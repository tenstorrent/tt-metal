# llama-nemotron-rerank-1b-v2 on Blackhole P150

[nvidia/llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) is a
cross-encoder reranker: it reads one `question:{query} \n \n passage:{passage}` sequence and returns one
relevance logit for that pair. It is Llama-3.2-1B with four changes:

| | Llama-3.2-1B | this model |
|---|---|---|
| attention | causal | bidirectional (padding keys masked) |
| output | next-token logits | final RMSNorm -> mean over real tokens -> `score` linear 2048 -> 1 |
| shape | 16 layers, hidden 2048, 32 q / 8 kv heads of 64, FFN 8192, llama3 RoPE | same |
| inputs | | up to 8192 tokens per pair |

There is no decode step and no KV cache: one forward per batch of pairs.

## Layout

```
llama_nemotron_rerank_1b/
  tt/model.py                  NemotronReranker: embedding, 16 layers, norm, score, pooling (plain ttnn + custom ops)
  tt/model_config.py           shapes, precisions, matmul / SDPA block tables (every knob documented with its measurement)
  tt/weights.py                checkpoint -> device tensors (Q/K row permutation for RoPE, attention scale folded into Wq)
  tt/runner.py                 TracedReranker: one trace per (batch {1,8,32}) x (length bucket), batching by bucket
  tt/custom_ops/               generic_op kernels: fused_heads_rope (new), fused_add_rmsnorm (from pplx_embed_4b)
  reference/hf_reference.py    HF CPU reference
  demo/serve.py                FastAPI rerank server
  demo/generator_vllm.py       NemotronRerankForScoring (same interface as bge_m3's BgeM3ForEmbedding)
  demo/rerank_client.py        server check + MIRACL-en load test
  demo/eval_miracl.py          device scores vs HF fp32 on MIRACL-en (nDCG@10, score error, Pearson)
  demo/bench.py                traced latency per (batch x length)
  perf_tools/                  matmul / SDPA sweeps, probes, cycle-level SoL report
  tests/pcc/                   fused_heads_rope, one decoder layer, end-to-end scores
```

## Run

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
M=models/demos/blackhole/llama_nemotron_rerank_1b

pytest -q $M/tests/pcc                                         # 7 tests
python $M/demo/bench.py --shapes 1x512 8x512 32x512 1x8192     # traced latency
python $M/demo/eval_miracl.py --eval-set miracl_eval.pt --batch 8
python $M/demo/serve.py --port 8000                            # server (one P150: --dp 1)
python $M/demo/rerank_client.py --url http://localhost:8000 --eval-set miracl_eval.pt
```

`miracl_eval.pt` is 40 MIRACL-en dev queries x 25 candidates (1000 pairs) with HF fp32 scores, built from
`mteb/MIRACLReranking`. Every knob in `RerankConfig` can be overridden for an A/B:
`bench.py --cfg mm_fidelity=LoFi w2_dtype=bfloat4_b`.

Tracy, cycle-level speed of light per op:

```bash
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
  python -m tracy -p -r -v -o prof $M/demo/bench.py --shapes 32x512 --iters 2 --signpost
python $M/perf_tools/opsol.py prof/reports/*/ops_perf_results_*.csv 32x512
```

## Server

`demo/serve.py` follows the embedding servers' layout (`pplx_embed_0_6b/demo/serve.py`): FastAPI + uvicorn,
one spawned worker process per chip holding a resident `TracedReranker`, an async pool handing requests
to idle chips, a collector thread resolving futures. The server tokenizes the prompt template and ships
token ids; the worker packs pairs by length bucket into traced batches of 1 / 8 / 32.

| endpoint | request | response |
|---|---|---|
| `POST /v1/rerank`, `/rerank` | `{query, documents: [str or {text}], top_n?, return_documents?}` (vLLM / Cohere shape) | `results: [{index, relevance_score, logit, document?}]` sorted by score |
| `POST /v1/score`, `/score` | `{text_1: query, text_2: str or [str]}` (vLLM shape) | `data: [{index, score}]` |
| `GET /health`, `/v1/models`, `/metrics` | | readiness, model id, per-request tokenize / device / wall ms |

`relevance_score` is sigmoid(logit) (`--score raw` returns the logit); `logit` is always included.
One P150, MIRACL-en through the server at concurrency 4: 132 pairs/s, nDCG@10 0.6840 (HF 0.6840).

## Accuracy

1000 MIRACL-en pairs, device vs HF fp32 (`eval_miracl.py`). The gate used for every change: Pearson >= 0.9995
and nDCG@10 within 0.015 of HF.

| | mean abs logit error | Pearson | nDCG@10 | MRR@10 |
|---|---|---|---|---|
| HF bf16 (CPU) | 0.064 | 0.99997 | 0.6845 | 0.6486 |
| this model, batch 8 | 0.207 | 0.99962 | 0.6835 | 0.6486 |
| this model, batch 32 | 0.212 | 0.99962 | 0.6859 | 0.6486 |
| HF fp32 (reference) | 0 | 1 | 0.6840 | 0.6486 |

## Performance (one P150)

Traced forward, inputs on device. "Sustained" is the median of the second half of 40 replays: under load the
board holds its 150 W limit by lowering the clock from 1.35 GHz to about 0.95 GHz, so it is the number a
loaded server sees; "best" is at full clock.

| batch x tokens | after bring-up | now, sustained | now, best | speed of light (cycles) |
|---|---|---|---|---|
| 1 x 128 | 12.8 ms | 5.97 ms | 5.96 ms | |
| 1 x 512 | 15.6 ms | 7.50 ms | 7.47 ms | 35.0 % |
| 8 x 512 | 76.2 ms | 48.8 ms | 40.5 ms | 50.9 % |
| 32 x 512 | 329 ms | 192 ms | 140 ms | 59.4 % |
| 1 x 2048 | 46.5 ms | 27.4 ms | 25.5 ms | 49.9 % |
| 1 x 8192 | 362 ms | 197 ms | 150 ms | 58.4 % |

Speed of light: ideal cycles for the matmuls and attention at the precision each op runs at
(130 cores x 4096 FLOP/cycle LoFi, / 2 for HiFi2, / 4 for HiFi4), divided by the sum of the ops' device
kernel cycles (`perf_tools/opsol.py`). Kernel durations, not firmware windows: an op that runs on fewer
cores than the next lets the idle cores start the next op's firmware early, so firmware windows overlap
and double count. The kernel-cycle totals match wall time at the measured clock (32x512: 187.8 M cycles
in 140 ms = 1.34 GHz).

32 x 512 per op (share of cycles, % of that op's compute speed of light): FF1+FF3 fused SwiGLU 42 % (84 %),
custom data-movement ops 21 % (head split + RoPE, residual add + RMSNorm; DRAM bound at ~65 % of 512 GB/s),
FF2 13 % (68 %), QKV 9 % (72 %), SDPA 8 % (58 %), WO 7 % (67 %).

## What was done

Bring-up: a model-local encoder on plain ttnn ops (no `tt_transformers` decoder machinery: no KV cache
fill, no paged attention), bidirectional SDPA with a padding mask, final norm, a per-token score head
(mean_t(x_t) . w = mean_t(x_t . w), so the head is one [B*S, 2048] x [2048, 32] matmul) and a masked sum,
divided by the real-token count on the host.

Data movement, in the order they landed (32x512 unless noted):

| change | why it is faster | effect |
|---|---|---|
| FF1 + FF3 + silu(gate) * up in one `minimal_matmul(fuse_swiglu=True)` over a tile-interleaved weight | one op writes the SwiGLU product instead of two matmul outputs and a multiply | part of 329 -> 276 ms with the matmul configs |
| fused head split + RoPE (`tt/custom_ops/fused_heads_rope`, new) | one pass over the fused QKV row replaces `nlp_create_qkv_heads` + two `rotary_embedding_llama` (three DRAM round trips over Q and K) | 1349 -> 729 us per layer |
| fused residual add + RMSNorm (`tt/custom_ops/fused_add_rmsnorm`, from pplx_embed_4b) from 4096 rows | one pass writes the residual and the normed tensor (4 DRAM passes instead of 5, one launch) | 797 -> 670 us per call |
| SDPA writes concatenated heads (`output_concat_heads`) | no `nlp_concat_heads` pass | |
| padding mask built on device from a [B, 1, 32, S] valid row, stored as bfp4 | the mask holds only 0 and -1e9, exact in bfp4 at a quarter of bf16's bytes; the host sends no S x S tensor | SDPA 32x512 1780 -> 1113 us |
| SDPA block map (`attn_mask_block_map`) | blocks with no real query or key are skipped | 1x2048 802 -> 389 us |
| ttnn SDPA change: block-map value 2 = every key real; the reader zero-fills those mask tiles (NOC zero writes, no DRAM read) and the streaming compute pops them unapplied | SDPA with a padding mask runs at unmasked speed on fully visible blocks; output bit-identical | 1x8192 36.4 -> 46.3 % of SoL; isolated 32x512 SDPA 1255 -> 663 us |
| softmax scale 1/8 folded into Wq (exact: power of two), SDPA at scale 1 | ttnn's SDPA wrapper otherwise multiplies the mask by 1/scale on every call | one [B, 1, S, S] pass per layer removed |
| activations up to 8.5 MB in L1 | bs1 intermediates never touch DRAM | 1x512 10.1 -> 8.9 ms |
| block-sharded RMSNorm up to 512 rows | stock rms_norm gives each core whole tile rows (16 cores at 512 rows) | 1x128 7.03 -> 6.27 ms |
| bfp8 matmul inputs (norm outputs, SwiGLU product) and bfp8 WO / FF2 outputs | half the bytes between ops; `minimal_matmul` runs best with in0 and weight in one dtype | FF2 38.7 -> 28.0 M cycles |

Knobs on compute ops: `minimal_matmul` block sizes swept per weight and row count (`perf_tools/sweep_mm.py`,
FF1+FF3 at 512 rows 210 -> 108 us); FF2 at LoFi (accuracy unchanged); SDPA q / k chunks per shape
(`perf_tools/sweep_sdpa.py`). Tried and rejected by the accuracy gate: LoFi on QKV / WO / FF1+FF3
(Pearson 0.9992-0.9993), bfp4 weights (0.987), HiFi2 SDPA (mean error 0.23 -> 0.30), bfp8 residual
stream (0.99944). No effect: raising the power limit (the card stays at 150 W).

## Open items toward 75 % of speed of light

* 32x512 needs 39 M fewer cycles (188 -> 149 M). The custom data-movement ops (40 M) are at ~65 % of DRAM
  bandwidth; removing them takes keeping activations in L1 across ops (row-blocked layer execution), which
  costs matmul efficiency at smaller M, or fusing them into matmul prologues / epilogues inside
  `minimal_matmul`. QKV / WO / FF2 / SDPA at 58-72 % have tuning room toward 80 %.
* bs1 (35 %): the matmuls at 512 rows run at 27-52 % of their limit; FF2 at LoFi is bound by its weight
  read. DRAM-sharded weights are the next step.
* `fused_heads_rope` with a bfp8 input or output writes wrong K / V (Q is correct); it rejects a dtype
  change until fixed. Fixed, it would halve that op's traffic and SDPA's reads.
