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
  tt/runner.py                 TracedReranker: packs pairs into rows, two traces (masked / unmasked) per
                               (batch {1,8,32}) x (row length) bucket
  tt/custom_ops/               generic_op kernels: matmul_qkv_rope (QKV matmul + head split + RoPE), matmul_add_norm
                               (WO / FF2 + residual add + RMSNorm on the L1-resident residual; embed_rmsnorm),
                               fused_add_rmsnorm (from pplx_embed_4b, < 4096 rows), fused_heads_rope (previous path)
  reference/hf_reference.py    HF CPU reference
  demo/serve.py                FastAPI rerank server
  demo/generator_vllm.py       NemotronRerankForScoring (same interface as bge_m3's BgeM3ForEmbedding)
  demo/rerank_client.py        server check + MIRACL-en load test
  demo/eval_miracl.py          device scores vs HF fp32 on MIRACL-en (nDCG@10, score error, Pearson)
  demo/bench.py                traced latency per (batch x length)
  perf_tools/                  matmul / SDPA sweeps, probes, cycle-level SoL report
  tests/pcc/                   fused_heads_rope, one decoder layer, end-to-end scores (padded, unpadded, packed rows)
```

## Run

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
M=models/demos/blackhole/llama_nemotron_rerank_1b

pytest -q $M/tests/pcc                                         # 9 tests
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
token ids; the worker packs the pairs into rows and runs them as traced batches of 1 / 8 / 32 rows (see
"Packed rows" below).

| endpoint | request | response |
|---|---|---|
| `POST /v1/rerank`, `/rerank` | `{query, documents: [str or {text}], top_n?, return_documents?}` (vLLM / Cohere shape) | `results: [{index, relevance_score, logit, document?}]` sorted by score |
| `POST /v1/score`, `/score` | `{text_1: query, text_2: str or [str]}` (vLLM shape) | `data: [{index, score}]` |
| `GET /health`, `/v1/models`, `/metrics` | | readiness, model id, per-request tokenize / device / wall ms |

`relevance_score` is sigmoid(logit) (`--score raw` returns the logit); `logit` is always included.
The model's own throughput is measured without the server (`perf_tools/miracl_runner.py`, "Packed rows"
below).

## Accuracy

1000 MIRACL-en pairs, device vs HF fp32 (`eval_miracl.py`). The gate used for every change: Pearson >= 0.9995
and nDCG@10 within 0.015 of HF.

| | mean abs logit error | Pearson | nDCG@10 | MRR@10 |
|---|---|---|---|---|
| HF bf16 (CPU) | 0.064 | 0.99997 | 0.6845 | 0.6486 |
| this model, batch 8 | 0.205 | 0.99966 | 0.6745 | 0.6361 |
| this model, batch 32 | 0.235 | 0.99956 | 0.6841 | 0.6486 |
| `TracedReranker`, packed rows (25-document requests) | 0.241 | 0.99957 | 0.6837 | |
| HF fp32 (reference) | 0 | 1 | 0.6840 | 0.6486 |

## Performance (one P150)

Traced forward, inputs on device. Each time is the median of the second half of 30 back-to-back replays:
under continuous load the board holds its 150 W limit by lowering the clock from 1.35 GHz to about 0.95 GHz,
so this is the latency under continuous load. "Now": every row is one pair that fills it, so SDPA runs
without a mask (`TracedReranker` does the same for such a batch). "Mask on": the same inputs with the
attention mask applied (`demo/bench.py --masked`). Any batch whose rows hold several packed pairs, or a
padded pair, runs with the mask, so this column is the time per batch of full rows in normal use; what
packing saves is rows per request, which is in the MIRACL table under "Packed rows".

| batch x tokens | after bring-up | now | now, mask on | speed of light (kernel cycles) |
|---|---|---|---|---|
| 1 x 128 | 12.8 ms | 6.27 ms | 6.41 ms | |
| 1 x 256 | | 6.47 ms | 6.70 ms | |
| 1 x 512 | 15.6 ms | 7.69 ms | 8.02 ms | 33.7 % |
| 1 x 1024 | | 15.2 ms | 16.7 ms | |
| 8 x 512 | 76.2 ms | 39.8 ms | 42.1 ms | 58.6 % |
| 32 x 512 | 329 ms | 190 ms | 193 ms | **71.3 %** (156.4 M cycles) |
| 1 x 2048 | 46.5 ms | 24.7 ms | 27.7 ms | 51.9 % |
| 1 x 4096 | | 66.6 ms | 77.2 ms | |
| 1 x 8192 | 362 ms | 188 ms | 221 ms | 64.1 % |

32 x 512 was at 75.9 % (147.0 M cycles, 175 ms) with FF1 + FF3 accumulating in bf16; packed rows put every
pair through that shape, where bf16 accumulation fails the accuracy gate, so it now accumulates in fp32
(see "Packed rows").

Speed of light: ideal cycles for the matmuls and attention at the precision each op runs at
(130 cores x 4096 FLOP/cycle LoFi, / 2 for HiFi2, / 4 for HiFi4), divided by the sum of the ops' device
kernel cycles (`perf_tools/opsol.py`). Kernel durations, not firmware windows: an op that runs on fewer
cores than the next lets the idle cores start the next op's firmware early, so firmware windows overlap
and double count. Cycles do not depend on the clock, so the speed-of-light figures hold at any clock. Each
shape was benchmarked in its own process on an idle board.

32 x 512 per op (share of cycles, % of that op's compute speed of light):

| op | M cycles | share | of its SoL |
|---|---|---|---|
| FF1 + FF3 + SwiGLU (`minimal_matmul`, HiFi2, fp32 accumulation) | 87.2 | 56 % | 76 % |
| FF2 + residual add + RMSNorm (`matmul_add_norm`, LoFi) | 24.0 | 15 % | 69 % |
| QKV + head split + RoPE (`matmul_qkv_rope`, HiFi2) | 18.3 | 12 % | 68 % |
| SDPA (HiFi4, unmasked) | 14.1 | 9 % | 59 % |
| WO + residual add + RMSNorm (`matmul_add_norm`, HiFi2) | 12.2 | 8 % | 67 % |
| embedding + first RMSNorm (`embed_rmsnorm`), score head | 0.6 | 0.4 % | |

## Packed rows

A trace runs one fixed shape, so a pair is padded to a row of S tokens (S = 128, 256, 512, ...). On MIRACL-en
(median pair 140 tokens, p90 256) one pair per row left 33 % of the token slots as padding, and every pad
token goes through all 16 layers. `TracedReranker` now packs several pairs back to back into each row (at
most 32 pairs per row, first-fit decreasing; `prepare_packed_inputs`):

* the `valid` input is a [32, S] segment matrix per row, row j = 1 on pair j's tokens;
* the attention mask is segments^T @ segments (1 where query and key are in the same pair) -> 0 / -1e9 in
  bfp4, so a pair never sees another pair or padding (`_segment_mask`);
* RoPE needs no change: a score depends only on the distance between query and key, and both are in the
  same pair, so a pair scores the same at any offset in the row;
* pooling is unchanged: segments @ per-token scores gives one sum per pair.

The runner picks the row length S with the least estimated cost, and splits the rows into batches of
1 / 8 / 32 at the least cost (rows run, repeat rows included, plus ~0.7 row per launch: 1 x 512 takes
7.7 ms, 8 x 512 39.8 ms). The old plan rounded 9 rows up to a batch of 32; it now runs 8 + 1.

MIRACL-en through tt-metal's traced runner directly (`TracedReranker.score_ids`, no server;
`perf_tools/miracl_runner.py`): 40 requests of 25 documents each, then all 1000 pairs in one call:

| | per request | 1000 pairs in one call | MIRACL Pearson | nDCG@10 |
|---|---|---|---|---|
| one pair per row, old batch plan | 142-156 ms | 385-400 pairs/s | 0.99960 | 0.6844 |
| packed rows, new batch plan | 58.7 ms | 596-604 pairs/s | 0.99957 | 0.6837 |

Packing first failed the accuracy gate (Pearson 0.99945): it runs every pair at 32 x 512 (16384 rows), where
FF1 + FF3 accumulated in bf16 DST. One pair per row at 32 x 512 gives 0.99930, so the shape, not the packing,
was the cause; the earlier batch-32 MIRACL number (0.99958) mostly ran 32 x 128 / 32 x 256 batches, which
never take that path. fp32 accumulation everywhere: 0.99959 (packed). Its 16384-row block (4, 16, 8, 1, 4)
costs 87.2 vs 77.8 M cycles for FF1 + FF3; (2, 32, 8, 1, 4) runs 79.4 M cycles but the board throttles it
to 335 ms per 32 x 512 replay under continuous load (182-188 ms with K 16).

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
| QKV matmul + head split + RoPE in one program (`tt/custom_ops/matmul_qkv_rope`) | minimal_matmul's dataflow with a new output stage: Q / K tiles rotated in L1 (HF half-split layout, lo' = lo cos - hi sin, hi' = hi cos + lo sin) and every tile written straight into the head layouts; the [B*S, 3072] QKV activation never reaches DRAM | QKV + heads 1406 -> 924 us per layer; 185.2 -> 175.1 M cycles |
| fused residual add + RMSNorm (`tt/custom_ops/fused_add_rmsnorm`, from pplx_embed_4b) from 4096 rows | one pass writes the residual and the normed tensor (4 DRAM passes instead of 5, one launch) | 797 -> 670 us per call |
| SDPA writes concatenated heads (`output_concat_heads`) | no `nlp_concat_heads` pass | |
| padding mask built on device from a [B, 1, 32, S] valid row, stored as bfp4 | the mask holds only 0 and -1e9, exact in bfp4 at a quarter of bf16's bytes; the host sends no S x S tensor | SDPA 32x512 1780 -> 1113 us |
| SDPA block map (`attn_mask_block_map`) | blocks with no real query or key are skipped | 1x2048 802 -> 389 us |
| SDPA without the mask when no pair is padded: `forward(masked=False)`, and `TracedReranker` keeps a second trace per bucket for such batches | an all-zero padding mask is still read from DRAM and added on every K chunk | 32x512 SDPA 1101 -> 623 us per layer; 1x8192 220 -> 188 ms sustained |
| softmax scale 1/8 folded into Wq (exact: power of two), SDPA at scale 1 | ttnn's SDPA wrapper otherwise multiplies the mask by 1/scale on every call | one [B, 1, S, S] pass per layer removed |
| activations up to 8.5 MB in L1 | bs1 intermediates never touch DRAM | 1x512 10.1 -> 8.9 ms |
| block-sharded RMSNorm up to 512 rows | stock rms_norm gives each core whole tile rows (16 cores at 512 rows) | 1x128 7.03 -> 6.27 ms |
| bfp8 matmul inputs (norm outputs, SwiGLU product) and bfp8 WO / FF2 outputs | half the bytes between ops; `minimal_matmul` runs best with in0 and weight in one dtype | FF2 38.7 -> 28.0 M cycles |
| L1-resident residual stream; WO and FF2 fused with the residual add + RMSNorm (`matmul_add_norm`) | the residual lives in L1 for the whole forward, block-sharded like the fused matmul's 13 x 10 grid (uneven shards, no padded copy); the matmul's output block never leaves L1, the epilogue adds it to the core's residual slice in place and writes only the normed output. The cores sharing rows swap partial sums of squares over the NoC. With the residual read from and written to DRAM in the epilogue, the matmul itself ran ~40 % slower | 169.7 -> 160.1 M cycles |
| RMSNorm gammas folded into the next matmul's weight rows (QKV, FF1+FF3, score head) | the fused norms skip the gamma multiply | part of 160.1 -> 151.6 M cycles |
| `embed_rmsnorm`: the embedding gathered straight into the resident residual, then the first RMSNorm | replaces `ttnn.embedding` (tiles to DRAM), the copy into the L1 shards and the norm: the 10 cores holding a row block split each 32-token tile-row, read whole 4 KB embedding rows (32 slices of 448 B per core ran DRAM at ~175 GB/s) and write each peer's column slice into its staging; the compute tilizes and normalizes block by block as rows arrive | 645 -> 304 us; 150.4 -> 149.9 M cycles |
| fused norm statistic on the FPU | the epilogue squared each h tile on the SFPU (~200 cycles a tile) and row-summed 7 packed x^2 tiles; now the FPU multiplies each packed h tile by itself into one accumulating DST tile per row (HiFi2) and row-sums that one tile, with one pack -> unpack handoff per block instead of per row | FF2 1209 -> 1150 us, WO 664 -> 612 us per layer; 149.8 -> 148.4 M cycles |
| activation blocks multicast down each core column in the fused WO / FF2 ops (`tt/custom_ops/kernels/dm_in0_mcast.cpp`) | minimal_matmul's in0 reader relays each block core to core (each core waits for its upstream neighbour, then for its downstream one, then writes); the column's first core now reads the block once and multicasts it | WO 602 -> 593 us, FF2 1129 -> 1124 us; 148.4 -> 148.1 M cycles (QKV got slower, 865 -> 884 us, left on the relay) |

Knobs on compute ops: `minimal_matmul` block sizes swept per weight and row count (`perf_tools/sweep_mm.py`,
FF1+FF3 at 512 rows 210 -> 108 us); FF2 at LoFi (accuracy unchanged); SDPA q / k chunks per shape
(`perf_tools/sweep_sdpa.py`). Tried and rejected by the accuracy gate: LoFi on QKV / WO / FF1+FF3
(Pearson 0.9992-0.9993), bfp4 weights (0.987), HiFi2 SDPA (mean error 0.23 -> 0.30), bfp8 residual
stream (0.99944). No effect: raising the power limit (the card stays at 150 W).

## Open items

* 32 x 512 is at 71.3 % of speed of light (75.9 % before FF1 + FF3 moved to fp32 accumulation). The rest
  is spread over the ops: FF1 + FF3 is at 76 % (85 % with bf16 accumulation, which fails the accuracy gate
  with packed rows; the faster fp32 K-32 block is throttled by the board, see "Packed rows"); the fused
  WO / FF2 matmuls run at ~76-84 % of their limit inside their K loops (FF2 at LoFi with 4 x 1 subblocks, which
  the 7-tile N slice per core forces) plus ~8 % of epilogue; QKV + RoPE's matmul runs at ~80 % of HiFi2 peak and
  waits ~10 % on its inputs (the rotation itself is ~1.5 %); SDPA's softmax bounds it (HiFi2 matmuls are no
  faster). The 13 x 10 grid costs 3 % (FF1 + FF3) to 10 % (WO / FF2: 64 output columns over 10 core rows)
  in padding.
* The matmul loop itself caps the achievable fraction: minimal_matmul's compute loop on tiles already in L1,
  with no data movement (`perf_tools/probe_mm_loop.py`, profiler cycles), runs at 93.5 % of FPU peak at HiFi2,
  96.5 % at HiFi4 and ~91-92 % at LoFi (about 2 cycles per tile-multiply for the source-register handoff
  between tiles). With every op at that ceiling, no epilogue, no input waits and the 13 x 10 grid's padding,
  32 x 512 would be at ~88.5 % (~90 % with WO / FF2 re-gridded to 5 columns per core, which does not fit next
  to the resident residual: 2.3 MB of CBs).
* Measured and rejected: bf16 DST with 8-tile subblocks inside the fused ops (slower), full-sync DST, the
  non-transposed grid (WO 823 vs 628 us), K block 16 for FF2 (does not fit next to the resident residual),
  interleaving the norm's second pass into the next K loop, LoFi y / RoPE multiplies (no faster; RoPE loses
  Q / K PCC), a 1D row-split score head (160 vs 149 us), bfp8 Q / K / V into SDPA (Pearson 0.99927), bfp4
  SwiGLU product, approximate SiLU, bfp8 Q into SDPA (its output takes Q's dtype: 657 -> 614 us, Pearson 0.99943),
  deeper (3- / 4-block) input buffers for FF1 + FF3 (no change), QKV with a core's whole N slice per block
  (1206 vs 861 us), SDPA q chunk 256 (does not fit next to the resident residual). QKV + RoPE waits ~20 % of
  its K loop on inputs (zones), but multicasting its weight blocks (935 us) or activations (884 us) instead of
  relaying them is slower, and N blocks of 4 / 6 or M blocks of 10 / 20 land at 862-954 us (861 now).
* A custom attention kernel for SDPA (all keys per query row, exp(S - c) without a row max, row sums from
  P [V | 1], 1 / l in DST, padding keys as zero rows of [V | 1]): correct (PCC 0.99969 with the precise exp)
  but 1298 us at 32 x 512, 968 us with the fast exp (PCC 0.99909), against ttnn's 610-660 us. The pack
  thread's exp + pack of the 16 score tiles per query row (~8 k cycles) exceeds the ~5 k of matmul per row;
  ttnn's streaming SDPA does a row in ~6.5 k, so it is within ~20 % of that bound. HF scores on MIRACL stay
  in [-69, 26] (every row's max >= -1.01), which is what made a max-free exp possible. ttnn's SDPA ignores
  SDPAProgramConfig.matmul_math_fidelity (only the ring-joint variant reads it): its matmuls run at the
  compute config's fidelity.
* FF1 + FF3 on the model's dataflow (minimal_matmul's legacy compute.cpp with multicast in0 / in1 and deeper
  weight CBs): 4927-5655 us against minimal_matmul's 3609 us in every orientation / NOC / RISC / writer
  arrangement. minimal_matmul binds the Metal 2.0 forks (compute_metal2.cpp etc.), whose SwiGLU runs on the
  pack thread inside the last K block (swiglu_in_k_loop); the legacy kernel runs it as a separate pass. Inside
  minimal_matmul a 16 x 8 weight block read takes ~15 k cycles against ~6 k alone: DRAM reads saturate at
  ~287 B/cycle (~390 GB/s) over any number of reading cores (perf_tools/probe_dram_read.py), and the read
  queue is shared with the activation reads and the output writes.
* FF2 with bf16 DST and a bf16 partial-sum CB, (5,16,7,5,1): 1116 -> 1101 us per layer (32x512 147.28 -> 146.98 M
  cycles, 75.9 %) but MIRACL batch 32 Pearson 0.99949, under the 0.9995 gate (batch 8 0.99963): rejected.
* WO with bf16 DST (fp32 partial sums in L1): 147.28 -> 147.20 M cycles, MIRACL batch 32 Pearson 0.99952 (from
  0.99956), batch 8 0.99966: not kept for a 0.05 % gain. Larger M blocks for WO / FF2 (minimal_matmul's best WO
  block is M 8: 506 us bare) do not fit next to the resident residual's 573 KB shard.
* A virtual 16 x 8 core grid for WO / FF2 (128 cores wired by explicit core lists, 8 columns per core instead of
  7 over 10 rows, padding 11 % -> 1.6 %): output correct, but WO 893-1478 us and FF2 1657-2374 us against 575 /
  1116 us. A virtual 13 x 10 grid matches the physical one (580 us), and 13 x 8 costs only its extra work
  (638 us); 16 x 8 needs 3 row blocks in the 2 leftover physical rows, and those cores run ~40 % slower than
  the 13 column groups (per-core kernel ends 1.18 M vs 0.85 M cycles), which sets the op's time.
* Windowed SDPA (`cu_window_seqlens`, stock ttnn): the batch flattened to one [1, H, B*S, 64] sequence with each
  pair's real tokens in one window and its padding in another, so no mask tensor at all. Correct, but 2161 us
  against 1098 us for the masked [B, H, S, 64] call at 32 x 512 with mixed lengths
  (`perf_tools/probe_sdpa_windowed.py`).
* bs1 (34 %): the matmuls at 512 rows run at 27-52 % of their limit; FF2 at LoFi is bound by its weight read.
  DRAM-sharded weights are the next step. 1 x 128 / 1 x 512 read ~0.3 ms slower than in the previous README
  (5.97 / 7.50 ms), not yet traced to a change.
