# E1: functional text encoder (stage report)

Scope: the EmbeddingGemma 2 text tower on one Blackhole P150a, bf16 weights (the checkpoint is
bf16, so the weights are exact) and bf16 activations, HiFi4 + fp32 accumulation, DRAM
interleaved. Reference: HF `EmbeddingGemma2TextModel` in fp32 on CPU (transformers 5.20.2
overlay, identical outputs to 5.21.0; see `work_log.md`). All numbers below are measured on
sjc-snva-tp100 on 2026-10-27 unless marked otherwise.

## Architecture as implemented (checked against `modeling_embedding_gemma2.py`, 5.20.2 = 5.21.0)

| item | HF | TT |
|---|---|---|
| embedding | `embed_tokens(ids) * sqrt(512)` (fp32 scale 22.6274) | scale folded into the bf16 table at load |
| PLE | `RMSNorm_512(reshape(Linear 512->12288 (x) * 512^-0.5, [.., 24, 512]))`, shared norm weight | one linear (scale folded into W), then slice `[i*512:(i+1)*512]` + RMSNorm per layer |
| attention | q/k/v no bias; q_norm, k_norm with weight, v_norm without; RoPE after norm; scaling 1.0; no softcap; `is_causal=False` | fused QKV linear, `nlp_create_qkv_heads`, `rms_norm`, `rotary_embedding` (HF rotate_half), SDPA `is_causal=False, scale=1.0, attn_mask` |
| layer kinds | sliding (0-4, 6-10, ...): hd 256, 2 KV, theta 1e4, `abs(q-k) <= 512`; full (5, 11, 17, 23): hd 512, 1 KV, theta 1e6, padding only | same, one `Attention` class driven by `AttentionConfig` |
| layer | sandwich norms; PLE block `h + norm(W_p(gelu_tanh(W_g h) * pli))`; `h *= layer_scalar` | same order; gelu `GeluVariant.Tanh` |
| RMSNorm | `x * (mean(x^2)+1e-6)^-0.5 * w` in fp32, weight as-is | `ttnn.rms_norm`, HiFi4 + fp32 dest acc |
| head | final norm, `Linear 512->768` per token | same |
| pooling | ST `Pooling(mean, include_prompt)` + `Normalize` | `Pooler`: `keep @ hidden` matmul (fp32 out), divide by count, divide by L2 norm |
| positions | `arange(S)` for every row | same (cos/sin rows 0..S-1) |

## Module contracts

All activations are `[B, 1, S, D]` bf16 TILE in DRAM, where `S` is the physical bucket (power of
two, 32 ... 32768) and `L <= S` the logical length. Rows at padded positions are computed but
never read back.

| module | forward | in | out |
|---|---|---|---|
| `TextEmbedding` | `(input_ids)` | `[B, S]` uint32 ROW_MAJOR | `[B,1,S,512]` |
| `PerLayerInput` | `(inputs_embeds)`; `layer_input(proj, i)` | `[B,1,S,512]` | `[B,1,S,12288]`; `[B,1,S,512]` |
| `PerLayerInputBlock` | `(h, pli_i)` | `[B,1,S,512]` x2 | `[B,1,S,512]` |
| `RMSNorm` | `(x)` | any rank, last dim = `dim` | same |
| `RotaryTables` | `(x, layer_type)` | `[B,H,S,hd]` | same |
| `AttentionMasks` | `prepare(S)` (setup); `(keep)` | keep `[B,1,1,S]` bf16 TILE, 1 = valid | `{"sliding_attention", "full_attention"}` additive `[B,1,S,S]` bf16 |
| `Attention` | `(x, mask)` | `[B,1,S,512]`, mask `[B,1,S,S]` | `[B,1,S,512]` |
| `MLP` | `(x)` | `[B,1,S,512]` | `[B,1,S,512]` |
| `EncoderLayer` | `(x, pli_i, mask)` | as above | `[B,1,S,512]` |
| `Pooler` | `(last_hidden, keep)` | `[B,1,S,768]`, `[B,1,1,S]` | `[B,1,1,768]` fp32, unit L2 |
| `TextModel` | `(input_ids, keep)` | as above | `(last_hidden [B,1,S,768] bf16, embedding [B,1,1,768] fp32)` |
| `TextEncoder` | `encode(ids, mask, traced=)` (host) | torch `[B, L]` | torch `[B, 768]` |

Mask semantics: `key_bias = (keep - 1) * 1e9`; full = `repeat(key_bias, S rows)`; sliding =
full + constant band (`-1e9` where `|q-k| > 512`, built once per bucket; none needed for S <= 513).
TT SDPA rejects `sliding_window_size` together with `attn_mask`
(`sdpa_device_operation.cpp:141`), so the functional path uses the dense mask for both kinds.
Every `from_state_dict` checks tensor shapes and writes the ttnn weight cache
(`~/.cache/ttnn/models/embedding_gemma2/<snapshot>/`).

## PCC vs HF (bar 0.995 per layer kind; cosine >= 0.99 per prompt)

Cases: batch 1 at L = 1, 7, 31, 33, 511, 512, 513, 1025, 8192; padded batches (1025, 513, 31)
and (512, 33); full stack also batch 8 (512, 480, 400, 300, 257, 129, 64, 7) and L = 32768.
Each module gets the HF input of that module; PCC is over valid tokens only.
Source: `pytest models/demos/embedding_gemma2/tests/pcc models/demos/embedding_gemma2/demo/demo.py`
-> 101 passed (249.2 s); metrics in `tt-ops/runs/embeddinggemma-2/e1/pcc_results.jsonl`. An earlier
run of the same suite in a separate process (before the `capture_all` change) gave the same
230 metric values bit for bit.

| layer kind | tests | min PCC | worst case |
|---|---|---|---|
| embedding (`embed_tokens * sqrt(512)`) | 11 | 0.999999 | all |
| per-layer inputs (24 slices, min over layers) | 11 | 0.999994 | L=1 |
| attention, sliding (layer 0, hd 256) | 11 | 0.999990 | L=8192 |
| attention, full (layer 5, hd 512) | 11 | 0.999945 | L=8192 |
| MLP (layers 0 and 5) | 22 | 0.999993 | layer 5, L=7 |
| encoder layer, sliding (layer 0) | 11 | 0.999980 | L=1025 |
| encoder layer, full (layer 5) | 11 | 0.999978 | L=8192 |
| all 24 layers, teacher forced (L=1025, 8192, padded b3) | 72 | 0.999978 | layer 5, L=8192 |

Full stack, free running (ids -> embedding):

| case | bucket | last-hidden PCC | embedding cosine (per row) |
|---|---|---|---|
| L=1 | 32 | 0.999985 | 0.999985 |
| L=7 | 32 | 0.999338 | 0.999764 |
| L=31 | 32 | 0.999207 | 0.999772 |
| L=33 | 64 | 0.999282 | 0.999821 |
| L=511 | 512 | 0.999303 | 0.999925 |
| L=512 | 512 | 0.999330 | 0.999931 |
| L=513 | 1024 | 0.999404 | 0.999926 |
| L=1025 | 2048 | 0.999381 | 0.999936 |
| L=8192 | 8192 | 0.999427 | 0.999980 |
| L=32768 | 32768 | 0.998819 | 0.999937 |
| b3 (1025, 513, 31) | 2048 | 0.999440 | 0.999936 / 0.999920 / 0.999732 |
| b2 (512, 33) | 512 | 0.999341 | 0.999932 / 0.999906 |
| b8 (512 ... 7) | 512 | 0.999548 | min 0.999607 (L=7 row) |

The per-layer PCC (>= 0.99998) is higher than the free-running last-hidden PCC (0.9988-0.9995)
because bf16 rounding compounds over 24 layers; the `layer_scalar` product (2.5e-6) does not hurt
because every sub-block is RMS-normalized. The lowest cosines are for the shortest prompts, where
the mean pools few tokens. No case needed a fix: the first implementation met every bar.

## Trace, determinism, fallback audit, watcher

| check | command | result |
|---|---|---|
| trace replay == eager | `tests/pcc/test_trace.py::test_trace_matches_eager` (b1 L100/77 in bucket 128, b1 L1025/2000 in 2048, b3 padded in 2048, b8 in 512) | bit-identical embedding and last hidden (max abs diff 0.0); replay cosine vs HF >= 0.999607; a second logical length reuses the bucket's trace |
| trace allocation safety | same file + demo with `TT_METAL_TRACE_ALLOC_TRACKING=1` | 7 passed, no unsafe live buffer (`e1/logs/trace_alloc_tracking.log`) |
| determinism | `test_trace.py::test_determinism` (b3 padded, b1 L8192) | 5 replays + 2 eager runs bit-identical |
| fallback audit | `tests/pcc/test_fallback_audit.py` | 0 torch calls in `TextModel.forward` (TorchFunctionMode), no ttnn host transfer called, `throw_exception_on_fallback=True` did not raise; Tracy report of one forward: 743 device ops, **0 host ops** |
| watcher | `TT_METAL_WATCHER=10 pytest tests/pcc/test_text_model.py -k "test_text_model and not s32768"` | 15 passed (122 s); `watcher.log` has no error/assert/sanitize/overflow line (`e1/watcher/generated/watcher/watcher.log`) |

Trace-allocation finding (measured): the first run of a new program shape allocates
program-cache buffers that stay alive (61 live buffers, 2 of them the uploaded inputs, after one eager
b2/S512 forward; the count stays 61 over three repeats). If that happens after a trace was captured, the buffers can sit in the trace's
intermediate region. `TextEncoder.capture_all` therefore uploads every bucket constant, allocates
every persistent input and compiles every shape before the first capture; trace outputs are
acknowledged as corruptible (each replay rewrites them).

Static audit: `torch`/`from_torch`/`to_torch` appear in `tt/` only in setup paths
(`from_state_dict`, `RotaryTables.build`, `AttentionMasks.prepare`) and in the host side of
`TextEncoder` (`prepare_inputs`, `read_*`).

## Performance (functional, unoptimized; traced, warmed, 20 replays)

`pytest models/demos/embedding_gemma2/tests/perf/perf.py -k traced -s`
(`device` = back-to-back `execute_trace` + one sync; `e2e` = input copy + replay + embedding readback).

| batch x seq | device ms | e2e ms | tokens/s | embeddings/s |
|---|---|---|---|---|
| 1 x 128 | 8.75 | 8.87 | 14.6k | 114 |
| 1 x 512 | 18.31 | 18.43 | 28.0k | 54.6 |
| 1 x 2048 | 58.98 | 59.11 | 34.7k | 17.0 |
| 8 x 512 | 74.48 | 74.67 | 55.0k | 107 |

tt-perf-report for one warmed eager forward, b1 x 512, signposts `PERF_TEXT`/`PERF_TEXT_END`
(`tracy/b1_s512/forward_perf_report.{txt,csv}`, stacked summary `forward_perf_report_stacked.csv`;
`Device Time` column in microseconds): 743 device ops, 18,134 us device time + 855 us op-to-op
gaps (matches the 18.3 ms trace replay).

| op | device us | share | count |
|---|---|---|---|
| SDPA | 7,175 | 39.6 % | 24 |
| Matmul | 4,905 | 27.1 % | 171 |
| RMSNorm (LayerNorm op) | 2,096 | 11.6 % | 217 |
| eltwise binary | 1,371 | 7.6 % | 149 |
| RoPE | 1,323 | 7.3 % | 48 |
| eltwise unary | 642 | 3.5 % | 51 |
| create/concat heads, slice, other | 623 | 3.4 % | 83 |

Obvious E5 levers [guess]: SDPA with dense masks on 32-64 cores at HiFi4 (native sliding
window / block-map masks, larger grids, LoFi/bfp8), 217 separate norms (fuse q/k norm + RoPE,
sharded norms), bf16 HiFi4 matmuls (bfp8 weights, sharding), the per-layer PLE slice.

## Context contract

`doc/context_contract.json`: HF `max_position_embeddings` 262144; supported and tested 32768
(PCC 0.998819, cosine 0.999937). The gap is the dense O(S^2) masks of this functional path
(6 GiB of masks at 32768, 24 GiB at 65536 of 32 GiB DRAM) and the fp32 CPU reference (24 GB RSS at
32768, about 4x that at 65536 > 61 GB host RAM). The model card's shared context is 8192.

## Limitations

- Dense additive masks: memory O(B * S^2) per forward (and per batch row); native
  sliding-window SDPA is deferred to E5 (approved in the brief).
- Power-of-two buckets: L = 513 runs at 1024, L = 1025 at 2048 (up to 2x compute waste).
- `capture_all` must list every shape before the first capture (see above).
- Decoder-only items (paged KV cache, decode, page table, current position, sampling, chunked
  prefill) are not applicable to this encoder and were not built.
- bf16 everywhere; lower-precision weights/activations are E6.

## Artifacts

- `tt-ops/runs/embeddinggemma-2/e1/logs/`: `pcc_full.log`, `watcher_run.log`, `perf.log`,
  `tracy.log`, `trace_alloc_tracking.log`, `audit_demo.log`
- `tt-ops/runs/embeddinggemma-2/e1/pcc_results.jsonl`, `perf_results.jsonl`, `watcher/`
- `doc/functional_text/tracy/b1_s512/`: tt-perf-report table, CSV and stacked CSV. The raw ops CSV
  (2.8 MB) stays at `generated/profiler/reports/2026_10_27_12_36_08/` (not committed).
