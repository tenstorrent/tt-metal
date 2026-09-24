# pplx-embed-v1-4B and Qwen3-Embedding-4B on Blackhole P150: performance, baseline to now

All numbers are **sustained latency at ISL 512**: the median of iterations 15–29 of a 30-iteration run,
after the board's power manager has settled the clock (≈1.1–1.3 GHz under load), on one P150 of a Galaxy
(12×10 = 120 worker cores; a p150a card exposes 13×10). Timed path is the extended trace: forward + pooling
+ I/O in one replay. Branch `arg/embed-4b-pplx-qwen3`; measured 2026-09-24.

## Headline

| batch | baseline (2026-09-22) | pplx-embed-4B now | Qwen3-Embedding-4B now | speedup | H200 | × H200 (pplx) | tok/s (pplx) |
|---|---|---|---|---|---|---|---|
| 1 | 28.8 ms | **17.6 ms** | 18.4 ms | 1.64× | 5.44 ms | 3.24× | 29.1k |
| 8 | 190.3 | **120.9** | 120.8 | 1.57× | 33.08 | 3.65× | 33.9k |
| 16 | 372.7 | **227.6** | 228.2 | 1.64× | 67.23 | 3.39× | 36.0k |
| 32 | 720.8 | **446.4** | 445.1 | 1.61× | 139.15 | 3.21× | 36.7k |

- **Baseline** is commit fb45b516f49: the stock tt_transformers prefill with the grid clamp that lets bs≥8 run
  on this 120-worker part, re-measured today with the same method (its Python against the current host
  library). The two models share one code path; Qwen3-Embedding-4B differs only by causal attention and
  last-token pooling, which the stack switches on from `HF_MODEL`.
- **Accuracy** was re-checked after every landing: STS-B Spearman 0.8161 (pplx, bucketed bs1 path),
  0.812–0.816 through the batched paths; Qwen3-Embedding-4B 0.807–0.819 (last token + EOS). Every landed
  change is bit-identical to the stock op or within noise of the stock op's accuracy.

## What changed, in order

Each row was measured as a same-chip A/B when it landed (best of 10 at the cold clock; the sustained metric
was adopted later). The last column is e2e latency after the step: bs1 / bs8 / bs16 / bs32, in ms.

| # | Change | Why it is faster | After |
|---|---|---|---|
| 1 | Stock prefill configs corrected: `minimal_matmul` output subblock 1×8 (was 1×1), `in0_block_w` cap 8 → 38, grid clamped to the 120-worker part | the batched matmuls ran one tile per DST pass; FF2 was pinned at 2-tile blocks | bs8 −12%, bs16 −13%, bs32 −18%, bs1 −4.4 ms |
| 2 | Head split and concat as model-local kernels | the fused QKV activation is scattered straight into heads | bs1 −2.7 ms, bs32 −8.8 ms |
| 3 | Extended trace as the timed path | forward, pooling and I/O in one replay; a 3.4 ms host bubble at bs1 is gone | 25.2 / 155.6 / 288.7 / 543.5 |
| 4 | Head split + Q/K RMSNorm + RoPE fused into one kernel, Q/K/V written in bfp8; QKV projection writes bfp8 | one pass replaces five ops and a typecast; SDPA reads half the bytes | 23.7 / 127.5 / 240.9 / 456.7 |
| 5 | Merged core ranges for the model-local kernels | ≈0.4 µs per core range per launch | 23.4 / 126.7 / 240.6 / 455.8 |
| 6 | Batched SDPA on all 120 cores with a 512-token K chunk | K and V read once | 23.3 / 126.2 / 239.6 / 450.6 |
| 7 | Residual add + RMSNorm fused (bs16+) | one kernel writes the residual sum and the normalised tensor: 4 DRAM passes → 2 | 23.3 / 126.2 / 234.8 / 443.5 |
| 8 | Matmul block sweeps at bs8 / bs16; SwiGLU product as one kernel at bs32 | in-model sweeps; SiLU and multiply in one pass over the FF1/FF3 outputs | 23.3 / 123.4 / 228.3 / 438.1 |
| 9 | bs1: legacy 2D-multicast matmuls on 12×8, coalesced weight reads, SDPA q-chunk 256 | at M=512 the 2D kernel beats `minimal_matmul` by 53–65%; a factory bug had blocked grids wider than 8 | 17.7 / 123.4 / 228.3 / 438.1 |
| 10 | bs>1: add + RMSNorm split over 4–5 cores per row, SDPA 12×8 at bs8, DRAM-interleaved weights at bs32 | all 120 cores busy in the norm; fuller SDPA work units; faster weight reads at M=16384 | 17.7 / 118.5 / 216.7 / 428.1 |
| 11 | SDPA writes the concatenated-heads layout directly (new op flag) | the concat pass (142 MB per layer at bs32) is gone | 17.6 / 115.3 / 221.0 / 425.5 |
| 12 | bs1: concat-free SDPA output; residual adds written in the norm's shard layout | the per-layer concat and 72 layout conversions are gone | 17.3 / 115.3 / 221.0 / 425.5 |

Sustained after step 12: **17.6 / 120.9 / 227.6 / 446.4 ms**. Configuration lives in
`demo/_common.py::apply_workload_env` (per-batch defaults, every knob overridable from the shell), the kernels in
`tt/custom_ops/`, the shared-code changes in `models/tt_transformers/tt/`, the SDPA op and the 2D matmul factory.

## Where the time goes now

Device-profiled bs32 (365 ms of kernels at the nominal clock; sustained e2e 446 ms): matmuls 65%, SwiGLU
product 13%, fused add+RMSNorm 8%, SDPA 8%, fused heads op 6%. bs1 (16.6 ms of kernels, 509 ops): matmuls
60%, SwiGLU product 14%, SDPA 11%, fused heads op 10%.

## Reproduce

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD MESH_DEVICE=P150
PY=./python_env/bin/python; M=models/demos/blackhole/pplx_embed_4b
TT_VISIBLE_DEVICES=4 $PY $M/demo/demo_bs1_isl512.py                     # 10 iterations, extended trace
bash $M/perf_tools/sustained_run.sh 32 6 30 pplx32                      # 30 iterations + tt-smi: cold best and sustained median
bash $M/perf_tools/sustained_run.sh 32 6 30 q3e32 "HF_MODEL=Qwen/Qwen3-Embedding-4B"
TT_VISIBLE_DEVICES=9 $PY $M/demo/eval_accuracy_tt.py                    # STS-B, pplx
TT_VISIBLE_DEVICES=9 HF_MODEL=Qwen/Qwen3-Embedding-4B $PY $M/demo/eval_accuracy_batched.py --batch 8 --pool last --eos
```
