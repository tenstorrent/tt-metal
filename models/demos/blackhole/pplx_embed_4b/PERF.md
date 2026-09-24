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

Effects are the same-chip A/B measured at landing (best of 10 at the cold clock; the sustained metric was
adopted late in the effort). e2e ms after each step: bs1 / bs8 / bs16 / bs32.

| # | change | mechanism | after |
|---|---|---|---|
| 1 | `minimal_matmul` output subblock 1×8 (was 1×1) | the batched matmuls ran one tile per DST pass | bs8 −12%, bs16 −13%, bs32 −18% |
| 2 | `in0_block_w` cap 8 → 38; head-split QKV and concat as model-local `generic_op`s | FF2 blocks were pinned at 2; head split scatters the fused QKV activation straight into heads | bs1 −4.4 ms and −2.7 ms |
| 3 | timed path = extended trace | pooling + I/O inside the replay; the 3.4 ms host bubble at bs1 is gone | 25.2 / 155.6 / 288.7 / 543.5 |
| 4 | fused head-split + Q/K RMSNorm + RoPE, emitting bfp8 Q and K/V | one pass over QKV replaces create_heads, q_norm, k_norm, two RoPE ops and a typecast | 23.7 / 135.3 / 250.4 / 474.3 |
| 5 | QKV projection writes bfp8; merged core ranges for the generic ops | 26 MB less per layer at bs8; ≈0.4 µs per core range per launch | 23.4 / 126.7 / 240.6 / 455.8 |
| 6 | batched SDPA on all 120 cores with a 512-token K chunk | one K chunk, no re-read | 23.3 / 126.2 / 239.6 / 450.6 |
| 7 | fused residual add + RMSNorm (bs16+), then row-split over 4–5 cores per row (bs8/16/32) | one op emits the residual sum and the normalised tensor: 4 DRAM passes → 2, 87% of DRAM bandwidth at bs32 | 234.8 / 443.5, later 118.5 / 216.7 / 428.1 |
| 8 | fused-SwiGLU matmul blocks at bs16, plain matmul blocks at bs8 | in-model block sweeps | 123.4 / 228.3 |
| 9 | SwiGLU product as one `generic_op` at bs32 (`silu_mul`) | SiLU then dest-reuse multiply | 438.1 |
| 10 | bs1: legacy 2D matmuls on 12×8 after fixing the factory's DRAM-bank walk, coalesced weight reads, SDPA q256 | the 2D multicast kernel at M=512 beats `minimal_matmul` by 53–65%; the bug had blocked grids wider than 8 | **17.7** / 123.4 / 228.3 / 438.1 |
| 11 | bs>1: SDPA 12×8 at bs8, DRAM-interleaved QKV/WO/W1/W3 weights at bs32 | fewer, fuller SDPA work units; interleaved weights read faster at M=16384 | 17.7 / 118.5 / 216.7 / 428.1 |
| 12 | SDPA writes the `[B, 1, S, H·d]` layout directly (new op flag `output_heads_concat`) | tile-id remap in the writer; the concat pass (142 MB per layer at bs32) is gone | 17.6 / 115.3 / 221.0 / 425.5 |
| 13 | bs1: concat-free SDPA output too; residual adds written in the norm's block-shard layout | the per-layer concat op and all 72 interleaved-to-sharded ops disappear | 17.3 (sustained 18.3 → 17.7) |

Configuration lives in `demo/_common.py::apply_workload_env` (per-batch defaults, every knob overridable
from the shell); the kernels in `tt/custom_ops/`; the shared-code changes in `models/tt_transformers/tt/`
(prefill norm, matmul grids and blocks, weight layouts) and in the SDPA op and the 2D matmul factory.

## Where the time goes now

Device-profiled bs32 (365 ms of kernels at the nominal clock; sustained e2e 446 ms): matmuls 65%, SwiGLU
product 13%, fused add+RMSNorm 8%, SDPA 8%, fused heads op 6%. bs1 (16.6 ms of kernels, 509 ops): matmuls
60%, SwiGLU product 14%, SDPA 11%, fused heads op 10%.

## What is left

Every wiring- or config-level item is at or near its ceiling; the remainder sits in three kernels, filed on
tenstorrent/tt-metal with repro scripts:

- `minimal_matmul` at K=2560, N=9728 runs at 74% of the LoFi peak (85% is the practical ceiling): ≈ −18 ms at
  bs32, and the bs8 fused-SwiGLU matmul runs at 43%: ≈ −26 ms at bs8 — #57626.
- SwiGLU epilogue inside FF1/FF3 instead of the SFPU-bound product op: ≈ −8 ms at bs32 — #57627.
- SDPA at 43% of its byte floor at bs32 and on 64 of 120 cores at bs1: −8…−16 ms at bs32 — #57628.

Details: `doc/POSITIVE_RESULTS.md` (every landing with its effect), `doc/NEGATIVE_RESULTS.md` (every rejected
experiment with numbers, §0–§50), `doc/PERF_GUIDE.md` (how to run, measure and profile).

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
