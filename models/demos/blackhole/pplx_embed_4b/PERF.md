# pplx-embed-v1-4B and Qwen3-Embedding-4B on Blackhole P150: performance, baseline to now

All numbers are **sustained latency at ISL 512**: the median of iterations 15–29 of a 30-iteration run,
after the board's power manager has settled the clock (≈1.1–1.3 GHz under load), on one P150 of a Galaxy
(12×10 = 120 worker cores; a p150a card exposes 13×10). Timed path is the extended trace: forward + pooling
+ I/O in one replay. Branch `arg/embed-4b-pplx-qwen3`.

## Headline

| batch | baseline | pplx-embed-4B now | Qwen3-Embedding-4B now | speedup | H200 | × H200 (pplx) | tok/s (pplx) |
|---|---|---|---|---|---|---|---|
| 1 | 45.773 ± 0.160 ms | **16.5 ms** | 17.4 ms | 2.77× | 5.44 ms | 3.03× | 31.0k |
| 8 | 190.065 ± 4.527 | **116.8** | 117.7 | 1.63× | 33.08 | 3.53× | 35.1k |
| 16 | 375.817 ± 5.810 | **216.7** | 220.1 | 1.73× | 67.23 | 3.22× | 37.8k |
| 32 | 726.944 ± 2.552 | **433.7** | 433.7 | 1.68× | 139.15 | 3.12× | 37.8k |

- **Baseline** is the reference measurement of pplx-embed-4B on Blackhole P150 as provided by the customer
  (mean ± spread, ms). The two models share one code path; Qwen3-Embedding-4B differs only by causal attention
  and last-token pooling, which the stack switches on from `HF_MODEL`.
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
| 13 | bs1: SDPA packs each GQA group's 4 query heads as one head (`pack_gqa_heads`, q192 on 11×8); the fused heads kernel batches every norm/RoPE phase across a unit's heads and keeps its constants resident in L1; both norms keep their block-shard output for the QKV/FF1/FF3 matmuls | K/V stream once per KV head instead of once per query head; 45 phase set-ups per unit → 9; the 72 sharded-to-interleaved ops are gone | **15.9** / 115.3 / 221.0 / 425.5 |
| 14 | bs8–32: SDPA keeps K/V in a core's buffers across its Q chunks of the same (batch, KV head) (`reuse_kv`, q128); the fused add+RMSNorm's short-lived operands (WO/FF2 outputs, both norm outputs, the post-attention sum at bs8/16) live in L1 | each core reads a KV head's K/V once instead of once per Q chunk (142 MB → 36 MB per call at bs16) and finer chunks fill the grid; the DRAM-bound norm halves its traffic (435 → 255 µs per call at bs32) | 15.9 / **110.5** / **212.9** / **416.2** |

Sustained after step 14: **16.5 / 116.8 / 216.7 / 433.7 ms**. Configuration lives in
`demo/_common.py::apply_workload_env` (per-batch defaults, every knob overridable from the shell), the kernels in
`tt/custom_ops/`, the shared-code changes in `models/tt_transformers/tt/`, the SDPA op and the 2D matmul factory.

## Data parallel on 32 chips

`demo/dp32_multiprocess.py --num-devices 32 --batch-size B --iterations 30 --warmup 5`: one resident model per
chip, workers released together after warm-up, per-chip latency = median of 30 extended-trace iterations,
throughput gated by the slowest chip (pplx-embed-4B, ISL 512, 32/32 chips active in every run).

| per-chip batch | global batch | per-chip median | slowest chip | vs one chip sustained | embeddings/s | tokens/s | scaling vs 32 × one chip |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 17.1 ms | 17.5 ms | +4% | 1,833 | 0.94 M | 95% |
| 4 | 128 | 70.1 | 72.0 | +2% | 1,777 | 0.91 M | 96% |
| 8 | 256 | 117.6 | 123.1 | +0.7% | 2,080 | 1.06 M | 95% |
| 16 | 512 | 218.4 | 235.0 | +0.8% | 2,179 | 1.12 M | 92% |
| 32 | 1,024 | 432.5 | 457.2 | −0.3% | 2,239 | 1.15 M | 95% |

The per-chip median matches the single-chip sustained numbers, so the chips do not interfere; the 4–6% lost
against ideal scaling is chip-to-chip spread (fastest to slowest chip: 16.0–17.7 ms at bs1, 402–461 ms at bs32),
which gates the synchronous aggregate.

## Where the time goes now

Share of device kernel time per batch (Tracy, signposted trace replay, one P150). The SwiGLU column is empty at
bs 8 and 16 because there the product runs inside the fused SwiGLU matmul.

| Batch | Matmul | SDPA | Fused heads | Norm + residual | SwiGLU product | Matmul + SDPA | Kernel sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 66.6% | 6.8% | 6.6% | 7.3% | 12.4% | 73.4% | 15.1 ms |
| 8 | 80.5% | 7.3% | 6.3% | 5.8% | 0.0% | 87.8% | 101.0 ms |
| 16 | 82.4% | 6.3% | 6.1% | 5.2% | 0.0% | 88.6% | 187.7 ms |
| 32 | 68.5% | 6.0% | 6.1% | 5.1% | 14.2% | 74.5% | 345.8 ms |

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
