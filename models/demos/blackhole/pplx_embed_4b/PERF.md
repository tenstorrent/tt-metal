# pplx-embed-v1-4B and Qwen3-Embedding-4B on Blackhole P150: performance, baseline to now

All numbers are **sustained latency at ISL 512**: the median of iterations 15–29 of a 30-iteration run,
after the board's power manager has settled the clock (≈1.15–1.35 GHz under load at the 160 W firmware power cap
the model applies when the device opens; the board default is 130 W), on one P150 of a Galaxy
(12×10 = 120 worker cores; a p150a card exposes 13×10). Timed path is the extended trace: forward + pooling
+ I/O in one replay. Branch `arg/embed-4b-pplx-qwen3`.

## Headline

| batch | baseline | pplx-embed-4B now | Qwen3-Embedding-4B now | speedup | H200 | × H200 (pplx) | tok/s (pplx) |
|---|---|---|---|---|---|---|---|
| 1 | 45.773 ± 0.160 ms | **15.7 ms** | 16.8 ms | 2.92× | 5.44 ms | 2.89× | 32.6k |
| 8 | 190.065 ± 4.527 | **86.2** | 88.5 | 2.20× | 33.08 | 2.61× | 47.5k |
| 16 | 375.817 ± 5.810 | **162.9** | 168.8 | 2.31× | 67.23 | 2.42× | 50.3k |
| 32 | 726.944 ± 2.552 | **325.9** | 325.7 | 2.23× | 139.15 | 2.34× | 50.3k |

- **Baseline** is the reference measurement of pplx-embed-4B on Blackhole P150 as provided by the customer
  (mean ± spread, ms). The two models share one code path; Qwen3-Embedding-4B differs only by causal attention
  and last-token pooling, which the stack switches on from `HF_MODEL`.
- **Accuracy** was re-checked after every landing: STS-B Spearman 0.8161 (pplx, bucketed bs1 path),
  0.812–0.817 through the batched paths; Qwen3-Embedding-4B 0.807–0.819 (last token + EOS). Every landed
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
| 15 | bs8–32: the fused SwiGLU is applied on `minimal_matmul`'s pack thread in each block's last K block (K_block 40), and bs32 runs the fused kernel too; QKV output kept in L1 at bs8/16; at bs32 QKV + heads run in two half-batch chunks with L1 outputs; norm output preallocated at the top of L1 | the SwiGLU SFPU pass overlaps the math thread's next subblock instead of a serialized epilogue (fused FF13 at bs16 2688 → 1691 µs, and bs32's FF1 + FF3 + product 4684 → 3248 µs); the heads op reads its input from L1 (bs16 322 → 210 µs) | 15.9 / **87.7** / **172.8** / **360.4** |
| 16 | bs8–32: SDPA sums the softmax rows on the math thread instead of the pack thread; the fused-SwiGLU sigmoid is sized for the bfp8 output; `minimal_matmul` output writers never defer to K block 0; the heads op writes K/V to L1 at bs8/16 (pplx-embed; the causal Qwen3 SDPA keeps K/V in DRAM) and bs32 runs QKV in four quarter-batch chunks | the pack thread was SDPA's slower thread (per call bs8/16/32 179/286/600 → 169/266/566 µs); 71% less SFPU work in the SwiGLU epilogue; fused FF13 at bs16/32 1686/3236 → 1650/3139 µs; SDPA reads K/V from L1 (−13–15% per call) | 16.5 / **90.8** / 180.7 / **379.6** |
| 17 | firmware power cap raised from the board's 130 W to 160 W for this model (`QWEN_TDP_LIMIT_WATTS`, applied by `apply_workload_env` when the device opens; 0 restores the board default) | from bs 8 up the fused kernels are power-limited at 130 W (settled clock ≈1.0–1.1 GHz); at 160 W the clock settles at ≈1.15–1.35 GHz and the chip-to-chip spread narrows | **15.7** / **86.2** / **162.9** / **325.9** |

Sustained after step 17: **15.7 / 86.2 / 162.9 / 325.9 ms**. At the board's 130 W cap, step 16's bs16
gain is eaten by the lower settled clock (cold best 172.8 → 166.9 ms, sustained unchanged). Configuration lives in
`demo/_common.py::apply_workload_env` (per-batch defaults, every knob overridable from the shell), the kernels in
`tt/custom_ops/`, the shared-code changes in `models/tt_transformers/tt/`, the SDPA op and the 2D matmul factory.

## Data parallel on 32 chips

`demo/dp32_multiprocess.py --num-devices 32 --batch-size B --iterations 30 --warmup 5`: one resident model per
chip, workers released together after warm-up, per-chip latency = median of 30 extended-trace iterations,
throughput gated by the slowest chip (pplx-embed-4B, ISL 512, 32/32 chips active in every run).

| per-chip batch | global batch | per-chip median | slowest chip | vs one chip sustained | embeddings/s | tokens/s | scaling vs 32 × one chip |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 15.6 ms | 15.8 ms | −1% | 2,029 | 1.04 M | 99% |
| 8 | 256 | 87.2 | 90.5 | +1% | 2,829 | 1.45 M | 95% |
| 16 | 512 | 166.6 | 181.0 | +2% | 2,829 | 1.45 M | 90% |
| 32 | 1,024 | 326.6 | 343.5 | +0% | 2,981 | 1.53 M | 95% |

The per-chip median stays within a few percent of the single-chip sustained numbers, so the chips do not interfere;
the 1–10% lost against ideal scaling is chip-to-chip spread (fastest to slowest chip: 15.4–16.1 ms at bs1, 308–344 ms at bs32),
which gates the synchronous aggregate. At the board's 130 W cap the spread was 4–14%; most of it was power throttling.

## Where the time goes now

Share of device kernel time per batch (Tracy, signposted trace replay, one P150). The SwiGLU column is empty at
bs 8 to 32 because there the product runs inside the fused SwiGLU matmul.

| Batch | Matmul | SDPA | Fused heads | Norm + residual | SwiGLU product | Matmul + SDPA | Kernel sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 66.6% | 6.8% | 6.6% | 7.3% | 12.4% | 73.4% | 15.1 ms |
| 8 | 78.7% | 7.9% | 5.4% | 7.8% | 0.0% | 86.6% | 76.3 ms |
| 16 | 81.0% | 6.7% | 5.3% | 6.8% | 0.0% | 87.7% | 142.2 ms |
| 32 | 80.9% | 6.5% | 6.0% | 6.4% | 0.0% | 87.4% | 278.0 ms |

## Reproduce

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD MESH_DEVICE=P150
PY=./python_env/bin/python; M=models/demos/blackhole/pplx_embed_4b
TT_VISIBLE_DEVICES=4 $PY $M/demo/demo_bs1_isl512.py                     # 10 iterations, extended trace
bash $M/perf_tools/sustained_run.sh 32 6 30 pplx32                      # 30 iterations + tt-smi: cold best and sustained median
bash $M/perf_tools/sustained_run.sh 32 6 30 q3e32 "HF_MODEL=Qwen/Qwen3-Embedding-4B"
bash $M/perf_tools/sustained_run.sh 32 6 30 pplx32_130w "QWEN_TDP_LIMIT_WATTS=130"  # at the board's 130 W power cap
TT_VISIBLE_DEVICES=9 $PY $M/demo/eval_accuracy_tt.py                    # STS-B, pplx
TT_VISIBLE_DEVICES=9 HF_MODEL=Qwen/Qwen3-Embedding-4B $PY $M/demo/eval_accuracy_batched.py --batch 8 --pool last --eos
```
