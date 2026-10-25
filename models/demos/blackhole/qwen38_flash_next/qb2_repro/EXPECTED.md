# Expected numbers (reference run)

Machine: QuietBox 2 = 2x P300c, 4 Blackhole dies as a 1x4 line, FW bundle 19.12.0, KMD 2.11.0, 245 GB RAM.
Code: pr/58690 head 0fb7f9d95b3 on its own base (not the current branch). Profile qb2, allocated context 32768, 2026-10-22.
Each number is the median over the repeats (decode/sampled 3, concurrency 2, prefill 2 per point), server-side TTFT.
Run-to-run spread in the reference run: decode ms/tok within 0.1 ms at P1/P2; MTP steps (P3) vary by text and run (19.7-22.1 ms).
The client forces 256 output tokens (ignore_eos), thinking off. Concurrency "agg tok/s" = total tokens / wall time.

Branch re-run (2026-10-23, branch atupe/qwen38-flash-next = main 985c23b1bdb + #60469 + #59945 + this model dir, same QB2, fresh BF4 cache):
- Acceptance: json 96/96 PASS. Every first divergence is the same as in the table below.
- P1 and P2 (no MTP): every number within 1% of the tables below.
- P3 decode: 19.51 ms/tok greedy (51.3 tok/s), 20.82 sampled (48.0 tok/s). The time per MTP pass is the same (46.1 ms greedy).
  The run accepted more drafts (2.38 vs 2.26 tokens per pass), so tok/s is 4.6% higher. P3 prefill TTFT within 0.3%.
- P3 prefill rows: decode ms/tok moves by up to 15%, because each request decodes only 16 tokens. Use the TTFT only.
- P4a: c1 41.6 tok/s per user (+9.8%, MTP acceptance), c2 agg 57.9, c4 agg 97.7 (+0.7%).
- P4b: c2 agg 58.8 (-3.4%), c4 agg 92.8 (-1.6%), c8 agg 140.5 (+1.3%).
- Tolerance to use: 1% for P1/P2; 10% for MTP decode (P3, P4) because acceptance changes with the generated text.

Prefill plan change (2026-10-25, branch default `QWEN38_PREFILL_PLAN=cost`, `docs/PREFILL.md`):
- Set `QWEN38_PREFILL_PLAN=legacy` to reproduce this file exactly. The legacy replay gives the same divergence table.
- Under `cost`, the json gate is 96/96 PASS. Two first divergences change: math 56 -> 61, summary 75 -> 1.
  The summary change is a near tie at step 1 (top-2 logit margin 0.25). All other records are the same.
- Under `cost`, the P1 and P2 prefill TTFT rows drop (see `docs/PREFILL.md`). Decode rows, P3 and P4 do not change.

## Acceptance (json 96/96 against the CPU greedy records)

Gate: json record 96/96 -> PASS. First divergence index per record (token position where device and CPU differ):

| record | first divergence |
|---|---|
| json | none (96/96) |
| chat | 39 |
| code | 32 |
| fact | 15 |
| list | 46 |
| math | 56 |
| multilingual | 9 |
| prose | 13 |
| refactor | 23 |
| sky | 19 |
| story | 6 |
| summary | 75 |

A later divergence is better, an earlier one is worse. Reference decode speed in the replay: 37.6-39.0 tok/s per record.

## Perf

Columns: server TTFT in s, decode ms per token, server tok/s, prefill ms per prompt token, aggregate tok/s (concurrency only).
P1 = plain server. P2 = --prefill-slab 2048. P3 = --mtp 4 --prefill-slab 2048. P4a = lanes 4, mtp 4. P4b = lanes 8, mtp 3.
Notes: lanes (P4) are greedy-only; their TTFT is about 0.3 s higher than the plain server. MTP gain depends on the prompt text.

### P1_plain

decode (medians)

| point | TTFT s | decode ms/tok | tok/s |
|---|---|---|---|
| decode | 0.365 | 25.91 | 38.59 |

sampled (medians)

| point | TTFT s | decode ms/tok | tok/s |
|---|---|---|---|
| sampled | 0.369 | 26.00 | 38.47 |

concurrency (medians)

| point | TTFT s | decode ms/tok | tok/s | agg tok/s |
|---|---|---|---|---|
| c1 | 0.364 | 25.90 | 38.61 | 36.71 |
| c2 | 0.368 | 25.91 | 38.60 | 36.69 |
| c4 | 0.368 | 25.91 | 38.60 | 36.70 |

prefill (medians)

| point | TTFT s | decode ms/tok | tok/s | prefill ms/tok |
|---|---|---|---|---|
| isl1024 | 1.464 | 26.99 | 37.05 | 1.403 |
| isl4096 | 5.293 | 28.47 | 35.13 | 1.285 |
| isl8192 | 10.463 | 28.47 | 35.13 | 1.274 |
| isl16384 | 20.952 | 28.41 | 35.21 | 1.277 |
| isl31000 | 39.764 | 28.53 | 35.05 | 1.282 |

### P2_slab2048

decode (medians)

| point | TTFT s | decode ms/tok | tok/s |
|---|---|---|---|
| decode | 0.364 | 25.90 | 38.60 |

prefill (medians)

| point | TTFT s | decode ms/tok | tok/s | prefill ms/tok |
|---|---|---|---|---|
| isl1024 | 1.459 | 26.96 | 37.09 | 1.398 |
| isl4096 | 3.492 | 28.43 | 35.17 | 0.846 |
| isl8192 | 4.810 | 28.45 | 35.15 | 0.584 |
| isl16384 | 7.434 | 28.44 | 35.16 | 0.452 |
| isl31000 | 10.342 | 28.41 | 35.20 | 0.333 |

### P3_mtp4_slab2048

decode (medians)

| point | TTFT s | decode ms/tok | tok/s |
|---|---|---|---|
| decode | 0.373 | 20.40 | 49.02 |

sampled (medians)

| point | TTFT s | decode ms/tok | tok/s |
|---|---|---|---|
| sampled | 0.374 | 21.78 | 45.91 |

prefill (medians)

| point | TTFT s | decode ms/tok | tok/s | prefill ms/tok |
|---|---|---|---|---|
| isl4096 | 3.684 | 27.67 | 36.62 | 0.892 |
| isl16384 | 8.204 | 21.21 | 47.16 | 0.499 |
| isl31000 | 11.796 | 24.36 | 44.18 | 0.380 |

### P4a_lanes4_mtp4

concurrency (medians)

| point | TTFT s | decode ms/tok | tok/s | agg tok/s |
|---|---|---|---|---|
| c1 | 0.675 | 26.42 | 37.87 | 34.54 |
| c2 | 1.069 | 28.75 | 34.91 | 59.52 |
| c4 | 1.815 | 32.51 | 30.77 | 96.99 |

### P4b_lanes8_mtp3

concurrency (medians)

| point | TTFT s | decode ms/tok | tok/s | agg tok/s |
|---|---|---|---|---|
| c1 | 0.677 | 26.09 | 38.34 | 34.91 |
| c2 | 1.049 | 28.86 | 34.65 | 60.89 |
| c4 | 1.799 | 33.15 | 30.18 | 94.25 |
| c8 | 3.359 | 41.59 | 24.07 | 138.73 |
