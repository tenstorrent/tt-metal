<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
SPDX-License-Identifier: Apache-2.0
-->
# Results: Qwen3-8B sampling on the x280 cores (P300, one chip)

Hardware: P300 (Blackhole), chip 0 only, 110 Tensix cores (a P150 has 130), L2CPU tile 0 (x280 harts 0-3 at
1750 MHz). Model: Qwen3-8B from tt_transformers, performance precision, unmodified. Firmware: sampling firmware,
stream-capable build (the accepted default image), RVV sampling library.

## Benchmark
128 input tokens, 256 output tokens, 3 warm-up + 10 measured runs in one process (re-prefilled), median ms per
output token (tokens/s/user), decode only. Every x280 row produced the same tokens as the host C reference.
x280 columns: this tree (`bench_row.sh`, default image `bh-irq-sampling/fw.bin`); host and stock-Tensix columns: the
same driver and protocol on the same chip with the tt-metal base of this tree.

| Batch | Setting | Host sampling | Stock Tensix sampling | x280 host-mediated | x280 device loop |
|---|---|---|---|---|---|
| 1 | T0.7/k50/p0.9 | 27.47 (36.4) | 26.85 (37.2), k32 | 26.34 (38.0) | **26.10 (38.3)** |
| 1 | greedy | 27.60 (36.2) | 26.74 (37.4) | - | **26.05 (38.4)** |
| 32 | T0.7/k50/p0.9 | 32.13 (31.1), 8 threads | **30.51 (32.8)**, k32 | 31.51 (31.7) | 30.94 (32.3), streamed |
| 32 | greedy | 31.98 (31.3) | **30.40 (32.9)** | - | 30.91 (32.4), streamed |
| 1 | T1.0/k0/p1.0 (K 1024) | 28.00 (35.7) | not served (k <= 32) | - | **26.62 (37.6)** |
| 32 | T1.0/k0/p1.0 (K 1024) | **33.11 (30.2)**, 8 threads | not served (k <= 32) | - | 34.84 (28.7), streamed |

Batch 1: the device loop is the fastest path (0.7 ms/token ahead of stock Tensix sampling, 1.4 ahead of host
sampling). Batch 32: 0.4-0.5 ms/token behind stock Tensix sampling, 1.1-1.2 ahead of host sampling; the remaining
cost is the uncached read of 32 logits rows on the x280 (two reader slots).

What the table says: sampling is 2-3 % of a decode step, so no row is far from any other, and the L2CPU rows are
within run-to-run noise of stock Tensix sampling. The rows that stock Tensix sampling does not serve (top-k above
32, here top-k 0 = the 1024-candidate cap, plain multinomial) show the cost of a full top-k on one tile: 0.5 ms per
step at batch 1 (ahead of the host C reference by 1.4 ms) and 3.9 ms at batch 32, where the host C reference on
eight Zen 5 threads is 1.7 ms ahead of one tile (four harts at 1750 MHz). Splitting the batch over the four tiles
(the multi-tile change that follows this PR) brings that row to 31.20 ms/token.

## x280 time per step (firmware timestamps, median / p99, us; device loop not streamed)
| Case | Wake-up | Logits read | Sampling | Write-back | Wait workers | Publish/other | Wake -> publish |
|---|---|---|---|---|---|---|---|
| b1 greedy | 3.3 / 5.3 | 0 (in place) | 28.0 / 46.9 | 0.1 / 0.2 | - | 3.5 / 3.6 | 31.7 / 50.6 |
| b1 T0.7 k50 p0.9 | 3.3 / 3.7 | 0 | 87.0 / 92.9 | 0.1 / 0.2 | - | 3.5 / 3.6 | 90.6 / 96.6 |
| b1 T1.0 k0 p1.0 (cap 1024) | 3.3 / 3.9 | 0 | 629.4 / 671.3 | 0.1 / 0.3 | - | 3.5 / 3.6 | 633.1 / 675.0 |
| b1 T0.6 k20 p0.95 | 3.3 / 3.8 | 0 | 73.0 / 76.2 | 0.1 / 0.2 | - | 3.5 / 3.7 | 76.7 / 80.0 |
| b32 all T0.7 k50 p0.9 | 8.0 / 14.2 | 587.7 / 1276.3 | 475.2 / 1020.1 | 1.1 / 6.3 | 68.2 / 79.7 | 4.4 / 5.9 | 1136.7 / 2371.6 |
| b32 mixed settings | 8.6 / 18.2 | 664.0 / 797.5 | 1415.1 / 1498.8 | 1.0 / 1.3 | 192.4 / 362.2 | 4.4 / 4.7 | 2276.2 / 2380.2 |
At batch 32 the per-hart columns are hart 0's eight users. Notify-end -> wait-release (Tensix wall clock) equals
the x280 wake-to-publish time; doorbell wake-up <= 2 us at batch 1. Streamed batch 32 (mixed): doorbell -> release
2.42 ms mean / 2.61 ms max over a 20-run soak; the push overlaps the x280.
Device tail at batch 32: forward 28.78 ms, untilize 0.06 ms, push of 32 rows 0.66 ms (1 core), ~1.5 us per program.

## Acceptance evidence (Qwen3-8B, this tree)
What is compared: the full token list per user (prefill token + 256 decode tokens) of the device loop against the
host C library sampling the same device logits in the host-mediated loop, same process, prompts, prefill, per-user
settings, seeds and step indices.
| Run | Result |
|---|---|
| batch 1, 5 prompts x 256 tokens, greedy / T0.7 k50 p0.9 / T1.0 k0 p1.0 / T0.6 k20 p0.95 | 1285/1285 tokens identical for each setting (25.99 / 26.05 / 26.60 / 26.03 ms/token) |
| batch 32, 256 tokens, mixed settings (u % 4, seed 1234 + u), 4 harts, streamed | 32/32 users, 8224/8224 tokens identical (31.62 ms/token) |
| negative control: one user's seed + 1 on the x280 side only | exactly that user diverges (decode step 5), 31 identical |
| host-mediated loop, batch 1, 5 x 256, T0.7 k50 p0.9: x280 vs host C library | 1280/1280 identical (compare:x280,hostref) |
| demo, batch 1 / batch 32 mixed | coherent text; 26.05 / 31.62 ms/token |
| host activity during decode | one `execute_trace` per step + asynchronous ring polls |
| soak (development tree, same kernels and firmware protocol): batch 32 mixed, 20 x 256 tokens in one process | 164,480/164,480 identical, 0 wait timeouts, 0 firmware errors |

## One timeout: warm restart and retry (`--retry-on-timeout 16 --inject-timeout-at 100`, batch 32 mixed)
Test hook: at decode step 100 hart 0 is made to hang (wfi with interrupts off, mailbox INJECT WFIPARK). The queued
wait ops hit their 50 ms bound, the host sees the wait status word, keeps the published tokens, warm-restarts the
firmware and re-issues the failed step with its step index.
| Model | Tokens after recovery | Recoveries | Cost |
|---|---|---|---|
| Qwen3-0.6B | 32/32 users, 8224/8224 identical | 2: warm restart via RNMI 200.9 ms, then the first retried request timed out once more and a second (cooperative) warm restart 0.6 ms recovered | run 16.36 vs 9.31 ms/token: +1.8 s for 256 steps |
| Qwen3-8B | 32/32 users, 8224/8224 identical | same pattern: 200.9 ms + 0.7 ms restarts | run 41.10 vs 31.62 ms/token: +2.4 s for 256 steps |
The cost is dominated by the queued steps that run into the 50 ms wait bound (the rest of the failed chunk and the
retried chunk): about chunk x (step time + 50 ms) per failed chunk, plus 0.2 s for the RNMI restart. Smaller chunks
or a shorter wait bound reduce it. Open: why the first request after the RNMI-path restart is not served within the
bound (a second, cooperative restart always fixed it).

## Deviations of the comparison
- P300 chip with 110 Tensix cores (one chip of the board used as a single-chip system), not a P150 (130 cores).
- Stock Tensix sampling runs at top-k 32 (its sampler's cap) and without a seed; no penalties in any arm.
- Host sampling at batch 32 uses 8 threads (1 thread: 33.88 ms/token).

## Limits
Batch 1 and 32 only (the firmware's QEMU suite covers 4, 9, 13); Qwen3-8B and Qwen3-0.6B; <= 256 generated
tokens; no stop conditions; one chip, one L2CPU tile.
