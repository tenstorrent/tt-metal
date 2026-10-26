# Toto 2.0 optimization candidates

Contract (`/home/andy/avan/fleet/toto2/CONTRACT.md`, version 1.2; exit also after any round under 5 % in total, U4):

| item | value |
|---|---|
| target | `forward` on Toto-2.0-2.5B-FT, shape `paper` (1024 univariate series x 2048 -> 64, 66 tokens per row), precision `accuracy` (bf16 activations and weights, HiFi2+ matmuls, RMSNorm statistics in fp32). Device time (median of 5 `forward_device` calls, each after a 0.3 s idle) is primary; steady wall time (median of 5 end-to-end calls) is secondary |
| guards (<= 3 % regression) | `long_context` (2.5B-FT, 32 x 8192 -> 1024, accuracy), `multivariate` (2.5B-FT, 128 x 8 variates x 2048 -> 64, accuracy), `paper` / `performance` (bfp8 weights, LoFi), `paper_313m` (313m, 1024 x 2048 -> 64, accuracy) |
| correctness floor | full PCC suite passes in one `pcc` job; target `pcc_normalized` >= 0.999 and `pcc_final` >= 0.99; performance guard >= 0.995 / 0.98; no candidate more than 0.0005 below the round-0 target `pcc_normalized` without an X gate |
| acceptance gate | >= 2 % target device time (wall for host-only changes), no guard regressing by > 3 %, every floor holding: `compare_bench.py <cand> <best> --target paper:accuracy --pcc-key pcc_normalized --pcc-floor 0.995`, then the target floor by hand |
| exit | two consecutive rounds each < 3 %; or matmuls > 65 % of FLOPs and non-matmul ops < 15 % of device time; or 6 rounds |
| commits | one per accepted change, `commit_unsigned.sh`, one short lowercase line, no trailers, model paths only |

Headline: `forward`, `paper`, `accuracy`, median of 5 steady runs; device time after a 0.3 s idle. Chip 2 (`7FD83D0A1372`).

| round | candidate | target wall ms | target device ms | vs best | guards | PCC (normalized) | decision | evidence |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline: fp32 dest acc, HiFi4 embed, fp32 head out, realistic-mix goldens | 8412 | 8209 | - | long_context 614 ms, multivariate 8183 ms, paper/performance 8006 ms, paper_313m 1241 ms | 0.99993 | baseline | iter_00, jobs 553 (suite 72 passed), 554 (bench), 526 (profile) |
