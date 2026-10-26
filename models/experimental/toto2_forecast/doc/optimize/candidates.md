# Toto 2.0 optimization candidates

Contract (`/home/andy/avan/fleet/toto2/CONTRACT.md`, version 1.2; exit also after any round under 5 % in total, U4):

| item | value |
|---|---|
| target | `forward` on Toto-2.0-2.5B-FT, shape `paper` (1024 univariate series x 2048 -> 64, 66 tokens per row), precision `accuracy` (bf16 activations and weights, HiFi2+ matmuls, RMSNorm statistics in fp32). Device time (median of 5 `forward_device` calls, each after a 0.3 s idle) is primary; steady wall time (median of 5 end-to-end calls) is secondary |
| guards (<= 3 % regression) | `long_context` (2.5B-FT, 32 x 8192 -> 1024, accuracy), `multivariate` (2.5B-FT, 128 x 8 variates x 2048 -> 64, accuracy), `paper` / `performance` (P1 per D2: bfp8 weights, HiFi2 matmuls and SDPA, fp32 dest acc, HiFi4 embed, fp32 head out), `paper_313m` (313m, 1024 x 2048 -> 64, accuracy) |
| correctness floor | `tests/golden/goldens.py` GATE_FLOORS / SMALL_CHECKPOINT_FLOORS / DRIFT as set by D2 (two-sided spread_ratio_p50, degenerate-case range, min / max / inactive floor-only), with D3: 4m/22m p50 0.9-1.1 and p99 1.2 (acc) / 1.5 (perf); 313m accuracy p99 1.25; D4 ruling 2: performance = P6 (P1 + bf16 / HiFi2 quantile head), the 313m INFO set is empty and every D2 floor is gated there; D4 ruling 1: an all-degenerate case (`rows_degenerate == rows`) has no drift except abs_err_max <= 1.5x its fixed record; drift records are bench 684 / suite 683 (accuracy) and the D4 condition 2 jobs (performance); `check_gates.py --suite` drift-checks 313m and 2.5B-FT suite records; `check_gates.py <cand> iter_00/bench.json`; full PCC suite passes in one `pcc` job; target `pcc_normalized` >= 0.999 and `pcc_final` >= 0.99; performance guard >= 0.995 / 0.98; no candidate more than 0.0005 below the round-0 target `pcc_normalized` without an X gate |
| acceptance gate | >= 2 % target device time (wall for host-only changes), no guard regressing by > 3 %, every floor holding: `compare_bench.py <cand> <best> --target paper:accuracy --pcc-key pcc_normalized --pcc-floor 0.995`, then the target floor by hand |
| exit | a round under 5 % in total (U4, skill section 5 item 6); performance ships at G3 only if >= 5 % faster than accuracy on paper, passing every floor on 2.5B-FT, and passing the D2 band floors on 313m with no INFO exemption (D2, D3); otherwise accuracy ships; or matmuls > 65 % of FLOPs and non-matmul ops < 15 % of device time; or 6 rounds |
| commits | one per accepted change, `commit_unsigned.sh`, one short lowercase line, no trailers, model paths only |

Headline: `forward`, `paper`, `accuracy`, median of 5 steady runs; device time after a 0.3 s idle. Chip 2 (`7FD83D0A1372`).

| round | candidate | target wall ms | target device ms | vs best | guards | PCC (normalized) | decision | evidence |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline: fp32 dest acc, HiFi4 embed, fp32 head out, realistic-mix goldens; performance = P1 (D2) | 8367 | 8204 | - | long_context 615 ms, multivariate 8193 ms, paper/performance (P1) 8108 ms, paper_313m 1241 ms | 0.99994 | baseline | iter_00, jobs 683 (suite on 9bc21877b2c), 684 (bench, the drift baseline `iter_00/bench.json`; job 635's is `bench_job635.json`), 632 (profile); D3 floor code re-checked by jobs 714 (suite) / 715 (bench) on 08164306d04 |

Disclosures (D2 item 6, D3 condition 5):
- Policy sweep job 580 ended "7 failed, 21 passed": every single_sparse bench record, accuracy included (pcc_normalized 0.952), failed the bench's contract `pcc_normalized` assert, which is ill-conditioned on an all-degenerate row; the bench now judges such cases by abs_err_max plus the degenerate p50 range (D2 item 4).
- Sweeps 580 and 628 (broken-output scoring) ran before commit 3100146 that holds their scripts.
- SwiGLU sweep job 698 started at 14:11:49, before `sweep_swiglu.py` was committed (14:12:24). Its results are discarded (D3 defect 1); it was cancelled and rerun on a committed sha.
| 1 | R1-3 fused SwiGLU (`minimal_matmul(fuse_swiglu=True)`; blocks 8/32/2/2/2 on d_model 2048, 8/16/4/1/4 on 1024; sweep job 720) | 6359 | 6170 | -24.8 % | long_context 504 ms (-18.0 %), multivariate 6157 ms (-24.9 %), paper/performance 6090 ms (-24.9 %), paper_313m 967 ms (-22.1 %) | 0.99994 | pending X (check_gates drift fails only on the 1-row all-degenerate single_sparse case) | iter_01, aeebce5ef19, suite 743 (75 passed), bench 744 |

R1-3 notes (D4 conditions 4 and 3):
- Precision change, not order-only (PLAN v1.14): the fused op computes silu(u) * g in the fp32 dest and rounds once to bf16; round 0 rounded silu(u) to bf16, then the bf16 mul rounded again.
- Suite moves vs `iter_00/pcc_metrics.json` (job 683), all inside the floors and now drift-checked on 313m / 2.5B-FT (`check_gates.py --suite`, PASS on `iter_01/pcc_swiglu.json`): 22m single_sparse accuracy pcc_normalized 0.9635 -> 0.8501 (all-degenerate, floors only); 313m single_sparse accuracy spread_ratio_p50 1.437 -> 1.236 (all-degenerate, inside 0.5-2.0); 4m / 22m missing accuracy p50 +0.020 / +0.039 (inside 0.9-1.1); 313m missing_16 accuracy abs_err_max 0.022 -> 0.078 (3.5x; below, condition 3).
- Moves within margin (bench 744 vs 684): long_context spread_ratio_inactive_p50 1.73 -> 1.90 (accuracy) and 1.69 -> 1.89 (performance), about 0.1 under the 2.0 ceiling; paper accuracy pcc_row_min_active 0.986 -> 0.975 (floor 0.9 past 64 rows).
- single_sparse broken outputs under D4 ruling 1: the full collapse scores abs_err_max 0.00214 and the 0.3x band 0.00150, under both the floors and a 1.5x abs_err_max ceiling; both are caught only by the 0.5-2.0 spread_ratio_p50 range, which stays gated. Accepted weakness: a single near-constant row whose band moves within 0.5x-2.0x passes.
- Table fixes (D4 defect 5): the round-0 "paper/performance (P1) 8108 ms" is job 684's number; job 715 measured 8096.3 ms. The X gate's single_sparse performance drift list also missed pcc_row_p01_active. single_sparse accuracy device time 55.2 -> 60.6 ms (+9.9 %) while performance went 58.5 -> 37.8 ms: one row (96 tokens) per forward is launch- and weight-read-bound and not a guard; not investigated further.
- Job 721 (policy variants) started at 14:44:01 on 29625d703fa, 62 s before aeebce5ef19 was committed: it ran the unfused FFN. Superseded by the D4 condition 2 jobs.
- P6 on paper_313m: spread_ratio_inactive_p01 0.419 against the 0.4 floor (P1: 0.576), the closest margin of the P6 records in job 721.
