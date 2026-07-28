# Distributed affine prefix optimization log

Target: Kimi-K3, `B=1`, `T=5120`, `SP=4`, `TP=2`, real FP32 affine-state
payload `[1,48,128,128]` per device. Acceptance is source-derived floor
efficiency >=60% on LoudBox.

Branch: `codex/kimi-affine-prefix-perf`

Base: `b9d91959aa3d86ae5c501a9daa170166613b4b28`
(`origin/mvasilijevic/codex/kimi-linear-kda`, fetched 2026-07-28).

## Baseline and ranking

Existing validated profiler capture: ten trace replay sessions 2--11,
representative evidence session 7. Exact function measured time is
10,535.604 us; projected floor is 696.180 us; efficiency is 6.608%. The 60%
threshold is `696.180 / 0.60 = 1,160.300 us`.

The stage medians do not sum to the exact-function median because each stage is
collapsed independently across devices/replays. Ranking is still valid for
selecting the first exposed target.

| Rank | In-function stage | Measured us | Floor us | Efficiency | Potential to 60% us |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Hillis--Steele distance 1 | 3,003.528 | 264.976 | 8.82% | 2,561.901 |
| 2 | Exclusive entry shift | 2,648.666 | 252.688 | 9.54% | 2,227.519 |
| 3 | Hillis--Steele distance 2 | 2,458.663 | 139.147 | 5.66% | 2,226.751 |
| 4 | Final-state broadcast | 2,095.147 | 190.289 | 9.08% | 1,778.000 |
| 5 | Apply entry transform | 1,338.169 | 12.288 | 0.92% | 1,317.689 |
| 6 | Apply inclusive transform | 183.401 | 12.288 | 6.70% | 162.921 |

Potential is `measured - floor/0.60`; it ranks opportunities but is not additive.

## Decision 0: optimize schedule before P2P internals

Evidence: the API returns states, not prefix transforms. Current execution
moves both A and B through two Hillis--Steele distances and another exclusive
shift. A direct state relay can cut cross-device transfers from 38 to 12 and
global matmuls from six to four for SP4. This is a stronger first experiment
than tuning the generic P2P kernel while preserving the same excessive call
graph.

Experiment 1 hypothesis: the removed launches/bytes outweigh the added serial
dependency for SP4xTP2. Gate it locally to this configuration and retain the
generic schedule for all others.

Status: exact-shape baseline reproduced in the isolated worktree; Experiment 1 is next.

## 2026-07-28: isolated exact-shape baseline

Validated command: `PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_distributed_affine_prefix_perf.py -q -s`. Pytest: 1 passed; wrapper: `SAFE_PYTEST_RESULT: PASS`. Raw CSV: `generated/profiler/reports/2026_07_28_15_00_07/ops_perf_results_2026_07_28_15_00_07.csv`.

Median exact-function device time is 10,524.204 us across trace replay sessions 2--11; projected floor is 696.180 us; efficiency is 6.615%. This is within 0.11% of the original report baseline, so the prior breakdown is reproduced. The isolated stage measurements are: distance 1 = 2,990.939 us, entry shift = 2,647.805 us, distance 2 = 2,458.315 us, final broadcast = 2,095.135 us, entry apply = 1,338.810 us, inclusive apply = 183.145 us.

Two rejected harness shapes are not baselines: a rank-5 `[SP,B,H,K,K]` transform first failed ND sharding and then the operation equal-shape invariant. Source and the correctness test establish the global transform shape as `[SP,H,K,K]`; its local SP shard `[1,H/TP,K,K]` already uses the leading singleton as batch. Also, profiler-mode `SAFE_PYTEST_RESULT: PASS` masks pytest failures, so the actual pytest verdict was used.


## 2026-07-28: workload audit

The existing `test_kda_tp_layer_device_perf` SP4xTP2 case is not the Kimi-K3
affine shape: it defines 32 global heads, hence 16 heads/device. It passed and
reported 7.722 ms/layer over ten trace replays, but its device-profiler buffers
also overflowed. Report `2026_07_28_14_52_56` is therefore an environment smoke
test, not an optimization baseline.

The operation has no weight tensors: its five inputs are runtime affine/state
tensors. Exact values do not affect the launched programs. The new isolated
microbenchmark uses deterministic data with the verified production geometry
`[1,48,128,128]` per device and profiles only this function.

## Experiment 1: direct state relay

Verdict: keep. The serial recurrence is correct and 2.450x faster.

Implementation: only when `sp_size == 4 && tp_size == 2`, relay the recurrent state from rank to rank and preserve the generic Hillis--Steele implementation for every other shape. This reduces cross-device P2P calls from 38 to 12 and matmuls from six to four.

Validation:

- `./build_metal.sh`: PASS.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_distributed_affine_prefix.py -q -s`: 2 passed, `SAFE_PYTEST_RESULT: PASS`. Both TP-axis placements, repeated execution, and trace replay passed. Worst observed PCC was 0.999999; worst max absolute error was 1.904368e-4.
- `PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_distributed_affine_prefix_perf.py -q -s`: 1 passed, `SAFE_PYTEST_RESULT: PASS`. Raw CSV: `generated/profiler/reports/2026_07_28_15_04_58/ops_perf_results_2026_07_28_15_04_58.csv`.

Exact device time fell from 10,524.204 to 4,294.868 us (59.19% reduction, 2.450x speedup). Against the campaign floor of 696.180 us this is 16.21% efficiency; the fixed 60% target remains 1,160.300 us. The schedule-specific floor becomes smaller because the new algorithm performs less work, so it is not used to move the acceptance threshold.

Next breakdown, sorted by potential gain to 60% of each stage floor:

| Rank | Stage | Measured us | Stage floor us | Potential to 60% us |
| ---: | --- | ---: | ---: | ---: |
| 1 | Final-state broadcast | 2,095.662 | 190.289 | 1,778.515 |
| 2 | Relay rank 2 to 3 and apply rank 3 | 889.488 | 69.316 | 773.961 |
| 3 | Relay rank 1 to 2 and apply rank 2 | 578.426 | 69.316 | 462.900 |
| 4 | Relay rank 0 to 1 and apply rank 1 | 557.935 | 69.316 | 442.408 |
| 5 | Initialize and apply rank 0 | 194.874 | 18.946 | 163.297 |

Decision 1: target P2P implementation before matmul. Source evidence shows each P2P uses one worker core and one fabric link (`send_program_factory.cpp:36-40,124`; `receive_program_factory.cpp:35-40,123`) and serially sends one 4 KiB FP32 tile per packet (`writer_send.cpp:76-105`). The 3.146 MB Kimi state therefore cannot use the available parallel fabric links. First test a minimal multi-link P2P change patterned after the existing direct-send implementation; do not assume `ttnn::broadcast` is usable because its 2D API has one exact sender coordinate, while Kimi needs one root per TP lane.

## Experiment 2: FP32 tiled multi-link point-to-point

Verdict: keep. The hardware exposes two forwarding links on the tested routes; striping FP32 tiled packets across them reduces every non-local P2P kernel median by 49.8--50.2% and reduces exact affine-prefix device time by 40.86%. The path was initially restricted to FP32 TILE transfers; Experiment 4 extends it to all TILE transfers after BF16 TILE coverage passed. ROW_MAJOR transfers retain the original single-worker, routing-plane-0 implementation. Worker count is capped by available links, four, and total packet count.

Validation:

- `./build_metal.sh`: PASS after formatting.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_distributed_affine_prefix.py -q -s`: 2 passed, `SAFE_PYTEST_RESULT: PASS`; both TP axes, repeat, and trace passed. Worst PCC remains 0.999999 and worst max absolute error remains 1.904368e-4.
- `PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_distributed_affine_prefix_perf.py -q -s`: 1 passed, `SAFE_PYTEST_RESULT: PASS`. Raw CSV: `generated/profiler/reports/2026_07_28_15_21_13/ops_perf_results_2026_07_28_15_21_13.csv`.

Median exact device time across sessions 2--11 is 2,539.813 us, down from 4,294.868 us (1.691x, 40.86%) and from the original 10,524.204 us (4.144x, 75.87%). Fixed-floor efficiency is now 27.41%; the 1,160.300 us target still requires another 2.189x. Per-device medians are 2,519.107, 2,520.620, 2,518.380, 2,518.195, 2,536.878, 2,539.191, 2,536.850, and 2,535.366 us.

Remaining critical P2P kernels, sorted by median duration (calls overlap across devices, so these are not additive):

| Rank | Transfer | Two-link us | One-link us | Reduction |
| ---: | --- | ---: | ---: | ---: |
| 1 | Broadcast TP1 rank 3 to 0 (3 hops) | 1,433.341 | 2,857.362 | 49.84% |
| 2 | Broadcast TP0 rank 3 to 0 (3 hops) | 1,422.024 | 2,826.874 | 49.70% |
| 3 | Relay TP1 rank 2 to 3 (1 physical neighbor, routed trace duration) | 1,238.331 | 2,484.601 | 50.16% |
| 4 | Relay TP0 rank 2 to 3 | 1,224.370 | 2,456.985 | 50.17% |
| 5 | Broadcast TP1 rank 3 to 1 (2 hops) | 1,208.927 | 2,410.413 | 49.85% |
| 6 | Broadcast TP0 rank 3 to 1 (2 hops) | 1,205.575 | 2,398.005 | 49.73% |
| 7 | Broadcast TP1 rank 3 to 2 (1 hop) | 983.775 | 1,962.440 | 49.87% |
| 8 | Broadcast TP0 rank 3 to 2 (1 hop) | 980.383 | 1,947.407 | 49.66% |
| 9 | Relay TP1 rank 1 to 2 | 625.835 | 1,248.198 | 49.86% |
| 10 | Relay TP0 rank 1 to 2 | 612.647 | 1,222.281 | 49.88% |
| 11 | Relay TP1 rank 0 to 1 | 402.788 | 802.761 | 49.82% |
| 12 | Relay TP0 rank 0 to 1 | 394.371 | 789.362 | 50.04% |

The generic P2P suite reports 28 passed and one deterministic BF16 row-major failure for `[1,1,8,16]`. A clean committed control produces the identical failure (only the first row arrives), proving it predates this experiment. The localized FP32 TILE path does not execute for that case.

Decision 2: reduce broadcast hop-work next. The current final-state loop sends rank 3 independently to ranks 0, 1, and 2 for each TP lane (twelve total fabric hops). A chained rank 3 -> 2 -> 1 -> 0 broadcast preserves six P2P calls but halves fabric hop-work to six and reuses the already received final-state tensor. Validate the dependency chain before accepting it.

## Experiment 3: chained final-state broadcast

Verdict: reject and revert. The chain `3 -> 2 -> 1 -> 0` halved nominal broadcast fabric hops from twelve to six and passed both TP-axis correctness cases, repeat, and trace with unchanged PCC/error. It did not improve performance: raw CSV `generated/profiler/reports/2026_07_28_15_27_33/ops_perf_results_2026_07_28_15_27_33.csv` measured a 2,540.412 us exact median versus 2,539.813 us for independent sends (0.02% slower, noise). Evidence rejects hop count as the limiting variable here: independent routed sends overlap, whereas the chain introduces data dependencies between one-hop transfers. No source change retained.

## Experiment 4: BF16 affine-state transport

Verdict: keep. FP32 remains the input, recurrence-compute, and returned-state dtype; only the communicated carry and P2P destination buffers are BF16. Each transfer payload falls from 3.146 MB to 1.573 MB. Tiled BF16 P2P uses the two-link path; the full generic P2P suite remains identical to the clean control at 28 passed plus the one pre-existing BF16 ROW_MAJOR failure.

Validation:

- `./build_metal.sh`: PASS.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_distributed_affine_prefix.py -q -s`: 2 passed, `SAFE_PYTEST_RESULT: PASS`; both TP axes, repeat, and trace passed. TP-axis 0 worst PCC changed from 0.999999 to 0.999996 and max absolute error from 1.904368e-4 to 3.162287e-4. TP-axis 1 remained at PCC 1.000000 and max absolute error 9.099394e-5.
- `PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_distributed_affine_prefix_perf.py -q -s`: 1 passed, `SAFE_PYTEST_RESULT: PASS`. Raw CSV: `generated/profiler/reports/2026_07_28_15_32_25/ops_perf_results_2026_07_28_15_32_25.csv`.

Median exact device time is 2,191.986 us, down from 2,539.813 us (13.70%) and from the original 10,524.204 us (4.801x, 79.17%). Fixed-floor efficiency is 31.76%; reaching 1,160.300 us still requires 1.889x. Per-device medians are 2,187.613, 2,189.978, 2,186.934, 2,185.344, 2,186.814, 2,189.602, 2,186.615, and 2,185.656 us. Typecast kernels cost a median 11.529 us.

Remaining P2P critical kernels, sorted by median duration: 1,019.147 us (TP1 broadcast 3 hops), 1,016.393 us (TP0 broadcast 3 hops), 882.776/881.337 us (2-hop broadcasts), 853.023/851.916 us (rank 2-to-3 relay), 745.916/745.779 us (1-hop broadcasts), 429.810/427.565 us (rank 1-to-2 relay), and 313.446/313.417 us (rank 0-to-1 relay). These calls overlap across devices and are not additive.

Decision 3: composite P2P launch structure is now the dominant issue. Payload compression and both physical forwarding links are already used, yet a 1.573 MB routed call remains 0.31--1.02 ms and exact time is 1.889x above target. Survey the existing persistent socket/direct-send KDA prototypes before designing a fused relay/prefix primitive; another rearrangement of the same composite calls is not supported by Experiment 3.

## Experiment 5: L1 affine-state transport buffers

Verdict: keep. Preserve the Experiment 4 BF16 transport dtype, but place communicated carry and destination buffers in interleaved L1 instead of DRAM. The 1.573 MB per-device tensor fits across distributed L1 and removes the transport endpoint DRAM round trips; FP32 matmul, add, and public return buffers remain unchanged.

Validation:

- `./build_metal.sh`: PASS.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_distributed_affine_prefix.py -q -s`: 2 passed, `SAFE_PYTEST_RESULT: PASS`; both TP axes, repeat, and trace passed. Precision is unchanged from Experiment 4: TP-axis 0 worst PCC 0.999996 and max absolute error 3.162287e-4; TP-axis 1 PCC 1.000000 and max absolute error 9.099394e-5.
- `PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_distributed_affine_prefix_perf.py -q -s`: 1 passed, `SAFE_PYTEST_RESULT: PASS`. Raw CSV: `generated/profiler/reports/2026_07_28_15_38_58/ops_perf_results_2026_07_28_15_38_58.csv`.

Median exact device time over measured sessions 2--11 is 1,901.893 us, down from 2,191.986 us (13.23%) and from the original 10,524.204 us (5.533x, 81.93%). Fixed-floor efficiency is 36.60%; reaching 1,160.300 us still requires 1.639x. Per-device medians over measured sessions 2--11 are 1,898.127, 1,900.009, 1,897.976, 1,896.303, 1,898.263, 1,900.602, 1,898.207, and 1,897.168 us.

Diagnosis: L1 exposes that steady P2P payload movement is no longer the primary limit. Most sender kernels are approximately 119.6--120.3 us for 1.573 MB (about 13.1 GB/s), while receiver program durations grow to 0.59--0.82 ms because they are queued early and wait for dependent relay computation or earlier sends. The next highest-gain target is redundant final-state broadcast work: rank 3 currently sends the same tensor independently to ranks 0, 1, and 2 on each TP line. Survey an existing fabric line-multicast primitive before implementing a dedicated broadcast.

## Backlog
