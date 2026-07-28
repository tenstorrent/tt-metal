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
