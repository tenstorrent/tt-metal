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

## Experiment 6: per-TP-line hardware multicast

Verdict: keep. Replace the three independent final-state P2P sends on each TP lane with the existing two-link CCL line-multicast. Correct `BroadcastProgramFactory` sender selection for cluster-axis collectives: every orthogonal mesh line now selects the rank matching the sender coordinate on the collective axis, consistent with the already per-line topology and neighbor calculation. This lets one operation broadcast from SP rank 3 independently on both TP lanes.

Validation:

- `./build_metal.sh`: PASS.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_distributed_affine_prefix.py -q -s`: 2 passed, `SAFE_PYTEST_RESULT: PASS`; both sequence-parallel axis placements exercise the corrected 2D per-line sender semantics, and repeat plus trace passed. Precision remains unchanged: TP-axis 0 worst PCC 0.999996 and max absolute error 3.162287e-4; TP-axis 1 PCC 1.000000 and max absolute error 9.099394e-5.
- `PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_distributed_affine_prefix_perf.py -q -s`: 1 passed, `SAFE_PYTEST_RESULT: PASS`. Raw CSV: `generated/profiler/reports/2026_07_28_15_44_53/ops_perf_results_2026_07_28_15_44_53.csv`.

Median exact device time over sessions 2--11 is 1,604.297 us, down from 1,901.893 us (15.65%) and from the original 10,524.204 us (6.560x, 84.75%). Fixed-floor efficiency is 43.40%; reaching 1,160.300 us still requires 1.383x. Per-device medians are 1,603.805, 1,602.502, 1,604.208, 1,603.226, 1,602.240, 1,601.054, 1,602.811, and 1,601.486 us.

Diagnosis: multicast removes five composite P2P launches and redundant payload sends per invocation. The two physical source devices execute the broadcast in approximately 68 us; longer 0.18--0.66 ms receiver durations are time queued behind the dependent relay and do not represent multicast bandwidth. The critical arithmetic remains four FP32 matmuls at a 154.525 us median, four adds at 22.253 us, three approximately 120 us steady P2P sends, and dtype/launch boundaries. Next test a BF16 recurrence fast path: retain FP32 public inputs and outputs, but typecast transforms once, compute and relay carry in BF16, and validate recurrent error before making a performance claim.

## Experiment 7: BF16 internal recurrence compute

Verdict: keep; campaign target reached. Public inputs, entry states, and final states remain FP32. The SP4xTP2 fast path typecasts `transform_a` and `transform_b` once into interleaved L1, performs the four recurrence matmuls and adds in BF16 L1, relays the BF16 carry directly, and typecasts only the returned entry/final states to FP32. This removes redundant carry typecasts and avoids recasting communicated BF16 state to FP32 solely for the next matmul.

Validation:

- `./build_metal.sh`: PASS.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_distributed_affine_prefix.py -q -s`: 2 passed, `SAFE_PYTEST_RESULT: PASS`; both TP axes, repeat, and trace passed. The precision cost is measurable and explicitly accepted by the existing production contract: TP-axis 0 worst PCC changes from 0.999996 to 0.999984 and max absolute error from 3.162287e-4 to 7.171780e-4. TP-axis 1 remains PCC 1.000000 and max absolute error 9.099394e-5.
- `PERF_REPS=10 scripts/run_safe_pytest.sh --profile models/experimental/kimi_delta_attention/tests/perf/test_distributed_affine_prefix_perf.py -q -s`: 1 passed, `SAFE_PYTEST_RESULT: PASS`. Raw CSV: `generated/profiler/reports/2026_07_28_15_49_20/ops_perf_results_2026_07_28_15_49_20.csv`.

Median exact device time over sessions 2--11 is 979.819 us, down from 1,604.297 us (38.93%) and from the original 10,524.204 us (10.741x, 90.69%). Fixed-floor efficiency is 71.05%, exceeding the requested 60% target by 11.05 percentage points and beating the 1,160.300 us threshold by 180.481 us. Session maxima are 978.683, 979.893, 980.410, 979.746, 979.972, 978.755, 981.285, 979.319, 980.138, and 978.677 us. Per-device medians are 978.594, 977.416, 978.520, 977.540, 979.483, 978.276, 979.379, and 978.160 us.

The four recurrence matmuls fall from a 154.525 us median to 20.249 us; adds fall from 22.253 us to 5.092 us. Remaining steady payload sends are approximately 119.6--119.9 us and source-side multicast is approximately 68.1 us. Receiver-side 0.18--0.71 ms durations continue to include queued dependency wait and are not additive transfer costs.

Decision 4: stop the target campaign at the first fully validated result above 60% rather than expand scope. Further work could fuse BF16 matmul/add with relay readiness or specialize P2P for persistent recurrence, but neither is required for the stated goal.

## 2026-07-29: generalize the state relay

Verdict: keep. Remove the `SP=4, TP=2` selection guard and use the same
source-derived state relay for every distributed mesh (`SP > 1`). Loop bounds
and mesh coordinates are derived from `sp_size`, `tp_size`, and
`sequence_parallel_axis`; broadcast link count is auto-selected. There is no
participant-count specialization.

Hypothesis: SP2xTP4 was still running the generic affine-transform
Hillis--Steele schedule, so applying the already validated state-relay
algorithm should eliminate its redundant transform communication without
regressing SP4xTP2.

Validation:

- `./build_metal.sh`: PASS.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_distributed_affine_prefix.py -q -s`:
  2 passed, `SAFE_PYTEST_RESULT: PASS`. SP4xTP2/axis 0 worst PCC is 0.999984
  and max absolute error is 7.171780e-4; SP2xTP4/axis 1 worst PCC is 0.999991
  and max absolute error is 4.910976e-4. Repeat and trace replay pass.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_sp_layer.py -q -s`:
  6 passed, `SAFE_PYTEST_RESULT: PASS`; worst printed output PCC is 0.999974
  and recurrent-state PCC is 0.999985.
- Real Kimi-K3 layer, real weights, `T=5120`, ten trace replays:
  SP2xTP4 passes at 13.472 ms/replay
  (`generated/profiler/reports/2026_07_29_06_19_07`), down from 14.605 ms
  (7.8%). The matched SP4xTP2 control passes at 12.462 ms/replay
  (`generated/profiler/reports/2026_07_29_06_21_03`), versus the prior
  12.423 ms (0.3%, noise).

The source-semantic trace attribution confirms the hypothesis: SP2xTP4
distributed-prefix time falls from 4.337 ms to 0.590 ms. It also rejects
distributed prefix as the remaining explanation for the SP2/SP4 ordering:
SP4's matched prefix is 2.378 ms. The largest remaining adverse SP2 deltas are
output projection/reduce-scatter (4.055 vs 2.532 ms) and local affine
composition (1.175 vs 0.508 ms). These are the next two targets, ranked by
measured opportunity.

## 2026-07-29: fuse local affine composition

Verdict: keep. The existing `KdaAffinePrefixOperation` already performs an
arbitrary-group Hillis--Steele scan on device. Add a compose-only mode that
returns the final inclusive `(A, B)` transform for each head and skips
initial-state application. The distributed path now calls that mode directly
instead of launching slices, matmuls, adds, concats, and final slices. The
implementation is parameterized by `groups_per_head`, `key_dim`, and
`val_dim`; it contains no SP/TP participant-count branches.

Validation:

- `./build_metal.sh`: PASS.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_distributed_affine_prefix.py -q -s`:
  2 passed, `SAFE_PYTEST_RESULT: PASS`; both mesh-axis placements, repeat, and
  trace replay pass with unchanged accuracy.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_sp_layer.py -q -s`:
  6 passed, `SAFE_PYTEST_RESULT: PASS`. This exercises `G=4,5,8,10` across
  both mesh-axis placements. Worst printed output PCC is 0.999969 against the
  serial reference and 0.999974 for chunked versus one-shot; worst recurrent
  PCC is 0.999917 and 0.999985 respectively.
- Real Kimi-K3 layer, real weights, `T=5120`, ten trace replays:
  SP2xTP4 passes at 12.332 ms/replay
  (`generated/profiler/reports/2026_07_29_06_32_52`), down from 13.472 ms
  after the generalized relay (8.5%) and 14.605 ms baseline (15.6%).
  SP4xTP2 passes at 12.012 ms/replay
  (`generated/profiler/reports/2026_07_29_06_34_38`), down from its matched
  12.462 ms control (3.6%).

The trace verifies the intended mechanism. SP2xTP4 local composition falls
from 24 operations and 1.175 ms to one `KdaAffinePrefixOperation` at
45.303 us (25.9x). SP4xTP2 falls from 12 operations and 0.508 ms to one
operation at 54.197 us (9.4x). The remaining SP2/SP4 wall gap is 0.320 ms,
with TP4 output projection/reduce-scatter still the largest adverse stage:
3.950 ms versus 2.434 ms.

## 2026-07-29: reduce fused output-collective traffic

Verdict: keep. Typecast the output-projection input to BF16 and request a BF16
output from the fused matmul/reduce-scatter. The change applies to every fused
KDA output collective and contains no SP/TP-count branch. Matmul accumulation
remains FP32 through the existing compute-kernel configuration.

Diagnosis: normal `MatmulReduceScatterAsyncDeviceOperation` does not overlap
the batch-1 projection and reduce-scatter. Its sender publishes readiness only
after the complete matmul batch, and the linear reduce-scatter consumes
full-width slices. At the matched task-2 baseline, SP2xTP4 spent a 3.950 ms
median in this stage versus 2.434 ms for SP4xTP2; endpoint matmul kernels took
approximately 2.65--2.70 ms, while inner ranks took approximately
3.93--3.96 ms. This isolates approximately 1.25--1.31 ms of TP4 collective
tail and supports traffic volume as the limiting factor.

Rejected controls:

- Changing only the MMRS output dtype produced mixed input/output page sizes
  and corrupted the fused collective (output PCC as low as -0.005 with values
  near 3e29). The retained path keeps input and output dtypes equal.
- The existing chunk-ready strided MMRS path is ring-only, while SP2xTP4
  resolves to the linear collective. Reusing it would not test or improve the
  target path. A correct chunked linear algorithm would require a new data
  layout and collective implementation, not a semaphore-only change.

Validation:

- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_sp_layer.py -q -s`:
  6 passed, `SAFE_PYTEST_RESULT: PASS`. Both mesh-axis placements, repeat, and
  trace replay pass. Worst printed serial-reference output PCC is 0.999964;
  worst chunked-versus-one-shot PCC is 0.999967.
- `scripts/run_safe_pytest.sh --run-all models/experimental/kimi_delta_attention/tests/test_tp_weights.py -q -s`:
  4 passed, 4 mesh-shape skips, `SAFE_PYTEST_RESULT: PASS`. TP8 layer output
  PCC is 0.999958; both 2D output placements pass at PCC 0.999994.
- Real Kimi-K3 layer, real weights, `T=5120`, ten trace replays:
  SP2xTP4 passes at 11.124 ms/replay
  (`generated/profiler/reports/2026_07_29_06_50_27`), down from 12.332 ms
  (9.8%). Its MMRS session-max median falls to 2.737 ms despite the added
  0.312 ms typecast.
- The matched SP4xTP2 control passes at 11.548 ms/replay
  (`generated/profiler/reports/2026_07_29_06_52_23`), down from 12.012 ms
  (3.9%). Its MMRS median falls to 1.960 ms and its added typecast is also
  0.312 ms.

SP2xTP4 is now 0.424 ms faster than SP4xTP2, reversing the prior 0.320 ms
deficit. The larger TP4 benefit is consistent with the traffic hypothesis.
Across the three general changes, SP2xTP4 improves from 14.605 to 11.124 ms
(23.8%).

## Backlog
