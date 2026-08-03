# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Findings Closed

- Two-way MLP gate/up packing was implemented and profiled after the first
  review found its rejection unearned. The candidate produced one
  `32 x 5120 x 34816` matmul and two exact slices per replay, but regressed
  full-attention batch-32 decode from 2386.7590 to 2388.7548 us/replay. It was
  reverted and the measured rejection is retained.
- Determinism is now explicit: the trace test snapshots mutable KV or
  convolution/recurrent state, executes, restores identical state, executes
  again, and requires bit-exact output equality.
- Motherboard-discovery and post-success nanobind teardown diagnostics are
  controlled environment warnings; PCC, fallback-hard-failure, and watcher
  evidence contain no stage-path failure.

## Independently Recomputed Performance

| Workload | Functional us | Final fused us |
|---|---:|---:|
| Decode full b1 | 2381.3743 | 2287.7420 |
| Decode full b32 | 2474.1482 | 2386.7590 |
| Decode linear b1 | 3058.8871 | 2921.2791 |
| Decode linear b32 | 21470.3822 | 20884.8516 |
| Prefill full b1 | 2694.6060 | 2584.4600 |
| Prefill full b32 | 72256.6030 | 68521.0110 |
| Prefill linear b1 | 10691.7000 | 10577.3160 |
| Prefill linear b32 | 316313.4630 | 312361.0220 |

The reviewer parsed retained CSVs directly. The rejected packed-MLP CSV sums
to 23887.548 us over ten replays. The final source contains separate gate/up
linears and no packed MLP key.

## Scope Inspected

- Graph-fusing, TT-device-usage, and stage-review skill contracts.
- `tt/fused_decoder.py`, fused-path/static/trace tests, README, work log,
  AutoDebug/AutoFix reports, context contract, correctness logs, trace logs,
  final profiler reports, rejected-candidate profiler report, and final watcher
  artifacts.
- Read-only Git/source/log inspection, CSV analysis, and `git diff --check`.
- No device or server was opened by the reviewer.

## Residual Risk

- The rejected candidate regression is small (about 0.084%), but its host
  median corroborates the device-profiler ordering. The retained graph still
  clearly beats every functional baseline.
- Determinism covers representative full- and linear-attention layers at batch
  32; all physical layers share these two reviewed implementations.
