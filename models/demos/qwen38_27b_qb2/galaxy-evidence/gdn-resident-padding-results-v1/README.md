# Resident GDN reader experiments: correctness passed, no model promotion

October 11, 2026. Measured full-model performance remains **20.884562 TSU at
B16/32K/TP4 (47.882258 ms)**. These experiments did not produce a meaningful
new speedup. Keep the production reader unchanged and move effort to projection
and collective dataflow. No full-model run or GPQA was launched for these changes.

## What changed and why

The resident recurrence broadcasts only row zero of its delta tile. The shared
reader nevertheless clears all 1024 FP32 values of each V tile before filling
32 live values. v1 tests skipping that clear and separately fills unused rows
with quiet NaNs to check that they cannot influence the result.

v2 additionally completes vector reads first, issues state reads into the
separate reserved CB5 storage, formats the vectors while state reads can be in
flight, and waits for those reads before publishing the input buffers. This
adds a vector-read barrier; its cost is included in timing. Both zeroed and
skipped padding variants are measured, with poison variants used for correctness.

The experiment replaces reader source only inside the opted-in hardware test.
Exact source anchors fail closed if the shared reader changes. All 120 recurrence
workers, two-item lookahead, shared-Q/K mapping, FP32 state/math, ownership,
writer, program arguments and model defaults remain unchanged. There is no new
production policy to enable from these results.

## Measured results

Native traces include the complete recurrence invocation. Each timing arm
executes 2592 identical-input updates from the same reset state: two warm
16-call traces and five samples of 32 such traces. Before/after controls and
all candidate final states/outputs match exactly on all four physical ranks.

| Run | Batch | Variant | Mean baseline, us | Candidate, us | Latency change | Projected saving across 48 layers, ms |
|---|---:|---|---:|---:|---:|---:|
| v1 | 16 | Skip V padding | 81.370 | 80.528 | -1.03% | +0.040 |
| v1 | 32 | Skip V padding | 142.746 | 148.027 | +3.70% | -0.254 |
| v2 | 16 | Skip V padding | 81.505 | 80.503 | -1.23% | +0.048 |
| v2 | 16 | Overlap, zero padding | 81.505 | 82.146 | +0.79% | -0.031 |
| v2 | 16 | Overlap, skip padding | 81.505 | 80.130 | -1.69% | +0.066 |
| v2 | 32 | Skip V padding | 143.134 | 147.367 | +2.96% | -0.203 |
| v2 | 32 | Overlap, zero padding | 143.134 | 157.713 | +10.19% | -0.700 |
| v2 | 32 | Overlap, skip padding | 143.134 | 164.474 | +14.91% | -1.024 |

Baseline is the mean of the before/after sample medians. Control drift is
0.16-0.54%. Every candidate has five samples; all raw values are retained.
The best B16 extrapolation is only **20.913 TSU**, about **+0.029 TSU**, if the
0.066-ms component saving transferred unchanged to the model. That is not a
model measurement and does not justify separate expensive qualification.
The B32 regression also argues against a generic reader change.

These measurements rule out clearing unused V rows as a large standalone
opportunity in this resident path. They do not identify the cause of the B32
regressions or establish that all vector preparation is cheap. Altered read
pacing and the extra barrier are possible contributors, not proven causes.

## Correctness and limits

- v1: 10 CPU tests; native compilation and hardware test passed in 41.9 seconds
  including the controller. v2: 22 CPU tests; native compilation and hardware
  test passed in 48.0 seconds including the controller.
- B16 and B32, physical TP4 ranks `[0,4,12,8]`, independent rank/user inputs,
  cancellation-sensitive first update, 64 changing-input updates, live-buffer
  rebinding, repeated CB wraps, finite outputs and dense FP32 reference checks.
- All zero/skip/poison modes, including overlapping variants, have bit-identical
  state and outputs versus the original reader. Read-only inputs and retained
  buffer addresses are unchanged. Both tests close the mesh cleanly; no reset.
- This is a component screen, not a 4096-changing-input or real-weight/full-model
  qualification. The longer mode is implemented but was not run for a result
  this small. No serving, precision, GPQA or cache feature qualification changes.
- Both logs retain a native warning about allocations with an active trace.
  Matching outputs do not clear that warning. Allocation tracking is still
  required before integrating a future winning implementation.

Jobs ran as bounded persistent user services, with the exclusive device lock,
5-second dispatch timeout, shorter pytest/controller timeouts and a 30-minute
systemd limit. They ran independently of the SSH/conversation connection and
were not configured to resume after reboot. At collection both transient units had already been
garbage-collected (`LoadState=not-found`, `MainPID=0`); successful completion is
established by the sealed controller receipts, JUnit and native PASS logs,
rather than interpreting the absent unit's default `Result=success` field.

## Reproduce and verify

From the repository root:

```sh
python3 models/demos/qwen38_27b_qb2/galaxy-evidence/gdn-resident-padding-results-v1/summarize.py
```

This verifies all 39 raw artifact hashes, source bindings, correctness and timing
receipts, then recreates [summary.json](summary.json). Compressed raw Python/C++
files preserve their original bytes through formatting hooks. The manifest
records both original host-relative paths and stored paths.

Native Metal was `a08819ddbe23077f8037d3802303939064868ff6`; branch base was
`bb87ae7444173aad68e9fd4ebc8eece87ec9b266`. The production recurrence source is
unchanged. Final experiment code matches the v2 frozen source manifest.

Launch arguments and environment are in each raw control directory's
`launch.json`; exact controllers are `run.py.gz`. Host directories are under
`/home/ttuser/qwen38-artifacts-20261007/gdn-resident-padding-*`.
Service names were `qwen38-gdn-resident-padding-v1-20261011.service` and
`qwen38-gdn-resident-padding-v2-20261011.service`.

## Next priority

The dedicated-consumer fusion and these reader changes have not realized the
earlier 1-3-ms GDN planning target. Retain them as evidence and test scaffolding,
not selected model changes. Investigate projection output delivery into the TP
collective and following consumer layout. The prior 62-case reader/block/core
sweep had no accepted gain, so repeating those knobs is not the next experiment.
Count input conversion, matmul, reshard and all-reduce in a matched traced
boundary; target fewer transfers/programs without changing numerical precision.
That work is a next implementation task, not an already running job or a promised
performance improvement. Thirty TSU still requires removing about 14.55 ms.
