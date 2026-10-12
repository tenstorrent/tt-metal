# Resident GDN scheduling and bank placement: completed, not selected

October 12, 2026 UTC. Neither experiment changes the measured full-model
result: **20.884562 TSU, 47.882258 ms at B16/32K/TP4**. The scheduling gain
projects to only +0.025 TSU; every bank-placement candidate regresses. No
production default, precision, serving configuration or native installation
was changed. No full-model, GPQA or 4096-update qualification was run.

## What changed

Scheduling tests three compute-source transformations independently and together:
retain the decay broadcast in spare FP32 DEST tile 2; load all four state tiles
with one copy configuration; combine both. Reader/writer, 120-worker ownership,
two-item lookahead, reduction order and separate multiply/add rounding stay
unchanged. The source transformer fails closed if its exact anchor changes.

Placement remaps complete work items over the same 120 workers. Each value
partition accesses one bank pair, `(partition, partition + 4)`, for the
eight-bank state interleave. Thirty workers are assigned per partition using
read, write or balanced distance cost. Kernel sources, work counts and disjoint
state/output ownership stay unchanged. Unit checks establish complete coverage.

The distance is **logical Manhattan distance to the native first-device bank
anchors**, not measured physical NoC hops or an optimal mapping for all ranks.
Reducing that heuristic cost demonstrably does not improve these timings.
Traffic concentration and directional NoC interactions remain hypotheses;
these tests do not establish the cause of the regression.

## Native results

Times are microseconds per recurrence call, excluding the separate epilogue.
Controls run before and after all three candidates. Negative latency change
means faster. Full-model savings are projections across 48 recurrence calls.

| Experiment | Batch | Control | Candidate | Change | Projected saving, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hoist decay | 16 | 81.829 | 80.828 | -1.22% | 0.048 |
| Block state loads | 16 | 81.829 | 81.463 | -0.45% | 0.018 |
| Both scheduling changes | 16 | 81.829 | 80.621 | -1.48% | 0.058 |
| Hoist decay | 32 | 142.986 | 143.424 | +0.31% | -0.021 |
| Block state loads | 32 | 142.986 | 143.446 | +0.32% | -0.022 |
| Both scheduling changes | 32 | 142.986 | 143.080 | +0.07% | -0.005 |
| Read-bank placement | 16 | 81.443 | 113.582 | +39.46% | -1.543 |
| Write-bank placement | 16 | 81.443 | 83.295 | +2.27% | -0.089 |
| Balanced placement | 16 | 81.443 | 101.803 | +25.00% | -0.977 |
| Read-bank placement | 32 | 142.570 | 222.958 | +56.38% | -3.859 |
| Write-bank placement | 32 | 142.570 | 164.055 | +15.07% | -1.031 |
| Balanced placement | 32 | 142.570 | 200.557 | +40.67% | -2.783 |

Control drift: scheduling 0.437%/0.199% at B16/B32; placement
0.046%/0.013%. Each arm uses five samples of 32 replays of a 16-call trace,
plus two warm replays: 2592 updates with exactly matching final state/output.
Small scheduling changes are not a large explanation for recurrence cost.
Keep future work focused on reducing arithmetic/intermediate traffic or larger
compatible boundaries instead of repeating these same knobs.

## Correctness and execution

- 12 scheduling and 17 placement CPU tests passed. Generated scheduling C++
  variants compiled through native generic-op JIT on the pinned installation.
- Both physical TP4 tests passed B16/B32 with cancellation-sensitive inputs,
  64 changing inputs, four independent ranks and four retained variant sessions.
  State and output were exact against baseline and passed dense FP32 checks.
- Inputs stayed read-only, addresses stayed stable, and trace-allocation
  tracking remained enabled. No live-buffer warning or device reset occurred.
- Host: `10.228.203.98`; devices `[0, 4, 12, 8]`; native Metal pin
  `a08819ddbe23077f8037d3802303939064868ff6`.
- Scheduling finished around 01:33:54 UTC; hardware pytest took 36.29 seconds.
  Placement finished around 08:14:51 UTC; hardware pytest took 34.38 seconds.
  The interval between experiments was not device execution time.
- Both bounded persistent systemd jobs are terminal and cleanup completed.
  They survived SSH disconnection but were not configured to resume after
  reboot. Controller/JUnit/native logs establish success independently of
  transient-unit garbage collection. The device lock was free at collection.

## Reproduce the report

```bash
python3 models/demos/qwen38_27b_qb2/galaxy-evidence/gdn-schedule-placement-results-v1/summarize.py
```

The script verifies all 35 sealed raw files, six current tested source files,
generated kernel hashes, CPU/native JUnit results, terminal receipts, exact
rank comparisons and timing arithmetic. `summary.json` contains full-precision
values. Raw Python, C++, XML and logs are compressed before repository formatting
to preserve their bytes. Complete launch environments and commands are retained
in each control directory; all device execution used `/tmp/tt-device.lock`.
