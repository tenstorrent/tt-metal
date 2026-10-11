# Profiling coverage and the next diagnostic

October 11, 2026, 02:32 UTC. Target: B16/32K, TP4, BFP8 weights/KV,
BF16 activations, FP32 recurrent state. This is an evidence audit and diagnostic
plan, not a new capture or a claim of measured DRAM utilization.

Update at03:13UTC: the current resident/epilogue phase diagnostic completed
all eight cases with all24 required labels and clean teardown. See
[results](../galaxy-evidence/gdn-combined-padding-results-v1/README.md) and
[launch evidence](../galaxy-evidence/gdn-pipeline-phase-launch-v1/README.md).
Phase analysis is now retained with those results: all 3,460,608 selected
events pair and expected item counts match. Hardware counters remain outstanding.

Update at04:31UTC: the bounded hardware-counter capture is active after two
diagnostic failures. The first was duplicate raw-log discovery after a passing
control; the second was an eight-byte BRISC firmware overflow with L1_0+FPU.
The stricter native one-group pass plan is now running after locked recovery.
No populated counter result is claimed yet. See
[launches and preserved failures](../galaxy-evidence/gdn-counter-launch-v1/README.md).

Update at04:47UTC: the single-FPU hardware pass passed output/state checks and
closed cleanly. Its CSV lacks counter type metadata: native mid-run export
skips the enrichment routine. A corrected counter-specific wrapper retains
final metadata processing; the new persistent capture is active. No populated
counter result is credited before its coverage check succeeds.

## Existing evidence

| Evidence | Established | Limit |
|---|---|---|
| [Compact full-model capture](../galaxy-evidence/compact-profile-v1/README.md) | Real weights, restored state, three replays on four ranks; 50.0873 ms unprofiled versus 52.3057 ms profiled | 4.43% whole-step overhead; predates resident recurrence and padding skip |
| [Generated-kernel attribution](../galaxy-evidence/compact-kernel-map-v1/README.md) | Exact source/hash attribution of recurrence, epilogue, convolution and preparation | Kernel lifetimes include waits; sums are not an additive critical path |
| [GDN phase diagnostic](../galaxy-evidence/gdn-phase-results-v2/README.md) | Eight cases, 24 calls, identical output/state on four ranks; 2,105,856 correctly paired selected events | Older nonresident recurrence, synthetic shared Q/K, no epilogue; compute zones include synchronization |
| [Epilogue padding A/B](../galaxy-evidence/gdn-epilogue-padding-results-v3/README.md) | Removing unused initialization reduced B16 packed-L1 component time from 66.254 to 39.336 us | 1.292-ms full-step saving is a 48-layer projection, awaiting combined model comparison |

The earlier phase diagnostic already provided a useful lead: with two buffers
at B32, accumulated input waits averaged 9.82 us/core while the instrumented
kernel median was 162.14 us. Reader L1 preparation averaged 65.88 us/core,
partly overlapped. This points toward compute/synchronization and formatting
costs. It does not prove input delivery is irrelevant or that removing reader
preparation would save 66 us. Do not repeat this diagnostic unchanged.

## Remaining attribution gaps

1. Current resident recurrence and epilogue now have a completed phase capture
   and paired-event analysis. Full-model critical-path reconciliation remains;
   diagnostic-only annotations retain math, barrier ordering and precision,
   with exact output comparisons on physical hardware. Math regions include
   synchronization and pack regions include register waits; they are not pure
   arithmetic/packing durations. Native profiling is active in both controls.
2. Existing reports have no populated physical DRAM utilization or NoC
   congestion data. Useful bytes divided by elapsed time and assumed peak is
   a model, not bus activity. Hardware counters are needed to distinguish
   math, unpack/pack, L1 contention and waits.
3. The complete dependent timeline remains open: quantify rank/core imbalance,
   launch gaps, collectives and overlap without adding concurrent lifetimes.
   A reader or writer that remains alive may be waiting.
4. The 512-GB/s/chip roofline is assumed peak, not measured bandwidth at the
   model's placement, transfer size and read/write mix. Calibrate matched
   streaming controls before calling the entire gap attainable.
5. These captures focus on decode. Prefill and eventual prefix/SSD serving
   need separate measured timelines before system-throughputput conclusions.
   AgentX remains gated on both features through serving.

## Verified capabilities of the pinned Metal revision

Inspected source at `a08819ddbe23077f8037d3802303939064868ff6`:

- `tt_metal/tools/profiler/perf_counters.hpp`: FPU/SFPU/math activity,
  pack/unpack, instruction stalls and L1 request/grant counter groups.
  Compute-core counters bracket TRISC1; readout occurs on BRISC. They do not
  directly describe the whole reader/writer lifetime.
- `tools/tracy/perf_counter_multipass.py`: reuse the native pass planner.
  Only one L1 mux bank per pass is valid. The pinned planner limits each
  pass to three counter groups because larger readout code overflows BRISC
  firmware text. Do not enable all groups in one mask.
- `tt_metal/llrt/rtoptions.cpp`: runtime rejects multiple L1 bank bits.
  Its example mask 47 selects five groups, exceeding the planner's documented
  three-group limit; it is not an appropriate all-in-one capture setting.
- L1 NoC-port request/grant counters measure the L1 interface, not DRAM
  controller bandwidth. Their ratios alone do not prove network congestion.
  Retain counter coverage and reference intervals.

Source support does not prove a successful capture on this installation.
A bounded smoke capture with meaningful counter records is required before
relying on them. Keep the pinned native installation unchanged.

## Next useful capture

Prepare a bounded diagnostic for current recurrence and epilogue, then the
low-efficiency output projection. Add SDPA and remaining matmuls after this
attribution works; do not start with another broad full-model trace.

1. Record uninstrumented component controls with actual B16 layouts and
   ownership. Restore identical inputs/state between passes; retain all-rank
   correctness, generated source hashes and measured instrumentation overhead.
2. Annotate resident input wait, update/reduction, packing, epilogue formatting,
   RMS/gating and writer waits. Do not reorder barriers to split one combined
   DMA interval, or accidentally change C++ variable scope with timing blocks.
3. Use native Tracy multipass for relevant math, pack, unpack, instruction and
   L1 banks. A complete Blackhole L1-bank survey needs six passes; begin with
   the groups relevant to these kernels. Preserve each pass's raw records.
4. Pair suspected contention with a placement/core-count A/B and matched
   streaming control. Counters provide a lead; reduced unprofiled latency
   with unchanged outputs establishes an optimization's value.
5. Reconcile winners with unprofiled full-model step time and critical path.
   Batch sub-1-TSU improvements before expensive G0/API/full GPQA.

At the original02:32UTC audit timestamp, this diagnostic was planned. It is
subsequently queued and completed as recorded above. At the original timestamp, the
combined-padding model comparison is running and its conditional qualification
follower is waiting. The follower requires at least 1 TSU measured gain and
does not automatically run another full profile. Do not instrument or replace
the running comparison in place.

## October 11 continuation after artifact headroom guard

Metadata-preserving FPU, pack and unpack captures passed complete per-core
coverage and exact outputs. The instruction device test passed and closed
cleanly, but analysis stopped at the 16-GiB free-space guard. This is a collector
failure, not a failed model output or a reason to reset clean hardware.

A persistent recovery sequence losslessly archived a closed historical 63-GiB
Tracy CSV, verified full decompression against the original SHA256 and left
75.65 GiB free. It then recovers the retained instruction analysis and
continues L1 banks 0-5 plus the final control. Exact failed and successful receipts
remain separate in `galaxy-evidence/gdn-counter-launch-v1`. Do not credit the
queued passes before their coverage and output checks complete. No new model
throughput or physical DRAM-utilization claim follows from this recovery.


At 05:13 UTC, offline instruction recovery passed all requested counters on
23,040 active-core operation records with exact state/output matches and no
device rerun. The persistent service advanced to the six L1-bank captures and
final control. `space-recovery-verified` retains this later observation; earlier
failed and in-progress snapshots remain unchanged.
