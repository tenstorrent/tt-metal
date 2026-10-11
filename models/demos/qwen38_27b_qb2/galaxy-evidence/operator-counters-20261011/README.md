# Completed GDN, projection and attention counters

Collected October 11, 2026. **Model performance remains 20.884562 TSU at
B16/32K/TP4 (47.882258 ms)**. This publication improves attribution and the
profiler; it does not contain a new model speedup or serving promotion.
The latest separately GPQA-qualified compact configuration remains
177/198 (89.39%).

## Completed and verified

- GDN: all ten groups (FPU, pack, unpack, instruction, L1 selectors 0-5),
  192 target rank/call records and 23,040 active-core call records per pass,
  exact cross-pass state/output matches and a final zero-mask control.
- Large operators: all 17 v4 captures completed: before/after controls plus
  five boundaries for each of instruction, L1_0 and L1_1. Every capture passed
  hardware correctness, exact outputs, complete requested counter coverage
  and device cleanup. Prior v2 FPU/pack/unpack captures also passed coverage.
- The five boundaries are local output/down matmuls, output/down including
  layout conversion and TP all-reduce, and the production accurate SDPA path.
  Projections use actual layer-0 BFP8 checkpoint weights. SDPA uses synthetic
  BFP8 shuffled pages, distinct user positions and poisoned future values;
  it is not a model-level attention accuracy qualification.
- All four physical ranks are checked: `[0,4,12,8]`. Three repeated calls per
  boundary must match exactly. Projection accuracy requires PCC >=0.999 and
  relative RMS <=0.015 against quantized-weight references; SDPA requires
  per-user PCC >=0.999 and relative RMS <=0.02.
- 41 focused CPU tests passed (24 operator-profile, 17 GDN-counter tests).
  The final v4 edit added explicit device-profiler drains; the parser and
  controller retain the tested v3 hashes. All four local profiler files match
  the successful frozen v4 manifest. Pre-commit passed.

The terminal v4 service returned exit 0 and is inactive with MainPID 0.
The runs survived a disconnected/paused conversation. They were not configured
to resume after a host reboot. No new device job was launched for collection.

## Measurements and interpretation

These are weighted active-core counter fractions: sum(value)/sum(reference),
not averages of per-core percentages. Grant fractions are sum(grants)/sum(requests)
for paired counters with nonzero requests. Groups were captured separately.

| Boundary | Math activity | Packer busy | Measured L1 unpack grant/request | Measured L1 pack grant/request |
|---|---:|---:|---:|---:|
| Output projection, local | 39.7% | 11.6% | 88.8-93.1% | 79.0% |
| MLP down projection, local | 50.6% | 5.2% | 88.9-95.1% | 82.0% |
| Accurate SDPA | 37.6% | 2.7% | 98.1-99.0% | 98.5% |
| Resident GDN recurrence | 31.8% | 1.3% | 97.0-100% | 98.5-98.8% |
| GDN epilogue | 45.8% | 6.0% | 94.2-99.0% | 98.3-98.7% |

SDPA's sampled NoC/L1 interfaces grant 98.5-100% of requests. GDN recurrence
is 98.4-99.5%. This does not support assuming pervasive arbitration denial at
these interfaces. It also does not rule out latency, reader placement, DRAM
transaction inefficiency or congestion elsewhere: these counters bracket
compute-core work, not every reader/writer or physical DRAM controller.

Projection pack/unpack interfaces show more request denial. However, packer
busy is only 5-12%, and thread-2 semaphore waits are high. A pack grant ratio
alone does not establish a saturated packer or quantify removable wall time.
Local and complete projection captures show similar target-matmul activity;
the complete boundary adds separate reshard and all-reduce operations.

GDN recurrence's ANY_THREAD_STALL fraction is 84.1%, including thread-1
nonzero-semaphore waits of 29.7% and thread-2 waits of 73.5%. These intervals
overlap and must not be added or interpreted as an 84% possible speedup.
Together with the earlier preparation phase timing, this prioritizes operand
formatting and producer/consumer handoffs over blind additional reader cores.
It remains a hypothesis to test with a matched unprofiled change.

Median of the four per-rank kernel medians, in microseconds:

| Target kernel | Before | After |
|---|---:|---:|
| Output local | 26.390 | 26.591 |
| Output, inside complete boundary | 26.818 | 26.481 |
| Down local | 66.218 | 66.201 |
| Down, inside complete boundary | 66.407 | 66.367 |
| SDPA | 800.287 | 800.526 |
| GDN recurrence, B16 skip, no extra phase zones | 79.172 | 79.950 |
| GDN epilogue, same case | 31.555 | 31.569 |

GDN control drift is about 0.98% for recurrence and 0.046% for epilogue.
The state-only 49.152-us ideal memory floor remains about 62% of recurrence
time. This excludes vector traffic, computation and synchronization.

The retained complete-boundary timelines are **eager Python-issued calls**.
Their firmware spans include host issue gaps (roughly 0.7-0.8 ms for complete
projections); those are not representative of the traced model's projection
latency. Kernel sums and separate family medians are also not additive model
wall time. The earlier matched full-model profile remains the source for
whole-step contributions. Activity counters are not roofline percentages or
physical DRAM-bandwidth utilization.

## Profiler failure and fix

v2 passed FPU/pack/unpack. Its instruction hardware test passed correctness
and cleanup, but analysis rejected incomplete compute-core counter coverage.
Native warnings explicitly reported full profiler DRAM buffers and dropped
markers. v3 split each counter group into individual operator boundaries;
the complete output projection still overflowed. These were lost profiling
records, not model output failures or a device hang.

v4 calls `ttnn.ReadDeviceProfiler(mesh)` before warmup, after warmup and after
each measured END marker. This drains device buffers into host storage outside
the measured samples. It keeps final counter metadata enrichment and leaves
`TT_METAL_PROFILER_MID_RUN_DUMP` unset. The existing strict coverage checks were
not weakened. All 17 captures subsequently passed without a reset or native
installation change. Prior failures and native warning excerpts are retained.

## Next experiments, in priority order

1. GDN operand delivery and recurrence-to-epilogue fusion: test removing
   repeated vector formatting and overlapping consumer preparation, retaining
   exact FP32 state math and the full 128-value-head normalization. Use the
   explicit ownership/credit design in [GDN-FUSION-PLAN.md](../../experiments/GDN-FUSION-PLAN.md).
   Combined engineering target remains 1-3 ms per model step, not a prediction
   established by these counters. Against 47.882 ms this would be about
   21.33-22.28 TSU. Do not count prior padding/resident gains again.
2. Projection dataflow across the complete boundary: keep producer output in
   the consumer's layout and inspect L1 allocation/port contention. The prior
   62-case reader/block/core sweep produced no accepted gain; do not repeat
   those knobs unchanged. A causal A/B must improve traced boundary latency,
   not just a counter fraction. Existing 1.5-3.5-ms roadmap range is still an
   unverified target, not new credit from this capture.
3. SDPA delivery/pipelining remains relevant, especially at long context, but
   current evidence does not prioritize a generic L1 contention fix. The
   previous remote-delivery prototype and partial-query work do not establish
   a large new gain. Preserve accurate exp and FP32 accumulation.

These are implementation priorities, **not queued jobs**. Thirty TSU still
requires removing about 14.55 ms; no single measured lead closes that gap.
Batch changes worth less than 1 TSU for expensive model qualification. Full
GPQA remains gated on a measured 25 TSU. AgentX remains gated on prefix caching
and SSD offload through serving; the passing native restore test is separate.

## Reproduction and provenance

Run from the repository root:

```sh
python3 models/demos/qwen38_27b_qb2/galaxy-evidence/operator-counters-20261011/summarize.py
```

This verifies all 105 sealed input hashes (including decompressed content) and
recreates [summary.json](summary.json). The raw manifests retain host paths,
source hashes, commands, counter inventories, JUnit and cleanup receipts.
Large instruction JSON and control CSVs are losslessly compressed for Git.
`raw/large-operator-publication-v1/overflow-excerpts.json` separately retains
the original log hashes and relevant line numbers. Raw native device traces
remain on host disk; the repository contains validated per-core aggregates.

Pinned native Metal: `a08819ddbe23077f8037d3802303939064868ff6`.
Successful host source: `/home/ttuser/qwen38-artifacts-20261007/large-operator-source-v4`.
Controller launch and full arguments: `raw/large-operator-control-v4/launch.json`.
Capture unit: `qwen38-large-operator-profile-v4-20261011.service`.
No serving defaults, precision settings or installed kernels changed.
