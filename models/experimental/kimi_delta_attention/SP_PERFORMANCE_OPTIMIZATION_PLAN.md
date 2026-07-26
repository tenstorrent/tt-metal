# KDA sequence-parallel performance optimization plan

## Purpose

This is the execution plan for getting the best defensible end-to-end KDA sequence-parallel performance from the current implementation. It starts from the proven LoudBox `SP=2 x TP=4` trace, because that is the only topology we can measure today, and transfers only validated mechanisms to Galaxy `SP=8 x TP=4`.

It optimizes the *slowest device's traced layer critical path*, not a host interval or a standalone collective. The existing protocol, correctness paths, and serial implementation remain the oracle while this work is in progress.

For topology, correctness, and bring-up history, see [SP_LOUDBOX_PLAN.md](SP_LOUDBOX_PLAN.md).

## Starting point and targets

The current accepted control is a complete, child-mesh captured `SP=2 x TP=4` forward at global `T=5120` on LoudBox:

| Metric | Current value |
| --- | ---: |
| Steady-state slowest-device median | **2.855 ms** |
| Steady-state range | 2.848--2.860 ms |
| Profiler report | `2026_07_26_15_59_04` |
| TP=8 control at the same global shape | 3.114 ms |
| Improvement versus TP=8 | 8.3% |
| Accepted primary gate | <= 2.958 ms |
| Existing stretch gate | <= 2.803 ms |

The trace used `KDA_SP_SPLIT_AFFINE=1`, two socket lanes, and a 512 KiB socket FIFO. Sessions 1--2 are cold/warm outliers; all comparisons use the median of sessions 3--11 from ten child-trace replays.

The current critical-path view on a representative slow device is:

| Operation / path | Duration | What it means |
| --- | ---: | --- |
| Fused TP4 output `MatmulReduceScatterAsyncDeviceOperation` | 879 us | Largest output-projection compute/CCL stage. |
| Runtime output `CloneOperation` | 830 us | Overlaps the MRS, but extends the output tail. |
| Union from MRS start to clone end | about **916 us** | First optimization target. |
| Causal short convolution | 452 us | Second local compute candidate, after proving its unhidden contribution. |
| Initial matmul | 420 us | Do not tune before the output tail. |
| Chunk GDN preparation | 310 us | Do not tune before the output tail. |
| Affine prefix | 181 us | Correct and not currently the dominant unhidden span. |

The clone ends only about 37 us after MRS. Therefore, removing the clone by itself can save **at most about 37 us** on the current critical path; it cannot by itself reach the 52 us stretch goal. A stretch win needs either both clone removal and a modest MRS reduction, or a larger change to the joint output tail.

| Endpoint | Required saving from 2.855 ms | Required reduction of the current 916 us output-tail union |
| --- | ---: | ---: |
| Existing stretch: 2.803 ms | 52 us | 5.7% |
| Strong LB result: 2.750 ms | 105 us | 11.5% |
| Ambitious LB result: 2.700 ms | 155 us | 16.9% |

`2.750 ms` is the best near-term endpoint worth planning against. `2.700 ms` is aspirational, not a commitment: it requires a material reduction in the fused output path rather than a cosmetic scheduling change.

## Measurement contract

Every retained change must satisfy all of the following.

1. Run the focused functional gate first, including target-shape output, recurrent-state, convolution-state, and boundary-token PCC. Require the existing end-to-end threshold, PCC >= 0.98.
2. Profile ten child-trace replays at global `T=5120`; take the median of the slowest device in sessions 3--11. Report the raw range and profiler report directory with the result.
3. Attribute a gain to the union of dependent operations on the slowest device. Never add overlapping operation durations to claim an end-to-end saving.
4. Retain a change only if it remains below 2.958 ms and improves the steady-state median by at least 15 us, unless it is a necessary enabler for a separately measured next step. Fifteen microseconds exceeds the observed replay spread enough to avoid accepting noise as an optimization.
5. After a retained SP2xTP4 improvement, rerun the TP8 control before making a comparative claim. Keep all existing trace-safety barriers and persistent allocations unless the experiment explicitly validates a replacement.

```bash
# Functional gate: use only the guarded runner.
KDA_SP_SPLIT_AFFINE=1 KDA_SP_SOCKET_LANES=2 \
  KDA_SP_SOCKET_FIFO_BYTES=524288 KDA_SP_TARGET_SHAPE=1 \
  scripts/run_safe_pytest.sh -q -s \
  models/experimental/kimi_delta_attention/tests/test_sp2_tp4.py

# Performance gate: sessions 3--11 are the steady-state sample.
KDA_SP_SPLIT_AFFINE=1 KDA_SP_SOCKET_LANES=2 \
  KDA_SP_SOCKET_FIFO_BYTES=524288 PERF_SEQ=5120 PERF_REPS=10 \
  PERF_CHILD_TRACE=1 scripts/run_safe_pytest.sh --profile -q -s \
  models/experimental/kimi_delta_attention/tests/perf/test_kda_sp2_tp4_layer_perf.py::test_kda_sp2_tp4_layer_device_perf
```

## Work sequence

### 0. Freeze and reproduce the baseline

**Status: complete.** Preserve the current `2.855 ms` report as the control. Before changing code, rerun the functional command and one profile if the hardware or runtime environment changes. This avoids chasing a board/runtime shift as if it were a code regression.

### 1. Make the output-tail dependency explicit

**Goal:** determine why the runtime `CloneOperation` exists and whether it is required for layout, lifetime, ownership, or a downstream consumer.

- Correlate the MRS and clone program hashes, tensors, allocations, and consumers on each TP4 rank. Record the exact dependency timeline: producer completion, clone start/end, and first consumer start.
- Verify whether the MRS result can be normalized to `[B, T, H/TP]` as a view or reshape rather than a materialized copy; preserve the known singleton-dimension handling that fixed the earlier diagnostic bug.
- Confirm that persistent buffers, trace capture, and SP-boundary handoff do not require the clone to create a fixed-address tensor.

**Exit criterion:** a short trace note names the clone's owner and consumer, and classifies it as removable, replaceable by a persistent destination, or required. This step does not claim a performance gain.

### 2. Remove or hide the clone without changing collective topology

**Goal:** recover the clone's exposed tail while keeping the proven fused Line MRS implementation.

Try these in order; each is a separate experiment:

1. Let the downstream layer consume the MRS result directly, using a metadata reshape/view when possible.
2. Allocate a persistent downstream-owned output destination and make the MRS result land in the required lifetime/layout, if the runtime API supports it.
3. Keep the existing result allocation but alias its lifetime safely across trace replays, eliminating only the redundant materialization.

**Expected critical-path gain:** 0--37 us directly. A larger result is only credible if the profile proves that removing the copy also removes CCL/memory contention. Do not claim that its 830 us duration is an 830 us layer win.

**Exit criterion:** retain only a PCC-clean change with a measured >=15 us end-to-end gain, or record why the clone is semantically required and proceed to step 3.

### 3. Reduce the fused TP4 output MRS tail

**Goal:** cut at least 30 us from MRS itself after, or together with, the clone work. Combined with the exposed clone tail, that is sufficient to make the 2.803 ms stretch target credible.

Prioritized experiments:

1. **Fuse the output layout contract.** Avoid a post-MRS layout conversion by having the matmul/MRS output already use the consumer's required shape, page layout, and persistent address.
2. **Producer/collective pipelining.** Determine whether the output projection can expose tiles to the Line reduce-scatter earlier without changing output numerics or trace order. Measure MRS start time and end time separately; earlier launch that does not shorten the dependent union is not a win.
3. **Joint matmul/MRS resource tuning.** Change grid, buffering, or tile scheduling only when the profiler identifies an MRS-side idle/bandwidth cause. Tune one dimension at a time and retain the setting only under the measurement contract.

**Expected critical-path gain:** 20--80 us for a justified schedule/layout improvement; 50--150 us is possible only from a real producer--CCL pipeline or epilogue fusion. These are hypotheses to test, not promised gains.

**Do not repeat known dead ends:** Ring output collectives deadlocked; the fixed Line MRS is the correct topology. Previous direct sweeps of worker counts, buffer counts, chunk cadence, and the `10x8` output grid did not produce a long-span win. Revisit one only with a new profiler-identified cause, not as a blind parameter sweep.

**Exit criterion:** reach <=2.803 ms if the output-tail work provides a clean 52 us total saving. Continue toward <=2.750 ms only while each retained change meets the 15 us evidence threshold.

### 4. Optimize the next unhidden local stage

**Goal:** after the output tail no longer dominates, re-profile and work only on the new largest dependent union.

The current likely order is causal short convolution, initial projection, then chunk GDN preparation. For causal convolution specifically, test a genuine producer/consumer fusion or cache/layout change before micro-tuning workers: the TP4 two-channel-block implementation exists to fit L1 and is a correctness constraint. Any fusion must retain both convolution-state PCC and the post-boundary-token check.

**Expected gain:** 20--60 us only if the affected portion is exposed on the layer critical path. Its standalone 452 us duration is not a budget that can be added to the output-tail saving.

### 5. Protect SP overlap while improving local work

**Goal:** ensure local output improvements do not re-expose sequence-parallel latency.

For every winner, compare affine prefix, socket transfer, barrier, grouped final scan, and output-tail timing before/after. The prefix uses the stable two-lane configuration; do not tune socket FIFO size, prefix lane count, or Hillis--Steele schedule unless one of those spans becomes visibly unhidden.

The required proof remains: no host state copy, no host stage fence on the captured path, fixed trace allocations, and full two-call cache reuse.

### 6. Transfer the winning design to Galaxy SP8xTP4

**Goal:** validate that the LB winner survives the native 32-device topology without incorrectly extrapolating LB latency.

1. First capture a clean Galaxy slowest-device baseline with the same output layout and Line TP4 MRS contract.
2. Apply one retained LB output-tail improvement at a time and repeat the Galaxy PCC, trace-stability, and ten-replay tests.
3. Attribute the new prefix/transport cost separately from the TP4 output tail. Only then decide whether prefix overlap is the new bottleneck.

The deployment stretch goal remains <=1.339 ms for the production-rank layer budget, but it is not an LB acceptance criterion. Galaxy performance claims start only after the first native traced baseline exists.

## Decision order and stop conditions

```text
output clone provenance
        |
        v
clone/layout elimination -----> fused MRS tail reduction
        |                                 |
        +--------- re-profile ------------+
                                          |
                     output tail no longer dominant?
                              | yes
                              v
                    causal conv / next unhidden stage
                              |
                              v
                       Galaxy SP8xTP4 transfer
```

Stop an experiment immediately and restore the last accepted configuration if it fails PCC, destabilizes trace replay, exceeds 2.958 ms, or has no reproducible >=15 us benefit. Do not trade away correctness, fixed-address trace safety, or device-queued SP ordering for an isolated microbenchmark.

## Completion definition

The LB portion is complete when the best reproducible configuration is documented with its code/configuration, PCC result, ten-replay raw values, profiler report ID, and TP8 control comparison; all lower-risk output-tail options have been either retained or ruled out with evidence. `<=2.803 ms` is the required stretch outcome. `<=2.750 ms` is the strong target; `<=2.700 ms` requires a separately demonstrated fused output-path improvement.

Galaxy completion is separate: a native `SP=8 x TP=4` trace must meet the protocol and correctness gates before its performance is compared with any LB number.
