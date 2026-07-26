# KDA sequence-parallel best-performance plan

## Purpose

This is the execution plan for getting the best defensible end-to-end KDA sequence-parallel performance from the current implementation. It starts from the proven LoudBox `SP=2 x TP=4` trace, because that is the only topology we can measure today, and transfers only validated mechanisms to Galaxy `SP=8 x TP=4`.

It optimizes the *slowest device's traced layer critical path*, not a host interval or a standalone collective. The existing protocol, correctness paths, and serial implementation remain the oracle while this work is in progress.

For topology, correctness, and bring-up history, see [SP_LOUDBOX_PLAN.md](SP_LOUDBOX_PLAN.md).

## Starting point and targets

The current accepted control is a complete, child-mesh captured `SP=2 x TP=4` forward at global `T=5120` on LoudBox:

| Metric | Current value |
| --- | ---: |
| Historical cloned SP2xTP4 control | 2.855 ms (`2026_07_26_15_59_04`) |
| First direct-output SP2xTP4 control | 2.813 ms (`2026_07_26_21_17_29`) |
| Previous retained SP2xTP4 control | 2.790 ms (`2026_07_26_21_37_43`) |
| Retained SP2xTP4 working control | **2.762 ms** (`2026_07_26_21_55_50`) |
| Retained-control steady-state range | 2.756--2.777 ms |
| Revalidated TP=8 control at the same global shape | **3.076 ms** (`2026_07_26_21_57_35`) |
| Improvement versus revalidated TP=8 | **10.22%** |
| Accepted primary gate | <= 2.958 ms |
| Existing stretch gate | <= 2.803 ms (**met**) |
| Strong LB target | <= 2.750 ms |
| Frontier LB target | <= 2.700 ms |

The trace used `KDA_SP_SPLIT_AFFINE=1`, two socket lanes, and a 512 KiB socket FIFO. Sessions 1--2 are cold/warm outliers; all comparisons use the median of sessions 3--11 from ten child-trace replays.

The original cloned trace identified the following critical-path candidates on a
representative slow device. The direct-output report contains no clone, so MRS
is the first stage to re-profile and classify before changing its schedule:

| Operation / path | Duration | What it means |
| --- | ---: | --- |
| Fused TP4 output `MatmulReduceScatterAsyncDeviceOperation` | 879 us | Largest output-projection compute/CCL stage. |
| Runtime output `CloneOperation` | 830 us | Overlaps the MRS, but extends the output tail. |
| Union from MRS start to clone end | about **916 us** | First optimization target. |
| Causal short convolution | 452 us | Second local compute candidate, after proving its unhidden contribution. |
| Initial matmul | 420 us | Do not tune before the output tail. |
| Chunk GDN preparation | 310 us | Do not tune before the output tail. |
| Affine prefix | 181 us | Correct and not currently the dominant unhidden span. |

The clone ends only about 37 us after MRS. Therefore, removing the clone by itself can save **at most about 37 us** on the original critical path; it could not by itself reach the original 52 us stretch goal. That work is now retained. The causal-convolution accumulator experiment subsequently proved that a local stage can still have an exposed contribution. Further wins must shorten the fused MRS tail or the next re-profiled exposed dependent stage.

| Endpoint | Required saving from 2.762 ms | Required reduction of the current 916 us output-tail union |
| --- | ---: | ---: |
| Existing stretch: 2.803 ms | **met by 41 us** | 4.5% |
| Next statistical retention point: 2.747 ms | 15 us | 1.6% |
| Strong LB result: 2.750 ms | 12 us | 1.3% |
| Frontier LB result: 2.700 ms | 62 us | 6.7% |

`2.750 ms` is the best credible near-term endpoint. Because a separately retained change still needs 15 us evidence, reaching it alone is not sufficient to retain a new independent micro-change. `2.700 ms` is a frontier target, not a commitment: it requires a material reduction in the fused output path rather than a cosmetic scheduling change.

## Performance target hierarchy

The purpose is to find the lowest *reproducible* layer latency, not to stop at
the first number below the historical target. The targets below prevent both
under-optimizing a now-close stretch target and accepting replay noise as a
win.

| Tier | LB slowest-device median | Evidence required | Decision |
| --- | ---: | --- | --- |
| Working control | 2.762 ms | PCC/trace gates and two ten-replay reports | Starting point for every experiment |
| Product stretch | <=2.803 ms | Same measurement contract | **Met** by the retained 8x7 configuration |
| Statistical improvement | <=2.747 ms | >=15 us versus 2.762 ms, PCC-clean | Next independently retainable step |
| Strong result | <=2.750 ms | At least two stable profiles and TP8 control rerun | Continue only if the next exposed critical path has a concrete cause |
| Frontier result | <=2.700 ms | A demonstrated fused producer/CCL or epilogue change | Explore only after the strong path is exhausted |

Small, directly coupled changes may be evaluated as one bundle when neither is
meaningful on its own. Every other change must clear the 15 us retention
threshold against the 2.762 ms control; a value merely below 2.803 ms is not
automatically evidence of an independent improvement.

## Best-performance campaign

This is the ordered LB campaign.  It is deliberately narrow: each row is a
specific critical-path hypothesis, rather than a parameter sweep.  The result
of a row is either a retained configuration with a ten-replay result or a
written negative result.  That is how we know that the reported best result is
the best one we could defensibly obtain on LoudBox, rather than simply the last
number observed.

| Order | Change class and hypothesis | Expected exposed layer saving | Retain only if | Primary evidence / implementation area |
| --- | --- | ---: | --- | --- |
| A | **Causal-conv accumulator probe (retained).** The four-tap BF16 causal convolution does not need an FP32 destination accumulator for the covered KDA contract, reducing local compute/L1 pressure without changing the rest of KDA. | 28 us measured | Target/trace PCC is clean and two profiles beat the pre-change 2.790 ms control by >=15 us | Default in `tt/layer.py`; `KDA_CAUSAL_CONV_FP32_ACC=1` is the accuracy-control override |
| B | **MRS producer/RS pipeline design.** Replace the all-producers-complete release with a per-slice readiness contract so a reduce-scatter slice can start only after its matching output tiles exist. | 30--100 us | A source-level dependency proof, PCC/trace stability, then >=15 us layer gain | Fused MRS factory, `OpSignaler`, Line RS worker partition and semaphores |
| C | **MRS epilogue/output-layout fusion.** Make output pages/layout/persistent ownership exactly match the next consumer, removing an exposed conversion or excess output write. | 15--50 us | A profile shows the dependent union shrinks; no unsafe output alias/deallocation | MRS output spec, consumer layout, trace-owned persistent buffers |
| D | **Re-profile and optimize the new exposed local stage.** After A--C, choose only the largest *dependent union*: causal convolution, initial projection, or chunk GDN. | 15--60 us | The selected stage is exposed in the slowest-device trace and a change beats its control by >=15 us | Stage-specific factory and profiler timeline |
| E | **Cross-stage overlap/fusion.** Only once local stages are individually near their limits, overlap a proven producer/consumer boundary or remove a materialized intermediate. | 20--80 us | The layer union decreases; standalone op duration is not accepted as evidence | Adjacent device programs, fixed trace buffers, cache handoff |
| F | **SP transport re-tuning.** Consider prefix lanes/FIFO/scan schedule only if the re-profile shows transport is exposed after A--E. | 10--40 us | The SP prefix-to-consumer dependency shortens without changing order or host synchronization | `tt/sp_layer.py`, socket transport profile |

The campaign has three concrete LB endpoints:

| Endpoint | Target | Interpretation |
| --- | ---: | --- |
| Best current retained result | 2.762 ms | Direct MRS output, 8x7 TP4 producer grid, BF16-destination causal convolution |
| Next proof point | <=2.747 ms | One independently reproducible >=15 us saving |
| Strong practical endpoint | <=2.750 ms | 12 us beyond the baseline; the next independent change still needs >=15 us evidence |
| Frontier endpoint | <=2.700 ms | Requires 62 us; treat as a design project (B/C/E), not tuning |

The sum of the upper bounds in the campaign is **not** a forecast: these
operations overlap.  The only valid total is the traced slowest-device layer
span.  A 2.700 ms result is possible only if profiling proves that at least one
currently serial dependency has been shortened or overlapped.

### Execution protocol for each campaign row

1. Write down the dependency being shortened and the maximum plausible exposed
   saving before changing code.  If there is no dependency-level hypothesis,
   do not run the experiment.
2. Run the focused PCC test with `scripts/run_safe_pytest.sh`, then the
   two-replay direct-output trace PCC when ownership, layout, cache, or program
   order changes.
3. Run the ten-replay `T=5120` child-trace profile.  Compare sessions 3--11
   against the **2.762 ms** baseline, keep the raw samples and identify the
   changed dependent union on the slowest device.
4. Retain only a clean >=15 us improvement.  Re-run it once from a fresh
   process for the strong-target path; then re-run TP8 before publishing a new
   SP2xTP4-versus-TP8 comparison.
5. For a loss, ambiguous result, or a win under 15 us, restore the last
   retained configuration and add the result to the execution log.  Do not
   convert noise into a default.

### LB exhaustion and Galaxy handoff

LB optimization is considered exhausted when rows A--F have either been
retained or ruled out with the protocol above, no remaining slowest-device
dependent union has a source-backed path to >=15 us, and the best configuration
has a second confirming profile.  At that point we publish the lowest measured
LB number, even if it is above the frontier target.

The Galaxy handoff is intentionally not a multiplication of LB latency by four.
The first native `SP=8 x TP=4` run must establish its own slowest-device trace;
then transfer retained A--E changes one at a time.  The desired production
property is the same per-device local work as LB SP2xTP4, while the global
sequence and SP transport are larger.  Any new exposed prefix/scan cost is a
Galaxy-specific optimization problem and must not be hidden by an LB claim.

## Execution log

### 2026-07-26: direct persistent MRS output

**Status: retained as the first direct-output LoudBox trace configuration.** The profiler
identified the output `CloneOperation` as the helper's explicit ownership copy:
the fused MRS API requires a persistent output buffer, while the normal helper
shares that buffer and must return a separately deallocatable tensor. The
result is not a reshape/view issue.

`KDA_MRS_DIRECT_OUTPUT=1` now lets a trace-only owner return that persistent
MRS result directly and retain it through trace replay. The ordinary path
continues to clone by default, so generic callers may deallocate their output.
The SP2xTP4 profiler is the explicit owner for the direct mode and does not
free those aliased outputs.

* The target-shape eager PCC gate and a new two-replay child-trace PCC gate
  pass. The latter checks both first- and second-call outputs after the
  device-resident recurrent/convolution cache changes.
* Ten child-trace replays at global `T=5120` produce no `CloneOperation` rows
  in report `2026_07_26_21_17_29`.
* Slowest-device sessions 3--11 are `[2808.091, 2816.711, 2819.827,
  2820.824, 2811.938, 2812.636, 2811.222, 2813.978, 2810.359] us`; median
  is **2812.636 us**. This is a **42.414 us (1.49%)** improvement over the
  2855.050 us cloned control and clears the 15 us retention threshold.
* At this point it remained 9.636 us above the 2803 us stretch goal. The next
  experiment was therefore MRS-side scheduling/layout work, not more SP socket
  tuning.

Run the retained LB configuration with:

```bash
KDA_MRS_DIRECT_OUTPUT=1 KDA_SP_SPLIT_AFFINE=1 KDA_SP_SOCKET_LANES=2 \
  KDA_SP_SOCKET_FIFO_BYTES=524288 PERF_SEQ=5120 PERF_REPS=10 \
  PERF_CHILD_TRACE=1 scripts/run_safe_pytest.sh --profile -q -s \
  models/experimental/kimi_delta_attention/tests/perf/test_kda_sp2_tp4_layer_perf.py::test_kda_sp2_tp4_layer_device_perf
```

### 2026-07-26: TP8 comparison rerun

The comparison control was rerun after retaining direct MRS output, as required
by the measurement contract. Report `2026_07_26_21_24_59` gives slowest-device
sessions 3--11 of `[3105.559, 3095.442, 3095.908, 3097.722, 3094.181,
3098.621, 3108.671, 3107.304, 3100.724] us`, with a **3098.621 us** median
and a 3094.181--3108.671 us range. The 2812.636 us SP2xTP4 result is therefore
285.985 us (9.23%) faster at the same global `T=5120` shape.

```bash
PERF_SEQ=5120 PERF_REPS=10 PERF_TRACE=1 \
  scripts/run_safe_pytest.sh --profile -q -s \
  models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py::test_kda_tp_layer_device_perf
```

### 2026-07-26: TP4 8x7 fused-MRS producer grid (retained)

The fused MRS source shows that each matmul worker calls
`synchronize_workers_and_signal_op()` only after its complete output region,
and the master waits for every producer before signalling Line RS. The default
8x6 grid therefore gives each producer 14 M tiles. Moving only the TP4 grid
height to seven keeps the eight-column N partition, lowers each producer to 12
M tiles, and leaves the Line RS allocation below `rs_offset=(0,7)`.

`SP2TP4KimiDeltaAttention` now uses 8x7 by default; `KDA_MRS_TP4_GRID_Y=6`
is retained solely as the control override. Target-shape PCC, the two-replay
direct-output trace PCC, and the normal SP2xTP4 suite pass.

* Report `2026_07_26_21_37_43` slowest-device sessions 3--11 are
  `[2793.523, 2797.547, 2786.255, 2789.959, 2792.539, 2791.667,
  2781.467, 2788.001, 2783.329] us`; median **2789.959 us**, range
  2781.467--2797.547 us.
* This is **22.677 us (0.81%)** faster than the 2812.636 us direct-output
  control. It clears the 15 us retention threshold and the 2803 us product
  stretch target by 13.041 us.
* The slowest MRS median is **855.675 us** (range 846.907--874.227 us), down
  from roughly 0.87 ms. This supports the producer-grid hypothesis without
  claiming a tile-level matmul/CCL overlap that the current barrier does not
  provide.

The TP8 control rerun after this retained change is report
`2026_07_26_21_40_09`: sessions 3--11 are `[3109.921, 3110.874, 3098.885,
3088.731, 3095.177, 3090.601, 3096.815, 3095.686, 3102.040] us`, median
**3096.815 us**. SP2xTP4 is thus **306.856 us (9.91%)** faster at the same
global T=5120 shape.

### 2026-07-26: fused-MRS worker-policy forwarding (ruled out)

The fused MRS factory retained `DEFAULT_WORKERS_PER_LINK = 1` in its operation
attributes but did not forward that field to the Line reduce-scatter builder;
the standalone factory does. Restoring the forwarding was a narrow
source-backed experiment, not a topology change. It passed target-shape PCC,
but report `2026_07_26_21_31_56` measured slowest-device sessions 3--11 of
`[2817.941, 2815.936, 2813.137, 2829.679, 2822.624, 2810.516, 2815.147,
2815.279, 2808.969] us`, a **2815.279 us** median. That is 2.643 us slower
than the 2812.636 us working control and has a wider range. The MRS slowest
device remained about 0.87--0.89 ms, so it did not expose a material latency
benefit. The forwarding change was reverted and `_ttnn.so` rebuilt; it is
recorded here to avoid repeating this worker-count hypothesis.

### 2026-07-26: causal-convolution three-block balance (ruled out)

TP4 causal convolution has 96 channel tiles and 80 time tiles. Replacing its
two 48-tile channel blocks (120 workers, with 40 workers carrying two time
tiles) with three 32-tile blocks would have aligned Q/K/V boundaries and given
all 120 workers two equal work items. The target-shape PCC gate passed, but
report `2026_07_26_21_45_15` measured a **2786.749 us** layer median, only
3.210 us below the 2789.959 us control. Its causal-convolution median remained
**453.293 us**. The result is inside replay variation and fails the 15 us
retention gate, so the 32-tile change was reverted and `_ttnn.so` rebuilt.

### 2026-07-26: BF16-destination causal convolution (retained)

The causal convolution is a four-tap BF16 operation.  Its independent FP32
destination accumulator was a plausible local compute/L1 cost, whereas the
rest of the KDA layer continues to use the existing FP32 destination
configuration.  The experiment was first enabled by
`KDA_CAUSAL_CONV_FP32_ACC=0`, then made the default after correctness and a
fresh profile confirmed it.  `KDA_CAUSAL_CONV_FP32_ACC=1` preserves the former
FP32-destination behavior for accuracy investigations.

* Target-shape PCC, convolution/recurrent state checks, and the two-replay
  direct-output trace PCC pass. The normal SP2xTP4 suite passes (3 passed; the
  separately gated trace test skips unless its opt-in variable is set).
* First profile report `2026_07_26_21_53_17` has slowest-device sessions 3--11
  `[2761.136, 2763.313, 2770.807, 2770.165, 2760.508, 2765.293, 2758.296,
  2760.664, 2763.839] us`, median **2763.313 us**. Its slowest causal-conv
  median is **424.463 us**.
* Fresh default-configuration report `2026_07_26_21_55_50` has sessions 3--11
  `[2762.089, 2756.296, 2756.853, 2756.681, 2764.972, 2761.627, 2776.860,
  2763.296, 2759.221] us`, median **2761.627 us**, range 2756.296--2776.860
  us. Its causal-conv median is **424.389 us**.
* The retained 2761.627 us result is **28.332 us (1.02%)** below the 2789.959
  us 8x7/direct-output control. It clears the 15 us retention threshold in
  two independent profiles; no standalone causal-convolution duration was
  added to the layer claim.

The required TP8 comparison rerun is report `2026_07_26_21_57_35`: sessions
3--11 are `[3073.064, 3071.245, 3074.370, 3077.398, 3081.919, 3077.154,
3076.084, 3074.819, 3081.248] us`, median **3076.084 us**, range
3071.245--3081.919 us. SP2xTP4 is therefore **314.457 us (10.22%)** faster
at global `T=5120` under the same default causal-convolution configuration.

### 2026-07-26: TP4 8x8 exact-M balance (ruled out by LoudBox capacity)

KDA's `T=2560` local span has 80 M tiles. An 8x8 matmul grid would therefore
give 64 producers exactly 10 M tiles each, eliminating the four padded M tiles
of the retained 8x7/12-tile configuration. This was a source-backed capacity
probe, not a grid sweep. The target-shape PCC gate passed, but the production
`T=5120` trace could not build: Line RS attempted to allocate logical core
`(0,10)` and the LoudBox worker grid has no such core. The failure occurs in
`ReduceScatterFusedOpSignaler::init_reduce_scatter`, before replay, because the
larger producer grid leaves insufficient rows for the complete Line RS worker
set at the real shape.

The temporary `KDA_MRS_TP4_GRID_Y=8` diagnostic override was reverted. This
proves that 8x7 is the maximum viable TP4 producer grid on this LoudBox for the
full KDA MRS/CCL program; future MRS work must improve the producer/CCL
dependency, worker partition, or output contract rather than consume another
CCL row.

## Measurement contract

Every retained change must satisfy all of the following.

1. Run the focused functional gate first, including target-shape output, recurrent-state, convolution-state, and boundary-token PCC. Require the existing end-to-end threshold, PCC >= 0.98.
2. Profile ten child-trace replays at global `T=5120`; take the median of the slowest device in sessions 3--11. Report the raw range and profiler report directory with the result.
3. Attribute a gain to the union of dependent operations on the slowest device. Never add overlapping operation durations to claim an end-to-end saving.
4. Retain a change only if it remains below 2.958 ms and improves the 2.762 ms working control by at least 15 us, unless it is a necessary enabler for a separately measured next step. Fifteen microseconds exceeds the observed replay spread enough to avoid accepting noise as an optimization.
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

**Status: complete.** Preserve `2.855 ms` as the historical cloned control and
`2.762 ms` as the retained working control. Before changing code, rerun the
functional command and one profile if the hardware or runtime environment
changes. This avoids chasing a board/runtime shift as if it were a code
regression.

### 1. Make the output-tail dependency explicit

**Goal:** determine why the runtime `CloneOperation` exists and whether it is required for layout, lifetime, ownership, or a downstream consumer.

- Correlate the MRS and clone program hashes, tensors, allocations, and consumers on each TP4 rank. Record the exact dependency timeline: producer completion, clone start/end, and first consumer start.
- Verify whether the MRS result can be normalized to `[B, T, H/TP]` as a view or reshape rather than a materialized copy; preserve the known singleton-dimension handling that fixed the earlier diagnostic bug.
- Confirm that persistent buffers, trace capture, and SP-boundary handoff do not require the clone to create a fixed-address tensor.

**Exit criterion: complete.** The clone is the required ownership copy from a
shared persistent MRS output buffer. The direct trace owner can safely retain
that buffer instead of cloning it; generic callers still require the default
clone contract.

### 2. Remove or hide the clone without changing collective topology

**Goal:** recover the clone's exposed tail while keeping the proven fused Line MRS implementation.

Try these in order; each is a separate experiment:

1. Let the downstream layer consume the MRS result directly, using a metadata reshape/view when possible.
2. Allocate a persistent downstream-owned output destination and make the MRS result land in the required lifetime/layout, if the runtime API supports it.
3. Keep the existing result allocation but alias its lifetime safely across trace replays, eliminating only the redundant materialization.

**Expected critical-path gain:** 0--37 us directly. A larger result is only credible if the profile proves that removing the copy also removes CCL/memory contention. Do not claim that its 830 us duration is an 830 us layer win.

**Exit criterion: complete.** Direct output is PCC-clean across two trace
replays and saves 42.414 us. It is retained only for owners that preserve the
persistent output lifetime. It enabled the later 8x7 MRS grid improvement;
the next independently retained change now needs 15 us beyond 2.762 ms.

### 3. Reduce the fused TP4 output MRS tail

**Goal:** find a reproducible MRS-side reduction of at least 15 us, then
continue only while the proven critical path supports progress toward the 12 us
strong-result budget. The product stretch gate is complete; the 15 us minimum
makes the next change statistically retainable.

**Status: in progress.** The 8x7 source-backed producer-grid change is
retained; the worker-policy forwarding experiment is ruled out. The remaining
high-value source change is tile-level producer/collective pipelining: the
current `OpSignaler` waits for every matmul producer before launching Line RS.
First establish a safe per-slice readiness contract; do not repeat worker,
buffer, chunk-count, or grid sweeps without new evidence.

Prioritized experiments:

1. **Prove the MRS limiter.** Inspect the fused MRS factory and profiler
   counters/timeline to classify the long tail. Verify which worker, link,
   buffer, and output-grid controls are actually supported by this operation.
   This is a read-only investigation with a written hypothesis before code is
   changed.
2. **Fuse the output layout contract.** Avoid a post-MRS layout conversion by
   having the matmul/MRS output already use the consumer's required shape,
   page layout, and persistent address.
3. **Producer/collective pipelining.** Determine whether the output projection
   can expose tiles to the Line reduce-scatter earlier without changing output
   numerics or trace order. Measure MRS start time and end time separately;
   earlier launch that does not shorten the dependent union is not a win.
4. **Joint matmul/MRS resource tuning.** Change grid, buffering, or tile
   scheduling only when the proven limiter exposes a supported control. Tune
   one dimension at a time and retain the setting only under the measurement
   contract.

**Expected critical-path gain:** 20--80 us for a justified schedule/layout improvement; 50--150 us is possible only from a real producer--CCL pipeline or epilogue fusion. These are hypotheses to test, not promised gains.

**Do not repeat known dead ends:** Ring output collectives deadlocked; the fixed Line MRS is the correct topology. Previous direct sweeps of worker counts, buffer counts, chunk cadence, and the `10x8` output grid did not produce a long-span win. Revisit one only with a new profiler-identified cause, not as a blind parameter sweep.

The 8x8 exact-M grid is also ruled out on LoudBox: it exhausts the logical
worker-grid rows required by the full Line RS program at `T=5120` (attempted
core `(0,10)`). 8x7 is therefore the largest valid TP4 producer grid without a
CCL worker-partition redesign.

**Pipeline design constraint (confirmed from source):** at TP4 the 72 output-N
tiles form four 18-tile reduce-scatter slices; the 8x7 matmul produces nine N
tiles per column. The actual KDA MRS has two batches. The matmul writer calls
`OpSignaler::synchronize_workers_and_signal_op()` once after each whole batch,
and the Line reader waits with `wait_for_matmul_batch(b)`. It already has a
coarse batch-0 RS / batch-1 matmul pipeline, but each batch signal still waits
for all 56 producers. Within a batch, Line RS workers partition a flattened
`M x 18` slice and process every target slice. A safe finer pipeline therefore
needs either (a) RS work repartitioned by N-stripe, with a readiness semaphore
for the two matching matmul columns across all seven M rows, or (b) a per-M-band
contract that maps each RS worker's flattened range to completed matmul rows.
Signalling the present global semaphore early would let a worker read unwritten
tiles and is not an experiment to run. Preserve persistent buffer addresses,
the two Line directions, the existing batch counter semantics, and the trace
replay contract in either design.

**Exit criterion:** a standalone retained change reaches <=2.747 ms, or a
directly coupled bundle reaches <=2.750 ms and then crosses the 15 us evidence
threshold when measured as a whole. Continue toward <=2.750 ms only while
each retained change meets the 15 us evidence threshold.

### 4. Optimize the next unhidden local stage

**Goal:** after the output tail no longer dominates, re-profile and work only on the new largest dependent union.

The current likely order is causal short convolution, initial projection, then chunk GDN preparation. A three-channel-block balance probe was PCC-clean but
not retained (3.210 us layer difference; unchanged convolution span). For causal convolution, test a genuine producer/consumer fusion or cache/layout change before further worker micro-tuning: the TP4 two-channel-block implementation exists to fit L1 and is a correctness constraint. Any fusion must retain both convolution-state PCC and the post-boundary-token check.

The one low-risk local probe still worth measuring before a fusion is the
opt-in BF16-destination causal-convolution configuration
`KDA_CAUSAL_CONV_FP32_ACC=0`.  It changes only the causal-convolution compute
configuration; the layer default remains FP32 destination accumulation.  It is
not retained on eager PCC alone: it must pass target-shape and two-replay trace
PCC and clear the same 15 us layer threshold.  If it loses or produces a
material PCC/state regression, revert the opt-in path and do not trade accuracy
for its local kernel duration.

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
