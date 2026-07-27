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
| Previous retained SP2xTP4 control | 2.762 ms (`2026_07_26_21_55_50`) |
| Historical SP2xTP4 control | 2.737 ms (18 steady samples; `2026_07_26_23_18_49`, `2026_07_26_23_20_34`) |
| Current retained SP2xTP4 result | **2.571 ms** (18 steady samples; `2026_07_27_00_09_30`, `2026_07_27_00_10_48`) |
| Current steady-state range | 2.561--2.582 ms |
| Revalidated TP=8 control at the same global shape | **3.074 ms** (`2026_07_27_00_04_23`) |
| Improvement versus revalidated TP=8 | **16.35%** |
| Accepted primary gate | <= 2.933 ms |
| Existing stretch gate | <= 2.803 ms (**met**) |
| 2.700 ms fusion target | **met** |

The trace used `KDA_SP_SPLIT_AFFINE=1`, two socket lanes, a 512 KiB socket FIFO, and the default fused QKV untilize path. Sessions 1--2 are cold/warm outliers; all comparisons use the median of sessions 3--11 from ten child-trace replays.

## Current assessment and active order

The retained implementation has removed the trace-owner output clone and now
reads the causal QKV prefix directly from the tiled projection, writing direct
Q/K/V results. The current **2.571 ms** result is the LoudBox layer frontier.
There is **no KDA-local consumer
after the output MRS**: the layer returns that tensor to its caller.  An MRS
"first-consumer layout fusion" is consequently an end-to-end model-integration
project, not a valid next layer benchmark experiment.  It must not displace a
measurable KDA dependency on LoudBox.

The 2.571 ms trace reclassifies the local stages: output MRS is the 854 us
stage but overlaps its 773 us waiting copy and has no KDA-local consumer;
initial projection is 420 us and Chunk-GDN preparation is 311 us. The input
projection retains all 4,232 channels for gate/decay/beta, already uses the
balanced 12x10 generic-matmul schedule, and exposes no source-backed local
fusion beyond the retained tiled producer. Chunk-GDN's mixed-precision and
layout alternatives were exhausted in bring-up. Neither is authorized for a
configuration sweep.

This is the active order for the rest of the LB campaign:

1. The causal-convolution reader/compute probes are exhausted; the tiled
   projection producer is retained with direct Q/K/V outputs.
2. Initial projection and Chunk-GDN preparation are ruled out for further LB
   sweeps by their balanced/exhausted source contracts and lack of an exposed
   data-contract change.
3. Move MRS-output/next-layer fusion and native SP8 transport work to a model
   integration or Galaxy run respectively.  LB layer latency cannot validate
   either claim.

### Remaining performance envelope

| Work item | Why it is still plausible | LB layer gain to seek | Stop condition |
| --- | --- | ---: | --- |
| Causal-conv reader path | The native reader issues 32 row reads for each of four taps per work item; a legal shape-aware NOC/data-layout reduction could shorten a real dataflow dependency. | 15--35 us | Profiler shows compute/writer dominates, or the layer gain is <15 us. |
| Causal-conv compute path | Each time tile performs four tilizes and 192 BF16 multiply/add tile operations for a 48-tile channel block. | 15--40 us | A change adds L1 pressure, loses PCC/trace safety, or does not shorten the dependent union. |
| Fused projection-to-convolution producer | It could remove the remaining row-major QKV materialization/read boundary while retaining the 4,232-channel projection for gates. | 30--70 us | No design can preserve auxiliary gates, convolution state, and trace-stable ownership without a second materialization. |
| Next exposed GDN/projection stage | Both are sizeable only if a fresh trace proves their layer contribution is no longer hidden. | 15--40 us | No source-backed fusion or resource cause; do not sweep configs. |
| Model-level MRS handoff | The MRS output has no local consumer, but a real following model op may consume its layout directly. | 15--50 us E2E only | No end-to-end first consumer or no shrinking E2E dependency union. |

The credible LB frontier remains **2.700 ms**.  It needs 37 us from the
2.737 ms control, so it likely requires the projection-to-convolution fusion
or an unusually successful causal-convolution reduction.  A lower number is
not forecast until a traced dependency shows it is attainable.

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

The clone ends only about 37 us after MRS. Therefore, removing the clone by itself can save **at most about 37 us** on the original critical path; it could not by itself reach the original 52 us stretch goal. That work is now retained. The causal-convolution accumulator experiment subsequently proved that a local stage can still have an exposed contribution. Further KDA-layer wins must shorten the next re-profiled exposed local dependency; MRS-tail work requires an end-to-end consumer.

| Endpoint | Required saving from 2.737 ms | Required reduction of the current 916 us output-tail union |
| --- | ---: | ---: |
| Existing stretch: 2.803 ms | **met by 41 us** | 4.5% |
| Next statistical retention point: 2.722 ms | 15 us | 1.6% |
| Strong LB result: 2.750 ms | **met** | n/a |
| Frontier LB result: 2.700 ms | 37 us | 4.0% |

`2.750 ms` is now exceeded. Because a separately retained change still needs 15 us evidence, the next independent acceptance point is `2.722 ms`. `2.700 ms` remains a frontier target, not a commitment: it requires a material reduction in a traced dependency rather than a cosmetic scheduling change.

## Performance target hierarchy

The purpose is to find the lowest *reproducible* layer latency, not to stop at
the first number below the historical target. The targets below prevent both
under-optimizing a now-close stretch target and accepting replay noise as a
win.

| Tier | LB slowest-device median | Evidence required | Decision |
| --- | ---: | --- | --- |
| Working control | 2.737 ms | PCC/trace gates and two ten-replay reports | Starting point for every experiment |
| Product stretch | <=2.803 ms | Same measurement contract | **Met** by the retained 8x7 configuration |
| Statistical improvement | <=2.722 ms | >=15 us versus 2.737 ms, PCC-clean | Next independently retainable step |
| Strong result | <=2.750 ms | At least two stable profiles and TP8 control rerun | **Met** by fused QKV materialization |
| Frontier result | <=2.700 ms | A demonstrated fused producer/CCL or epilogue change | Explore only after the strong path is exhausted |

Small, directly coupled changes may be evaluated as one bundle when neither is
meaningful on its own. Every other change must clear the 15 us retention
threshold against the 2.737 ms control; a value merely below 2.803 ms is not
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
| B | **QKV materialization fusion (retained).** Produce exactly the 3,072-channel causal-convolution input from the 4,232-channel tiled projection without a tiled slice plus a second untilize, while preserving the auxiliary-gate view of the projection. | 24.8 us measured | Generic op eager/replay, SP PCC/trace stability, and two profiles | `untilize_with_unpadding` block-interleaved writer and the KDA projection-to-convolution boundary |
| C | **Causal-convolution dependency reduction (next).** First classify reader/NOC, compute, and writer time; then make one shape-specific reader, layout, or compute change to the exposed component. The two-block policy and three-block balance probe are explicitly ruled out. | 15--40 us | The layer dependent union shrinks by >=15 us; target/state PCC and trace replay remain clean | `reader_kda_causal_conv1d.cpp`, `kda_causal_conv1d.cpp`, causal-conv factory |
| D | **Projection-to-causal producer fusion.** Remove a full projection-to-convolution store/read only if the same producer can preserve the full 4,232-channel auxiliary-gate view and causal state contract. | 30--70 us | The saved boundary is visible in the layer union; no hidden second materialization or unsafe trace alias | projection output contract and KDA causal-conv input boundary |
| E | **Re-profile and optimize one exposed local stage.** Choose only initial projection or Chunk-GDN preparation after C/D.  No generic matmul, GDN, worker-grid, or mixed-precision sweep is in scope. | 15--40 us | The selected stage is exposed in the slowest-device trace and a change beats its control by >=15 us | Stage-specific factory and profiler timeline |
| F | **Model/transport handoff.** Investigate MRS output-layout fusion only with a real next-layer consumer; investigate SP transport only on native Galaxy after LB local work is exhausted. | 15--50 us | An E2E or native-SP dependent union, respectively, shortens without host synchronization | model integration / `tt/sp_layer.py` |

The campaign has three concrete LB endpoints:

| Endpoint | Target | Interpretation |
| --- | ---: | --- |
| Best current retained result | **2.571 ms** | Direct MRS output, 8x7 TP4 producer grid, direct-Q/K/V tiled producer |
| Strong practical endpoint | <=2.750 ms | **Met** |
| Fusion endpoint | <=2.700 ms | **Met** by tiled projection-to-causal fusion |

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
   against the **2.737 ms** baseline, keep the raw samples and identify the
   changed dependent union on the slowest device.
4. Retain only a clean >=15 us improvement.  Re-run it once from a fresh
   process for the strong-target path; then re-run TP8 before publishing a new
   SP2xTP4-versus-TP8 comparison.
5. For a loss, ambiguous result, or a win under 15 us, restore the last
   retained configuration and add the result to the execution log.  Do not
   convert noise into a default.

### MRS pipeline investigation (completed; do not repeat on LB)

**Decision:** the implementation was PCC- and trace-clean, and shortened the
MRS kernel, but the immediate A/B layer gain was only 12.133 us.  It is below
the 15 us retention threshold and has been reverted.  The mapping below is
kept as design evidence only; it is not the next implementation step.

The best remaining opportunity is not a new grid setting.  It is the coarse
readiness boundary between the TP4 output matmul and its fused Line
reduce-scatter (MRS).  The current two-batch pipeline overlaps batch 0 MRS
with batch 1 matmul, but within each batch all 56 matmul producers complete
before every Line-RS worker is released.  At local `T=2560`, a TP4 output slice
is 18 N tiles; the retained 8x7 matmul grid has eight N columns and seven M
rows.  Line-RS workers consume roughly five M rows per partition.  A useful
pipeline must therefore prove that each receiver sees exactly the tiles it is
about to read; waking the existing global semaphore earlier is incorrect.

| Step | Deliverable and decision | Expected layer effect |
| --- | --- | ---: |
| 1. Restore a clean control | Rebuild/install the unmodified CCL host extension, then recover the device only through the guarded test runner. Capture the target PCC control before changing an ABI. | 0 us |
| 2. Make the kernel ABI cache-safe | Any reader runtime-argument layout change must also change the directly compiled `line_reduce_scatter_minimal_async_reader.cpp` source or its compile-time defines. Header-only ABI edits can reuse a stale JIT kernel that decodes the old arg count and hangs. Prove the deployed Python extension and JIT kernel are from the same source revision before a device run. | Safety gate |
| 3. Build a non-invasive readiness map | Derive `Line RS logical worker -> [M, 18] pages -> matmul M-band` from existing stripe offsets. Choose M-band release: all eight N-column producers for a 12-row matmul band must finish before that band is released. | Design gate |
| 4. Add per-M-band readiness without a reader ABI change | Allocate one epoch semaphore per matmul M band. Each band elects one of its eight producers after a row-local synchronization; it signals only the physical Line-RS cores whose stripe intersects that band. The reader derives its one or two needed bands from its existing stripe offsets. | 0 us; required enabler |
| 5. Validate the complete schedule while feature-gated | Run target-shape PCC, recurrent/convolution-state checks, and two direct-output trace replays with the new schedule. The disabled path must reproduce the accepted control bit-for-bit at the operation-contract level. | Correctness gate |
| 6. Measure and retain/revert | Profile ten replays. Retain only if the slowest-device layer median is <=2.747 ms, the MRS-to-consumer union visibly shrinks, and a second profile agrees. Otherwise revert the feature and record whether the limiter was producer synchronization, RS transport, or a newly exposed stage. | 15--100 us hypothesis |
| 7. Re-profile before further work | If the union does not shrink, restore the control and select C (output layout/epilogue) or D (new exposed local stage); do not tune semaphore counts or worker grids blindly. | Evidence-driven |

The safe design has two invariants: a receiver waits for every M band that
intersects its own pages, and a band is released only after all eight of its
N-column producers have committed their tiles. Batch epochs must remain
monotonic across trace replays so a prior replay cannot satisfy a later wait.
The implementation intentionally uses a distinct semaphore per M band, rather
than mixing independent bands in a shared counted semaphore: a Line-RS stripe
needs either one or two band epochs, never an ambiguous partial count. This
keeps the reader runtime ABI unchanged. A neutral one-word reader-ABI probe
stalled in the Line-RS transport wait and remains reverted.

For the retained `T=5120` LB shape, the host-side mapping is exact. Each of
the two KDA batches contains an `80 x 72`-tile matmul output. TP4 scatters the
72 N tiles into four 18-tile slices. The 8x7 matmul grid owns `12 x 9` tiles
per producer (the final M row is padded), while Line RS has 16 logical workers
per direction/link set. `reduce_scatter_get_tile_offsets()` assigns each
logical worker 90 pages, i.e. five M rows times 18 N tiles. The two physical
directions duplicate the same logical stripe and must receive the same
readiness epoch.

| RS logical worker | M rows it consumes | TP4 matmul M bands required | Release condition |
| --- | --- | --- | ---: |
| `3k` | `15k..15k+4` | one | all eight N columns in that band complete |
| `3k+1` | `15k+5..15k+9` | one | all eight N columns in that band complete |
| `3k+2` | `15k+10..15k+14` | two adjacent | each band is independently complete |

The last partial group follows the same intersection rule rather than a
hard-coded count. This proves that a single early global signal is unsafe, and
also shows why the first completed M band can start useful RS work before all
56 producers complete.

#### Measured pipeline decision and endgame

| Outcome of step 6 | Interpretation | Next bounded action | Best credible endpoint |
| --- | --- | --- | ---: |
| `<=2.747 ms`, MRS union shrinks | The release barrier was exposed and the pipeline is real. | Confirm once; then inspect whether the first consumer can overlap the final RS tail. | 2.700--2.747 ms |
| PCC/trace clean, but <15 us gain | Most MRS time is transport or the old barrier was hidden. | Revert the schedule and pursue output-layout/epilogue fusion; do not add more semaphore granularity. | 2.747--2.762 ms |
| PCC/trace failure or hang | Mapping, epoch, or cached-kernel contract is unsafe. | Revert; retain the mapping analysis only. Verify generated program hashes and device state through the guarded runner. | 2.762 ms control |
| MRS improves but layer does not | A following local stage now dominates. | Re-profile the slowest-device dependent union and select only causal-conv, initial projection, or GDN if it is exposed. | Depends on re-profile |

The post-pipeline priority order is fixed by evidence, not preference:

1. Eliminate an exposed MRS output layout/store or make it the first consumer's
   persistent layout.
2. If that is already direct, overlap the final MRS tail with a dependency-safe
   first consumer, or optimize exactly one newly exposed local stage.
3. Only after local work is exhausted, revisit SP transport on the basis of a
   native `SP=8 x TP=4` trace.

This is the full LB endgame: no grid sweep, semaphore micro-tuning, or socket
parameter sweep is in scope unless the preceding profiler report names the
relevant span as the slowest exposed dependency.

### QKV materialization fusion (retained)

The canonical control's exposed local sequence was input projection (416.776
us), tiled QKV slice (79.879 us), untilize (93.513 us), and causal convolution
(423.534 us). The direct projected-input attempt missed retention by 0.816 us
because it widened untilize. The public operation instead exposed a generic
block-interleaved writer bug: tiles wholly past a narrower requested output
row underflowed the write-size calculation and issued an invalid NOC transfer.

| Step | Deliverable and decision | Gate |
| --- | --- | --- |
| 1 | Add an exact-shape tiled `[1, 2560, 4232] -> [1, 2560, 3072]` operation-level regression. | Eager and captured replay pass via `scripts/run_safe_pytest.sh`. |
| 2 | Repair the block-interleaved writer to consume out-of-output tiles without a NOC write. | Generic width-unpadding behavior; no KDA special case. |
| 3 | Make KDA directly untilize the QKV prefix while retaining full projection storage for auxiliary gates. | Target PCC, state checks, and two direct-output trace PCC replays pass. |
| 4 | Measure twice at global `T=5120`, then rerun TP8. | 2.738 and 2.735 ms reports; 2.737 ms combined median; TP8 3.073 ms. |

`KDA_FUSED_QKV_UNTILIZE=1` is now the default and `=0` is the diagnostic
opt-out. The next active item is the causal-convolution dependency analysis
below. The initial projection already uses a balanced 12x10/120-core generic
matmul schedule; there is no program-config sweep in this plan. Chunk-GDN
mixed-precision knobs have likewise been exhausted in bring-up work and are
not a low-risk next experiment.

### Causal-convolution dependency plan (active)

This is a two-experiment maximum, not an open-ended kernel-tuning exercise.
The operation consumes row-major QKV and produces tiled Q/K/V.  At TP4 its
`Ct=96` channels are deliberately split into two `worker_Ct=48` blocks;
`distribute_channel_blocks()` maps 80 time tiles in each block over 60 cores,
leaving 40 cores with two work items and 80 with one.  The four tap tensors are
read once into each worker's L1 and reused.  That makes changing the block
count, grid, or tap caching a known bad direction unless a new profile proves
otherwise.

| Step | Exact action | Success evidence | Exit / next action |
| --- | --- | --- | --- |
| C1: classify | Capture the standard ten-replay trace, inspect the slowest-device `KdaCausalConvOperation` BRISC/NCRISC/TRISC timings and correlate them with the immediately dependent QKV/Chunk-GDN timeline. Record whether reader, compute, or writer is the exposed limiter. | One named subkernel dominates a layer-visible dependency; raw sessions and report ID are recorded. | If no component is exposed, skip directly to E; do not tune causal conv. |
| C2a: reader only | If dataflow dominates, prototype one legal shape-aware reduction in row-read transactions or a producer layout that avoids a repeated conversion. Preserve the 48-tile L1 budget and state-boundary reads. | Target PCC, state/boundary PCC, two trace replays, then >=15 us layer gain twice. | Retain or revert; no alternate reader variants after one negative result. |
| C2b: compute only | If TRISC dominates, prototype one fusion that removes a demonstrated conversion/pass while retaining four-tap causal order and BF16 destination behavior. | Same correctness gate and a shorter causal-conv-to-consumer union, not only a shorter kernel. | Retain or revert; then proceed to D. |
| C3: producer fusion | Only if C2 cannot reach 15 us, write a design note for a fused projection-to-causal producer. It must provide QKV rows in causal order while retaining the full projection for the gate branch. | Architecture review proves a single materialization is truly removed before code changes. | Implement only after the design is proven; otherwise move to E. |

The source locations are intentionally explicit: the distribution and L1
budget live in `chunk_gdn_phased_program_factory.cpp`; the 32-row/four-tap
dataflow is in `reader_kda_causal_conv1d.cpp`; the four tilize/multiply/add
passes are in `kda_causal_conv1d.cpp`.  This prevents a generic worker-grid
sweep from being mistaken for causal-convolution optimization.

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

### 2026-07-26: per-M-band Line-MRS readiness pipeline (ruled out for LB retention)

The fused Line MRS barrier was replaced behind `KDA_MRS_STRIPE_PIPELINE=1`
with one readiness epoch per 12-row matmul band.  Eight N-column producers
synchronize within a band, then its elected master signals exactly the Line-RS
physical cores whose five-row stripe intersects that band.  Each reader derives
its one or two required M bands from its existing tile offsets; no reader
runtime argument was added.  This is the safe granularity for the retained 8x7
producer grid.

An initial target-shape timeout was traced with `tt-triage` to a host runtime
argument framing defect: the fine path emitted the legacy 56-worker group size
before its row-local eight-worker `OpSignaler` payload.  The kernel therefore
decoded the following arguments at the wrong offsets.  Removing that legacy
word from the fine path fixed the schedule. Target-shape PCC and the two-replay
direct-output trace PCC then passed.

The MRS kernel did improve materially, but the layer did not improve enough to
retain the additional scheduling complexity:

| Configuration | Report | Steady sessions 3--11, slowest-device layer median | MRS median | Decision |
| --- | --- | ---: | ---: | --- |
| Per-M-band pipeline | `2026_07_26_22_44_51` | **2751.764 us** (2748.298--2758.142) | 759.953 us | Clean, but provisional |
| Immediate global-release control | `2026_07_26_22_46_45` | **2763.897 us** (2752.394--2769.175) | 767.412 us | Control |

The immediate A/B layer difference is **12.133 us**, below the independent
15 us retain threshold, even though the pipeline reduces the MRS duration.
That proves the residual MRS saving is largely hidden by the layer dependency
graph.  The feature-gated implementation is reverted; retain the mapping and
the ABI lesson, but do not add semaphore granularity or a different producer
grid without a profiler result that makes MRS an exposed layer tail. The next
work is output-layout/first-consumer fusion, then the newly largest exposed
dependent local stage.

### 2026-07-26: direct projected-input causal convolution (ruled out for LB retention)

The slowest-device trace showed an exposed QKV-prefix materialization before
the causal convolution: slice the leading 3,072 Q/K/V channels from the
4,232-channel projection, then untilize the slice.  The causal-convolution
reader was temporarily extended to consume the QKV prefix directly from the
wider row-major projection, using its true 4,232-element row pitch while
leaving the projection live for the auxiliary-gate path.  This removed the
materialized QKV slice but widened the untilize input.

The focused SP2xTP4 PCC and the two-replay direct-output trace PCC both pass.
The two independent ten-replay profiles were too close to the retention floor:

| Report | Sessions 3--11, slowest-device layer median | Gain versus 2.762 ms control |
| --- | ---: | ---: |
| `2026_07_26_23_00_32` | 2747.938 us | 13.689 us |
| `2026_07_26_23_02_15` | 2746.364 us | 15.263 us |

The combined 18-sample median is **2747.443 us**, a **14.184 us** gain: below
the independent 15 us retention rule.  The direct reader and relaxed input
contract are reverted.  The result confirms that this boundary is near the
right scale, but the wider untilize absorbs enough of the slice saving that a
future attempt must remove or avoid that conversion as well; it must not be
re-enabled as a default on the basis of the single 2746.364 us run.

### 2026-07-26: projected-to-QKV `untilize_with_unpadding` fusion (ruled out as unsafe)

The next low-code attempt was to use the public `untilize_with_unpadding`
operation directly on the tiled 4,232-channel projection, ending at the
3,072-channel QKV prefix.  This would have retained the established causal
convolution ABI while removing both the tiled slice and its separate untilize.
The opt-in target-shape test timed out before it reached numerical comparison;
the guarded runner captured `tt-triage` data and automatically reset all eight
devices.  The timeout subsequently surfaced on the next queued allocation,
which means the new untilize path is not safe for this captured interleaved
projection shape.

The opt-in source is reverted and no profiler result is claimed.  Revisit this
only as a dedicated untilize implementation/factory investigation with a
minimal operation-level reproduction and trace-safety proof; it is not a
valid layer-level tuning flag.

### 2026-07-26: generic QKV `untilize_with_unpadding` repair (retained)

The minimal exact-shape regression showed that the wide-row heuristic selected
the block-interleaved untilize writer for the tiled `[1, 2560, 4232]`
projection. The writer partitions work from the padded input width. For a
requested 3,072-channel output, rightmost producer tiles lie wholly outside
the destination row. The previous `width_size - (end - output_width)` clamp
underflowed when `start_column > output_width`, producing an invalid enormous
NOC transfer and the earlier trace stall.

The generic writer now consumes those CB tiles but emits a write only for their
intersection with the output row. The new operation-level regression covers
both eager execution and captured replay at the exact KDA shape. KDA uses the
direct projection-to-QKV untilize by default; `KDA_FUSED_QKV_UNTILIZE=0`
retains the former slice-plus-untilize path for diagnosis.

* Target-shape SP2xTP4 PCC and two direct-output child-trace replays pass,
  including recurrent state, convolution state, and the boundary token.
* Report `2026_07_26_23_18_49` steady sessions 3--11 are `[2737.836,
  2746.530, 2737.366, 2742.068, 2732.651, 2747.897, 2743.244, 2738.461,
  2731.667] us`; median **2738.461 us**.
* Independent report `2026_07_26_23_20_34` sessions are `[2740.970,
  2738.348, 2735.878, 2728.221, 2729.490, 2734.105, 2735.421, 2736.233,
  2727.979] us`; median **2735.421 us**.
* The combined 18-sample median is **2736.799 us**, a **24.828 us (0.90%)**
  gain over the former 2761.627 us retained control. It clears the independent
  15 us retention rule.
* The required TP8 rerun, `2026_07_26_23_23_02`, has a **3072.976 us** median
  from sessions 3--11. SP2xTP4 is **336.177 us (12.28%)** faster at the same
  global `T=5120` shape.

### 2026-07-26: prefix-only block-untilize schedule (ruled out for LB retention)

After retaining generic width-unpadding correctness, the block-interleaved
factory was temporarily changed to schedule only the 96 output-prefix tiles
per KDA row while retaining the input's 133-tile reader stride. The exact
operation regression, target-shape PCC, and direct-output trace PCC all pass.
The runtime was rebuilt and installed before profiling; the report therefore
contains the actual factory schedule, not only the JIT writer update.

Report `2026_07_26_23_31_00` reduced the canonical untilize operation from
about 104 us to **93.791 us** (112 to 99 cores), but its slowest-device layer
sessions 3--11 have a **2730.847 us** median. That is only **5.952 us** below
the 2736.799 us retained control, so the extra factory scheduling complexity
does not clear the 15 us retention gate. The schedule is reverted and the
runtime rebuilt; keep only the safe writer underflow fix and the retained QKV
fusion.

### 2026-07-26: causal-convolution C1/C2 probes (ruled out; proceed to C3 design)

The retained report `2026_07_26_23_20_34` identifies causal convolution as a
reader/compute lock-step operation rather than a writer-limited operation. On
the slowest checked device/session its 422.977 us firmware span consists of a
421.799 us BRISC reader and a 419.870 us TRISC1 compute kernel. This confirms
that the causal reader and four-pass tilize/multiply/add loop are the only
useful local targets; it also rules out writer and worker-grid tuning.

Two bounded C2 experiments were completed and reverted:

1. Alternating the independent activation-row reads between the two Blackhole
   NOCs timed out during the target-shape PCC gate. `tt-triage` captured
   causal readers in `noc_async_read_barrier()` while compute waited for the
   activation CB. The normal runner recovered the device. This is a route/
   transaction-completion incompatibility, not a numerical failure; do not
   retry dual-NOC reader routing without a dedicated NOC contract.
2. Removing the two explicit source-format reconfigurations preceding
   `mul_bcast_rows_init_short()` was PCC- and direct-output-trace clean.
   Report `2026_07_26_23_42_03` produced a 2.732964 ms steady median from
   sessions 3--11 (range 2.726476--2.736628 ms), only **3.835 us** below the
   2.736799 ms control. Causal convolution fell to 416--419 us, proving that
   the local kernel improvement is mostly hidden. The code is reverted because
   it does not satisfy the 15 us layer-retention gate.

**C3 feasibility result:** the producer fusion cannot be implemented as a
small boundary change. The input projection must remain tiled and live at its
full 4,232-channel width for the decay, output-gate, and beta slices. The
native causal op explicitly validates a row-major QKV input. Making it consume
the tiled 3,072-channel prefix needs a new tiled causal-convolution program:
the four causal taps are shifted by 0--3 *rows*, so three taps cross a 32-row
tile boundary. The generic tiled roll path is native only for tile-aligned
shifts and otherwise performs `untilize -> roll -> tilize`, which recreates
the materialization we are trying to remove. There is no existing tile-row
shift primitive to reuse.

The producer-fusion design is therefore a new kernel project with an expected
30--70 us upside, not a patch to the current row-major causal kernel. Do not
add further causal reader/compute micro-variants on LoudBox.

#### Tiled producer project preflight (required before reimplementation)

An earlier prototype exists in history as `83654eb5fc9` and was reverted by
`0c5beb02b0a`. It did prove the desired boundary (read tiled projection prefix,
untilize locally, convolve, write tiled output/state), but its long-path PCC
was invalid: the first three rows of each 32-row block were wrong and attempts
to make cross-tile prefix ownership explicit deadlocked. It also predates the
TP4 `Ct=96` geometry. Do not cherry-pick it.

The replacement must be a new, feature-gated operation with this contract:

1. Split `Ct=96` into the existing two `worker_Ct=48` blocks; a one-block
   tiled prototype exceeds the proven Blackhole L1 envelope. Each block owns
   all 80 time tiles over 60 cores.
2. Read projection tiles with the physical projected row stride (`Pt=133`),
   but materialize only QKV prefix tiles `0..95`. The first raw packed-face
   approach corrupted each tile-boundary prefix (one-tile PCC passed, long
   PCC failed at 0.969238). The retained correctness bridge locally untilizes
   the preceding QKV tile row, then copies its last three row-major rows. It
   is the authoritative source for a future lower-traffic prefix mechanism.
3. Make each channel block write only its own QKV output tiles and its own
   1,536-byte range of the three row-major convolution-state pages. No core
   may write another block's state range.
4. First prove identity-tap exactness at a long sequence with the first three
   rows of *every* tile block checked, then prove four-tap state continuity,
   then run the target SP2xTP4 PCC and two direct trace replays. Only after
   those gates pass may it receive a ten-replay profile.

This is a real device-program project, not a retry of the rejected public
untilize or generic tiled-FIR routes.

#### 2026-07-27: C3 tiled producer retained

`KDA_TILED_PROJECTION_CAUSAL_CONV=1` now selects
`KdaTiledCausalConvOperation` for long prefill. The operation reads the QKV
prefix directly from the tiled 4,232-column projection, splits TP4's 96 QKV
tiles into two 48-tile channel blocks, emits direct tiled Q/K/V outputs plus
the row-major three-token carry, and leaves the full tiled projection live for
the decay/output-gate/beta branch. `KDA_TILED_PROJECTION_CAUSAL_CONV_MIN_SEQUENCE`
is a correctness-only threshold override; the default is 640.

Correctness gates all passed using `scripts/run_safe_pytest.sh`:

1. A one-local-tile TP4 control (`KDA_SP_TEST_SEQ=64`, minimum sequence 32)
   passed, proving channel-block output/state ownership.
2. The target global-T=1280 SP2xTP4 PCC gate passed after replacing raw face
   decoding with a locally-untilized predecessor tile. It covers output,
   recurrent state, convolution state, and the first token after the SP
   boundary at PCC >= 0.98.
3. The direct-MRS two-child-trace PCC gate passed with the tiled producer.

Two independent global-T=5120, ten-replay child-trace profiles passed:

| Report | Sessions 3--11 slowest-device spans (us) | Median |
|---|---|---:|
| `2026_07_27_00_00_36` | 2604.784, 2606.611, 2600.627, 2608.664, 2608.067, 2602.567, 2598.269, 2604.676, 2606.572 | 2604.784 |
| `2026_07_27_00_02_12` | 2600.023, 2604.060, 2600.996, 2598.458, 2595.775, 2604.390, 2598.846, 2605.879, 2603.247 | 2600.996 |

The combined 18-sample median is **2603.654 us / 2.604 ms**, a **133.145 us
(4.87%)** improvement over the retained 2736.799 us control. It clears the
2.700 ms stretch threshold. The new fused causal operation's slowest-device
firmware median is about 410.4 us (kernel about 356.1 us); the generic
`UntilizeWithUnpadding` QKV materialization is absent from the trace. The
remaining `SliceDeviceOperation` wait span is about 282 us, so the next
optimization must use a new trace to establish whether its consumers are now
exposed.

The required refreshed TP8 control is
`2026_07_27_00_04_23`: sessions 3--11 are 3079.379, 3076.447, 3073.614,
3077.724, 3085.540, 3069.881, 3065.909, 3066.701, and 3065.869 us, for a
**3073.614 us / 3.074 ms** median. This is statistically consistent with the
prior 3072.976 us TP8 control. The combined SP2xTP4 tiled-producer median is
therefore **469.960 us (15.29%) faster** than current TP8 at the same global
T=5120.

#### 2026-07-27: direct Q/K/V writer retained

The first tiled producer wrote one QKV tensor and then used three width-slice
operations. The slowest-device trace showed three approximately 28 us physical
slice kernels plus a 280 us dependent wait. The producer writer now routes each
local channel tile directly to Q, K, or V while preserving the two-block state
partition. The target PCC and two-child direct-output trace gate both pass.

Two independent global-T=5120 ten-replay profiles passed:

| Report | Sessions 3--11 slowest-device spans (us) | Median |
|---|---|---:|
| `2026_07_27_00_09_30` | 2581.081, 2572.864, 2570.663, 2575.945, 2571.013, 2565.995, 2560.956, 2574.577, 2576.965 | 2572.864 |
| `2026_07_27_00_10_48` | 2577.650, 2567.856, 2567.586, 2581.818, 2581.197, 2570.644, 2569.687, 2571.597, 2570.639 | 2570.644 |

The combined 18-sample median is **2571.305 us / 2.571 ms**. It is **32.349
us** below the combined-QKV producer and **165.494 us (6.05%)** below the
2736.799 us control, so it is retained as the current LoudBox configuration.

## Measurement contract

Every retained change must satisfy all of the following.

1. Run the focused functional gate first, including target-shape output, recurrent-state, convolution-state, and boundary-token PCC. Require the existing end-to-end threshold, PCC >= 0.98.
2. Profile ten child-trace replays at global `T=5120`; take the median of the slowest device in sessions 3--11. Report the raw range and profiler report directory with the result.
3. Attribute a gain to the union of dependent operations on the slowest device. Never add overlapping operation durations to claim an end-to-end saving.
4. Retain a change only if it remains below 2.933 ms and improves the 2.737 ms working control by at least 15 us, unless it is a necessary enabler for a separately measured next step. Fifteen microseconds exceeds the observed replay spread enough to avoid accepting noise as an optimization.
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
`2.737 ms` as the retained working control. Before changing code, rerun the
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
the next independently retained change now needs 15 us beyond 2.737 ms.

### 3. Reduce the fused TP4 output MRS tail

**Status: deferred from the KDA-layer campaign.** The KDA layer returns the
MRS tensor; it does not launch its next model consumer.  The remaining
output-layout/epilogue idea therefore has no layer-local producer/consumer
union to optimize.  Its only valid implementation home is an end-to-end model
integration benchmark.  The MRS scheduling results and constraints below are
kept as historical evidence and a handoff note, not as active LB work.

**Status: LB scheduling work exhausted for now.** The 8x7 source-backed
producer-grid change is retained; worker-policy forwarding and the PCC-clean
per-M-band producer/collective pipeline are ruled out.  The latter shortened
MRS but improved the full layer by only 12.133 us.  Do not repeat worker,
buffer, chunk-count, grid, or semaphore sweeps without a new trace proving an
exposed MRS tail; follow the QKV materialization plan above first.

If an end-to-end harness supplies a real first consumer, the only prioritized
experiments are:

1. **Fuse the output layout contract.** Avoid a post-MRS layout conversion by
   having the matmul/MRS output already use the consumer's required shape,
   page layout, and persistent address.
2. **Overlap the final MRS tail with a dependency-safe first consumer.** Do
   this only if a fresh trace shows an exposed producer/consumer boundary.
3. **Joint matmul/MRS resource tuning.** Change grid, buffering, or tile
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

**Exit criterion:** an end-to-end retained change reaches its independently
measured target.  It cannot change the KDA-layer 2.737 ms control.  Continue
the LB layer campaign with section 4.

### 4. Optimize the next unhidden local stage

**Status: active. Goal:** retain the C3 tiled producer, then re-profile and
work only on the next largest dependent union.

The tiled producer has now completed the causal-convolution step and is
retained behind its explicit environment flag. Its TP4 two-channel-block
implementation is a correctness and L1 constraint. A three-channel-block
balance probe remains not retained (3.210 us layer difference; unchanged
convolution span). Next, use the new slowest-device trace to choose exactly
one exposed union; do not reopen causal micro-tuning.

The BF16-destination causal-convolution configuration is already retained as
the default. `KDA_CAUSAL_CONV_FP32_ACC=1` remains the accuracy-control
override. Do not reopen this accumulator choice unless a new correctness or
production-accuracy requirement demands it.

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
retained clone + QKV materialization removals
                    |
                    v
      causal-conv subkernel classification (C1)
             | reader/compute exposed
             v
   one bounded causal-conv experiment (C2)
             | >=15 us layer win
             v
     retain and re-profile next dependency
             | no causal opportunity / <15 us
             v
 projection-to-causal fusion design (C3) --> initial projection or GDN
             |
             v
 Galaxy SP8xTP4 native profile; model-level MRS handoff separately
```

Stop an experiment immediately and restore the last accepted configuration if it fails PCC, destabilizes trace replay, exceeds 2.933 ms, or has no reproducible >=15 us benefit. Do not trade away correctness, fixed-address trace safety, or device-queued SP ordering for an isolated microbenchmark.

## Completion definition

The LB portion is complete when the best reproducible configuration is documented with its code/configuration, PCC result, ten-replay raw values, profiler report ID, and TP8 control comparison; causal-conv C1/C2, producer-fusion feasibility, and every profiler-proven next local dependency have been either retained or ruled out with evidence. `<=2.803 ms` is the required stretch outcome. `<=2.750 ms` is the strong target; `<=2.700 ms` requires a separately demonstrated causal/projection fusion or another traced local dependency reduction.

Galaxy completion is separate: a native `SP=8 x TP=4` trace must meet the protocol and correctness gates before its performance is compared with any LB number.
