# Next GDN decode candidate: restore fused local convolution

CPU-only audit. No device execution or TTNN import by this worker. Existing
files/defaults are unchanged. The new [candidate module](../../tt/recurrent_next_candidate.py) contains
an opt-in local convolution wrapper, a state-publication source transformer,
and a pure host value-column partition plan. None is hardware-validated.

The strongest next experiment is **fusing the seven convolution binary ops,
BF16 SiLU and history update**, before spending more effort on the already
optimized recurrence SFPU. The multichip `_local_batched_conv` currently
implements four FP32 products, three ordered additions (last output BF16),
BF16 SiLU, history slice/concat/copy. The frozen single-chip decoder already
has an arithmetic-compatible fused untilize-once helper, but the multichip
override does not use it. Its hardcoded10240 width is an adaptation issue,
not a reason to retain all these dispatches for local width2560.

## Evidence and expected scope

The parent's `optimized_multichip_decoder/perf/interim_profile2_l0_b32_s33/`
decode report shows the convolution region as four BinaryNg multiplies
(roughly6us each), two FP32 BinaryNg adds (~7 and5us), final BF16 add (~6us),
BF16 unary SiLU (~5us), plus input/history slices and concat/copy. Those seven
binaries are about42us of the reported64us total binary time. This is a
specific dispatch/data-movement opportunity adjacent to recurrence, with no
change to its BF16 state/FP32 live arithmetic.

This report uses the merged decode report for those approximate durations.
Raw trace CSV attributes sometimes retain different logical-shape metadata
from cached calls, and raw per-rank timings do not equal merged report values;
do not infer a shape bug or promise exact savings from them. Current B32
recurrence is about125us and total uninstrumented decode about673us as supplied
by parent. Removing every42us convolution binary would be only a6.2% ceiling
on that different-regime total; the replacement kernel still costs time.
No achieved gain is claimed.

Source contracts inspected:

- `tt/multichip_decoder.py::_local_batched_conv`, `_gdn`, `_local12_recurrent`.
- `tt/optimized_decoder.py::_conv107_conv_decode_sfpu` and its three
  `_conv107__CONV_RM_PIPE2_*` kernels; `_fused_delta_row_GDN_COMPACT_*` kernels.
- `tt/recurrent_variants.py::sources`, including `cumulative_exp_scalar`.
- Previous diagnosis `doc/optimized_multichip_decoder/debug_recurrent/AUTODEBUG.md`.

The dead-padding/constants/grouped-QK/gate-cache/exp experiments are not
repeated. State-streaming takes the already-selected transformed sources.

## Implemented opt-in convolution candidate

`local_conv(token, history, taps, output=None, cores=None)` adapts frozen
untilize-once pipeline2 to local shapes:

| Tensor | Shape / dtype / layout |
|---|---|
| compact projection input | `[1,B,2560]` BF16 TILE DRAM |
| raw token after one untilize | `[1,B,2560]` BF16 ROW_MAJOR DRAM |
| persistent raw history | `[B,3,2560]` BF16 ROW_MAJOR DRAM, updated in place |
| each tap | `[1,1,2560]` BF16 ROW_MAJOR DRAM |
| output consumed directly by recurrence | `[B,1,2560]` BF16 ROW_MAJOR DRAM |

Each user has three independent contiguous channel chunks:1024,1024,512
elements. Their byte offsets are0,2048,4096 and lengths2048,2048,1024 in a
5120-byte row. Reader and writer transfer only the valid last1024 bytes; there
is no overrun into the following row. The last chunk's unused CB half is dead:
the unchanged compute has only elementwise operations, and only the first
1024 output bytes are written. Tests must poison the unused half before
acceptance if there is any concern about hidden lane coupling. There is no
reduction, transpose or broadcast in this convolution compute.

At B32 there are96 tasks, hence96 active cores and one task/core on11x10.
The existing two-entry CB protocol is retained unchanged for smaller core
limits, including odd final task groups. CBs0–7,9,10 remain two BF16 tiles
each:40KiB/core. At B1 there are three tasks/cores; latency may regress despite
fewer dispatches, so B1 must be measured separately.

The compute C++ string is **byte-identical** to the frozen helper. It retains:

1. Four distinct FP32 products, then the same left-associated FP32 sum.
2. Explicit native FP32→BF16 rounding before SiLU, matching the multichip
   `ttnn.silu(ttnn.add(..., dtype=BF16))` boundary.
3. Native BF16 SiLU calculation while retaining the correct FP32 destination
   address mode, then BF16 output packing. No FP32 SiLU substitution or FMA.
4. History writes sourced from old slot1, old slot2 and raw current token;
   carry never contains convolved output. Writer completion tokens protect
   input storage; compute alone advances input CB read pointers.

The wrapper keeps the existing one-time native untilize. It removes the
standalone product/add/SiLU kernels and history slice/concat/copy. It does not
fuse model projections, alter GDN gates, or touch recurrence state arithmetic.

The [probe CLI](../../tests/probe_multichip_conv_candidate.py) supplies an
instance-only full-layer hook installed by a temporary `from_state_dict`
wrapper. It preserves the existing implementation for legacy-shaped prefill
tails and records actual candidate call counts. Equivalent manual decode hook:

```python
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_next_candidate import local_conv

original_conv = model._local_batched_conv
model._local_batched_conv = lambda qkv, history: (
    local_conv(qkv, history, model.taps_rm)
    if tuple(qkv.shape) == (1, history.shape[0], 2560)
    else original_conv(qkv, history)
)
```

The wrapper deliberately requires compact `[1,B,2560]` decode. Do not install
it on a legacy `[B,1,2560]` route without an explicit entry adapter. B1 is
covered only when the parent's `batched_conv_b1` selects this function. Eager
warmups and capture must preserve the same history/output ownership as current
decoder tests; an optional preallocated output can be provided by the caller.

Validation order: compare candidate and original convolution with separate
identical nonzero histories and distinct rank/user/channel tokens, requiring
**bitwise output and updated-history equality** over multiple decode steps.
Use B1/2/3/17/31/32, default cores plus `cores=1`/`cores=7` to cross ring wraps,
and values near the BF16 rounding/SILU transition boundaries. Then test
unchanged complete recurrence output/state, full-layer changed-input trace,
watcher, and current decoder accuracy gates. Time complete convolution with
untilize and carry update, then whole decode with the selected cumulative
recurrence variant. Do not compare only the fused kernel against a whole
unfused boundary.

## Parent-run CLI sequence

Run these under the parent's serialized device wrapper from the repository
root. This worker executed CPU checks only. The focused probe requires exact
BF16 bit patterns for **both outputs and raw histories**, for every rank/user
and every eager/changed-input trace step. It uses independent identical history
allocations, distinct rank/user/channel inputs, and compares each trace to its
saved eager baseline. The default extra B3/one-core case exercises nine tasks
through the two-slot pipeline, including the odd final group. Timing covers
untilize, convolution, SiLU and history writes; queued trace timing alternates
baseline/candidate order and uses endpoint synchronization.

Mandatory focused smoke first:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_conv_candidate \
  --batches 32 --low-cores 0 --steps 4 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/conv_smoke.json
```

Full shape suite, optional zero-input/history modes, and dead-padding poison:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_conv_candidate \
  --modes normal zero_qkv zero_history \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/conv_exact_suite.json
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_conv_candidate \
  --batches 3 32 --cores 1 --low-cores 0 --poison-dead-padding \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/conv_poison.json
```

The poison build writes BF16 NaNs to the unused upper 1024 bytes of each tail
chunk in all eight input CBs after reservation and before reading useful bytes.
The frozen elementwise compute remains byte-identical. Poison mode disables
timing; it must produce finite useful outputs and exact raw histories.

After focused exactness passes, forward the full-layer runner's options:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_conv_candidate \
  --full-layer --trace --repeats 100 --layer 0 --batch 32 --length 33 \
  --reference models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/delivery_perf_ref_l0_b32_s33.pt \
  --policy-json '{"flat_prefill":true,"flat_finish":true,"flat_prefill_correction":true,"persistent_ccl":true,"recurrent_variant":"cumulative_exp_scalar"}' \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/conv_full_l0_b32_s33.json
```

Full-layer B1 **requires `"batched_conv_b1":true`**, whose default is false.
The CLI fails if its hook is never exercised. This flag also selects raw-packed
`cumulative_exp_scalar` recurrence at B1, so compare the coherent family against
both the same-flag unfused control and the original policy baseline. The same-
flag comparison isolates convolution; the original-policy comparison includes
the recurrence route change. Exact B1 commands, in that order:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder \
  --trace --repeats 100 --layer 0 --batch 1 --length 512 \
  --reference models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/delivery_perf_ref_l0_b1_s512.pt \
  --policy-json '{"flat_prefill":true,"flat_finish":true,"flat_prefill_correction":true,"persistent_ccl":true,"recurrent_variant":"cumulative_exp_scalar","batched_conv_b1":true}' \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/conv_b1_same_flag_control.json
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_conv_candidate \
  --full-layer --trace --repeats 100 --layer 0 --batch 1 --length 512 \
  --reference models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/delivery_perf_ref_l0_b1_s512.pt \
  --policy-json '{"flat_prefill":true,"flat_finish":true,"flat_prefill_correction":true,"persistent_ccl":true,"recurrent_variant":"cumulative_exp_scalar","batched_conv_b1":true}' \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/conv_b1_family.json
python -m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder \
  --trace --repeats 100 --layer 0 --batch 1 --length 512 \
  --reference models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/delivery_perf_ref_l0_b1_s512.pt \
  --policy-json '{"flat_prefill":true,"flat_finish":true,"flat_prefill_correction":true,"persistent_ccl":true,"recurrent_variant":"cumulative_exp_scalar"}' \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/conv_b1_original_policy.json
```

Full-layer mode imports the existing runner only after parsing, forwards its
arguments without changing its validation/trace logic, and appends source
hashes, candidate/legacy hook counts and failure details to its JSON. It does
not edit the decoder, its defaults, or the frozen baseline. The focused probe
captures both implementations separately and releases both traces explicitly.
Neither path performs a reset. Watcher runs and uninstrumented performance runs
remain separate parent commands.

## Smaller recurrence-only candidate: stream persisted value columns

`stream_state_sources(reader, compute, writer)` expects the already-selected
`cumulative_exp_scalar` reader/compute and frozen writer. It returns all three
sources; caller must wire the returned writer too. It does not change CB
descriptors, runtime arguments, state dtype/layout, reader, or arithmetic.

Current compute retains four K-row state tiles in DST for each value column
`n`, completes prediction/update/current FP32 readout, packs persisted state,
and publishes nothing until all four value columns finish. Writer therefore
starts each head's32KiB state transfer after all its arithmetic.

The transformer replaces one reserve/push16 with four reserve/push4 groups
on CB11. It packs the current column contiguously at indices0..3, and writer
maps stream `(n,m)` back to original page `16*h+4*m+n`. Writer waits for four
state tiles, writes that column, barriers, then pops those four. Output CB12
still publishes/writes four readout tiles per complete head. Reader CB5 already
holds all old state before compute starts, so earlier persistence cannot
overwrite unread old state. Compute never rereads persisted CB11: live update
and readout remain in FP32 DST, while only persisted copies receive native
BF16 RNE. Every arithmetic helper and its order are unchanged.

This overlaps state writes with later value-column compute but saves zero
DRAM bytes and zero SFPU instructions. It adds per-column publication/barriers
and can lose. It is more conservative than discarding BF16 RNE or changing
readout to use rounded state, both of which change the numerical contract.

Exact integration change required in a parent-owned wrapper:

```python
reader_source, compute_source = sources(..., "cumulative_exp_scalar", grouped=group_qk)
writer_source = _fused_delta_row_GDN_COMPACT_WRITER
reader_source, compute_source, writer_source = stream_state_sources(
    reader_source, compute_source, writer_source
)
# Existing reader/compute descriptors stay identical except these source strings.
# Replace the writer descriptor's source argument with writer_source.
```

Current `_local12_recurrent(source_overrides=...)` accepts only reader/compute,
so passing this three-source result directly would be wrong. Parent must
extend that hook or use an isolated wrapper. No such existing-file edit was
made here. Keep this independent from convolution so timing attribution is
clear. Exact old/new recurrent output **and state**, low-core multi-head
wraps, changed-input trace, and watcher are required before selection.

The separate [streaming probe](../../tests/probe_multichip_stream_state.py)
now supplies that isolated wrapper. It extracts the existing recurrent probe's
`main` via AST into a private namespace and changes only the compared sources,
binding, metadata and warmup ordering. Baseline is `cumulative_exp_scalar`;
candidate is the same reader/compute with `stream_state_sources` applied.
Both compute wrappers are fixed `FunctionType` snapshots with private globals:
one binds the frozen writer, the other the streaming writer. The dispatcher
checks the candidate triple's writer against that fixed binding, then forwards
the reader/compute pair. No model globals are changed at setup or per call.
Every device tensor and both kernels' warmups precede both trace captures.

CPU preparation, then parent-serialized first smoke and complete controls:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_stream_state --cpu-check
TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_stream_state \
  --batches 32 --state-dtypes bf16 --modes normal --low-cores 0 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/stream_state_smoke.json
TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_stream_state \
  --batches 1 2 3 17 31 32 --state-dtypes bf16 fp32 --modes normal zero_qkv zero_state \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/stream_state_suite.json
```

The default extra B3/one-core case crosses user, grouped-QK and state-CB ring
boundaries. `--grouped-qk off` provides the independent grouping control;
`--poison-dead-padding` poisons both matched cumulative sources and disables
timing. The retained gates require exact integer bit patterns for outputs and
state, finite values for every rank/user, changed-input traces versus saved
eager outputs/state, and equality after matched timed replay counts.

The concrete protocol hazards are addressed as follows:

- Reader CB5 contains the entire old head after its NoC read barrier and
  `state.push_back(16)`. Compute waits for all 16 before publishing an updated
  column. Early DRAM writes cannot change old tiles still needed in CB5.
- CB11 retains 32 pages. Four-page reserve/push and wait/pop groups divide that
  capacity exactly, so no group crosses the ring end. Writer offsets `m`
  within each group map to matrix page `16*h + 4*m + n`.
- Writer barriers each four-page DMA before popping its source group. Compute
  never reads CB11; live update/readout stays in FP32 DST. The old comment
  requiring complete output before reading persisted state is not a live
  dependency of this fused compute.
- CB12 remains eight pages with four-page publication per head, preserving
  output/padding writes and their final barrier. State BF16/FP32 format,
  native persistence rounding, descriptors and core ownership are unchanged.
- This adds four state-write barriers plus the original final output barrier
  per head, versus the old single barrier. It may increase NoC contention or
  synchronization cost despite earlier writes; no gain is claimed.

If the parent later prefers a regular triple-source hook, the minimal optional
patch in `_local12_recurrent` is:

```python
writer_source = _fused_delta_row_GDN_COMPACT_WRITER
if source_overrides is None:
    # Existing reader/compute source selection remains here.
    ...
else:
    assert len(source_overrides) in (2, 3)
    reader_source, compute_source = source_overrides[:2]
    if len(source_overrides) == 3:
        writer_source = source_overrides[2]
# Writer descriptor only:
kernel(writer_source, ttnn.WriterConfigDescriptor(), wr, wct)
```

That patch is not needed for the new standalone probe and was not applied.
The CPU check validates both grouping/poison choices, unchanged arithmetic
helpers, fixed writer snapshots, exact state page mapping and four-page CB
wraps up to 384 heads/core. Device compilation, exactness, tracing and speed
remain parent-run work.

## More intrusive scheduling alternative: value-column ownership

The new pure host `value_column_partition` returns a concrete work plan, not a
device implementation. It splits the independent `(head,n)` space into
contiguous ranges, preserving each column's complete four-K-segment reduction
on one core. No K reduction moves between cores, so the native SFPU reduction
order need not change.

At B32 existing head ownership gives54 cores four heads and56 cores three,
i.e. max16 value columns/core. Column ownership assigns1536 columns over110
cores, max14: a12.5% reduction of the maximum column count, not a predicted
latency gain. It produces440 head fragments instead of384 and220 Q/K
preparations instead of164, so normalizing extra Q/K can erase the gain. At B1
it offers48 cores/one column versus12 cores/four columns, a more plausible
parallelism win, with duplicated Q/K preparation and fixed-launch costs.

Exact necessary edits for an isolated implementation:

- Use returned `(first_head, head_count, first_n, last_n_exclusive)` in all
  three kernels' runtime args. For each head, apply the first/last column masks
  only on the matching boundary head; interior heads own n0..3.
- Reader fetches state page `16*h+4*m+n` only for owned columns, retaining
  original CB offsets. Otherwise it wastes state traffic or could read state
  another core has already overwritten. Read all four raw V tiles initially
  for simple protocol compatibility; only selected V columns are consumed.
- Compute keeps the same ordered m0..3 loops and state/readout arithmetic,
  but changes only the outer n range. Output/state CBs can initially retain
  their full capacities/reserve counts; unwritten entries must never be read.
- Writer emits only owned state pages and output n tiles. Each `(h,n)` owns
  disjoint64-byte row-face output slices. Tail-padding writes must also be
  restricted to owned n tiles, otherwise final-head fragments duplicate them.
- Grouped-QK prepare/pop predicates keep their global-head logic, with local
  boundary preparation. State/readout collection must restore original layout
  directly; no additional reduction or tensor gather is permitted.

This is a larger protocol change than state streaming and should follow the
small convolution experiment. No device source for this plan is claimed.

## State bandwidth limits and rejected shortcuts

BF16 state is384 x128 x128 x2 =12,582,912 bytes/device at B32; one read and one
write is25,165,824 bytes, about49.15us at the parent's512GB/s ceiling. The
ordinary recurrent kernel cannot reduce those bytes while preserving the
same per-step state contract. Streaming changes overlap, not this floor.

Keeping one layer's state in L1 could produce misleading isolated timings:
each of48 GDN layers needs this12MiB state at B32, totaling576MiB/device before
weights/other activations, far beyond aggregate worker L1. A final stacked
model cannot retain every layer's state this way. Sequential logical state
tiles are DRAM-interleaved; one contiguous32KiB DMA/head is not legal without
a physical placement change. Unpacking weights/state at lower precision,
factoring decay outside the prediction reduction, FMA replacement, TF32
contraction, and reading rounded persisted state for current readout are
excluded from this exactness-preserving proposal.

Gate fusion is a separate modest follow-up: `_decay` performs FP32 bias add,
standalone FP32 softplus and FP32 negative-A multiply; beta is native BF16
sigmoid. Fusing these requires preserving the native sigmoid template,
FP32 softplus mode and explicit BF16 beta rounding, then writing the current
`[B,1,12]` gate layout directly to remove two reshape kernels. Merely setting
`unfused_g=False` changes an already-explicit arithmetic choice and is not an
exact custom implementation. No gate fusion code or measured claim is included
because the reused exact convolution helper is a more concrete next step.

## CPU checks performed

- Source rewrite anchors match current frozen convolution and selected
  cumulative recurrence strings; convolution compute is byte-identical.
- For B1–32, all convolution byte ranges cover each5120-byte output/history
  row exactly once, with64-byte alignment and no tail overrun.
- 160 value-column ownership cases (B1–32, core limits1/12/48/96/110) cover
  each `(head,n)` exactly once. B1 and B32 fragment counts above are calculated
  from the actual partition helper.
- Python AST/compile and formatting checks; no hardware claim follows from
  source/ownership validation. Runtime compilation, exactness and speed remain
  parent-run work.
- Probe/candidate import checks actively reject TTNN imports. CLI forwarding
  preserves `--batch` for the existing runner instead of abbreviating it to
  focused `--batches`. Both normal and NaN-poison source transforms retain
  byte-identical compute, and 160 convolution ownership/core-limit cases cover
  every useful byte exactly once. Black/isort checks passed in `python_env`.
