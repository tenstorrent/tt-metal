# AutoDebug: expert buffer lifetime and preceding CCL writes

## Verdict

**No model-level use-after-free or direct CCL-payload/L1-output alias is established.**
The production expert method keeps raw gate/up alive until the entire expert
call returns, and its gate/up slices own separate allocations. Attention RS
staging, RS output and AG output are DRAM; raw GU and both slices are L1. A late
write to the intended collective payload address therefore cannot hit GU merely
because an allocator reuses an address. A wrong NoC target, kernel-local L1
overwrite, or producer state error remains possible and needs evidence.

The passing raw-GU retention control has a larger effect than extending GU
through the slices. It changes both the suffix after the expert call and the
**next forward's attention/router allocation prefix**. This confound should be
measured with integer-only addresses before proposing a lifetime fix.

The reported changed columns 0:32 across all eight experts correspond to one
sparse producer, logical worker (0,0), which also sends input A. They map to
eight different slice output pages and slice workers. This favors inspecting
that producer's local state and CB placement over an unqualified external
write to one interleaved output bank. It is a localization clue, not proof.

## Scope and evidence

This is a delegated, source-only AutoFix/AutoDebug audit. The local skill files
and the preceding BF8, indexed-expert and stack-CCL reports were read. No TTNN
import, device access, target execution, reset, runtime change or C++ change was
made. The shared checkout contains other agents' authorized work.

- HEAD: `9a529836fc91b1117a48f6d63ef30455a69cd42d`.
- `tt/multichip_decoder.py` SHA256:
  `8b59370cda6f4ff88157de123123509036f2e91e8054000c809752e21f933175`.
- Diagnostic source SHA256 at inspection:
  `864d2110a16bb3a2591f3fc19aaa896161ed152e48c4193883aa0ad5d46bd74c`.
- The hardware owner reports stable logical and physical expert inputs,
  routes and indices; only rank 1's first gate tile changes, for all eight
  indexed slots. Up remains exact.
- `bfp8_boundary_retain_gu.json` passes 128 steps; slice retention in
  `bfp8_boundary_retain_slices_v2.json` still fails at step 2. Original-class
  K88 and 40 ms inter-replay host-delay controls fail. Frozen original experts
  pass 128 replays with exact physical BF16 operands, including stable NaNs.
- These are existing hardware-owner observations, not experiments run by this
  audit. The isolated frozen pass does not reproduce the whole-layer allocator
  or preceding-program state.

Paths below are relative to the repository root; model paths abbreviate
`models/autoports/google_gemma_4_26b_a4b_it/` as `model/`.

## Concrete source findings

### 1. GU remains owned through both slices, GELU/mul, down and final mixing

`model/tt/optimized_decoder.py:225-241` assigns GU to a local variable, replaces
that local with its reshape, produces gate/up slices, then runs hidden, down,
permute and mix before returning. There is no `del gu`, `deallocate`, rebinding
to another allocation, or helper-scope exit between GU production and its two
slice consumers. The shape-only reshape preserves the underlying allocation;
the prior stack report verified this tensor-view ownership path.

The slices are materialized copies here. Their width changes from 384 to 192,
so the no-op branch at `slice/slice.cpp:181-183` does not apply. The TILE path
calls `prim::slice` at lines 372-384, whose
`slice_device_operation.cpp:256-265` allocates a new device tensor when no
preallocated output is supplied. The final view only changes metadata of that
new output. Thus retaining gate/up does not retain the raw GU allocation.

Consequently, a change that merely retains raw GU until after `hidden` would
not extend its present production lifetime. Adding such retention would not be
a source-justified fix.

### 2. The diagnostic retains previous GU through the next attention prefix

`model/tests/diagnose_attention_ccl_boundaries.py:88-99` stores selected expert
handles in `BoundaryExperts.boundaries`, independently of the decoder's own
dictionary. `BoundaryDecoder._forward` clears the decoder dictionary at entry
(line 228), but `BoundaryExperts._chunk` clears its dictionary only when the
next expert call starts (line 99).

For GU-only retention, the sequence is:

1. Forward A retains GU in the expert dictionary through shared MLP, grouped
   MoE reduction, fused tail and return.
2. Forward B clears the decoder dictionary, while the expert dictionary still
   owns A's GU during attention, post-attention norm, residual and routing.
3. B enters `_chunk`, clears the expert dictionary, then allocates its selected
   weights, GU, slices and remaining expert tensors.

The second warm forward and capture therefore see a different allocation
history from the original class. After capture, the retained capture GU also
remains owned, although replay itself does not rerun the Python dictionaries.
Passing with retained GU cannot distinguish a suffix overwrite, prefix
placement change, different trace addresses or prior-program state by itself.

### 3. Intended attention CCL payload writes target DRAM

`model/tt/multichip_decoder.py:919-924` casts the attention result and copies it
to DRAM before allreduce. `models/demos/gemma4/config.py:96-127` gives both RS
and AG DRAM output memory. No intermediate-memory override is supplied.

`reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.cpp:213-244`
chooses the input tensor's memory config for the Linear intermediate, and
doubles its leading dimension for the two directions. Here this is BF8 DRAM
with physical shape `[2,1,32,2816]`, 176 tiles, 191488 bytes. RS output is BF8
DRAM `[1,1,32,704]` physically; AG output is BF8 DRAM `[1,1,32,2816]`.
The wrapper drops its staging handle on return
(`reduce_scatter_minimal_async.cpp:114-119`). That release can permit later
**DRAM** reuse. It does not permit a correct DRAM NoC write to become an L1 write.

The allocator uses separate managers for DRAM and L1
(`tt_metal/impl/allocator/allocator.cpp:176-205,240-244`). Numerical base-address
equality across memory types is not physical aliasing. A late collective write
would need a matching target memory type, target core/bank, byte interval and
overlapping execution lifetime to implicate GU or a slice.

CCL kernels do use L1 CBs, packet headers, persistent global semaphores and mux
resources. Those are distinct from the DRAM payload destinations. A claim
about them requires the specific resource and address; tensor addresses alone
cannot prove or refute corruption of a producer's local CB.

### 4. Neither generic async launch nor the slice omits the obvious drain

Trace command assembly sets each program's GO wait count to the cumulative
completed-worker count (`tt_metal/impl/program/dispatch.cpp:3197-3204`;
`tt_metal/distributed/fd_mesh_command_queue.cpp:1583-1616`). Ordinary programs
on this command queue are not intentionally launched as an unordered set just
because a Python function is named `async`. A transport transaction surviving
worker completion would be a separate kernel/fabric completion defect.

The selected tile-slice reader reads one complete tile, read-barriers, and
pushes it (`slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp:37-52`).
The writer flushes before popping each tile and ends with a write barrier
(`writer_unary_interleaved_start_id.cpp:43-50`). No missing caller-level
retention or obvious missing final slice write barrier was found.

The Linear RS writer ends with write and atomic barriers, disconnect/termination
handling, then another pair of barriers
(`line_reduce_scatter_minimal_async_writer.cpp:428-447`). The AG default writer
similarly barriers before connection close and again afterwards
(`minimal_default_writer.cpp:678-700`). These source facts demote a simple
missing-final-barrier story; they do not prove correct delivery of every fabric
packet or exclude corruption of the barriers' own state.

### 5. The corruption pattern groups by sparse producer, not slice destination

GU has 96 BF16 tiles: eight slots times 12 N tiles. The first gate tile for
slot `s` is GU page `12*s`, produced by sparse worker N=0, logical core (0,0).
The corresponding gate slice page is `6*s`. Tile slice distributes its 48
output pages across 48 one-tile workers on this available grid
(`slice_program_factory_tile.cpp:28-34,99-129,164-181`). There is no special
first-column data movement; source page selection is the same dimensional
stride calculation for all workers.

The separate indexed-expert audit confirms that (0,0) is also the input-A
multicast sender. Its CBs, sender/receiver state and inherited hardware state
deserve correlation with the address trace. A single corrupt producer output
tile reused across expert slots is structurally compatible with the symptom.
An external write hypothesis must explain all eight distinct destination
pages, rather than only a single L1 bank address.

## Smallest useful next experiment

Keep the original output-only runner as the acceptance reproducer. The hardware
owner is running a separate Watcher control. Do not change production code or
add another broad retention variant until the following metadata comparison is
available.

### A. Capture integer-only addresses without retaining extra tensors

In the diagnostic's existing `retain` callbacks, record metadata **before**
the retention-selection condition, and return the original tensor immediately.
Store only Python integers, strings, lists and dictionaries:

- forward label: warm 1, warm 2 or capture; boundary name; call ordinal;
- `int(value.buffer_address())`, shape, padded shape, dtype and memory config;
- raw GU and immediately reshaped GU addresses, to verify their alias;
- at `_forward` entry and immediately before `_chunk` clears its dictionary,
  the names and scalar addresses of previous retained expert handles.

Use the existing attention, router and expert boundary set initially. It
already covers attention WO/cast/DRAM/RS/AG, promoted/sharded normalization,
expert inputs, route/index tensors, selected weights, raw GU, gate/up, hidden,
down, permuted down and mixed output. A copied shared-MLP implementation is not
needed for the first pass. If the first pass implicates a suffix allocation,
then extend scalar logging to that specific suffix.

`Tensor.buffer_address()` is a host metadata accessor
(`ttnn/cpp/ttnn-nanobind/pytensor.cpp:1380-1389`). Do not call `to_torch`, reshape,
clone, synchronize, or hold temporary device-tensor lists in this logger.
Host metadata logging during capture can alter enqueue timing; require the
no-retention instrumented graph still to reproduce before interpreting it.

For GU and gate/up, obtain physical page mappings while each tensor is live
using `ttnn._ttnn.reports.get_buffer_pages(mesh)`, filtering by buffer address
and memory type and immediately serializing its scalar records. This function
walks allocator metadata and computes addresses; it does not read device memory
(`ttnn/core/reports.cpp:104-168`). Record `core_x`, `core_y`, `page_index`,
`page_address`, `page_size` and `buffer_type`. The table's device ID represents
the mesh allocator entry, so do not claim it independently measures each rank.

**Instrumentation caveat:** the interleaved branch increments `bank_id` before
placing it into `BufferPageInfo` (`reports.cpp:147-162`), while `core` and
`page_address` were computed from the previous ID. Its reported `bank_id` is
therefore one ahead modulo the bank count in this source. Use the recorded core
coordinates and page address, or correct that field explicitly. Do not patch
this unrelated reporting issue as part of the runtime investigation.

Compare no-GU-retention versus GU retention, with identical warm/capture
sequencing. Focus on GU pages `0,12,24,36,48,60,72,84` and gate pages
`0,6,12,18,24,30,36,42`. Compare byte intervals within the same memory type and
core/bank; total tensor bytes are not an interleaved per-core extent.

### B. Separate the two retention effects, only if A still reproduces

In a diagnostic-only variant, clear the expert's previous retained dictionary
at `_forward` entry as well as the decoder dictionary, then retain the current
GU through the suffix exactly as before. Record the actual addresses. This
removes previous-GU ownership from the next prefix while preserving current-GU
ownership after the expert call.

- If this variant fails and the existing retain-GU variant passes, the earlier
  pass depends on previous-forward prefix placement/state; it does not prove a
  suffix use-after-free.
- If both pass with different prefix placement, suffix retention remains a
  candidate, but a producer/state effect is still not excluded.
- If scalar addresses differ in an implicated range, that is a candidate alias
  map, not evidence of an illegal write. Correlate it with the first changed
  physical page and the actual kernel destination.

If tensor ranges cannot explain the pattern, inspect CB/NoC state on sparse
producer (0,0), separately from profiling. Program-local CBs grow from the L1
base and tensor buffers normally allocate downward; ordinary CB/tensor overlap
is checked against the lowest live tensor allocation
(`tt_metal/impl/buffers/buffer.cpp:426`;
`tt_metal/impl/program/program.cpp:1823-1829`). An out-of-bounds kernel write can
still bypass that host-level check. The exact CB interval and kernel writes
are required before attributing a failure to this mechanism.

## Status

Unresolved. This audit refutes premature GU release within the expert call,
slice-as-view ownership, direct intended DRAM-payload/L1 aliasing, and ordinary
unordered trace-program launch as explanations. It identifies the retained
previous-forward dictionary as an experimental confound and supplies a
host-metadata-only overlap test. No speculative implementation fix, hardware
result, build result or performance claim is asserted. This report is docs-only;
whitespace validation is the applicable repository check.

## Follow-up: N2 address capture and physical snapshot audit

The hardware owner supplied `bfp8_boundary_n2_slices_addresses.json` and its
PT snapshots. Runtime remains `8b59370c`; diagnostic SHA256 is
`dc22ebba887ee93fa2279d7e5c3481de4e1d5ea5a60d319101c7e2309548ccd8`.
This control uses N2/K44 on 6x1, retaining gate/up/hidden and the existing outer
boundaries, with scalar address logging and physical reads. It fails at step 0,
position 4096: rank 2's logical gate has 254 changed elements, maximum delta
0.328125, confined to the first 64 columns. Logical up is exact. The changed
region doubles with sparse sender worker (0,0)'s N ownership. This is stronger
producer-localization evidence than the original first-32-column observation.

### Address findings

CPU analysis examined all 221 scalar records. Relevant addresses are:

| Boundary | Warm 1 | Warm 2 | Capture |
| --- | --- | --- | --- |
| Attention WO FP32 L1 | `0x172c80` | `0x17c000` | `0x1727c0` |
| Expert input BF16 L1 | `0x177480` | `0x176800` | `0x177480` |
| Raw/reshaped GU BF16 L1 | `0x174480` | `0x171800` | `0x174480` |
| Gate BF16 L1 | `0x171480` | `0x171000` | `0x171fc0` |
| Up BF16 L1 | `0x170c80` | `0x170800` | `0x1717c0` |
| Hidden BF16 L1 | `0x170480` | `0x170000` | `0x170fc0` |

Raw GU and reshaped GU have the same base in every phase; gate/up have distinct
bases. Neither captured GU nor either slice shares a base with a logged current
attention/router tensor. Gate's one BF16 page per used bank occupies
`[0x171fc0,0x1727c0)`, ending exactly at the retained WO base. Up and hidden
occupy the adjacent lower 2048-byte intervals. Nothing in those intervals
establishes an overlap. The scalar records omit actual bank/core mappings,
internal CB allocations and unlogged temporary tensors, so this is not a
complete physical memory-safety proof.

The only captured same-base reuse between separately produced logged tensors
is tiled expert mix weights followed by shared input, both at `0x176c80`.
That reuse occurs after the expert call returns and its mix weights are dead;
it is expected. The other repeated bases are known aliases: indices, expert
input/call argument, raw/reshaped GU, and expert output/routed local output.
RS and AG remain DRAM at `0x5aa680` and `0x5ab340`; their numerical addresses
also differ from GU and slices.

The previous-dictionary records directly confirm that warm-2 gate/up/hidden
remain live at capture's expert entry even though the decoder dictionary was
cleared earlier. They are released at that entry and replaced by the new
captured slices. This measures the earlier source-predicted retention effect;
it does not identify an illegal write.

### Physical-input correction and repeated lane pattern

Unlike the earlier N1 hidden-retention case, this N2 run does **not** have a
bit-identical physical expert input. CPU inspection of the saved PT tensors
finds exactly 1822 differing padded elements on every rank: 1596 change from
positive zero to positive infinity, and 226 from positive infinity to positive
zero. Their snapshot bit pairs are `0x00000000 -> 0x7f800000` and the reverse.
All occur outside logical row 0. These are not NaN-sign or payload differences.
The JSON's zero maximum finite-pair delta excludes these finite/nonfinite
transitions and must not be read as proof that physical operands are stable.

The 254 changed logical gate elements have a second consistent structure:
columns modulo 16 are `{0,1,4,7,11,12,13,15}`, repeated across all four 16-column
faces of columns 0:64 and all eight experts. Slots 0 through 6 each have 32 changed
values; slot 7 has 30 because columns 0 and 7 are equal in the stored BF16
results. Absolute deltas vary by expert and column. This suggests checking
compute/pack lane state on the producer in addition to its A-read protocol;
an A-input defect has not yet been proven. The repeated mask is empirical,
not a diagnosis of a particular register or SFPU operation.

### Revised next control

The address comparison demotes additional broad retention A/B runs. The
hardware owner is preparing the coordinator's narrower control: copy only
the original N1/K44 expert activation tensor to DRAM immediately before the
original expert method, preserving dtype, full physical TILE contents,
weights, indices and compute/program configurations. Compare the original
output-only replay gate, and record source/destination addresses. A pass would
localize sensitivity to activation placement/read path or the inserted copy's
ordering; it would not alone prove a lifetime fix. A failure leaves the
producer-local compute/pack and sender protocol investigation active.

The separate sender/kernel audit owns the smallest subsequent protocol or
register-state experiment. Kernel changes should wait for that diagnosis.
The N2 padding variation must remain separate from the older N1 frozen and
whole-layer controls with bit-stable physical inputs. No hardware or runtime
modification was performed by this follow-up; Python standard-library JSON
analysis and CPU-only `torch.load(..., map_location="cpu", weights_only=True)`
were used to examine the existing artifacts.
