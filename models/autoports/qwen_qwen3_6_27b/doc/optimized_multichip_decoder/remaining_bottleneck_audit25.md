# Remaining decoder experiments after timing24

CPU-only independent audit, 2026-11-13. No TTNN/torch imports, compilation,
device access, source changes, or new measurements. Only this report is new.
Paths below are relative to this model unless they start with `ttnn/`.

## What the evidence supports

`gdn22_timing24.csv` and `attention22_timing24.csv` are candidate measurements,
not final-default evidence. GDN regression B32/S513 prefill remains
43.588–43.686 ms across its five variants; attention remains35.254–35.444 ms
across three. The recent recurrence/hidden-norm/packed-QK changes primarily
target decode. The corresponding JSON policies confirm that the first temporal
chunk has16384 logical rows: GDN joint projection uses four4096-row pieces;
attention projection inherits512-row pieces. The final S1 tail is separate.
Changing temporal alignment or blindly increasing those capacities would
repeat `prefill_chunks7`/`prefill_alignment8`, which already contain adapted
geometry and coherent whole-layer evidence.

The latest available fusions20 profiles are B32/S33, **not S513**. In the fused
attention prefill table, op3226 is a207us material reshape, op3228 a40us head
creation, and the preceding QKV/gate slices cost18+24us. Later head concatenation
and compacting reshape cost32+99us. GDN has96us input flatten,100us head-output
reshape and134us final unflatten; attention's final unflatten is also134us.
These are observed material operations, not free metadata. They cannot be
multiplied by513/33 to invent a long-prefill saving. Trace-prefill23's negligible
long attention improvement likewise argues against blaming all35ms on host
dispatch.

Decode has approximately125us recurrence and64.5us of two AG/two RS in the
fusions20 GDN table. The four decode matrices already reach roughly72–87% of
their modeled DRAM ceilings. The280/195us floors quoted in earlier audits are
**decode** bounds, not prefill bounds; prefill needs its own operation-count,
activation-traffic and collective-traffic accounting. None of the proposals
below establishes70% whole-layer roofline, and none justifies stopping early.

## 1. Eliminate projection-piece copies while preserving winning arithmetic

Highest priority for the long-prefill investigation. Keep the selected512-row
attention and4096-row joint-GDN matmul partitions, their weights, math configs,
subblocks and output rounding. Change the memory traversal around them.

Source: `tt/optimized_decoder.py:499–514` slices each flat activation piece and
concatenates results; `tt/packed_correction_candidate.py:267–305` additionally
runs the correction combine for each piece and concatenates its BF16 results.
Neither source provides an input-row/output-row offset interface. At B32/S512
this is32 attention projection calls or four joint-GDN calls. This proposal
does not change their count or reread-weight schedule initially.

Concrete implementation route: a bounded native-program adapter retaining the
selected2D compute kernel, with the original whole-input TensorAccessor and
`row_start/32 * K_tiles` added to the reader tile origin. Write each result into
the correct final-output tile interval. Existing native mechanics are in
`ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp:82,232–290`
and `.../writer_bmm_tile_layout.cpp:12–18,44–75`; the actual selected program
factory's padding-capable writer must be adapted, not replaced blindly with
this simpler illustrative writer. For GDN, keep the existing intermediate
FP32 joint output and exact combine math first, but give the combine writer a
global BF16 row offset to remove the final concat. Then adapt the input reader
to remove the activation slices. The output has a single owner per logical
tile; no two partial programs may clear each other's padded output.

`ttnn.linear(optional_output_tensor=...)` alone is insufficient: native
`matmul_device_operation.cpp:229–240` requires matching output volume, so a
full-output tensor is not a legal drop-in for a smaller piece. A TTNN slice is
not a demonstrated zero-copy subview either. Do not count replacing concat
with slice_write as removing its traffic without measuring both.

Discriminator: compare complete selected projection+combine boundaries for
rows1056,4096,16384 and awkward logical lengths, then whole B32/S513. Report
slice/concat bytes and timings from a fresh long-prefill profile before
attributing the43/35ms total. Preserve projection bits if compute is unchanged;
retain unchanged whole-layer contribution/state gates regardless. Verify
piece edges, final padding, rank/user-distinct inputs, output-address rebound,
queued reuse and changed-input traces. This is a new traversal family, not a
repeat of the376 geometry configurations. No speed claim follows merely from
removing copies.

## 2. Direct compact-packed prefill QKV to final head tensors

Bound the first candidate to an arithmetic-free producer/consumer change;
leave current native Q/K norm, batched partial RoPE, cache fill and SDPA intact.
Read `[1,B*S,3584]` packed QKVG directly and emit Q`[B,6,S,256]`,
K/V`[B,1,S,256]`, and the existing compact gate. This removes both packed
section slices and the intermediate padded `[B,S,2048]` material reshape before
head creation (`tt/multichip_decoder.py:701–707`). It differs from the active
**decode output gating** helper and from merely enabling packed decode QKRoPE.

Use the existing native head-shuffle readers/writers as the basis:
`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/{reader,writer}_tm_tile_layout_nlp_create_qkv_heads_runtime_rows.cpp`.
They already map disjoint `(batch,head,tile-row)` output owners and handle
multi-tile256-wide heads. Their current input model assumes each user's
sequence is independently tile-padded; `runtime_rows=True` changes cache
specialization, **not** that layout. A direct call does not solve compact
nonaligned-user input. Adapt the reader to compute logical source row
`user*S + token`, packed stride112tiles, and Q/K/V offsets0/48/56tiles.
For S divisible by32, transfer full tiles. For nonaligned S, assemble a
destination tile from aligned source-face row segments and explicitly zero
nonlogical rows. Gate starts at tile64 and remains compact; optionally leave
its old slice in the first version to isolate QKV movement. Every64-byte DRAM
transaction must respect source/destination alignment and valid overfetch.

Coherent measurement boundary: packed projection output through the unchanged
norm/RoPE/cache-ready outputs, then whole prefill. No restore to the intermediate
`[B,S,2048]` layout. The S33 table's207us reshape+40us head creation+section
slices defines an observed target boundary, not an achievable saving. The
S512 first chunk is tile-aligned, so whether this becomes material at long
prefill must be established separately. Test S1/3/31/33/511/512/513, all user
boundaries, poisoned padding, exact Q/K/V/gate and cache comparisons, mixed
lengths and resumed starts. Do not alter public logical alignment requirements.

## 3. Recurrent live-row Q/K preparation, isolated from prediction changes

This is an earlier **untried** hypothesis in `recurrent_epilogue_audit.md:172`,
not the already measured prediction-register, column, native prepared-state,
head-major readout or BF16-pack families. It is smaller and lower-risk than a
new persistent-state layout, although its total saving may be modest.

`tt/optimized_decoder.py:5379–5420` squares four complete FP32 tiles, folds
them in order, reduces across columns, adds epsilon/rsqrts, normalizes four
tiles, separately scales Q and packs before transpose/column expansion. Only
input row0 reaches the prepared columns. Preserve the existing REDUCE_ROW
helper and all packs/transposes initially; restrict only pointwise square,
ordered fold, scalar epsilon/rsqrt, normalization and separate Q scale to the
consumed lanes. Use the exact existing accurate primitives, epsilon bits
0x358637bd and Q-scale bits0x3db504f3. `VectorMode::R` is not by itself a proof
of row0-only execution. Preserve the native lane map and reduction input bits.

Current B32 ownership prepares164 Q/K pairs per rank, including split triples;
the critical four-head owners prepare two groups. A logical32-fold dead-row
reduction is **not** a32-fold recurrence speedup: state traffic, transpose,
packs, reduction and update/readout remain. Existing CB descriptors already
have two-head capacity despite stale comments (`multichip_decoder.py:1340–1363`,
`debug_recurrent/AUTODEBUG.md:64`), so do not propose double buffering as new.

First expose prepared Q/K for bit comparison before recurrence: zero vectors,
epsilon-scale vectors, extreme finite magnitudes and poisoned dead rows.
Then compare complete recurrence output and BF16/FP32 persistence over all32
batches, split triples, one-core wraps, changed input/state traces and512-step
stress. Use the best cumulative-exp-scalar control; only compose with kept
prediction/column changes after the isolated result. A failure to preserve
live-row reduction bits warrants lane/rounding adaptation, not relaxing PCC.

## Scope and exclusions

No additional standalone AGMM, MMRS placement, reader-count, CCL-dtype or
head-major epilogue rerun is recommended without a new structural hypothesis;
their adapted complete-boundary evidence is already retained. DiT/minimal
distributed norm, joint convolution, new attention gate, BF16 joint output,
prediction composition and precise native recurrence are active elsewhere.
The remaining AG/RS operations implement the selected intra-layer TP contract;
deleting them independently is invalid. The three experiments above retain
the1280-wide rank-local inter-layer residual and add no inter-layer collective.

This audit supplies source-based candidates and discriminating tests, not a
stage-review verdict, hardware certification, final policy or stop-rule proof.
