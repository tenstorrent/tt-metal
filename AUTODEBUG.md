# AutoDebug: `ttnn.gather` on ROW_MAJOR BLOCK/WIDTH-sharded tensors with an unaligned shard row (issue #62153)

Inspection-only analysis on `main` at `8ac40e3f904`. No hardware runs were done. Claims that need hardware to prove are marked as such.

## Summary

**Finding 1 (root cause of all five repro cases).** In a ROW_MAJOR BLOCK_SHARDED or WIDTH_SHARDED buffer, one page is one shard row of `shard_W * element_size` bytes (the "row size"). The device stores those pages at a pitch of `round_up(row size, 16 B)` (the "aligned page size"). The RM gather kernels get the unaligned row size from [`per_shard_page_size_bytes`](ttnn/cpp/ttnn/operations/data_movement/common/common.cpp#L898-L909) and use it for two jobs:

- as the page pitch inside a shard, so row `h` is addressed at `h * row size` instead of `h * aligned page size`;
- as the NOC chunk size, so the chunks of one row are packed into the circular buffer (CB) at multiples of the row size, and half of them start 8 B away from a 16 B boundary.

When the row size is a multiple of 16 B, the two values are equal and nothing goes wrong. When the row size is not a multiple of 16 B, the kernels read or write the wrong SRAM bytes:

- **Sharded input:** wrong data goes into the input CB, so the output values are wrong.
- **Sharded output:** the output rows land at the wrong addresses, so the output values are wrong.
- **Sharded index:** wrong index values, including the 8 undefined padding bytes per row, are read and then used without a bounds check as load addresses. My guess is that a load from an address outside SRAM stalls the data-movement core, which is the hang.

**Finding 2 (the smallest place to fix Finding 1).** The gather wrapper and the device operation have no check for an unaligned BLOCK/WIDTH shard row. The two sibling RM ops that use the same NOC helpers, slice and fold, move such tensors to interleaved SRAM before their native kernels run. The gather wrapper checks only whether the logical W is a multiple of 32, and only for the input and index tensors, never for the output. All repro widths (96, 1984) are multiples of 32, so no workaround path runs.

**Finding 3 (independent bug in `RmSingleRowMultiCore`, also hit by case 5).** The multi-core factory gives core `c` a column range that starts at `w_start = 31 * c` for W_index = 1984 on 64 cores. The index read and the output write start at byte offset `w_start * element_size`, which is not 16 B aligned for most cores. The local CB address is always 16 B aligned. The Wormhole NOC needs matching low 4 address bits on both ends, so most cores read and write shifted bytes. The unaligned per-core offsets occur on interleaved tensors as well. A committed test comment already records that this factory "returns wrong elements".

**Proposed minimal fix:**

1. In the gather wrapper, move any unaligned BLOCK/WIDTH-sharded RM input, index, or output to interleaved SRAM around the native call, as slice and fold do.
2. In the device operation, make the same condition a hard error, as slice's factory does.
3. In `RmSingleRowMultiCore`, split W_index in steps that keep every per-core byte offset 16 B aligned (32 B for DRAM).

After items 1 and 2, `per_shard_page_size_bytes()` equals `buffer->aligned_page_size()` for every tensor these factories see, which is what the Metal 2.0 port needs.

### Case matrix

The index tensor is always uint32. "Row size" is `shard_W * element_size` of the sharded tensor. The predicted float32 results are a direct test of Finding 1: for cases 1, 2 and 4 the float32 row size is 48 B, which is aligned.

| Case (test id) | Factory | Sharded tensor | Row size bf16 / f32 | Finding 1 | Finding 3 | Expected bf16 | Expected f32 |
|---|---|---|---|---|---|---|---|
| 1 `RM_interleaved_to_width_subaligned_shard_row` | SingleCore | output, width, (32,12) | 24 / 48 B | bf16 only | no | wrong values | pass |
| 2 `RM_width_subaligned_shard_row_input_to_interleaved` | SingleCore | input, width, (32,12) | 24 / 48 B | bf16 only | no | wrong values | pass |
| 3 `RM_width_subaligned_shard_row_index_to_interleaved` | SingleCore | index, width, (32,6) | 24 / 24 B (uint32) | both | no | hang | hang |
| 4 `RM_interleaved_to_block_subaligned_shard_row` | SingleCore | output, block 2x8, (16,12) | 24 / 48 B | bf16 only | no | wrong values | pass |
| 5 `RM_multicore_interleaved_to_width_subaligned_shard_row` | MultiCore | output, width, (4,62) | 124 / 248 B | both | both | wrong values | wrong values |

## Observations and interpretations

Direct observations, from the issue (not reproduced here):

- Wormhole n150, all tensors in SRAM, ROW_MAJOR, `dim=-1`.
- Sharded input or sharded output with an unaligned BLOCK/WIDTH shard row gives wrong values.
- A sharded index with an unaligned shard row hangs the device.
- The affected factories are `RmSingleRowSingleCore` and `RmSingleRowMultiCore`.

Interpretations (this report):

- Findings 1 to 3 above.
- The issue does not say which dtypes failed. The f32 column of the case matrix is a prediction.
- The issue does not include an all-interleaved control for case 5. Finding 3 predicts that such a control also fails.

## How the device stores an RM BLOCK/WIDTH-sharded buffer

- **Page size.** For BLOCK or WIDTH sharding, the RM page width is the shard width ([page_config.cpp:117-124](tt_metal/impl/tensor/spec/layout/page_config.cpp#L117-L124)), so one page is one shard row of `shard_W * E` bytes.
- **Page pitch.** Pages are stored at a pitch of `Buffer::aligned_page_size() = align(page_size, alignment)` ([buffer.cpp:844](tt_metal/impl/buffers/buffer.cpp#L844)), where the alignment is 16 B for Wormhole SRAM. For a 24 B row the pitch is 32 B, so each row has 8 bytes of padding.
- **Padding contents.** The host write path sends `pages * aligned_page_size` bytes but copies only `page_size` bytes of tensor data into each slot ([dispatch.cpp:622-685](tt_metal/impl/buffers/dispatch.cpp#L622-L685)). The command-queue space is reserved with `reserve_space<uint8_t*, true>` ([dispatch.cpp:636](tt_metal/impl/buffers/dispatch.cpp#L636)), and data reservations are never zeroed ([device_command.hpp:249-260](tt_metal/impl/dispatch/device_command.hpp#L249-L260)). The 8 padding bytes therefore hold whatever bytes were in the issue queue at that position.
- **Kernel-side page geometry.** The kernel side describes the buffer in pages. For case 2 the tensor is (32, 8) pages and a shard is (32, 1) pages. `noc_async_read_sharded` / `noc_async_write_sharded` compute `pages_per_row = 8` from these values ([common.hpp:494-501](ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp#L494-L501)). They then move one chunk of `tensor.get_aligned_page_size()` bytes per page and advance the local SRAM address by the same amount ([common.hpp:502-522](ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp#L502-L522), and [common.hpp:440-460](ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp#L440-L460) for writes).
- **Wormhole NOC alignment rule.** "The alignment of source and destination start addresses (i.e., bits [3:0]) must be identical. If the alignment is not identical, the one from destination address is assumed." ([noc.h:57-60](tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc.h#L57-L60)). A mismatch therefore moves the wrong bytes and raises no error. Watcher reports it as `DebugSanitizeNocAlignment` ([sanitize.h:697-707](tt_metal/hw/inc/internal/debug/sanitize.h#L697-L707)). Concat states the same rule: a NOC transfer that starts unaligned "copies the wrong bytes rather than failing" ([concat_device_operation.cpp:239-241](ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp#L239-L241)).

## Finding 1: the RM gather kernels use the shard row size as the page pitch

### Where the two values diverge

The earliest point where the two calculations diverge is [common.cpp:898-909](ttnn/cpp/ttnn/operations/data_movement/common/common.cpp#L898-L909):

```cpp
if (mc.is_sharded() && (BLOCK_SHARDED || WIDTH_SHARDED)) {
    return spec.shape[1] * t.element_size();              // unaligned row size
}
if (mc.is_sharded()) {
    return t.buffer()->aligned_page_size();               // HEIGHT: aligned
}
```

The HEIGHT branch was changed to the aligned size by #47299 (`6b9f0a992ad`). The description of #47299 reports that gather hung on HEIGHT-sharded tensors with unaligned rows. It names two causes: a `pages_per_row` error in the NOC helpers, and this pitch divergence ("`per_shard_page_size_bytes` returns an unpadded row size that diverges from L1's 16-aligned stride"). #47299 fixed the HEIGHT branch only. The BLOCK/WIDTH branch still returns the unaligned size.

Both RM factories pass the unaligned row size to the kernels as a runtime argument:

- `RmSingleRowSingleCore`: [gather_program_factory.cpp:475-480](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L475-L480) and [gather_program_factory.cpp:482-506](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L482-L506).
- `RmSingleRowMultiCore`: [gather_program_factory.cpp:627-660](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L627-L660).

The kernels pass the unaligned row size as the third `TensorAccessor` argument. That argument replaces the aligned page size that the host already put in the compile-time arguments:

- [gather_reader_rm_single_row_single_core.cpp:40-42](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_rm_single_row_single_core.cpp#L40-L42)
- [gather_writer_rm_single_row_single_core.cpp:40-43](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_rm_single_row_single_core.cpp#L40-L43)
- [gather_reader_rm_single_row_multi_core.cpp:46-47](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_rm_single_row_multi_core.cpp#L46-L47)
- [gather_writer_rm_single_row_multi_core.cpp:45-48](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_rm_single_row_multi_core.cpp#L45-L48)

The accessor stores that value ([tensor_accessor.h:95-100](tt_metal/hw/inc/api/tensor/tensor_accessor.h#L95-L100)) and uses it as the page pitch: `bank_base + bank_page_offset * aligned_page_size + offset` ([tensor_accessor.h:335-343](tt_metal/hw/inc/api/tensor/tensor_accessor.h#L335-L343)).

### What goes wrong when the row size is not a multiple of 16 B

**(a) Pitch error.** Row `h` of a shard is addressed at `h * row size`, but the data is at `h * aligned page size`. Only row 0 is correct. All accesses stay inside the per-core allocation, because the kernel's pitch is smaller than the real pitch, so nothing faults.

| Case (bf16) | Row size / aligned page size | Kernel address for h = 0, 1, 2 | True address for h = 0, 1, 2 |
|---|---|---|---|
| 1, 2, 3, 4 | 24 / 32 B | 0, 24, 48 | 0, 32, 64 |
| 5 (core c writes shard c/2 at offset 62·(c%2)) | 124 / 128 B | 0, 124, 248 (+62 for odd c) | 0, 128, 256 (+62 for odd c) |

**(b) Alignment error.** The split loop places chunk `k` of a row at `CB base + k * row size` ([common.hpp:509-521](ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp#L509-L521)). The CB base is 32 B aligned ([program.cpp:2401](tt_metal/impl/program/program.cpp#L2401)). For a 24 B row the local addresses are therefore 0 and 8 mod 16 in turn. The remote address is `24h` mod 16 at the kernel's address, or 0 mod 16 at the true address. Half the chunks of every row have mismatched low bits, so the NOC moves the wrong bytes. The alignment error stays even if only the pitch is corrected. Any kernel-level fix must also change how the chunks are placed in the CB.

### Sharded input (case 2): wrong values

The writer kernel reads input row `h` into the input CB through the sharded helper ([gather_writer_rm_single_row_single_core.cpp:52-60](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_rm_single_row_single_core.cpp#L52-L60)). Because of errors (a) and (b), the CB receives a mix of other rows' data and padding bytes. The index values are correct, because the index tensor is interleaved, so every load stays inside the 192 B input CB. The output values are wrong, and nothing else goes wrong.

### Sharded output (cases 1, 4, 5): wrong values

The gather loop fills the output CB correctly. The writer then writes row `h` to `h * row size` instead of `h * aligned page size` ([gather_writer_rm_single_row_single_core.cpp:62-70](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_rm_single_row_single_core.cpp#L62-L70)). In case 1 the writes cover bytes [0, 768) of each 1024 B shard, so true rows 24 to 31 are never written, and the host reads stale SRAM for them. The rows below 24 hold misplaced data. Error (b) also shifts half of the chunks. The host reads the buffer at the true pitch and gets wrong values.

### Sharded index (case 3): hang

**Step 1: wrong index values.** The reader reads index row `h` with 16 chunks of 24 B from 16 shards at shard offset `24h` ([gather_reader_rm_single_row_single_core.cpp:52-60](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_rm_single_row_single_core.cpp#L52-L60)). By the pitch error alone, for every `h` with `h % 4 != 0` (24 of the 32 active cores), 2 of every 6 index words come from the 8 padding bytes of a shard row. As described above, the host write path never sets those padding bytes to a defined value. Error (b) moves more bytes. The exact set of padding words read depends on the hardware behavior described in [noc.h:57-60](tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc.h#L57-L60).

**Step 2: no bounds check.** The reader then calls `get_value_from_stick(input_cb, global_index, 2)` with no bounds check ([gather_reader_rm_single_row_single_core.cpp:71-77](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_rm_single_row_single_core.cpp#L71-L77), [gather_common.hpp:11-15](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_common.hpp#L11-L15)). A padding word `g` becomes a `volatile uint16_t` load at `(input_cb + 2g) mod 2^32`. Wormhole Tensix SRAM covers only `[0, 0x16E000)` ([dev_mem_map.h:32-33](tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h#L32-L33)). Most 32-bit values therefore point into the unmapped gap or into the local memory and register space at `0xFFB00000` and above.

**Step 3, unproven: the hang.** My guess is that one of these loads stalls the data-movement RISC-V core, so the program never completes. The repo has no statement about Wormhole loads to unmapped addresses. Some register reads in that range are documented to block, for example the PC buffer read in [tensix.h:56-58](tt_metal/hw/inc/internal/tt-1xx/wormhole/tensix.h#L56-L58).

Supporting evidence:

- #47299 reports that gather hung on HEIGHT-sharded tensors with unaligned rows, and names the same pitch divergence as one of two causes.
- Only the index-sharded variant feeds misread bytes into an address calculation. This matches the observation that the input-sharded and output-sharded variants of the same tensor sizes return wrong values without hanging.

Checked and ruled out as hang causes for case 3:

- A CB reserve/push/wait/pop count mismatch between reader and writer: each loop pushes and pops once per CB, and both kernels get the same loop count.
- A mismatch between issued NOC transactions and barriers: every read and write is followed by its barrier.
- A store out of range: stores are bounded by `w < W_index` inside the output CB.

## Finding 2: gather has no guard for an unaligned BLOCK/WIDTH shard row

[`needs_rm_irregular_composite`](ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L26-L43) sends work to the TILE workaround path only when the input or index is BLOCK/WIDTH-sharded **and** its logical W is not a multiple of 32. The predicate has two gaps:

- It never checks the shard row size, so an unaligned shard row with W = 96 or W = 1984 passes.
- It never checks the output memory config or `out`.

[`validate_on_program_cache_miss`](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp#L46-L125) has no such check either. The other wrapper steps do not change the repro tensors before the device op:

- The codegen path requires TILE layout.
- For `dim=-1` at rank 4, the transpose and the 4D step do nothing.
- The input `ttnn::slice` is a no-op ([slice.cpp:209-211](ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp#L209-L211)).

The sibling ops show the intended contract:

- **Slice wrapper.** [`has_subaligned_shard_row`](ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp#L49-L60) sends both input and output through an interleaved-SRAM hop ([slice.cpp:62-90](ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp#L62-L90), [slice.cpp:360-385](ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp#L360-L385)). The commit that added it (#55262, `1b0d9d1258a`) describes this exact defect for slice: "otherwise every row after the first is mis-addressed and slice silently returns wrong data". It also says that fold already had the same guard.
- **Slice factory.** [`check_accessor_page_size`](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp#L282-L306) raises a hard error when the per-shard page size differs from the aligned page size. Its comment explains why the Metal 2.0 port needs the two to be equal: the ported kernels take the aligned page size from the tensor binding, not from a runtime argument.
- **Fold.** [fold.cpp:512-532](ttnn/cpp/ttnn/operations/data_movement/fold/fold.cpp#L512-L532) moves an unaligned BLOCK/WIDTH-sharded RM tensor to interleaved SRAM.
- **Slice test for the output side.** `test_slice_rm_bw_sharded_subaligned_shard_row_out` uses the same 96-wide tensor and the same 8-core, 24 B shard row as gather case 1. It relies on `to_memory_config` to produce the unaligned sharded output, so that conversion path is already covered.

The missing wrapper guard is the boundary to fix. For the single-core cases, the passing configurations (aligned shard rows, or interleaved tensors) differ from the failing ones only in whether the shard row is a multiple of 16 B. A wrapper guard can test that property before any kernel runs. Slice and fold already fix the same defect with a wrapper guard.

## Finding 3: `RmSingleRowMultiCore` gives cores byte offsets that are not NOC-aligned

The factory splits W_index into equal column ranges ([gather_program_factory.cpp:550-553](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L550-L553)). The default Wormhole n150 grid is 8x8 = 64 cores, so W_index = 1984 gives 31 columns per core and `w_start = 31c`. The reader reads the index slice at byte offset `w_start * 4` ([gather_reader_rm_single_row_multi_core.cpp:43-61](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_rm_single_row_multi_core.cpp#L43-L61)). The writer writes the output slice at byte offset `w_start * E` ([gather_writer_rm_single_row_multi_core.cpp:42-71](ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_rm_single_row_multi_core.cpp#L42-L71)). The local CB address is always 0 mod 16.

| core c | w_start | index read offset mod 16 | bf16 output offset mod 16 | f32 output offset mod 16 |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 31 | 12 | 14 | 12 |
| 2 | 62 | 8 | 12 | 8 |
| 3 | 93 | 4 | 10 | 4 |
| 4 | 124 | 0 | 8 | 0 |

With bf16 output, 48 of the 64 cores read the index with mismatched low address bits, and 56 of the 64 cores write the output with mismatched low address bits. This does not depend on sharding: for an interleaved tensor the helper issues one transfer at that offset ([common.hpp:491-492](ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp#L491-L492)). Because the destination alignment wins, my guess is that each core gets index values from a few columns to the left (still in range, so no hang) and writes its slice shifted by a few bytes.

Evidence:

- The Wormhole NOC rule quoted above.
- argmax sizes its per-core work to avoid exactly this ([argmax_multi_core_program_factory.cpp:221-225](ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_multi_core_program_factory.cpp#L221-L225)).
- A committed test comment says this factory "returns wrong elements" for a 151936-wide interleaved row, where the per-core offsets are also unaligned. That test skips the value check above 1920 columns ([test_gather_codegen_routing.py:76-82](tests/ttnn/nightly/unit_tests/operations/data_movement/test_gather_codegen_routing.py#L76-L82)).

No value-checking test exercises this factory. The cases labelled "multi-core" in `test_universal_input_tm_gather.py` use an index width of 1024, which is below the 1920 threshold that [`select_program_factory`](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp#L24-L33) applies to W_index. Those cases therefore run `RmSingleRowSingleCore`. See `RM_interleaved_multi_core` ([test_universal_input_tm_gather.py:406-414](tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py#L406-L414)) and `test_gather_rm_multicore_sharded`.

For case 5, Finding 1 (pitch and alignment of the 124 B output shard row) and Finding 3 both apply. Fixing either one alone does not make case 5 pass.

## Proposed minimal fix

### Fix A: wrapper reroute (covers Findings 1 and 2)

Add in [gather.cpp](ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L26-L43) a predicate equal to slice's `has_subaligned_shard_row`, restricted to BLOCK/WIDTH sharding:

```cpp
// A BLOCK/WIDTH-sharded ROW_MAJOR buffer stores one shard row per page at a pitch of
// buffer->aligned_page_size(). The RM gather kernels split a row into shard_W * element_size
// chunks and stride pages by that same value, so the two agree only when the shard row is a
// multiple of the buffer alignment. Same guard as slice's has_subaligned_shard_row and fold.
inline bool has_subaligned_bw_shard_row(const tt::tt_metal::MemoryConfig& mc, uint32_t element_size) {
    if (!mc.is_sharded() || !mc.shard_spec().has_value()) {
        return false;
    }
    const auto layout = mc.memory_layout();
    if (layout != TensorMemoryLayout::BLOCK_SHARDED && layout != TensorMemoryLayout::WIDTH_SHARDED) {
        return false;
    }
    const uint32_t alignment = (mc.buffer_type() == BufferType::DRAM) ? hal::get_dram_alignment()
                                                                       : hal::get_l1_alignment();
    return (mc.shard_spec()->shape[1] * element_size) % alignment != 0;
}
```

In `gather_dispatch`, for ROW_MAJOR, just before the `ttnn::prim::gather` call ([gather.cpp:385-400](ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L385-L400)):

1. **Input and index.** If `padded_input_tensor` or `padded_index_tensor` meets the predicate, replace it with `ttnn::to_memory_config(t, l1_interleaved)`, using the interleaved-SRAM config that `gather_dispatch` already builds for its TILE workaround path. Checking after `pre_gather_transform_tensor` also covers tensors that a transpose or slice step produced.
2. **Output.** Get the effective output spec by calling `GatherDeviceOperation::compute_output_specs` with the same arguments. If that spec meets the predicate, with the output element size, then:
   - call the prim with an interleaved-SRAM output and no `out`;
   - convert the result with `ttnn::to_memory_config(result, spec.memory_config())`, or `ttnn::copy` it into the preallocated `out`, as slice's `finalize_into_preallocated` does.

   The effective spec matters. When the caller asks for a sharded output with no shard spec, `compute_output_specs` copies the index's shard spec in **element** units ([gather_device_operation.cpp:136-153](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp#L136-L153), [transpose_utils.cpp:109-145](ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_utils.cpp#L109-L145)). Index elements are 4 B and output elements may be 2 B. An aligned index shard row of 4 elements (16 B) therefore becomes an unaligned bf16 output row of 8 B.

Fix A uses the same `to_memory_config` conversions that slice and fold already rely on for unaligned shard rows. Fix A does not change the kernels.

### Fix B: device-op backstop (needed by the Metal 2.0 port)

In `validate_on_program_cache_miss`, for ROW_MAJOR, raise a hard error if the input, the index, or the computed output spec meets the predicate. Point the message to `ttnn::gather`, as slice's `check_accessor_page_size` does. After Fix A and Fix B, `per_shard_page_size_bytes()` equals `buffer->aligned_page_size()` for every tensor that reaches `RmSingleRowSingleCore` or `RmSingleRowMultiCore`. The port can then take the page size from the tensor binding and drop the runtime page-size arguments.

### Fix C: aligned work split in `RmSingleRowMultiCore` (covers Finding 3)

In [gather_program_factory.cpp:550-560](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L550-L560), split W_index in units of `granule` columns, where `granule * E` is a multiple of the alignment for both the index read and the output write:

```cpp
const uint32_t alignment = std::max(input_index_tensor_buffer->alignment(), output_tensor_buffer->alignment());
const uint32_t granule = std::lcm(alignment / std::gcd(alignment, input_index_elem_size),
                                  alignment / std::gcd(alignment, output_elem_size));
const uint32_t num_units = tt::div_up(W_index, granule);
// split_work_to_cores(core_grid, num_units, true) -> units_per_core_g1 / g2
// per core: w_per_core = std::min(units * granule, W_index - w_cursor)   // the last core takes the tail
// CB sizes: max_w_per_core = units_per_core_g1 * granule
```

Example: a uint32 index and bf16 output in SRAM give `granule = lcm(4, 8) = 8`. W_index = 1984 is then 248 units, split as 56 cores × 32 columns and 8 cores × 24 columns. Every offset is a multiple of 32 B for the index and 16 B for the output. The kernels need no change, because `w_per_core` and `w_start` are already runtime arguments.

### Optional hardening (not needed for the repro)

Bound the index value in the RM gather loops, for example: if `global_index >= W_input`, write 0. The Python docstring already says that an out-of-range index gives an unspecified value ([gather_nanobind.cpp:27](ttnn/cpp/ttnn/operations/data_movement/gather/gather_nanobind.cpp#L27)), and an unspecified value should not be a hang. With this bound, a corrupt index can no longer hang the device.

### Not recommended as the minimal fix

Changing `per_shard_page_size_bytes` to return the aligned size is not enough on its own. `noc_async_*_sharded` uses the same value as the chunk size, so each chunk would copy the 8 padding bytes into the CB. A kernel-level fix needs a separate pitch and payload in the helper, plus a staging copy to restore 16 B alignment in the CB. A kernel-level fix is more work and more risk than Fix A.

## Other potential issues

These do not explain the repro, or are not proven.

1. **The composite workaround ignores `out`.** The existing TILE workaround path calls `gather_native(..., std::nullopt, ...)` and returns `to_memory_config(rm_out, requested_mc)` ([gather.cpp:316-348](ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L316-L348)). A preallocated `out` is never written, and its memory config is not used. If Fix A reuses this path instead of a new reroute, this gap must be closed.
2. **`per_shard_page_size_bytes` treats sharding layouts differently.** HEIGHT sharding returns the aligned size and BLOCK/WIDTH sharding returns the unaligned size. The header comment documents this ([common.hpp:440-441](ttnn/cpp/ttnn/operations/data_movement/common/common.hpp#L440-L441)). Every caller must guarantee aligned BLOCK/WIDTH shard rows. Slice does this with its guard and its factory check; gather does neither.
3. **Test gaps.**
   - No value-checking test reaches `RmSingleRowMultiCore` (see Finding 3).
   - No committed test has an unaligned BLOCK/WIDTH shard row for gather.
   - The `RM_*_regular_*` cases use 64 B or 256 B rows, which are aligned.
4. **Out-of-date documentation.** [Gather.md:82](ttnn/cpp/ttnn/operations/data_movement/gather/docs/Gather.md#L82) says sharded memory is not supported, and the Python docstring lists TILE layout and interleaved memory only ([gather_nanobind.cpp:43-71](ttnn/cpp/ttnn/operations/data_movement/gather/gather_nanobind.cpp#L43-L71)). Both RM factories and the tests support sharded ROW_MAJOR.
5. **TILE factories not examined.** `SingleRowSingleCore` and `SingleRowMultiCore` were not examined for the same index bounds issue.

## Remaining uncertainty and suggested hardware checks

| Claim | Status | Check that would decide it |
|---|---|---|
| Finding 1 addresses and alignment for cases 1 to 5 | proven from code | Watcher run: expect `DebugSanitizeNocAlignment` in cases 1 to 5 |
| f32 variants of cases 1, 2, 4 pass | prediction | `pytest ...::test_gather_rm -k "subaligned and f32 and not index"` |
| Case 3 hangs because of out-of-range loads from padding index values | unproven final step | DPRINT `global_index` on a core with `h % 4 != 0`, or add the index bound and check whether the hang becomes wrong values |
| An all-interleaved (1,1,4,1984) RM gather returns wrong values (Finding 3) | prediction, consistent with the committed test comment | Add that case with an exact value check |
| Exact bytes that land after a mismatched NOC transfer | uncertain beyond the noc.h sentence | not needed for the fix |
| Case 5 numbers assume the 8x8 default grid | true for default fast dispatch on n150 | a different grid changes the offsets but not the conclusion, unless every offset happens to align |
