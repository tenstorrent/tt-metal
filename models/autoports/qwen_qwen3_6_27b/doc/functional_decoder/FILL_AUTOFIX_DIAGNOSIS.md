# AutoFix discriminator: small row-major INT32 device fill

## Diagnosis

**Confirmed source mismatch:** the interleaved `moreh_full` factory sizes its single-entry fill buffer using the output's **unaligned page size**, but `writer_full.cpp` transfers the accessor's **aligned page size**. For `(1,1)` INT32 on this Blackhole DRAM path, the logical page is 4 bytes and the writer requests 64 bytes. This matches the Watcher failure exactly. Padding the fill to `(1,32)` INT32 makes both sizes 128 bytes and removes this specific buffer-overrun condition. Slicing the completed device tensor to `(1,B)` before reshaping to `(B,)`, for `1 <= B <= 32`, has a separate aligned-buffer implementation and does not reproduce the same sizing mismatch by source inspection.

This is a source-proven workaround for the identified byte-count mismatch, **not a measured passing device result**. No TTNN import, device use, target execution, reset, or runtime edit was performed during this analysis.

## Runtime evidence

`full_contract_device_fill.log:32-45` records Watcher stopping a 64-byte NoC write from local L1 `0x01b7c0` to DRAM, with `NOC transaction overflows a circular buffer`. The active BRISC kernel is `ttnn/cpp/ttnn/operations/full/device/kernels/writer_full.cpp`; the other compute/data-movement kernels are blank. Earlier length cases passed through 1025 and 17, and the failure begins after that sequence, consistent with the subsequent unaligned resume path creating its current-position tensor.

The current target code in `tt/functional_decoder.py:236-244` invokes `ttnn.moreh_full((1, x.shape[0]), absolute, ..., dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)` and reshapes to batch width. Batch one therefore selects the failing one-element row. The shared address spelling `0x1b7c0` in this Watcher message and the unrelated dispatch-fence evidence is not a causal connection: this failure is on a worker core, with a directly identified fill kernel and circular-buffer bound violation.

## Producer/consumer byte ledger

| Boundary | Existing behavior | Width 1 INT32 | Width 32 INT32 |
|---|---|---:|---:|
| Output page payload | `FullInterleavedProgramFactory`, `full_program_factory_interleaved.cpp:42-46`, takes `output.mesh_buffer().page_size()` | 4 B | 128 B |
| Fill-buffer requested entry size | Same file `make_fill_value_dfb`, `.entry_size = page_size`, `.num_entries = 1` | 4 B requested (allocator may enforce its own minimum) | 128 B |
| Initialized data | `elems_per_page = page_size / element_size`; `writer_full.cpp` INT32 loop writes that many values | 1 integer | 32 integers |
| NoC source transfer | `writer_full.cpp:66-67`, `noc.async_write(dfb, s, s.get_aligned_page_size(), ...)` | 64 B observed | 128 B |

`tt_metal/impl/buffers/buffer.cpp:880` defines aligned page size by rounding `page_size()` to buffer alignment. The writer's source transfer must fit its source fill buffer; using the destination's aligned stride without aligning that source violates the invariant for small rows. This is the earliest inconsistent size calculation. Suppressing Watcher or reducing the write check would conceal an actual over-read and is not a repair.

Width 16 would also produce 64 bytes on this device, but width 32 produces 128 bytes, accommodates all requested batches through 32, and remains a complete aligned source page. The target workaround does not depend on accidental allocation padding.

## Proposed target-local change

```python
batch = x.shape[0]
# For the current contract covering batches 1..32:
padded_position = ttnn.moreh_full(
    (1, 32), absolute, self.mesh,
    dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
)
position = ttnn.reshape(padded_position[:, :batch], (batch,))
```

Preserve an explicit batch-bound contract or derive the physical width by rounding batch up to a multiple of 32 if wider batches are already supported; do not silently truncate a batch greater than 32. All 32 physical values contain the same absolute position, so logical selection preserves the existing scalar-position semantics. Independent starts in the list-of-requests API are handled by separate calls and remain independent.

This workaround remains target-local, as requested. The upstream minimal repair would be to make full's source DFB entry cover the aligned transfer, with consistent initialization of those bytes; that broader runtime change is not proposed for this stage's edits.

## Why the slice and reshape do not repeat the defect

- For a contiguous slice beginning at width zero, `slice_program_factory_rm.cpp:182-212` computes `single_alignment=max(source_alignment,destination_alignment)`, zero misalignment, and `stick_size_aligned=round_up(output_row_bytes, alignment)`. Its DFB entry uses that aligned size, unlike full's unaligned entry.
- For `B=1..16`, output payload `4*B` fits a 64-byte aligned DFB; for `B=17..32`, it fits a 128-byte DFB. The source filled row is 128 bytes in every case. Thus any aligned source read needed for the output remains within the source row, and the output's aligned allocation covers the corresponding write.
- The row-major slice writer uses its computed `stick_size` and performs a write barrier before releasing entries; its source DFB size was derived from aligned geometry. See `slice/device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp` and `slice_program_factory_rm.cpp`.
- `slice.cpp:435-446` launches `ttnn::prim::slice` on the device and adjusts shape with a view. This path performs no host numerical conversion.
- `(1,B)` to `(B,)` keeps the last dimension unchanged and preserves row-major memory config. `reshape_view/reshape.cpp:760-770` selects `PerformView` for precisely that condition. This is a metadata reshape with the same device data, not a fallback.
- The ordinary `ttnn.moreh_full` C++ path invokes device `prim::full`; its attached Python golden function is not the measured runtime when comparison mode is disabled. This change must continue using `moreh_full`, not the small-int32 creation `full` path already identified as uploading host data.

## Focused validation still required

Parent should run the smallest Watcher-enabled device-only position construction and value check for batches `1, 2, 15, 16, 17, 31, 32`, with representative absolute positions including `1, 31, 33, 511, 262143`. These span the logical and 64/128-byte boundaries. Setup/comparison reads may be outside the guarded pass; patch `ttnn.from_torch` and `ttnn.to_torch` to fail within the construction/pass boundary. Then rerun the previously failing unaligned continuation battery under Watcher and its per-row contribution/negative-history checks. The source analysis predicts no fill CB overrun; only those runs can certify the device repair and preserve the functional stage gates.

## Other requested cleanup

The owner already prepended `triage_long/HOST_DIAGNOSIS.md` with the capture-interference correction. It now states that full tt-triage preceded the queue/debugger snapshots and that preexisting FetchQ failure was not established. The never-run standalone private-memory helper is already absent. No additional changes to those owner-corrected artifacts were needed.
