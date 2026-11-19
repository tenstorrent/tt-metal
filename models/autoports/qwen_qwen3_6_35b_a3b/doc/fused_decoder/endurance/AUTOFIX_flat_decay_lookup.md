# Flat FP32 decay-table gather: source rejection

Source-only; no hardware or runtime edits. Proposed table `[1,2097152]` FP32 with UINT32 indices `[1,32]`, where index[h]=code[h]+65536*h, has correct lookup semantics and in-range indices (maximum2097151). Native gather explicitly accepts UINT32 indices (`gather/device/gather_device_operation.cpp:82–87`), and all addresses fit exact integer arithmetic. **The proposed native gather cannot fit its required per-core circular buffer.**

For ROW_MAJOR, `GatherDeviceOperation::select_program_factory` (`gather_device_operation.cpp:24–33`) selects the row-distributed factory whenever W_index<=1920. Thus this32-output candidate uses `RmSingleRowSingleCore` and H=1, allocating one active core. `gather_program_factory.cpp:379–416` computes input_stick_size_bytes=W_input*element_size and directly allocates that entire size as its input CB:2097152*4=8,388,608bytes, before any other buffers. The existing32-row table instead allocates65536*4=262,144bytes on each of32 active cores.

This is not a sparse address reader. `kernels/dataflow/gather_writer_rm_single_row_single_core.cpp` copies the full input row into its local CB before the reader performs element lookups. Adding output columns to force the column-distributed factory does not help: that path still stages the full row on every participating core, with a two-page input CB (`gather_program_factory.cpp:596–616`), doubling the local requirement. Width sharding of the DRAM tensor likewise does not replace the full-row CB; the common accessor assembles the row from shards.

The BF16-only codegen implementation offers a bounded-residency streaming fallback, but `gather/codegen/gather_codegen_supported.cpp:77–81` rejects FLOAT32 input. Changing layout to TILE therefore cannot enable that fallback while preserving the exact FP32 table. `ttnn.tosa.gather` is only an index-expand wrapper around ordinary `ttnn.gather`, so it has the same blocker. No native-device experiment of the proposed flat shape is justified by this source contract.

Keep the existing per-head FP32 RM table for this interface. A true flat sparse FP32 lookup requires a different supported native reader, or native work outside the permitted model-only scope; UINT32 offset arithmetic alone does not solve the capacity problem.

## Follow-up: setup-transposed table, gather dim0

`table_T[65536,32]` FP32 RM with `indices[1,32]` and dim0 is semantically valid: output[0,h]=table_T[code[h],h]. It passes native dtype/rank/non-gather-dimension validation. It does **not** produce32 sparse128-byte row reads in this implementation.

`gather.cpp:118–133` calls `reduction_common::perform_transpose(input,is_dim_last,dim,-1)` for both input and index. `reduction_common.cpp:13–16` calls actual `ttnn::transpose` whenever the gather axis is not last. Consequently the device primitive receives exactly table `[1,1,32,65536]` and indices `[1,1,32,1]`, the existing layout. It stages the same262144-byte row on each of32 cores and reads8MiB total. `post_gather_transform_tensor` then transposes the result back. The large table transpose is a runtime device data movement operation; a program-cache hit does not memoize its tensor contents.

The device validation source even documents that operands have already been transposed and arbitrary-axis handling is a future improvement (`gather_device_operation.cpp`, non-last dimension loop). Reader compile-time args include H/W_input/W_index, not an arbitrary gather axis. Bypassing the wrapper with a dim0 primitive call would therefore not implement the requested indexing.

No candidate patch was produced: this legal call adds an8MiB table transpose while preserving the same full-row lookup. The existing `tests/gate_lut_layout_probe.py` already contains both transposed and original-layout adapters if a runtime confirmation is desired, but source disproves the claimed sparse-read mechanism. Prefill/runtime source remains unchanged.
