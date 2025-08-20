Dram Prefetcher program cache review

Status: Reviewed — no program cache issues found.

Summary
- Operation: `ttnn/cpp/ttnn/operations/prefetcher/prefetcher` (old infra ProgramWithCallbacks).
- The program creates circular buffers bound to a provided `GlobalCircularBuffer` and a tensor of addresses (`tensor_addrs`).
- Runtime args at creation include per-core VC/bank selection, page sizes, num pages, tile sizes, and per-core block sizes; these are derived from hashed inputs (tensor shapes/layouts, global CB mapping, num_layers, performance mode) and remain valid on cache hits.
- Override callback updates only the dynamic circular buffer address for `tensor_addrs_cb` to the current `tensor_addrs` buffer, which is the only runtime-only value that changes between runs.

Key references
- `device/dram_prefetcher_op_multi_core.cpp`
  - Override uses `UpdateDynamicCircularBufferAddress(program, tensor_addrs_cb, *tensor_addrs_buffer)`.
  - Reader/writer runtime arguments that depend on shapes and mapping are set during creation and need not be updated on cache hits.

Recommendation
- No changes required.
