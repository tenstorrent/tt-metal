## Program cache review: experimental/ccl/all_gather_concat_heads_fused

OP reviewed
- `ttnn::operations::experimental::ccl::all_gather_concat_heads_fused` (old infra; `ProgramWithCallbacks`).

Creation path summary
- Builds reader/writer for all-gather plus on-device concat/tilize pipeline; uses global semaphores and optional NOC1.
- Sets static CB handles for output tilize path and per-core slice parameters.

Override behavior (cache-hit path)
- Updates dynamic CB address for `cb_q_output` to new output tensor buffer.
- For sender cores: reader `[0] = input.addr; [1] = semaphore.addr`; writer `[0] = temp.addr; [1] = semaphore.addr`.
- Updates concat reader runtime args per core with fresh base addresses of temp and input buffers.

Findings
- All runtime-only addresses are refreshed. Per-core indices/counts are recomputed from shapes (hashed properties).

Conclusion
- No program cache issues found. No failing cache test required.
