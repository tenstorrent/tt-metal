## Program cache review: experimental/ccl/all_broadcast_async

OP reviewed
- `ttnn::operations::experimental::ccl::all_broadcast_async` (old infra; `ProgramWithCallbacks`).

Creation path summary
- Builds sender reader/writer kernels per link; computes per-link tile ranges and sets runtime args.
- Writer args include output buffer base, global semaphore address, and optional fabric connection args.
- Supports TILE and row-major layouts; handles sharded inputs via helper that extends CT/RT args.

Override behavior (cache-hit path)
- Callback updates for each sender core:
  - Reader: `[0] = input.buffer()->address()`.
  - Writer: `[0] = output[ring_index].buffer()->address()`, `[1] = out_ready_semaphore.address()`, `[9] = barrier_semaphore.address()`.
- All per-invocation addresses refreshed; counts/indices derived from hashed shape remain unchanged.

Findings
- No missing runtime-argument updates detected. Semaphores and buffer base addresses are correctly refreshed.
- Default hashing includes inputs’ shape/layout/dtype/memory_config via the op wrapper, so cache keys are adequate.

Conclusion
- No program cache issues found. No failing cache test required.
