## Program cache review: experimental/ccl/all_gather_matmul_async

OP reviewed
- `ttnn::operations::experimental::ccl::all_gather_matmul_async_*` fused pipeline (old infra; `ProgramWithCallbacks`).

Creation path summary
- Builds matmul program and an async all-gather program; fuses via a combined override.
- Runtime args include buffer base addresses, per-core indices, semaphores, and optional barrier semaphore.

Override behavior (cache-hit path)
- Fused override updates:
  - Matmul stage addresses (input/output/weights) via its provided override.
  - All-gather stage addresses and semaphores for sender cores; barrier semaphore address refreshed if present.

Findings
- Correct override sequencing and index stability across cache hits. No under-keying observed.

Conclusion
- No program cache issues found. No failing cache test required.
