## Program cache review: experimental/ccl/all_gather_matmul

OP reviewed
- `ttnn::operations::experimental::ccl::all_gather_matmul_*` fused pipeline (old infra; `ProgramWithCallbacks`).

Creation path summary
- Constructs matmul program (with its own override) and fuses it with all-gather helper program.
- Optional datacopy stage can be included for debugging; its override also updates buffer addresses.

Override behavior (cache-hit path)
- Fused override calls:
  - Matmul override: refreshes buffer addresses and any kernel CB bindings.
  - All-gather override: refreshes input and output addresses and semaphores for sender cores.
  - Optional datacopy override: refreshes source/destination addresses.

Findings
- Override chaining preserves correct address updates for all fused stages. No stale address usage detected.

Conclusion
- No program cache issues found. No failing cache test required.
