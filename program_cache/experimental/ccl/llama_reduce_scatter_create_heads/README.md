## Program cache review: experimental/ccl/llama_reduce_scatter_create_heads

OP reviewed
- `ttnn::operations::experimental::ccl::llama_reduce_scatter_create_heads` (new infra; mesh program cache).

Creation/override summary
- Creates per-mesh programs; shared variables include kernel IDs, CB handles, and core ranges.
- Binds CBs to input and packet buffers; per-core runtime args include q/k/v base addresses and role flags.
- Override updates CB base addresses and refreshes cross-device semaphore and q/k/v base addresses for all cores.

Findings
- Address updates are complete (input, packet, q/k/v outputs, semaphore). Per-core role indices remain consistent.

Conclusion
- No program cache issues found. No failing cache test required.
