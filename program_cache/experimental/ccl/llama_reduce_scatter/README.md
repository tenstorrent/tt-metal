## Program cache review: experimental/ccl/llama_reduce_scatter

OP reviewed
- `ttnn::operations::experimental::ccl::llama_reduce_scatter` (new infra; mesh program cache).

Creation/override summary
- Creates per-mesh programs and stores shared variables: kernel IDs, CB handles, core ranges.
- Uses globally allocated CBs bound to input/output/packet buffers; per-core runtime args encode role and packet slices.
- Override updates dynamic CB base addresses and refreshes cross-device semaphore addresses in reader/writer args.

Findings
- Correctly updates CB base addresses for input/output/packet buffers on cache hit.
- Semaphore address refreshed in both reader and writer args for all cores; per-core roles preserved.

Conclusion
- No program cache issues found. No failing cache test required.
