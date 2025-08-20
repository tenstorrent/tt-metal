AllReduce (experimental CCL) — Program Cache Review

- Summary: This OP is a composite that delegates to existing CCL ops and does not define its own device ProgramFactory or override logic. It selects one of two strategies based on input shape/topology and calls sub-ops via `tt::tt_metal::operation::run(...)`:
  - ReduceScatter → AllGather
  - AllGather → local reduce (sum)

- Files reviewed:
  - `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce/device/all_reduce_op.cpp` — strategy selection and composition
  - `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce/all_reduce.cpp` — registered op wrapper

- Program cache behavior:
  - No separate program cache entries are created for this wrapper; cache hits/misses occur inside the invoked sub-ops:
    - `ttnn::ReduceScatter`
    - `ttnn::AllGather`
  - Both sub-ops were already reviewed (marked DONE) and handle cache-hit overrides for buffer addresses and other runtime-only args.

- Notable constraints affecting testing:
  - Requires >1 device; will `TT_FATAL` otherwise.
  - Only supported with Fast Dispatch (`TT_METAL_SLOW_DISPATCH_MODE` must not be set).

- Findings:
  - No additional override/runtime-arg handling is implemented at this wrapper level; there is no evidence of stale runtime arguments being reused.
  - Hashing/caching correctness is inherited from the sub-ops. The wrapper’s attributes that influence strategy selection (tensor shape, topology, num_links, ring size) are forwarded to sub-ops, so the compiled program choice and cache keys live there.

- Suggested follow-ups: None. No tests added for this wrapper; rely on sub-op program-cache tests for coverage.
