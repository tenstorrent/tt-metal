Split Query/Key/Value And Split Heads — Program Cache Review

Status: Reviewed — no program cache issues found

Summary
- This OP uses the old (type-erased) ProgramWithCallbacks flow.
- On cache hit, the override updates all runtime-only buffer base addresses for input (reader) and outputs Q/K/V (writer), which is required for correctness.

Key locations reviewed
- Non-sharded program factory override updates:
  - `ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/split_query_key_value_and_split_heads_program_factory.hpp`:
    - reader override updates input base address at runtime
    - writer override updates Q/K/V base addresses at runtime
- Sharded program factory override updates:
  - Uses dynamic CB address updates for input and each output CB

Details
- Non‑sharded path
  - Reader runtime args during create: `[in_addr, in_tile_id]`.
  - Writer runtime args during create: `[q_addr, k_addr, v_addr, out_tile_id, out_tile_id_with_transpose]`.
  - Override updates only addresses (indices 0..2) which are runtime-only; tile ids are derived from hashed shape/grid and need not change on cache hits.
- Sharded path
  - Circular buffers are created with globally allocated addresses; on cache hit, the override calls `UpdateDynamicCircularBufferAddress(...)` for input and each of the three outputs, ensuring fresh buffer bases are used.

Why no issue was filed
- All runtime-only buffer bases are updated on cache hit.
- Order and indices of runtime args in overrides match creation order for the respective kernels.
- Compile-time args depend on hashed properties (dtype, layout, memory location, tile counts); cache keys will differ if these change, avoiding stale code reuse.

Suggested tests (optional, passing)
- A two-run cache test that verifies cache entry increments on first run and that second run (with new tensor allocations but unchanged hashed properties) produces identical outputs.
