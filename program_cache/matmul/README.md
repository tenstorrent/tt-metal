Matmul — Program Cache Review

Status: Reviewed — no program cache issues found (baseline multi_core path examined)

Summary
- Old infra op with multiple program factories; examined `matmul_multi_core` path.
- No explicit custom hash in file; old infra defaults to hashing op type and input tensors, which cover shapes/dtypes/layouts/mem configs driving compile-time constants and grid split.
- Override updates runtime-only buffer base addresses for inputs A/B and output per core; counts/offsets are derived from hashed shapes and remain valid on cache hit.

Key locations reviewed
- `ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_program_factory.cpp`
  - Reader args at create include `[a_addr, b_addr, Mt, Kt, Nt, MtKt, KtNt, B, bcast_batch, tile_offset, tiles_per_core, MtNt]`.
  - Writer args at create `[dst_addr, tiles_per_core, tile_offset]`.
  - Override updates: reader indices 0–1, writer index 0.

Notes
- Other optimized/mcast/sharded program factories follow the same callback pattern; spot-checking shows they update buffer bases on cache hit.
