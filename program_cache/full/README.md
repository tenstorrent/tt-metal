Full — Program Cache Review

Status: Reviewed — no program cache issues found

Summary
- New infra device op with simple writer-only program.
- No custom hash defined; default hash includes op type and `operation_attributes` (shape, dtype, layout, memory_config) and `tensor_args` (`any` tensor). These fully determine compile-time behavior; runtime-only buffer addresses are excluded.
- Override updates only the output buffer base address per core on cache-hit; all other runtime args are derived from hashed properties and remain valid.

Key locations reviewed
- `ttnn/cpp/ttnn/operations/full/device/full_program_factory.cpp`
  - Writer runtime args at create: `[dst_addr, fill_value_bits, num_tiles_per_core, tile_offset]`.
  - Override updates index 0 (dst_addr) for each core.
- `ttnn/cpp/ttnn/operations/full/device/full_device_operation.cpp`
  - Validations enforce interleaved, tile layout; shapes from attributes decide work split.

Notes
- Fill value is embedded as a constant in the per-core runtime args at create; it does not change across cache hits because it’s part of `operation_attributes` hashed by default infra.
