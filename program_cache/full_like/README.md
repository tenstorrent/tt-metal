Full Like — Program Cache Review

Status: Reviewed — no program cache issues found

Summary
- New infra device op; writer-only program similar to `full` but shapes come from input.
- Default hash is used (no custom hash). Determinants include op attributes (fill_value, dtype, layout, memory_config) and input tensor properties, which define grid/work split and compile-time constants.
- Override updates only output buffer base address per core; other runtime args are derived from hashed properties.

Key locations reviewed
- `ttnn/cpp/ttnn/operations/full_like/device/full_like_factory.cpp`
  - Writer runtime args at create: `[dst_addr, fill_value_bits, num_tiles_per_core, tiles_offset]`.
  - Override updates index 0 (dst_addr) for each core.
- `ttnn/cpp/ttnn/operations/full_like/device/full_like_device_operation.cpp`
  - Validations ensure interleaved/tile layout; dtype conversion safeguards.

Notes
- `fill_value` and chosen `dtype` are in `operation_attributes` and thus hashed; cache entries won’t be erroneously reused across differing fill values or dtypes.
