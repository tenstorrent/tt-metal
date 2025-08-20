Index Fill — Program Cache Review

Status: Reviewed — no program cache issues found

Summary
- Old-infra-style program factory within new infra device op. Reader and writer kernels with per-core runtime args.
- Default hash used; determinants include op attributes (`dim`, `value`, output mem_config) and input/index tensor shapes/dtypes/mem configs, which drive compile-time constants (unit sizes, flags, grid split).
- Override updates runtime-only buffer base addresses for input, index, and output per core; all size/offset args are recomputed based on hashed properties.

Key locations reviewed
- `ttnn/cpp/ttnn/operations/index_fill/device/index_fill_multi_core_factory.cpp`
  - Reader args at create: `[input_addr, index_addr, fill_value_bits, input_unit_size, index_unit_size, unit_offset, num_rows_per_core, num_rows_to_fill_per_index, input_dim_size]`.
  - Writer args at create: `[output_addr, num_rows_per_core, unit_offset, output_unit_size]`.
  - Override updates: reader indices 0–1; writer index 0.
- `ttnn/cpp/ttnn/operations/index_fill/device/index_fill_device_operation.cpp`
  - Validations constrain shapes/layouts; ensures consistency for cache hits.

Notes
- `fill_value` is part of `operation_attributes` and thus hashed; cache entries won’t be reused across different fill values.
