moreh_abs_pow — Program Cache Review

Status: Reviewed — no program cache issues found

Summary
- New infra op; factory sets reader, writer, and compute kernels with per-core runtime args derived from input shapes and attribute `p`.
- Default hash path used; determinants include op attributes (p, compute kernel config, output mem_config) and input tensor properties that affect compile-time constants and work split.
- Override updates input and output buffer base addresses per core; counts/offsets and flags remain valid as they are derived from hashed properties.

Key locations reviewed
- `ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow/device/moreh_abs_pow_program_factory.cpp`
  - Reader create args: `[in_addr, in_is_dram, decimal_bits, num_units_per_core, Wt, tile_offset, origin_w]`.
  - Writer create args: `[out_addr, out_is_dram, num_units_per_core, Wt, tile_offset]`.
  - Compute create args: `[num_units_per_core, Wt, origin_w, floored_p, p_is_negative]`.
  - Override updates reader[0] and writer[0] only (buffer bases).
