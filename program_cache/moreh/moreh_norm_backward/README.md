Moreh Norm Backward — Program Cache Review

- Status: Reviewed — no program-cache issues found

Summary
- The factory sets reader args with input, output, output_grad base addresses plus compile-time/hashed values (decimal from p, dims, tile counts). Writer args include output buffer address and tile metadata.
- On cache-hit, override updates all runtime-only values (buffer base addresses) and leaves hashed/derived values unchanged.

Key references
- Factory create: `ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/moreh_norm_backward_program_factory.cpp`
  - Reader args order set in create (addresses, decimal, num_tiles_per_core, tile_offset, dims): around L230–L241.
  - Writer args order set (output address, num_tiles_per_core, tile_offset): around L245–L250.
- Cache-hit override updates:
  - Reader base addresses updated: around L282–L286.
  - Writer base address updated: around L289–L292.

Notes
- `p`, dims, and shapes live in `operation_attributes_t`/`tensor_args_t` and participate in the default hash, so reusing the same cache entry across runs implies these values do not change. Only DRAM base addresses vary per run and are correctly overridden.
