## Program cache review: examples/example_multiple_return

OP reviewed
- `ttnn::prim::example_multiple_return` implemented by `ExampleMultipleReturnDeviceOperation::SingleCore`.

Program factory and override summary
- Program creation sets runtime args per core for:
  - Reader DM kernel args: `[0] = src_buffer_address, [1] = num_tiles_per_core, [2] = num_tiles_written`.
  - Writer DM kernel args: `[0] = dst_buffer1_address, [1] = dst_buffer2_address, [2] = num_tiles_per_core, [3] = num_tiles_written`.
- Compile-time args include destination DRAM flags per optional output. These derive from `return_output1/2` and buffer types.
- Grid is fixed to `1x1`; only one core is used.

Override behavior (cache-hit path)
- Reader: updates `[0]` with the new input buffer base address for core `{0,0}`.
- Writer: updates `[0]` and `[1]` with the new output buffer base addresses for core `{0,0}` (only if corresponding optional outputs are present).
- Counts/offsets are not changed in override, which is correct because they are shape-derived and hashed properties remain constant across cache hits.

Hashing and factory selection
- Default hash is used. Attributes `return_output1/2` influence output specs and writer compile-time flags; differing these across runs will select a different program (cache miss), which is correct.

Findings
- No missing runtime-argument overrides detected for the single-core configuration.
- No under- or over-keying observed with default hashing for constant shapes/attributes.

Conclusion
- Reviewed with no program cache issues identified. No failing cache test required.
