Halo — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/halo_device_operation.hpp`
- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/halo_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.cpp`

Findings:
- Uses the old type-erased infra with an override callback.
- Non-inplace path updates runtime-only values on cache hits:
  - Dynamic CB base addresses for source and output via UpdateDynamicCircularBufferAddress.
- In-place path similarly updates source/output CB base addresses.
- Kernel configuration CBs (padding/gather/local/remote configs) are created from device buffers and their storages are intentionally captured in the cached program when capture_buffers=true. This avoids needing to update their addresses during overrides and keeps content/address stable across cache hits for identical attributes.
- Program keying relies on default hashing of operation attributes which include config_, parallel_config_, pad_val_, remote_read_, transpose_mcast_, max_out_nsticks_per_core_, output_memory_config_, is_out_tiled_, in_place_. These cover codegen-affecting choices and prevent under-keying.

No program-cache issues identified.

Suggested optional tests:
- Two-run cache test that reallocates input/output tensors (new buffer addresses) with identical op attributes and asserts correctness on cache hit.
- Variant covering in-place mode when legal (input and output buffers aligned as enforced by validation), confirming override updates only source/output CBs.
