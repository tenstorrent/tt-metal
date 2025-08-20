GELU Backward — Program Cache Review

Status: Reviewed — no program cache issues found

Summary
- New infra device op with a dedicated program factory and explicit override.
- Cache hash includes operation attributes (approximation mode, output dtype/mem_config) and both input tensors' dtype/mem_config and input padded volume.
- On cache-hit, override correctly updates runtime-only values (buffer base addresses and per-core tile counts/offsets) for reader, compute, and writer.

Key locations reviewed
- `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/gelu_backward_program_factory.cpp`
  - Runtime args at create:
    - reader: `[src0_addr, src1_addr, num_tiles_per_core, num_tiles_written, 0, 0, num_cores_y]`
    - compute: `[num_tiles_per_core, 1]`
    - writer: `[dst_addr, num_tiles_per_core, num_tiles_written]`
  - Override updates: reader indices 0–3, compute index 0, writer indices 0–2.
- Hash: `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/gelu_backward_device_operation.cpp`
  - Uses `operation::hash_operation<GeluBackwardDeviceOperation>(...)` with program factory index, input/grad dtypes & mem configs, and input padded volume. This prevents under-keying; buffer addresses and per-core offsets are not hashed.

Notes
- Program cache is per-device; the per-device grid used for work-splitting is stable across runs on the same device, matching cache scope.
