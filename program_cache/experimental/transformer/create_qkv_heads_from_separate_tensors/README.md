### create_qkv_heads_from_separate_tensors program cache review

- **OP**: `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors`

Status: Reviewed — no program cache issues found.

Findings
- **Infra**: Old type-erased operation (`tt::tt_metal::operation::ProgramWithCallbacks`).
- **Override coverage**: The override callback updates dynamic circular buffer base addresses for all tensors on cache-hit:
  - Input Q buffer CB and input KV buffer CB
  - Output Q/K/V buffers CBs
- **Kernel arg ordering**: Address updates are applied via `UpdateDynamicCircularBufferAddress` on CB handles captured at create-time; there are no per-kernel positional runtime-arg vectors that can drift.
- **Hashing/compile-time determinants**: Program creation depends on input tensor shapes/dtypes/layouts/sharding, `num_q_heads`, `num_kv_heads`, `head_dim`, `transpose_k_heads`, and output `MemoryConfig`. Default hashing of attributes and tensors covers these, preventing under-keying. There are no runtime-only scalars besides buffer addresses.
- **CB sizes/page sizes**: Derived from per-core tile counts and tile size at create-time; fixed for a given hashed configuration.

Key references
- `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/device/create_qkv_heads_from_separate_tensors_program_factory.cpp`
  - Override updates: lines where `UpdateDynamicCircularBufferAddress` is invoked for both inputs and all outputs.

Recommendation
- None required. Optionally add a two-run cache test varying only buffer base addresses to exercise cache-hit override path.
