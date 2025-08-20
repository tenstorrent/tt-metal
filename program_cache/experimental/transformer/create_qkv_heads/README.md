### create_qkv_heads program cache review

- **OP**: `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads`

Status: Reviewed — no program cache issues found.

Findings
- **Infra**: Old type-erased operation (`tt::tt_metal::operation::ProgramWithCallbacks`).
- **Override coverage**: The override callback updates dynamic circular buffer base addresses for all tensors on cache-hit:
  - Input QKV buffer CB
  - Output Q/K/V buffers CBs
- **Kernel arg ordering**: No per-kernel runtime argument vectors are used; address updates are applied via `UpdateDynamicCircularBufferAddress` for CBs captured at create-time, avoiding index drift issues.
- **Hashing/compile-time determinants**: Compile-time args depend on input tensor shape/dtype/layout/sharding, `num_q_heads`, `num_kv_heads`, `head_dim`, `transpose_k_heads`, and output `MemoryConfig`. These are covered by default hashing of operation attributes and input tensors, preventing under-keying. No runtime-only scalars exist beyond buffer addresses.
- **CB sizes/page sizes**: Computed from per-core tile counts and tile size at create-time; not runtime-varying for a given hashed configuration.

Key references
- `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads/device/create_qkv_heads_program_factory.cpp`
  - Override updates: lines where `UpdateDynamicCircularBufferAddress` is called for input and all three outputs.
- `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads/device/create_qkv_heads_device_operation.cpp`
  - Validation and spec/layout derivation for compile-time parameters.

Recommendation
- None required. Optional: add a simple two-run cache test that varies only buffer base addresses to exercise the override path.
