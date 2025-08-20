SDPA Decode — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_op.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_op.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`

Findings:
- Uses the old type-erased infra with an override callback.
- Override updates all runtime-only values on cache hits:
  - Reader: Q/K/V/cur_pos/page_table/attn_mask addresses and `page_table_stick_size`.
  - Writer: output buffer base address; reducer/output coordination indices.
  - Compute: per-core flags/indices derived from hashed inputs.
  - For sharded outputs, updates dynamic CB out address via `UpdateDynamicCircularBufferAddress`.
- Program hash includes flags that affect codegen (scale, memory config, program config, compute config, k_chunk_size, paged_attention, is_causal, use_mla, head_dim_v, presence of optional tensors, and tensors themselves for page table sizing), avoiding under/over-keying.

No program-cache issues identified.

Suggested optional test:
- Two-run cache test varying only buffer addresses (and optional presence/absence values held constant) to confirm override path correctness.
