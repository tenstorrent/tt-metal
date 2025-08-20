KV Cache — Program Cache Review

Status: Reviewed — no program cache issues found

Summary
- Old infra operation `UpdateCache` with custom `compute_program_hash(...)` including op type and input tensor properties.
- Two paths: UPDATE and FILL. Both set per-core runtime args, and overrides update all runtime-only values on cache hit, including buffer base addresses and key indices (cache_start_id, tile_update_offset) that depend on per-invocation `update_idx`/`batch_idx`.

Key locations reviewed
- `ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op.cpp`
  - `compute_program_hash(...)` uses `operation::hash_operation<UpdateCache>(this->op_type, input_tensors)`.
  - Validations ensure shaped properties remain consistent across cache hits.
- `ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op_multi_core.cpp`
  - UPDATE path override updates reader args `[dst_addr, src_addr, ..., cache_start_id]` and writer args `[dst_addr, ..., cache_start_id, ..., tile_update_offset]`. Updates sharded CB address when applicable.
  - FILL path override updates reader `[src_addr]` and writer `[dst_addr, cache_start_id]` and sharded CB address if needed.

Notes
- `update_idx` and `batch_idx` are runtime-only and not part of the hash; they are correctly applied in override lambdas using the captured program-time state combined with the current operation’s fields.
