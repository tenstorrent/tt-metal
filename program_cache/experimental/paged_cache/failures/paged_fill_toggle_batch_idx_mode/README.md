### Issue: Toggling batch_idx mode (tensor vs scalar) across cache-hit

- OP: `ttnn/cpp/ttnn/operations/experimental/paged_cache`
- Program factory: `device/paged_fill_cache_program_factory.cpp`

What this test does:
- First run compiles with batch_idx passed as a tensor (uses CB and loads it in the writer kernel).
- Second run reuses the cached program but passes batch_idx as a scalar fallback.

Why this should expose a bug on cache-hit:
- The program hash for `PagedUpdateCacheDeviceOperation` is computed from op attributes and input/optional tensors. The path (tensor vs scalar) is encoded as compile-time args in the writer, so toggling it should either:
  - produce a different program hash; or
  - keep the hash the same but have the override logic update the correct runtime args for the other path.
- Current implementation sets compile-time args for `use_batch_idx_tensor` and CB setup during create(...), and override only updates buffer addresses and per-core args. Switching modes across runs likely leads to stale compile-time configuration on cache-hit.

References:
- `device/paged_fill_cache_program_factory.cpp`: writer compile-time args indices include flags for batch tensor presence; override at lines that update `writer_args[...]` cannot change compile-time path.
- `device/paged_cache_operation.cpp`: custom hash only uses `op_type` and tensors, not the batch mode flag explicitly.

Failure mode:
- PCC mismatch on the second run when toggling from tensor to scalar for `batch_idx`.

Repro:
```bash
pytest -q program_cache/experimental/paged_cache/failures/paged_fill_toggle_batch_idx_mode/test_paged_fill_toggle_batch_idx_mode.py::test_paged_fill_toggle_batch_idx_mode_program_cache -s --disable-warnings
```

Suggested fix:
- Include the effective batch mode (tensor vs scalar) in the program hash for FILL, or
- Unify the path so both modes use the same compile-time configuration and read batch_idx via CB regardless, with scalar written into a 1-element CB.
