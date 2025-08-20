MatmulReduceScatterAsync (experimental CCL) — Program Cache Review

- Summary: Fused op that composes Matmul and ReduceScatterMinimalAsync into one program with fused override callbacks.
  It defines a custom `compute_program_hash(...)` but currently under-keys by excluding matmul determinants.

- Files reviewed:
  - `device/matmul_reduce_scatter_async_op.hpp/.cpp` — struct, create_program_at, compute_program_hash
  - `device/multi_core/matmul_reduce_scatter_async_op_multi_core.cpp` — builds fused program and fuses override callbacks

- Program cache behavior:
  - Custom hash: uses `hash_operation<MatmulReduceScatterAsync>(...)` with RS parameters and input[0] properties only.
  - Missing in hash: matmul program configuration and weight tensor properties that influence compiled kernels.
  - On cache-hit, overrides update runtime-only args for both sub-ops, but cannot correct a mismatched compiled matmul.

- Issue(s) found: Under-keyed program hash
  - See `failures/test_matmul_reduce_scatter_async_cachehit_underkey_matmul.py` which changes only matmul program_config
    between runs; first run seeds the cache, second run reuses cached program → expected PCC fail on cache-hit.

- Suggested fixes:
  - Add matmul determinants and weight tensor properties to the hash, or compute and include the selected matmul
    ProgramFactory index and its determinants.
