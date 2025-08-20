Test: matmul_reduce_scatter_async — under-keyed program hash for matmul configuration

- Issue: The custom `compute_program_hash(...)` in `MatmulReduceScatterAsync` omits the matmul program configuration and
  weight tensor properties. On cache-hit, if the matmul program configuration changes (e.g., different `in0_block_w` or
  `out_block_w`), the same cached program is reused, leading to incorrect behavior.

- Suspected root cause:
  - `ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_op.cpp`
    `MatmulReduceScatterAsync::compute_program_hash(...)` L164-L191 — hashes only ReduceScatter fields and input[0]
    properties. It does not include `matmul_struct` attributes (program config, compute kernel config, dtype overrides)
    nor the weight tensor shape/dtype/memory config.
  - The fused program’s creation wires matmul and reduce-scatter override callbacks together, but reuse of a program
    compiled for a different matmul layout/config cannot be corrected by runtime overrides.

- Failure mode: PCC mismatch on the second run (cache-hit), when only the matmul `program_config` is changed.

- Reproduction:
  - Test file: `program_cache/experimental/ccl/matmul_reduce_scatter_async/failures/test_matmul_reduce_scatter_async_cachehit_underkey_matmul.py`
  - Command:
    - `pytest -q program_cache/experimental/ccl/matmul_reduce_scatter_async/failures -s --disable-warnings`

- Environment notes:
  - Requires multi-device mesh (>=2 devices) and Fast Dispatch (SLOW_DISPATCH disabled).
  - Uses persistent buffers and global semaphores as required by the op.

- Suggested fix:
  - Include matmul determinants in `compute_program_hash(...)`, e.g. serialize fields from `matmul_struct` that affect
    kernel/codegen selection: program grid, in0_block_w, out_subblock_h/w, per_core_M/N, out_block_w, transpose flags,
    fused activation, user_run_batched, dtype, memory_config, and properties of `weight_tensor`.
  - Alternatively, select the matmul ProgramFactory first and hash its index plus its determinants, similar to other
    device operations patterns.
