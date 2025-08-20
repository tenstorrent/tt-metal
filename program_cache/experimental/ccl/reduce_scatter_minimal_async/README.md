# Program cache review — experimental/ccl/reduce_scatter_minimal_async

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- The op uses the old (type-erased) program infra and wires an `override_runtime_arguments_callback` for both Ring and Line variants.
- Overrides correctly update all runtime-only values on cache-hit:
  - Input/intermediate/output buffer base addresses for reader/writer kernels
  - Semaphore addresses used for inter-device synchronization and per-batch sync
  - Barrier semaphore address (when present)
- Hashing includes all parameters that select kernels and affect codegen, avoiding under-keying for common cases.

## Details and references
- Program creation and overrides:
  - Ring helper: `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp`
    - Reader rt args indices 0–3 updated on cache-hit (input, intermediate, out-ready, batch-ready)
    - Writer rt args indices 0–1, 4–5, and 14 updated on cache-hit (intermediate, output, semaphores, barrier addr)
  - Line helper: same file
    - Reader rt args indices 0–3 updated (input/intermediate/output, semaphore)
    - Writer rt args indices 0–1, 4–6, and 14 updated (intermediate/output, semaphores, barrier addr)
- Hashing:
  - `ReduceScatterMinimalAsync::compute_program_hash(...)` includes: `dim`, `num_links`, `ring_size`, `output_mem_config`, `intermediate_mem_config`, `topology`, `barrier_semaphore.has_value()`, `using_persistent_buffers`, `sub_device_id`, `cluster_axis`, `chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel`, and `input_tensors`. This covers kernel/runtime selection knobs.
  - Mesh workload hash further incorporates tensor coordinates, ensuring per-range cache separation.

## Observations / potential risks
- Device list order is not explicitly hashed; unicast/mcast routing parameters are embedded at compile time. In typical runs, device ordering remains stable within a process, so this is acceptable. If mesh composition changes between first and second runs without changing shapes/topology, a conservative approach would be to incorporate a stable descriptor of participating devices into the hash.
- Booleans controlling barrier usage are compile-time constants derived from hashed fields (`barrier_semaphore.has_value()` and `using_persistent_buffers`), so they do not need runtime overrides.

## Recommendation
- No changes required. Optionally consider hashing a stable device-topology descriptor if future usage expects dynamic device sets under identical tensor properties.
