## Program cache review — ccl/reduce_scatter

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra with `ProgramWithCallbacks` and proper override callback.
- Hashing: default `hash_operation<ReduceScatter>(operation, input_tensors)`; determinants include op fields (reduce op, scatter dim, num_links, ring_size, topology, optional worker/buffer counts) and input tensor properties. No runtime buffer addresses are hashed.
- Overrides on cache-hit update base addresses used by worker kernels:
  - Receiver kernel: updates input and output base addresses (arg indices 0,1).
  - Sender kernel: updates output base address (arg index 0).
  - If line topology and split worker range exists, CCL send kernel gets input base address at arg index 0.
  - Reference: `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/host/reduce_scatter_full_worker_grid.cpp:L917-L959`.
- Compile-time args/defines are derived from hashed determinants (pages, shapes, formats, topology, grid), so consistent across runs with identical hashed inputs.
