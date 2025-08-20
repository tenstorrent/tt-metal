ReduceScatterAsync (experimental CCL) — Program Cache Review

- Summary: New infra device operation with `ProgramWithCallbacks` builder in `reduce_scatter_async_program.cpp`.
  It defines a custom `compute_program_hash(...)` that appears to omit `num_links_preferred` from the key.

- Files reviewed:
  - `device/reduce_scatter_async_op.hpp/.cpp` — op struct, `create_program_at`, `compute_program_hash`
  - `device/reduce_scatter_async_program.cpp` — constructs worker command streams and builds override callback

- Program cache behavior:
  - Custom hash: `hash_operation<ReduceScatterAsync>(binary_op, scatter_dim, ring_size, topology, cluster_axis, input0 shape/layout/dtype/mem_config)`.
  - Potential omission: `num_links_preferred` is not included in the hash, yet it affects program construction
    (routing/fabric utilization) inside `build_reduce_scatter_async_program(...)`.
  - Override callback correctly updates per-core buffer base addresses for input, remote input, partial outputs, and
    final outputs. Argument ordering is centralized via overrider maps.

- Issue found: Under-keyed program hash (num_links omitted)
  - See `failures/test_reduce_scatter_async_cachehit_underkey_numlinks.py` which holds hashed properties constant and
    varies only `num_links` between runs. First run seeds cache; second run hits cache with a different link count →
    expected PCC failure on cache-hit.

- Suggested fix:
  - Include `num_links_preferred` (effective link count) in `compute_program_hash(...)`.
  - If `num_links_preferred` is computed dynamically via cluster axis path, include the resolved link count used to
    build the program.
