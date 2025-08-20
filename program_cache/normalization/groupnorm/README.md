GroupNorm — Program Cache Review

- Status: Reviewed — no program-cache issues found (old infra callback-based)

Summary
- This OP uses the legacy type-erased infra and returns `ProgramWithCallbacks`. The override path is handled via a stored callback that updates buffer addresses for input(s) and output per core in both unsharded and sharded multi-core variants.
- Creation wires reader/writer kernels with compile-time args derived from tensor accessors; on cache-hit the callback is invoked to update all DRAM base addresses and any per-core offsets, matching the indices/order used at creation.

Key references
- Entry and program selection: `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_op.cpp`.
- Multi-core sharded program wiring (addresses, per-core iteration): `device/multi_core/groupnorm_op_multi_core.cpp`.

Notes
- The code ensures per-core address updates using the same core iteration used in create. No runtime-only scalars are compiled in without override coverage.
