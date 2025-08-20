Uniform OP — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/uniform/device/uniform_device_operation.hpp`
- `ttnn/cpp/ttnn/operations/uniform/device/uniform_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/uniform/device/uniform_device_operation.cpp`

Findings:
- Custom hash zeroes the runtime-only `seed` (`compute_program_hash` sets `seed = 0`), preventing cache fragmentation across seed changes.
- On cache hit, `override_runtime_arguments(...)` updates:
  - Compute kernel arg 0: per-core `seed` (adds core index when non-zero; generates fresh seed when zero).
  - Writer kernel arg 0: output buffer base address.
- Other runtime args (`from`, `to`, `tile_offset`, `units_per_core`) are derived from hashed properties (operation attributes, tensor shape/layout) and remain constant for a given cache key; not updating them on cache hit is correct.
- MemoryConfig and dtype affect compile-time paths (writer defines, DRAM vs L1) and are included in the hash, so cache entries won’t be incorrectly reused across those changes.

No issues identified with program cache usage for this OP.

Suggested tests (optional):
- Two-run cache test varying only tensor buffers and `seed` to confirm cache-hit path executes correctly and output differs as expected when `seed == 0` (new random seeds used on each run).
