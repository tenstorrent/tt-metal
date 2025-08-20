Rand operation program cache review

Status: Reviewed — no program cache issues found.

Summary
- Program factory: `ttnn/cpp/ttnn/operations/rand/device/rand_program_factory.cpp`.
- Override updates per-run addresses and RNG seed only:
  - Compute kernel arg[0]: per-core seed uses `attributes.seed + core_index` when `seed != 0`, otherwise a new random seed via `get_random_seed()`.
  - Writer kernel arg[0]: output buffer base address.
- Other compute/runtime args (from/to, tile_offset, units_per_core) are derived from hashed attributes and shape at creation time and remain valid on cache hits.
- Custom hash zeros the seed so repeated calls with `seed=0` reuse the same cached program but produce fresh random numbers via override.

Key references
- `rand_device_operation.cpp` — compute_program_hash zeros `seed` before hashing attributes, preventing cache fragmentation.
- `rand_program_factory.cpp` — override updates only non-hashed runtime values (output addr, seed), preserving correctness.

Recommendation
- No changes required.
