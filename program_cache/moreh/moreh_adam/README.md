Moreh Adam – Program Cache Review

Findings: Reviewed `override_runtime_arguments` and hashing. No program-cache issues found.

- Override updates all runtime-only values on cache hit:
  - Reader kernel: updates input/output buffer base addresses and runtime scalars `lr` and `step`.
  - Writer kernel: updates output buffer base addresses.
  - Compute kernel: updates `step` per core.
- Hashed vs runtime-only:
  - `compute_program_hash` zeroes out `step` and `lr`, so cache keys exclude these runtime-only values. Other attributes (`beta1`, `beta2`, `eps`, `weight_decay`, `amsgrad`, shapes/layouts/memory) remain in the hash and are not updated at override time.
- Argument order consistency: Override indices match those used during `create(...)` for reader and writer kernels.
- Per-core coverage: Iterates all cores using the same `core_group_1`/`core_group_2` selection as in `create(...)`.

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/moreh_adam_program_factory.cpp` – create/override logic and runtime-arg indices.
- `ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/moreh_adam_device_operation.cpp` – custom `compute_program_hash` excluding `step` and `lr`.

Status: Reviewed – no issues identified.
