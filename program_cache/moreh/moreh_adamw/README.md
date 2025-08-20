Moreh AdamW – Program Cache Review

Findings: Reviewed `create(...)`, `override_runtime_arguments(...)`, and hashing. No program-cache issues found.

- Override correctness:
  - Reader: updates all input buffer base addresses and runtime scalars `lr`, `beta1_exponent`, `beta2_exponent`, and `step` on cache-hit.
  - Writer: updates all output buffer base addresses.
  - Compute: updates `step` per core.
- Hashed vs runtime-only:
  - `compute_program_hash` excludes `step` and `lr` so cache keys are not fragmented by those runtime-only values.
  - Other attributes (`beta1`, `beta2`, `eps`, `weight_decay`, `amsgrad`) and tensor shapes/layouts are part of the hash; they are intentionally not overridden.
- Argument order: Override indices match `create(...)` for reader and writer kernels.
- Per-core coverage: Iteration and `core_group_1`/`core_group_2` handling mirror `create(...)`.

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/multi_core_program_factory.cpp` – runtime arg indices and per-core logic.
- `ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/moreh_adamw_device_operation.cpp` – custom program hash excluding `step` and `lr`.

Status: Reviewed – no issues identified.
