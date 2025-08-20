## Program cache review — ccl/barrier

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra program created via `barrier_with_workers` and returned as `ProgramWithCallbacks` without a custom hash.
- Hashing uses default `hash_operation<Barrier>(operation, input_tensors)`. Determinants are topology and the input tensor properties; no per-run addresses are hashed.
- Runtime overrides: this program does not store an override callback; kernels use only device/mesh-derived addresses and semaphores created within `create_program_at(...)`, so there are no per-run buffer base addresses to update.
  - See `ttnn/cpp/ttnn/operations/ccl/barrier/device/host/barrier_full_worker_grid.cpp:L128-L135` where all runtime args are set during creation.
- Therefore, cache correctness is preserved; repeated runs with the same hashed inputs reuse the compiled program with no need for overrides.
