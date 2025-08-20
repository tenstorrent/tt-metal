Sampling — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_op.hpp`
- `ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_op.cpp`
- `ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_program_factory.cpp`

Findings:
- Uses the old type-erased infra with an override callback.
- Override updates runtime-only buffer base addresses on cache hits for all involved tensors:
  - Reader: input values and input indices buffers.
  - Writer (per-core): output, temp, k, and p buffers.
- Random seed is compiled into compute args from op attribute `seed` when provided; default hashing includes attributes and tensor args, so cache entries differ per-seed as expected.
- Grid selection and CB sizes depend on shapes and optional sub_core_grids; these are part of default hashing via op attributes/tensors.

No program-cache issues identified.

Suggested optional tests:
- Two-run cache test reusing same seed and shapes but reallocating all buffers; expect one program cache entry and matching results across runs.
- Additional variant with different seed to confirm separate cache entries are created.
