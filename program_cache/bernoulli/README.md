## Program cache review — bernoulli

Status: Reviewed — no program cache issues found.

Key findings
- Hashing: Custom hash zeroes the seed so it does not fragment the cache while other determinants remain hashed.
  - Reference: `ttnn/cpp/ttnn/operations/bernoulli/device/bernoulli_device_operation.cpp:L69-L73`.
- Program creation: Compile-time args and defines are derived from hashed properties (input/output dtype, memory configuration, DRAM/L1 placement via tensor/memory config).
  - Reader/compute/writer kernel compile-time args and writer defines: `ttnn/cpp/ttnn/operations/bernoulli/device/bernoulli_program_factory.cpp:L66-L86` and `L88-L106`.
- Runtime overrides on cache-hit correctly update only non-hashed values:
  - Reader kernel: updates input buffer base address at arg index 0.
  - Compute kernel: updates random seed at arg index 0 (per core); `tile_offset` and `units_per_core` remain unchanged across runs for identical hashed inputs.
  - Writer kernel: updates output buffer base address at arg index 0.
  - Reference (create-time arg order and override): `ttnn/cpp/ttnn/operations/bernoulli/device/bernoulli_program_factory.cpp:L119-L131` and `L142-L169`.

Notes
- `tile_offset` and `units_per_core` are derived from output volume and core split; for identical shapes/device grid they are stable and correctly not overridden.
- The device cache is per-device; grid selection is device-derived and consistent for cache hits on the same device.

Suggested tests
- None required; behavior appears correct given hashing and override paths above.
