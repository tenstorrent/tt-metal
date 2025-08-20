Grid Sample program cache review

Status: Reviewed — no program cache issues found.

Summary
- Factory: `device/grid_sample_program_factory.cpp` (old infra ProgramWithCallbacks).
- Override updates input, grid, and output buffer base addresses for reader and writer kernels.
- Work partitioning and stick counts are computed from hashed shapes/layouts and remain stable for cache hits.
- Compute kernel uses compile-time args only; no runtime args to override there.

Recommendation
- No changes required.
