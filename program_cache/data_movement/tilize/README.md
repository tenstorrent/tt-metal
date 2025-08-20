## Program cache review — data_movement/tilize

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra with single-core, interleaved multi-core, block multi-core, and sharded variants; each returns a program with an override.
- Hashing: default determinants include tile shapes, block sizes, grid partitioning, and sharding; runtime addresses excluded.
- Overrides update:
  - Single-core: reader arg[0]=input; writer arg[0]=output.
  - Interleaved/block multi-core: update reader/writer base addresses for each core.
  - Sharded: update dynamic CB addresses for input/output.
  - References: `device/tilize_program_factory.cpp` override lambdas in each variant.
