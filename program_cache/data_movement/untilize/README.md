## Program cache review — data_movement/untilize

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra with multiple multi-core strategies (sub-core grids, column-parallel, block, generic), plus single-core and sharded variants; all provide overrides.
- Hashing: default determinants include pack-vs-untilize selection, tile/block sizes, sharding and grid partitioning; runtime addresses excluded.
- Overrides update reader/writer base addresses per core, or dynamic CB addresses when using sharded CBs. Coordinator/lookup tensors are recomputed on creation and their addresses are updated when applicable.
  - Reference: `device/untilize_program_factory.cpp` override lambdas for each variant.
