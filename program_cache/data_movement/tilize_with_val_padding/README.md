## Program cache review — data_movement/tilize_with_val_padding

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra with single-core, multi-core interleaved, multi-core block, and sharded variants; each provides an override.
- Hashing: default determinants include pad value (packed), shapes, block/grid partitioning, and sharding; runtime addresses excluded.
- Overrides update:
  - Single-core and multi-core interleaved/block: reader/writer base addresses per core; pad value is compile-time or stable runtime arg derived from op attrs.
  - Sharded: updates dynamic CB addresses for input/output.
  - References: `device/tilize_with_val_padding_program_factory.cpp` override lambdas in each variant.
