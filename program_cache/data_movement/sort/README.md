## Program cache review — data_movement/sort

Status: Reviewed — no program cache issues found.

Findings
- New-infra op with three program factories: single-row single-core, cross-core data exchange multi-core, and single-row multi-core. Each stores kernel ids/core ranges and provides overrides.
- Hashing: default determinants include sort direction/stability flags, shapes (Ht, Wt), tile sizes, and grid usage; runtime addresses excluded.
- Overrides update input/value/index base addresses per core; where lookup tensors are used, their buffer address is refreshed.
  - References: `device/sort_program_factory.cpp` override methods for each variant.
