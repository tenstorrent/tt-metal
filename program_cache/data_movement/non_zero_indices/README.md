## Program cache review — data_movement/non_zero_indices

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased single-core path with override callback.
- Hashing: determinants include element size and aligned size; addresses excluded.
- Override updates input/output addresses and recomputed aligned/actual element counts.
  - Reference: `device/non_zero_indices_program_factory.cpp:L91-L111`.
