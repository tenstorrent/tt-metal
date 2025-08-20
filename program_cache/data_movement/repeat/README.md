## Program cache review — data_movement/repeat

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra with two RM variants (repeat last dim, repeat higher dim). Both return programs with overrides.
- Hashing: default determinants include repeat count and shapes; runtime addresses excluded.
- Overrides update reader runtime args for all cores with new input/output base addresses; other loop bounds remain constant.
  - References: `device/host/repeat_program_factory.cpp` overrides in lambdas for both variants.
