## Program cache review — data_movement/reshape_on_device

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra; single-core tiled and multi-core RM variants. Both supply override callbacks.
- Hashing: default determinants include old/new stick sizes, shapes, and grid split; runtime addresses excluded.
- Overrides update:
  - Single-core: reader arg[0]=src; writer arg[0]=dst.
  - Multi-core: recompute all per-core reader/writer runtime args on cache-hit and set them, based on current input/output shapes; addresses updated accordingly.
  - References: `device/reshape_program_factory.cpp` override lambdas for both variants.
