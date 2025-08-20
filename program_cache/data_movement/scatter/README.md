## Program cache review — data_movement/scatter

Status: Reviewed — no program cache issues found.

Findings
- New-infra op with program factory storing reader/writer kernel ids and cores set; override updates per-core base addresses for input/index/src/output.
- Hashing: default determinants include chunk sizes, tile/page sizes, shapes, and grid split; runtime addresses excluded.
- Override updates:
  - Reader args[0..2] = input, index, source addresses; Writer arg[0] = output address.
  - Reference: `device/scatter_program_factory.cpp:L187-L199`.
