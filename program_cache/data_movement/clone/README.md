## Program cache review — data_movement/clone

Status: Reviewed — no program cache issues found.

Findings
- New-infra op with `ProgramFactory` and proper override to update reader/writer base addresses per core.
- Hashing: default determinants include layout (tiled vs RM), dtype conversion need, and shapes; runtime addresses excluded.
- Override updates:
  - Reader arg[0] = input buffer address; Writer arg[0] = output buffer address for each core.
  - Reference: `device/clone_program_factory.cpp:L168-L184`.
