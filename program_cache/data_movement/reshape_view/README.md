## Program cache review — data_movement/reshape_view

Status: Reviewed — no program cache issues found.

Findings
- New-infra tiled reshape view using a host-computed mapping tensor. Program includes override callback.
- Hashing: default determinants include mapping tensor shape (derived from input/output shapes), tile/face shapes; runtime addresses excluded.
- Overrides update reader arg[0] with input address and writer arg[0] with output address for each utilized core. Mapping tensor device buffer is cached in closure so it’s stable across runs.
  - Reference: `device/host/reshape_tiled_program_factory.cpp:L418-L441`.
