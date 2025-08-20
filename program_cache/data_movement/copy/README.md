## Program cache review — data_movement/copy

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra, program created for interleaved vs sharded and tiled vs RM, with override callback provided.
- Hashing: default determinants include layout, sharding, dtype conversion, shapes/sizes; runtime addresses excluded.
- Override updates per-core reader/writer base addresses at index 0; sharded variants have additional sharding RT args but buffer addresses are still updated correctly.
  - Reference: `device/copy_program_factory.cpp:L181-L205`.
