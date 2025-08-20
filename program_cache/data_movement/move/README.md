## Program cache review — data_movement/move

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra; two main variants:
  - Overlap-safe interleaved/tiled path updating reader args[0]=src, [1]=dst per core.
  - Sharded path using dynamic CB addresses for src/dst and recomputing runtime args that depend on chunk size from buffer addresses.
- References:
  - Overlap path override: `device/move_program_factory.cpp:L181-L199`.
  - Sharded path override: `device/move_program_factory.cpp:L262-L293`.
