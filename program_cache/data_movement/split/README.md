## Program cache review — data_movement/split

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased tiled split into two outputs; program factory provides override callback.
- Hashing: default determinants include split axis/strategy via compile-time args and shapes; runtime addresses excluded.
- Overrides update reader input base and both writer output base addresses for each core.
  - Reference: `device/split_program_factory.cpp` override lambda in `split_last_dim_two_chunks_tiled`.
