## Program cache review — data_movement/sharded/interleaved_to_sharded

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased style factory builds per-shard programs and provides override callback.
- Hashing: default determinants include shard spec, layout (tile/RM), dtype conversion flag, widths/heights, num_slices; runtime addresses excluded.
- Overrides update:
  - Reader arg[0] = input base; when running partial slicing, updates start index arg to recomputed `starting_idx_h`.
  - Writer: if DRAM, arg[0] = output base; if L1, updates dynamic CB address for output.
  - Reference: `device/interleaved_to_sharded_program_factory.cpp:L363-L402`.
