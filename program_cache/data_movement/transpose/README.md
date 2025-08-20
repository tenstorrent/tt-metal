## Program cache review — data_movement/transpose

Status: Reviewed — no program cache issues found.

Findings
- New-infra op with multiple variants: CN, HC (tiled interleaved and general), WH (RM and tiled), and sharded versions for HC/WH; each defines an override.
- Hashing: default determinants include layout (RM vs TILE), padding flags/values, shapes, tile sizes, sharding specs, and grid splits; runtime addresses excluded.
- Overrides update per-core reader/writer addresses and recompute counts/indices based on current tensor shapes; sharded variants update dynamic CB addresses (and sizes) for input/output CBs.
  - Reference: `device/transpose_program_factory.cpp` override callbacks for CN/HC/WH (both interleaved and sharded paths).
