## Program cache review — data_movement/slice

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra with RM tiled/strided and sharded variants; each returns a program and override callback.
- Hashing: default determinants include slice start/end/step, layout, sharding, shapes; runtime addresses excluded.
- Overrides update:
  - RM multi-core: recompute CB sizing and per-core reader/writer args; update CB sizes and set runtime args; writer arg[0] set to output base.
  - RM strided single-core: update input/output addresses.
  - Sharded: update dynamic CB addresses for input/output.
  - References: `device/slice_program_factory.cpp` override lambdas in each factory (`slice_rm_multi_core`, `slice_rm_strided_single_core_n_dims`, `slice_rm_multi_core_sharded`, and `slice_tile_multi_core` via helper).
