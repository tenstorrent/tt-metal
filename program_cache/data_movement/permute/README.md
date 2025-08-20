## Program cache review — data_movement/permute

Status: Reviewed — no program cache issues found.

Findings
- New-infra permute op with multiple tiled variants. Each factory provides an override that updates per-core reader/writer base addresses.
- Hashing: default determinants include permutation `dims`, tile shapes, padding flags/values, shapes; runtime addresses excluded.
- Overrides update:
  - Tile-invariant and tile-row-invariant: reader arg[0]=src, writer arg[0]=dst for all cores.
  - Generic tiled: same address updates; compute (transpose) kernels get updated counts via runtime args.
  - Reference: `ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_tiled_program_factory.cpp` override methods for each variant.
