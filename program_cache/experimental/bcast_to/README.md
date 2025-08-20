## Program cache review: experimental/bcast_to

OP reviewed
- `ttnn::prim::bcast_to` implemented by `BcastToOperation::BcastToTileFactory`.

Creation path summary
- Computes output tile count and splits work across `compute_with_storage_grid_size`.
- Reader runtime args (13): `[0]=input.addr, [1]=start_n, [2]=start_c, [3]=start_t, [4]=start_th, [5]=start_tw, [6]=tiles_per_core, [7]=iHt*iWt*iC*(iN>1), [8]=iHt*iWt*(iC>1), [9]=oN, [10]=oC, [11]=oHt, [12]=oWt`.
- Writer runtime args (14): `[0]=output.addr, [1..12]=same as reader indices/dims, [13]=start_tile_id`.
- Compute runtime args (12): indices/dims excluding addresses and `start_tile_id`.
- Compile-time args include DRAM flags and CB indices.

Override behavior (cache-hit path)
- Recomputes per-core start indices and counts identically to creation and updates runtime args for all active cores.
- Updates input/output base addresses; preserves hashed properties (shape, layout, memory_config) via selection and validation.
- Uses stored `compute_with_storage_grid_size` ensuring per-core coverage matches creation.

Findings
- No missing runtime-argument updates detected. Address, index, and count parameters are correctly refreshed.
- Hashing/selection: default hash with fixed factory; differing shapes/layouts or memory configs would change selection and miss cache, which is correct.

Conclusion
- No program cache issues found. No failing cache test required.
