# DeepSeek V4 Decode Core Grids

## Goal

Allow all `matmul_decode` tensors in `layers.py` to use the regular row-wise
`ttnn.num_cores_to_corerangeset` core selection instead of forcing an exact
rectangle. This permits unrelated core ranges to reuse L1 addresses under
per-core allocation. Reshard sharded normalization inputs to rectangular grids
immediately before RMSNorm, whose sharded kernel requires rectangular geometry.

## Scope

- Update `LinearDecode` input-A, weight-B, and sharded-output grids.
- Update `BatchedLinearDecode` input-A and weight-B grids. Its output remains
  DRAM-interleaved as produced by the operation.
- Cover both `DeepSeekV4RMSNorm` and `_rms_norm_unweighted`.
- Preserve tensor shapes, shard shapes, width-sharded layout, row-major
  orientation, dtypes, and numerical behavior.
- Do not modify unrelated sharding helpers or other model components.

## Design

Add a small helper in `layers.py` that returns the regular row-wise core range
set for a requested core count:

```python
ttnn.num_cores_to_corerangeset(
    num_cores,
    device.compute_with_storage_grid_size(),
    row_wise=True,
)
```

Use this helper for every sharded memory configuration owned by
`LinearDecode` and `BatchedLinearDecode`.

Add a normalization-boundary helper that:

1. Returns an interleaved input unchanged.
2. Reads the existing shard specification from a sharded input.
3. Uses the same number of cores to construct an exact rectangular core range.
4. Creates a memory configuration preserving the existing memory layout,
   buffer type, shard shape, and orientation.
5. Calls `ttnn.to_memory_config` only when the existing grid differs from the
   rectangular target.

Both weighted and unweighted RMSNorm paths call this helper immediately before
`ttnn.rms_norm`. Existing logic that chooses whether an interleaved tensor
should first become width-sharded remains intact.

## Error Handling

The existing `rectangular_core_range_set` validation remains responsible for
rejecting a core count that cannot form an exact rectangle on the device.
Decode core selection continues to rely on TTNN validation for core counts
outside the device grid.

## Testing

Add focused tests that verify:

- Decode configurations use the regular row-wise core range set for
  non-rectangular core counts.
- A sharded normalization input is converted to a rectangular grid while
  preserving core count and shard geometry.
- An already rectangular or interleaved normalization input does not receive
  an unnecessary reshard.

Run the focused DeepSeek V4 tests and Python lint checks covering edited files.
