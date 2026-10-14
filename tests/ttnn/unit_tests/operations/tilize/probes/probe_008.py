import torch, ttnn
from ttnn.operations.tilize.tilize_program_descriptor import create_program_descriptor
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
for shape, layout in [
    ((1, 1, 32, 1024), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ((1, 1, 1024, 32), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ((1, 1, 256, 256), ttnn.TensorMemoryLayout.BLOCK_SHARDED),
]:
    H, W = shape[-2:]
    if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(W // 32 - 1, H // 32 - 1))})
    else:
        n = H // 32 if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED else W // 32
        grid = ttnn.num_cores_to_corerangeset(n, dev.compute_with_storage_grid_size(), row_wise=True)
    mem = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR))
    for ih, oh in [(32, 16), (1, 32)]:
        t = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
            tile=ttnn.Tile([ih, 32]),
        )
        spec = ttnn.TensorSpec(
            ttnn.Shape(shape),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            layout,
            mem.shard_spec,
            buffer_type=ttnn.BufferType.L1,
            tile=ttnn.Tile([oh, 32]),
        )
        o = ttnn.allocate_tensor_on_device(spec, dev)
        pd = create_program_descriptor(t, o)
        print(
            "RES",
            layout,
            ih,
            oh,
            sorted(set(f.buffer_index for c in pd.cbs for f in c.format_descriptors)),
            len(pd.kernels),
        )
ttnn.close_device(dev)
