import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    for tensor_shape, shard_shape, out_shard in [
        ([4, 128, 128], [2, 64, 64], None),
        ([3, 160, 160], [2, 64, 64], None),
        ([5, 4, 160, 160], [2, 3, 64, 96], None),
        ([23, 96, 160], [4, 64, 96], None),
        ([3, 160, 160], [2, 64, 64], [1, 64, 96]),
    ]:
        nd = ttnn.NdShardSpec(shard_shape=shard_shape, grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR)
        mc = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd)
        x = torch.rand(tensor_shape, dtype=torch.bfloat16)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
        print(
            f"IN  {tensor_shape} shard={shard_shape}: logical={list(t.shape)} padded={list(t.padded_shape)} "
            f"page={t.buffer_page_size()} aligned={t.buffer_aligned_page_size()} npages={t.buffer_num_pages()} "
            f"mem_layout={t.memory_config().memory_layout} sspec={list(t.memory_config().shard_spec.shape) if t.memory_config().shard_spec else None}"
        )
        oshape = out_shard if out_shard else shard_shape
        ondc = ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.L1,
            nd_shard_spec=ttnn.NdShardSpec(shard_shape=oshape, grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR),
        )
        spec = ttnn.TensorSpec(
            ttnn.Shape(tensor_shape),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            ondc.nd_shard_spec,
            ondc.buffer_type,
            ttnn.Tile([32, 32]),
        )
        o = ttnn.allocate_tensor_on_device(spec, device)
        print(
            f"OUT shard={oshape}: logical={list(o.shape)} padded={list(o.padded_shape)} page={o.buffer_page_size()} "
            f"npages={o.buffer_num_pages()} mem_layout={o.memory_config().memory_layout} "
            f"sspec={list(o.memory_config().shard_spec.shape) if o.memory_config().shard_spec else None}"
        )
        print()
finally:
    ttnn.close_device(device)
