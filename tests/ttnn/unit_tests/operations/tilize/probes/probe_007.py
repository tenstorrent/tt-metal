import torch, ttnn

dev = ttnn.open_device(device_id=0)
try:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    for shape, ss in [([4, 128, 128], [2, 64, 64]), ([3, 160, 160], [2, 64, 64])]:
        nd = ttnn.NdShardSpec(shard_shape=ss, grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR)
        mc = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd)
        t = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(torch.bfloat16)
        tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
        buf = tt.buffer()
        print(
            f"shape={shape} shard={ss}: num_pages={tt.buffer_num_pages()} page_size={tt.buffer_page_size()} "
            f"num_cores={nd.grid.num_cores()} addr={tt.buffer_address()}"
        )
        print(
            f"   buffer type={buf.buffer_type()} size_per_bank? total buffer size={buf.size()} page_size(buf)={buf.page_size()}"
        )
finally:
    ttnn.close_device(dev)
