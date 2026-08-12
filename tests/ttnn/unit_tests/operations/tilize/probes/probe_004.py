import torch, ttnn


def crs(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in ranges})


dev = ttnn.open_device(device_id=0)
try:

    def info(name, t):
        mc = t.memory_config()
        print(
            f"--- {name}: shape={list(t.shape)} layout={t.layout} page_size={t.buffer_page_size()} "
            f"aligned_page={t.buffer_aligned_page_size()} num_pages={t.buffer_num_pages()}"
        )
        if mc.is_sharded():
            try:
                print("    shard_spec:", mc.shard_spec)
            except Exception as e:
                print("    shard_spec err", type(e).__name__)
            try:
                print("    nd_shard_spec:", mc.nd_shard_spec)
            except Exception as e:
                print("    nd err", type(e).__name__)
            cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(t)
            print("    cores:", [(c.x, c.y) for c in cores])
            cb = ttnn.cb_descriptor_from_sharded_tensor(0, t)
            fd = cb.format_descriptors[0]
            print("    cb total_size:", cb.total_size, "cb page:", fd.page_size, "cores:", cb.core_ranges)

    mc_h = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(((0, 0), (3, 0))), (128, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    x = torch.randn(1, 1, 512, 64).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc_h)
    info("HEIGHT rm", t)
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc_h)
    info("HEIGHT tile", tt)

    mc_w = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(((0, 0), (3, 0))), (64, 128), ttnn.ShardOrientation.ROW_MAJOR),
    )
    y = torch.randn(1, 1, 64, 512).bfloat16()
    t2 = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc_w)
    info("WIDTH rm", t2)
    t2t = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc_w)
    info("WIDTH tile", t2t)

    mc_b = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs(((0, 0), (1, 1))), (64, 64), ttnn.ShardOrientation.COL_MAJOR),
    )
    z = torch.randn(1, 1, 128, 128).bfloat16()
    t3 = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc_b)
    info("BLOCK COL rm", t3)
    t3t = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc_b)
    info("BLOCK COL tile", t3t)

    nd = ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(ttnn.Shape((2, 32, 64)), crs(((0, 0), (1, 0))), ttnn.ShardOrientation.ROW_MAJOR),
    )
    w = torch.randn(4, 32, 64).bfloat16()
    t4 = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=nd)
    info("ND rm", t4)
    t4t = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=nd)
    info("ND tile", t4t)

    nd2 = ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(ttnn.Shape((1, 1, 64, 64)), crs(((0, 0), (1, 1))), ttnn.ShardOrientation.ROW_MAJOR),
    )
    t5 = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=nd2)
    info("ND2 rm", t5)
    t5t = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=nd2)
    info("ND2 tile", t5t)
finally:
    ttnn.close_device(dev)
