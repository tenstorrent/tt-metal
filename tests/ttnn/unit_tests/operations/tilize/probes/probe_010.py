import torch, ttnn

device = ttnn.open_device(device_id=0)
try:

    def show(name, t):
        mc = t.memory_config()
        print(f"--- {name}: shape={list(t.shape)} padded={list(t.padded_shape)} layout={t.layout}")
        print("   mem_layout=", mc.memory_layout, "buf=", mc.buffer_type)
        ss = mc.shard_spec
        nd = mc.nd_shard_spec
        print("   shard_spec=", (list(ss.shape), ss.orientation, ss.grid.num_cores()) if ss else None)
        print("   nd_spec=", (list(nd.shard_shape), nd.orientation, nd.grid.num_cores()) if nd else None)
        print(
            "   page_size=",
            t.buffer_page_size(),
            "aligned_page=",
            t.buffer_aligned_page_size(),
            "num_pages=",
            t.buffer_num_pages(),
            "is_sharded=",
            t.is_sharded(),
        )
        try:
            cb = ttnn.cb_descriptor_from_sharded_tensor(0, t)
            print(
                "   CB total_size=",
                cb.total_size,
                "page=",
                cb.format_descriptors[0].page_size,
                "cores=",
                cb.core_ranges.num_cores(),
                "addr=",
                cb.buffer_address(),
            )
        except Exception as e:
            print("   CB err:", e)

    grid4 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    grid2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    grid22 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})

    mc_h = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid4, (128, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    x = torch.randn(1, 1, 512, 64).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc_h)
    show("height RM in", t)

    spec = ttnn.TensorSpec(
        ttnn.Shape([1, 1, 512, 64]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mc_h.memory_layout,
        mc_h.shard_spec,
        mc_h.buffer_type,
        ttnn.Tile([32, 32]),
    )
    o = ttnn.allocate_tensor_on_device(spec, device)
    show("height TILE out", o)

    nd = ttnn.NdShardSpec(ttnn.Shape([1, 1, 64, 64]), grid2, ttnn.ShardOrientation.ROW_MAJOR)
    mc_nd = ttnn.MemoryConfig(ttnn.BufferType.L1, nd)
    print("nd cfg: layout=", mc_nd.memory_layout, "shard_spec=", mc_nd.shard_spec, "nd=", mc_nd.nd_shard_spec)
    x2 = torch.randn(1, 1, 128, 64).bfloat16()
    t2 = ttnn.from_torch(x2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc_nd)
    show("nd RM in", t2)
    spec2 = ttnn.TensorSpec(
        ttnn.Shape([1, 1, 128, 64]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mc_nd.memory_layout,
        mc_nd.shard_spec,
        mc_nd.buffer_type,
        ttnn.Tile([32, 32]),
    )
    try:
        o2 = ttnn.allocate_tensor_on_device(spec2, device)
        show("nd TILE out", o2)
    except Exception as e:
        print("nd out alloc err:", type(e).__name__, e)

    mc_b = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid22, (64, 64), ttnn.ShardOrientation.COL_MAJOR),
    )
    x3 = torch.randn(1, 1, 128, 128).bfloat16()
    t3 = ttnn.from_torch(x3, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc_b)
    show("block COL RM in", t3)
    print("max_l1_unreserved=", ttnn.get_max_worker_l1_unreserved_size())
    print("corerange grid22 rowwise:", [(c.x, c.y) for c in ttnn.corerange_to_cores(grid22, 4, True)])
    print("corerange grid22 colwise:", [(c.x, c.y) for c in ttnn.corerange_to_cores(grid22, 4, False)])
finally:
    ttnn.close_device(device)
