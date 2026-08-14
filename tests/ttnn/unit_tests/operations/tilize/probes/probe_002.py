import torch, ttnn


def crs(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in ranges})


device = ttnn.open_device(device_id=0)
try:
    cases = [
        (
            "HEIGHT_ROW",
            [1, 1, 512, 64],
            crs(((0, 0), (3, 0))),
            (128, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            "WIDTH_ROW",
            [1, 1, 64, 512],
            crs(((0, 0), (3, 0))),
            (64, 128),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            "BLOCK_COL",
            [1, 1, 128, 128],
            crs(((0, 0), (1, 1))),
            (64, 64),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            "HEIGHT_COL",
            [1, 1, 256, 64],
            crs(((0, 0), (0, 3))),
            (64, 64),
            ttnn.ShardOrientation.COL_MAJOR,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
    ]
    for name, shape, grid, sshape, orient, scheme in cases:
        spec = ttnn.ShardSpec(grid, sshape, orient)
        mc = ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, spec)
        t = ttnn.from_torch(
            torch.randn(shape).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=mc,
        )
        m = t.memory_config()
        print("=== ", name)
        print("  memory_layout:", m.memory_layout, "is_sharded:", m.is_sharded())
        print(
            "  shard_spec:",
            m.shard_spec,
        )
        print("  nd_shard_spec:", getattr(m, "nd_shard_spec", "ATTR-MISSING"))
        print("  buffer_page_size:", t.buffer_page_size(), "padded:", t.padded_shape, "elem:", t.element_size())
        cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(t)
        print("  optimal cores:", [(c.x, c.y) for c in cores])
        print(
            "  corerange_to_cores rowwise:", [(c.x, c.y) for c in ttnn.corerange_to_cores(grid, grid.num_cores(), True)]
        )
        print(
            "  corerange_to_cores colwise:",
            [(c.x, c.y) for c in ttnn.corerange_to_cores(grid, grid.num_cores(), False)],
        )
        ttnn.deallocate(t)

    # nd
    nd_grid = crs(((0, 0), (1, 1)))
    nd = ttnn.NdShardSpec(ttnn.Shape((1, 1, 64, 64)), nd_grid, ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.BufferType.L1, nd)
    t = ttnn.from_torch(
        torch.randn([1, 1, 128, 128]).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )
    m = t.memory_config()
    print("=== ND")
    print("  memory_layout:", m.memory_layout, "is_sharded:", m.is_sharded())
    print("  shard_spec:", m.shard_spec)
    print("  nd_shard_spec:", m.nd_shard_spec, m.nd_shard_spec.shard_shape, m.nd_shard_spec.orientation)
    print("  buffer_page_size:", t.buffer_page_size())
    print("  optimal cores:", [(c.x, c.y) for c in ttnn.get_optimal_worker_cores_for_sharded_tensor(t)])
finally:
    ttnn.close_device(device)
