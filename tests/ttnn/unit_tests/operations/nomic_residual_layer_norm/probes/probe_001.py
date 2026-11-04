import torch, ttnn, math

device = ttnn.open_device(device_id=0)
try:
    grid = device.compute_with_storage_grid_size()
    print("grid", grid.x, grid.y)
    for B, S, rs in [(8, 512, 4), (2, 512, 1), (2, 500, 1), (4, 512, 2)]:
        St = math.ceil(S / 32)
        groups = B * math.ceil(St / rs)
        n = 3 * groups
        cores = ttnn.num_cores_to_corerangeset(n, grid, True)
        spec = ttnn.NdShardSpec(ttnn.Shape([1, 1, rs * 32, 256]), cores, orientation=ttnn.ShardOrientation.ROW_MAJOR)
        mc = ttnn.MemoryConfig(ttnn.BufferType.L1, spec)
        t = torch.randn(B, 1, S, 768).to(torch.bfloat16)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        back = ttnn.to_torch(x)
        print(B, S, rs, "cores", n, "roundtrip_equal", torch.equal(back, t), "mc_equal", x.memory_config() == mc)
        print(
            "  layout", x.memory_config().memory_layout, "legacy shard_spec", x.memory_config().shard_spec is not None
        )
        cb = ttnn.cb_descriptor_from_sharded_tensor(0, x, core_ranges=cores) if True else None
        print("  cb total_size", cb.total_size, "expected", rs * 8 * 2048)
        ttnn.deallocate(x)
finally:
    ttnn.close_device(device)
