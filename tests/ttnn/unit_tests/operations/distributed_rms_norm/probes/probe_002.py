import ttnn, torch, time

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    t = ttnn.from_torch(
        torch.zeros(32, 2048),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    t0 = time.time()
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    print("PROBE time", time.time() - t0)
    print(
        "PROBE banks",
        mv.num_banks,
        mv.total_bytes_per_bank,
        mv.largest_contiguous_bytes_free_per_bank,
        mv.total_bytes_allocated_per_bank,
    )
    print("PROBE blocks", list(mv.block_table)[:6])
finally:
    ttnn.close_mesh_device(mesh)
