import ttnn, torch

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    t = ttnn.from_torch(
        torch.zeros(32, 2048),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    mv = (
        ttnn._ttnn.device.get_memory_view(mesh, ttnn.BufferType.L1)
        if hasattr(ttnn._ttnn.device, "get_memory_view")
        else None
    )
    print("PROBE has get_memory_view", mv is not None, [n for n in dir(ttnn) if "memory_view" in n.lower()])
    print("PROBE base", ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1), "addr", t.buffer_address())
    if mv is not None:
        print("PROBE banks", mv.num_banks, mv.total_bytes_per_bank, mv.largest_contiguous_bytes_free_per_bank)
        print("PROBE blocks", list(mv.block_table)[:5])
finally:
    ttnn.close_mesh_device(mesh)
