import ttnn, torch

d = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
try:
    print("BASE", ttnn.get_allocator_base_address(d, ttnn.BufferType.L1))
    v = ttnn.get_memory_view(d, ttnn.BufferType.L1)
    print("VIEW", v.total_bytes_per_bank, v.num_banks)
    for b in list(v.block_table)[:4]:
        print("BLK", dict(b))
    t = ttnn.from_torch(torch.zeros(64, 64), layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.get_memory_view(d, ttnn.BufferType.L1)
    for b in list(v.block_table)[:4]:
        print("BLK", dict(b))
    print("UNRES", ttnn.get_max_worker_l1_unreserved_size())
except Exception as e:
    print("ERR", e)
ttnn.close_mesh_device(d)
