import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    base = (
        ttnn.device.GetAllocatorBaseAddress(device, ttnn.BufferType.L1)
        if hasattr(ttnn.device, "GetAllocatorBaseAddress")
        else None
    )
    v = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    print(
        "PROBE base",
        base,
        "total",
        v.total_bytes_per_bank,
        "alloc",
        v.total_bytes_allocated_per_bank,
        "largest free",
        v.largest_contiguous_bytes_free_per_bank,
        "banks",
        v.num_banks,
    )
    print("PROBE table0", v.block_table[:5])
    t1 = ttnn.from_torch(
        torch.randn(16384, 512),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    t2 = ttnn.from_torch(
        torch.randn(1024, 512),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    t1.deallocate()
    v = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    print(
        "PROBE after: total",
        v.total_bytes_per_bank,
        "alloc",
        v.total_bytes_allocated_per_bank,
        "largest free",
        v.largest_contiguous_bytes_free_per_bank,
    )
    print("PROBE table", v.block_table)
    print("PROBE t2 addr", t2.buffer_address())
    print(
        "PROBE grid",
        device.compute_with_storage_grid_size(),
        [a for a in dir(ttnn.device) if "ddress" in a or "Alloc" in a],
    )
finally:
    ttnn.close_device(device)
