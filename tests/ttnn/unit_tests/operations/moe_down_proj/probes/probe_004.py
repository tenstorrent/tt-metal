import torch, ttnn
from ttnn.operations.moe_down_proj import moe_down_proj_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    base = ttnn.get_allocator_base_address(device, ttnn.BufferType.L1)
    v = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    print(
        "base",
        base,
        "banks",
        v.num_banks,
        "total/bank",
        v.total_bytes_per_bank,
        "free",
        v.total_bytes_free_per_bank,
        "largest",
        v.largest_contiguous_bytes_free_per_bank,
    )
    print([b for b in v.block_table][:5])
    R = 8416
    x = ttnn.from_torch(
        torch.zeros(1, 1, R, 3072),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R, 768]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    v = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    live = [int(b["address"]) for b in v.block_table if b["allocated"] == "yes"]
    print(
        "x addr",
        x.buffer_address(),
        "y addr",
        y.buffer_address(),
        "min live",
        min(live),
        "free below",
        min(live) - base,
    )
    print("lowest_occ", getattr(device, "lowest_occupied_compute_l1_address", None))
    grid = device.compute_with_storage_grid_size()
    print("grid", grid, "budget const", pd.cb_budget_bytes(grid.x * grid.y, False, False))
    lay = pd.select_layout(grid, R // 32)
    for k in (8, 6, 4, 3, 2):
        for w in (1, 2, 3, 4):
            rows = pd._div_up(pd._div_up(R // 32, w), lay.num_m_groups)
            print(k, w, rows, pd.footprint_bytes(rows, k, lay.core_n_tiles, 32, w))
finally:
    ttnn.close_device(device)
