import torch, ttnn
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_program_descriptor as pd
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu import default_compute_kernel_config

device = ttnn.open_device(device_id=0)
try:
    emb, capacity, HIDDEN = 7168, 1024, 2048
    x = ttnn.from_torch(
        torch.zeros((1, 1, capacity, emb), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wg = ttnn.from_torch(
        torch.zeros((emb, HIDDEN), dtype=torch.bfloat16),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wu = ttnn.from_torch(
        torch.zeros((emb, HIDDEN), dtype=torch.bfloat16),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wd = ttnn.from_torch(
        torch.zeros((HIDDEN, emb), dtype=torch.bfloat16),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cnt = ttnn.from_torch(
        torch.zeros(256, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    idx = ttnn.from_torch(
        torch.zeros(8, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, capacity, emb]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    g = device.compute_with_storage_grid_size()
    mb = pd.make_mailbox(device, int(g.x) * int(g.y))
    d = pd.create_program_descriptor(
        x,
        wg,
        wu,
        wd,
        cnt,
        idx,
        out,
        mb,
        local_expert_id=3,
        input_m_tiles=capacity // 32,
        compute_kernel_config=default_compute_kernel_config(),
    )
    total = 0
    for cb in d.cbs:
        total += cb.total_size
    print("num cbs", len(d.cbs), "L1 total bytes", total, "KB", total / 1024.0)
    print("num semaphores", len(d.semaphores))
    print("reader ct len", len(d.kernels[0].compile_time_args))
    print("writer ct len", len(d.kernels[1].compile_time_args))
    print("compute ct len", len(d.kernels[2].compile_time_args))
    print("reader rt len (0,0)", len(d.kernels[0].runtime_args[0][0]))
    print(
        "x page",
        x.buffer_page_size(),
        "cnt page",
        cnt.buffer_aligned_page_size(),
        "idx page",
        idx.buffer_aligned_page_size(),
    )
finally:
    ttnn.close_device(device)
