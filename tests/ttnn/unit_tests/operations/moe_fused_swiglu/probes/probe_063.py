import os, torch, ttnn
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_program_descriptor as D
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu import default_compute_kernel_config

EMB, CAP, HID = 7168, 5120, 2048
dev = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(
        torch.zeros((1, 1, CAP, EMB), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gu_mc, dn_mc = D.weight_memory_configs(dev, EMB, HID)
    mk = lambda s, mc: ttnn.from_torch(
        torch.zeros(s, dtype=torch.bfloat16),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=mc,
    )
    wg, wu, wd = mk((EMB, HID), gu_mc), mk((EMB, HID), gu_mc), mk((HID, EMB), dn_mc)
    u32 = lambda t: ttnn.from_torch(
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    counts, idx = u32(torch.zeros(256, dtype=torch.int32)), u32(torch.zeros(8, dtype=torch.int32))
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, CAP, EMB]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    g = dev.compute_with_storage_grid_size()
    mbox = D.make_mailbox(dev, int(g.x) * int(g.y))
    d = D.create_program_descriptor(
        x,
        wg,
        wu,
        wd,
        counts,
        idx,
        out,
        mbox,
        local_expert_id=3,
        input_m_tiles=CAP // 32,
        compute_kernel_config=default_compute_kernel_config(),
    )
    names = {v: k for k, v in vars(D).items() if k.startswith("CB_") and isinstance(v, int)}
    tot = 0
    rows = []
    for cb in d.cbs:
        idxs = [fd.buffer_index for fd in cb.format_descriptors]
        rows.append((names.get(idxs[0], f"cb{idxs[0]}"), cb.total_size))
        tot += cb.total_size
    rows.sort(key=lambda r: -r[1])
    print(
        f"### M_BLOCK={D.M_BLOCK} X_KB={os.environ.get('MOE_SWIGLU_X_KB','0')} "
        f"DEPTH_X={D.DEPTH_X} GU_KCHUNKS={D.GU_KCHUNKS} GRID={os.environ.get('MOE_SWIGLU_GRID','')}"
    )
    for n, s in rows:
        if s > 4096:
            print(f"  {n:22s} {s:9,d}")
    print(f"  {'TOTAL':22s} {tot:9,d}   free_of_1572864 = {1572864-tot:,d}")
finally:
    ttnn.close_device(dev)
