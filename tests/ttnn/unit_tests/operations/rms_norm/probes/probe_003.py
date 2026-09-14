import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import derive_blocking

dev = ttnn.open_device(device_id=0)
try:
    g = dev.compute_with_storage_grid_size()

    # per-row scale: row r (global) = r+1 -> rms(row) = r+1 -> output must be 1.0 everywhere
    def run(shape, mem_config, label):
        H, W = shape[-2], shape[-1]
        x = (torch.arange(H).float().reshape(1, 1, H, 1) + 1.0).expand(*shape).contiguous().to(torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem_config)
        b = derive_blocking(tx, None, g.x, g.y)
        print(
            f"== {label} {shape}: regime={b.regime} Cw={b.num_w_splits} rect={b.rect_a}x{b.rect_b} groups={b.num_row_groups} B={b.block_rows} Wc_max={b.core_w_tiles_max}"
        )
        out = ttnn.to_torch(
            rms_norm(tx, memory_config=mem_config if mem_config != ttnn.DRAM_MEMORY_CONFIG else None)
        ).float()
        per_row = out[0, 0, :, 0]
        bad = (per_row - 1.0).abs() > 0.02
        print("   rows != 1.0:", bad.nonzero().flatten().tolist()[:40])
        for r in [0, 1, 31, 32, 33, 63]:
            if r < H:
                print(f"   row {r}: out[:4]={out[0,0,r,:4].tolist()}  out[W-2:]={out[0,0,r,W-2:].tolist()}")

    sharded = ttnn.create_sharded_memory_config(
        shape=(64, 64),
        core_grid=ttnn.CoreGrid(x=8, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    run((1, 1, 64, 512), sharded, "R3 sharded 8 cores Rt=2")
    run((1, 1, 640, 4096), ttnn.DRAM_MEMORY_CONFIG, "R2 interleaved rows>1/block")
    run((1, 1, 64, 512), ttnn.DRAM_MEMORY_CONFIG, "interleaved same shape")
finally:
    ttnn.close_device(dev)
