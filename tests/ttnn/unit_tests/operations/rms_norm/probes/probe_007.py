import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import derive_blocking

dev = ttnn.open_device(device_id=0)
try:
    g = dev.compute_with_storage_grid_size()
    grid16 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(2, 1)),
        }
    )
    grid8 = ttnn.CoreGrid(x=8, y=1)

    def run(shape, grid, shard_w, label, show_rows=None):
        H, W = shape[-2], shape[-1]
        mc = ttnn.create_sharded_memory_config(
            shape=(H, shard_w),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        xr = torch.arange(H).float() % 97 + 1.0
        x = xr.reshape(1, 1, H, 1).expand(*shape).contiguous().to(torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        b = derive_blocking(tx, None, g.x, g.y)
        Rt, Wt = H // 32, W // 32
        nb = -(-Rt // b.block_rows)
        out = ttnn.to_torch(rms_norm(tx, memory_config=mc)).float()
        dev_map = (out.reshape(Rt, 32, Wt, 32) - 1.0).abs().amax(dim=(1, 3))
        bad_rows = (dev_map > 0.02).any(dim=1).nonzero().flatten().tolist()
        print(
            f"== {label} {shape}: Cw={b.num_w_splits} B={b.block_rows} Rt={Rt} blocks={nb} -> bad tile-rows {bad_rows}"
        )
        if show_rows is not None and bad_rows:
            tr = bad_rows[0]
            col0 = out[0, 0, tr * 32 : (tr + 1) * 32, 0]
            xs = xr[tr * 32 : (tr + 1) * 32]
            implied = xs / col0  # x * rstd_used == out  -> 1/rstd_used = x/out = implied rms actually used
            print("   x rows     :", xs.int().tolist())
            print("   out col0   :", [round(v, 3) for v in col0.tolist()])
            print("   implied rms:", [round(v, 2) for v in implied.tolist()])

    run((1, 1, 1024, 512), grid16, 32, "16c 13+3", show_rows=True)
    run((1, 1, 640, 512), grid16, 32, "16c 13+3 Rt=20")
    run((1, 1, 512, 512), grid16, 32, "16c 13+3 Rt=16")
    run((1, 1, 1024, 512), grid8, 64, "8c 8x1 Rt=32")
    run((1, 1, 1024, 256), grid8, 32, "8c 8x1 W=256 Rt=32")
finally:
    ttnn.close_device(dev)
