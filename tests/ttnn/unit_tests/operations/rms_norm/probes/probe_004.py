import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import derive_blocking

dev = ttnn.open_device(device_id=0)
try:
    g = dev.compute_with_storage_grid_size()

    def grid16():
        return ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(2, 1)),
            }
        )

    def run(shape, shard_shape, grid, dtype, label):
        H, W = shape[-2], shape[-1]
        x = (torch.arange(H).float().reshape(1, 1, H, 1) + 1.0).expand(*shape).contiguous()
        tdt = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
        mc = ttnn.create_sharded_memory_config(
            shape=shard_shape,
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tx = ttnn.from_torch(x.to(tdt), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        b = derive_blocking(tx, None, g.x, g.y)
        print(
            f"== {label} {shape}: regime={b.regime} Cw={b.num_w_splits} rect={b.rect_a}x{b.rect_b} B={b.block_rows} Wc={b.core_w_tiles_max} active={b.groups[0].num_active}"
        )
        out = ttnn.to_torch(rms_norm(tx, memory_config=mc)).float()
        Rt, Wt = H // 32, W // 32
        wc = shard_shape[1] // 32
        # per (tile-row, core-slice) mean; expected 1.0
        m = out.reshape(Rt, 32, Wt // wc, wc * 32).mean(dim=(1, 3))
        torch.set_printoptions(precision=3, linewidth=200)
        print(m)

    run((1, 1, 128, 512), (128, 32), grid16(), ttnn.float32, "16 cores 13+3, fp32")
    run((1, 1, 128, 512), (128, 32), grid16(), ttnn.bfloat16, "16 cores 13+3, bf16")
    run((1, 1, 128, 512), (128, 32), ttnn.CoreGrid(x=8, y=2), ttnn.bfloat16, "16 cores 8x2 rect, bf16")
    run((1, 1, 128, 512), (128, 64), ttnn.CoreGrid(x=8, y=1), ttnn.bfloat16, "8 cores 8x1 rect, bf16")
finally:
    ttnn.close_device(dev)
