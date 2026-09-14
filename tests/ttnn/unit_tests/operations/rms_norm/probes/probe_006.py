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
    shape = (1, 1, 1024, 512)
    H, W = shape[-2], shape[-1]
    mc = ttnn.create_sharded_memory_config(
        shape=(H, 32),
        core_grid=grid16,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = (torch.arange(H).float().reshape(1, 1, H, 1) % 97 + 1.0).expand(*shape).contiguous().to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    b = derive_blocking(tx, None, g.x, g.y)
    print(
        f"regime={b.regime} Cw={b.num_w_splits} rect={b.rect_a}x{b.rect_b} B={b.block_rows} Rt={b.tensor_row_tiles} blocks={-(-b.tensor_row_tiles//b.block_rows)}"
    )
    Rt, Wt = H // 32, W // 32
    torch.set_printoptions(precision=2, linewidth=240, sci_mode=False)
    for trial in range(3):
        out = ttnn.to_torch(rms_norm(tx, memory_config=mc)).float()
        dev_map = (out.reshape(Rt, 32, Wt, 32) - 1.0).abs().amax(dim=(1, 3))  # (tile-row, core slice)
        bad = dev_map > 0.02
        print(f"--- trial {trial}: bad cells={int(bad.sum())} / {Rt*Wt}")
        if bad.any():
            rows_bad = bad.any(dim=1).nonzero().flatten().tolist()
            cols_bad = bad.any(dim=0).nonzero().flatten().tolist()
            print("   bad tile-rows:", rows_bad)
            print("   bad core slices:", cols_bad)
            r0 = rows_bad[0]
            print(f"   dev_map[{r0}] =", dev_map[r0].tolist())
            print(f"   out[{r0}*32, per slice first elem] =", out[0, 0, r0 * 32, ::32].tolist())
finally:
    ttnn.close_device(dev)
