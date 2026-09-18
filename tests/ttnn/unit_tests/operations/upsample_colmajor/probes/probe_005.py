import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)


def run(H, C, gx, gy, orient, shard_h):
    x = torch.randn(1, H, H, C).bfloat16()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    cw = C // (gx if orient == ttnn.ShardOrientation.ROW_MAJOR else gy)
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(grid, [shard_h, cw], orient)
    )
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    t = ttnn.to_memory_config(t, mc)
    y = ttnn.upsample(t, (2, 2))
    ref = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2).float(), scale_factor=2, mode="nearest").permute(
        0, 2, 3, 1
    )
    out = ttnn.to_torch(y).float()
    bad = (out - ref).abs().reshape(-1, C).max(dim=1).values > 0
    idx = bad.nonzero().flatten()
    print(
        f"PROBE H={H} C={C} grid={gx}x{gy} {str(orient).split('.')[-1]} shard=[{shard_h},{cw}] maxdiff={(out-ref).abs().max().item():.3f} bad_rows={int(bad.sum())}/{bad.numel()} first={idx[0].item() if len(idx) else -1} last={idx[-1].item() if len(idx) else -1}"
    )


try:
    run(64, 640, 8, 8, ttnn.ShardOrientation.ROW_MAJOR, 512)  # baseline (model path)
    run(64, 640, 8, 10, ttnn.ShardOrientation.COL_MAJOR, 512)  # COL_MAJOR, not ragged
    run(64, 640, 11, 10, ttnn.ShardOrientation.COL_MAJOR, 384)  # COL_MAJOR, ragged (target)
    run(32, 1280, 11, 10, ttnn.ShardOrientation.COL_MAJOR, 96)  # COL_MAJOR, ragged (target)
    run(64, 640, 10, 10, ttnn.ShardOrientation.ROW_MAJOR, 416)  # ROW_MAJOR ragged
finally:
    ttnn.close_device(dev)
