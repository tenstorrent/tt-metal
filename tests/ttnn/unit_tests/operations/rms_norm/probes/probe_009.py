import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    grid16 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(2, 1)),
        }
    )
    shape = (1, 1, 640, 512)
    H, W = 640, 512
    mc = ttnn.create_sharded_memory_config(
        shape=(H, 32),
        core_grid=grid16,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    xr = torch.arange(H).float() % 97 + 1.0
    x = xr.reshape(1, 1, H, 1).expand(*shape).contiguous().to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    out = ttnn.to_torch(rms_norm(tx, memory_config=mc)).float()
    tr = 10
    col0 = out[0, 0, tr * 32 : (tr + 1) * 32, 0]
    xs = xr[tr * 32 : (tr + 1) * 32]
    implied = xs / col0
    print("row-in-tile | x | out | implied rms (should == x)")
    for i in range(32):
        flag = "" if abs(col0[i].item() - 1.0) < 0.02 else "  <-- BAD"
        print(f"{i:2d} | {int(xs[i]):3d} | {col0[i].item():.4f} | {implied[i].item():.2f}{flag}")
    # also: is the whole 32x512 tile-row identical across columns?
    same = (out[0, 0, tr * 32 : (tr + 1) * 32, :] == out[0, 0, tr * 32 : (tr + 1) * 32, 0:1]).all().item()
    print("all columns identical within the tile-row:", same)
finally:
    ttnn.close_device(dev)
