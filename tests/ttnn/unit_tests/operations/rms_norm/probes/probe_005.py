import torch, ttnn
from eval.golden_tests.rms_norm.helpers import run_rms_norm, CheckOutputError
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    cases = [
        (
            ((128, 512),),
            dict(
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                gamma_mode="no_gamma",
                gamma_dtype="none",
                gamma_layout="none",
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                fp32_dest_acc_en=True,
            ),
        ),
        (
            ((128, 512),),
            dict(
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                gamma_mode="no_gamma",
                gamma_dtype="none",
                gamma_layout="none",
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                fp32_dest_acc_en=True,
            ),
        ),
        (
            ((128, 512),),
            dict(
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                gamma_mode="no_gamma",
                gamma_dtype="none",
                gamma_layout="none",
                memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
                fp32_dest_acc_en=True,
            ),
        ),
    ]
    for inputs, axes in cases:
        try:
            run_rms_norm(inputs, device=dev, **axes)
            print("== HARNESS PASS", inputs, axes["dtype"], axes["memory_layout"])
        except CheckOutputError as e:
            print("== HARNESS FAIL", inputs, axes["dtype"], axes["memory_layout"], str(e)[:160])
    # direct call, random input, same sharded geometry as the harness
    torch.manual_seed(0)
    x = torch.randn(128, 512)
    mc = auto_shard_config(
        [128, 512], ttnn.TensorMemoryLayout.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=dev
    )
    print(
        "shard spec:",
        mc.shard_spec.shape,
        [(r.start.x, r.start.y, r.end.x, r.end.y) for r in mc.shard_spec.grid.ranges()],
    )
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    out = ttnn.to_torch(rms_norm(tx, memory_config=mc)).float()
    ref = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)
    err = (out - ref).abs()
    print("direct random: max|diff|=", err.max().item())
    per_tile = err.reshape(4, 32, 16, 32).amax(dim=(1, 3))
    torch.set_printoptions(precision=3, linewidth=220)
    print(per_tile)
finally:
    ttnn.close_device(dev)
