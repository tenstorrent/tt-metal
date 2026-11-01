import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED
orig = pd.select_sharded_blocking


def spy(*a, **kw):
    blk = orig(*a, **kw)
    print(
        "CASE blk k=%d b=%d groups=%d rows=%s owner=%s"
        % (blk.k, blk.block_row_tiles, len(blk.groups), blk.groups[0][0].core_rows, blk.opts.owner_combine)
    )
    return blk


pd.select_sharded_blocking = spy
for shape in [(1, 1, 1024, 320), (1, 1, 320, 320), (7136, 736), (1, 1, 2048, 320)]:
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    mc = auto_shard_config(list(shape), WS, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = False
    out = ttnn.to_torch(rms_norm(xt, compute_kernel_config=cfg, memory_config=mc)).float()
    xf = x.float()
    exp = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
    err = (out - exp).abs().reshape(-1, shape[-1]).max(-1).values
    bad = (err > 0.1).nonzero().flatten()
    print(
        "CASE",
        shape,
        "bad rows:",
        len(bad),
        "first:",
        bad[:8].tolist(),
        "tilerows:",
        sorted(set((bad // 32).tolist()))[:20],
    )
ttnn.close_device(device)
