import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    o = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    return o * g.to(torch.float32).reshape(-1) if g is not None else o


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
TD = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}
cases = [
    ((1, 1, 32, 4096), ttnn.bfloat16, True),
    ((1, 1, 32, 8192), ttnn.bfloat16, True),  # L1-tight: 1 core, 256 tiles in + out
    ((128, 8192), ttnn.bfloat16, True),  # 4 cores, same shard size
    ((1, 1, 8192, 1024), ttnn.bfloat16, True),  # rows_max = 3 tile-rows per core
    ((2, 4, 128, 512), ttnn.float32, True),
    ((1, 1, 64, 128), ttnn.float32, True),
    ((1, 1, 64, 128), ttnn.bfloat8_b, True),
    ((4, 8, 32, 256), ttnn.bfloat8_b, False),  # no gamma
    ((1, 1, 2048, 256), ttnn.bfloat16, False),
]
for shape, dt, has_g in cases:
    try:
        mc = auto_shard_config(list(shape), HS, layout=ttnn.TILE_LAYOUT, dtype=dt, device=device)
        x = torch.randn(shape, dtype=TD[dt])
        tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        g = tg = None
        if has_g:
            g = torch.randn(1, 1, 1, shape[-1], dtype=TD[dt])
            tg = ttnn.from_torch(g, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        ht, wt = pd._tile_geometry(tx)
        p = pd._select_placement(device, device.compute_with_storage_grid_size(), tx, ht, wt, True)
        blk = pd._derive_blocking(tx, tg, 110, p, sharded_in=True, sharded_out=True)
        out = ttnn.to_torch(rms_norm(tx, gamma=tg, memory_config=mc)).to(torch.float32)
        print(
            f"{shape} {dt} g={has_g} shard={list(mc.shard_spec.shape)} cores={p.num_cores} rows={p.rows_core_max} "
            f"WT={blk.Wt} chunk={blk.wt_chunk} nw={blk.nw} htb={blk.ht_block} cb={blk.cb_total_bytes} "
            f"PCC={pcc(out, ref(x,g)):.6f}",
            flush=True,
        )
    except Exception as e:
        print(f"{shape} {dt} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
ttnn.close_device(device)
