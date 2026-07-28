import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
print(
    "L1 bank:",
    ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank,
    "-> total budget",
    pd._l1_total_budget(device),
    flush=True,
)
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
TD = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}


def ref(x, g, eps=1e-6):
    xf = x.to(torch.float32)
    return xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps) * g.to(torch.float32).reshape(-1)


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# The L1-tight corner, each case fully deallocated so the next starts clean.
for shape, dt in [
    ((1, 1, 32, 4096), ttnn.float32),
    ((1, 1, 128, 4096), ttnn.float32),
    ((1, 1, 32, 4096), ttnn.bfloat16),
    ((1, 1, 32, 8192), ttnn.bfloat16),
    ((128, 8192), ttnn.bfloat16),
    ((1, 1, 32, 8192), ttnn.bfloat8_b),
    ((1, 1, 32, 8192), ttnn.float32),
]:
    tx = tg = out = None
    try:
        mc = auto_shard_config(list(shape), HS, layout=ttnn.TILE_LAYOUT, dtype=dt, device=device)
        x = torch.randn(shape, dtype=TD[dt])
        g = torch.randn(1, 1, 1, shape[-1], dtype=TD[dt])
        tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tg = ttnn.from_torch(g, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        ht, wt = pd._tile_geometry(tx)
        p = pd._select_placement(device, device.compute_with_storage_grid_size(), tx, ht, wt, True)
        blk = pd._derive_blocking(
            tx, tg, 110, p, sharded_in=True, sharded_out=True, l1_total_budget=pd._l1_total_budget(device)
        )
        out = rms_norm(tx, gamma=tg, memory_config=mc)
        r = ttnn.to_torch(out).to(torch.float32)
        print(
            f"OK   {shape} {dt.name} cores={p.num_cores} chunk={blk.wt_chunk} nw={blk.nw} "
            f"prog={blk.program_cb_bytes} shard={blk.resident_shard_bytes} PCC={pcc(r, ref(x,g)):.6f}",
            flush=True,
        )
    except Exception as e:
        print(f"FAIL {shape} {dt.name}: {type(e).__name__}: {str(e)[:180]}", flush=True)
    finally:
        for t in (out, tg, tx):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
ttnn.close_device(device)
