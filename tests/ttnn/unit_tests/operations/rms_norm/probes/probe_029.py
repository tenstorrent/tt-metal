import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import _plan
from eval.sharding import auto_shard_config
from eval.metrics import check_output


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (o * g.to(torch.float32).reshape(-1)) if g is not None else o


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rms(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).pow(2).mean().sqrt() / b.std()).item()


_ML = ttnn.TensorMemoryLayout
device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 3232, 96), (1, 1, 4064, 160)]:
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.bfloat16)
        gm = torch.randn(shape[-1], dtype=torch.bfloat16)
        exp = ref(x, gm)
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = False
        cfg.math_approx_mode = False
        for name, ml in [("INTERLEAVED", None), ("WIDTH", _ML.WIDTH_SHARDED), ("HEIGHT", _ML.HEIGHT_SHARDED)]:
            outs = []
            for trial in range(3):
                mc = (
                    ttnn.DRAM_MEMORY_CONFIG
                    if ml is None
                    else auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
                )
                tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
                tg = ttnn.from_torch(
                    gm.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                kw = {} if ml is None else {"memory_config": tx.memory_config()}
                if trial == 0:
                    p = _plan(
                        device,
                        tx,
                        has_gamma=True,
                        bytes_={
                            "in_tile": 2048,
                            "out_tile": 2048,
                            "gamma_tile": 2048,
                            "stat_tile": 4096,
                            "bf16_tile": 2048,
                        },
                    )
                    print(
                        f"{shape} {name} plan: g={p['num_row_groups']} s={p['num_hidden_slices']} S={p['slice_hidden_tiles']} B={p['block_rows']} shard_rows={p['shard_rows']}",
                        flush=True,
                    )
                out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg, **kw)
                outs.append(ttnn.to_torch(out))
            same = all(torch.equal(outs[0], o) for o in outs[1:])
            print(
                f"  {shape} {name}: PCC {pcc(outs[0],exp):.6f} relRMS {rms(outs[0],exp):.4f} deterministic={same}",
                flush=True,
            )
finally:
    ttnn.close_device(device)
