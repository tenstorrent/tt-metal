# Is the residual m_eff=4 race in the h all-gather (still a LOOPBACK send) or still in x?
# The bfp8_tile path has NO x loopback but the SAME h loopback. 4 reps each.
import torch, ttnn
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu

TILE, HIDDEN = 32, 2048
NG, NL, LID, GID = 256, 8, 3, 137


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def build(emb, cap, count, fmt, device):
    torch.manual_seed(42)
    x = torch.randn((1, 1, cap, emb), dtype=torch.float32)
    if count < cap:
        x[:, :, count:, :] = 100.0
    wg = torch.randn((emb, HIDDEN))
    wu = torch.randn((emb, HIDDEN))
    wd = torch.randn((HIDDEN, emb))
    dt, lay = (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) if fmt == "bf16_rm" else (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    tx = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tw = [
        ttnn.from_torch(
            w.to(torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in (wg, wu, wd)
    ]
    c = torch.zeros(NG, dtype=torch.int32)
    c[GID] = count
    idx = torch.tensor([(11 + 37 * i) % NG for i in range(NL)], dtype=torch.int32)
    idx[LID] = GID
    f = lambda t: ttnn.from_torch(
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    xr = x[0, 0, :count, :].to(torch.bfloat16).to(torch.float32)
    h = torch.nn.functional.silu(xr @ wg) * (xr @ wu)
    return tx, tw, f(c), f(idx), h @ wd


device = ttnn.open_device(device_id=0)
try:
    for fmt in ("bf16_rm", "bfp8_tile"):
        for emb, cap, count in ((6144, 2048, 128), (7168, 5120, 128)):
            tx, tw, tc, ti, ref = build(emb, cap, count, fmt, device)
            vals = []
            for rep in range(4):
                out = moe_fused_swiglu(tx, tw[0], tw[1], tw[2], tc, ti, LID)
                got = ttnn.to_torch(out)[0, 0, :count, :].to(torch.float32)
                vals.append(round(pcc(ref, got), 5))
            spread = max(vals) - min(vals)
            print(
                f"{fmt} emb={emb} cap={cap} count={count} m_eff=4: PCC={vals} spread={spread:.5f} "
                f"{'STABLE' if spread < 1e-4 else 'RACY'}"
            )
finally:
    ttnn.close_device(device)
