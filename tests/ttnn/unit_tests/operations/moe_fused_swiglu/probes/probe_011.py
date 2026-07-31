# Acceptance-style fixture (randn + hostile padding sentinel). Sweep emb x m_eff and report
# overall PCC plus PER-TOKEN-TILE-ROW PCC, so we see whether the damage is emb-specific,
# m_eff-specific, and which rows carry it.
import torch, ttnn
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu

TILE, HIDDEN = 32, 2048
NG, NL, LID, GID = 256, 8, 3, 137
SENT = 100.0


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def build(emb, cap, count, device):
    torch.manual_seed(42)
    x = torch.randn((1, 1, cap, emb), dtype=torch.float32)
    if count < cap:
        x[:, :, count:, :] = SENT
    wg = torch.randn((emb, HIDDEN))
    wu = torch.randn((emb, HIDDEN))
    wd = torch.randn((HIDDEN, emb))
    tx = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
    h = torch.nn.functional.silu(xr @ wg)
    h = h * (xr @ wu)
    return tx, tw, f(c), f(idx), h @ wd


device = ttnn.open_device(device_id=0)
try:
    for emb, cap, count in (
        (6144, 2048, 128),
        (7168, 1024, 128),
        (6144, 1024, 96),
        (7168, 1024, 96),
        (6144, 1024, 64),
        (6144, 2048, 256),
        (7168, 1024, 64),
    ):
        tx, tw, tc, ti, ref = build(emb, cap, count, device)
        m_t = (count + 31) // 32
        m_eff = 1
        while m_eff < min(m_t, 8):
            m_eff <<= 1
        for rep in range(2):
            out = moe_fused_swiglu(tx, tw[0], tw[1], tw[2], tc, ti, LID)
            got = ttnn.to_torch(out)[0, 0, :count, :].to(torch.float32)
            rows = [
                round(pcc(ref[r * 32 : min((r + 1) * 32, count)], got[r * 32 : min((r + 1) * 32, count)]), 4)
                for r in range(m_t)
            ]
            print(
                f"emb={emb} cap={cap} count={count} M_t={m_t} m_eff={m_eff} rep{rep}: "
                f"PCC={pcc(ref,got):.5f}  per-tile-row={rows}"
            )
finally:
    ttnn.close_device(device)
