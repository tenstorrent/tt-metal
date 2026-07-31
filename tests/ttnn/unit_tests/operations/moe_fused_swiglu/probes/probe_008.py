# Localize the m_eff regression: all-ones input -> EVERY output element must equal
# hidden*emb^2. Per-token-tile-row + per-emb-column error tells us whether the damage is a
# whole tile-row (an x multicast round) or a column band (the Hn / Ne split).
import torch, ttnn
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu

TILE, HIDDEN = 32, 2048
NG, NL, LID, GID = 256, 8, 3, 137


def counts_t(count, device):
    c = torch.zeros(NG, dtype=torch.int32)
    c[GID] = count
    idx = torch.tensor([(11 + 37 * i) % NG for i in range(NL)], dtype=torch.int32)
    idx[LID] = GID
    f = lambda t: ttnn.from_torch(
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return f(c), f(idx)


def run(device, emb, capacity, count, fmt):
    x = torch.ones((1, 1, capacity, emb))
    wg = torch.ones((emb, HIDDEN))
    wu = torch.ones((emb, HIDDEN))
    wd = torch.ones((HIDDEN, emb))
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
    tc, ti = counts_t(count, device)
    out = moe_fused_swiglu(tx, tw[0], tw[1], tw[2], tc, ti, LID)
    return ttnn.to_torch(out)[0, 0, :, :].to(torch.float32)


device = ttnn.open_device(device_id=0)
try:
    for fmt in ("bf16_rm", "bfp8_tile"):
        for emb, cap, cnt in ((7168, 1024, 32), (7168, 1024, 64), (6144, 2048, 128), (7168, 1024, 255)):
            got = run(device, emb, cap, cnt, fmt)[:cnt]
            expect = float(HIDDEN) * float(emb) * float(emb)
            rel = (got - expect).abs() / expect
            m_t = (cnt + 31) // 32
            print(f"--- {fmt} emb={emb} cap={cap} count={cnt} M_t={m_t}")
            for tr in range(m_t):
                band = rel[tr * 32 : min((tr + 1) * 32, cnt)]
                badcols = (band.max(dim=0).values > 0.05).nonzero().flatten().tolist()
                print(
                    f"    tile-row {tr}: maxrel={band.max().item():.4f} badcols={len(badcols)}/{emb} "
                    f"first={badcols[:6]} val0={got[tr*32,0].item():.4e} (want {expect:.4e})"
                )
finally:
    ttnn.close_device(device)
