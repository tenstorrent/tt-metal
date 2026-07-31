# x[row] = const(row//32 + 1) -> out scales exactly as (tile-row+1)^2 per hidden tile.
# Run each config TWICE to separate a deterministic mapping bug from a race.
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


def run(device, emb, cap, count):
    x = torch.zeros((1, 1, cap, emb))
    for r in range(cap):
        x[0, 0, r, :] = float(r // 32 + 1)
    wg = torch.ones((emb, HIDDEN))
    wu = torch.ones((emb, HIDDEN))
    wd = torch.zeros((HIDDEN, emb))
    for i in range(HIDDEN):
        wd[i, i] = 1.0
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
    tc, ti = counts_t(count, device)
    out = moe_fused_swiglu(tx, tw[0], tw[1], tw[2], tc, ti, LID)
    return ttnn.to_torch(out)[0, 0, :, :HIDDEN].to(torch.float32)


device = ttnn.open_device(device_id=0)
try:
    # (emb, cap, count): the three acceptance failures + two known-good controls
    for emb, cap, count in ((6144, 2048, 128), (6144, 1024, 96), (7168, 1024, 64), (7168, 1024, 255), (7168, 1024, 32)):
        m_t = (count + 31) // 32
        for rep in range(2):
            got = run(device, emb, cap, count)
            # per token tile-row: mean over the 64 hidden tiles, normalised by tile-row 0
            base = None
            line = []
            for tr in range(m_t):
                v = got[tr * 32].reshape(HIDDEN // TILE, TILE)[:, 0]
                m = v.mean().item()
                if base is None:
                    base = m
                line.append(round(m / base, 3))
            want = [round(((tr + 1) / 1.0) ** 2, 3) for tr in range(m_t)]
            ok = all(abs(line[i] - want[i]) < 0.06 * want[i] for i in range(m_t))
            print(
                f"emb={emb} cap={cap} count={count} m_t={m_t} rep{rep}: "
                f"{'OK ' if ok else 'BAD'} got={line} want={want}"
            )
finally:
    ttnn.close_device(device)
