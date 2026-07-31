# Which hidden tiles / which token tile-rows are wrong at m_eff < M_BLOCK?
#   W_down = hidden->emb identity  => out[:, :HIDDEN] IS h
#   W_gate per-hidden-tile constant => h's value encodes the hidden TILE index
#   x[row] = const(row+1)           => h also encodes the token ROW
# So a zero/stale hidden tile and a wrong-row x both show up directly.
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


def run(device, emb, cap, count, x):
    wg = torch.zeros((emb, HIDDEN))
    for nt in range(HIDDEN // TILE):
        wg[:, nt * TILE : (nt + 1) * TILE] = float(nt + 1)
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
    emb, cap = 7168, 1024
    for count in (64, 128, 255):
        # x row r = (r//32 + 1): constant per TOKEN TILE-ROW, so h encodes the tile-row too.
        x = torch.zeros((1, 1, cap, emb))
        for r in range(cap):
            x[0, 0, r, :] = float(r // 32 + 1)
        got = run(device, emb, cap, count, x)
        m_t = (count + 31) // 32
        print(f"=== count={count} M_t={m_t}")
        for tr in range(m_t):
            row = got[tr * 32]  # one token row
            per_tile = row.reshape(HIDDEN // TILE, TILE)[:, 0]
            # expected scales as (tile+1) * xval^2 * emb^2 -> normalise by tile 0
            base = per_tile[0].item()
            ratio = (per_tile / max(base, 1e-9)).tolist()
            want = [float(nt + 1) for nt in range(HIDDEN // TILE)]
            bad = [nt for nt in range(HIDDEN // TILE) if abs(ratio[nt] - want[nt]) > 0.05 * want[nt]]
            print(f"  tile-row {tr}: tile0={base:.4e}  bad hidden tiles {len(bad)}/64 -> {bad[:12]}")
            if bad:
                print(f"      ratio at bad: {[round(ratio[nt],3) for nt in bad[:12]]}")
finally:
    ttnn.close_device(device)
