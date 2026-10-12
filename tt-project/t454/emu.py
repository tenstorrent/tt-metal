import math, sys, torch
import models.tt_dit.models.transformers.minimax_h3.vsa_stages_minimax_h3 as st
from models.tt_dit.pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS as T, build_sla_geometry

st.from_torch = lambda x, **kw: x


def ref(q, k, v, ratio, blkq, blkk=64):  # q [H, L, d]
    H, L, d = q.shape

    def mp(x, b):
        pad = -L % b
        xp = torch.nn.functional.pad(x, (0, 0, 0, pad))
        s = xp.reshape(H, -1, b, d).sum(-2)
        c = torch.full((s.shape[1], 1), float(b), dtype=x.dtype)
        c[-1] = b - pad
        return s / c

    sc = mp(q, blkq) @ mp(k - k.mean(-2, keepdim=True), blkk).transpose(-1, -2)
    K = sc.shape[-1]
    kk = max(1, min(K, int(ratio * K)))
    ids = torch.topk(sc, kk, dim=-1).indices
    out = torch.empty_like(q)
    for h in range(H):
        for qb in range(sc.shape[1]):
            keys = torch.cat([torch.arange(b * blkk, min(L, (b + 1) * blkk)) for b in sorted(ids[h, qb].tolist())])
            qs = slice(qb * blkq, min(L, (qb + 1) * blkq))
            p = torch.softmax(q[h, qs] @ k[h, keys].T / math.sqrt(d), -1)
            out[h, qs] = p @ v[h, keys]
    return out, ids


def ours(q, k, v, g, sparsity, qbt):
    H, L, d = q.shape
    s = st.MiniMaxH3VSACoarseStage(
        g,
        sparsity=sparsity,
        selection="sla",
        sla_q_block_tiles=qbt,
        padded_pooling=True,
        head_dim=d,
        mesh_device=None,
        sp_axis=1,
    )
    qp, kp, vp = (g.pack_rows(x, dim=-2) for x in (q, k, v))
    sp, R, S = g.sp_factor, g.tiles_per_shard, s.slots_per_shard
    rl = g.padded_len // sp
    aq = s.a_t_q.reshape(g.padded_len, S).double()
    ak = s.a_t_kv.reshape(g.padded_len, S).double()
    kct = torch.cat([kp[:, i * rl : (i + 1) * rl].transpose(-1, -2) @ ak[i * rl : (i + 1) * rl] for i in range(sp)], -1)
    cm = s.cand_mask.reshape(-1).double()
    counts = s.block_counts_tensor().reshape(-1)
    shift, padn = s.coarse_slots_shift, S - R
    out = torch.zeros_like(qp)
    sel = {}
    for i in range(sp):
        qc = (qp[:, i * rl : (i + 1) * rl].transpose(-1, -2) @ aq[i * rl : (i + 1) * rl]).transpose(-1, -2)
        sc = qc @ kct + cm
        ids = torch.topk(sc, s.k_pad, dim=-1).indices[:, :R, : s.k]
        for h in range(H):
            for r in range(R):
                t = i * R + r
                blocks = sorted({int(b) - (int(b) >> shift) * padn if shift else int(b) for b in ids[h, r]})
                sel[(h, t)] = blocks
                keys = torch.cat([torch.arange(b * T, b * T + int(counts[b])) for b in blocks])
                qs = slice(t * T, (t + 1) * T)
                p = torch.softmax(qp[h, qs] @ kp[h, keys].T / math.sqrt(d), -1)
                out[h, qs] = p @ vp[h, keys]
    return g.unpack_rows(out, dim=-2), sel, s


if __name__ == "__main__":
    L = int(sys.argv[1])
    sp = int(sys.argv[2])
    qbt = int(sys.argv[3])
    H, d = 2, 32
    torch.manual_seed(0)
    q, k, v = (torch.randn(H, L, d, dtype=torch.float64) for _ in range(3))
    k = k + 0.5
    g = build_sla_geometry(L, (1, 1, 1), sp_factor=sp, q_block_tiles=qbt)
    o_ref, ids_ref = ref(q, k, v, 0.15, qbt * T)
    o, sel, s = ours(q, k, v, g, 0.85, qbt)
    print("k", s.k, "k_pad", s.k_pad, "R", g.tiles_per_shard, "slots", s.slots_per_shard, "ntiles", g.n_tiles)
    nreal = math.ceil(L / T)
    bad = 0
    for h in range(H):
        for t in range(nreal):
            rs = sorted(ids_ref[h, t // qbt].tolist())
            if rs != sel[(h, t)]:
                bad += 1
    print("sel mismatches", bad, "of", H * nreal, " max|out diff|", (o - o_ref).abs().max().item())
