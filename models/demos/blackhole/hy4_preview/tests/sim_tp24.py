"""CPU (torch only) simulation of the tp24 dataflow (24 cores: 3 per bank, experts a % 3 == group, per-group 8-core exchange, act within the group,
per-core partial down over own experts, leader 3-way sum) against hy4_moe_tp32_layout.emulate_chip, plus semaphore bookkeeping checks.
Run: python models/demos/blackhole/hy4_preview/tests/sim_tp24.py   (random bf16 tiles, E = 40 experts, T = 1..16)"""

import torch

from models.demos.blackhole.hy4_preview.tt import hy4_moe_tp32_layout as L

NB, KB, NC = 8, 24, 8
bf = lambda t: t.to(torch.bfloat16).float()


def sim(x, ids, sc, w01, w2s, E, limit=10.0, down_bf16=True):
    T = x.shape[0]
    MT = 2 if T > 8 else 1
    xt = x.float().reshape(T, 192, 32)
    active = sorted(set(ids.flatten().tolist()))
    na_all = len(active)
    out = torch.zeros(T, L.HIDDEN)
    part_out = {}  # (b, grp) -> [6][T][4*32]
    sem_p_expect = {}
    for grp in range(3):
        own = [active[3 * q + grp] for q in range((na_all - grp + 2) // 3)] if na_all > grp else []
        nq = len(own)
        ng = (nq + 7) // 8
        # phase A on core (b, grp): partial P[q][b] = [4 tiles][T rows][32], bf16 (packed to cb_p)
        P = torch.zeros(NB, nq, 4, T, 32)
        for b in range(NB):
            for q, e in enumerate(own):
                for kl in range(KB):
                    for j in range(4):
                        P[b, q, j] += xt[:, KB * b + kl] @ w01[b, (e * KB + kl) * 4 + j].float()
        P = bf(P)
        # reducer of q is member q % 8: needs 8 partials; sem count check
        recv = {r: 0 for r in range(NC)}
        for g in range(ng):
            for i in range(min(8, nq - 8 * g)):
                recv[(8 * g + i) % NC] += NC  # 8 cores each send one inc for this expert
        nj = {me: ((nq - me + NC - 1) // NC if nq > me else 0) for me in range(NC)}
        for me in range(NC):
            assert recv[me] == NC * nj[me], (grp, me, recv[me], nj[me])
        # act per q (computed by reducer q % 8), broadcast to the 8 group cores
        act = []
        for q, e in enumerate(own):
            s = torch.zeros(T)
            for t in range(T):
                for k in range(8):
                    if int(ids[t, k]) == e:
                        s[t] = sc[t, k].float()
            gu = P[:, q].sum(0)  # [4][T][32]
            g_ = torch.cat([gu[0], gu[1]], 1)
            u_ = torch.cat([gu[2], gu[3]], 1)
            a = torch.nn.functional.silu(g_.clamp(max=limit)) * u_.clamp(-limit, limit) * s[:, None]
            act.append(bf(a))
        # down on core (b, grp): partial over own experts, [s 6][T][128]
        for b in range(NB):
            o = torch.zeros(6, T, 128)
            for s_ in range(6):
                acc = torch.zeros(T, 4, 32)
                for q, e in enumerate(own):
                    at = act[q].reshape(T, 2, 32)
                    for n in range(4):
                        for k in range(2):
                            acc[:, n] += at[:, k] @ w2s[b, ((s_ * E + e) * 8) + k * 4 + n].float()
                        if down_bf16:
                            acc[:, n] = bf(acc[:, n])
                o[s_] = acc.reshape(T, 128)
            part_out[(b, grp)] = bf(o)
    # leader sum
    for b in range(NB):
        tot = bf(part_out[(b, 0)] + part_out[(b, 1)] + part_out[(b, 2)])
        for s_ in range(6):
            col = (KB * b + 4 * s_) * 32
            out[:, col : col + 128] = tot[s_]
    return out


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])


if __name__ == "__main__":
    E = 40
    torch.manual_seed(0)
    w01 = bf(torch.randn(NB, E * KB * 4, 32, 32) * 0.02)
    w2s = bf(torch.randn(NB, 6 * E * 8, 32, 32) * 0.02)
    for T in (1, 2, 4, 8, 9, 12, 16):
        x = bf(torch.randn(T, 6144))
        ids = torch.stack([torch.randperm(E)[:8] for _ in range(T)])
        sc = bf(torch.rand(T, 8) * 0.4 + 0.05)
        ref = L.emulate_chip(x, ids, sc, w01, w2s, E=E)
        got = sim(x, ids, sc, w01, w2s, E)
        print(f"T={T} distinct={len(set(ids.flatten().tolist()))} pcc={pcc(got, ref):.6f}", flush=True)
