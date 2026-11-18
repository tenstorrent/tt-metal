# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""EP-4 with weights on Y: the host reference of the layout (WORKLOG "EP-4 with weights on Y (LoudBox)").

Mesh R x C (C = 4: the EP group is a mesh row, along X). Chip column c holds experts [EPC c, EPC (c + 1)), EPC = E / C,
sharded over the column's R chips (OWN = EPC / R each) and streamed around the column's Y ring, so every chip of the
column computes all EPC of them. Every chip owns n tokens; S = n padded to 32.

Per chip (r, c), the buffers of the design (names as in the WORKLOG):
- route [n]: the X mask of the row's chips holding >= 1 of the token's experts, own chip excluded, as
  sparse_dispatch_2d's route word (bit r C + c', order 0 = X first, link << 17);
- grp [C n, K]: ids / gate weights of every token of the row (home-major: home c' tokens at [c' n, (c' + 1) n));
- lists: per local expert (le = e - EPC c) its count, token slots home S + t (into xbuf) and O rows, (home, t) order;
- runs: per (home, t) with >= 1 local expert: base row in O, count, the experts and gate weights (k order);
- O rows: a run's rows are contiguous, base(home, t) + j, j = the expert's rank among the run's (k order);
- ready[le]: the runs whose last local expert in stream order is le;
- mask [n] (at the home chip): which of the row's chips sent a partial (own chip included);
- cnt [C]: per source chip, the partials that land on this home chip.

The math: O rows unweighted (bf16), the pre-sum w . row in fp32 -> bf16 partial a (token, expert chip), the home pass
the sum of the present partials (slot order 0..C-1) plus the shared expert's row. Every sum in a fixed order."""

from dataclasses import dataclass

import torch

E_DEF, K_DEF, H_DEF, I_DEF, IS_DEF = 640, 12, 3840, 640, 2944


def pad32(n):
    return -(-n // 32) * 32


@dataclass(frozen=True)
class Geo:
    R: int  # mesh rows = the Y ring's length
    C: int = 4  # mesh columns = the EP group
    E: int = E_DEF
    K: int = K_DEF

    @property
    def EPC(self):
        return self.E // self.C

    @property
    def OWN(self):
        return self.EPC // self.R


LB = Geo(2)
GLX = Geo(8)

# Galaxy lengths and their per-chip load (n = T / 32); the LoudBox runs T = 8 n
LENGTHS = {"302K": 302373, "588K": 588369, "874K": 874365, "1M": 1000000}


def n_per_chip(name):
    return round(LENGTHS[name] / 32)


# ---------------------------------------------------------------------------------------------------------- routing
# zipf: a logit bias -A log(rank) over a random order of the experts (rank 1 hottest), on top of N(0, 1) logits.
# ZIPF_A gives an expert-load CV of about 1 at top-12 of 640 (zipf_cv() measures it).
ZIPF_A = 0.25


def zipf_bias(E, seed, a=ZIPF_A):
    g = torch.Generator().manual_seed(seed + 7919)
    rank = torch.empty(E)
    rank[torch.randperm(E, generator=g)] = torch.arange(1, E + 1, dtype=torch.float32)
    return -a * torch.log(rank)


def routing(geo: Geo, n, kind="uniform", seed=0, chunk=8192, a=None):
    """ids [R, C, n, K] int64 (score order) and gate weights [R, C, n, K] fp32 (softmax over all E, renormalized over
    the top K = softmax of the top-K logits), from logits N(0, 1) (+ the zipf bias)."""
    R, C, E, K = geo.R, geo.C, geo.E, geo.K
    bias = torch.zeros(E) if kind == "uniform" else zipf_bias(E, seed, ZIPF_A if a is None else a)
    assert kind in ("uniform", "zipf"), kind
    g = torch.Generator().manual_seed(seed)
    ids = torch.empty(R * C * n, K, dtype=torch.int64)
    w = torch.empty(R * C * n, K)
    for a in range(0, R * C * n, chunk):
        b = min(a + chunk, R * C * n)
        lg = torch.randn(b - a, E, generator=g) + bias
        v, i = lg.topk(K, dim=-1)
        ids[a:b], w[a:b] = i, torch.softmax(v, dim=-1)
    return ids.reshape(R, C, n, K), w.reshape(R, C, n, K)


def expert_cv(ids, E):
    load = torch.bincount(ids.flatten(), minlength=E).double()
    return float(load.std() / load.mean())


# ---------------------------------------------------------------------------------------------------------- layout
@dataclass
class Chip:
    """One chip's (r, c) derived buffers (see the module docstring)."""

    route: torch.Tensor  # [n] int64 route words
    counts: torch.Tensor  # [EPC] rows a local expert
    lists_tok: list  # per local expert: token slots home S + t (int64), (home, t) order
    lists_row: list  # per local expert: O rows
    run_home: torch.Tensor  # [nruns] home column
    run_t: torch.Tensor  # [nruns] token
    run_base: torch.Tensor  # [nruns] first O row
    run_cnt: torch.Tensor  # [nruns]
    run_exp: torch.Tensor  # [nruns, K] local experts (k order), -1 past the count
    run_w: torch.Tensor  # [nruns, K] gate weights (k order), 0 past the count
    R_rows: int  # O rows used
    mask: torch.Tensor  # [n] int64: bit c' = home token t has a partial from the row's chip c' (own included)
    cnt: torch.Tensor  # [C] partials landing here from each of the row's chips
    pair_row: torch.Tensor = None  # [C, n, K] O row of (home, t, k) on this chip, -1 if not local


def route_words(geo: Geo, ids, link=0, row0=0):
    """[rows, C, n] route words of mesh rows row0.. (ids [rows, C, n, K]; sparse_dispatch_2d, up to 16 chips: mask |
    order << 16 | link << 17; order 0 = X first). link: an int, or a function of the token index."""
    R, C = ids.shape[0], geo.C
    assert R * C <= 16, "one route word only up to 16 chips"
    col = ids // geo.EPC  # [R, C, n, K]
    hit = torch.zeros(*ids.shape[:3], C, dtype=torch.bool)
    hit.scatter_(-1, col, True)
    cc = torch.arange(C)
    hit &= cc.view(1, C, 1, 1) != cc.view(1, 1, 1, C)  # own chip excluded
    bit = 1 << ((row0 + torch.arange(R)).view(R, 1, 1, 1) * C + cc.view(1, 1, 1, C))
    mask = (hit.long() * bit).sum(-1)
    lk = link if not callable(link) else link(torch.arange(ids.shape[2]))
    return mask | (torch.as_tensor(lk) << 17)


def home_masks(geo: Geo, ids):
    """[R, C, n] bit c' set = the token holds >= 1 expert on its row's chip c' (own included)."""
    col = ids // geo.EPC
    hit = torch.zeros(*ids.shape[:3], geo.C, dtype=torch.bool)
    hit.scatter_(-1, col, True)
    return (hit.long() << torch.arange(geo.C)).sum(-1)


def chip_layout(geo: Geo, ids, w, r, c, S=None, keep_pairs=False):
    """Chip (r, c)'s Chip from the routing of its mesh row (ids / w [R, C, n, K])."""
    C, K, EPC = geo.C, geo.K, geo.EPC
    n = ids.shape[2]
    S = S or pad32(n)
    gi, gw = ids[r], w[r]  # [C, n, K]: the group exchange's grp
    sel = (gi // EPC) == c  # [C, n, K]
    cnt_tok = sel.sum(-1)  # [C, n]
    flat_cnt = cnt_tok.flatten()
    base = torch.cumsum(flat_cnt, 0) - flat_cnt  # exclusive
    j = torch.cumsum(sel.long(), -1) - 1
    prow = torch.where(sel, base.view(C, n, 1) + j, torch.full_like(j, -1))
    Rr = int(flat_cnt.sum())
    # lists: pairs in (home, t, k) order, stable-sorted by local expert -> each list in (home, t) order
    hh, tt, kk = sel.nonzero(as_tuple=True)
    le = gi[hh, tt, kk] - EPC * c
    order = torch.sort(le, stable=True).indices
    le_s, tok_s, row_s = le[order], (hh * S + tt)[order], prow[hh, tt, kk][order]
    counts = torch.bincount(le_s, minlength=EPC)
    splits = counts.tolist()
    lists_tok, lists_row = list(torch.split(tok_s, splits)), list(torch.split(row_s, splits))
    # runs: (home, t) with >= 1 local expert, in (home, t) order; experts / weights compacted in k order
    has = flat_cnt > 0
    rid = has.nonzero(as_tuple=True)[0]
    rk = torch.sort((~sel).long().reshape(C * n, K), dim=-1, stable=True).indices[rid]  # local ks first, k order
    sel_f, gi_f, gw_f = sel.reshape(C * n, K)[rid], gi.reshape(C * n, K)[rid], gw.reshape(C * n, K)[rid]
    take = lambda a: torch.gather(a, 1, rk)
    valid = take(sel_f.long()).bool()
    run_exp = torch.where(valid, take(gi_f) - EPC * c, torch.full_like(rk, -1))
    run_w = torch.where(valid, take(gw_f), torch.zeros_like(gw_f))
    hm = home_masks(geo, ids)[r, c]
    cnt = torch.tensor([int(((hm >> s) & 1).sum()) for s in range(C)])  # partials landing on (r, c) from each chip
    return Chip(
        route=route_words(geo, ids[r : r + 1], row0=r)[0, c] if geo.R * C <= 16 else None,
        counts=counts,
        lists_tok=lists_tok,
        lists_row=lists_row,
        run_home=rid // n,
        run_t=rid % n,
        run_base=base[rid],
        run_cnt=flat_cnt[rid],
        run_exp=run_exp,
        run_w=run_w,
        R_rows=Rr,
        mask=hm,
        cnt=cnt,
        pair_row=prow if keep_pairs else None,
    )


def partial_counts(geo: Geo, ids):
    """[R, C(src), C(home)] partials a (source, home) pair: home tokens with >= 1 expert on the source chip."""
    hm = home_masks(geo, ids)  # [R, C(home), n]
    return torch.stack([((hm >> s) & 1).sum(-1) for s in range(geo.C)], 1)


def chip_rows(geo: Geo, ids):
    """[R, C] O rows a chip: the row's (token, k) pairs on its experts."""
    col = ids // geo.EPC  # [R, C(home), n, K]
    return torch.stack([(col == c).sum((1, 2, 3)) for c in range(geo.C)], 1)


# ---------------------------------------------------------------------------------------------------------- readiness
def lane_orders(R, EPC, G, kind="lane", lane_groups=None, sched_kw=None):
    """Per Y position (chip of the column cycle, 0..R-1) the compute order of each group: [pos][g] -> [local experts].
    Local expert le of lane group g is lane-local i = le // G (le % G == g), as ws_lane.make_tables. kind:
    - "lane": the lane's static schedule (ws_lane_sched.lane_sched on the R-chip column cycle, the 5 x 5 lane's
      packets and slots), every group a Y lane;
    - "own_first" / "interleaved": the lane's sets of own / remote experts per group, own first, or alternating;
    - "split": lane_groups of the G groups are Y lanes holding only remote experts, the rest only own ones
      (sequentially, any order)."""
    from models.demos.ws_expert.ws_lane_sched import lane_plan, lane_sched, slot_types

    kw = dict(P=14, GU=10, C=7, keep=2, types=slot_types("d0,d1,d0d1,any"))
    kw.update(sched_kw or {})
    if kind == "split":
        LG = lane_groups
        own_per = EPC // R
        out = []
        for pos in range(R):
            # remote experts: those of the column's other chips; own: this chip's (owner = le % R by position)
            own = [le for le in range(EPC) if le % R == pos]
            rem = [le for le in range(EPC) if le % R != pos]
            per = [[] for _ in range(G)]
            for k, le in enumerate(rem):
                per[k % LG].append(le)
            for k, le in enumerate(own):
                per[LG + k % (G - LG)].append(le)
            out.append(per)
        assert all(len(o) == own_per for o in [[le for le in range(EPC) if le % R == p] for p in range(R)])
        return out
    E = EPC // G
    owners, _ = lane_plan(R, E)
    sch = lane_sched(R, E, **kw) if R > 1 else dict(order={0: list(range(E))})
    out = []
    for pos in range(R):
        order = sch["order"][pos]
        per = []
        for g in range(G):
            seq = [i * G + g for i in order]
            if kind != "lane":
                own = [i * G + g for i in order if owners[i] == pos]
                rem = [i * G + g for i in order if owners[i] != pos]
                if kind == "own_first":
                    seq = own + rem
                else:
                    assert kind == "interleaved", kind
                    seq = [v for pair in zip(own, rem) for v in pair] + own[len(rem) :] + rem[len(own) :]
            per.append(seq)
        out.append(per)
    return out


def positions(per_group, EPC):
    """[EPC] stream position of each local expert in (0, 1]: its index in its group's order / the group's length (the
    groups run side by side), ties broken by group (a key for 'last')."""
    pos = torch.zeros(EPC, dtype=torch.float64)
    key = torch.zeros(EPC, dtype=torch.float64)
    G = len(per_group)
    for g, seq in enumerate(per_group):
        for k, le in enumerate(seq):
            pos[le] = (k + 1) / len(seq)
            key[le] = pos[le] + g * 1e-9
    assert (pos > 0).all(), "every local expert in one group's order"
    return pos, key


def ready_lists(ch: Chip, key, EPC):
    """ready[le]: the runs (indices into ch's runs) whose last local expert in stream order is le."""
    kx = torch.where(ch.run_exp >= 0, key[ch.run_exp.clamp(min=0)], torch.full(ch.run_exp.shape, -1.0, dtype=key.dtype))
    last = torch.gather(ch.run_exp, 1, kx.argmax(1, keepdim=True)).squeeze(1)
    order = torch.sort(last, stable=True).indices
    return list(torch.split(order, torch.bincount(last, minlength=EPC).tolist())), last


def readiness(ch: Chip, pos, key, grid=None):
    """F(x) on grid: the share of the chip's runs ready (last expert done) at stream progress x; and the x^c model
    (each run's experts at independent uniform positions)."""
    grid = torch.linspace(0, 1, 101, dtype=torch.float64) if grid is None else grid
    _, last = ready_lists(ch, key, pos.numel())
    px = pos[last]
    F = (px.view(1, -1) <= grid.view(-1, 1) + 1e-12).double().mean(1)
    model = (grid.view(-1, 1) ** ch.run_cnt.double().view(1, -1)).mean(1)
    return grid, F, model


def hidden_share(grid, F, S, C):
    """Fluid model of the combine under the stream: runs become ready at F(x) (x = stream progress, the stream takes S),
    the combine sends at most a share S / C of all partials per unit x (it takes C alone). Returns the share sent by the
    stream's end (hidden) and the tail C (1 - hidden)."""
    rate = S / C
    q, sent = 0.0, 0.0
    for a in range(1, len(grid)):
        dx = float(grid[a] - grid[a - 1])
        q += float(F[a] - F[a - 1])
        s = min(q, rate * dx)
        q -= s
        sent += s
    return sent, (1 - sent) * C


# ---------------------------------------------------------------------------------------------------------- math
def expert_ref(xq, wg, wu, wd, h_dt=None):
    """SwiGLU expert on the quantized inputs: silu(x Wg) * (x Wu), h rounded like the lane (h_dt), then Wd."""
    from models.demos.ws_expert.bench_util import quantize

    h = torch.nn.functional.silu(xq @ wg) * (xq @ wu)
    if h_dt is not None:
        h = quantize(h, h_dt)
    return h @ wd


def presum(O, ch: Chip, ridx=None):
    """Partials [nruns, H] bf16: sum_j w_j O[base + j] in fp32, in the run's order."""
    ridx = torch.arange(ch.run_cnt.numel()) if ridx is None else ridx
    acc = torch.zeros(ridx.numel(), O.shape[1])
    cnt, base, w = ch.run_cnt[ridx], ch.run_base[ridx], ch.run_w[ridx]
    for j in range(int(cnt.max()) if ridx.numel() else 0):
        m = cnt > j
        acc[m] += w[m, j : j + 1] * O[base[m] + j].float()
    return acc.to(torch.bfloat16)


def home_pass(P, mask, shared):
    """y [n, H] bf16 = the present partial slots (mask bit c, c = 0..C-1 in order) + the shared row, fp32 accumulate.
    P [n, C, H]; absent slots never read (no 0 x NaN)."""
    acc = torch.zeros(P.shape[0], P.shape[2])
    for s in range(P.shape[1]):
        m = ((mask >> s) & 1).bool()
        acc[m] += P[m, s].float()
    return (acc + shared.float()).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------------------- report
def report(geo: Geo, n, kinds=("uniform", "zipf"), seed=0, orders=None, S_ms=None, C_ms=None, log=print):
    """Readiness / capacity stats for a mesh at n tokens a chip."""
    out = {}
    for kind in kinds:
        ids, w = routing(geo, n, kind, seed)
        rows = chip_rows(geo, ids).double()
        pc = partial_counts(geo, ids).double()
        log(
            f"{kind}: expert CV {expert_cv(ids, geo.E):.2f}; O rows a chip max/mean {rows.max() / rows.mean():.3f} "
            f"(mean {rows.mean() / n:.2f} n, max {rows.max() / n:.2f} n); partials a (src, home) max/mean "
            f"{pc.max() / pc.mean():.3f} (mean {pc.mean() / n:.3f} n)"
        )
        ch = chip_layout(geo, ids, w, 0, 0)
        rc = ch.run_cnt.double()
        log(
            f"  chip (0,0): runs {rc.numel()} = {rc.numel() / (geo.C * n):.3f} of the row's tokens, experts a run mean {rc.mean():.2f}"
        )
        res = {}
        for name, per in (orders or {}).items():
            pos, key = positions(per[0], geo.EPC)
            grid, F, model = readiness(ch, pos, key)
            pts = {x: float(F[int(x * 100)]) for x in (0.25, 0.5, 0.75, 0.9)}
            x3 = {x: x**3 for x in pts}
            line = f"  {name:12s} F(.25/.5/.75/.9) " + " ".join(f"{pts[x]:.3f}" for x in pts)
            line += "  x^c " + " ".join(f"{float(model[int(x * 100)]):.3f}" for x in pts)
            line += "  x^3 " + " ".join(f"{x3[x]:.3f}" for x in pts)
            if S_ms is not None:
                h, tail = hidden_share(grid, F, S_ms, C_ms)
                g3 = grid**3
                h3, t3 = hidden_share(grid, g3, S_ms, C_ms)
                line += f"  hidden {h * 100:.0f}% tail {tail:.2f} ms (x^3: {h3 * 100:.0f}%, {t3:.2f})"
            log(line)
            res[name] = dict(F=F, model=model, grid=grid)
        out[kind] = dict(rows=rows, pc=pc, res=res)
    return out
