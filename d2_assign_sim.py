#!/usr/bin/env python3
"""Offline D2 study: how evenly do work-based assignments of Q blocks to cores spread a real Sol-Attn mask?

    python3 d2_assign_sim.py mask_tau1_r4736.npz [cores.txt]

mask_*.npz: written by the pre-pass with SOL_DUMP_MASK_DIR set (block mask, batch dim = SP device, head dim = the TP
devices' heads concatenated). cores.txt: optional per-core rows from sol_profile_summary.py (second argument); with
it the work model is fitted to measured compute time and profiled devices are matched to mask devices.

Per device (one TP x SP position: 14 heads x nqc Q blocks over the 110 SDPA cores), the work of a Q block is
    w = kept blocks (all shards) + beta (the per-block-independent part: pooled entries, per-piece fixed cost),
and the schemes are:
    position     today's factory split: flat (head, Q block) order, the first (H*nqc % cores) cores take one more
    group-swap   static head groups (the cores fully inside a head); members keep their block counts and swap which
                 blocks (largest-first onto the lightest member that still has a free slot); boundary cores fixed
    group-free   as group-swap, but members may take any count up to the slot cap
    head-split   heads split over the cores by cumulative work (cores may share two heads at the boundaries), then
                 largest-first inside each head, slot cap
    global-lpt   largest-first over every block of the device, ignoring heads (a lower bound, not a design)
Printed: max / mean core work per device (1.00 = perfect), and the device-max against today's.
"""

import heapq
import sys

import numpy as np

CORES = 110
GRID_X = 11


def block_work(d, beta):
    """kept[h, q] per device from the dump; returns {(t, s): kept array [H, nqc]}."""
    mask = d["mask"]
    shards, nb, nb_pad, nqc = d["shards"], d["nb"], d["nb_pad"], d["nqc"]
    sp_axis = d["sp_axis"]
    mesh = d["mesh_shape"]
    tp = mesh[1 - sp_axis]
    cols = np.concatenate([np.arange(s * nb_pad, s * nb_pad + nb) for s in range(shards)])
    heads_per = mask.shape[1] // tp
    out = {}
    for s in range(mask.shape[0]):
        for t in range(tp):
            sub = mask[s, t * heads_per : (t + 1) * heads_per, :nqc][..., cols].astype(np.float64)
            kept = sub.sum(-1)
            out[(t, s)] = kept + beta
    return out


def position_split(n_items):
    base, extra = divmod(n_items, CORES)
    counts = [base + (1 if i < extra else 0) for i in range(CORES)]
    starts = np.cumsum([0] + counts[:-1])
    return [list(range(starts[i], starts[i] + counts[i])) for i in range(CORES)]


def core_loads(work_flat, assign):
    return np.array([work_flat[a].sum() if a else 0.0 for a in assign])


def lpt(items, weights, n_bins, caps, start=None):
    """Largest first onto the lightest bin with room. items: indices; caps: per-bin max count; start: initial loads
    (a core shared with another head starts at the work it does there)."""
    heap = [(0.0 if start is None else float(start[b]), b) for b in range(n_bins)]
    heapq.heapify(heap)
    bins = [[] for _ in range(n_bins)]
    for i in sorted(items, key=lambda i: -weights[i]):
        held = []
        while True:
            load, b = heapq.heappop(heap)
            if len(bins[b]) < caps[b]:
                break
            held.append((load, b))
        bins[b].append(i)
        heapq.heappush(heap, (load + weights[i], b))
        for h in held:
            heapq.heappush(heap, h)
    return bins


def schemes(work, cap):
    H, nqc = work.shape
    flat = work.reshape(-1)
    pos = position_split(H * nqc)
    res = {"position": core_loads(flat, pos)}

    # Head groups of the position split: cores whose items all lie in one head.
    head_of = lambda i: i // nqc
    groups, boundary = {}, []
    for c, items in enumerate(pos):
        hs = {head_of(i) for i in items}
        if len(hs) == 1 and items:
            groups.setdefault(hs.pop(), []).append(c)
        else:
            boundary.append(c)
    for name, free in (("group-swap", False), ("group-free", True)):
        assign = [list(pos[c]) for c in range(CORES)]
        for h, members in groups.items():
            items = [i for c in members for i in pos[c]]
            caps = [cap if free else len(pos[c]) for c in members]
            bins = lpt(items, flat, len(members), caps)
            for c, b in zip(members, bins):
                assign[c] = b
        res[name] = core_loads(flat, assign)

    # Head split by cumulative work: core c covers work [c, c+1) * total / CORES of the head-major sequence; a head
    # owned by cores [c0, c1] is then split largest-first over them (a shared core gets its share of the head).
    total = flat.sum()
    head_tot = work.sum(1)
    bounds = np.concatenate([[0], np.cumsum(head_tot)]) / total * CORES
    loads = np.zeros(CORES)
    for h in range(H):
        c0, c1 = int(np.floor(bounds[h])), min(CORES - 1, int(np.ceil(bounds[h + 1])) - 1)
        members = list(range(c0, c1 + 1))
        share = np.array([min(bounds[h + 1], c + 1) - max(bounds[h], c) for c in members])
        caps = [max(1, int(np.ceil(cap * s))) for s in share]
        while sum(caps) < nqc:  # a head's blocks must fit its cores' slots
            caps[int(np.argmax(np.array(share) / np.array(caps)))] += 1
        per_core = total / CORES
        bins = lpt(list(range(h * nqc, (h + 1) * nqc)), flat, len(members), caps, [(1 - s) * per_core for s in share])
        for c, b in zip(members, bins):
            loads[c] += flat[b].sum() if b else 0.0
    res["head-split"] = loads

    bins = lpt(list(range(H * nqc)), flat, CORES, [cap] * CORES)
    res["global-lpt"] = np.array([flat[b].sum() if b else 0.0 for b in bins])
    return res


def read_cores(path):
    rows = [l.split() for l in open(path).read().splitlines()[1:]]
    by_dev = {}
    for r in rows:
        by_dev.setdefault(r[0], []).append((int(r[1]), int(r[2]), float(r[4])))
    out = {}
    for dev, cs in by_dev.items():
        xs = sorted({c[0] for c in cs})
        ys = sorted({c[1] for c in cs})
        busy = np.zeros(CORES)
        for x, y, b in cs:
            i = ys.index(y) * GRID_X + xs.index(x)
            if i < CORES:
                busy[i] = b
        out[dev] = busy
    return out


def main():
    z = np.load(sys.argv[1])
    m = z["meta"]
    d = {
        "mask": z["mask"],
        "shards": int(m[0]),
        "rows": int(m[1]),
        "nb": int(m[2]),
        "nb_pad": int(m[3]),
        "nqc": int(m[4]),
        "sp_axis": int(m[5]),
        "mesh_shape": (int(m[6]), int(m[7])),
    }
    cap = -(-14 * d["nqc"] // CORES)
    print(
        "mask %s: %d SP x %s mesh, nqc %d, nb %d, slot cap %d"
        % (sys.argv[1], d["shards"], d["mesh_shape"], d["nqc"], d["nb"], cap)
    )

    beta = 0.0
    if len(sys.argv) > 2:
        meas = read_cores(sys.argv[2])
        kept = block_work(d, 0.0)
        # Model per core under today's split: busy = a * kept + b * chunks + c. Match each profiled device to the mask
        # device whose kept-per-core vector correlates best, then least squares over all matched cores.
        model = {k: core_loads(v.reshape(-1), position_split(v.size)) for k, v in kept.items()}
        chunks = np.array([len(a) for a in position_split(next(iter(kept.values())).size)], dtype=float)
        pairs, used = [], set()
        for dev, busy in meas.items():
            best = max((k for k in model if k not in used), key=lambda k: np.corrcoef(busy, model[k])[0, 1])
            used.add(best)
            pairs.append((busy, model[best], np.corrcoef(busy, model[best])[0, 1]))
        y = np.concatenate([p[0] for p in pairs])
        X = np.column_stack(
            [np.concatenate([p[1] for p in pairs]), np.tile(chunks, len(pairs)), np.ones(len(pairs) * CORES)]
        )
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        fit = X @ coef
        print(
            "fit: busy = %.3f us x kept + %.1f us x chunks + %.1f us; r = %.3f; per-device match r median %.3f"
            % (coef[0], coef[1], coef[2], np.corrcoef(y, fit)[0, 1], np.median([p[2] for p in pairs]))
        )
        beta = coef[1] / coef[0] if coef[0] > 0 else 0.0
        print("beta (per-block-independent work, in kept-block units per Q block) = %.1f" % beta)

    work = block_work(d, beta)
    tot = {}
    for key, w in sorted(work.items()):
        for name, loads in schemes(w, cap).items():
            tot.setdefault(name, []).append((loads.max() / loads.mean(), loads.max()))
    base_max = max(m for _, m in tot["position"])
    print("\nscheme        max/mean per device (median, worst)   device-max vs position")
    for name, v in tot.items():
        r = [a for a, _ in v]
        print(
            "  %-11s %6.3f  %6.3f                           %6.3f"
            % (name, np.median(r), max(r), max(m for _, m in v) / base_max)
        )


if __name__ == "__main__":
    main()
