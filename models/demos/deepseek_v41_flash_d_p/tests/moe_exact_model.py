# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.55a (owner 2026-11-13, MoE exact replicas, step 1): the CPU model of the exact split and its copy deadline.

No device. tests/test_moe_exact_bound.py feeds it one profiled chunk's measured routed_experts_combine times and the
router's counts (MOE_COUNTS); it fits the per-chip expert time, then plans the exact split per layer and evaluates it
with copies that take time.

Terms
-----
* slot: one of a chip's experts (12 a chip on the 8x4); its home is that chip. counts[k][j]: tokens of chip k's slot j.
* tile: 32 rows; an expert with n tokens costs ceil(n / 32) row tiles of compute.
* per-chip expert time (the fitted model): a chip's own experts finish at
      reach_k = max(A x active experts, B x row tiles)
  A = one expert's weight read at the DRAM rate (the weight-bound floor), B = one row tile of one expert's GLU FFN
  (fitted). The layer's experts+combine program takes C0 + max_k(finish_k) (C0 fitted: the combine's tail and fixed
  costs).
* exact split (the owner's design): after the counts, per layer, the planner picks hot experts on the busiest chips
  and, for each, a receiving chip (least loaded, one replica a chip: the 13th slot) and how many of its tokens go
  there. The receiver computes the replica LAST, after its own experts (it reaches it at reach_r); the home computes the
  hot expert FIRST, so the weights leave the home from the start.
* copy: one expert's weights (19.9 MB bf4) from the home to the receiver, store and forward along the dimension-ordered
  torus path (axis 0 first, then axis 1; a tie at half a ring goes the + way), each link crossed once per hot expert
  (a drop-off tree). Time at the measured link rate (P.47: 0.71 ms one hop, ~0.765 ms four hops, i.e. 18.3 us an extra
  hop), slowed by the hot experts sharing a link (fair share: x the most experts on any link of the path), starting
  when the home starts the expert (after its earlier hot experts) plus one weight read (the gate/up blocks land in L1).
* deadline: the replica's compute may start on the first sub-block (4 tiles) as blocks arrive; the rest waits for the
  whole copy:  finish_r = max(A (active + 1), reach_r + B tiles_rep, t_land + B max(0, tiles_rep - 4)).
  A copy is "on time" when it lands before the receiver reaches it (t_land <= reach_r).
* max mode: the same planner with copies that take no time (t_land = 0): what the split can save at most.
* deadline mode: the planner knows the copy times (the counts and the mesh are known on device before dispatch) and
  splits so the receiver's late start is accounted for.
"""

from __future__ import annotations

import math

import numpy as np

T1_MS = 0.71  # P.47: one 19.9 MB expert copy, one hop, one link pair
HOP_MS = (0.765 - 0.71) / 3  # P.47: 0.75-0.78 ms at four hops
SUB_TILES = 4  # the flat expert's sub-block: up to 128 rows


def own_counts(chips: list, spc: int) -> np.ndarray:
    """[chips][chips x spc] (MOE_COUNTS file rows) -> [chips, spc]: each chip's own slots."""
    a = np.asarray(chips, dtype=np.int64)
    n = a.shape[0]
    return np.stack([a[k, k * spc : (k + 1) * spc] for k in range(n)])


def tiles(n) -> np.ndarray:
    return (np.asarray(n) + 31) // 32


def reach(own: np.ndarray, A: float, B: float) -> np.ndarray:
    """Every chip's own experts' finish (ms) with no replica: max(A x active, B x tiles)."""
    act = (own > 0).sum(1)
    return np.maximum(A * act, B * tiles(own).sum(1))


def fit(samples: list, A: float) -> dict:
    """samples: [(measured ms, own counts [chips, spc])]. Fits measured = C0 + max_k reach_k(A, B) for B (grid, then a
    finer grid) with C0 = the mean residual; A fixed (unidentifiable from the busiest chip, which is compute-bound)."""
    T = np.array([t for t, _ in samples], dtype=float)
    act = np.array([(o > 0).sum(1) for _, o in samples], dtype=float)  # [S, chips]
    til = np.array([tiles(o).sum(1) for _, o in samples], dtype=float)

    def sse(B):
        W = np.maximum(A * act, B * til).max(1)
        c0 = float((T - W).mean())
        r = T - W - c0
        return float((r**2).sum()), c0, W

    best = None
    for grid in (np.arange(0.0005, 0.08, 0.0005),):
        for B in grid:
            e, c0, _ = sse(B)
            if best is None or e < best[0]:
                best = (e, B, c0)
    lo = max(best[1] - 0.0005, 1e-6)
    for B in np.linspace(lo, best[1] + 0.0005, 201):
        e, c0, _ = sse(B)
        if e < best[0]:
            best = (e, B, c0)
    e, B, c0 = best
    _, _, W = sse(B)
    pred = c0 + W
    ss_tot = float(((T - T.mean()) ** 2).sum())
    return {
        "A_ms": A,
        "B_ms": float(B),
        "C0_ms": float(c0),
        "r2": 1.0 - e / ss_tot if ss_tot > 0 else float("nan"),
        "rmse_ms": math.sqrt(e / len(T)),
        "samples": len(T),
        "pred_ms": pred.tolist(),
    }


# ---------------------------------------------------------------------------------------------------------------- mesh
def _ring_steps(a: int, b: int, n: int) -> list:
    """Shortest steps from a to b on a ring of n (ties at n / 2: the + way)."""
    d = (b - a) % n
    if d == 0:
        return []
    if d <= n - d:
        return [+1] * d
    return [-1] * (n - d)


def path_links(src: int, dst: int, shape: tuple) -> list:
    """Directed links (from chip, to chip) of the dimension-ordered torus path src -> dst: axis 0 first, then axis 1."""
    R, C = shape
    r, c = divmod(src, C)
    r1, c1 = divmod(dst, C)
    out = []
    for s in _ring_steps(r, r1, R):
        nr = (r + s) % R
        out.append((r * C + c, nr * C + c))
        r = nr
    for s in _ring_steps(c, c1, C):
        nc = (c + s) % C
        out.append((r * C + c, r * C + nc))
        c = nc
    return out


# ------------------------------------------------------------------------------------------------------------- planner
def _key(f: np.ndarray) -> tuple:
    """The planner's objective: the busiest finish first, ties broken by the next ones (1 ns steps: a move that only
    evens two chips at the top still counts)."""
    top = np.sort(np.asarray(f, dtype=float))[::-1][:4]
    return tuple(round(float(v), 6) for v in top)


class Plan:
    """One layer's exact split. nh[k, j]: tokens chip k keeps of its slot j; rep[r] = (home k, slot j, tokens) on the
    receiver r; hot[k]: chip k's hot slots in compute order (first = first computed)."""

    def __init__(self, own: np.ndarray, shape: tuple, A: float, B: float, copies: bool, paths: int = 1):
        self.own = own
        self.nh = own.copy()
        self.shape = shape
        self.n = own.shape[0]
        self.A, self.B = A, B
        self.copies = copies
        self.paths = paths
        self.rep: dict = {}
        self.hot: dict = {}
        self.act = (own > 0).sum(1)
        self._land = {}

    # timing --------------------------------------------------------------------------------------------------------
    def home_reach(self, k: int, nh_row=None) -> float:
        row = self.nh[k] if nh_row is None else nh_row
        act = int(self.act[k])  # (a hot slot is still read: its weights feed the copy)
        return max(self.A * act, self.B * float(tiles(row).sum()))

    def t_src(self, k: int, j: int, hot=None) -> float:
        """When the home's copy of slot j starts flowing: one weight read after the home starts the expert, which comes
        after its earlier hot slots (each max(A, B x the tiles it keeps))."""
        t = 0.0
        for jj in (self.hot if hot is None else hot).get(k, []):
            if jj == j:
                break
            t += max(self.A, self.B * float(tiles(self.nh[k, jj])))
        return t + self.A

    def schedule(self, rep=None, hot=None) -> dict:
        """{receiver: t_land}: every hot slot's tree (the union of its receivers' dimension-ordered paths, each link
        once), the trees in order of their start (t_src); each link carries one copy at a time (FIFO: a copy takes
        T1 / paths of the link, starting when the link is free and the upstream link has started); a receiver's copy
        has landed when the last link of its path has finished, plus a hop's latency for each extra hop."""
        rep = self.rep if rep is None else rep
        hot = self.hot if hot is None else hot
        if not self.copies:
            return {r: 0.0 for r in rep}
        per = T1_MS / self.paths
        trees = {}
        for r, (k, j, _) in rep.items():
            trees.setdefault((k, j), []).append(r)
        order = sorted(trees, key=lambda kj: (self.t_src(kj[0], kj[1], hot), kj))
        free: dict = {}
        land = {}
        for k, j in order:
            t0 = self.t_src(k, j, hot)
            done: dict = {}  # link -> (start, end) of this tree's copy on it
            for r in sorted(trees[(k, j)], key=lambda r: len(path_links(k, r, self.shape))):
                p = path_links(k, r, self.shape)
                up_s, up_e = t0, t0
                for l in p:
                    if l not in done:
                        s = max(up_s, free.get(l, 0.0))
                        done[l] = (s, max(s + per, up_e))
                        free[l] = done[l][1]
                    up_s, up_e = done[l]
                land[r] = up_e + HOP_MS * max(len(p) - 1, 0)
        return land

    def recv_finish(self, r: int, n_rep: int, t_land: float, nh_row=None) -> float:
        own_r = self.home_reach(r, nh_row)
        tr = float(tiles(n_rep))
        f = max(self.A * (int(self.act[r]) + 1), own_r + self.B * tr)
        if self.copies:
            f = max(f, t_land + self.B * max(0.0, tr - SUB_TILES))
        return f

    def finishes(self, land=None, rep=None) -> np.ndarray:
        rep = self.rep if rep is None else rep
        land = self.schedule(rep) if land is None else land
        f = np.array([self.home_reach(k) for k in range(self.n)], dtype=float)
        for r, (k, j, x) in rep.items():
            f[r] = self.recv_finish(r, x, land[r])
        return f

    # planning ------------------------------------------------------------------------------------------------------
    def step(self, n_rcv: int = 16, n_exp: int = 2) -> bool:
        """One greedy move: from the busiest chip's piece of a hot slot (its own slot's kept tokens, or the replica it
        received), part of the tokens to one new receiver (a new replica of that slot, copied from the slot's home),
        chosen (piece, receiver, split) to minimise the layer's finishes (busiest first). The copy times of every
        replica are re-scheduled with the new tree; within one candidate they are taken at the current split.
        False when nothing improves the layer."""
        f = self.finishes()
        cur = _key(f)
        ks = int(np.argmax(f))
        pieces = []  # (home k, slot j, from the replica ks holds?)
        if ks in self.rep:
            k, j, _ = self.rep[ks]
            pieces.append((k, j, True))
        pieces += [(ks, int(j), False) for j in np.argsort(-self.nh[ks], kind="stable")[:n_exp] if self.nh[ks, j] > 0]
        free = [r for r in range(self.n) if r not in self.rep and r != ks]
        free.sort(key=lambda r: f[r])
        best = None
        for k, j, from_rep in pieces:
            for r in free[:n_rcv]:
                if r == k:
                    continue
                hot = dict(self.hot)
                if j not in hot.get(k, []):
                    hot[k] = hot.get(k, []) + [j]
                rep = dict(self.rep)
                rep[r] = (k, j, 1)
                land = self.schedule(rep, hot)
                others = self.finishes(land, self.rep) if self.copies else f.copy()
                n0 = int(self.rep[ks][2]) if from_rep else int(self.nh[ks, j])

                def pair(x):
                    if from_rep:
                        fh = self.recv_finish(ks, n0 - x, land[ks]) if n0 - x > 0 else self.home_reach(ks)
                    else:
                        row = self.nh[k].copy()
                        row[j] = n0 - x
                        if ks in self.rep:  # (a receiver giving away part of its own slot still computes its replica)
                            fh = self.recv_finish(ks, self.rep[ks][2], land[ks], row)
                        else:
                            fh = self.home_reach(k, row)
                    return fh, self.recv_finish(r, x, land[r])

                lo, hi = 1, n0  # smallest x with the receiver no faster than the source
                while lo < hi:
                    mid = (lo + hi) // 2
                    fh, fr = pair(mid)
                    if fr >= fh:
                        hi = mid
                    else:
                        lo = mid + 1
                for x in {max(1, lo - 1), lo}:
                    if x >= n0 and from_rep:
                        continue  # (a replica moved whole is a move, not a split)
                    fh, fr = pair(x)
                    g_f = others.copy()
                    g_f[ks], g_f[r] = fh, fr
                    g = _key(g_f)
                    if best is None or g < best[0]:
                        best = (g, k, j, from_rep, r, x, hot)
        if best is None:
            return False
        g, k, j, from_rep, r, x, hot = best
        old = (self.nh.copy(), dict(self.rep), dict(self.hot))
        if from_rep:
            k0, j0, x0 = self.rep[ks]
            self.rep[ks] = (k0, j0, x0 - x)
        else:
            self.nh[k, j] -= x
        self.rep[r] = (k, j, x)
        self.hot = hot
        if not _key(self.finishes()) < cur:  # (the exact schedule at the new split: keep the move only if it helps)
            self.nh, self.rep, self.hot = old
            return False
        return True

    def run(self) -> "Plan":
        while len(self.rep) < self.n - 1 and self.step():
            pass
        return self

    def report(self) -> dict:
        land = self.schedule()
        f = self.finishes(land)
        reps = []
        for r, (k, j, x) in sorted(self.rep.items()):
            tl = land[r]
            own_r = self.home_reach(r)
            hops = len(path_links(k, r, self.shape))
            # the source window: the home holds the expert's gate/up blocks in L1 while it computes it
            t0 = self.t_src(k, j)
            win = max(self.A, self.B * float(tiles(self.nh[k, j])))
            reps.append(
                {
                    "home": k,
                    "slot": j,
                    "recv": r,
                    "tokens": int(x),
                    "home_kept": int(self.nh[k, j]),
                    "hops": hops,
                    "t_src_ms": round(t0, 4),
                    "t_land_ms": round(tl, 4),
                    "reach_ms": round(own_r, 4),
                    "on_time": bool(tl <= own_r + 1e-12),
                    "src_window_short": bool(self.copies and tl - t0 > win + self.A),
                }
            )
        return {"finish_ms": f.tolist(), "busiest_ms": float(f.max()), "replicas": reps}


def plan_layer(own: np.ndarray, shape: tuple, A: float, B: float, copies: bool, paths: int = 1) -> dict:
    p = Plan(own, shape, A, B, copies, paths).run()
    return p.report()


def layer_bound(own: np.ndarray, shape: tuple, A: float, B: float, paths: int = 1) -> dict:
    """One layer: before (no replica), the exact split with free copies (max), with timed copies (deadline)."""
    before = reach(own, A, B)
    mx = plan_layer(own, shape, A, B, copies=False)
    dlp = Plan(own, shape, A, B, copies=True, paths=paths).run()
    dl = dlp.report()
    dlp.copies = False  # (the deadline plan with free copies: max mode keeps the better of the two)
    alt = float(dlp.finishes().max())
    if alt < mx["busiest_ms"]:
        mx = {"busiest_ms": alt, "replicas": dl["replicas"]}
    mean = float(before.mean())
    return {
        "before_ms": float(before.max()),
        "mean_ms": mean,
        "reach_ms": [round(float(v), 4) for v in before],
        "tokens_chip": own.sum(1).tolist(),
        "max_ms": mx["busiest_ms"],
        "deadline_ms": dl["busiest_ms"],
        "save_max_ms": float(before.max()) - mx["busiest_ms"],
        "save_deadline_ms": float(before.max()) - dl["busiest_ms"],
        "n_rep_max": len(mx["replicas"]),
        "n_rep_deadline": len(dl["replicas"]),
        "on_time": sum(r["on_time"] for r in dl["replicas"]),
        "src_window_short": sum(r["src_window_short"] for r in dl["replicas"]),
        "replicas_deadline": dl["replicas"],
    }


def layer_task(args) -> dict:
    """A pool worker's unit: (key, own counts, shape, A, B, paths) -> (key, layer_bound(...))."""
    key, own, shape, A, B, paths = args
    return key, layer_bound(np.asarray(own), tuple(shape), A, B, paths=paths)
