"""NoC link load per K step, from routing every flow of a config over the physical grid (no fitted constants).

Flows per K step: each in0 / in1 reader pulls its block from the interleaved banks (DRAM: the bank's endpoint for the
reader's NoC; interleaved L1: every worker core), and each mcast sender multicasts its block over its rectangle.
Routing: NOC0 goes +x then +y, NOC1 goes -y then -x, both wrapping (torus). A link = (noc, axis, x, y), the hop leaving
router (x, y) in that NoC's direction. The output is the most-loaded link's bytes per K step (and which NoC), so the
step can be charged max_link_bytes / link rate. Physical placement: logical worker (i, j) -> the first usable worker
columns / rows of the SoC descriptor (harvested rows on the n150 and the p150b's dispatch column are not known here,
so the first rows / columns are assumed).
"""
import numpy as np

ARCH = {
    "wh": dict(
        X=10,
        Y=12,
        cols=[1, 2, 3, 4, 6, 7, 8, 9],
        rows=[1, 2, 3, 5, 7, 8, 9, 10, 11],  # this n150: row 4 harvested (UMD SocDescriptor)
        chan=[
            [(0, 0), (0, 1), (0, 11)],
            [(0, 5), (0, 6), (0, 7)],
            [(5, 0), (5, 1), (5, 11)],
            [(5, 2), (5, 9), (5, 10)],
            [(5, 3), (5, 4), (5, 8)],
            [(5, 5), (5, 6), (5, 7)],
        ],
        banks=[
            (0, 2, 2),
            (0, 1, 1),
            (1, 0, 0),
            (1, 2, 2),
            (2, 1, 1),
            (2, 2, 2),
            (3, 0, 0),
            (3, 1, 1),
            (4, 2, 2),
            (4, 0, 0),
            (5, 0, 0),
            (5, 2, 2),
        ],
        grid=(8, 8),
    ),
    "bh": dict(
        X=17,
        Y=12,
        # bh-30 p150b (tt_umd SocDescriptor): Tensix columns 7 and 10 harvested; the 11x10 worker grid takes the first 11 of
        # the 12 remaining columns (the last one goes to dispatch)
        cols=[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15],
        rows=list(range(2, 12)),
        chan=[
            [(0, 0), (0, 1), (0, 11)],
            [(0, 2), (0, 10), (0, 3)],
            [(0, 9), (0, 4), (0, 8)],
            [(0, 5), (0, 7), (0, 6)],
            [(9, 0), (9, 1), (9, 11)],
            [(9, 2), (9, 10), (9, 3)],
            [(9, 9), (9, 4), (9, 8)],
            [(9, 5), (9, 7), (9, 6)],
        ],
        banks=[(0, 2, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1), (4, 2, 1), (5, 2, 1), (6, 2, 1), (7, 2, 1)],
        grid=(11, 10),
    ),
}


def _route(L, a, noc, s, t, b):
    X, Y = a["X"], a["Y"]
    x, y = s
    if noc == 0:
        while x != t[0]:
            L[0, 0, x, y] += b
            x = (x + 1) % X
        while y != t[1]:
            L[0, 1, x, y] += b
            y = (y + 1) % Y
    else:
        while y != t[1]:
            L[1, 1, x, y] += b
            y = (y - 1) % Y
        while x != t[0]:
            L[1, 0, x, y] += b
            x = (x - 1) % X


def _mcast(L, a, noc, s, x0, x1, y0, y1, b):
    """multicast over the physical rectangle [x0..x1] x [y0..y1]: to the near corner, then one row, then down each column"""
    if isinstance(_ONLY, int):
        return
    X, Y = a["X"], a["Y"]
    if noc == 0:
        _route(L, a, 0, s, (x0, y0), b)
        x = x0
        while x != x1:
            L[0, 0, x, y0] += b
            x = (x + 1) % X
        for x in range(x0, x1 + 1):
            y = y0
            while y != y1:
                L[0, 1, x, y] += b
                y = (y + 1) % Y
    else:
        _route(L, a, 1, s, (x1, y1), b)
        y = y1
        while y != y0:
            L[1, 1, x1, y] += b
            y = (y - 1) % Y
        for y in range(y0, y1 + 1):
            x = x1
            while x != x0:
                L[1, 0, x, y] += b
                x = (x - 1) % X


def _phys(a, i, j):
    return a["cols"][i], a["rows"][j]


_ONLY = None  # None: every flow; "nodram": all but the DRAM reads; int b: only the DRAM reads from bank b, weight 1


def _read(L, a, noc, reader, src):
    """one byte read by `reader`, spread evenly over the interleaved banks"""
    if src == 0:
        if _ONLY == "nodram":
            return
        for b, (ch, e0, e1) in enumerate(a["banks"]):
            if isinstance(_ONLY, int):
                if b == _ONLY:
                    _route(L, a, noc, a["chan"][ch][e0 if noc == 0 else e1], reader, 1.0)
                continue
            _route(L, a, noc, a["chan"][ch][e0 if noc == 0 else e1], reader, 1.0 / len(a["banks"]))
        return
    if isinstance(_ONLY, int):
        return
    elif src == 1:
        gx, gy = a["grid"]
        for i in range(gx):
            for j in range(gy):
                _route(L, a, noc, _phys(a, i, j), reader, 1.0 / (gx * gy))


def unit_loads(arch, fam, R, C, P, cores, gx, gy, src_a, src_b):
    """(U0, U1): link loads per byte of one in0 block / one in1 block (each reader's block, each sender's mcast)."""
    a = ARCH[arch]
    sh = (2, 2, a["X"], a["Y"])
    U0, U1 = np.zeros(sh), np.zeros(sh)
    ph = lambda i, j: _phys(a, i, j)
    if fam == "2d":  # in0: left column reads + mcasts along its row (NOC1); in1: top row reads + mcasts down (NOC0)
        for j in range(R):
            s = ph(0, j)
            _read(U0, a, 1, s, src_a)
            if C > 1:
                _mcast(U0, a, 1, s, ph(1, j)[0], ph(C - 1, j)[0], s[1], s[1], 1.0)
        for i in range(C):
            s = ph(i, 0)
            _read(U1, a, 0, s, src_b)
            if R > 1:
                _mcast(U1, a, 0, s, s[0], s[0], ph(i, 1)[1], ph(i, R - 1)[1], 1.0)
    elif fam in (
        "1d_in0",
        "1d_in1",
    ):  # one sender at (0,0) mcasts to the whole grid; every core reads the other operand
        rows = int(np.ceil(P / gx))
        cs = [(k % gx, k // gx) for k in range(P)]
        snd, own, ns, no, ss, so = (U0, U1, 1, 0, src_a, src_b) if fam == "1d_in0" else (U1, U0, 0, 1, src_b, src_a)
        s = ph(0, 0)
        _read(snd, a, ns, s, ss)
        if P > 1:
            _mcast(snd, a, ns, s, ph(0, 0)[0], ph(gx - 1, 0)[0], ph(0, 0)[1], ph(0, max(rows, gy) - 1)[1], 1.0)
        for i, j in cs:
            _read(own, a, no, ph(i, j), so)
    elif fam == "reuse":  # every core reads both (in0 on NOC0, in1 on NOC1)
        for k in range(int(cores)):
            c = ph(k % gx, k // gx)
            _read(U0, a, 0, c, src_a)
            _read(U1, a, 1, c, src_b)
    return U0.reshape(-1), U1.reshape(-1)


def link_bytes(g, d, part="all"):
    """max link bytes per K step for each row (0 for multicore / sharded-only). part: "all" (reads + multicasts), "read"
    (only the memory reads) or "mcast" (only the multicast traffic)."""
    global _mcast, _read
    keep = _mcast, _read
    if part == "read":
        _mcast = lambda *a, **k: None
    elif part == "mcast":
        _read = lambda *a, **k: None
    try:
        return _link_bytes(g, d, part)
    finally:
        _mcast, _read = keep


def _link_bytes(g, d, part):
    out = np.zeros(len(d))
    gx = d.grid_x.fillna(1).to_numpy(int)
    gy = d.grid_y.fillna(1).to_numpy(int)
    fuse = d.fuse_batch.fillna(0).to_numpy() == 1
    Mrows = np.where(fuse, g["B"] * g["Mt"], g["Mt"])  # a fused batch stacks its rows
    R2 = np.nan_to_num(np.ceil(Mrows / d.per_core_M.to_numpy(float)))
    C2 = np.nan_to_num(np.ceil(g["Nt"] / d.per_core_N.to_numpy(float)))
    keys = np.array(
        [
            f"{part}|{g['arch'][i]}|{g['fam'][i]}|{int(g['rd0'][i])}|{int(g['rd1'][i])}|{int(g['cores'][i])}|{gx[i]}|{gy[i]}|{g['src_a'][i]}|{g['src_b'][i]}|{int(R2[i])}|{int(C2[i])}"
            for i in range(len(d))
        ]
    )
    b0 = g["obh"] * g["kb"] * g["tb_a"]
    b1 = g["kb"] * g["obw"] * g["tb_b"]
    cache = {}
    for k in np.unique(keys):
        m = keys == k
        i = np.flatnonzero(m)[0]
        arch, fam = g["arch"][i], g["fam"][i]
        if fam == "multicore":
            continue
        if k not in cache:
            cores = int(g["cores"][i])
            R, C = (int(R2[i]), int(C2[i])) if fam == "2d" else (0, 0)
            cache[k] = unit_loads(
                arch, fam, R, C, cores, cores, int(gx[i]), int(gy[i]), int(g["src_a"][i]), int(g["src_b"][i])
            )
        U0, U1 = cache[k]
        nz = (U0 > 0) | (U1 > 0)
        if not nz.any():
            continue
        out[m] = (np.outer(b0[m], U0[nz]) + np.outer(b1[m], U1[nz])).max(axis=1)
    return out


def link_bytes_banked(g, d, pats):
    """max link bytes per K step, averaged over the K steps, with each step's DRAM reads coming only from the banks that
    step touches (bankload.patterns): bank camping concentrates the traffic on those banks' links."""
    global _ONLY
    out = np.zeros(len(d))
    gx = d.grid_x.fillna(1).to_numpy(int)
    gy = d.grid_y.fillna(1).to_numpy(int)
    fuse = d.fuse_batch.fillna(0).to_numpy() == 1
    Mrows = np.where(fuse, g["B"] * g["Mt"], g["Mt"])
    R2 = np.nan_to_num(np.ceil(Mrows / d.per_core_M.to_numpy(float)))
    C2 = np.nan_to_num(np.ceil(g["Nt"] / d.per_core_N.to_numpy(float)))
    b0 = g["obh"] * g["kb"] * g["tb_a"]
    b1 = g["kb"] * g["obw"] * g["tb_b"]
    keys = [
        f"{g['arch'][i]}|{g['fam'][i]}|{int(g['rd0'][i])}|{int(g['rd1'][i])}|{int(g['cores'][i])}|{gx[i]}|{gy[i]}|{g['src_a'][i]}|{g['src_b'][i]}|{int(R2[i])}|{int(C2[i])}"
        for i in range(len(d))
    ]
    parts = {}

    def decomp(i, k):
        if k in parts:
            return parts[k]
        global _ONLY
        arch, fam = g["arch"][i], g["fam"][i]
        cores = int(g["cores"][i])
        R, C = (int(R2[i]), int(C2[i])) if fam == "2d" else (0, 0)
        args = (arch, fam, R, C, cores, cores, int(gx[i]), int(gy[i]), int(g["src_a"][i]), int(g["src_b"][i]))
        try:
            _ONLY = "nodram"
            o0, o1 = unit_loads(*args)
            nb = len(ARCH[arch]["banks"])
            B0, B1 = [], []
            for b in range(nb):
                _ONLY = b
                u0, u1 = unit_loads(*args)
                B0.append(u0)
                B1.append(u1)
        finally:
            _ONLY = None
        parts[k] = (o0, o1, np.array(B0), np.array(B1))
        return parts[k]

    groups = {}
    for i in range(len(d)):
        if g["fam"][i] == "multicore":
            continue
        groups.setdefault((keys[i],) + pats[i], []).append(i)
    for gk, idx in groups.items():
        k, A0, sa, Bs0, sb, nb = gk
        o0, o1, B0, B1 = decomp(idx[0], k)
        idx = np.array(idx)
        acc = np.zeros(len(idx))
        for j in range(nb):  # the bank sets rotate with the K step; nb steps cover every rotation
            ua = o0 + B0[[(x + j * sa) % nb for x in A0]].mean(axis=0)
            ub = o1 + B1[[(x + j * sb) % nb for x in Bs0]].mean(axis=0)
            nz = (ua > 0) | (ub > 0)
            if nz.any():
                acc += (np.outer(b0[idx], ua[nz]) + np.outer(b1[idx], ub[nz])).max(axis=1)
        out[idx] = acc / nb
    return out
