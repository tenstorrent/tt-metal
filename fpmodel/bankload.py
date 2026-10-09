"""DRAM banks touched per K step: interleaved tensors put page i in bank i mod NB, so a K step whose tiles all land in a
few banks (Kt sharing a factor with NB, narrow K blocks) runs on those banks' bandwidth, not the chip's.

in0 tile (m, k) is page m*Kt + k (k*Mt + m when A is transposed); in1 tile (k, n) is page k*Nt + n (n*Kt + k when B
is transposed). In one K step every in0 reader pulls its current block's rows and every in1 reader its block's columns,
over the same window of kb K tiles (the multicasts keep the cores in step)."""
import numpy as np

NB = {"wh": 12, "bh": 8}
_cache = {}


def _touched(nb, stride_r, rows, stride_k, kb):
    """distinct banks of {(r*stride_r + k*stride_k) mod nb : r in rows, k < kb}"""
    key = (nb, stride_r, rows, stride_k, kb)
    if key not in _cache:
        r = np.asarray(rows, np.int64)[:, None] * stride_r
        k = np.arange(min(kb, nb), dtype=np.int64)[None, :] * stride_k
        _cache[key] = len(np.unique((r + k) % nb))
    return _cache[key]


def _rows(readers, per_core, block, total):
    """row (or column) indices one K step touches: each reader's current block, readers per_core apart"""
    idx = set()
    for r in range(int(min(readers, total))):
        for i in range(int(min(block, total))):
            v = r * per_core + i
            if v < total:
                idx.add(v)
            if len(idx) >= 48:  # enough rows to cover every residue
                break
        if len(idx) >= 48:
            break
    return tuple(sorted(idx))


def banks_touched(g, d):
    """(banks touched by the in0 read, by the in1 read) per K step, and the bank count, per row"""
    arch = d.arch_.to_numpy()
    ta_ = d.transpose_a.fillna(0).to_numpy() if "transpose_a" in d else np.zeros(len(d))
    tb_ = d.transpose_b.fillna(0).to_numpy() if "transpose_b" in d else np.zeros(len(d))
    pcM = d.per_core_M.fillna(1).to_numpy()
    pcN = d.per_core_N.fillna(1).to_numpy()
    fuse = d.fuse_batch.fillna(0).to_numpy() == 1
    out_a, out_b, nb = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    for i in range(len(d)):
        n = NB[arch[i]]
        nb[i] = n
        Mt, Kt, Nt = int(g["Mt"][i]), int(g["Kt"][i]), int(g["Nt"][i])
        kb = int(g["kb"][i]) if np.isfinite(g["kb"][i]) and g["kb"][i] > 0 else Kt
        Mrows = (
            int(Mt * g["B"][i]) if fuse[i] else Mt
        )  # fused batches continue the rows (batch b starts at page b*Mt*Kt)
        iv = lambda v, dflt: int(v) if np.isfinite(v) and v > 0 else dflt
        rows = _rows(iv(g["rd0"][i], 1), iv(pcM[i], 1), iv(g["obh"][i], Mrows), Mrows)
        cols = _rows(iv(g["rd1"][i], 1), iv(pcN[i], 1), iv(g["obw"][i], Nt), Nt)
        if ta_[i]:
            out_a[i] = _touched(n, 1, rows, Mt, kb)  # stored K x M: page k*Mt + m
        else:
            out_a[i] = _touched(n, Kt, rows, 1, kb)
        if tb_[i]:
            out_b[i] = _touched(n, Kt, cols, 1, kb)  # stored N x K: page n*Kt + k
        else:
            out_b[i] = _touched(n, 1, cols, Nt, kb)
    return out_a, out_b, nb


def _set(nb, stride_r, rows, stride_k, kb):
    r = np.asarray(rows, np.int64)[:, None] * stride_r
    k = np.arange(min(kb, nb), dtype=np.int64)[None, :] * stride_k
    return tuple(int(v) for v in np.unique((r + k) % nb))


def patterns(g, d):
    """per row: (in0 banks of the first K step, in0 bank shift per K step, in1 banks, in1 shift, bank count)"""
    arch = d.arch_.to_numpy()
    ta_ = d.transpose_a.fillna(0).to_numpy() if "transpose_a" in d else np.zeros(len(d))
    tb_ = d.transpose_b.fillna(0).to_numpy() if "transpose_b" in d else np.zeros(len(d))
    pcM = d.per_core_M.fillna(1).to_numpy()
    pcN = d.per_core_N.fillna(1).to_numpy()
    fuse = d.fuse_batch.fillna(0).to_numpy() == 1
    out = []
    cache = {}
    for i in range(len(d)):
        n = NB[arch[i]]
        Mt, Kt, Nt = int(g["Mt"][i]), int(g["Kt"][i]), int(g["Nt"][i])
        kb = int(g["kb"][i]) if np.isfinite(g["kb"][i]) and g["kb"][i] > 0 else Kt
        Mrows = int(Mt * g["B"][i]) if fuse[i] else Mt
        iv = lambda v, dflt: int(v) if np.isfinite(v) and v > 0 else dflt
        key = (
            n,
            Mt,
            Kt,
            Nt,
            kb,
            Mrows,
            iv(g["rd0"][i], 1),
            iv(pcM[i], 1),
            iv(g["obh"][i], Mrows),
            iv(g["rd1"][i], 1),
            iv(pcN[i], 1),
            iv(g["obw"][i], Nt),
            bool(ta_[i]),
            bool(tb_[i]),
        )
        if key not in cache:
            rows = _rows(key[6], key[7], key[8], Mrows)
            cols = _rows(key[9], key[10], key[11], Nt)
            if ta_[i]:
                A, sa = _set(n, 1, rows, Mt, kb), kb * Mt
            else:
                A, sa = _set(n, Kt, rows, 1, kb), kb
            if tb_[i]:
                Bs, sb = _set(n, Kt, cols, 1, kb), kb
            else:
                Bs, sb = _set(n, 1, cols, Nt, kb), kb * Nt
            cache[key] = (A, sa % n, Bs, sb % n, n)
        out.append(cache[key])
    return out
