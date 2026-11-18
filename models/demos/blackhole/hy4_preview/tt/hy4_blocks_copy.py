# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B45b (HY4_BLOCKS=1, default OFF): row-local page-list copy of the Hy4 shadow block store (tt/hy4_blocks_store.py).

Copies pool pages ``src`` -> ``dst`` INSIDE one mesh row, for every pool tensor (78 kv, 21 key, MTP kv/key), by a SINGLE launch of the b45x ``xrow_copy.cpp`` JIT kernel per (tensor, run of consecutive pages): src and dst
are the same pool tensor (disjoint page ranges: active region <-> shadow region), no stage tensor, no gather, no persistent L1 (generic_op program, same as slice_copy). Rows other than ``row`` skip via the control page.
Page unit math as hy4_xrow (kv pool: 64 rows of 704 B per pool page; key pool: 8 tile pages of 1088 B bfp8 / 2048 B bf16). The shadow pages are ordinary extra pages at the END of the pool tensors (pool pages = P + S).
"""

import os
import time

from models.demos.blackhole.hy4_preview.tt import hy4_xrow as X


def runs_of(pairs):
    """[(src_page, dst_page)] -> [(src0, dst0, n)] maximal runs consecutive in BOTH src and dst (order kept)."""
    out = []
    for s, d in pairs:
        if out and out[-1][0] + out[-1][2] == s and out[-1][1] + out[-1][2] == d:
            out[-1][2] += 1
        else:
            out.append([s, d, 1])
    return [tuple(r) for r in out]


def check_disjoint(pairs):
    src, dst = {s for s, _ in pairs}, {d for _, d in pairs}
    if len(dst) != len(pairs):
        raise ValueError("page copy: duplicate destination page")
    if src & dst:
        raise ValueError("page copy: a page is both source and destination")


def ref_copy(pool, pairs, units):
    """CPU reference: pool = ndarray [pages*units, ...]; returns a copy with dst pages = src pages."""
    out = pool.copy()
    for s, d in pairs:
        out[d * units : (d + 1) * units] = pool[s * units : (s + 1) * units]
    return out


class Hy4PageCopy:
    """tensors: list of (name, ttnn pool tensor[, kind]) as Hy4Xrow (kind 'kv' | 'key', default by last dim 656)."""

    def __init__(self, md, tensors, key_dtype="bfloat8_b"):
        import ttnn

        self.md = md
        self.rows = int(md.shape[0])
        self.ncores = int(os.environ.get("HY4_BLOCKS_CORES", "32"))
        self.batch = int(os.environ.get("HY4_BLOCKS_B", "16"))
        self.shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        self.key_page = X.KEY_PAGE_B[key_dtype]
        self.tensors, self.kind = [], {}
        for x in tensors:
            name, t = x[0], x[1]
            self.kind[name] = x[2] if len(x) > 2 else ("kv" if int(t.shape[-1]) == 656 else "key")
            self.tensors.append((name, t))
        self._ctls = {}
        self.stats = dict(copies=0, pages=0, seconds=0.0)

    def _ctl_row(self, row):
        """control page skipping every mesh row but ``row`` (cached; persistent small DRAM tensors, build before the traces)"""
        if row not in self._ctls:
            self._ctls[row] = X.Hy4Xrow._ctl(self, 0, row)
        return self._ctls[row]

    def copy(self, row, pairs, only=None):
        """pool[row][dst] = pool[row][src] for every (src, dst) page pair and every pool tensor; synchronous. Raises on error (caller then recomputes cold)."""
        import ttnn

        if not pairs:
            return 0.0
        check_disjoint(pairs)
        runs = runs_of(pairs)
        ctl = self._ctl_row(row)
        t0 = time.perf_counter()
        for name, t in self.tensors[:only]:
            u, pb = (X.KV_UNITS, X.KV_PAGE_B) if self.kind[name] == "kv" else (X.KEY_UNITS, self.key_page)
            for s0, d0, n in runs:
                X.Hy4Xrow._prog(self, t, t, ctl, n * u, s0 * u, d0 * u, pb)
        ttnn.synchronize_device(self.md)
        dt = time.perf_counter() - t0
        self.stats["copies"] += 1
        self.stats["pages"] += len(pairs)
        self.stats["seconds"] += dt
        return dt
