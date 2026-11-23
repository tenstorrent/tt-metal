# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill -> decode handoff in one program: per cache set (one core each), the k / v conv window state.x_hist
([32, C] bf16, rows = given source rows, the rest zero) and the post-attn / post-mlp conv ring vectors (dense fp32
[C], each a source row widened to fp32).

prefill_model._handoff built these with ~15 slice / concat / copy programs per cache set and wrote the 384 ring
vectors from the host (one read, 384 host writes). The values are the same source elements (bf16 copied, or widened
to fp32 exactly), so the decode state is bit-equal.

Call: PfHandoff(device)(jobs), jobs: [(x_hist, xrows, vecs)] with xrows[d] = (src, row) or None (zero row d) and
vecs = [(dense fp32 ring tensor, src, row)]. Every src: bf16 TILE DRAM [.., rows, C]; row = absolute row.
"""

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

BLK = 8


class PfHandoff:
    def __init__(self, device):
        self.device = device

    @staticmethod
    def ok(t):
        return t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT and t.memory_config().buffer_type == ttnn.BufferType.DRAM and not t.is_sharded()  # fmt: skip

    def __call__(self, jobs):
        c = jobs[0][0].shape[-1]
        wt = c // 32
        vec_pages = F.pages(c)
        cores, crs = F.core_list(self.device, len(jobs))
        nv = max(len(v) for _, _, v in jobs)
        rt, io = {}, []
        for core, (xh, xrows, vecs) in zip(cores, jobs):
            assert xh.dtype == ttnn.bfloat16 and tuple(xh.shape)[-2:] == (32, c), xh.shape
            groups = {}  # (id(src), tile row) -> (src, [(r, kind, idx)])
            for d, sr in enumerate(xrows):
                if sr is not None:
                    src, row = sr
                    groups.setdefault((id(src), row // 32), (src, []))[1].append((row % 32, 0, d))
            for v, (ring, src, row) in enumerate(vecs):
                assert ring.dtype == ttnn.float32, ring.dtype
                groups.setdefault((id(src), row // 32), (src, []))[1].append((row % 32, 1, v))
            args = [xh.buffer_address(), len(vecs)] + [r.buffer_address() for r, _, _ in vecs]
            args += [len(groups)]
            for (_, trow), (src, rows) in groups.items():
                assert self.ok(src) and src.shape[-1] == c, (src.dtype, src.shape)
                args += [src.buffer_address(), trow, len(rows)] + [x for rr in rows for x in rr]
                io.append(src)
            rt[core] = args
            io += [xh] + [r for r, _, _ in vecs]
        named = F.fixed_rt(dict(wt=wt, blk=BLK, vec_pages=vec_pages), rt)
        k = F.kernel("pf_handoff.cpp", ttnn.ReaderConfigDescriptor(), named, rt, crs)
        cbs = [F.cb(0, ttnn.bfloat16, wt, cores=crs), F.cb(1, ttnn.float32, nv * vec_pages, cores=crs),
               F.cb(2, ttnn.bfloat16, BLK, cores=crs)]  # fmt: skip
        seen, uniq = set(), []
        for t in io:
            if id(t) not in seen:
                seen.add(id(t))
                uniq.append(t)
        F.run(uniq, [k], cbs)
