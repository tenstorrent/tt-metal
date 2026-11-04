# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16: the indexer-score + running top-512 phase (M-B2 section 2 S2 / S3; coordinator 2026-10-25 07:15:21Z),
standalone (one generic_op launch, traced by the caller). Kernels: kernels/ix/ixp_{s,r,c,f}.cpp (ixp_common.hpp has
the protocol); host CB / cycle model: tests/s16/ix_phase_model.py.

Per chip: R query rows (the chip's SP rows) x 32 indexer heads x 128, query-major q [1, 1, R x 32, 128] bf16 TILE
(query t = rows 32 t .. 32 t + 31: its heads); head weights w [1, 1, R, 32] bf16 ROW_MAJOR (the model's w x 32^-0.5 x
128^-0.5: relu(q.k / sqrt 128) w 32^-0.5 = relu(q.k) (w 32^-0.5 128^-0.5) since the scale is positive); pooled keys
[1, 1, P, 128] bf16 TILE (the slot's pools in pool order); output sel [1, 1, R, 512] uint32 ROW_MAJOR: per query the
512 pool ids with the largest scores among its visible pools (p < (CS + t + 1) / 4), any order, 0xFFFFFFFF padding.
Cores: members = an exact rectangle of 64 cores (NQ = ceil(R / 64) queries each, the decode I4 compute + the DM
filter), one streamer core outside it (keys read once per launch, multicast)."""

from __future__ import annotations

import math
import os

TOPK = 512
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "ix")


class IxPhase:
    def __init__(self, mesh, rows: int, pools: int, *, members=None, streamer=None, kc: int = 4, depth: int = 4,
                 cap: int = 1024, fidelity: str = "HiFi4", diag: int = 0):  # fmt: skip
        import ttnn

        self.ttnn, self.mesh = ttnn, mesh
        grid = mesh.compute_with_storage_grid_size()
        gx, gy = int(grid.x), int(grid.y)
        if members is None:  # v7: every core but the last column (one multicast rectangle); the streamer in that column
            mx = int(os.environ.get("GLM53_IXP_MEMBER_COLS", "") or (gx - 1))
            members = [ttnn.CoreCoord(x, y) for y in range(gy) for x in range(mx)]
            if streamer is None:
                streamer = ttnn.CoreCoord(mx, 0)
        if streamer is None:
            streamer = ttnn.CoreCoord(gx - 1, 0)
        assert int(streamer.x) < gx and int(streamer.y) < gy, (streamer, grid)
        self.members, self.streamer = list(members), streamer
        xs = sorted({int(c.x) for c in members})
        ys = sorted({int(c.y) for c in members})
        assert len(xs) * len(ys) == len(members) and xs == list(range(xs[0], xs[-1] + 1)) and ys == list(range(ys[0], ys[-1] + 1)), \
            "members must be an exact rectangle (one multicast)"  # fmt: skip
        self.rect = (xs[0], ys[0], xs[-1], ys[-1])
        self.rows, self.pools, self.kc, self.depth, self.cap = int(rows), int(pools), int(kc), int(depth), int(cap)
        self.nm = len(members)
        self.nq = -(-self.rows // self.nm)
        assert 1 <= self.nq <= 32, ("<= 32 resident queries per member (one score tile row each)", self.nq)
        assert self.cap >= 2 * TOPK and self.kc <= 4, (self.cap, self.kc)
        self.nkt = -(-self.pools // 32)
        self.nblk = -(-self.nkt // self.kc)
        self.fidelity = getattr(ttnn.MathFidelity, fidelity)
        self.diag = int(
            diag
        )  # timing diagnostics (wrong selections): 1 = no filter scan, 2 = no compute math, 3 = both
        self.calls = 0

    def _noc(self, c):
        p = self.mesh.worker_core_from_logical_core(c)  # as pf_stream.PfStreamMM._noc
        return int(p.x), int(p.y)

    def program(self, q, w, keys, sel, cs: int, cs_t=None, base_t=None):
        """cs_t / base_t (IXP_CS_TENSOR, the served layer): the chunk start and this chip's first row as one-element int32
        device tensors (expand_rows' start / base): one trace replays the launch for every chunk; cs is then ignored"""
        ttnn = self.ttnn
        crs = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])  # noqa: E731
        lo, hi = ttnn.CoreCoord(self.rect[0], self.rect[1]), ttnn.CoreCoord(self.rect[2], self.rect[3])
        mem = ttnn.CoreRangeSet([ttnn.CoreRange(lo, hi)])
        st = crs([self.streamer])
        both = ttnn.CoreRangeSet([ttnn.CoreRange(lo, hi), ttnn.CoreRange(self.streamer, self.streamer)])
        nq, kc = self.nq, self.kc
        cb = lambda idx, fmt, page, pages, ranges: ttnn.CBDescriptor(total_size=page * pages, core_ranges=ranges, format_descriptors=[  # noqa: E731
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)])  # fmt: skip
        bf, f32, u32 = ttnn.bfloat16, ttnn.float32, ttnn.uint32
        nqa = -(-nq // 2)  # NCRISC filters rows [0, nqa), BRISC [nqa, nq)
        self.nqa = nqa
        cbs = [cb(0, bf, 2048, 4 * nq, mem), cb(1, bf, 2048, nq, mem), cb(2, bf, 2048, self.depth * 4 * kc, mem),
               cb(3, bf, 2048, nq * kc, mem), cb(4, f32, 4096, 2 * 2 * kc, mem), cb(5, u32, nqa * self.cap * 8, 1, mem),
               cb(6, f32, 4096, 2 * 2 * kc, mem), cb(7, u32, max(1, nq - nqa) * self.cap * 8, 1, mem),
               cb(8, f32, 4096, 2 * kc, mem), cb(9, bf, 2048, 1, mem),
               cb(0, bf, 2048, 2 * 4 * kc, st)]  # fmt: skip
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=both, initial_value=0) for i in range(3)]
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        addr = lambda t: int(t.buffer_address())  # noqa: E731
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        brisc = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)
        ncrisc = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, math_approx_mode=False, fp32_dest_acc_en=True,
                                          dst_full_sync_en=False)  # fmt: skip
        x0, y0 = self._noc(lo)
        x1, y1 = self._noc(hi)
        sx, sy = self._noc(self.streamer)

        cst = cs_t is not None
        assert cst == (base_t is not None), "IXP_CS_TENSOR: both the start and the row base tensors"
        defs = ([("IXP_DIAG", str(self.diag))] if self.diag else []) + ([("IXP_CS_TENSOR", "1")] if cst else [])

        def k(src, ct, rt, cores, cfg):
            rta = ttnn.RuntimeArgs()
            for c, a in rt.items():
                rta[c[0]][c[1]] = [int(v) for v in a]
            return ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                         core_ranges=cores, compile_time_args=[int(v) for v in ct], runtime_args=rta,
                                         defines=defs, config=cfg)  # fmt: skip

        rt_r, rt_c, rt_f = {}, {}, {}
        for i, c in enumerate(self.members):
            q0 = min(i * nq, self.rows)
            n = max(0, min(nq, self.rows - q0))
            xy = (int(c.x), int(c.y))
            rt_r[xy] = [addr(q), addr(w), q0 if n else 0, n if n else 1, self.nblk, sx, sy, int(i == 0), addr(sel),
                        addr(cs_t) if cst else int(cs), self.pools] + ([addr(base_t)] if cst else [])  # fmt: skip
            rt_c[xy] = [self.nblk]
            rt_f[xy] = [addr(sel), q0 if n else 0, n, self.nblk, addr(cs_t) if cst else int(cs), self.pools] + (
                [addr(base_t)] if cst else []
            )
        s_xy = (int(self.streamer.x), int(self.streamer.y))
        kernels = [
            k(
                "ixp_s.cpp",
                [kc, self.depth, self.nm] + acc(keys),
                {s_xy: [addr(keys), self.nkt, x0, y0, x1, y1]},
                st,
                brisc,
            ),
            k(
                "ixp_r.cpp",
                [nq, kc, self.cap, nqa, self.depth]
                + acc(q)
                + acc(w)
                + acc(sel)
                + (acc(cs_t) + acc(base_t) if cst else []),
                rt_r,
                mem,
                brisc,
            ),
            k("ixp_c.cpp", [nq, kc], rt_c, mem, cc),
            k(
                "ixp_f.cpp",
                [nq, kc, self.cap, nqa] + acc(sel) + (acc(cs_t) + acc(base_t) if cst else []),
                rt_f,
                mem,
                ncrisc,
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs), [q, w, keys, sel] + (
            [cs_t, base_t] if cst else []
        )

    def __call__(self, q, w, keys, sel, cs: int = 0, cs_t=None, base_t=None):
        prog, io = self.program(q, w, keys, sel, cs, cs_t, base_t)
        self.ttnn.generic_op(io, prog)
        self.calls += 1
        return sel

    def estimate_us(self) -> float:
        """the host model's compute term (tests/s16/ix_phase_model.py at HiFi4, fp32 S)"""
        mm = 64 if self.fidelity == self.ttnn.MathFidelity.HiFi4 else 32
        per_blk = self.nq * (4 * self.kc * mm + 40) + self.nq * self.kc * mm + self.kc * 60 + 2 * 300  # v2
        return self.nblk * per_blk / 1350.0 + math.ceil(self.nkt * 8192 / 512e3)
