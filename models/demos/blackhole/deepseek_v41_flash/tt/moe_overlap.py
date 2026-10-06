# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared expert overlapped with the unified-MoE dispatch on Tensix sub-devices (DSV41_MO_OVERLAP=1, default OFF).

The layer's shared expert (own-token rows, replicated weights) is independent of the routed path, and ``dispatch`` is a fabric-bound op that needs only a
handful of worker cores.  Like models/demos/deepseek_v3_d_p/tt/moe/tt_moe.py, the Tensix grid is split into a one-row "dispatch" sub-device (row 0) and a
"shared expert" sub-device (the remaining rows); inside the window [load manager .. clear manager] the dispatch runs on sub-device 0 and the shared expert
matmuls / GLU on sub-device 1, concurrently.  ttnn forbids loading / clearing a sub-device manager inside a trace capture, so the traced chunk forward is
captured as SEVERAL traces split at the load / clear points (``SegTrace``, the DSV3 ``SubDeviceTraceController`` scheme) and replayed segment by segment.
"""

import math
import os

import ttnn

TILE = 32


def enabled():
    return os.environ.get("DSV41_MO_OVERLAP", "0") == "1"


def prep_enabled():
    """build-time: split the shared-expert weights (needed by ``enabled()`` runs; DSV41_MO_OVERLAP=prep builds them but leaves the overlap off, for in-process A/B)"""
    return os.environ.get("DSV41_MO_OVERLAP", "0") in ("1", "prep")


def _divisor_le(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _subblock(pm, pn, cap=4):
    best = (1, 1)
    for h in range(1, pm + 1):
        if pm % h:
            continue
        for w in range(1, pn + 1):
            if pn % w == 0 and h * w <= cap and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def mm_cfg(grid_xy, m_tiles, k_tiles, n_tiles):
    gx, gy = grid_xy
    pm = math.ceil(m_tiles / gy)
    pn = math.ceil(n_tiles / gx)
    sh, sw = _subblock(pm, pn)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=_divisor_le(k_tiles, 8),
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


class SegTrace:
    """Chunk trace captured as segments split at sub-device manager load / clear (the manager cannot be switched inside a capture).
    Drop-in for the single ``begin_trace_capture`` .. ``end_trace_capture`` / ``execute_trace`` of the prefill chunk forward.
    """

    def __init__(self, md):
        self.md = md
        self.prog = []  # ("trace", tid) | ("load", mgr) | ("clear", None)
        self.cur = None
        self.capturing = False

    def begin(self):
        self.prog = []
        self.capturing = True
        self.cur = ttnn.begin_trace_capture(self.md, cq_id=0)

    def end(self):
        ttnn.end_trace_capture(self.md, self.cur, cq_id=0)
        self.prog.append(("trace", self.cur))
        self.cur = None
        self.capturing = False

    def _split(self, kind, mgr):
        ttnn.end_trace_capture(self.md, self.cur, cq_id=0)
        self.prog.append(("trace", self.cur))
        if kind == "load":
            self.md.load_sub_device_manager(mgr)
        else:
            self.md.clear_loaded_sub_device_manager()
        self.prog.append((kind, mgr))
        self.cur = ttnn.begin_trace_capture(self.md, cq_id=0)

    def load(self, mgr):
        if self.capturing:
            self._split("load", mgr)
        else:
            self.md.load_sub_device_manager(mgr)

    def clear(self):
        if self.capturing:
            self._split("clear", None)
        else:
            self.md.clear_loaded_sub_device_manager()

    def replay(self, blocking=False):
        for kind, p in self.prog:
            if kind == "trace":
                ttnn.execute_trace(self.md, p, cq_id=0, blocking=False)
            elif kind == "load":
                self.md.load_sub_device_manager(p)
            else:
                self.md.clear_loaded_sub_device_manager()
        if blocking:
            ttnn.synchronize_device(self.md)

    def release(self):
        loaded = False
        try:
            for kind, p in self.prog:
                if kind == "trace":
                    ttnn.release_trace(self.md, p)
                elif kind == "load":
                    self.md.load_sub_device_manager(p)
                    loaded = True
                else:
                    self.md.clear_loaded_sub_device_manager()
                    loaded = False
        finally:
            if loaded:
                self.md.clear_loaded_sub_device_manager()
            self.prog = []


class SDOverlap:
    """Process-wide: the two-sub-device manager and the trace splitter (``seg`` is set while a chunk trace is being captured)."""

    _inst = {}

    @classmethod
    def get(cls, md):
        if id(md) not in cls._inst:
            cls._inst[id(md)] = cls(md)
        return cls._inst[id(md)]

    def __init__(self, md):
        self.md = md
        g = md.compute_with_storage_grid_size()
        self.gx, self.gy = g.x, g.y
        rows = int(os.environ.get("DSV41_MO_DISPATCH_ROWS", "1"))
        d = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, rows - 1))})
        s = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        self.s_cores = s
        self.s_grid = (g.x, g.y - rows)
        self.mgr = md.create_sub_device_manager([ttnn.SubDevice([d]), ttnn.SubDevice([s])], 0)
        self.d_id, self.s_id = ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)
        self.seg = None  # SegTrace while capturing / None (eager or replay)
        self.loaded = False
        self._keep = (
            []
        )  # temporaries of the overlapped shared expert: freed only at the NEXT layer's window (the async kernels may still read them)

    def load(self):
        (self.seg.load if self.seg is not None else self.md.load_sub_device_manager)(self.mgr)
        self.loaded = True

    def clear(self):
        (self.seg.clear if self.seg is not None else self.md.clear_loaded_sub_device_manager)()
        self.loaded = False

    def split_weights(self, sh):
        """separate gate / up bf8 weights of a DSV41SharedExpertV2 (fused1d: w01 = [gate | up]) next to w01 (+24 MB / layer / chip)"""
        if getattr(sh, "wg", None) is None:
            n = sh.inter
            sh.wg = ttnn.slice(sh.w01, [0, 0, 0, 0], [1, 1, sh.dim, n], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sh.wu = ttnn.slice(sh.w01, [0, 0, 0, n], [1, 1, sh.dim, 2 * n], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def shared(self, sh, h):
        """Shared expert of ``h`` [1,1,M,D] bf16 on sub-device 1 (same maths / dtypes as prefill_layer.shared_big) -> [1,1,M,D] fp32."""
        self.split_weights(sh)
        m, kt, nt = h.shape[2] // TILE, sh.dim // TILE, sh.inter // TILE
        c1, c2 = mm_cfg(self.s_grid, m, kt, nt), mm_cfg(self.s_grid, m, nt, kt)
        mm = lambda x, w, pc, dt: ttnn.matmul(
            x,
            w,
            program_config=pc,
            compute_kernel_config=sh.ckc,
            dtype=dt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sub_device_id=self.s_id,
        )
        g, u = mm(h, sh.wg, c1, sh.mid_dtype), mm(h, sh.wu, c1, sh.mid_dtype)
        act = ttnn.multiply(
            g,
            u,
            input_tensor_a_activations=sh.act_a,
            input_tensor_b_activations=sh.act_b,
            dtype=sh.mid_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sub_core_grids=self.s_cores,
        )
        out = mm(act, sh.w2, c2, ttnn.float32)
        for t in self._keep:
            ttnn.deallocate(t)
        self._keep = [g, u, act]
        return out
