# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSV41_DEV_ENGRAM=1: the Engram rows of the decode verify block are produced INSIDE the verify trace.

Host per round (cheap): the token history cache write + the (W-1) committed lookback tokens of every user + the block's raw token ids, packed into the widened ``ctrl`` upload
(``fill_ctrl``).  Device, at the top of the trace (``DevEngramRows.rows``):
  1. engram_hash_dec.cpp  : ctrl -> ids [TTOT, 64] (global table rows, every chip hashes the whole block, one token per kernel instance)
  2. engram_gather_dec.cpp: rows of this chip's table shard (fp8 + e8m0 -> bf16), zero rows for ids owned by other chips, TOKEN-major output order
  3. reduce_scatter over the mesh rows (dim 2) + all_reduce over the mesh columns: chip (r, *) ends with the rows of the T tokens of mesh row r = ``rows_cat`` [T, NL*K*256] row-major
The tables (rows sharded over the 32 chips) are allocated once per process (``DevEngramTables``); prefill keeps the host table.
The numpy / python mirrors (``emu_ids``, ``emu_rows_order``) document the exact index math and are what tests/test_engram_dev_cpu.py checks against the shipped host path."""

import numpy as np
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import engram_table as et

KDIR = "models/demos/blackhole/deepseek_v41_flash/tt/engram_hash_kernels"
i32, u8, bf16 = ttnn.int32, ttnn.uint8, ttnn.bfloat16
K = et.K


def ctrl_width(ttot, utot, w=4):
    """ctrl words per row: [token, pos] + ttot raw ids + utot * (w - 1) compressed lookbacks, rounded up to 16 words (64 B)."""
    return -(-(2 + ttot + utot * (w - 1)) // 16) * 16


def fill_ctrl(ctrl, X, base, st, rows=None, w=4):
    """Fill the Engram words of ``ctrl`` (int32 numpy / torch [R, CW], R >= 1; every row gets the same words) for the block X [B, n] at positions base [B].
    ``st`` = numpy view of the host hash state: dict(cache=[users, max_seq] int, tmap=[V], dead=int).  Writes the block's compressed tokens into the history cache exactly like
    FastNgramHash / RaggedNgramHash do (positions base .. base + n - 1; stale speculative entries beyond the accepted ones are overwritten by the next round) and reads the w - 1 committed
    lookbacks of column 0.  Returns nothing."""
    ctrl = ctrl.numpy() if isinstance(ctrl, torch.Tensor) else ctrl
    X = (X.numpy() if isinstance(X, torch.Tensor) else np.asarray(X)).astype(np.int64)
    base = (base.numpy() if isinstance(base, torch.Tensor) else np.asarray(base)).reshape(-1).astype(np.int64)
    B, n = X.shape
    crow = np.arange(B) if rows is None else np.asarray(rows).astype(np.int64).reshape(B)
    cache, tmap, dead = st["cache"], st["tmap"], int(st["dead"])
    pos = base[:, None] + np.arange(n, dtype=np.int64)[None, :]
    cache[crow[:, None], pos] = tmap[X]
    ttot = B * n
    lb = np.empty((B, w - 1), dtype=np.int64)
    blocked = np.zeros(B, dtype=bool)
    for s in range(1, w):
        p = base - s
        v = cache[crow, np.maximum(p, 0)]
        blocked = blocked | (p < 0) | (v == dead)  # sticky
        lb[:, s - 1] = np.where(blocked, -1, v)
    ctrl[:, 2 : 2 + ttot] = X.reshape(1, -1).astype(np.int32)
    ctrl[:, 2 + ttot : 2 + ttot + B * (w - 1)] = lb.reshape(1, -1).astype(np.int32)


def emu_ids(ctrl_row, ttot, n, tmap, mult, primes, offsets, pad, w=4):
    """Python mirror of engram_hash_dec.cpp: ctrl page 0 (int list / array) -> ids [ttot, NL, (w-1)*H] (python int64 arithmetic, wrapping like the reference's int64).
    mult [NL, w], primes [NL, w-1, H], offsets [NL, (w-1)*H] numpy int64."""
    nl, h = mult.shape[0], primes.shape[2]
    out = np.zeros((ttot, nl, (w - 1) * h), dtype=np.int64)
    for t in range(ttot):
        u, j = divmod(t, n)
        tk, blocked = [], False
        for s in range(w):
            v = int(tmap[int(ctrl_row[2 + t - s])]) if s <= j else int(ctrl_row[2 + ttot + u * (w - 1) + (s - j - 1)])
            blocked = blocked or v < 0
            tk.append(pad if blocked else v)
        for l in range(nl):
            rolling = 0
            for s in range(w):
                prod = np.int64(tk[s]) * np.int64(mult[l, s])  # int64 wrap-around, same as torch
                rolling = prod if s == 0 else np.bitwise_xor(rolling, prod)
                if s >= 1:
                    for hh in range(h):
                        out[t, l, (s - 1) * h + hh] = int(rolling) % int(primes[l, s - 1, hh]) + int(
                            offsets[l, (s - 1) * h + hh]
                        )
    return out


def emu_rows_order(ids, tables):
    """Python mirror of engram_gather_dec.cpp + the combine: ids [ttot, NL, K] global rows, tables [NL] objects with ``.rows(LongTensor) -> bf16`` -> [ttot, NL*K*256] bf16 in the
    rows_cat layout (token-major, layer, k)."""
    ttot, nl, k = ids.shape
    out = torch.zeros(ttot * nl * k, 256, dtype=torch.bfloat16)
    for j in range(ttot * nl * k):
        t, rem = divmod(j, nl * k)
        layer, kk = divmod(rem, k)
        out[j] = tables[layer].rows(torch.tensor([int(ids[t, layer, kk])]))[0]
    return out.reshape(ttot, nl * k * 256)


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


class DevEngramTables:
    """The two Engram tables sharded over all chips (about 3.17 GB / chip / layer) + the hash constants.  Built once (batch independent: kept across reconfigure)."""

    def __init__(self, mesh, host_rows, layer_ids, log=print, chips=None):
        import time

        self.mesh, self.layer_ids = mesh, list(layer_ids)
        assert len(self.layer_ids) == 2, "the gather kernel is built for exactly the 2 Engram layers"
        n = mesh.get_num_devices()
        self.n = n
        st = host_rows.engram.hash
        self.W = int(st.layout.max_ngram_size)
        self.tmap = st.token_map.numpy().astype(np.int64)
        self.dead = int(st.DEAD)
        self.pad = int(st.pad_id)
        mult = st.multipliers.numpy().astype(np.int64)
        primes = st.primes.numpy().astype(np.int64)
        offs = st.offsets.numpy().astype(np.int64)
        self.NL, self.H = int(mult.shape[0]), int(primes.shape[2])
        assert self.NL == 2 and self.H * (self.W - 1) == K, (self.NL, self.W, self.H)  # real checkpoint: W = 4 (2/3/4-grams), H = 8 heads
        self.mult, self.primes, self.offs = mult, primes, offs
        # hash constants (layout of engram_hash.cpp: [mult lo/hi (l,s)] [primes (l,s-1,h)] [offsets (l,col)] [pad])
        c = np.zeros(128, dtype=np.uint32)
        m64 = mult.view(np.uint64)
        mh = np.stack([(m64 & 0xFFFFFFFF).astype(np.uint32), (m64 >> 32).astype(np.uint32)], -1).reshape(-1)
        c[: mh.size] = mh
        o = mh.size
        pr = primes.reshape(-1)
        assert pr.max() < 2**31
        c[o : o + pr.size] = pr.astype(np.uint32)
        o += pr.size
        of = offs.reshape(-1)
        c[o : o + of.size] = of.astype(np.uint32)
        o += of.size
        c[o] = np.uint32(self.pad)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        up = lambda t: ttnn.from_torch(
            t,
            device=mesh,
            dtype=i32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.consts = up(torch.from_numpy(c.view(np.int32)).reshape(1, 128))
        tm = np.zeros((len(self.tmap), 16), dtype=np.int32)
        tm[:, 0] = self.tmap.astype(np.int32)
        self.tmap_dev = up(torch.from_numpy(tm))
        # the tables
        self.nrows = [int(host_rows.tables[l].w.shape[0]) for l in self.layer_ids]
        self.rpc = [et.rows_per_chip(nr, n) for nr in self.nrows]
        self.tables = []
        for l, rpc, nr in zip(self.layer_ids, self.rpc, self.nrows):
            rows, scales = et.alloc_table(mesh, rpc)
            self.tables.append((rows, scales, nr))
        ttnn.synchronize_device(mesh)
        self.load_s = 0.0
        for l, (rows, scales, nr), rpc in zip(self.layer_ids, self.tables, self.rpc):
            t0 = time.time()
            th, tw = et.load_table(mesh, rows, scales, host_rows.tables[l], rpc, chips)
            self.load_s += time.time() - t0
            log(f"DEV_ENGRAM table layer {l}: host-build {th:.1f} s, device-write {tw:.1f} s ({rpc} rows/chip)")
        self.info = ttnn.from_torch(
            torch.from_numpy(self._info()),
            device=mesh,
            dtype=i32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        self.state = dict(tmap=self.tmap, dead=self.dead)

    def _info(self):
        info = np.zeros((self.n, 16), dtype=np.int32)
        for c in range(self.n):
            for l, nr in enumerate(self.nrows):
                rpc = self.rpc[l]
                info[c, 2 * l] = c * rpc
                info[c, 2 * l + 1] = max(0, min(nr, (c + 1) * rpc) - c * rpc)
        return info


class DevEngramRows:
    """The in-trace rows producer for one verify block geometry: T = users per mesh row * n tokens per mesh row, TTOT = T * mesh rows."""

    def __init__(self, tabs, T, n, ncores=96, hash_cores=64, mode=0, num_links=2):
        self.tabs, self.mesh, self.T, self.n = tabs, tabs.mesh, T, n
        self.mrows, self.mcols = tuple(self.mesh.shape)
        self.TTOT = T * self.mrows
        assert self.TTOT % n == 0 and self.T % n == 0
        self.UTOT = self.TTOT // n
        self.W, self.NL, self.H = tabs.W, tabs.NL, tabs.H
        self.CW = ctrl_width(self.TTOT, self.UTOT, self.W)
        self.NROWS = self.TTOT * self.NL * K
        self.mode, self.num_links = mode, num_links
        grid = self.mesh.compute_with_storage_grid_size()
        allc = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
        # hash: one instance per token (2 instances per core)
        self.h_inst = min(self.TTOT, 2 * hash_cores)
        self.h_per = -(-self.TTOT // self.h_inst)
        self.h_cores = allc[: -(-self.h_inst // 2)]
        # gather
        self.g_cores = allc[:ncores]
        self.g_inst = 2 * len(self.g_cores)
        self.g_per = -(-self.NROWS // self.g_inst)
        self.ids = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.TTOT, 64]), i32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        self.part = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.NROWS, 256]), bf16, ttnn.ROW_MAJOR_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )

    @staticmethod
    def _cset(cores):
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores]
        )

    def hash(self, ctrl):
        cs = self._cset(self.h_cores)
        cbsz = (512 + self.CW * 4 + 63) // 64 * 64 + self.h_per * (self.W * 64 + 256)
        cbsz = (cbsz + 63) // 64 * 64
        cbs = [
            ttnn.CBDescriptor(
                total_size=cbsz,
                core_ranges=cs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=i32, page_size=cbsz)],
            )
            for i in range(2)
        ]
        tb = self.tabs
        common = [x.buffer_address() for x in (ctrl, tb.tmap_dev, tb.consts, self.ids)]
        acc = _acc(ctrl) + _acc(tb.tmap_dev) + _acc(tb.consts) + _acc(self.ids)
        kernels = []
        for inst, cfg in enumerate((ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor())):
            rt = ttnn.RuntimeArgs()
            for k, (cx, cy) in enumerate(self.h_cores):
                idx = 2 * k + inst
                t0 = idx * self.h_per
                cnt = max(0, min(self.h_per, self.TTOT - t0)) if idx < self.h_inst else 0
                rt[cx][cy] = [min(t0, self.TTOT), cnt]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/engram_hash_dec.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cs,
                    compile_time_args=[
                        self.mode,
                        self.TTOT,
                        self.n,
                        self.CW,
                        self.W,
                        self.NL,
                        self.H,
                        self.h_per,
                        inst,
                    ]
                    + acc,
                    runtime_args=rt,
                    common_runtime_args=common,
                    config=cfg,
                )
            )
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        prog.custom_program_hash = (0x3E8 << 40) | (
            hash((self.mode, self.TTOT, self.n, self.CW, self.h_per, len(self.h_cores), tuple(acc))) & ((1 << 40) - 1)
        )
        ttnn.generic_op([ctrl, tb.tmap_dev, tb.consts, self.ids], prog)
        return self.ids

    def gather(self, ids):
        cs = self._cset(self.g_cores)
        per = self.g_per
        maxtok = (per + self.NL * K - 1) // (self.NL * K) + 1
        cbsz = (maxtok * 256 + 64 + per * (256 + 64 + 512) + 63) // 64 * 64
        cbs = [
            ttnn.CBDescriptor(
                total_size=cbsz,
                core_ranges=cs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=u8, page_size=cbsz)],
            )
            for i in range(2)
        ]
        tb = self.tabs
        (t0_, s0_, _), (t1_, s1_, _) = tb.tables
        common = [x.buffer_address() for x in (ids, tb.info, t0_, s0_, t1_, s1_, self.part)]
        acc = _acc(ids) + _acc(tb.info) + _acc(t0_) + _acc(s0_) + _acc(t1_) + _acc(s1_) + _acc(self.part)
        kernels = []
        for inst, cfg in enumerate((ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor())):
            rt = ttnn.RuntimeArgs()
            for k, (cx, cy) in enumerate(self.g_cores):
                row0 = min((2 * k + inst) * per, self.NROWS)
                rt[cx][cy] = [row0, max(0, min(per, self.NROWS - row0))]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/engram_gather_dec.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cs,
                    compile_time_args=[self.TTOT, self.NL, K, per, inst] + acc,
                    runtime_args=rt,
                    common_runtime_args=common,
                    config=cfg,
                )
            )
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        prog.custom_program_hash = (0x3E9 << 40) | (
            hash((self.TTOT, len(self.g_cores), per, tuple(acc))) & ((1 << 40) - 1)
        )
        ttnn.generic_op([ids, tb.info, t0_, s0_, t1_, s1_, self.part], prog)
        return self.part

    def rows(self, ctrl):
        """ctrl: the widened device ctrl of the block (this chip's shard [T, CW]) -> [1, 1, T, NL*K*256] bf16 row-major on every chip of mesh row r = the rows of the T tokens of mesh row r."""
        ids = self.hash(ctrl)
        part = self.gather(ids)
        topo = ttnn.Topology.Linear
        tiled = (self.T * self.NL * K) % 32 == 0  # tile layout combine: 98-138 us vs 360-660 us for the 512 B row-major pages (tests/test_engram_dev_comb.py)
        if tiled:
            part = ttnn.to_layout(part, ttnn.TILE_LAYOUT)
        rs = ttnn.reduce_scatter(part, dim=2, cluster_axis=0, num_links=self.num_links, topology=topo)
        ar = ttnn.all_reduce(rs, cluster_axis=1, num_links=self.num_links, topology=topo)
        if tiled:
            ar = ttnn.to_layout(ar, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(ar, [1, 1, self.T, self.NL * K * 256])
