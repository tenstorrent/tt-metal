# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-to-device cross-row copy (DSV41_XROW_D2D=1, default off): the donor row's pool latents and key blocks travel over the fabric (``ttnn.point_to_point``, one op per mesh column: row ``sri`` ->
row ``dst`` of the same column) instead of through the host.

latents: per chunk of CH rows of the concatenated sources (all ``PO.SOURCES`` of the hit, one row stream): ``paged_gather_rows`` on the donor row (SKIP ids on the other rows) into a persistent slab G,
8 x p2p G -> OUT (receiving row), ``paged_scatter_rows`` of OUT into the pool at the receiving row's new pages (SKIP ids on the other rows). The ids are the only host data (2 x 4 B per row).
keys: ``slot_copy`` (tt/xrow_ops.py) gathers the donor slots of a group of KG blocks into a contiguous stage per layer, 8 x p2p, ``slot_copy`` scatters it into the receiving row's new slots
(columns hold DIFFERENT key data: every column's pair moves its own).
All buffers are persistent (allocated before the first trace capture, ``PagedApc.warm_xrow``), fixed shapes (one program each), reused in order on CQ 0 (no per-chunk synchronisation: the p2p
receiver-ready handshake plus the in-order device queues serialise reuse). ``warm`` builds every (sender row, receiver row, column, spec) p2p program (a program-cache miss costs 17-21 ms and a
mesh-wide barrier: never in the serving path).
"""
import os
import time

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as PO
from models.demos.blackhole.deepseek_v41_flash.tt import xrow_ops as XO

SKIP = 0xFFFFFFFF
CH = int(os.environ.get("DSV41_XROW_D2D_CH", "16384"))  # latent rows per chunk (16 MiB)
KG = int(os.environ.get("DSV41_XROW_D2D_KG", "128"))  # key blocks per group


def enabled():
    return os.environ.get("DSV41_XROW_D2D", "0") == "1"


def keys_enabled():
    return enabled() and os.environ.get("DSV41_XROW_D2D_KEYS", "1") == "1"


def key_pairs(src, cps):
    """[(donor slot, new slot)] of the blocks that get a key slot in the receiving row; ``src`` the donor blocks ((.., .., kslot, ..)*), ``cps`` [(block index, chunk plan with ``key_slot``)]"""
    out = []
    for i, cp in cps:
        if cp.key_slot >= 0:
            if src[i][2] < 0:
                raise RuntimeError(f"donor block {i} has no key slot")
            out.append((int(src[i][2]), int(cp.key_slot)))
    return out


def latent_rows(pt, hit, layer_ids, phys_src_fn, phys_dst_fn):
    """concatenated physical rows (donor row, receiving row) of the latents of the ``hit`` tokens of every source in ``layer_ids``, same order on both sides"""
    ps, pd = [], []
    for k, s_ in enumerate(PO.SOURCES):
        if s_ in layer_ids:
            n_ = hit // PO.SRC_RATIO[s_]
            ps.append(phys_src_fn(pt, k, torch.arange(n_)).long())
            pd.append(phys_dst_fn(s_, torch.arange(n_)).long())
    return torch.cat(ps), torch.cat(pd)


class XrowD2D:
    def __init__(self, model, key_store=None):
        self.m = model
        self.md = model.md
        self.rows, self.cols = model.rows, model.cols
        self.ks = key_store
        md = self.md
        mc = ttnn.DRAM_MEMORY_CONFIG

        def z(shape, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=layout, device=md, memory_config=mc)

        self.G = z([1, 1, CH, PO.HEAD_DIM])
        self.OUT = z([1, 1, CH, PO.HEAD_DIM])
        self.INT = self._intermediate(self.G)
        self.kst = {}
        if self.ks is not None:
            for lid, st in self.ks.store.items():
                R = int(st.shape[2])
                a, b = z([KG, 1, R, XO.IDIM], ttnn.TILE_LAYOUT), z([KG, 1, R, XO.IDIM], ttnn.TILE_LAYOUT)
                self.kst[lid] = (a, b, self._intermediate(a))
        self.shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        self.stats = {}

    def _intermediate(self, like):
        """the p2p intermediate (packet staging) tensor of a transfer of ``like`` between any two rows (same spec for all pairs), or None (the op allocates one per call)"""
        try:
            spec = ttnn.p2p_compute_intermediate_tensor_spec(
                like, ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(1, 0), ttnn.Topology.Linear
            )
            return ttnn.allocate_tensor_on_device(spec, self.md)
        except Exception as ex:  # (older binding: the op allocates it)
            print(f"xrow_d2d: no explicit p2p intermediate tensor ({ex!r})", flush=True)
            return None

    def p2p(self, src, dst, sri, dr, inter):
        for c in range(self.cols):
            ttnn.point_to_point(
                src,
                ttnn.MeshCoordinate(sri, c),
                ttnn.MeshCoordinate(dr, c),
                topology=ttnn.Topology.Linear,
                output_tensor=dst,
                intermediate_tensor=inter,
            )

    def _ids(self, row, vals):
        """uint32 RM [rows, 1, CH] sharded over the mesh rows: ``vals`` (torch long [m]) on ``row``, SKIP everywhere else"""
        ids = torch.full((self.rows, 1, CH), SKIP, dtype=torch.int64)
        if row is not None:
            ids[row, 0, : vals.numel()] = vals
        return ttnn.from_torch(
            ids.to(torch.int32),
            device=self.md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard,
        )

    def move_latents(self, sri, dr, phys_src, phys_dst):
        """pool[dr][phys_dst[i]] = pool[sri][phys_src[i]] for the concatenated sources, in place; enqueued, NOT synchronised (-> tensors to free after the next synchronize)"""
        pool = self.m.pool
        n = int(phys_src.numel())
        keep = []
        for a in range(0, n, CH):
            m = min(CH, n - a)
            d_s = self._ids(sri, phys_src[a : a + m])
            d_d = self._ids(dr, phys_dst[a : a + m])
            PO.paged_gather_rows(pool.pool, ttnn.reshape(d_s, [1, CH]), out=self.G)
            self.p2p(self.G, self.OUT, sri, dr, self.INT)
            PO.paged_scatter_rows(pool.pool, self.OUT, d_d)
            keep += [d_s, d_d]
        return keep

    def move_keys(self, sri, dr, pairs):
        """key store slots: [(donor slot on row sri, new slot on row dr)] of every layer's store; enqueued, NOT synchronised (-> tensors to free after the next synchronize)"""
        keep = []
        for g0 in range(0, len(pairs), KG):
            grp = pairs[g0 : g0 + KG]
            n = len(grp)
            tg = XO._tab(self.md, {sri: [s for s, _ in grp]}, {sri: list(range(n))}, KG)
            ts = XO._tab(self.md, {dr: list(range(n))}, {dr: [d for _, d in grp]}, KG)
            keep += [tg, ts]
            for lid, st in self.ks.store.items():
                stage, out, inter = self.kst[lid]
                XO.slot_copy(tg, st, stage, KG, KG)
                self.p2p(stage, out, sri, dr, inter)
                XO.slot_copy(ts, out, st, KG, KG)
        return keep

    def warm(self, latents=True):
        """build every p2p program used (all ordered row pairs, all columns, every spec) and the gather / scatter / slot_copy programs, with all-SKIP ids / tables (no pool or store data is touched)"""
        t0 = time.perf_counter()
        keep = []
        pool = self.m.pool
        d_s, d_d = self._ids(None, torch.zeros(0, dtype=torch.long)), self._ids(None, torch.zeros(0, dtype=torch.long))
        PO.paged_gather_rows(pool.pool, ttnn.reshape(d_s, [1, CH]), out=self.G)
        PO.paged_scatter_rows(pool.pool, self.OUT, d_d)
        keep += [d_s, d_d]
        for sr in range(self.rows):
            for dr in range(self.rows):
                if sr != dr:
                    self.p2p(self.G, self.OUT, sr, dr, self.INT)
                    for lid, (stage, out, inter) in self.kst.items():
                        self.p2p(stage, out, sr, dr, inter)
        if self.ks is not None:
            tab = XO._tab(self.md, {}, {}, KG)
            keep.append(tab)
            for lid, st in self.ks.store.items():
                stage, out, _ = self.kst[lid]
                XO.slot_copy(tab, st, stage, KG, KG)
                XO.slot_copy(tab, out, st, KG, KG)
        ttnn.synchronize_device(self.md)
        for x in keep:
            ttnn.deallocate(x)
        return time.perf_counter() - t0
