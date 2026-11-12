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
CH = int(os.environ.get("DSV41_XROW_D2D_CH", "8192"))  # latent rows per chunk (16 MiB)
KH = 32  # halo blocks per stage (>= the layers of a halo part)
KG = int(os.environ.get("DSV41_XROW_D2D_KG", "128"))  # key blocks per group


def enabled():
    return os.environ.get("DSV41_XROW_D2D", "0") == "1"


def halos_enabled():
    return keys_enabled() and os.environ.get("DSV41_XROW_D2D_HALOS", "1") == "1"


def halo_pairs(src, cps):
    """[(part, donor slot, new slot)] of the tip's halo slots: the donor tip block ``src[-1]`` (halo A slot at [3], B at [4]) -> the receiving chunk plans' ``halo_a`` / ``halo_b``"""
    tip = src[-1]
    out = []
    for _, cp in cps:
        for pn, dslot, sslot in (("A", cp.halo_a, tip[3]), ("B", cp.halo_b, tip[4])):
            if dslot >= 0 and sslot >= 0:
                out.append((pn, int(sslot), int(dslot)))
    return out


def mode():
    """DSV41_XROW_D2D_MODE: ``rows`` (default): the slab is all_gathered over the ROW axis (cluster_axis 0; no persistent L1, one program per shape) and the destination row picks the donor row's
    block; ``p2p``: one ttnn.point_to_point per column pair (every cache entry parks a 64 B GlobalSemaphore in L1 on every worker core: 12 pairs x 8 columns x 4 specs = 24 KiB, which made the B=16
    prefill SDPA static CBs clash with L1 buffers at boot)"""
    return os.environ.get("DSV41_XROW_D2D_MODE", "rows")


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
    def __init__(self, model, key_store=None, halo_store=None):
        self.m = model
        self.md = model.md
        self.rows, self.cols = model.rows, model.cols
        self.ks = key_store
        self.mode = mode()
        assert self.mode in ("rows", "p2p"), self.mode
        md = self.md
        mc = ttnn.DRAM_MEMORY_CONFIG

        def z(shape, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=layout, device=md, memory_config=mc)

        self.G = z([1, 1, CH, PO.HEAD_DIM])
        self.OUT = z([1, 1, CH, PO.HEAD_DIM]) if self.mode == "p2p" else None
        self.INT = self._intermediate(self.G)
        self.kst = {}
        if self.ks is not None:
            for lid, st in self.ks.store.items():
                R = int(st.shape[2])
                a = z([KG, 1, R, XO.IDIM], ttnn.TILE_LAYOUT)
                b = z([KG, 1, R, XO.IDIM], ttnn.TILE_LAYOUT) if self.mode == "p2p" else None
                self.kst[lid] = (a, b, self._intermediate(a))
        self.hs = halo_store
        if self.hs is not None:
            st = next(iter(self.hs.store.values()))
            shp = [KH, 1, int(st.shape[2]), int(st.shape[3])]
            a = z(shp, ttnn.TILE_LAYOUT)
            b = z(shp, ttnn.TILE_LAYOUT) if self.mode == "p2p" else None
            self.hst = (a, b, self._intermediate(a))
        self.shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        self.stats = {}

    def _intermediate(self, like):
        """the p2p intermediate (packet staging) tensor of a transfer of ``like`` between any two rows (same spec for all pairs), or None (the op allocates one per call)"""
        if self.mode != "p2p":
            return None
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

    def _xfer(self, src, dst, sri, dr, inter):
        """move the donor row's shard of ``src`` to the receiving row -> (tensor to read on the receiving row, block offset of the donor row's data in it, tensor to free after its last use or None)"""
        if self.mode == "p2p":
            self.p2p(src, dst, sri, dr, inter)
            return dst, 0, None
        full = ttnn.all_gather(
            src, dim=0, cluster_axis=0, num_links=1, topology=ttnn.Topology.Ring
        )  # [rows * K, ...]: row r's shard at block r
        return full, sri * int(src.shape[0]), full

    def _ids(self, row, vals, width=CH, off=0):
        """uint32 RM [rows, 1, width] sharded over the mesh rows: ``vals`` (torch long [m]) at ``off`` on ``row``, SKIP everywhere else"""
        ids = torch.full((self.rows, 1, width), SKIP, dtype=torch.int64)
        if row is not None:
            ids[row, 0, off : off + vals.numel()] = vals
        return ttnn.from_torch(
            ids.to(torch.int32),
            device=self.md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard,
        )

    def _latent_chunk(self, sri, dr, src_ids, dst_ids):
        """one chunk (<= CH rows; None = all SKIP: warm-up); -> tensors to free after the next synchronize"""
        pool = self.m.pool
        e = torch.zeros(0, dtype=torch.long)
        d_s = self._ids(None if src_ids is None else sri, e if src_ids is None else src_ids)
        PO.paged_gather_rows(pool.pool, ttnn.reshape(d_s, [1, CH]), out=self.G)
        if self.mode == "p2p":
            self.p2p(self.G, self.OUT, sri, dr, self.INT)
            d_d = self._ids(None if dst_ids is None else dr, e if dst_ids is None else dst_ids)
            PO.paged_scatter_rows(pool.pool, self.OUT, d_d)
            return [d_s, d_d]
        full = ttnn.all_gather(
            self.G, dim=0, cluster_axis=0, num_links=1, topology=ttnn.Topology.Ring
        )  # [rows, 1, CH, 512]
        full = ttnn.reshape(full, [1, 1, self.rows * CH, PO.HEAD_DIM])
        d_d = self._ids(None if dst_ids is None else dr, e if dst_ids is None else dst_ids, self.rows * CH, sri * CH)
        PO.paged_scatter_rows(pool.pool, full, d_d)
        ttnn.deallocate(full)  # (enqueued behind the scatter, same queue)
        return [d_s, d_d]

    def move_latents(self, sri, dr, phys_src, phys_dst):
        """pool[dr][phys_dst[i]] = pool[sri][phys_src[i]] for the concatenated sources, in place; enqueued, NOT synchronised (-> tensors to free after the next synchronize)"""
        n = int(phys_src.numel())
        keep = []
        for a in range(0, n, CH):
            keep += self._latent_chunk(sri, dr, phys_src[a : a + CH], phys_dst[a : a + CH])
        return keep

    def _slots(self, st, stage, out, inter, sri, dr, src_slots, dst_slots, K):
        """stage[i] = st[src_slots[i]] on the donor row; moved to the receiving row; st[dst_slots[i]] = received block i there. None slots = all-SKIP tables (warm-up). -> tensors to free"""
        n = 0 if src_slots is None else len(src_slots)
        tg = XO._tab(self.md, {sri: src_slots or []}, {sri: list(range(n))}, K)
        XO.slot_copy(tg, st, stage, K, K)
        rcv, off, tmp = self._xfer(stage, out, sri, dr, inter)
        ts = XO._tab(self.md, {dr: [off + i for i in range(n)]}, {dr: dst_slots or []}, K)
        XO.slot_copy(ts, rcv, st, K, K)
        if tmp is not None:
            ttnn.deallocate(tmp)
        return [tg, ts]

    def move_keys(self, sri, dr, pairs):
        """key store slots: [(donor slot on row sri, new slot on row dr)] of every layer's store; enqueued, NOT synchronised (-> tensors to free after the next synchronize)"""
        keep = []
        for g0 in range(0, len(pairs), KG):
            grp = pairs[g0 : g0 + KG]
            for lid, st in self.ks.store.items():
                stage, out, inter = self.kst[lid]
                keep += self._slots(st, stage, out, inter, sri, dr, [s for s, _ in grp], [d for _, d in grp], KG)
        return keep

    def move_halos(self, sri, dr, parts):
        """tip halos: [(part, donor slot, new slot)]; the halo store [slots * NL, 1, 128, 512] holds the NL layers of a slot as consecutive blocks; enqueued, NOT synchronised"""
        keep = []
        stage, out, inter = self.hst
        for pn, sslot, dslot in parts:
            NL = len(self.hs.layers[pn])
            assert NL <= KH
            keep += self._slots(
                self.hs.store[pn],
                stage,
                out,
                inter,
                sri,
                dr,
                [sslot * NL + l for l in range(NL)],
                [dslot * NL + l for l in range(NL)],
                KH,
            )
        return keep

    def warm(self):
        """build every program used (gather / scatter / slot_copy / all_gather, or every p2p entry of ``p2p`` mode) with all-SKIP ids / tables (no pool or store data is touched)"""
        t0 = time.perf_counter()
        keep = self._latent_chunk(0, 1, None, None)
        if self.ks is not None:
            for lid, st in self.ks.store.items():
                stage, out, inter = self.kst[lid]
                keep += self._slots(st, stage, out, inter, 0, 1, None, None, KG)
        if self.hs is not None:
            stage, out, inter = self.hst
            for st in {id(v): v for v in self.hs.store.values()}.values():
                keep += self._slots(st, stage, out, inter, 0, 1, None, None, KH)
        if (
            self.mode == "p2p"
        ):  # every ordered row pair (a cache miss costs a mesh-wide barrier: never in the serving path)
            for sr in range(self.rows):
                for dr in range(self.rows):
                    if sr != dr:
                        self.p2p(self.G, self.OUT, sr, dr, self.INT)
                        for lid, (stage, out, inter) in self.kst.items():
                            self.p2p(stage, out, sr, dr, inter)
                        if self.hs is not None:
                            self.p2p(self.hst[0], self.hst[1], sr, dr, self.hst[2])
        ttnn.synchronize_device(self.md)
        for x in keep:
            ttnn.deallocate(x)
        return time.perf_counter() - t0
