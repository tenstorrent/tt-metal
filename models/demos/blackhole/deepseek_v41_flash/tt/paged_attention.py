# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged decode attention of DeepSeek-V4.1-Flash (``DSV41_PAGED=1``): same projections / RoPE / output path as ``attention.py``, but the KV state
lives in the shared row-major pool of ``paged_ops.PagedKVPool`` and the attention itself is ``ttnn.transformer.sparse_sdpa`` over index rows
that ``paged_ops.paged_kv_step`` builds on the device (design: docs/superpowers/specs/2026-10-02-dsv41-kv-paged-capacity-design.md, sections 4-5).

  window-only layers : indices = the valid ring rows (<= 128) of the layer's static ring region.
  compressed layers  : indices = ring rows + the compressed latents of the page pool (all entries while <= 512 exist, otherwise the indexer's
                       top-512 ids, ``st["topk_ids"]``), translated through the user's page table.

Interface (GPT-OSS style): ``forward(x, st)`` with ``st`` from ``DSV41PagedStepState`` (position tensor, rope rows) and the persistent
``page_table`` tensor living in ``kvpool``.
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P
from models.demos.blackhole.deepseek_v41_flash.tt.attention import (
    HEAD_DIM,
    LOCAL_HEADS,
    N_GROUPS,
    NH,
    PAD_HEADS,
    DSV41Attention,
    DSV41CompressedAttention,
)

TOPK_COMP = 512


def paged_enabled():
    return os.environ.get("DSV41_PAGED", "0") == "1"


class _PagedMixin:
    def _paged_init(self, kvpool, ring_slot):
        self.kv = kvpool
        self.ring_slot = ring_slot
        md = self.mesh_device
        # sparse_sdpa attention sink: [1,1,1,H=32] bf16 RM per device (this column's heads in 1..8; row 0 is the kv "head"), pre-divided by the scale
        sink = (self._sink_raw.float() / self.scale).reshape(N_GROUPS, LOCAL_HEADS)
        s = torch.zeros(N_GROUPS, PAD_HEADS)
        s[:, 1:NH] = sink
        self.sp_sink = ttnn.from_torch(
            s.reshape(1, 1, 1, N_GROUPS * PAD_HEADS).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(self.rows, self.cols)),
        )
        self.ckc_sp = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("DSV41_ATTN_SDPA_FID", "HiFi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    # ---- the paged step -------------------------------------------------------------------------------------------------------------
    def _paged_attend(self, q, lat, st, ratio, src_off, topk_out):
        """q: [1,T,(9)32,512] RoPE'd tile (row 0 = kv vector, rows 1..8 = heads) -> attention output [1,T,32,512] (tile), same layout as SDPA decode."""
        T = self.T
        ids = st.get("topk_ids") if ratio else None
        idx = P.paged_kv_step(
            self.kv.pool,
            q,
            lat,
            st["pos"],
            self.kv.page_table,
            ids,
            ring_base=self.kv.ring_base(self.ring_slot),
            layer_key=self.ring_slot,
            ratio=ratio,
            src_off=src_off,
            topk_out=topk_out,
            ring_rows=self.kv.ring_rows,
            nq=getattr(self, "nq", 1),  # spec verify blocks: rows of one user are consecutive (user = row // nq)
        )
        q32 = ttnn.reshape(q, (1, T, PAD_HEADS, HEAD_DIM), (1, T, PAD_HEADS, HEAD_DIM))
        qh = ttnn.to_layout(ttnn.permute(q32, (0, 2, 1, 3)), ttnn.ROW_MAJOR_LAYOUT)  # [1,32,T,512] RM (heads x users)
        o = ttnn.transformer.sparse_sdpa(
            qh,
            self.kv.pool,
            idx,
            HEAD_DIM,
            kv_format=(
                ttnn.transformer.SparseKVFormat.BF16
                if self.kv.dtype == ttnn.bfloat16
                else ttnn.transformer.SparseKVFormat.FP8_E4M3
            ),
            scale=self.scale,
            k_chunk_size=128,
            compute_kernel_config=self.ckc_sp,
            attention_sink=self.sp_sink,
        )
        ttnn.deallocate(qh)
        ttnn.deallocate(idx)
        o = ttnn.permute(ttnn.to_layout(o, ttnn.TILE_LAYOUT), (0, 2, 1, 3))  # [1,T,32,512]
        return o


class DSV41PagedAttention(_PagedMixin, DSV41Attention):
    """Window-only layers (0, 1, MTP): paged ring."""

    def __init__(self, mesh_device, mesh_config, ccl_manager, w, freqs_cis, kvpool, ring_slot, users_per_row=4):
        super().__init__(mesh_device, mesh_config, ccl_manager, w, freqs_cis, users_per_row=users_per_row, max_seq=32)
        ttnn.deallocate(self.cache)
        self.cache = None
        self._sink_raw = w["attn_sink"]
        self._paged_init(kvpool, ring_slot)

    def load_window(self, kv_rows, S=None):
        """kv_rows [B, S, 512]: positions 0..S-1 of the window cache (position-indexed) -> ring slots pos % RING (host staging)."""
        S = kv_rows.shape[1]
        ring = torch.zeros(kv_rows.shape[0], self.kv.ring_rows, HEAD_DIM)
        for pos in range(max(0, S - P.WINDOW), S):
            ring[:, pos % self.kv.ring_rows] = kv_rows[:, pos]
        self.kv.stage_ring(self.ring_slot, ring)

    def load_ring(self, ring):
        """ring [B, 128, 512] in the reference layout (slot = pos % 128) -> ring region (host staging)."""
        self.kv.stage_ring(self.ring_slot, ring.float()[:, : self.kv.ring_rows])

    def forward(self, x, st):
        q, kv, k = self._qkv(x, st)
        ttnn.deallocate(kv)
        ttnn.deallocate(k)
        o = self._paged_attend(q, None, st, 0, 0, P.WINDOW)
        return self._finish(o, st)


class DSV41PagedCompressedAttention(_PagedMixin, DSV41CompressedAttention):
    """Compressed layers: paged ring + shared page pool. Owners (``comp_w`` given) also run the compressor and write the latent pages;
    readers (``source`` given) only read the pages of their source."""

    def __init__(
        self,
        mesh_device,
        mesh_config,
        ccl_manager,
        w,
        freqs_cis,
        ratio,
        comp_w,
        kvpool,
        ring_slot,
        src_layer,
        users_per_row=4,
        source=None,
        indexer=None,
    ):
        super().__init__(
            mesh_device,
            mesh_config,
            ccl_manager,
            w,
            freqs_cis,
            ratio,
            comp_w,
            users_per_row=users_per_row,
            max_comp=32,
            source=source,
        )
        ttnn.deallocate(self.cache)
        self.cache = None
        self.src_layer = src_layer
        self.indexer = indexer  # DSV41DecodeIndexer of index-source layers (owners also append their key), or None
        self._sink_raw = w["attn_sink"]
        self._paged_init(kvpool, ring_slot)
        assert P.SRC_RATIO[src_layer] == ratio

    def load_state(self, window_kv, comp_kv, kv_state, score_state, start_pos=None):
        """Seed: window_kv [B,128,512] ring (reference layout, slot = pos % 128), comp_kv [B,Lc,512] (owners; staged into the pool pages),
        compressor state as in the non-paged class. The pool itself is uploaded by ``kvpool.stage_commit()`` after all layers staged.
        """
        B = window_kv.shape[0]
        self.kv.stage_ring(self.ring_slot, window_kv.float().reshape(B, -1, HEAD_DIM)[:, : self.kv.ring_rows])
        md = self.mesh_device
        if self.source is None:
            self.kv.stage_comp(self.src_layer, comp_kv.float().reshape(B, -1, HEAD_DIM))
            if self.ratio > 1:
                rs = ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(self.rows, self.cols))
                mk = lambda i: ttnn.from_torch(
                    torch.cat([kv_state[:, i], score_state[:, i]], dim=-1)
                    .float()
                    .reshape(1, 1, B, 2 * HEAD_DIM)
                    .contiguous(),
                    device=md,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rs,
                )
                for i in range(self.ratio):
                    self.cs_state[i] = mk(i)
                if start_pos is not None:
                    self.prev_cs = mk((start_pos - 1) % self.ratio)

    def step_inputs(self, positions, st=None):  # the legacy host-built inputs are not used by the paged path
        raise NotImplementedError("use DSV41PagedStepState")

    def _compress_step(self, x, st):
        """as the base class, but keeps the pooled latent BEFORE RoPE (``_lat_pre``): the indexer key is derived from it."""
        self._lat_pre = None
        orig = self._rope_rows

        def capture(lat, c, s, name="SW"):
            self._lat_pre = (
                ttnn.clone(lat) if self.indexer is not None else lat
            )  # the fused RoPE rotates ``lat`` in place
            return orig(lat, c, s, name)

        self._rope_rows = capture
        try:
            return super()._compress_step(x, st)
        finally:
            del self._rope_rows

    def forward(self, x, st):
        lat = self._compress_step(x, st) if self.source is None else self.source.last_lat
        self.last_lat = lat
        q, kv, k = self._qkv(x, st)
        ttnn.deallocate(kv)
        ttnn.deallocate(k)
        if (
            self.indexer is not None
        ):  # index-source layer: (owners: append this step's key,) then refresh the selection shared by the next layers
            if self.source is None:
                self.indexer.append_key(self._lat_pre, st)
            st["topk_ids"] = self.indexer.forward(x, self._last_qr, st)
        o = self._paged_attend(
            q, lat if self.source is None else None, st, self.ratio, P.SRC_OFF[self.src_layer], P.WINDOW + TOPK_COMP
        )
        return self._finish(o, st)


class DSV41PagedStepState:
    """Position-derived inputs of the paged attention layers, computed on the device from the ``pos`` tensor (trace safe, no per-position
    mask table: causality / visibility is carried by the sparse indices). One object per layer kind (window / ratio 2 / ratio 1).

    Rope tables are [P, 512] bf16 (C = 1, S = 0 off the rotated dims) gathered with ``ttnn.embedding``: 1 KiB/position per table, 2 tables per kind
    (design doc section 2 lists the compact [P, 64] alternative for > 100k contexts). ``with_indexer``: also the indexer inputs (128-wide query / key
    rope rows, entry index of the key write, valid length)."""

    def __init__(self, attn, max_pos=None, with_indexer=False, per_user_valid=False):
        self.per_user_valid = (
            per_user_valid  # ragged batches: every user has its own number of valid entries (own position)
        )
        self.md, self.T, self.rows, self.cols = attn.mesh_device, attn.T, attn.rows, attn.cols
        self.ratio = getattr(attn, "ratio", 0)
        self.with_indexer = with_indexer and self.ratio > 0
        Pn = max_pos or attn.cos_tab.shape[0]
        self.max_pos = Pn
        pos = torch.arange(Pn)
        c, s = attn._rope_inputs(pos)
        rep = ttnn.ReplicateTensorToMesh(self.md)
        up = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16).contiguous(),
            device=self.md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.t = {"C": up(c), "S": up(s)}
        if self.with_indexer:
            self.t["iC"] = up(
                torch.cat([torch.ones(Pn, 64), c[:, HEAD_DIM - 64 :].float()], dim=-1)
            )  # 128-wide (index head dim), rope on the last 64
            self.t["iS"] = up(torch.cat([torch.zeros(Pn, 64), s[:, HEAD_DIM - 64 :].float()], dim=-1))

    def _gather(self, name, idx, shape):
        return ttnn.to_layout(
            ttnn.reshape(ttnn.embedding(idx, self.t[name], layout=ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT
        )

    def build(self, pos):
        """pos int32 [T] row-major device tensor -> st dict."""
        T, W = self.T, HEAD_DIM
        idx = ttnn.typecast(ttnn.reshape(pos, [T, 1]), ttnn.uint32)
        st = {"pos": pos}
        st["Ch"] = self._gather("C", idx, [1, T, 1, W])
        st["Sh"] = self._gather("S", idx, [1, T, 1, W])
        st["nSh"] = ttnn.neg(st["Sh"])
        if self.ratio:
            r = self.ratio
            posf = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(pos, [1, 1, 1, T]), ttnn.TILE_LAYOUT), ttnn.float32
            )  # arithmetic in tile layout
            col = lambda t, dt: ttnn.reshape(ttnn.to_layout(ttnn.typecast(t, dt), ttnn.ROW_MAJOR_LAYOUT), [T, 1])
            gi = col(
                ttnn.relu(ttnn.add(posf, float(1 - r))), ttnn.uint32
            )  # rope position of a freshly pooled latent: max(pos + 1 - ratio, 0)
            st["Cg"], st["Sg"] = self._gather("C", gi, [1, 1, T, W]), self._gather("S", gi, [1, 1, T, W])
            if self.with_indexer:
                st["C"], st["S"] = self._gather("iC", idx, [T, 1, 1, 128]), self._gather("iS", idx, [T, 1, 1, 128])
                st["iCg"], st["iSg"] = self._gather("iC", gi, [1, 1, T, 128]), self._gather("iS", gi, [1, 1, T, 128])
                st["ent"] = ttnn.reshape(
                    col(ttnn.floor(ttnn.multiply(posf, 1.0 / r)), ttnn.int32), [T]
                )  # entry of the group in progress
                nu = ttnn.floor(
                    ttnn.multiply(ttnn.add(posf, 1.0), 1.0 / r)
                )  # valid entries of EVERY user, [1,1,1,T] float32 tile
                # topk_large_indices takes ONE valid length: the largest of the batch (>= 512); users with fewer entries are masked per user
                # in ``DSV41DecodeIndexer.select`` (``st["nvalid"]``) and the index build drops ids >= the user's own N.
                vmax = nu if T == 1 else ttnn.max(nu, dim=-1, keepdim=True)
                st["valid"] = ttnn.to_layout(
                    ttnn.typecast(ttnn.clamp(ttnn.slice(vmax, [0, 0, 0, 0], [1, 1, 1, 1]), min=512.0), ttnn.uint32),
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                if self.per_user_valid:
                    nrm = ttnn.to_layout(nu, ttnn.ROW_MAJOR_LAYOUT)  # [1,1,1,T]
                    st["nvalid"] = ttnn.to_layout(
                        ttnn.reshape(nrm, [T, 1, 1, 1]), ttnn.TILE_LAYOUT
                    )  # float32 [T,1,1,1] tile
        return st
