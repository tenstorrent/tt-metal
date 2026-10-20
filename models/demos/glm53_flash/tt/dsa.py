# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA (DeepSeek sparse attention, MLA with NoPE) layer for GLM-5.3-Flash on one device.

HF ``Glm5NextTextAttention`` (qk_rope_head_dim = 0, so no rotary anywhere):
    q = q_b(rmsnorm(q_a(x)))                      [T, 64, 256]
    c = rmsnorm(kv_a(x))                           [T, 512]   (the latent, cached per token)
    k_h = c @ Wkb1[h]^T, v_h = c @ Wkb2[h]^T       (kv_b = [64 x (256 k | 256 v), 512])
    o = softmax(q k^T / 16) v -> concat heads -> o_proj
The indexer selects the top 512 complete key pools (2048 tokens) plus the incomplete tail, so a
query at position a sees every visible token while floor((a+1)/4) <= 512, i.e. a <= 2050: there
the layer is exact causal attention (dense path). Beyond, the pooled-key indexer picks the keys
(sparse path, ``TtDSAIndexer`` + ``sparse_sdpa``).

Device design (section 5.3 of the design doc, single chip = Phase 1):
  * prefill: the latent is written to the paged cache with ``paged_fill_cache`` and the pooled-key
    index is maintained. Query rows [0, 2048) run the dense causal SDPA on materialized K/V; rows
    from 2048 on (they always see >= 2048 pooled tokens, so the top-k row is a full valid prefix)
    run the indexer selection + ``sparse_sdpa`` on the absorbed q. A prompt may arrive in
    consecutive chunks (``start`` = the chunk's first absolute position, a block multiple): the
    chunk's latent fills the pages from block ``start // 32``, the index continues from ``start``
    (the ring holds the previous chunk's tail) and the chunk rows attend over the full prefix, the
    dense rows through the windowed causal SDPA (K/V from the cached latent of [0, start) + the
    chunk's, Q offset ``start``) and the sparse rows exactly as in the single-shot path.
  * decode: absorbed MLA. q_lat[h] = q[h] @ Wkb1[h] (256 -> 512), then
    ``paged_flash_multi_latent_attention_decode`` over the latent cache for every user slot
    (exact through position 2050), then o[h] = o_lat[h] @ Wkb2[h]^T (512 -> 256), concat heads,
    o_proj. Users at position >= 2051 get their row replaced by the per-user indexer +
    ``sparse_sdpa`` result (host chunk geometry, not traced). The latent of the new token is
    appended with ``paged_update_cache`` first and the index is maintained for every user.
  * ``sparse_sdpa`` reads an index row up to its FIRST sentinel, so every row handed to it must be
    a contiguous valid prefix: that holds exactly because only queries with >= 2048 visible pooled
    tokens reach it.

Cache: ``[max_blocks, 1, block_size, 512]`` bf16, paged; page_table ``[B, max_blocks_per_seq]`` int32.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.custom_ops.index_score_decode.op import IndexScoreDecode
from models.demos.glm53_flash.custom_ops.paged_pool_gather.op import PagedPoolGather
from models.demos.glm53_flash.custom_ops.topk_rows.op import TopkRows
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.indexer import TtDSAIndexer, TtIndexCache, height_shard_rows
from models.demos.glm53_flash.tt.matmul_configs import DecodeLinear
from models.demos.glm53_flash.tt.mesh import GLMMesh, selector_tiles, single_chip

TILE = 32
CORES_PER_USER_DECODE = 4
# decode block: "fused" (GLM53_DSA_FUSED=1: stem matmul -> ONE dsa_decode_state program -> folded q_b.W1 matmul ->
# flash-MLA decode -> folded W2.o_proj matmul, 7 ops) | "composite" (default)
DSA_FUSED_DEFAULT = "fused" if os.environ.get("GLM53_DSA_FUSED", "0") == "1" else "composite"
# which absorb matrices the fused block folds into its projections: "both" (default) | "q" | "o" | "none"
# (an unfolded side runs the composite's per-head bmm + transposes; a precision / perf attribution knob)
DSA_FOLD = os.environ.get("GLM53_DSA_FOLD", "both")


def sparse_decode_enabled() -> bool:
    """The traced multi-user sparse decode (default). ``GLM53_SPARSE_DECODE=eager`` keeps the dense
    flash-MLA over the real cache + per-user eager sparse rows (host geometry) as the A/B reference."""
    return os.environ.get("GLM53_SPARSE_DECODE", "traced") != "eager"


class StageTimer:
    """Wall time per stage of the eager decode step (test instrumentation). ``mark(name)`` synchronizes
    the device and charges the time since the previous mark to ``name``; the allocator's DRAM bytes at
    every boundary are recorded in ``dram``. Never enable it in a traced or served path: the
    synchronizations serialize dispatch. Install with ``TtDSA.set_stage_timer``."""

    def __init__(self, device):
        self.device = device
        self.times: dict[str, float] = {}
        self.dram: dict[str, int] = {}
        self.order: list[str] = []
        self._t: float | None = None

    def reset(self):
        self.times, self.dram, self.order, self._t = {}, {}, [], None

    def start(self):
        ttnn.synchronize_device(self.device)
        self._t = time.perf_counter()

    def mark(self, name: str):
        ttnn.synchronize_device(self.device)
        now = time.perf_counter()
        if self._t is not None:
            self.times[name] = self.times.get(name, 0.0) + (now - self._t)
        if name not in self.order:
            self.order.append(name)
        try:
            mv = ttnn.device.get_memory_view(self.device, ttnn.BufferType.DRAM)
            self.dram[name] = mv.num_banks * mv.total_bytes_allocated_per_bank
        except Exception:  # pragma: no cover - informational
            pass
        self._t = time.perf_counter()

    @property
    def total(self) -> float:
        return sum(self.times.values())

    def report(self) -> str:
        gib = 2**30
        return "  ".join(
            f"{k} {self.times.get(k, 0.0) * 1e3:.2f}ms" + (f" [{self.dram[k] / gib:.2f}GiB]" if k in self.dram else "")
            for k in self.order
        )


def _compute(device, fidelity=ttnn.MathFidelity.HiFi2, fp32_acc=True, packer_l1_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


def _upload(device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def decode_position_tensors(
    device, host_positions: list[int], kpool: int, mesh: GLMMesh | None = None, partition: bool = False
):
    """(positions, positions // kpool, positions % kpool) as [B] int32 ROW_MAJOR device tensors
    (row-sharded over a multi-row ``mesh``: ``host_positions`` then lists ``rows * B`` entries, row-major;
    ``partition`` = the user-partitioned cache: the row's users are split over the columns too, chip (r, c)
    gets ``B_row / cols`` entries)."""

    def up(vals):
        t = torch.tensor(vals, dtype=torch.int32)
        return _meta_upload(device, t, ttnn.int32, mesh, partition)

    return up(host_positions), up([p // kpool for p in host_positions]), up([p % kpool for p in host_positions])


def _meta_upload(device, t: torch.Tensor, dtype, mesh: GLMMesh | None, partition: bool = False):
    """Host -> device ROW_MAJOR tensor; on a multi-row mesh dim 0 (the user list, row-major) is sharded over
    the rows, and with ``partition`` (user-partitioned cache) over the columns as well (dim 0 on both mesh
    axes: row r's slice is split again into ``cols`` pieces, chip (r, c) gets piece c)."""
    if mesh is not None and (mesh.rows > 1 or (partition and mesh.cols > 1)):
        return mesh.upload(
            t, dtype, rows=0, cols=0 if partition and mesh.cols > 1 else None, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _meta_mapper(mesh: GLMMesh | None, partition: bool):
    """The mesh mapper of ``_meta_upload`` for in-place refreshes (None on one chip)."""
    if mesh is None or not (mesh.rows > 1 or (partition and mesh.cols > 1)):
        return None
    return mesh.shard_2d(0, 0 if partition and mesh.cols > 1 else None)


@dataclass
class SparseDecodeMeta:
    """Per-step metadata of the traced multi-user sparse decode: one entry per user slot, derived on the
    host from the positions exactly like ``pool_ids`` / ``ring_slots`` and refreshed in place (``update``),
    so the captured decode graph reads the live values at every replay. With ``P`` = index_kpool, ``K`` =
    index_select_pools (512), ``n = (a + 1) // P`` complete pools and ``t = (a + 1) % P`` tail tokens of a user
    at position ``a`` (the token being decoded sees [0, a]):

    * ``valid_len[u]`` [1] uint32: the top-k's search width in pools, ``max(n, K)``: a sparse user ranks its
      ``n >= K`` complete pools only (the incomplete pool ``n`` and the stale rows beyond are never looked
      at); a dense user (``n < K``) gets a throw-away top-k over ``K`` columns that the gather ignores (it lists the
      user's pools in order).
    * ``tail_pool`` [B,1,1,1] uint32: ``n``, the user's incomplete pool, gathered into compact slot ``K``.
    * (the compact pool list itself is built by the gather kernel from the ranked ids and ``tail_pool``: the
      ranked pools for a sparse user, the pools ``0 .. K-1`` in order for a dense user (``n < K``, whose tail
      pool ``n`` then sits at slot ``n``), the tail pool at slot ``K``)
    * ``cur_pos`` [B] int32: the last attended row of the compact cache for the flash-MLA decode: ``a`` for a
      dense user (rows 0..a = tokens 0..a, exact dense attention) and ``P*K - 1 + t`` for a sparse user (the
      2048 selected tokens then the ``t`` tail tokens; the tail slot's other rows and the padding are hidden).
    A dense user is one with ``n < K``, i.e. ``a < P*K - 1`` (2047); from there on every top-K pool is a real
    complete pool and the two forms coincide (HF-exact through 2050 either way)."""

    valid_len: list[ttnn.Tensor]
    tail_pool: ttnn.Tensor
    cur_pos: ttnn.Tensor
    host: list[int]
    users_per_chip: int
    # [B, 1] uint32: the same per-user top-k bounds as ``valid_len`` in one tensor (one 4-byte page per user) for
    # the batched ``topk_large_indices`` call over all users' score rows (``GLM53_TOPK_BATCHED``, default on)
    valid_len_rows: ttnn.Tensor | None = None

    @staticmethod
    def host_values(positions: list[int], cfg: GLM53FlashConfig):
        """(valid [B], tail [B,1,1,1], cur_pos [B]) as host tensors."""
        P, K = cfg.index_kpool, cfg.index_select_pools
        B = len(positions)
        valid, tail, cur = [], [], []
        for u, a in enumerate(positions):
            n, t = (a + 1) // P, (a + 1) % P
            dense = n < K
            valid.append(max(n, K))
            tail.append(n)
            cur.append(a if dense else P * K - 1 + t)
        return (
            torch.tensor(valid, dtype=torch.int64),
            torch.tensor(tail, dtype=torch.int64).reshape(B, 1, 1, 1),
            torch.tensor(cur, dtype=torch.int32),
        )

    @staticmethod
    def _chips(mesh: GLMMesh | None, partition: bool) -> tuple[int, int]:
        """(row groups, column groups) the user list is split over."""
        if mesh is None:
            return 1, 1
        return mesh.rows, (mesh.cols if partition else 1)

    @classmethod
    def from_positions(
        cls, device, positions: list[int], cfg: GLM53FlashConfig, mesh: GLMMesh | None = None, partition: bool = False
    ) -> "SparseDecodeMeta":
        """``positions`` lists every user slot of the mesh (``rows * B`` on a multi-row mesh, row-major; with the
        user-partitioned cache (``partition``) chip (r, c) owns users ``[c*B/cols, (c+1)*B/cols)`` of row r)."""
        R, C = cls._chips(mesh, partition)
        B = len(positions) // (R * C)
        valid, tail, cur = cls.host_values(positions, cfg)
        # per local user u: a [R, C] tensor (chip (r, c)'s user u), sharded to one element per chip
        vl = valid.reshape(R, C, B)
        valid_t = [
            mesh.upload(
                vl[:, :, u].contiguous(), ttnn.uint32, rows=0, cols=1 if C > 1 else None, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            if mesh is not None and (R > 1 or C > 1)
            else ttnn.from_torch(
                vl[0, 0, u : u + 1].contiguous(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            for u in range(B)
        ]
        return cls(
            valid_len=valid_t,
            tail_pool=_meta_upload(device, tail, ttnn.uint32, mesh, partition),
            cur_pos=_meta_upload(device, cur, ttnn.int32, mesh, partition),
            host=list(positions),
            users_per_chip=B,
            valid_len_rows=_meta_upload(device, valid.reshape(-1, 1).contiguous(), ttnn.uint32, mesh, partition),
        )

    def update(self, positions: list[int], cfg: GLM53FlashConfig, mesh: GLMMesh | None = None, partition: bool = False):
        """Refresh every tensor in place for the new positions (trace-replay safe)."""
        R, C = self._chips(mesh, partition)
        B = self.users_per_chip
        assert len(positions) == R * C * B, (len(positions), R, C, B)
        valid, tail, cur = self.host_values(positions, cfg)
        mapper = _meta_mapper(mesh, partition)
        vmapper = mesh.shard_2d(0, 1 if C > 1 else None) if mesh is not None and (R > 1 or C > 1) else None

        def put(t: torch.Tensor, dtype, dst: ttnn.Tensor, m=mapper):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=m), dst
            )

        vl = valid.reshape(R, C, B)
        for u in range(B):
            put(
                vl[:, :, u].contiguous() if vmapper is not None else vl[0, 0, u : u + 1].contiguous(),
                ttnn.uint32,
                self.valid_len[u],
                vmapper,
            )
        put(tail, ttnn.uint32, self.tail_pool)
        if self.valid_len_rows is not None:
            put(valid.reshape(-1, 1).contiguous(), ttnn.uint32, self.valid_len_rows)
        put(cur, ttnn.int32, self.cur_pos)
        self.host = list(positions)


class TtLatentCache:
    """Paged latent cache for one DSA layer (and its page table) plus the pooled-key index, device-owned."""

    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        *,
        max_batch: int,
        max_seq_len: int,
        block_size: int = 32,
        dtype=ttnn.bfloat16,
        mesh: GLMMesh | None = None,
        partition_users: bool = False,
    ):
        """``max_batch`` user slots (per mesh row). ``partition_users`` on a multi-column mesh = the user-partitioned
        layout (doc/mesh/sparse_1m.md section 3): column chip c owns users ``[c*B/cols, (c+1)*B/cols)`` of its
        row, so the chip allocates ``users_per_chip + 1`` slots (the last one the prefill scratch slot) instead
        of ``max_batch``; decode addresses the local slots, prefill fills the scratch slot on every column and
        the user's slot on its owner (one prompt in flight per chip, chunks back to back)."""
        self.device = device
        self.mesh = mesh
        self.block_size = block_size
        self.max_batch = max_batch
        self.partition = bool(partition_users) and mesh is not None and mesh.cols > 1
        if self.partition:
            assert max_batch % mesh.cols == 0, (max_batch, mesh.cols)
            self.users_per_chip = max_batch // mesh.cols
            self.scratch: int | None = self.users_per_chip
            self.slots = self.users_per_chip + 1
        else:
            self.users_per_chip, self.scratch, self.slots = max_batch, None, max_batch
        self.blocks_per_seq = -(-max_seq_len // block_size)
        self.max_blocks = self.blocks_per_seq * self.slots
        self.dtype = dtype
        self.latent_dim = cfg.kv_lora_rank
        self.cache = ttnn.zeros(
            (self.max_blocks, 1, block_size, self.latent_dim),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # identity page table [slots, blocks_per_seq]: slot b owns blocks [b*bps, (b+1)*bps)
        pt = torch.arange(self.max_blocks, dtype=torch.int32).reshape(self.slots, self.blocks_per_seq)
        self.set_page_table(pt)
        self.rows_per_user = self.blocks_per_seq * block_size
        # pooled-key index (its own identity table; a slot's pools are contiguous regardless of the latent table)
        self.index = TtIndexCache(
            device,
            cfg,
            max_batch=self.slots,
            max_seq_len=self.rows_per_user,
            mesh=mesh,
            users_per_chip=self.users_per_chip if self.partition else None,
            scratch=self.scratch,
        )

    def set_page_table(self, page_table: torch.Tensor):
        """Any [slots, blocks_per_seq] mapping (vLLM-style); every block id must be unique. ``page_table`` is
        the decode table (the first ``users_per_chip`` rows when partitioned: the paged ops pair batch entry
        b with row b) and ``page_table_full`` the whole table."""
        assert tuple(page_table.shape) == (self.slots, self.blocks_per_seq), page_table.shape
        assert page_table.unique().numel() == page_table.numel(), "page table blocks must be unique"
        self.page_table_host = page_table.to(torch.int32).contiguous()
        self.page_table_full = ttnn.from_torch(
            self.page_table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        self.page_table = (
            ttnn.slice(self.page_table_full, [0, 0], [self.users_per_chip, self.blocks_per_seq])
            if self.partition
            else self.page_table_full
        )
        self._prefill_views: dict = {}

    def prefill_slot(self, user: int) -> int:
        """The slot every chip reads during the prefill of ``user`` (the scratch slot when partitioned)."""
        return self.scratch if self.partition else user

    def _prefill_views_for(self, user: int) -> list[ttnn.Tensor]:
        """[1, blocks_per_seq] page-table views a prefill of ``user`` fills (batch_idx 0): partitioned = the per-chip
        owner/scratch row (the user's slot on its owner column, the scratch slot elsewhere) and the scratch row."""
        key = ("fill", user)
        if key not in self._prefill_views:
            if not self.partition:
                self._prefill_views[key] = [
                    ttnn.slice(self.page_table_full, [user, 0], [user + 1, self.blocks_per_seq])
                ]
            else:
                owner, slot = user // self.users_per_chip, user % self.users_per_chip
                rows = torch.stack(
                    [self.page_table_host[slot if c == owner else self.scratch] for c in range(self.mesh.cols)]
                )  # [cols, bps]
                self._prefill_views[key] = [
                    self.mesh.upload(rows, ttnn.int32, cols=0, layout=ttnn.ROW_MAJOR_LAYOUT),
                    self.mesh.upload(
                        self.page_table_host[self.scratch : self.scratch + 1], ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                    ),
                ]
        return self._prefill_views[key]

    def fill(self, latent: ttnn.Tensor, user: int, start: int = 0):
        """latent [1, 1, T_pad, 512] (T_pad % block_size == 0) -> pages of ``user`` from position
        ``start`` (a block multiple): page-table view from block ``start // block_size`` on."""
        src = latent if latent.dtype == self.dtype else ttnn.typecast(latent, self.dtype)
        if start == 0 and not self.partition:
            ttnn.experimental.paged_fill_cache(self.cache, src, self.page_table, batch_idx=user)
            return
        if start % self.block_size:
            raise ValueError(f"fill start {start} must be a multiple of the block size {self.block_size}")
        b0 = start // self.block_size
        for full in self._prefill_views_for(user):
            view = full if b0 == 0 else ttnn.slice(full, [0, b0], [1, self.blocks_per_seq])  # device slice, no upload
            ttnn.experimental.paged_fill_cache(self.cache, src, view, batch_idx=0)
            if view is not full:
                ttnn.deallocate(view)

    def update(self, latent_rows: ttnn.Tensor, positions: ttnn.Tensor):
        """latent_rows [1, B, 1, 512] (one token per user), positions [B] int32 row-major."""
        sh = height_shard_rows(self.device, latent_rows)
        ttnn.experimental.paged_update_cache(self.cache, sh, update_idxs_tensor=positions, page_table=self.page_table)
        ttnn.deallocate(sh)

    def user_rows(self, user: int, n_rows: int) -> ttnn.Tensor:
        """The first ``n_rows`` (rounded up to whole blocks) latent rows of ``user`` in page-table
        order as a fresh TILE [1, 1, rows, 512] tensor."""
        n_blocks = min(self.blocks_per_seq, -(-n_rows // self.block_size))
        ids = self.page_table_host[self.prefill_slot(user), :n_blocks].tolist()
        if ids == list(range(ids[0], ids[0] + n_blocks)):  # contiguous (identity-like) fast path
            blocks = ttnn.slice(self.cache, [ids[0], 0, 0, 0], [ids[0] + n_blocks, 1, self.block_size, self.latent_dim])
        else:
            parts = [ttnn.slice(self.cache, [b, 0, 0, 0], [b + 1, 1, self.block_size, self.latent_dim]) for b in ids]
            blocks = ttnn.concat(parts, dim=0)
            for p in parts:
                ttnn.deallocate(p)
        return ttnn.reshape(blocks, [1, 1, n_blocks * self.block_size, self.latent_dim])

    def user_rows_rm(self, user: int, n_rows: int) -> ttnn.Tensor:
        """``user_rows`` as a fresh ROW_MAJOR [1, 1, rows, 512] tensor (the sparse_sdpa kv contract)."""
        rows = self.user_rows(user, n_rows)
        rm = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(rows)
        return rm


class TtDSA(LightweightModule):
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        attn_state_dict: dict[str, torch.Tensor],
        *,
        layer_idx: int,
        weight_dtype=ttnn.bfloat8_b,
        mesh: GLMMesh | None = None,
        fused_decode: str | bool | None = None,
    ):
        """``attn_state_dict``: raw checkpoint tensors of ``self_attn.`` (prefix stripped, fp8 dequantized).
        ``fused_decode``: "fused" | "composite" (default: the ``GLM53_DSA_FUSED`` knob); ``True`` = "fused".

        On a mesh the heads are split over the columns (``mesh.tp``, ``self.H`` = local heads): the
        latent stems (q_a, kv_a, their norms) and the indexer are replicated, q_b / kv_b / o_proj are
        head-sharded, the latent cache and the pooled-key index are replicated across the columns
        (Phase-2 first step; the user-partitioned cache comes next), and the output projection is
        all-reduced over the columns."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        tp = self.mesh.tp
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, Dq, Dv, L = cfg.num_attention_heads, cfg.qk_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        assert H % tp == 0, (H, tp)
        self.H_total = H
        self.H, self.Dq, self.Dv, self.L = H // tp, Dq, Dv, L
        self.scale = cfg.mla_softmax_scale
        sd = attn_state_dict
        up = self.mesh.upload
        self.w_qa = up(sd["q_a_proj.weight"].t(), weight_dtype)  # [4096,1536]
        self.qa_norm = up(sd["q_a_layernorm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.w_qb = up(sd["q_b_proj.weight"].t(), weight_dtype, cols=-1)  # [1536, H*256] head-major
        self.w_kva = up(sd["kv_a_proj_with_mqa.weight"].t(), weight_dtype)  # [4096, 512]
        self.kva_norm = up(sd["kv_a_layernorm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        wkb = sd["kv_b_proj.weight"].reshape(H, Dq + Dv, L)  # [H, 512, L]
        wkb1 = wkb[:, :Dq, :]  # k_nope[h] = c @ wkb1[h]^T   -> [H, 256, L]
        wkb2 = wkb[:, Dq:, :]  # v[h]      = c @ wkb2[h]^T   -> [H, 256, L]
        # prefill: K = c @ Wk^T for all heads at once: [T, L] @ [L, H*256]; same for V
        self.w_k_all = up(
            wkb1.transpose(0, 1).reshape(Dq, H, L).permute(2, 1, 0).reshape(L, H * Dq), weight_dtype, cols=-1
        )
        self.w_v_all = up(wkb2.permute(2, 0, 1).reshape(L, H * Dv), weight_dtype, cols=-1)
        # decode (absorbed): per-head [H, 256, L] for q -> latent, and [H, L, 256] for latent -> v space
        self.w_absorb_q = up(wkb1.reshape(1, H, Dq, L), weight_dtype, cols=1)
        self.w_absorb_o = up(wkb2.transpose(1, 2).reshape(1, H, L, Dv), weight_dtype, cols=1)
        self.w_o = up(sd["o_proj.weight"].t(), weight_dtype, cols=-2)  # [H*256, 4096] row-parallel
        self.indexer = TtDSAIndexer(
            device, cfg, {k[len("indexer.") :]: v for k, v in sd.items() if k.startswith("indexer.")}
        )
        # Decode stem: the five narrow projections of x (q_a 1536 | kv_a 512 | indexer k 128 | pool gate 128 |
        # indexer head weights Hi) as ONE [4096 x 2336] matmul (74 column blocks on 74 cores; the separate
        # matmuls sat on 4-48 cores for ~170 us) followed by tile-aligned column slices. bf16 weights keep
        # the indexer's numerics (its weights are bf16; GLM53_DSA_STEM_DTYPE=bfloat8_b halves the read).
        Lq, Hi, Di = cfg.q_lora_rank, self.indexer.Hi, self.indexer.D
        stem = torch.cat(
            [
                sd["q_a_proj.weight"].t().float(),
                sd["kv_a_proj_with_mqa.weight"].t().float(),
                sd["indexer.wk.weight"].t().float(),
                sd["indexer.index_kpool_compress_gate"].t().float(),
                sd["indexer.weights_proj.weight"].float().t() * Hi**-0.5,
            ],
            dim=-1,
        )
        stem_dtype = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}[
            os.environ.get("GLM53_DSA_STEM_DTYPE", "bfloat16")
        ]
        self.w_stem = up(stem, stem_dtype)  # [4096, Lq + L + Di + Di + Hi], replicated
        b = [0, Lq, Lq + L, Lq + L + Di, Lq + L + 2 * Di, Lq + L + 2 * Di + Hi]
        assert all(v % 32 == 0 for v in b), b
        self.stem_bounds = {
            "qa": (b[0], b[1]),
            "kva": (b[1], b[2]),
            "k": (b[2], b[3]),
            "g": (b[3], b[4]),
            "w": (b[4], b[5]),
        }
        # prefill rows from here on use the sparse path (they see >= 2048 pooled tokens: full index rows)
        self.sparse_row_start = cfg.index_topk  # 2048
        # decode: dense attention is HF-exact while floor((a+1)/4) <= index_topk/4, i.e. a <= 2050
        self.dense_exact_max_pos = cfg.index_topk + cfg.index_kpool - 2  # 2050
        self.hifi2 = _compute(device)
        self.hifi4 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)
        # decode-shaped (M <= 32) projections use the 1D mcast configs of tt/matmul_configs.py
        self.lin_qa = DecodeLinear(device, self.w_qa)
        self.lin_stem = DecodeLinear(device, self.w_stem)
        self.lin_qb = DecodeLinear(device, self.w_qb)
        self.lin_kva = DecodeLinear(device, self.w_kva)
        self.lin_o = DecodeLinear(device, self.w_o)
        grid = device.compute_with_storage_grid_size()
        self.sdpa_prefill_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False
        )
        self.full_grid = grid
        self._decode_cfgs: dict[int, ttnn.SDPAProgramConfig] = {}
        self._pad_cache: dict[tuple, ttnn.Tensor] = {}
        self._cu_windows: dict[int, ttnn.Tensor] = {}  # chunked-prefill SDPA window bounds per key length
        self.sdpa_compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.stage_timer: StageTimer | None = None
        if fused_decode is None:
            self.fused_decode = DSA_FUSED_DEFAULT
        elif isinstance(fused_decode, bool):
            self.fused_decode = "fused" if fused_decode else "composite"
        else:
            self.fused_decode = str(fused_decode)
        if self.fused_decode == "fused":
            # the dsa_decode_state program's constants (fp32 norm weights, the ape block, the selectors) and the
            # folded absorb weights: q_lat = q_resid @ (Wqb[h] Wkb1[h]) per head -> packed [1,1,B,H*L] (the
            # flash-MLA decode layout after one reshape), y = o_lat_packed @ stack_h(Wkb2[h]^T Wo[h]) (one matmul)
            idx = self.indexer
            f32 = lambda t: up(t.float().reshape(1, 1, 1, -1), ttnn.float32)
            self.qa_norm_w32 = f32(sd["q_a_layernorm.weight"])
            self.kva_norm_w32 = f32(sd["kv_a_layernorm.weight"])
            self.knorm_w32 = f32(sd["indexer.k_norm.weight"])
            self.knorm_b32 = f32(sd["indexer.k_norm.bias"])
            ape = torch.full((1, 1, TILE, idx.D), -1e9)
            ape[0, 0, : idx.P] = sd["indexer.index_kpool_compress_ape"].float().reshape(idx.P, idx.D)
            self.ape_block = up(ape, ttnn.float32)
            self.selectors = up(selector_tiles(), ttnn.bfloat16)
            wqb = sd["q_b_proj.weight"].t().float().reshape(Lq, H, Dq)  # [Lq, H, Dq]
            w_qb1 = torch.einsum("lhd,hdm->lhm", wqb, wkb1.float()).reshape(Lq, H * L)  # [Lq, H*L]
            self.w_qb1 = up(w_qb1, weight_dtype, cols=-1)
            wo = sd["o_proj.weight"].t().float().reshape(H, Dv, cfg.hidden_size)  # [H, Dv, hidden]
            w_o2 = torch.einsum("hdm,hmn->hdn", wkb2.transpose(1, 2).float(), wo).reshape(H * L, cfg.hidden_size)
            self.w_o2 = up(w_o2, weight_dtype, cols=-2)
            self.lin_qb1 = DecodeLinear(device, self.w_qb1)
            self.lin_o2 = DecodeLinear(device, self.w_o2)
        # the sparse-decode latent gather (model-local generic op) and its persistent compact output per batch size
        self.pool_gather = PagedPoolGather(device, pool_size=cfg.index_kpool)
        self._compact: dict[int, ttnn.Tensor] = {}
        # the per-row-bounded top-k over all users' score rows (model-local generic op) and its persistent output
        self.topk_rows = TopkRows(device, k=cfg.index_select_pools)
        self._top_buf: dict[int, ttnn.Tensor] = {}
        # the batched one-query-per-user indexer scorer (model-local generic op) and its persistent score rows
        self.index_scorer = IndexScoreDecode(device)
        self._score_buf: dict[int, ttnn.Tensor] = {}

    def set_stage_timer(self, timer: StageTimer | None):
        """Install (or remove, ``None``) the per-stage timer on this layer and its indexer."""
        self.stage_timer = timer
        self.indexer.stage_timer = timer

    def _mark(self, name: str):
        if self.stage_timer is not None:
            self.stage_timer.mark(name)

    def _decode_sdpa_config(self, batch: int) -> ttnn.SDPAProgramConfig:
        """Flash MLA decode splits each user's heads over cores and tree-reduces the partial results;
        its c_19 staging CB is sized (cores_per_user - 1) * (out + 2) fp32 tiles: 36 tiles * 4 KB * 11
        = 1.6 MB at 12 cores/user with the 512-wide latent. 4 cores/user keeps the op at ~1.2 MB."""
        if batch not in self._decode_cfgs:
            want = min(self.full_grid.x * self.full_grid.y, CORES_PER_USER_DECODE * batch)
            x = min(self.full_grid.x, want)
            y = max(1, min(self.full_grid.y, want // x))
            self._decode_cfgs[batch] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
                q_chunk_size=0,
                k_chunk_size=32,
                exp_approx_mode=False,
            )
        return self._decode_cfgs[batch]

    # ---------------------------------------------------------------- shared stems
    def _q_resid(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,4096] -> q_resid [1,1,T,1536] bf16 (q_a -> rmsnorm)."""
        qa = self.lin_qa(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return ttnn.rms_norm(qa, epsilon=self.cfg.rms_norm_eps, weight=self.qa_norm, compute_kernel_config=self.hifi4)

    def _q(self, q_resid: ttnn.Tensor) -> ttnn.Tensor:
        """q_resid [1,1,T,1536] -> q [1,1,T,H*256] bf16 (q_b)."""
        return self.lin_qb(q_resid, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)

    def _latent(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,4096] -> c [1,1,T,512] bf16 (kv_a -> rmsnorm)."""
        c = self.lin_kva(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return ttnn.rms_norm(c, epsilon=self.cfg.rms_norm_eps, weight=self.kva_norm, compute_kernel_config=self.hifi4)

    def _absorb_bmm_config(self, N: int, in0_block_w: int):
        """Decode-shaped per-head batched matmul: one core per head (H <= 24 blocks on a 12x2 grid), the
        whole [32 x N] output block per core. tests/test_bmm_sweep.py (chip 0): absorb_q 65.8 -> 24.1 us,
        absorb_o 106.4 -> 23.2 us (default dispatch puts them on 16 / 8 cores)."""
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(12, 2),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=N // 32,
        )

    def _absorb_q(self, q: ttnn.Tensor) -> ttnn.Tensor:
        """q [1,1,T,H*Dq] -> q_lat [1,H,T,L] bf16 (per-head q @ Wkb1)."""
        T = q.shape[-2]
        qh = ttnn.transpose(ttnn.reshape(q, [1, T, self.H, self.Dq]), 1, 2)  # [1,H,T,Dq]
        kw = {"program_config": self._absorb_bmm_config(self.L, 1)} if T <= 32 and self.H <= 24 else {}
        q_lat = ttnn.matmul(qh, self.w_absorb_q, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2, **kw)
        ttnn.deallocate(qh)
        return q_lat

    def _o_lat_to_heads(self, o_lat: ttnn.Tensor) -> ttnn.Tensor:
        """o_lat [1,H,T,L] -> concat-heads v-space [1,1,T,H*Dv] (per-head @ Wkb2^T)."""
        T = o_lat.shape[-2]
        kw = {"program_config": self._absorb_bmm_config(self.Dv, 4)} if T <= 32 and self.H <= 24 else {}
        o = ttnn.matmul(
            o_lat, self.w_absorb_o, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2, **kw
        )  # [1,H,T,Dv]
        return ttnn.reshape(ttnn.transpose(o, 1, 2), [1, 1, T, self.H * self.Dv])

    def _o_proj(self, o_heads: ttnn.Tensor) -> ttnn.Tensor:
        y = self.lin_o(o_heads, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return self.mesh.all_reduce_cols(y)  # TP partial -> replicated over the row's columns

    def _cu_window(self, n_keys: int) -> ttnn.Tensor:
        """``cu_window_seqlens`` = [0, n_keys] for the windowed (prefix-aware) causal SDPA: one window
        over the whole prefix. Chunk geometry, not data: built once per key length and cached."""
        if n_keys not in self._cu_windows:
            self._cu_windows[n_keys] = ttnn.from_torch(
                torch.tensor([0, n_keys], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        return self._cu_windows[n_keys]

    def _dense_attention(
        self, q: ttnn.Tensor, c: ttnn.Tensor, prefix: ttnn.Tensor | None = None, start: int = 0
    ) -> ttnn.Tensor:
        """Causal attention of q [1,1,T,H*Dq] (rows at absolute positions ``start + i``) over the latent
        of the full prefix: ``prefix`` [1,1,start,L] (the user's cached rows before this chunk, None
        when start == 0) followed by this chunk's c [1,1,T,L]; K/V are materialized from the latent.
        With a prefix the SDPA runs in windowed mode (one window over all keys, Q offset ``start``),
        which builds the causal mask on device at the global positions and is bit-identical to the
        plain causal kernel's rows [start, start + T). Returns concat-heads v-space [1,1,T,H*Dv]."""
        T = q.shape[-2]
        H, Dq, Dv = self.H, self.Dq, self.Dv
        if prefix is not None:
            c = ttnn.concat([prefix, c], dim=-2)  # [1,1,start+T,L]
        S = c.shape[-2]
        k = ttnn.linear(c, self.w_k_all, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,S,H*Dq]
        v = ttnn.linear(c, self.w_v_all, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,S,H*Dv]
        if prefix is not None:
            ttnn.deallocate(c)
        qh = ttnn.transpose(ttnn.reshape(q, [1, T, H, Dq]), 1, 2)
        kh = ttnn.transpose(ttnn.reshape(k, [1, S, H, Dq]), 1, 2)
        vh = ttnn.transpose(ttnn.reshape(v, [1, S, H, Dv]), 1, 2)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        windowed = {} if start == 0 else {"cu_window_seqlens": self._cu_window(S), "windowed_q_token_offset": start}
        o = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            is_causal=True,
            scale=self.scale,
            program_config=self.sdpa_prefill_cfg,
            compute_kernel_config=self.sdpa_compute,
            **windowed,
        )  # [1,H,T,Dv]
        ttnn.deallocate(qh)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)
        return ttnn.reshape(ttnn.transpose(o, 1, 2), [1, 1, T, H * Dv])

    # --------------------------------------------------------------------- prefill
    def prefill_state(self, x: ttnn.Tensor, cache: TtLatentCache, user: int, seq_len: int, start: int = 0):
        """The state half of ``prefill_forward`` without the attention output: the chunk's latent goes into
        the user's pages [start, start + T_pad) and the pooled-key index is maintained for positions
        [start, start + seq_len) (same contract as ``prefill_forward``: ``start % block_size == 0``,
        chunks in order). Used to seed long contexts synthetically (the 1 M-position cache test) and
        for prefix-state construction where the chunk's own attention output is not needed."""
        if start % cache.block_size:
            raise ValueError(f"start {start} must be a multiple of the cache block size {cache.block_size}")
        c = self._latent(x)
        cache.fill(c, user, start=start)
        ttnn.deallocate(c)
        self.indexer.maintain_prefill(x, cache.index, user, seq_len, start=start)

    def prefill_forward(
        self, x: ttnn.Tensor, cache: TtLatentCache, user: int, seq_len: int, start: int = 0
    ) -> ttnn.Tensor:
        """One user, one chunk: x row i is the token at absolute position ``start + i``
        (``start % block_size == 0``; ``start == 0`` = a new prompt). x [1,1,T_pad,4096] with
        T_pad % block_size == 0 and the logical ``seq_len`` <= T_pad; returns [1,1,T_pad,4096] (rows
        past seq_len are garbage). Fills the user's latent pages [start, start + T_pad) and maintains
        the pooled-key index for positions [start, start + seq_len) (the ring must hold the previous
        chunk's tail rows, i.e. chunks of a prompt are prefilled in order). Rows at absolute positions
        < 2048 run the dense causal attention over the full prefix (K/V from the cached latent rows +
        this chunk's); rows from 2048 on use the indexer + sparse_sdpa, exactly as the single-shot path."""
        T = x.shape[-2]
        if start % cache.block_size:
            raise ValueError(f"start {start} must be a multiple of the cache block size {cache.block_size}")
        end = start + seq_len
        split = min(T, max(0, self.sparse_row_start - start))  # chunk rows [0, split) are dense
        q_resid = self._q_resid(x)
        q = self._q(q_resid)  # [1,1,T,H*Dq]
        c = self._latent(x)  # [1,1,T,L]
        cache.fill(c, user, start=start)
        self.indexer.maintain_prefill(x, cache.index, user, seq_len, start=start)
        prefix = cache.user_rows(user, start) if start and split else None  # [1,1,start,L]
        if split == T:
            ttnn.deallocate(q_resid)
            o = self._dense_attention(q, c, prefix, start)
            ttnn.deallocate(q)
            ttnn.deallocate(c)
            if prefix is not None:
                ttnn.deallocate(prefix)
            return self._o_proj(o)
        # dense rows [0, split) (absolute positions [start, 2048))
        o_dense = None
        if split:
            o_dense = self._dense_attention(
                ttnn.slice(q, [0, 0, 0, 0], [1, 1, split, self.H * self.Dq]),
                ttnn.slice(c, [0, 0, 0, 0], [1, 1, split, self.L]),
                prefix,
                start,
            )
            if prefix is not None:
                ttnn.deallocate(prefix)
        ttnn.deallocate(c)
        # sparse rows [split, T) (absolute positions from max(start, 2048))
        idx = self.indexer
        x_s = ttnn.slice(x, [0, 0, split, 0], [1, 1, T, x.shape[-1]]) if split else x
        qr_s = ttnn.slice(q_resid, [0, 0, split, 0], [1, 1, T, q_resid.shape[-1]]) if split else q_resid
        q_i, w_i = idx.queries_and_weights(x_s, qr_s)
        ttnn.deallocate(q_resid)
        indices = idx.select(
            q_i,
            w_i,
            cache.index,
            cache.index.prefill_slot(user),
            query_start=start + split,
            num_pools_valid=end // idx.P,
        )
        ttnn.deallocate(q_i)
        ttnn.deallocate(w_i)
        q_lat = self._absorb_q(
            ttnn.slice(q, [0, 0, split, 0], [1, 1, T, self.H * self.Dq]) if split else q
        )  # [1,H,T-split,L]
        ttnn.deallocate(q)
        q_rm = ttnn.to_layout(q_lat, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(q_lat)
        kv_rm = cache.user_rows_rm(user, start + T)
        o_rm = ttnn.transformer.sparse_sdpa(
            self._pad_heads_rm(q_rm),
            kv_rm,
            indices,
            v_dim=self.L,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=self.scale,
            k_chunk_size=128,
        )  # [1,H,T-split,L] row-major
        ttnn.deallocate(q_rm)
        ttnn.deallocate(indices)
        ttnn.deallocate(kv_rm)
        o_lat = self._unpad_heads(ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT))
        ttnn.deallocate(o_rm)
        o_sparse = self._o_lat_to_heads(o_lat)  # [1,1,T-split,H*Dv]
        ttnn.deallocate(o_lat)
        if o_dense is None:  # the whole chunk lies in the sparse region
            return self._o_proj(o_sparse)
        o = ttnn.concat([o_dense, o_sparse], dim=-2)  # [1,1,T,H*Dv]
        ttnn.deallocate(o_dense)
        ttnn.deallocate(o_sparse)
        return self._o_proj(o)

    # ---------------------------------------------------------------------- decode
    def decode_forward(
        self,
        x: ttnn.Tensor,
        cache: TtLatentCache,
        positions: ttnn.Tensor,
        pool_ids: ttnn.Tensor,
        ring_slots: ttnn.Tensor,
        host_positions: list[int] | None = None,
        dense_positions: ttnn.Tensor | None = None,
        sparse_meta: SparseDecodeMeta | None = None,
    ) -> ttnn.Tensor:
        """x [1,1,B,4096] (one token per user at position a = positions[b]); ``pool_ids`` = a // 4 and
        ``ring_slots`` = a % 4 ([B] int32 device tensors, host-derived like ``positions``). Appends the
        latent, maintains the pooled-key index, then attends over [0, a] with the dense flash-MLA decode
        for every slot (HF-exact through position 2050, trace-safe). With ``host_positions`` (the same
        values), users at a >= 2051 get their row replaced by the per-user indexer + sparse_sdpa result
        (host chunk geometry, not traced). Without ``host_positions`` a user beyond 2050 is attended
        densely, which is NOT HF semantics: FunctionalDecoder enforces the routing.
        ``dense_positions`` (optional [B] int32, host-derived like ``pool_ids``): the positions the dense
        flash-MLA decode attends up to instead of ``positions``. The dense row of a sparse user is
        discarded, so ``min(a, dense_exact_max_pos)`` per user bounds that op's cost to 2051 keys at
        any context length (it reads the whole prefix otherwise: 1 GiB per layer at 1 M positions) and
        is the identity for every user still on the dense path.
        ``sparse_meta`` (a ``SparseDecodeMeta`` of this step's positions) selects the traced multi-user
        path instead: every user, dense or sparse, is attended over a compact cache gathered from its
        selected pools (``_decode_unified``), with no host geometry and no whole-prefix copy; the dense
        flash-MLA over the real cache and the per-user eager rows are then not run at all."""
        B = x.shape[-2]
        if self.stage_timer is not None:
            self.stage_timer.start()
        if self.fused_decode == "fused":
            return self._fused_decode(x, cache, positions, host_positions, dense_positions)
        stem = self.lin_stem(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,2336]
        parts = {k: ttnn.slice(stem, [0, 0, 0, a], [1, 1, B, b]) for k, (a, b) in self.stem_bounds.items()}
        ttnn.deallocate(stem)
        q_resid = ttnn.rms_norm(
            parts["qa"], epsilon=self.cfg.rms_norm_eps, weight=self.qa_norm, compute_kernel_config=self.hifi4
        )
        ttnn.deallocate(parts["qa"])
        q = self._q(q_resid)  # [1,1,B,H*Dq]
        c = ttnn.rms_norm(
            parts["kva"], epsilon=self.cfg.rms_norm_eps, weight=self.kva_norm, compute_kernel_config=self.hifi4
        )
        ttnn.deallocate(parts["kva"])
        part = cache.partition  # user-partitioned cache: this chip appends / indexes / attends its own users only
        if part:
            if sparse_meta is None:
                raise NotImplementedError("the user-partitioned cache decodes through the traced sparse path only")
            loc = lambda t: self.mesh.partition_cols(t, 2)  # [1,1,B,*] replicated -> [1,1,B/cols,*] (my users)
            c_loc, k_loc, g_loc = loc(c), loc(parts["k"]), loc(parts["g"])
            for t in (c, parts["k"], parts["g"]):
                ttnn.deallocate(t)
            B_loc = c_loc.shape[-2]
            cache.update(ttnn.reshape(c_loc, [1, B_loc, 1, self.L]), positions)
            ttnn.deallocate(c_loc)
            self.indexer.maintain_decode(x, cache.index, pool_ids, ring_slots, kg=(k_loc, g_loc))
            w_idx = loc(parts["w"])
            ttnn.deallocate(parts["w"])
            q_resid_loc = loc(q_resid)
        else:
            cache.update(ttnn.reshape(c, [1, B, 1, self.L]), positions)
            ttnn.deallocate(c)
            self.indexer.maintain_decode(x, cache.index, pool_ids, ring_slots, kg=(parts["k"], parts["g"]))
            w_idx = parts["w"]  # [1,1,B,Hi] indexer head weights (sparse rows only)
            q_resid_loc = q_resid
        self._mark("stems+append+index_maintain")
        q_lat = self._absorb_q(q)  # [1,H,B,L]
        ttnn.deallocate(q)
        if sparse_meta is not None:
            if part:
                # heads -> users: every column holds its 16 heads of all B users; the owner needs all 64 heads of
                # its B/cols users (the inverse exchange after attention)
                q_t = ttnn.transpose(q_lat, 1, 2)  # [1,B,H_loc,L]
                q_x = self.mesh.all_to_all_cols(q_t, in_dim=1, out_dim=2)  # [1,B/cols,H,L]
                ttnn.deallocate(q_t)
                q_full = ttnn.transpose(q_x, 1, 2)  # [1,H,B/cols,L]
                ttnn.deallocate(q_x)
                self._mark("a2a_heads_to_users")
                oh_full = self._decode_unified(q_resid_loc, q_full, cache, sparse_meta, w_idx)
                ttnn.deallocate(q_full)
                ttnn.deallocate(q_resid_loc)
                o_t = ttnn.transpose(oh_full, 1, 2)  # [1,B/cols,H,L]
                ttnn.deallocate(oh_full)
                o_x = self.mesh.all_to_all_cols(o_t, in_dim=2, out_dim=1)  # [1,B,H_loc,L]
                ttnn.deallocate(o_t)
                oh = ttnn.transpose(o_x, 1, 2)  # [1,H_loc,B,L]
                ttnn.deallocate(o_x)
                self._mark("a2a_users_to_heads")
            else:
                oh = self._decode_unified(q_resid, q_lat, cache, sparse_meta, w_idx)
            ttnn.deallocate(q_resid)
            ttnn.deallocate(q_lat)
            ttnn.deallocate(w_idx)
            y = self._o_proj(self._o_lat_to_heads(oh))
            self._mark("out_proj")
            return y
        q_dec = ttnn.transpose(q_lat, 1, 2)  # [1,B,H,L] (decode layout: batch, padded heads, dim)
        self._mark("absorb_q")
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_dec,
            cache.cache,
            head_dim_v=self.L,
            page_table_tensor=cache.page_table,
            cur_pos_tensor=positions if dense_positions is None else dense_positions,
            scale=self.scale,
            program_config=self._decode_sdpa_config(B),
            compute_kernel_config=self.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,B,H,L]
        ttnn.deallocate(q_dec)
        oh = ttnn.transpose(o_lat, 1, 2)  # [1,H,B,L]
        ttnn.deallocate(o_lat)
        self._mark("dense_mla_decode")
        sparse_users = (
            [u for u, a in enumerate(host_positions) if a > self.dense_exact_max_pos] if host_positions else []
        )
        if sparse_users:
            sparse_rows = self._decode_sparse_rows(x, q_resid, q_lat, cache, host_positions, sparse_users, w=w_idx)
            rows = [
                sparse_rows[u] if u in sparse_rows else ttnn.slice(oh, [0, 0, u, 0], [1, self.H, u + 1, self.L])
                for u in range(B)
            ]
            oh_new = rows[0] if B == 1 else ttnn.concat(rows, dim=-2)
            ttnn.deallocate(oh)
            oh = oh_new
        ttnn.deallocate(q_resid)
        ttnn.deallocate(q_lat)
        ttnn.deallocate(w_idx)
        y = self._o_proj(self._o_lat_to_heads(oh))
        self._mark("out_proj")
        return y

    def _fused_decode(self, x, cache: TtLatentCache, positions, host_positions, dense_positions):
        """The 7-op dense decode block: stem matmul -> ``dsa_decode_state`` (q_a norm, latent append, index ring +
        pool maintenance, all in place) -> folded q_b.Wkb1 matmul -> transpose to the decode head layout (the
        packed rows are a free view of the head-major tiles) -> flash-MLA decode -> transpose -> folded
        Wkb2.o_proj matmul (+ the TP all-reduce). Sparse users (positions > 2050) get
        their head block replaced by the per-user indexer + sparse_sdpa result as in the composite path."""
        B, H, L = x.shape[-2], self.H, self.L
        idx = cache.index
        b = self.stem_bounds
        stem = self.lin_stem(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,Ws]
        q_resid = ttnn.transformer.dsa_decode_state(
            stem,
            cache.cache,
            cache.page_table,
            positions,
            idx.ring_k,
            idx.ring_g,
            idx.pool_cache,
            self.qa_norm_w32,
            self.kva_norm_w32,
            self.knorm_w32,
            self.knorm_b32,
            self.ape_block,
            self.selectors,
            b["qa"][0],
            b["kva"][0],
            b["k"][0],
            b["g"][0],
            self.cfg.q_lora_rank,
            L,
            idx.D,
            idx.P,
            idx.pool_blocks,
            eps_rms=self.cfg.rms_norm_eps,
            eps_ln=1e-6,
        )  # [1,1,B,Lq]
        self._mark("stem+state")
        if DSA_FOLD in ("both", "q"):
            q_lat = self.lin_qb1(q_resid, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,H*L]
            # B <= 32: the packed tiled rows ARE the head-major [1,H,B,L] tile sequence (a free view); the decode
            # layout is then one multicore transpose. (A logical reshape would shuffle rows at ~5 GB/s.)
            q_lat_h = ttnn.experimental.view(q_lat, [1, H, B, L]) if B <= TILE else self._heads_major(q_lat, B)
        else:
            q_lat_h = self._absorb_q(self._q(q_resid))  # [1,H,B,L] (composite absorb)
            q_lat = None
        q_dec = ttnn.transpose(q_lat_h, 1, 2)  # [1,B,H,L]
        self._mark("absorb_q")
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_dec,
            cache.cache,
            head_dim_v=L,
            page_table_tensor=cache.page_table,
            cur_pos_tensor=positions if dense_positions is None else dense_positions,
            scale=self.scale,
            program_config=self._decode_sdpa_config(B),
            compute_kernel_config=self.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,B,H,L]
        self._mark("dense_mla_decode")
        sparse_users = (
            [u for u, a in enumerate(host_positions) if a > self.dense_exact_max_pos] if host_positions else []
        )
        if sparse_users:
            w_idx = ttnn.slice(stem, [0, 0, 0, b["w"][0]], [1, 1, B, b["w"][1]])
            sparse_rows = self._decode_sparse_rows(x, q_resid, q_lat_h, cache, host_positions, sparse_users, w=w_idx)
            blocks = [
                ttnn.transpose(sparse_rows[u], 1, 2)
                if u in sparse_rows
                else ttnn.slice(o_lat, [0, u, 0, 0], [1, u + 1, H, L])
                for u in range(B)
            ]  # [1,1,H,L] per user
            o_new = blocks[0] if B == 1 else ttnn.concat(blocks, dim=1)
            ttnn.deallocate(o_lat)
            o_lat = o_new
            ttnn.deallocate(w_idx)
        ttnn.deallocate(q_dec)
        ttnn.deallocate(q_lat if q_lat is not None else q_lat_h)
        ttnn.deallocate(q_resid)
        ttnn.deallocate(stem)
        oh = ttnn.transpose(o_lat, 1, 2)  # [1,H,B,L]
        ttnn.deallocate(o_lat)
        if DSA_FOLD in ("both", "o"):
            # the head-major tiles read as the packed [1,1,B,H*L] rows (free view, B <= 32)
            o_p = ttnn.experimental.view(oh, [1, 1, B, H * L]) if B <= TILE else self._heads_packed(oh, B)
            y = self.lin_o2(o_p, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,hidden] (TP partial)
            ttnn.deallocate(oh)
            y = self.mesh.all_reduce_cols(y)
        else:
            y = self._o_proj(self._o_lat_to_heads(oh))  # composite absorb_o + o_proj
        self._mark("out_proj")
        return y

    def _heads_major(self, q_lat: ttnn.Tensor, B: int) -> ttnn.Tensor:
        """[1,1,B,H*L] -> [1,H,B,L] for B > 32 (several tile rows: no free view), through the row-major layout."""
        rm = ttnn.to_layout(q_lat, ttnn.ROW_MAJOR_LAYOUT)
        t = ttnn.to_layout(ttnn.reshape(rm, [1, B, self.H, self.L]), ttnn.TILE_LAYOUT)
        ttnn.deallocate(rm)
        out = ttnn.transpose(t, 1, 2)
        ttnn.deallocate(t)
        return out

    def _heads_packed(self, oh: ttnn.Tensor, B: int) -> ttnn.Tensor:
        """[1,H,B,L] -> [1,1,B,H*L] for B > 32, through the row-major layout."""
        t = ttnn.transpose(oh, 1, 2)  # [1,B,H,L]
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(t)
        out = ttnn.to_layout(ttnn.reshape(rm, [1, 1, B, self.H * self.L]), ttnn.TILE_LAYOUT)
        ttnn.deallocate(rm)
        return out

    # ------------------------------------------------ traced multi-user sparse decode
    def _compact_page_table(self, B: int, out_blocks: int) -> ttnn.Tensor:
        """Identity page table [B, out_blocks] of the compact gathered cache (a constant per batch size)."""
        key = ("compact_pt", B, out_blocks)
        if key not in self._pad_cache:
            pt = torch.arange(B * out_blocks, dtype=torch.int32).reshape(B, out_blocks)
            self._pad_cache[key] = ttnn.from_torch(
                pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
        return self._pad_cache[key]

    def _zero_rows(self, shape: tuple) -> ttnn.Tensor:
        key = ("zeros",) + tuple(shape)
        if key not in self._pad_cache:
            self._pad_cache[key] = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return self._pad_cache[key]

    def _decode_unified(self, q_resid, q_lat, cache: TtLatentCache, meta: SparseDecodeMeta, w) -> ttnn.Tensor:
        """All users of the chip in one trace-safe graph: per user the indexer scores its pooled keys
        (``score_decode_row``) and the top-512 pools are selected under the user's own causal bound
        (``select_decode_pools`` with ``meta.valid_len[u]``); then, batched, the pool lists are completed
        with the tail pool and the dense users' in-order lists (``meta``), the selected pools are gathered
        out of the paged latent cache into a compact cache (``paged_pool_gather``: 2052 rows per user, one
        KB per selected token) and ONE ``paged_flash_multi_latent_attention_decode`` attends every user
        over its compact rows ``[0, cur_pos]``. Returns [1,H,B,L] bf16 (the ``oh`` layout)."""
        idx = self.indexer
        B = q_lat.shape[-2]
        Hi, D, P, K = idx.Hi, idx.D, idx.P, self.cfg.index_select_pools
        q_i, w_i = idx.queries_and_weights(q_resid, q_resid, w=w)  # [1,Hi,B,D], [1,1,B,Hi]
        # GLM53_SCORER_BATCHED=1 (default): ONE index_score_decode program scores every user's query (its 32 heads
        # as the rows of one query tile row) against its pooled keys -> [B,1,1,W] score rows, no per-user calls,
        # no 32-row score blocks (needs pools_per_user % 256 == 0: TtIndexCache rounds to 256); 0 = one shared
        # indexer_score_dsa call per user
        scorer_batched = os.environ.get("GLM53_SCORER_BATCHED", "1") == "1"
        batched = meta.valid_len_rows is not None and os.environ.get("GLM53_TOPK_BATCHED", "1") == "1"
        if scorer_batched:
            assert batched, "GLM53_SCORER_BATCHED needs the batched top-k (GLM53_TOPK_BATCHED=1)"
            q4 = ttnn.transpose(q_i, 1, 2)  # [1,B,Hi,D]
            ttnn.deallocate(q_i)
            W = cache.index.pools_per_user
            # GLM53_TOPK_SEGMENTS=1: the scorer masks the columns beyond each user's bound (-inf) and writes S segment
            # rows per user; the top-k runs in two stages (S cores per user, then the S*512 candidates) instead of one
            # serial row per core (1.26 ms at 262 K pools)
            S = self.topk_rows.segments_for(W) if os.environ.get("GLM53_TOPK_SEGMENTS", "0") == "1" else 1
            if B not in self._top_buf:
                self._top_buf[B] = ttnn.zeros(
                    (B, 1, 1, K), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                )
            if S > 1:
                key = (B, S)
                if key not in self._score_buf:
                    self._score_buf[key] = (
                        ttnn.zeros(
                            (B * S, 1, 1, W // S), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                        ),
                        ttnn.zeros(
                            (B, 1, 1, S * K), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                        ),
                        ttnn.zeros(
                            (B, 1, 1, S * K), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                        ),
                    )
                seg_buf, ids1, vals1 = self._score_buf[key]
                self.index_scorer(
                    q4, w_i, cache.index.pools_batched(), seg_buf, valid_rows=meta.valid_len_rows, segments=S
                )
                ttnn.deallocate(q4)
                ttnn.deallocate(w_i)
                self._mark("index_score")
                top = self.topk_rows.segmented(seg_buf, S, ids1, vals1, self._top_buf[B])  # [B,1,1,K] uint32
                self._mark("topk")
                return self._gather_and_attend(top, q_lat, cache, meta, B, P, out_persistent=True)
            if B not in self._score_buf:
                self._score_buf[B] = ttnn.zeros(
                    (B, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                )
            rows = self.index_scorer(q4, w_i, cache.index.pools_batched(), self._score_buf[B])  # [B,1,1,W] bf16 RM
            ttnn.deallocate(q4)
            ttnn.deallocate(w_i)
            self._mark("index_score")
            top = self.topk_rows(rows, meta.valid_len_rows, self._top_buf[B])  # [B,1,1,K] uint32, per-row bounds
            self._mark("topk")
            return self._gather_and_attend(top, q_lat, cache, meta, B, P, out_persistent=True)
        # per-user 32-row query blocks with the user's query in row 0: the B rows followed by 31 zero
        # rows, sliced at [u, u + 32) (rows past the user's are other users' queries or zeros: scored and
        # dropped by the row-0 slice inside score_decode_row)
        q_big = ttnn.concat([q_i, self._zero_rows((1, Hi, TILE - 1, D))], dim=2)  # [1,Hi,B+31,D]
        w_big = ttnn.concat([w_i, self._zero_rows((1, 1, TILE - 1, Hi))], dim=2)  # [1,1,B+31,Hi]
        ttnn.deallocate(q_i)
        ttnn.deallocate(w_i)
        self._mark("index_queries")
        # GLM53_TOPK_BATCHED=1 (default): the top-k runs ONCE over all users' score rows with per-row bounds
        # (``valid_len_rows``) through the model-local ``topk_rows`` op: one row costs the same as B rows in one call
        # (1.14 ms at 262 K pools); 0 = one shared topk_large_indices call per user (its single bound)
        rows_or_tops = []
        for u in range(B):
            qb = ttnn.slice(q_big, [0, 0, u, 0], [1, Hi, u + TILE, D])
            wb = ttnn.slice(w_big, [0, 0, u, 0], [1, 1, u + TILE, Hi])
            row = idx.score_decode_row(qb, wb, cache.index, u)  # [1,1,1,W]
            ttnn.deallocate(qb)
            ttnn.deallocate(wb)
            self._mark("index_score")
            if batched:
                rows_or_tops.append(row)
            else:
                rows_or_tops.append(idx.select_decode_pools(row, meta.valid_len[u]))  # [1,1,1,K] uint32
                ttnn.deallocate(row)
                self._mark("topk")
        ttnn.deallocate(q_big)
        ttnn.deallocate(w_big)
        if batched:
            rows = rows_or_tops[0] if B == 1 else ttnn.concat(rows_or_tops, dim=0)  # [B,1,1,W] bf16 row-major
            if B > 1:
                for t in rows_or_tops:
                    ttnn.deallocate(t)
            if B not in self._top_buf:  # persistent output of the generic op
                self._top_buf[B] = ttnn.zeros(
                    (B, 1, 1, K), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                )
            top = self.topk_rows(rows, meta.valid_len_rows, self._top_buf[B])  # [B,1,1,K] uint32, per-row bounds
            ttnn.deallocate(rows)
            self._mark("topk")
        else:
            top = rows_or_tops[0] if B == 1 else ttnn.concat(rows_or_tops, dim=0)  # [B,1,1,K]
            if B > 1:
                for t in rows_or_tops:
                    ttnn.deallocate(t)
        return self._gather_and_attend(top, q_lat, cache, meta, B, P, out_persistent=batched)

    def _gather_and_attend(
        self, top, q_lat, cache: TtLatentCache, meta: SparseDecodeMeta, B: int, P: int, *, out_persistent: bool
    ):
        """The batched tail of ``_decode_unified``: complete the ranked pool lists with the tail pool and the dense
        users' in-order lists, gather the pools into the compact cache, attend. ``top`` [B,1,1,K] uint32
        (``out_persistent``: a persistent generic-op output, not deallocated)."""
        # the slot -> pool mapping (ranked pools, dense users' pools in order, the tail pool) is done by the gather
        if (
            B not in self._compact
        ):  # persistent output of the gather (the generic op's io tensors keep their addresses under trace)
            self._compact[B] = ttnn.zeros(
                self.pool_gather.output_shape(top, self.L),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        compact = self.pool_gather(cache.cache, cache.page_table, top, meta.tail_pool, self._compact[B])
        if not out_persistent:
            ttnn.deallocate(top)
        out_blocks = compact.shape[0] // B
        self._mark("pool_gather")
        q_dec = ttnn.transpose(q_lat, 1, 2)  # [1,B,H,L]
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_dec,
            compact,
            head_dim_v=self.L,
            page_table_tensor=self._compact_page_table(B, out_blocks),
            cur_pos_tensor=meta.cur_pos,
            scale=self.scale,
            program_config=self._decode_sdpa_config(B),
            compute_kernel_config=self.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,B,H,L]
        ttnn.deallocate(q_dec)
        oh = ttnn.transpose(o_lat, 1, 2)  # [1,H,B,L]
        ttnn.deallocate(o_lat)
        self._mark("compact_mla_decode")
        return oh

    def _decode_sparse_rows(
        self, x, q_resid, q_lat, cache: TtLatentCache, positions: list[int], users: list[int], w=None
    ):
        """Per-user indexer selection + sparse_sdpa rows {u: [1,H,1,L]} for the listed users (positions
        > dense_exact_max_pos, so every index row is a full valid prefix). ``w``: the indexer head
        weights [1,1,B,Hi] when the caller already has them (the fused decode stem)."""
        idx = self.indexer
        D = idx.D
        q_i, w_i = idx.queries_and_weights(x, q_resid, w=w)  # [1,Hi,B,D], [1,1,B,Hi]
        out = {}
        for u in users:
            a = positions[u]
            # indexer_score_dsa wants a tile-aligned query block and chunk start: score the 32 positions
            # [a - a%32, a - a%32 + 32) with this user's query placed at row r = a % 32 (zero rows elsewhere,
            # scored but discarded); the selection keeps row r only (one row through the expansion + top-k)
            start, r = a - a % TILE, a % TILE
            qu = self._place_row(ttnn.slice(q_i, [0, 0, u, 0], [1, idx.Hi, u + 1, D]), r)
            wu = self._place_row(ttnn.slice(w_i, [0, 0, u, 0], [1, 1, u + 1, idx.Hi]), r)
            self._mark("index_queries")
            indices = idx.select(qu, wu, cache.index, u, query_start=start, num_pools_valid=(a + 1) // idx.P, row=r)
            q_rm = ttnn.to_layout(ttnn.slice(q_lat, [0, 0, u, 0], [1, self.H, u + 1, self.L]), ttnn.ROW_MAJOR_LAYOUT)
            self._mark("q_rm")
            kv_tile = cache.user_rows(u, a + 1)  # the user's whole prefix: a fresh TILE copy ...
            self._mark("kv_slice")
            kv_rm = ttnn.to_layout(kv_tile, ttnn.ROW_MAJOR_LAYOUT)  # ... untilized for the sparse_sdpa kv contract
            ttnn.deallocate(kv_tile)
            self._mark("kv_untilize")
            o_rm = ttnn.transformer.sparse_sdpa(
                self._pad_heads_rm(q_rm),
                kv_rm,
                indices,
                v_dim=self.L,
                kv_format=ttnn.transformer.SparseKVFormat.BF16,
                scale=self.scale,
                k_chunk_size=128,
            )  # [1,H,1,L]
            self._mark("sparse_sdpa")
            ttnn.deallocate(kv_rm)
            ttnn.deallocate(q_rm)
            ttnn.deallocate(indices)
            out[u] = self._unpad_heads(ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT))
            self._mark("sparse_out")
        ttnn.deallocate(q_i)
        ttnn.deallocate(w_i)
        return out

    def selected_pools(self, x: ttnn.Tensor, cache: TtLatentCache, user: int, position: int) -> list[int]:
        """Test hook: the pool ids the TRACED path selects (``score_decode_row`` + ``select_decode_pools``
        under the user's own bound) for one query ``x`` [1,1,1,4096] of ``user`` at ``position`` on the
        current index cache (read only); descending score, ``index_select_pools`` entries."""
        idx = self.indexer
        Hi, D, P, K = idx.Hi, idx.D, idx.P, self.cfg.index_select_pools
        q_resid = self._q_resid(x)
        q_i, w_i = idx.queries_and_weights(x, q_resid)
        qb = ttnn.concat([q_i, self._zero_rows((1, Hi, TILE - 1, D))], dim=2)
        wb = ttnn.concat([w_i, self._zero_rows((1, 1, TILE - 1, Hi))], dim=2)
        row = idx.score_decode_row(qb, wb, cache.index, user)
        valid = ttnn.from_torch(
            torch.tensor([max((position + 1) // P, K)], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        top = idx.select_decode_pools(row, valid)
        pools = [int(v) for v in ttnn.to_torch(top).flatten().tolist()]
        for t in (q_resid, q_i, w_i, qb, wb, row, valid, top):
            ttnn.deallocate(t)
        return pools

    def selected_ids(self, x: ttnn.Tensor, cache: TtLatentCache, user: int, position: int) -> list[int]:
        """Test hook: the raw index row (token ids, 0xFFFFFFFF sentinels) the indexer selects for one
        query ``x`` [1,1,1,4096] (the normalized attention input) of ``user`` at ``position`` on the
        CURRENT index cache (read only)."""
        idx = self.indexer
        q_resid = self._q_resid(x)
        q_i, w_i = idx.queries_and_weights(x, q_resid)
        start, r = position - position % TILE, position % TILE
        qu = self._place_row(q_i, r)
        wu = self._place_row(w_i, r)
        ids_t = idx.select(qu, wu, cache.index, user, query_start=start, num_pools_valid=(position + 1) // idx.P, row=r)
        ids = ttnn.to_torch(ids_t).flatten().tolist()
        for t in (q_resid, q_i, w_i, ids_t):
            ttnn.deallocate(t)
        return [int(i) for i in ids]

    # ------------------------------------------------------------- sparse_sdpa head padding
    def _pad_heads_rm(self, q_rm: ttnn.Tensor) -> ttnn.Tensor:
        """sparse_sdpa wants H % 32 == 0: pad a row-major [1, H, T, L] query with zero heads (their
        outputs are discarded by ``_unpad_heads``); a no-op when the local head count is tile-aligned."""
        H = q_rm.shape[1]
        if H % TILE == 0:
            return q_rm
        pad = TILE - H % TILE
        key = ("qpad", pad, q_rm.shape[-2], q_rm.shape[-1])
        if key not in self._pad_cache:
            self._pad_cache[key] = ttnn.zeros(
                (1, pad, q_rm.shape[-2], q_rm.shape[-1]),
                dtype=q_rm.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        return ttnn.concat([q_rm, self._pad_cache[key]], dim=1)

    def _unpad_heads(self, o: ttnn.Tensor) -> ttnn.Tensor:
        if o.shape[1] == self.H:
            return o
        return ttnn.slice(o, [0, 0, 0, 0], [1, self.H, o.shape[-2], o.shape[-1]])

    def _place_row(self, row: ttnn.Tensor, r: int) -> ttnn.Tensor:
        """[.., 1, W] -> [.., 32, W] with ``row`` at row r and zeros elsewhere: a broadcast multiply by the
        one-hot column ``e_r`` [1,1,32,1] (cached per r), so every r runs the same program (a zeros +
        concat build changes geometry with r and recompiled at every new ``position % 32``)."""
        key = ("row_mask", r)
        if key not in self._pad_cache:
            m = torch.zeros(1, 1, TILE, 1)
            m[0, 0, r, 0] = 1.0
            self._pad_cache[key] = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return ttnn.multiply(row, self._pad_cache[key])
