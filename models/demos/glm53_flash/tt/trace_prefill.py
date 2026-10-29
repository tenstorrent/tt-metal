# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced fixed-length prefill for the resident mesh model (``GLM53MeshModel.capture_prefill`` /
``prefill_traced``; design in doc/mesh/trace_prefill.md).

One trace per supported chunk length ``T`` (``supported_prefill_lengths``: {128, 256, 512, 1024, 2048}
rounded up to a multiple of ``32 * rows``, within the dense range of the DSA layers). A prompt of ``S``
tokens is padded to the smallest supported ``T >= S`` and replayed through that trace; the replay
gives the eager ``prefill``'s last-token logits (the padding rows never feed a valid row: causal
attention, per-row MLPs / mHC, ttKDA clips its chronology at ``actual_end``).

What the eager prefill takes from the host, and the persistent device input that replaces it (all are
refreshed with ``ttnn.copy_host_to_device_tensor`` before a replay, nothing is allocated after the
captures):

* the token ids: ``PrefillTraceInputs.ids`` ``[1,1,1,T]`` uint32, row-sharded like the decode ids;
* the prompt length: ``end`` ``[1,1,1,1]`` uint32 replicated = ttKDA's ``actual_end`` (the KDA carries
  stop at the valid end; ``actual_start`` is the cached zero scalar);
* the user slot: ``PrefillSlot`` (one per model, shared by every layer and length): the slot index as a
  ``[1]`` int32 tensor for the paged fills (latent cache, pooled-key cache, rings; and the KDA recurrent
  state written through ``paged_fill_cache`` on the ``[B*H*K/32, 1, 32, V]`` view of the batched state)
  plus one-hot user masks for the KDA conv history (``dst * (1 - m) + new * m`` is exact in bf16);
* the pooled-key ring tail: ``ring_sel`` ``[1,1,32,T]`` bf16, rows ``j < 4`` one-hot at the last prompt
  position ``p`` with ``p % 4 == j`` (selected on device with a HiFi4 matmul: exact) and written as one
  ring block; every pool of the padded chunk is written, the pools past the prompt are rewritten by
  decode before they become selectable;
* the last token: the head's collapse + final norm run on every local row, then the 32-row tile holding
  the owner row's local last row is cut with the tensor-argument ``ttnn.slice`` (``tile_start`` /
  ``tile_end`` ``[4]`` int32 replicated: the slice program keys on the length, the offset is a runtime
  tensor; the node tt_transformers uses after its bucketed prefill traces,
  ``Transformer.process_logits_after_prefill_trace``), the LM head + greedy run on that tile and the
  prompt's id is row ``(S-1) % 32`` of the owner row's chip (``PrefillTraceResult.row``).

Outputs: the logits ``[1,1,32,V]`` and ids ``[1,1,32,1]`` are copied inside the trace into buffers
allocated before any capture (``out_logits`` / ``out_ids``, the qwen38 generator's pattern), so they
survive the replays of other traces; the last layer's ``streams`` stay the capture's transient output
(read them before any other replay).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.glm53_flash.tt.indexer import ring_selector
from models.demos.glm53_flash.tt.mesh import GLMMesh
from models.demos.glm53_flash.tt.moe import pad_row_mask_host, wave_row_mask_host

TILE = 32
DENSE_ROWS = 2048  # TtDSA.sparse_row_start: rows at positions < 2048 attend densely (the traced range, Phase A)
RING_ROWS = 4  # index_kpool: the raw ring holds the current pool's rows (position % 4)
DEFAULT_LENGTHS = (128, 256, 512, 1024, 2048)


def supported_prefill_lengths(rows: int, max_rows: int, lengths=DEFAULT_LENGTHS) -> list[int]:
    """The traced chunk lengths for a ``rows``-row mesh whose caches hold ``max_rows`` positions: every
    requested length rounded up to a multiple of ``32 * rows`` (the row slabs stay tile-aligned), kept
    when it lies in the dense range, fits the caches and holds whole pool tiles (``T % 128 == 0``)."""
    unit = TILE * rows
    out = set()
    for L in lengths:
        T = -(-int(L) // unit) * unit
        if T <= DENSE_ROWS and T <= max_rows and T % (TILE * RING_ROWS) == 0:
            out.add(T)
    return sorted(out)


def pick_prefill_length(S: int, lengths) -> int | None:
    """The smallest supported length >= S (None when none fits: the caller prefills eagerly)."""
    fits = [T for T in lengths if T >= S]
    return min(fits) if fits else None


def _mapper(mesh: GLMMesh, rows_dim: int | None):
    if mesh.is_single:
        return None
    if rows_dim is not None and mesh.rows > 1:
        return mesh.shard_rows(rows_dim)
    return mesh.replicate()


def _upload(mesh: GLMMesh, t: torch.Tensor, dtype, layout, *, rows_dim: int | None = None, mapper=None) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, rows_dim) if mapper is None else mapper,
    )


def _refresh(mesh: GLMMesh, t: torch.Tensor, dst: ttnn.Tensor, *, rows_dim: int | None = None, mapper=None):
    """Host values -> the persistent device tensor ``dst`` (same shape / dtype / layout / placement)."""
    host = ttnn.from_torch(
        t.contiguous(),
        dtype=dst.dtype,
        layout=dst.layout,
        mesh_mapper=_mapper(mesh, rows_dim) if mapper is None else mapper,
    )
    ttnn.copy_host_to_device_tensor(host, dst)


# ----------------------------------------------------------------------------------------- user slot
@dataclass
class PrefillSlot:
    """Device-side user slot of the traced prefill (one per model). ``set_user`` refreshes every tensor
    for the prompt's slot before a replay; the tensors are allocated once, before any trace capture."""

    mesh: GLMMesh
    max_batch: int
    partition: bool  # user-partitioned DSA caches (per-column owner / scratch slots)
    users_per_chip: int
    cache_idx: ttnn.Tensor  # [1] int32 RM: the latent / pooled-key / ring slot the paged fills write (per column when partitioned)
    scratch_idx: ttnn.Tensor | None  # scratch-layout partitioned caches: the scratch slot every column fills as well
    user_idx: ttnn.Tensor  # [1] int32 RM replicated: the KDA slot (recurrent-state paged fill)
    conv_mask: ttnn.Tensor  # [taps, B, 1] bf16 TILE: 1.0 on row ``user``
    conv_keep: ttnn.Tensor  # [taps, B, 1] bf16 TILE: 1.0 - conv_mask
    rec_page_table: ttnn.Tensor | None  # [B, H*K/32] int32: identity page table of the recurrent-state view
    kda_zero: KdaState | None  # read-only zero initial state of one user (ttKDA only reads its input state)
    user: int = -1

    @classmethod
    def allocate(cls, mesh: GLMMesh, decoders) -> "PrefillSlot":
        """``decoders``: the model's ``FunctionalDecoder`` layers (any mix of KDA / DSA layers)."""
        decoders = list(decoders)
        max_batch = decoders[0].max_batch
        kda = [d for d in decoders if d.is_kda]
        dsa = [d for d in decoders if not d.is_kda]
        partition = bool(dsa and dsa[0].partition)
        users_per_chip = max_batch // mesh.cols if partition else max_batch
        # the scratch slot every column fills as well (scratch layout only; the heads2users layout has none: the
        # non-owner columns' ``cache_idx`` names the dummy row ``users_per_chip`` of the page tables instead)
        has_scratch = partition and dsa[0].cache.scratch is not None
        rm, i32 = ttnn.ROW_MAJOR_LAYOUT, ttnn.int32
        # one slot value per column when partitioned (the column-sharded upload needs ``cols`` chunks): user 0's
        cache_idx = _upload(
            mesh,
            cls._cache_slots_host(partition, users_per_chip, mesh.cols, 0),
            i32,
            rm,
            mapper=cls._cache_mapper(mesh, partition),
        )
        scratch_idx = _upload(mesh, torch.tensor([users_per_chip], dtype=torch.int32), i32, rm) if has_scratch else None
        user_idx = _upload(mesh, torch.zeros(1, dtype=torch.int32), i32, rm)
        rec_pt, kda_zero, taps = None, None, 3
        if kda:
            attn = kda[0]._require_weights().attn
            taps = attn.kcfg.conv_kernel_size - 1
            nblk = attn.H * attn.K // TILE
            rec_pt = _upload(mesh, torch.arange(max_batch * nblk, dtype=torch.int32).reshape(max_batch, nblk), i32, rm)
            kda_zero = attn.allocate_state(1)
        mask, keep = cls._conv_masks(taps, max_batch, 0)
        return cls(
            mesh=mesh,
            max_batch=max_batch,
            partition=partition,
            users_per_chip=users_per_chip,
            cache_idx=cache_idx,
            scratch_idx=scratch_idx,
            user_idx=user_idx,
            conv_mask=_upload(mesh, mask, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            conv_keep=_upload(mesh, keep, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            rec_page_table=rec_pt,
            kda_zero=kda_zero,
            user=0,
        )

    @staticmethod
    def _cache_mapper(mesh: GLMMesh, partition: bool):
        if mesh.is_single:
            return None
        return mesh.shard_cols(0) if partition and mesh.cols > 1 else mesh.replicate()

    @staticmethod
    def _conv_masks(taps: int, B: int, user: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros(taps, B, 1)
        mask[:, user, 0] = 1.0
        return mask, 1.0 - mask

    @staticmethod
    def _cache_slots_host(partition: bool, users_per_chip: int, cols: int, user: int) -> torch.Tensor:
        """The slot every column chip's fills write: the user's slot (non-partitioned, [1]), or ([cols]) its slot on
        the owner column and the scratch slot / dummy row (``users_per_chip`` either way) elsewhere
        (``TtLatentCache._prefill_views_for``)."""
        if not partition:
            return torch.tensor([user], dtype=torch.int32)
        owner, slot = divmod(user, users_per_chip)
        return torch.tensor([slot if c == owner else users_per_chip for c in range(cols)], dtype=torch.int32)

    def _cache_slots(self, user: int) -> torch.Tensor:
        return self._cache_slots_host(self.partition, self.users_per_chip, self.mesh.cols, user)

    def set_user(self, user: int):
        if not 0 <= user < self.max_batch:
            raise ValueError(f"user {user} >= max_batch {self.max_batch}")
        _refresh(
            self.mesh, self._cache_slots(user), self.cache_idx, mapper=self._cache_mapper(self.mesh, self.partition)
        )
        _refresh(self.mesh, torch.tensor([user], dtype=torch.int32), self.user_idx)
        mask, keep = self._conv_masks(int(self.conv_mask.shape[0]), self.max_batch, user)
        _refresh(self.mesh, mask, self.conv_mask)
        _refresh(self.mesh, keep, self.conv_keep)
        self.user = user


# ------------------------------------------------------------------------------------- wave slot
@dataclass
class WaveSlot:
    """Device-side targets of a traced WAVE (``doc/mesh/row_ownership.md``, phase 1: one slot index per wave): every
    chip's own fill target (``cache_idx``: the user's slot on its owner column, the dummy row elsewhere and on every
    column of an idle row), and the per-row KDA carry blend masks (``rec_mask`` / ``rec_keep`` ``[B,1,1,1]`` fp32,
    ``conv_mask`` / ``conv_keep`` ``[taps,B,1]`` bf16: one-hot at ``user`` on a member row, all-keep on an idle row),
    so a replay writes each member row's carry into ITS slot and leaves an idle row's slot untouched. Allocated once
    before any capture; ``set`` refreshes the values per wave."""

    mesh: GLMMesh
    max_batch: int
    partition: bool
    users_per_chip: int
    cache_idx: ttnn.Tensor  # [1] int32 RM per chip (row-major chip order)
    scratch_idx: ttnn.Tensor | None
    conv_mask: ttnn.Tensor  # [taps, B, 1] bf16 TILE per row
    conv_keep: ttnn.Tensor
    rec_mask: ttnn.Tensor  # [B, 1, 1, 1] fp32 TILE per row
    rec_keep: ttnn.Tensor
    kda_zero: KdaState | None
    _taps: int = 3
    has_dsa: bool = True  # without DSA layers an idle row has no cache to protect
    user: int = -1
    idle: tuple = ()

    @classmethod
    def allocate(cls, mesh: GLMMesh, decoders) -> "WaveSlot":
        decoders = list(decoders)
        max_batch = decoders[0].max_batch
        kda = [d for d in decoders if d.is_kda]
        dsa = [d for d in decoders if not d.is_kda]
        partition = bool(dsa and dsa[0].partition)
        users_per_chip = max_batch // mesh.cols if partition else max_batch
        has_scratch = partition and dsa[0].cache.scratch is not None
        rm, i32 = ttnn.ROW_MAJOR_LAYOUT, ttnn.int32
        rows = mesh.rows
        chips = cls._chip_slots_host(partition, users_per_chip, rows, mesh.cols, 0, ())
        cache_idx = _upload(mesh, chips, i32, rm, mapper=mesh.shard_2d(0, 0))
        scratch_idx = _upload(mesh, torch.tensor([users_per_chip], dtype=torch.int32), i32, rm) if has_scratch else None
        kda_zero, taps = None, 3
        if kda:
            attn = kda[0]._require_weights().attn
            taps = attn.kcfg.conv_kernel_size - 1
            kda_zero = attn.allocate_state(1)
        cm, ck, rmk, rk = cls._masks_host(taps, max_batch, rows, 0, ())
        return cls(
            mesh=mesh,
            max_batch=max_batch,
            partition=partition,
            users_per_chip=users_per_chip,
            cache_idx=cache_idx,
            scratch_idx=scratch_idx,
            conv_mask=_upload(mesh, cm, ttnn.bfloat16, ttnn.TILE_LAYOUT, rows_dim=0),
            conv_keep=_upload(mesh, ck, ttnn.bfloat16, ttnn.TILE_LAYOUT, rows_dim=0),
            rec_mask=_upload(mesh, rmk, ttnn.float32, ttnn.TILE_LAYOUT, rows_dim=0),
            rec_keep=_upload(mesh, rk, ttnn.float32, ttnn.TILE_LAYOUT, rows_dim=0),
            kda_zero=kda_zero,
            _taps=taps,
            has_dsa=bool(dsa),
            user=0,
            idle=(),
        )

    @staticmethod
    def _chip_slots_host(
        partition: bool, upc: int, rows: int, cols: int, user: int, idle, has_dsa: bool = True
    ) -> torch.Tensor:
        """[rows * cols] int32: the slot every chip's paged fills write for a wave into ``user``."""
        if not partition:
            if idle and has_dsa:
                raise NotImplementedError("a traced wave with idle rows needs a partitioned cache (dummy rows)")
            return torch.full((rows * cols,), user, dtype=torch.int32)
        owner, slot = divmod(user, upc)
        out = []
        for r in range(rows):
            for c in range(cols):
                out.append(upc if (r in idle or c != owner) else slot)
        return torch.tensor(out, dtype=torch.int32)

    @staticmethod
    def _masks_host(taps: int, B: int, rows: int, user: int, idle):
        """Row-concatenated on dim 0 (``rows_dim=0`` hands row r its block): the conv masks ``[rows * taps, B, 1]`` ->
        ``[taps, B, 1]`` per chip, the recurrent masks ``[rows * B, 1, 1, 1]`` -> ``[B, 1, 1, 1]`` per chip."""
        cm = torch.zeros(rows, taps, B, 1)
        rmk = torch.zeros(rows, B, 1, 1, 1)
        for r in range(rows):
            if r not in idle:
                cm[r, :, user, 0] = 1.0
                rmk[r, user] = 1.0
        cm = cm.reshape(rows * taps, B, 1)
        rmk = rmk.reshape(rows * B, 1, 1, 1)
        return cm, 1.0 - cm, rmk, 1.0 - rmk

    def set(self, user: int, idle) -> None:
        if not 0 <= user < self.max_batch:
            raise ValueError(f"user {user} >= max_batch {self.max_batch}")
        idle = tuple(sorted(int(r) for r in idle))
        rows, cols = self.mesh.rows, self.mesh.cols
        _refresh(
            self.mesh,
            self._chip_slots_host(self.partition, self.users_per_chip, rows, cols, user, idle, self.has_dsa),
            self.cache_idx,
            mapper=self.mesh.shard_2d(0, 0),
        )
        cm, ck, rmk, rk = self._masks_host(self._taps, self.max_batch, rows, user, idle)
        for host, dst in ((cm, self.conv_mask), (ck, self.conv_keep), (rmk, self.rec_mask), (rk, self.rec_keep)):
            _refresh(self.mesh, host, dst, rows_dim=0)
        self.user, self.idle = user, idle


@dataclass
class WaveTraceInputs:
    """The persistent inputs of one traced wave length ``T`` (rows per mesh row; every member at start 0)."""

    T: int
    hidden: int
    ids: ttnn.Tensor  # [1,1,1,rows*T] uint32 RM row-sharded on dim 3: row r embeds its own T tokens
    starts: ttnn.Tensor  # [1,1,1,1] uint32 per row: 0 (the KDA actual_start)
    ends: ttnn.Tensor  # [1,1,1,1] uint32 per row: the row's valid length (ttKDA actual_end)
    ring_sel: ttnn.Tensor  # [1,1,32,T] bf16 per row: its ring-row selector (zeros on an idle row)
    row_mask: ttnn.Tensor  # [1,1,T,32] bf16 per row: the MoE pad-row mask of its chunk
    tile_start: ttnn.Tensor  # [4] int32 per row: the row's last token's tile
    tile_end: ttnn.Tensor
    onehot: ttnn.Tensor  # [1,1,32,32] bf16 per row: row 0 picks the row's last token inside the tile
    out_logits: ttnn.Tensor | None  # [1,1,1,V] bf16 per row: the row's last-token logits, copied inside the trace
    valids: tuple = ()
    trace_id: object = None
    streams: ttnn.Tensor | None = None

    @classmethod
    def allocate(cls, mesh: GLMMesh, T: int, hidden: int, vocab: int | None) -> "WaveTraceInputs":
        if T % TILE:
            raise ValueError(f"wave T={T} must be a multiple of {TILE}")
        rows = mesh.rows
        i32, rm = ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        one = lambda t, dt, lay, rd: _upload(mesh, t, dt, lay, rows_dim=rd)
        return cls(
            T=T,
            hidden=hidden,
            ids=one(torch.zeros(1, 1, 1, rows * T, dtype=torch.int32), ttnn.uint32, rm, 3),
            starts=one(torch.zeros(rows, 1, 1, 1, dtype=torch.int32), ttnn.uint32, rm, 0),
            ends=one(torch.full((rows, 1, 1, 1), T, dtype=torch.int32), ttnn.uint32, rm, 0),
            ring_sel=one(torch.zeros(rows, 1, TILE, T), ttnn.bfloat16, ttnn.TILE_LAYOUT, 0),
            row_mask=one(wave_row_mask_host(T, (T,) * rows), ttnn.bfloat16, ttnn.TILE_LAYOUT, 2),
            tile_start=one(torch.tensor([0, 0, T - TILE, 0] * rows, dtype=torch.int32), i32, rm, 0),
            tile_end=one(torch.tensor([1, 1, T, hidden] * rows, dtype=torch.int32), i32, rm, 0),
            onehot=one(torch.zeros(rows, 1, TILE, TILE), ttnn.bfloat16, ttnn.TILE_LAYOUT, 0),
            out_logits=one(torch.zeros(rows, 1, 1, vocab), ttnn.bfloat16, ttnn.TILE_LAYOUT, 0) if vocab else None,
        )

    def set_wave(self, mesh: GLMMesh, ids_host: torch.Tensor, valids: tuple[int, ...]) -> None:
        """``ids_host`` [1,1,1,rows*T] int32 (row r's padded chunk at [r*T, (r+1)*T)), ``valids[r]`` its token count
        (0 = idle)."""
        rows, T = mesh.rows, self.T
        if len(valids) != rows or any(not 0 <= v <= T for v in valids):
            raise ValueError(f"wave valids {valids} for {rows} rows of {T}")
        _refresh(mesh, ids_host, self.ids, rows_dim=3)
        _refresh(mesh, torch.tensor(valids, dtype=torch.int32).reshape(rows, 1, 1, 1), self.ends, rows_dim=0)
        sel = torch.cat([ring_selector(v, T, RING_ROWS) if v else torch.zeros(1, 1, TILE, T) for v in valids], dim=0)
        _refresh(mesh, sel, self.ring_sel, rows_dim=0)
        _refresh(mesh, wave_row_mask_host(T, valids), self.row_mask, rows_dim=2)
        last = [max(1, v) - 1 for v in valids]
        _refresh(
            mesh,
            torch.tensor([[0, 0, (n // TILE) * TILE, 0] for n in last], dtype=torch.int32).reshape(-1),
            self.tile_start,
            rows_dim=0,
        )
        _refresh(
            mesh,
            torch.tensor([[1, 1, (n // TILE) * TILE + TILE, self.hidden] for n in last], dtype=torch.int32).reshape(-1),
            self.tile_end,
            rows_dim=0,
        )
        onehot = torch.zeros(rows, 1, TILE, TILE)
        for r, n in enumerate(last):
            onehot[r, 0, 0, n % TILE] = 1.0
        _refresh(mesh, onehot, self.onehot, rows_dim=0)
        self.valids = tuple(valids)


# ---------------------------------------------------------------------------------- per-length inputs
@dataclass
class PrefillTraceInputs:
    """The persistent inputs of one traced length and, once captured, its trace id and output tensors."""

    T: int  # the padded chunk length (rows of the whole chunk)
    T_loc: int  # rows per mesh row (sequence-parallel slabs)
    hidden: int  # the head's hidden width (the tile slice's last bound)
    ids: ttnn.Tensor  # [1,1,1,T] uint32 RM, row-sharded (each row embeds its own slab)
    end: ttnn.Tensor  # [1,1,1,1] uint32 RM replicated: the prompt length S (ttKDA actual_end)
    ring_sel: ttnn.Tensor  # [1,1,32,T] bf16 TILE replicated: ring-row selector (rows j < 4)
    row_mask: ttnn.Tensor  # [1,1,T,32] bf16 TILE row-sharded: column 0 = 1.0 on rows < S (the MoE pad-row mask)
    tile_start: ttnn.Tensor  # [4] int32 replicated: [0, 0, tile0, 0], the last token's 32-row tile of the local rows
    tile_end: ttnn.Tensor  # [4] int32 replicated: [1, 1, tile0 + 32, hidden]
    out_logits: ttnn.Tensor | None  # [1,1,32,V] bf16 TILE: the tile's logits, copied inside the trace
    out_ids: ttnn.Tensor | None  # [1,1,32,1] uint32 RM: the tile's greedy ids, copied inside the trace
    S: int = 0
    trace_id: object = None
    streams: ttnn.Tensor | None = None  # the capture's last-layer output [1,1,T_loc,16384] (transient)

    @classmethod
    def allocate(cls, mesh: GLMMesh, T: int, hidden: int, vocab: int | None) -> "PrefillTraceInputs":
        """``vocab`` None: no head, no output buffers."""
        rows = mesh.rows
        if T % (TILE * rows):
            raise ValueError(f"T={T} must be a multiple of {TILE * rows} on a {rows}-row mesh")
        T_loc = T // rows
        i32, rm = ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        return cls(
            T=T,
            T_loc=T_loc,
            hidden=hidden,
            ids=_upload(mesh, torch.zeros(1, 1, 1, T, dtype=torch.int32), ttnn.uint32, rm, rows_dim=3),
            end=_upload(mesh, torch.tensor([[[[T]]]], dtype=torch.int32), ttnn.uint32, rm),
            ring_sel=_upload(mesh, torch.zeros(1, 1, TILE, T), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            row_mask=_upload(mesh, pad_row_mask_host(T, T), ttnn.bfloat16, ttnn.TILE_LAYOUT, rows_dim=2),
            tile_start=_upload(mesh, torch.tensor([0, 0, T_loc - TILE, 0], dtype=torch.int32), i32, rm),
            tile_end=_upload(mesh, torch.tensor([1, 1, T_loc, hidden], dtype=torch.int32), i32, rm),
            out_logits=_upload(mesh, torch.zeros(1, 1, TILE, vocab), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            if vocab
            else None,
            out_ids=_upload(mesh, torch.zeros(1, 1, TILE, 1, dtype=torch.int32), ttnn.uint32, rm) if vocab else None,
        )

    @property
    def outputs(self) -> tuple:
        """(streams, logits, ids) after a replay: the transient last-layer output and the copied tile outputs."""
        return self.streams, self.out_logits, self.out_ids

    # ---- host values
    @staticmethod
    def ring_selector(S: int, T: int) -> torch.Tensor:
        """[1,1,32,T]: row j < 4 selects the last prompt position p <= S-1 with p % 4 == j (``indexer.ring_selector``:
        shared with the eager shape-static maintenance)."""
        return ring_selector(S, T, RING_ROWS)

    def owner_row(self, S: int) -> int:
        """The mesh row whose slab holds the prompt's last token."""
        return (S - 1) // self.T_loc

    def last_row(self, S: int) -> int:
        """The row of the last token inside the 32-row tile the trace's head sees."""
        return ((S - 1) % self.T_loc) % TILE

    def set_prompt(self, mesh: GLMMesh, ids_host: torch.Tensor, S: int):
        """``ids_host`` [1,1,1,T] int32 (padded prompt ids), ``S`` the prompt length."""
        if not 1 <= S <= self.T:
            raise ValueError(f"prompt length {S} does not fit the {self.T}-token trace")
        _refresh(mesh, ids_host, self.ids, rows_dim=3)
        _refresh(mesh, torch.tensor([[[[S]]]], dtype=torch.int32), self.end)
        _refresh(mesh, self.ring_selector(S, self.T), self.ring_sel)
        _refresh(mesh, pad_row_mask_host(self.T, S), self.row_mask, rows_dim=2)  # rows >= S leave the router
        tile0 = ((S - 1) % self.T_loc) // TILE * TILE  # the same local offset on every row (the owner row is read)
        _refresh(mesh, torch.tensor([0, 0, tile0, 0], dtype=torch.int32), self.tile_start)
        _refresh(mesh, torch.tensor([1, 1, tile0 + TILE, self.hidden], dtype=torch.int32), self.tile_end)
        self.S = S


@dataclass
class PrefillTraceResult:
    """What ``GLM53MeshModel.prefill_traced`` returns: the trace's outputs and where the prompt's token is."""

    streams: ttnn.Tensor  # the last layer's output rows (this row's slab), transient
    logits: ttnn.Tensor | None  # [1,1,32,V] bf16: the logits of the last token's tile (persistent buffer)
    ids: ttnn.Tensor | None  # [1,1,32,1] uint32: its greedy ids (persistent buffer)
    owner_row: int  # the mesh row holding the last token
    row: int  # the last token's row inside the tile
    S: int
    T: int
