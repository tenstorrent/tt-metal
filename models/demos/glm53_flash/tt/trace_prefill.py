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
from models.demos.glm53_flash.tt.mesh import GLMMesh

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
    scratch_idx: ttnn.Tensor | None  # partitioned caches: the scratch slot every column fills as well
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
        rm, i32 = ttnn.ROW_MAJOR_LAYOUT, ttnn.int32
        cache_idx = _upload(mesh, torch.zeros(1, dtype=torch.int32), i32, rm, mapper=cls._cache_mapper(mesh, partition))
        scratch_idx = _upload(mesh, torch.tensor([users_per_chip], dtype=torch.int32), i32, rm) if partition else None
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

    def _cache_slots(self, user: int) -> torch.Tensor:
        """The slot every column chip's fills write: the user's slot (non-partitioned), or its slot on the
        owner column and the scratch slot elsewhere (``TtLatentCache._prefill_views_for``)."""
        if not self.partition:
            return torch.tensor([user], dtype=torch.int32)
        owner, slot = divmod(user, self.users_per_chip)
        return torch.tensor(
            [slot if c == owner else self.users_per_chip for c in range(self.mesh.cols)], dtype=torch.int32
        )

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
        """[1,1,32,T]: row j < 4 selects the last prompt position p <= S-1 with p % 4 == j (a zero row when the
        prompt has no such position: that ring slot then holds zeros, harmless because the pool it belongs to is
        incomplete until decode rewrites the slot)."""
        sel = torch.zeros(1, 1, TILE, T)
        for j in range(RING_ROWS):
            p = S - 1 - ((S - 1 - j) % RING_ROWS)
            if p >= 0:
                sel[0, 0, j, p] = 1.0
        return sel

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
