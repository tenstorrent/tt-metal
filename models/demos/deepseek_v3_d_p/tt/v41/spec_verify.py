# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The verify step of speculative decode: a slot's ``1 + K`` tokens (the
committed next token, then ``K`` drafts) as ``1 + K`` rows of the batched decode tile, at consecutive positions, in
one step; the host accepts the longest matching prefix and commits.

``TtV41VerifyDecoder`` is the batched decoder (``tt/v41/batch_decode.py``) with the per-row pool addresses as step
inputs instead of the tile row = pool slot rule:

* the window ring has ``RING + ROWS`` slots per layer and a position's slot is ``position % ring_slots``
  (``V41DecodePool.position_rings``: the join places the carry by position), so the later rows' writes never land
  in an earlier row's window; each row's 128 window slots are a step input (``wslots``, the slots of its positions
  ``p - 127 .. p``) offset on device by its user's row base per layer;
* every write scatters by an absolute row (``ring<layer>``, ``comp<source>``, ``key<source>``: ``scatter_rows`` with
  stride 0 over the 32 index words, the sentinel on idle rows), the selection's pins and the attention's visible
  bounds come per row from its position, so a rejected row's writes sit past every committed pin and are rewritten
  when the position is really processed;
* a ratio-2 row pools with the previous row's projections of the same step (rows shifted by one, a block's first
  row with its slot's carry) and the step keeps every row's projections in a scratch; the next step's first ops
  scatter the committed row's into the slot's carry (``carry<source>`` index: the only state a rejection undoes);
* the index selection loops over the live rows (``range_scores`` with the row apart from the user; the candidate
  scorer with the row's pin and blocks);
* the Engram ids of row ``j`` come from the slot's history extended by the block's first ``j`` tokens.

The sampler takes every row's argmax (or, for a slot with sampling parameters, the served sampler's draw on every
one of its rows: the match rule, exact for greedy drafts); ``decode`` returns each slot's rows'
tokens and leaves the slot where it was; ``commit(slot, accepted)`` advances it (the drafts
up to the first mismatch and the bonus token are the step's output). A slot stepping with ``K = 0`` is a plain row of the same graph. One live slot
per rank at most (its rows are ``u .. u + K`` of its rank's tile, within the MoE's real rows), the captures keyed by
the live rows' layout.

With a drafter attached (``attach_drafter``; its rings at the pool's ``ring_slots``, position-based) a step with
``K >= 1`` drafts runs the drafter inside its own graph after the sampler (``TtDSparkDrafter.fused_draft``): every
verify row's tap (computed in the step) is written to the drafter's rings at the row's slot and the drafter proposes
on every accept outcome's block at once, so ``decode`` also returns, in ``drafts[slot][a]``, the five drafts that
follow accepting ``a`` of the block; the round's host side is the accept,
the commit and the next step's inputs.

Under ``V41_DECODE_C1_MOE=1`` (the c1 capture) the MoEs run the live
slot's ``1 + K`` rows through ``C1Moe`` (``rows=1 + K``: the rows gathered to every chip, a selected expert computed
on all of them at once, the weights masking the rows that did not select it) instead of the 32-row tile; the
sublayer norms follow ``V41_DECODE_NORM`` and the host inputs ``V41_DECODE_HOST_INPUTS`` as the batched step's. The
index selection is per live row either way (``V41_DECODE_C1_INDEX``'s pages are not read).
"""

from __future__ import annotations

import os
import time

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.v41.attention import TOPK_ALIGN, TtV41Attention
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import ROWS, DeviceSampler, TtV41BatchDecoder
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads
from models.demos.deepseek_v3_d_p.tt.v41.indexer import CandidateBlocks, DecodePins, TtV41Indexer, _gather_cols
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import range_scores, split_geometry
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.v41.moe_c1 import C1_SELECT_ROWS, C1Moe
from models.demos.deepseek_v3_d_p.tt.v41.pool import RING, scatter_rows
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_e4m3_qdq, fp8_qdq
from models.demos.deepseek_v3_d_p.tt.v41.rope import cos_sin
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

SENTINEL32 = -1  # int32 host value of the uint32 sentinel
C1_ONEHOT_SENTINEL = 32  # the identity table's zero row (moe_c1._SENTINEL)


class TtV41VerifyDecoder(TtV41BatchDecoder):
    POOL_RING_SLOTS = RING + ROWS
    C1_ROWS = C1_SELECT_ROWS  # the c1 MoE on 1 + K rows, K <= 3

    def __init__(self, *args, real_rows: int | None = None, **kwargs):
        """``real_rows``: the MoE tile's real rows a rank (``padding_config``, the decode routing), decoupled from
        the pool's slots (``users_per_rank``) so that every slot of a rank can hold a block: ``users_per_rank x
        (1 + K)`` for ``K`` drafts a slot, at most 32 (the earlier geometry realised the rows through a
        pool of as many slots). Default: the slots (the c1 geometry: one block a rank, its draft rows past the
        slots' with the c1 MoE)."""
        self.real_rows = real_rows
        super().__init__(*args, **kwargs)
        assert self.real_rows is None or self.users <= self.real_rows <= ROWS, (self.real_rows, self.users)
        assert self.single_query, "verify rows need V41_DECODE_SINGLE_QUERY=1 (one scorer per live row)"
        assert self.packer is not None, "verify rows need V41_DECODE_HOST_INPUTS=packed or one"
        assert self.sample_on_device, "verify rows take every row's argmax on device"
        self.live_rows = 1  # the live slot's rows this step (one layout on every live rank)
        self.pool.position_rings = True
        self.ring_slots = self.pool.ring_slots
        self.rows: dict[int, list[int]] = {}  # slot -> its tile rows this step
        self.live: list[tuple[int, int]] = []  # (row, user) of the live rows this step, in row order
        self.histories: list = []  # per block row j: the Engram histories [U, n - 1] after the slots' row j
        self.history_index: dict[int, int] = {}  # slot -> its row of ``histories``
        self._layouts: dict[tuple, dict] = {}  # the step's row layout per (slot, block length) tuple (``_layout``)
        self._ring_consts: tuple | None = None  # per layer: the slot rows and the ring row (``_host_inputs``)
        self._no_draft_rows: tuple | None = None  # the fused call's rows when no fused call runs (constants)
        self.pending_carry: dict[int, int] = {}  # slot -> the scratch row to commit into its carry
        self.scratch = {
            src: tuple(self._replicated(torch.zeros(1, 1, ROWS, self.config.HEAD_DIM), ttnn.float32) for _ in range(2))
            for src in self.pool.carry
        }
        self.sentinel_topk = self._replicated(
            torch.full((1, 1, 1, self.config.INDEX_TOPK), SENTINEL32, dtype=torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self._windows: dict[int, ttnn.Tensor] = {}
        self._uoff: dict[int, ttnn.Tensor] = {}
        # with a seeder on the model: the drafter's tap of every tile row, [1, 1, ROWS, n_taps * hidden/tp] bf16 per
        # chip (the hc-mean block inputs of the tap layers, chip-major), computed inside the step and kept at a fixed
        # address per capture (``taps`` after a step: the drafter's ``stage(tap32=)`` input)
        self.taps = None
        self.capture_taps: dict[str, ttnn.Tensor] = {}
        self.drafter = None  # ``attach_drafter``
        self.draft_plain = False  # the drafter runs on a plain step too (one live slot a rank; ``blocks = 1``)
        self.drafts: dict[int, list[list[int]]] = {}  # slot -> per accept outcome a: the five drafts after it
        self.draft_layout: tuple | None = None  # (user, blocks) of the fused draft call this step, or None
        self.draft_rows: tuple | None = None  # the live slot's tile rows this step (``block_rows``)
        # the merged index selection of the live rows (V41_VR_MERGE=rows, the default; ``row`` is the per-row chain
        # ): per (tile rows, split rows R) the replicated index page that stacks the rows'
        # pins, built before any capture (``prepare_merge``; a constant allocated after a capture may be overwritten
        # by its replay)
        self.merge_rows = os.environ.get("V41_VR_MERGE", "rows") == "rows"
        self._stack_idx: dict[tuple, ttnn.Tensor] = {}

    def attach_drafter(self, drafter, users: list[int] | None = None, max_k: int = 3) -> None:
        """Run ``drafter`` (a ``TtDSparkDrafter`` at this pool's ``ring_slots``) inside the steps with drafts; its
        constants for the live user indices ``users`` (default: every one) and ``K <= max_k`` are allocated now."""
        assert drafter.ring_slots == self.ring_slots, (drafter.ring_slots, self.ring_slots)
        assert drafter.users in (1, self.users), (drafter.users, self.users)
        assert self.model.dspark is not None, "the fused draft call needs the model's seeder (dspark_weights)"
        self.drafter = drafter
        drafter.gather_sp = self.pool.gather_sp  # the standalone call's drafts gathered over SP: one readback
        layouts = []
        for u in list(range(self.users)) if users is None else users:
            for k in range(max_k + 1):
                try:
                    layouts.append(self.block_rows(u, k, one_slot=True))
                except AssertionError:
                    pass
        drafter.prepare_fused(layouts)
        self.prepare_merge(layouts)

    def prepare_merge(self, layouts: list[tuple]) -> None:
        """The stacked pin index pages of every live-row layout (``block_rows`` tuples) the merged selection may
        see, for every split geometry of the pool's index sources: entry ``l * R + r`` is layout row ``l``'s tile
        row (a replicated uint32 row-major constant; the merged path takes a layout only when its page exists)."""
        if not self.merge_rows:
            return
        widths = {self.pool.slots[src].cap + 32 for src in self.pool.slots if src >= 0}
        for rows in layouts:
            if len(rows) < 2:
                continue
            for width in widths:
                n_rows = split_geometry(width)[0]
                key = (tuple(rows), n_rows)
                if key in self._stack_idx:
                    continue
                idx = torch.tensor([row for row in rows for _ in range(n_rows)], dtype=torch.int32)
                self._stack_idx[key] = self._replicated(
                    idx.view(1, 1, 1, len(rows) * n_rows), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                )

    def moe_real_rows(self) -> int:
        return self.real_rows or self.users

    @property
    def slot_stride(self) -> int:
        """The tile rows a slot owns in the multi-slot geometry: ``real_rows / users_per_rank``."""
        return self.moe_real_rows() // self.users

    def block_rows(self, u: int, k: int, one_slot: bool) -> tuple:
        """The tile rows of user ``u``'s block of ``1 + k`` tokens: its own row, then ``k`` rows. With the c1 MoE
        and one live slot the draft rows are the tile's rows past the users' (``users .. users + k - 1``: the c1
        MoE gathers any rows; the 32-row MoE dispatches the real rows only), else the slot's own rows
        ``u s .. u s + k`` for the stride ``s = real_rows / users_per_rank`` (``s = 1`` without decoupled real
        rows: ``u .. u + k``, within the slots' rows)."""
        if k and self.c1 and one_slot:
            rows = (u,) + tuple(range(self.users, self.users + k))
        else:
            s = self.slot_stride
            rows = tuple(range(u * s, u * s + k + 1))
            assert (
                rows[-1] < self.moe_real_rows()
            ), f"user {u}: {k} drafts need rows {rows} within {self.moe_real_rows()}"
        assert rows[-1] < ROWS and k < self.C1_ROWS, (rows, k)
        return rows

    # --- the step's layout ---------------------------------------------------------------------------------------
    def variant(self, tokens: dict) -> str:
        """The capture key: the live slots' count (one: the c1 MoE), the set of (user, rows) layouts over the ranks
        (the graph is one program on every rank: a rank's live rows are its inputs, the sentinel elsewhere; the
        user index and the row count are baked in) and whether the drafter runs."""
        layouts = sorted({(slot % self.users, len(block)) for slot, block in tokens.items()})
        parts = [f"{u}.{rows}" for u, rows in layouts]
        c1 = "c" if self.c1 and len(tokens) == 1 else ""
        return ("vd:" if self._fused_layout(tokens) else "v:") + f"{c1}{len(tokens)}:" + ",".join(parts)

    def _fused_layout(self, tokens: dict) -> tuple | None:
        """(user, blocks) of the fused draft call for ``tokens``: one live slot per live rank, every one at the same
        user index and block length (the drafter's graph is one per rank); else None (no drafts proposed: a step
        of several slots a rank, or of slots at different user indices, runs as plain or verify rows alone).
        ``draft_plain``: a step of one token a slot proposes too (``blocks = 1``: the served round's first step)."""
        if self.drafter is None or not tokens:
            return None
        ranks = [slot // self.users for slot in tokens]
        if len(set(ranks)) != len(ranks):
            return None  # several slots a rank: the standalone draft call after the step 
        layouts = {(slot % self.users, len(block)) for slot, block in tokens.items()}
        if len(layouts) != 1:
            assert all(len(block) == 1 for block in tokens.values()), f"one (user, rows) layout a step: {layouts}"
            return None
        ((user, rows),) = layouts
        return (user, rows) if rows > 1 or self.draft_plain else None

    def join(self, slot: int, tokens: torch.Tensor, state) -> None:
        super().join(slot, tokens, state)
        self.pending_carry.pop(slot, None)
        u, s = slot % self.users, self.slot_stride
        if s > 1 and any(c is not None for c in state.compressor_carry.values()):
            # the join placed the odd prompt's carry at the slot's user row ``u`` (``pool.join_ops``), which a
            # one-slot step's first row reads; a multi-slot step's first row is tile row ``u s``: the carry
            # copied there too (a later step's commit re-places it at its own first row)
            idx = torch.full((1, 1, 1, ROWS), SENTINEL32, dtype=torch.int32)
            idx[0, 0, 0, u] = u * s
            idx = self._replicated(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            for k0, s0 in self.pool.carry.values():
                for dst in (k0, s0):
                    src_rows = ttnn.to_layout(dst, ttnn.ROW_MAJOR_LAYOUT)
                    dst_rows = ttnn.to_layout(dst, ttnn.ROW_MAJOR_LAYOUT)
                    scatter_rows(src_rows, dst_rows, idx, rows=ROWS)
                    ttnn.copy(ttnn.to_layout(dst_rows, ttnn.TILE_LAYOUT), dst)
                    ttnn.deallocate(src_rows)
                    ttnn.deallocate(dst_rows)
            ttnn.deallocate(idx)

    # --- host inputs ---------------------------------------------------------------------------------------------
    def _layout(self, tokens: dict, order: list[int]) -> dict:
        """The step's row layout for ``tokens`` (slot -> block; ``order`` the slots sorted): the slots' tile rows,
        the live (row, user) pairs, the draft rows, and the entries' rank, row, block index and user as index
        tensors (one entry a tile row)."""
        users = self.users
        rows_of, live, taken, entries = {}, [], set(), []
        draft_rows = None
        for slot in order:
            block = tokens[slot]
            r, u = divmod(slot, users)
            k = len(block) - 1
            assert len(block) == self.live_rows, "one (user, rows) layout on every live rank"
            block_rows = self.block_rows(u, k, one_slot=len(tokens) == 1)
            draft_rows = block_rows
            rows = []
            for j in range(len(block)):
                row = block_rows[j]
                assert (r, row) not in taken, f"slot {slot}: row {row} of rank {r} already live"
                taken.add((r, row))
                entries.append((r, row, j, u))
                rows.append(row)
            rows_of[slot] = rows
            live += [(row, u) for row in rows if (row, u) not in live]  # one selection per tile row
        e = torch.tensor(entries, dtype=torch.int64).view(-1, 4)
        # the Engram rows' flat index, block row j of every slot in ``order`` for each j (``hash_.step``'s order)
        flat = [(slot // users) * ROWS + rows_of[slot][j] for j in range(self.live_rows) for slot in order]
        return {
            "rows": rows_of,
            "live": live,
            "draft_rows": draft_rows,
            "er": e[:, 0],
            "erow": e[:, 1],
            "ej": e[:, 2],
            "eu": e[:, 3],
            "flat": torch.tensor(flat, dtype=torch.int64),
        }

    def _host_inputs(self, tokens: dict[int, list[int]], sampling: dict | None = None) -> dict:
        cfg, sp, users, pool = self.config, self.sp, self.users, self.pool
        tick = self._tick("in")
        ids = torch.zeros(sp, 1, ROWS, dtype=torch.int32)
        positions = torch.zeros(sp, ROWS, dtype=torch.int64)
        active = torch.zeros(sp, ROWS, dtype=torch.bool)
        user_of = torch.zeros(sp, ROWS, dtype=torch.int64)
        first = torch.zeros(sp, ROWS, dtype=torch.float32)
        carry_rows = {src: torch.full((sp, ROWS), SENTINEL32, dtype=torch.int32) for src in pool.carry}
        c1sel = torch.full((sp, ROWS), C1_ONEHOT_SENTINEL, dtype=torch.int32)
        c1put = torch.full((sp, ROWS), C1_ONEHOT_SENTINEL, dtype=torch.int32)
        self.live_rows = max((len(block) for block in tokens.values()), default=1)
        # the step's row layout depends on the slots and their block lengths alone: the entries' (rank, row, j,
        # user) and the slots' rows are built once per layout and kept; the tokens and the
        # positions are the step's
        order = sorted(tokens)
        key = tuple((slot, len(tokens[slot])) for slot in order)
        layout = self._layouts.get(key)
        if layout is None:
            layout = self._layouts[key] = self._layout(tokens, order)
        self.rows, self.live, self.draft_rows = layout["rows"], layout["live"], layout["draft_rows"]
        for slot in order:
            committed = self.pending_carry.pop(slot, None)
            if committed is not None:
                # scratch row ``committed`` -> the carry row ``_compress`` reads for this block's first row: its
                # own tile row (``k0 * first`` is row-aligned), which is the slot's user row ``u`` in the c1
                # geometry and ``u s`` with decoupled real rows
                for src in carry_rows:
                    carry_rows[src][slot // users, committed] = self.rows[slot][0]
        if order:
            er, erow, ej, eu = layout["er"], layout["erow"], layout["ej"], layout["eu"]
            tok = torch.tensor([int(t) for slot in order for t in tokens[slot]], dtype=torch.int32)
            pos = torch.tensor([self.slots[slot].position for slot in order], dtype=torch.int64)
            pos = pos.repeat_interleave(self.live_rows) + ej
            c1sel[er, ej], c1put[er, erow] = erow.to(torch.int32), ej.to(torch.int32)  # the c1 MoE's row maps
            ids[er, 0, erow] = tok
            positions[er, erow] = pos
            active[er, erow] = True
            user_of[er, erow] = eu
            first[er, erow] = (ej == 0).float()
        tick("rows")
        ints = {"token": ids.view(sp, ROWS)}
        if self.c1:
            assert self.live_rows <= self.C1_ROWS, (self.live_rows, self.C1_ROWS)
            ints["c1sel"], ints["c1put"] = c1sel, c1put
        floats, rope, visible_cols = {}, {}, {}
        if self.sample_on_device:
            # the argmax on every row by default; a slot with sampling parameters (``sampling``: slot -> (k, p,
            # temperature), temperature 0 greedy) samples on every one of its rows as the batched step does on
            # one: the served sampler's draw on row j given the block's first j + 1 tokens, and ``accept``'s
            # "the draw equals the next draft" is the exact match rule for greedy drafts (the committed stream has the sequential served sampler's law; at temperature 0 it is the greedy
            # longest prefix)
            ints["k"] = torch.ones(sp, ROWS, dtype=torch.int32)
            floats["p"] = torch.zeros(sp, ROWS, dtype=torch.float32)
            floats["temp"] = torch.ones(sp, ROWS, dtype=torch.float32)
            for slot, (sk, sp_, st) in (sampling or {}).items():
                if slot not in tokens or st <= 0:
                    continue
                r = slot // users
                for row in self.rows[slot]:
                    ints["k"][r, row] = max(1, min(int(sk), 32)) if sk > 0 else 32
                    floats["p"][r, row], floats["temp"][r, row] = float(sp_), 1.0 / float(st)
        rope["rope0"] = self.rope_tables.rows(False, positions)
        rope["ropec"] = self.rope_tables.rows(True, positions)
        rope["ropeg"] = self.rope_tables.rows(True, positions // 2 * 2)
        tick("rope")
        # the ring: a position's slot, as an absolute row of every layer's KV tensor (idle rows the sentinel); the
        # layers at once (forty ``where`` calls a step before)
        slot_of = positions % self.ring_slots
        layers = self.model.layers
        if self._ring_consts is None:
            self._ring_consts = (
                torch.tensor([pool.slot_rows(layer) for layer in layers], dtype=torch.int64).view(-1, 1, 1),
                torch.tensor([pool.ring_row(layer) for layer in layers], dtype=torch.int64).view(-1, 1, 1),
            )
        slot_rows_l, ring_row_l = self._ring_consts
        base = user_of.unsqueeze(0) * slot_rows_l + ring_row_l + slot_of.unsqueeze(0)  # [layers, sp, ROWS]
        ring = torch.where(active.unsqueeze(0), base, torch.full_like(base, SENTINEL32)).to(torch.int32)
        for i, layer in enumerate(layers):
            ints[f"ring{layer}"] = ring[i]
        # the window: each row's 128 slots (positions p - 127 .. p), offset on device by the user's row base per layer
        wslots = (positions.view(sp, ROWS, 1) - (RING - 1) + torch.arange(RING).view(1, 1, RING)) % self.ring_slots
        wslots = torch.where(active.view(sp, ROWS, 1), wslots, torch.zeros_like(wslots))
        for src, slot in pool.slots.items():
            visible_cols[f"wu{src}"] = (user_of * slot.rows).float()
        for src in pool.carry:
            ints[f"carry{src}"] = carry_rows[src]
        visible_cols["first"] = first
        visible_cols["notfirst"] = 1.0 - first
        for src, slot in pool.slots.items():
            if src < 0:
                continue
            ratio = cfg.compress_ratio(src)
            visible = torch.where(active, (positions + 1) // ratio, torch.zeros_like(positions))
            newest = torch.where(visible > 0, visible - 1, torch.full_like(visible, SENTINEL32)).to(torch.int32)
            writes = active & (positions % ratio == ratio - 1)
            comp = torch.where(
                writes,
                user_of * slot.rows + pool.comp_row(src) + positions // ratio,
                torch.full_like(positions, SENTINEL32),
            )
            key = torch.where(
                writes, user_of * (slot.cap + 32) + positions // ratio, torch.full_like(positions, SENTINEL32)
            )
            ints[f"newest{ratio}"] = newest  # the same for every source of the ratio
            ints[f"comp{src}"], ints[f"key{src}"] = comp.to(torch.int32), key.to(torch.int32)
            visible_cols[f"uoff{src}"] = (user_of * slot.rows + pool.comp_row(src)).float()
        tick("addresses")
        wide = {}
        self.draft_layout = self._fused_layout(tokens)
        if self.drafter is not None and self.draft_layout is None:
            # no fused call this step (the multi-slot round: the standalone call drafts after it): the packer's
            # layout is fixed, so its rows travel as constants (the RoPE of 256 positions
            # and the block rows were built and uploaded for nothing)
            if self._no_draft_rows is None:
                rope_dim = cfg.QK_ROPE_HEAD_DIM
                self._no_draft_rows = (
                    torch.full((sp, ROWS), SENTINEL32, dtype=torch.int32),
                    torch.zeros(sp * ROWS, rope_dim),
                    (torch.zeros(sp * ROWS, rope_dim, dtype=torch.bfloat16),) * 2,
                )
            ints["dring"], zeros, rope["dblock"] = self._no_draft_rows
            wide["dcos"] = wide["dsin"] = zeros
            tick("draft_rows")
        elif self.drafter is not None:
            # the fused draft call's rows: every verify row's ring slot and fp32 RoPE pair (the ring write), the
            # block rows' bf16 pair (block a's row i at p + a + 1 + i; the rows past the blocks at p + 1 + row)
            user, blocks = self.draft_layout
            # the ring row of every verify row's tap: the user's ring with rings per slot (``TtDSparkDrafter(users=)``)
            ring_row = slot_of + (user_of * self.drafter.ring_slots if self.drafter.users > 1 else 0)
            ints["dring"] = torch.where(active, ring_row, torch.full_like(slot_of, SENTINEL32)).to(torch.int32)
            cos, sin = cos_sin(cfg, False, positions.flatten())
            wide["dcos"], wide["dsin"] = cos.view(sp * ROWS, -1), sin.view(sp * ROWS, -1)
            base = positions[:, user].view(sp, 1)  # the live slot's frontier p per rank
            offs = torch.arange(ROWS).view(1, ROWS)
            offs = torch.where(offs < 5 * blocks, offs // 5 + 1 + offs % 5, offs + 1)
            bpos = torch.where(active[:, user].view(sp, 1), base + offs, torch.zeros(sp, ROWS, dtype=torch.int64))
            rope["dblock"] = self.rope_tables.rows(False, bpos)
            tick("draft_rows")
        inputs = self.packer.pack(ints, floats, rope, visible_cols, wide)
        inputs["wslots"] = pool.per_rank(
            wslots.float().view(sp, 1, ROWS, RING), ttnn.float32, ttnn.TILE_LAYOUT, device=False
        )
        tick("pack")
        if self.model.engram:
            # the hash of row j of every block at once (one ``step`` call a block row over the live slots, not
            # one a row), the histories after each of the block's tokens kept per slot
            hash_ = self.model.engram_hash
            h = torch.stack([self.slots[slot].engram_history.view(-1) for slot in order])  # [U, n - 1]
            # the histories after row j of every block, one [U, n - 1] tensor a row (``step`` returns a fresh
            # one); ``commit`` clones the accepted row's for its slot (128 clones a step before)
            self.histories, self.history_index = [], {slot: i for i, slot in enumerate(order)}
            ids_rows = []
            for j in range(self.live_rows):
                row_ids, h = hash_.step(tok[j :: self.live_rows], h)
                self.histories.append(h)
                ids_rows.append(row_ids)  # [U, n_layers, n_hash_cols]
            ids_rows = torch.cat(ids_rows, dim=0)
            flat = layout["flat"]
            tick("engram_hash")
            inputs["engram"] = {}
            live = active.flatten().nonzero().flatten()
            for l, e in self.model.engram.items():
                ids_l = torch.zeros(sp * ROWS, hash_.n_hash_cols, dtype=torch.int64)
                ids_l[flat] = ids_rows[:, hash_.layer_index(l)]
                inputs["engram"][l] = e.prepare_host(ids_l, sp * ROWS, live_rows=live)
            tick("engram_rows")
        return inputs

    # --- one step ------------------------------------------------------------------------------------------------
    def decode(self, tokens: dict, sampling: dict | None = None) -> dict:
        """``tokens``: slot -> its block (the committed next token, then its drafts) -> slot -> the argmax of each
        of its rows. The slots do not advance: ``commit`` does. A slot given one token as an int (the batched
        decoder's contract: the generator's warm-up) gets its argmax back as an int and advances by one."""
        scalar = {slot for slot, block in tokens.items() if not isinstance(block, (list, tuple))}
        if scalar:
            tokens = {slot: ([int(block)] if slot in scalar else block) for slot, block in tokens.items()}
            out = self.decode(tokens, sampling)
            for slot in scalar:
                self.commit(slot, 0)
            return {slot: (v[0] if slot in scalar else v) for slot, v in out.items()}
        for slot, block in tokens.items():
            user = self.slots[slot]
            assert user.position + len(block) <= self.pool.max_seq_len, f"slot {slot} at {user.position}: full"
        self.items = {}
        t0 = time.time()
        inputs = self._host_inputs(tokens, sampling)
        t1 = time.time()
        self._stage(inputs)
        t2 = time.time()
        variant = self.variant(tokens)
        self.last_variant = variant
        if not self.trace:
            mode = "eager"
        elif variant in self.captures:
            mode = "replay"
        elif self.eager_counts.get(variant, 0) >= self.warmup:
            mode = "capture"
        else:
            mode = "eager"
        self.last = {"mode": mode, "inputs_s": round(t1 - t0, 4), "stage_s": round(t2 - t1, 4), "variant": variant}
        if self.input_timing:
            self.last["items"] = dict(self.items)
        if mode == "eager":
            self.eager_counts[variant] = self.eager_counts.get(variant, 0) + 1
            logits = self._forward(self.on_block)
        else:
            if mode == "capture":
                self.captures[variant] = self._capture_step()
                self.capture_taps[variant] = self.taps
            capture = self.captures[variant]
            self.taps = self.capture_taps.get(variant)
            if self.replay_sync:
                ttnn.synchronize_device(self.mesh_device)
            t3 = time.time()
            capture.controller.replay(blocking=False)
            t4 = time.time()
            ttnn.synchronize_device(self.mesh_device)
            t5 = time.time()
            logits = capture.logits
            self.last.update(sync_s=round(t3 - t2, 4), device_s=round(t5 - t3, 4), issue_s=round(t4 - t3, 4))
            if self.on_block is not None:
                for layer, x in capture.blocks:
                    self.on_block(layer, x)
        t6 = time.time()
        read = DeviceSampler.rows_to_host(logits)  # [sp, 32 (+ 32 drafts)]
        out = {slot: [int(read[slot // self.users, r]) for r in rows] for slot, rows in self.rows.items()}
        self.drafts = {}
        if self.draft_layout is not None:
            blocks = self.draft_layout[1]
            for slot in self.rows:
                r = slot // self.users
                self.drafts[slot] = [[int(v) for v in read[r, ROWS + 5 * a : ROWS + 5 * a + 5]] for a in range(blocks)]
        if mode == "eager":
            ttnn.deallocate(logits)
        self.last["read_s"] = round(time.time() - t6, 4)
        self.pool.step += 1
        return out

    def commit(self, slot: int, accepted: int) -> None:
        """After ``decode``: ``accepted`` of the slot's drafts matched (the rows ``0 .. accepted`` are committed):
        the slot advances by ``accepted + 1``, its Engram history follows, and the next step commits row
        ``accepted``'s ratio-2 projections as its carry."""
        rows = self.rows[slot]
        assert 0 <= accepted < len(rows), (accepted, rows)
        self.slots[slot].position += accepted + 1
        if self.model.engram:
            self.slots[slot].engram_history = self.histories[accepted][self.history_index[slot]].clone()
        self.pending_carry[slot] = rows[accepted]

    @staticmethod
    def accept(block: list[int], argmax: list[int]) -> tuple[int, list[int]]:
        """The acceptance rule on a block ``[x, d_1 .. d_K]`` and its rows' tokens (the argmaxes, or the sampled
        draws): the accepted count ``a`` (the drafts ``d_1 .. d_a`` equal the tokens of rows ``0 .. a - 1``) and
        the emitted tokens ``argmax[0 .. a]`` (the accepted drafts and the bonus). With sampled rows this is the
        match rule (``round_match``): exact against sequential served sampling."""
        a = 0
        while a < len(block) - 1 and block[a + 1] == argmax[a]:
            a += 1
        return a, argmax[: a + 1]

    # --- the graph -----------------------------------------------------------------------------------------------
    def _commit_carries(self) -> None:
        """The previous step's committed rows' projections into the slots' carry rows (``carry<source>``)."""
        for src, (k0, s0) in self.pool.carry.items():
            idx = self.inputs[f"carry{src}"]
            for scratch, dst in zip(self.scratch[src], (k0, s0)):
                src_rows = ttnn.to_layout(scratch, ttnn.ROW_MAJOR_LAYOUT)
                dst_rows = ttnn.to_layout(dst, ttnn.ROW_MAJOR_LAYOUT)
                scatter_rows(src_rows, dst_rows, idx, rows=ROWS)
                ttnn.copy(ttnn.to_layout(dst_rows, ttnn.TILE_LAYOUT), dst)
                ttnn.deallocate(src_rows)
                ttnn.deallocate(dst_rows)

    def _forward(self, on_block, c1: bool = False):
        c1 = self.c1 and len(self.rows) == 1  # the c1 MoE: one live slot (its 1 + K rows)
        model = self.model
        self.inputs = dict(self.packer.unpack(self.buffers), engram=self.buffers.get("engram", {}))
        self.inputs["token"] = ttnn.reshape(self.inputs["token"], (1, 1, ROWS))
        self.inputs["wslots"] = self.buffers["wslots"]
        if "dcos" in self.inputs:
            self.inputs["dmain"] = (self.inputs["dcos"], self.inputs["dsin"])
        self._windows, self._uoff = {}, {}
        self._commit_carries()
        selection = {}
        h = model.embedding(self.inputs["token"])
        x = mhc_expand(ttnn.typecast(h, ttnn.float32), self.config.HC_MULT)
        pre = self.pre_mix
        taps = []
        if c1:
            sel, put = C1Moe.onehots(self.inputs["c1sel"], self.inputs["c1put"], self.c1_table)
            rows = self.live_rows
        moes = self.c1_moes if c1 else self.moes
        for layer, block, moe in zip(model.layers, model.blocks, moes):
            if layer in model.engram:
                x = model.engram[layer](x, self.inputs["engram"][layer])
            if model.dspark is not None and layer in self.config.DSPARK_TARGET_LAYER_IDS:
                taps.append(model.dspark.tap(x))  # the block's input: the tap of every row
            attn_norm, ffn_norm = self.decode_norms.get(layer, (block.attn_norm, block.ffn_norm))
            x, pre = block.residual(
                x,
                pre,
                attention=lambda h, a=block.attn, n=attn_norm: self._attention(a, n(h), {}, selection),
                ffn=(
                    (lambda h, m=moe, n=ffn_norm: m(n(h), sel, put, rows))
                    if c1
                    else (lambda h, m=moe, n=ffn_norm: self._ffn(m, n(h)))
                ),
            )
            if on_block is not None:
                on_block(layer, x)
        if taps:
            self.taps = ttnn.concat(taps, dim=3)
        final = ttnn.typecast(model.blocks[-1].residual.final_collapse(x, pre), ttnn.bfloat16)
        logits, _ = model.head(final, 0)
        ids = self.sampler.sample(logits, self.inputs)  # [1, 1, 1, ROWS]: the rows' argmaxes
        if self.draft_layout is not None:
            drafts = self.drafter.fused_draft(self.taps, ids, self.inputs, self.draft_rows, self.draft_layout[0])
            ids = ttnn.concat([ids, drafts], dim=3)  # [1, 1, 1, 64]
        return self.sampler.gather(ids)

    def _capture_step(self, c1: bool = False):
        """The step's capture, the drafter's MoEs on the trace controller too when its call is in the graph."""
        if self.draft_layout is None:
            return super()._capture_step(c1)
        from models.demos.deepseek_v3_d_p.tt.v41.decode import _Trace

        controller = SubDeviceTraceController(self.mesh_device)
        moes = list(self.moes) + [stage.ffn for stage in self.drafter.stages]
        for moe in moes:
            moe.set_trace_controller(controller)
        blocks = []
        ttnn.synchronize_device(self.mesh_device)
        controller.begin_capture()
        try:
            logits = self._forward(lambda layer, x: blocks.append((layer, x)), c1=c1)
        finally:
            controller.end_capture()
            for moe in moes:
                moe.set_trace_controller(None)
        return _Trace(controller, logits, 0, blocks)

    def _window_rows(self, layer: int):
        """``sparse_sdpa`` window rows of the step ``[1, 1, ROWS, RING]`` uint32: each row's slots offset by its
        user's row base in ``layer``'s KV tensor."""
        if layer not in self._windows:
            pool = self.pool
            w = ttnn.add(self.inputs["wslots"], self.inputs[f"wu{pool.source_of(layer)}"])
            w = ttnn.add(w, float(pool.ring_row(layer)))
            self._windows[layer] = ttnn.to_layout(
                ttnn.typecast(ttnn.typecast(w, ttnn.int32), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT
            )
        return self._windows[layer]

    def _user_offsets(self, src: int):
        """``[1, 1, ROWS, 1]`` int32 tiled: each row's offset of its user's compressed row 0 in ``src``'s pool."""
        if src not in self._uoff:
            self._uoff[src] = ttnn.typecast(self.inputs[f"uoff{src}"], ttnn.int32)
        return self._uoff[src]

    def _attention(self, attn: TtV41Attention, x, masks: dict, selection: dict):
        layer, pool, cfg = attn.layer, self.pool, self.config
        cos, sin = self.inputs["ropec" if attn.ratio else "rope0"]
        xq = fp8_qdq(x)
        qr = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wq_a")), weight=attn.q_norm, epsilon=attn.eps)
        q = attn._dense(fp8_qdq(qr), "wq_b")
        kv = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wkv")), weight=attn.kv_norm, epsilon=attn.eps)
        kv_format = pool.kv_format if attn.ratio else MlaKvCacheFormat.BF16_RM
        reference_values = kv_format == MlaKvCacheFormat.BF16_RM
        kv = attn._rope(kv, cos, sin)
        kv_tensor = pool.kv_tensor(layer)
        rows = self._encode(fp8_qdq(kv) if reference_values else kv, kv_format)
        scatter_rows(rows, kv_tensor, self.inputs[f"ring{layer}"], rows=ROWS)
        ttnn.deallocate(rows)

        compressed = None
        if attn.ratio:
            ratio, src = attn.ratio, cfg.kv_source(layer)
            if attn.is_kv_source:
                latent = self._compress(attn.compressor, x, layer)
                c_rope = self.inputs["ropeg" if ratio == 2 else "ropec"]
                keys = attn.index_keys(latent, c_rope)
                comp_kv = attn._rope(latent, *c_rope)
                rows = self._encode(fp4_e4m3_qdq(comp_kv) if reference_values else comp_kv, kv_format)
                scatter_rows(rows, pool.kv[src], self.inputs[f"comp{src}"], rows=ROWS)
                ttnn.deallocate(rows)
                keys = ttnn.to_layout(keys, ttnn.ROW_MAJOR_LAYOUT)
                pool.write_index_keys(src, keys, self.inputs[f"key{src}"], rows=ROWS)
                ttnn.deallocate(keys)
            if attn.is_index_source:
                compressed, candidates = self._index(attn.indexer, x, qr, src, None, selection.get("candidates"))
                selection["topk"] = compressed
                if candidates is not None:
                    selection["candidates"] = candidates
            else:
                compressed = selection["topk"]
        rows = self._index_rows(layer, compressed)

        k_chunk = next(c for c in (128, 64, 32) if rows.shape[-1] % c == 0)
        if self.sdpa_local and self.tp > 1:
            # the batched step's V41_DECODE_SDPA=local: this chip's heads alone, padded with
            # zero heads to sparse_sdpa's 32; no all-gather of the queries, no partition of the output
            out = ttnn.transformer.sparse_sdpa(
                q_heads(q, cos, sin, attn.trans_mat, attn.local_heads, attn.rope_dim, out_heads=attn.sdpa_heads),
                kv_tensor,
                rows,
                attn.head_dim,
                kv_format=kv_format.sparse_sdpa_format,
                scale=attn.scale,
                k_chunk_size=k_chunk,
                attention_sink=attn.sink_local,
            )
            out = o_heads(
                out, cos, ttnn.neg(sin), attn.trans_mat, attn.o_groups // self.tp, attn.rope_dim, heads=attn.local_heads
            )
            return attn._o_proj(out)
        q = attn.ccl.tp_all_gather(q, dim=3)
        out = ttnn.transformer.sparse_sdpa(
            q_heads(q, cos, sin, attn.trans_mat, attn.heads, attn.rope_dim),
            kv_tensor,
            rows,
            attn.head_dim,
            kv_format=kv_format.sparse_sdpa_format,
            scale=attn.scale,
            k_chunk_size=k_chunk,
            attention_sink=attn.sink,
        )
        out = o_heads(out, cos, ttnn.neg(sin), attn.trans_mat, attn.o_groups, attn.rope_dim)
        if self.tp > 1:
            out = ttnn.mesh_partition(out, dim=1, cluster_axis=TP_AXIS)
        return attn._o_proj(out)

    @staticmethod
    def _shift_rows(t):
        """``[1, 1, ROWS, W]`` tiled fp32 -> the rows shifted down by one (row i = row i - 1; row 0 = row 0)."""
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        w = rm.shape[3]
        head = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, 1, w])
        body = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, ROWS - 1, w])
        out = ttnn.to_layout(ttnn.concat([head, body], dim=2), ttnn.TILE_LAYOUT)
        for u in (rm, head, body):
            ttnn.deallocate(u)
        return out

    def _compress(self, compressor, x, layer: int):
        """As the batched step's, with a row pooling against the previous row's projections unless it is its
        block's first row (then its slot's carry), and this step's projections kept in the scratch for ``commit``."""
        if compressor.ratio == 1:
            kv = compressor._project(x, compressor.wkv, ttnn.bfloat16)
            return ttnn.rms_norm(kv, weight=compressor.norm, epsilon=compressor.eps)
        xf = ttnn.typecast(x, ttnn.float32)
        kv = compressor._project(xf, compressor.wkv, ttnn.float32)
        score = compressor._project(xf, compressor.wgate, ttnn.float32)
        k0, s0 = self.pool.carry[layer]
        first, notfirst = self.inputs["first"], self.inputs["notfirst"]
        k_prev = ttnn.add(ttnn.multiply(k0, first), ttnn.multiply(self._shift_rows(kv), notfirst))
        s_prev = ttnn.add(ttnn.multiply(s0, first), ttnn.multiply(self._shift_rows(score), notfirst))
        m = ttnn.maximum(s_prev, score)
        e0, e1 = ttnn.exp(ttnn.subtract(s_prev, m)), ttnn.exp(ttnn.subtract(score, m))
        pooled = ttnn.divide(ttnn.add(ttnn.multiply(k_prev, e0), ttnn.multiply(kv, e1)), ttnn.add(e0, e1))
        scratch_k, scratch_s = self.scratch[layer]
        ttnn.copy(kv, scratch_k)
        ttnn.copy(score, scratch_s)
        return ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=compressor.norm, epsilon=compressor.eps)

    def _index(self, indexer: TtV41Indexer, x, qr, source: int, mask, candidates: dict | None):
        """The indexer of every live row over its user's keys (the candidate blocks keyed by tile row)."""
        cfg, pool = self.config, self.pool
        cap = pool.slots[source].cap
        pins = DecodePins(self.inputs[f"newest{cfg.compress_ratio(source)}"])
        cos, sin = self.inputs["ropec"]
        q = ttnn.linear(fp8_qdq(qr), indexer.wq_b, compute_kernel_config=indexer.compute_kernel_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=indexer.heads, num_kv_heads=0, transpose_k_heads=False
        )
        q = indexer.qdq(indexer.rope(q, cos, sin))
        w = indexer.ccl.tp_all_reduce(ttnn.linear(x, indexer.weights_proj))
        k = pool.index_k[source]
        published = {} if indexer.is_candidate_source else None
        blocks_of = pool.index_k_blocks.get(source)
        in_blocks = (
            blocks_of is not None and candidates is not None and any(c.ids is not None for c in candidates.values())
        )
        q_rm, w_rm = indexer.decode_query(q, w)
        parts = {}
        dense: list[tuple[int, ttnn.Tensor]] = []  # (tile row, its split-layout score) of the densely scored rows
        for row, u in self.live:
            mine = candidates[row] if candidates is not None else None
            if in_blocks and mine.ids is not None:
                pin_r = ttnn.slice(pins.newest, [0, 0, 0, row], [1, 1, 1, row + 1])
                base = u * (cap + 32) // cfg.CANDIDATE_BLOCK_SIZE
                parts[row] = indexer.select_user_in_blocks(
                    q_rm, w_rm, blocks_of, pin_r, mine.ids, row, cap, base, cfg.INDEX_TOPK
                )
                continue
            dense.append((row, range_scores(q_rm, w_rm, k, pins.newest, u, cap + 32, row=row)))
        n_rows = split_geometry(cap + 32)[0]
        stack_key = (tuple(row for row, _ in dense), n_rows)
        if len(dense) > 1 and stack_key in self._stack_idx:
            # the live rows' selections merged in one chain: the scores stacked along the
            # rows, the pins gathered into one page, one select over all of them, the rows sliced apart
            stacked = ttnn.concat([score for _, score in dense], dim=2)
            idx = self._stack_idx[stack_key]
            wide = pins.newest
            if idx.shape[3] > wide.shape[3]:
                wide = ttnn.concat([wide] * (-(-idx.shape[3] // wide.shape[3])), dim=3)
            newest = _gather_cols(wide, idx)  # [1, 1, 1, L * R]
            mine = candidates[dense[0][0]] if candidates is not None else None  # every row's is None here
            topk_all, blocks_all = indexer.select_split_rows(
                stacked, newest, len(dense), cap + 32, cfg.INDEX_TOPK, mine
            )
            for l, (row, _) in enumerate(dense):
                parts[row] = ttnn.slice(topk_all, [0, 0, l, 0], [1, 1, l + 1, cfg.INDEX_TOPK])
                if published is not None:
                    ids = blocks_all.ids
                    published[row] = CandidateBlocks(
                        None if ids is None else ttnn.slice(ids, [0, 0, l, 0], [1, 1, l + 1, ids.shape[3]])
                    )
        else:
            for row, score in dense:
                mine = candidates[row] if candidates is not None else None
                pin_r = ttnn.slice(pins.newest, [0, 0, 0, row], [1, 1, 1, row + 1])
                topk, blocks = indexer.select_split(score, pin_r, cap + 32, cfg.INDEX_TOPK, mine)
                if published is not None:
                    published[row] = blocks
                parts[row] = topk
        rows = [parts.get(i, self.sentinel_topk) for i in range(ROWS)]
        return ttnn.concat(rows, dim=2), published

    def _index_rows(self, layer: int, compressed):
        window = self._window_rows(layer)
        if compressed is None:
            return window
        src = self.config.kv_source(layer)
        comp = ttnn.typecast(ttnn.to_layout(compressed, ttnn.TILE_LAYOUT), ttnn.int32)
        comp = ttnn.where(ttnn.eq(comp, -1), -1, ttnn.add(comp, self._user_offsets(src)))
        comp = ttnn.to_layout(ttnn.typecast(comp, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        rows = ttnn.concat([window, comp], dim=-1)
        assert rows.shape[3] % TOPK_ALIGN == 0, tuple(rows.shape)
        return rows
