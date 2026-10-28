# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side serving plan of the vLLM adapter (``tt/generator_vllm.py``): the slot geometry of a
``GLM53MeshModel``, the column-aware placement of requests on its fixed user slots, the logical block
table vLLM manages above the device-owned paged latent cache, the prefill plan (traced vs chunked)
and the greedy-only device-sampling admission. Pure Python (no ttnn, no device), so the placement and
translation rules are testable on a CPU (``tests/test_generator_vllm_host.py``).

Vocabulary (design doc ``_agents/vllm_glm53/DESIGN.md``):

* **physical slot** ``(row, user)``: one of the mesh model's ``rows x users_per_row`` fixed decode user
  slots; flat index ``row * users_per_row + user`` is the order of ``GLM53MeshModel.load_ids`` /
  ``set_positions`` / ``ids_to_host`` (row r's users first). With the user-partitioned DSA cache, column
  ``c`` of a row owns users ``[c * B / cols, (c + 1) * B / cols)``: a user's latent pages, pooled index
  and sparse-attention work live on that column's chip.
* **logical slot**: the vLLM plugin's state-slot index in ``[0, max_num_seqs)`` (``empty_slots`` on a
  prefill, ``slot_remap`` on a decode, ``release_request(slot)``). The plugin assumes the model keeps one
  state per logical slot and gathers it on a remap; this adapter never moves device state: it keeps the
  logical -> physical indirection (``SlotTable``) and permutes the host-side token / position vectors.
* **request capacity**: the sequence-parallel prefill (v1) writes a prompt into slot ``user`` of EVERY
  row, so the rows replicate the same ``users_per_row`` prompts and the engine serves ``users_per_row``
  distinct requests (``rows_independent=False``). Per-row prefill ownership (the mesh README's "next
  step") lifts that to ``rows x users_per_row``; the table is written for both.
* **wave** (``rows_independent``, design ``_agents/vllm_glm53/ROW_OWNERSHIP.md``): one prefill call of the
  mesh model in which every mesh row runs one chunk of a DIFFERENT prompt (row-local attention and KDA
  scan, the per-token expert collectives as in decode). ``WavePlanner`` turns the prefill rows of a vLLM
  step into an ordered list of such waves (``WaveStep``) and of sequence-parallel single-prompt prefills
  (``SPStep``: today's path, taken only while it is exact, ``SlotTable.sp_eligible``). Phase 1 of the
  device side needs every member of a wave on the same slot index (``wave_affinity="user"``); phase 2
  lifts that (``"row"``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

# The sparse prefill's prefix read (``TtLatentCache.user_rows_rm``: the user's latent rows [0, start + T)) rounds its
# row count up to a multiple of this (0 = exact): the kv shape of a continued chunk then depends on the prefix bucket,
# not on the exact start, and the served warm-up builds every bucket's programs (``_warmup_prefix_starts``). The
# served adapter sets it to max(chunk, max_seq_len / 8 rounded to the chunk) before the caches are built.
PREFIX_BUCKET_ENV = "GLM53_PREFIX_BUCKET"

TILE = 32
DENSE_PREFILL_ROWS = 2048  # the traced per-length prefill covers prompts up to this many tokens
PREFILL_BUCKETS_ENV = "GLM53_PREFILL_BUCKETS"
# The eager prefill's padded chunk lengths (``eager_pad_length``; re-exported by tt/mesh_model.py). 8192 is deliberately
# absent: a 1024-row slab per chip (8192 tokens on the 8 rows) puts the KDA sequence-parallel scan's static dataflow
# buffers (~1005 KiB per core, slab-independent) and the prefill's L1-resident chunk terms (~608 KiB per core at 1024
# rows) over the 1.5 MiB L1 ("Statically allocated dataflow buffers ... clash with L1 buffers"); slabs <= 512 rows fit
# and slabs >= ~1216 rows spill the terms to DRAM (the 16384 chunk path). 256..2048 are the traced lengths (their
# per-length KDA prefill layers exist from the captures) and 16384 the serving chunk, so these buckets cost no DRAM; a
# 4096 bucket added ~1 GB of per-length buffers at boot and left the 1 M x 32-slot model unable to allocate the
# indexer's chunk transient (smoke r9, 2026-10-27).
DEFAULT_PREFILL_BUCKETS = (256, 512, 1024, 2048, 16384)
DEFAULT_DEVICE_PREFILL_CHUNK = 16384  # tokens per eager prefill chunk on the 8x4 (32 K exceeds the KDA owner cores)
DEFAULT_BLOCK_SIZE = 32  # the latent page (TtLatentCache.block_size); the vLLM block size must match it
DEFAULT_WAVE_ROWS = 2048  # rows per mesh row per wave (= today's 16 K chunk on 8 rows: memory-neutral)
WAVE_AFFINITIES = ("user", "row")


# ----------------------------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class ServingGeometry:
    """The fixed-slot geometry of one engine: a ``(rows, cols)`` mesh model with ``users_per_row`` user
    slots per row holding ``max_seq_len`` positions each."""

    rows: int
    cols: int
    users_per_row: int
    max_seq_len: int
    partitioned: bool = True  # user-partitioned DSA cache (column c owns users [c*B/cols, (c+1)*B/cols))
    rows_independent: bool = False  # per-row prefill ownership (waves); False = SP prefill v1 fills slot u of every row
    block_size: int = DEFAULT_BLOCK_SIZE
    device_prefill_chunk: int = DEFAULT_DEVICE_PREFILL_CHUNK  # the SP prefill's device chunk (tokens of one prompt)
    traced_prefill_lengths: tuple[int, ...] = ()
    pad_token_id: int = 0
    # waves (rows_independent): rows per mesh row per wave (every member's chunk is walked in these), which members
    # may share a wave ("user": the same slot index on distinct rows, phase 1; "row": distinct rows, phase 2) and
    # whether a request still takes the sequence-parallel path while that is exact (``SlotTable.sp_eligible``)
    wave_rows: int = DEFAULT_WAVE_ROWS
    wave_affinity: str = "user"
    sp_fallback: bool = True

    def __post_init__(self):
        if self.rows < 1 or self.cols < 1 or self.users_per_row < 1:
            raise ValueError(f"bad mesh geometry rows={self.rows} cols={self.cols} users_per_row={self.users_per_row}")
        if self.partitioned and self.cols > 1 and self.users_per_row % self.cols:
            raise ValueError(
                f"the user-partitioned cache needs users_per_row ({self.users_per_row}) % cols ({self.cols}) == 0"
            )
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len {self.max_seq_len} < 1")
        if self.block_size % TILE:
            raise ValueError(f"block_size {self.block_size} must be a multiple of {TILE} (the latent page is 32 rows)")
        unit = TILE * self.rows
        if self.device_prefill_chunk < unit or self.device_prefill_chunk % unit:
            raise ValueError(f"device_prefill_chunk {self.device_prefill_chunk} must be a positive multiple of {unit}")
        if self.wave_affinity not in WAVE_AFFINITIES:
            raise ValueError(f"wave_affinity {self.wave_affinity!r}: expected one of {WAVE_AFFINITIES}")
        # a wave chunk below the dense range would put rows with dense prefixes of different lengths in one wave
        # (not SPMD); at >= 2048 rows only a prompt's first chunk (start 0, no prefix) has dense rows
        if self.wave_rows < DENSE_PREFILL_ROWS or self.wave_rows % TILE:
            raise ValueError(f"wave_rows {self.wave_rows} must be a multiple of {TILE} and >= {DENSE_PREFILL_ROWS}")

    @property
    def users_per_chip(self) -> int:
        """Slots a column chip owns per row (the memory planner's ``slots_per_chip`` without the scratch slot)."""
        return self.users_per_row // self.cols if (self.partitioned and self.cols > 1) else self.users_per_row

    @property
    def physical_slots(self) -> int:
        return self.rows * self.users_per_row

    @property
    def request_capacity(self) -> int:
        """Distinct requests the engine serves at once: the plugin's ``max_num_seqs`` must equal it."""
        return self.physical_slots if self.rows_independent else self.users_per_row

    @property
    def blocks_per_seq(self) -> int:
        return -(-self.max_seq_len // self.block_size)

    @property
    def chunk_unit(self) -> int:
        """Prefill chunk alignment: a chunk start must be a multiple of 32 (the FunctionalDecoder contract) and,
        under the sequence-parallel prefill, every chunk but the last a multiple of ``32 * rows`` (tile-aligned
        row slabs); a wave's per-row chunks need the 32-row alignment only."""
        return TILE if self.rows_independent else TILE * self.rows

    def wave_user_order(self) -> list[int]:
        """The slot indices in the order the wave-affine placement fills them (``wave_affinity="user"``): index
        ``k`` of column 0, index ``k`` of column 1, ... so consecutive slot indices sit on different columns
        (decode balance) while every index is filled across all the rows before the next (one wave per burst)."""
        if self.partitioned and self.cols > 1:
            per_col = [list(self.users_of_column(c)) for c in range(self.cols)]
        else:
            per_col = [list(range(self.users_per_row))]
        return [col[k] for k in range(len(per_col[0])) for col in per_col]

    def column_of(self, user: int) -> int:
        if not (0 <= user < self.users_per_row):
            raise ValueError(f"user {user} outside [0, {self.users_per_row})")
        return user // self.users_per_chip if (self.partitioned and self.cols > 1) else 0

    def users_of_column(self, col: int) -> range:
        if not (self.partitioned and self.cols > 1):
            return range(self.users_per_row)
        return range(col * self.users_per_chip, (col + 1) * self.users_per_chip)

    def flat(self, row: int, user: int) -> int:
        return row * self.users_per_row + user

    def check_max_num_seqs(self, max_num_seqs: int) -> None:
        if int(max_num_seqs) != self.request_capacity:
            raise ValueError(
                f"max_num_seqs={max_num_seqs} must equal the engine's request capacity {self.request_capacity} "
                f"({self.rows}x{self.cols} mesh, {self.users_per_row} users per row, rows_independent="
                f"{self.rows_independent}): the mesh model decodes fixed user slots, so the plugin's slot count and "
                "the model's slot count must agree"
            )

    def check_memory_plan(self, plan) -> None:
        """``plan`` is a ``tt/memory_plan.py`` ``MemoryPlan``: the slots must fit the per-chip DRAM at ``max_seq_len``."""
        if not plan.fits(self.users_per_row, self.max_seq_len):
            raise ValueError(
                f"{self.users_per_row} users per row x {self.max_seq_len} positions do not fit the per-chip DRAM "
                f"(planner: at most {plan.max_users_per_row(self.max_seq_len)} users per row at this context, or "
                f"{plan.max_isl(self.users_per_row)} positions for {self.users_per_row} users per row)"
            )


# -------------------------------------------------------------------------------------------- slot table
@dataclass
class SlotState:
    row: int
    user: int
    position: int = 0  # the position the next decode token of this slot is consumed at
    phase: str = "free"  # free | prefill | decode
    prefilled_to: int = 0  # tokens of the prompt written so far (chunked prefill bookkeeping)
    # every chunk so far went through the sequence-parallel path (its replicas sit in the other rows' slot ``user``,
    # so a later SP chunk finds the prefix there); a wave chunk exists on the owner row only and clears it for good
    sp_only: bool = True


class SlotTable:
    """Logical (plugin) slot -> physical ``(row, user)`` placement, column-aware.

    Placement rule (``allocate``): among the columns with a free user slot pick the one with the fewest
    live users (ties -> lowest column), then the lowest free user of that column (and, with independent
    rows, the row with the fewest live users). Every column chip of a row serves its own users' sparse
    attention and holds their caches, so spreading requests over the columns balances per-chip decode
    work and keeps the next prompt's prefill scratch next to a column that has room.
    Independent rows with ``wave_affinity="user"`` (phase 1 of the device side: every member of a wave
    shares the slot index) fill one slot index across all the rows before the next
    (``ServingGeometry.wave_user_order``), so a burst of up to ``rows`` arrivals forms one wave.

    A remap (``apply_remap``: ``row i reads the state at slot remap[i]``) permutes the table instead of
    moving device state; ``apply_moves`` is the same permutation in the plugin's ``old -> new`` dictionary
    form (``note_state_slots_moved``). Either is applied exactly once per accepted decode.
    """

    def __init__(self, geometry: ServingGeometry):
        self.g = geometry
        self.n = geometry.request_capacity
        self._phys: list[SlotState | None] = [None] * self.n
        self._occupied: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------ queries
    def physical(self, logical: int) -> SlotState | None:
        self._check_logical(logical)
        return self._phys[logical]

    def live(self) -> list[int]:
        return [i for i, s in enumerate(self._phys) if s is not None]

    def column_loads(self) -> list[int]:
        loads = [0] * self.g.cols
        for s in self._phys:
            if s is not None:
                loads[self.g.column_of(s.user)] += 1
        return loads

    def row_loads(self) -> list[int]:
        loads = [0] * self.g.rows
        for s in self._phys:
            if s is not None:
                loads[s.row] += 1
        return loads

    def flat_indices(self, logical: int) -> list[int]:
        """Every flat device index holding this request's state: one per row when the rows replicate the
        prompt (the SP prefill writes slot ``user`` of every row), the one ``(row, user)`` otherwise."""
        s = self._require(logical)
        if self.g.rows_independent:
            return [self.g.flat(s.row, s.user)]
        return [self.g.flat(r, s.user) for r in range(self.g.rows)]

    def read_index(self, logical: int) -> int:
        """The flat device index the request's outputs are read from (row 0's copy when the rows replicate)."""
        s = self._require(logical)
        return self.g.flat(s.row if self.g.rows_independent else 0, s.user)

    def in_prefill(self) -> list[int]:
        return [i for i, s in enumerate(self._phys) if s is not None and s.phase == "prefill"]

    # ------------------------------------------------------------------ placement
    def allocate(self, logical: int, *, at: tuple[int, int] | None = None, slot_rank=None) -> SlotState:
        """Give ``logical`` a physical slot for a new prompt. A logical slot the plugin hands out again
        (``empty_slots`` of a prefill) is released first: the plugin reuses a slot only after its previous
        request finished, and a resumed request re-prefills from position 0.
        ``at``: the exact free ``(row, user)`` to take (a prefix-cache hit continues the slot holding the prefix:
        ``SlotPrefixCache``); ``slot_rank(row, user) -> int``: a tie-break after the column / row balance (0 = a slot
        without retained content, higher = more recently used retained content, so an unused slot is consumed before a
        resident prefix is dropped)."""
        self._check_logical(logical)
        if self._phys[logical] is not None:
            self.release(logical)
        col_loads, row_loads = self.column_loads(), self.row_loads()
        best = None  # (key, row, user)
        rank = slot_rank or (lambda row, user: 0)
        if at is not None:
            row, user = int(at[0]), int(at[1])
            if (row, user) in self._occupied:
                raise RuntimeError(f"physical slot {(row, user)} is live; a prefix-cache hit cannot take it")
            if not (0 <= row < self.g.rows and 0 <= user < self.g.users_per_row):
                raise ValueError(f"physical slot {(row, user)} outside the {self.g.rows}x{self.g.users_per_row} slots")
            best = (None, row, user)
        elif self.g.rows_independent and self.g.wave_affinity == "user":
            # wave-affine: the first slot index (column-interleaved order) with a free row, its lowest free row
            for user in self.g.wave_user_order():
                for row in range(self.g.rows):
                    if (row, user) not in self._occupied:
                        best = (None, row, user)
                        break
                if best is not None:
                    break
        else:
            rows = range(self.g.rows) if self.g.rows_independent else (0,)
            # column balance first, then row balance, then the lowest indices
            for col in range(self.g.cols):
                for row in rows:
                    for user in self.g.users_of_column(col):
                        if (row, user) not in self._occupied:
                            key = (col_loads[col], row_loads[row], rank(row, user), col, row, user)
                            if best is None or key < best[0]:
                                best = (key, row, user)
        if best is None:
            raise RuntimeError(
                f"no free physical slot for a new request: {len(self.live())} of {self.n} logical slots live, "
                f"column loads {col_loads}; admission is the scheduler's job (max_num_seqs must equal the capacity)"
            )
        _, row, user = best
        state = SlotState(row=row, user=user, phase="prefill")
        self._phys[logical] = state
        self._occupied.add((row, user))
        return state

    def sp_eligible(self, logical: int) -> bool:
        """Whether the next prefill chunk of ``logical`` may take the sequence-parallel path exactly: always with
        replicated rows (the only path there); with independent rows only when the policy allows it, every chunk
        so far was SP (the prefix exists on every row) and slot ``user`` holds no live request on any other row
        (the SP chunk writes replicas into those slots; a free slot's contents are garbage by contract)."""
        s = self._require(logical)
        if not self.g.rows_independent:
            return True
        if not self.g.sp_fallback or not s.sp_only:
            return False
        return not any((r, s.user) in self._occupied for r in range(self.g.rows) if r != s.row)

    def release(self, logical: int) -> None:
        self._check_logical(logical)
        s = self._phys[logical]
        if s is None:
            return
        self._occupied.discard((s.row, s.user))
        self._phys[logical] = None

    # ------------------------------------------------------------------ remaps
    def apply_remap(self, remap: Sequence[int]) -> dict[int, int]:
        """``remap[i] = j``: row/slot ``i`` takes the state of slot ``j`` (an absolute permutation of all
        ``n`` slots). Returns the plugin-form ``moves`` (``old -> new`` for every moved slot)."""
        remap = [int(v) for v in remap]
        if len(remap) != self.n or sorted(remap) != list(range(self.n)):
            raise ValueError(f"slot_remap {remap} is not a permutation of {self.n} slots")
        old = list(self._phys)
        self._phys = [old[j] for j in remap]
        return {j: i for i, j in enumerate(remap) if i != j}

    def apply_moves(self, moves: dict[int, int]) -> None:
        """The same permutation as ``apply_remap`` in ``old -> new`` form (``note_state_slots_moved``)."""
        if not moves:
            return
        remap = list(range(self.n))
        for old_slot, new_slot in moves.items():
            remap[int(new_slot)] = int(old_slot)
        self.apply_remap(remap)

    # ------------------------------------------------------------------ device vectors
    def device_feed(
        self, tokens: dict[int, int], positions: dict[int, int], park_position: int = 0, park: dict | None = None
    ):
        """Flat ``[rows * users_per_row]`` token and position lists for ``load_ids`` / ``set_positions``:
        live logical slots at their physical indices (every row's copy when the rows replicate), idle
        physical slots parked at ``(pad_token_id, park_position)``. ``tokens`` / ``positions`` are keyed by
        logical slot and must agree. ``park``: ``{(row, user): position}`` of idle slots that keep resident
        prefix content (``SlotPrefixCache``): parked at their own end, so the idle decode step writes its
        latent row / pool past the content instead of over row 0 (every replica row when the rows replicate)."""
        if set(tokens) != set(positions):
            raise ValueError(f"tokens for slots {sorted(tokens)} but positions for {sorted(positions)}")
        ids = [self.g.pad_token_id] * self.g.physical_slots
        pos = [park_position] * self.g.physical_slots
        for (row, user), p in (park or {}).items():
            if not (0 <= p < self.g.max_seq_len):
                raise ValueError(f"park position {p} of slot {(row, user)} outside [0, {self.g.max_seq_len})")
            for r in [row] if self.g.rows_independent else range(self.g.rows):
                pos[self.g.flat(r, user)] = int(p)
        for logical, tok in tokens.items():
            p = positions[logical]
            if not (0 <= p < self.g.max_seq_len):
                raise ValueError(f"slot {logical}: position {p} outside [0, {self.g.max_seq_len})")
            for flat in self.flat_indices(logical):
                ids[flat] = int(tok)
                pos[flat] = int(p)
        return ids, pos

    # ------------------------------------------------------------------ helpers
    def _check_logical(self, logical: int) -> None:
        if not (0 <= int(logical) < self.n):
            raise ValueError(f"logical slot {logical} outside [0, {self.n})")

    def _require(self, logical: int) -> SlotState:
        s = self.physical(logical)
        if s is None:
            raise RuntimeError(f"logical slot {logical} holds no request (never prefilled or already released)")
        return s


# ------------------------------------------------------------------------------------- block tables
class LogicalBlockTable:
    """vLLM's block ids above the device-managed per-slot latent pages.

    The device cache is allocated per user slot at build time (``TtLatentCache``: an identity page table
    of ``blocks_per_seq`` pages per slot, the pooled-key index contiguous per slot, both per DSA layer and
    local to the owner column). vLLM's block tables therefore do not address device memory: the KV pool
    the plugin sizes (``get_max_tokens_all_users`` = ``max_num_seqs x max_model_len``) only accounts for
    admission, and a row's ids are checked for shape and uniqueness here, never uploaded.
    """

    def __init__(self, geometry: ServingGeometry):
        self.g = geometry

    def blocks_needed(self, end_position: int) -> int:
        """Pages a request holding positions ``[0, end_position)`` needs."""
        return -(-int(end_position) // self.g.block_size)

    def check_row(self, table_row, end_position: int) -> int:
        """``table_row``: one request's block ids (any sequence / 1-D tensor). Returns the number of pages the
        request needs; raises when the table is too narrow for ``end_position`` or the needed prefix repeats
        a block id (a vLLM bookkeeping fault the device would silently tolerate)."""
        if end_position < 0 or end_position > self.g.max_seq_len:
            raise ValueError(f"end position {end_position} outside [0, {self.g.max_seq_len}]")
        need = self.blocks_needed(end_position)
        # tensor ops on the needed prefix only: iterating a 1 M-context row ([32768] ids) in Python cost ~50 ms per
        # live user per decode step (the served step's largest host share, 2026-10-27 profile)
        row = torch.as_tensor(table_row).reshape(-1)
        if row.numel() < need:
            raise ValueError(
                f"block table row of {row.numel()} entries cannot hold {end_position} positions ({need} pages)"
            )
        prefix = row[:need]
        if need and bool((prefix < 0).any()):
            raise ValueError(f"block table prefix {prefix.tolist()} has a negative block id")
        if need and int(torch.unique(prefix).numel()) != need:
            raise ValueError(f"block table prefix {prefix.tolist()} repeats a block id")
        return need


# ------------------------------------------------------------------------------------- prefill plan
@dataclass(frozen=True)
class PrefillPlan:
    kind: str  # "traced" | "eager"
    chunks: tuple[tuple[int, int], ...]  # (start, end) of every device prefill call, in order
    padded_length: int | None = None  # the traced length the prompt is padded to (traced only)

    @property
    def start(self) -> int:
        return self.chunks[0][0]

    @property
    def end(self) -> int:
        return self.chunks[-1][1]


class PrefillPlanner:
    """Which device prefill a vLLM prefill call takes.

    * ``start == 0``, ``end - start`` within a captured traced length and tracing on: one replay of the
      per-length traced prefill (``GLM53MeshModel.prefill_traced``), the prompt padded to that length.
    * otherwise the eager sequence-parallel prefill (``GLM53MeshModel.prefill``) in device chunks of
      ``device_prefill_chunk`` tokens from ``start``: every chunk but the last a multiple of ``32 * rows``.

    vLLM's chunk boundaries (``prefill_chunk_alignment``) must keep ``start % 32 == 0`` and chunks must be
    contiguous per request (the FunctionalDecoder's ``next_start`` contract); the last chunk of a prompt may
    end anywhere.
    """

    def __init__(self, geometry: ServingGeometry, traced_available: Callable[[int], bool] | None = None):
        self.g = geometry
        self._traced = traced_available or (lambda n: n in geometry.traced_prefill_lengths)

    def traced_length(self, n_tokens: int) -> int | None:
        fits = [T for T in self.g.traced_prefill_lengths if T >= n_tokens]
        return min(fits) if fits else None

    def captured_length(self, n_tokens: int) -> int | None:
        """``traced_length`` when that trace is captured, else None."""
        T = self.traced_length(n_tokens)
        return T if T is not None and self._traced(T) else None

    @staticmethod
    def walk(start: int, end: int, chunk: int, *, split_tail: bool = False) -> tuple[tuple[int, int], ...]:
        """``[start, end)`` in consecutive device chunks of at most ``chunk`` tokens. ``split_tail``: when ``end``
        is not a 32-multiple, the last chunk ends on the prompt's last 32-boundary and the remaining < 32 tokens form
        one more chunk, so the KDA state at that boundary can be snapshotted (``SlotPrefixCache``: a multi-turn
        request whose response never crossed a 32-boundary hits at exactly that position)."""
        out = []
        s0 = start
        while s0 < end:
            e = min(end, s0 + chunk)
            out.append((s0, e))
            s0 = e
        if split_tail and end % TILE and out:
            s_last, _ = out[-1]
            cut = end // TILE * TILE
            if cut > s_last:
                out[-1] = (s_last, cut)
                out.append((cut, end))
        return tuple(out)

    def check_range(self, start: int, end: int) -> tuple[int, int]:
        start, end = int(start), int(end)
        if start < 0 or start % TILE:
            raise ValueError(f"prefill chunk start {start} must be a non-negative multiple of {TILE}")
        if end <= start:
            raise ValueError(f"prefill chunk [{start}, {end}) is empty")
        if end > self.g.max_seq_len:
            raise ValueError(f"prefill chunk end {end} exceeds max_seq_len {self.g.max_seq_len}")
        return start, end

    def plan(self, start: int, end: int, *, enable_trace: bool = True, split_tail: bool = False) -> PrefillPlan:
        start, end = self.check_range(start, end)
        n = end - start
        if start == 0 and enable_trace:
            T = self.captured_length(n)
            if T is not None:
                return PrefillPlan("traced", ((0, end),), T)
        return PrefillPlan("eager", self.walk(start, end, self.g.device_prefill_chunk, split_tail=split_tail))


# ------------------------------------------------------------------------------------------- waves
@dataclass(frozen=True)
class WaveMember:
    """One request's chunk inside a wave: its logical slot, its physical ``(row, user)`` and ``[start, end)``."""

    logical: int
    row: int
    user: int
    start: int
    end: int

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class WaveCall:
    """What the mesh model's ``prefill_wave`` / ``prefill_traced_wave`` receive: at most one member per mesh row
    (rows without a member are idle: pad tokens into dummy targets on the device), every member's chunk padded to
    ``t_pad`` rows (its valid end is ``end - start``), the uniform slot index ``user`` (phase 1) or None (phase 2:
    each member's own), ``traced_length`` the captured per-row length that fits every member (all at ``start == 0``)
    or None (eager). ``ids`` maps a member's row to its chunk's token ids (filled by the adapter; the planner leaves
    it empty). The model returns ``{row: (first greedy token, last-token logits or None)}`` per member."""

    rows: int
    t_pad: int
    members: tuple[WaveMember, ...]
    user: int | None = None
    traced_length: int | None = None
    ids: dict = field(default_factory=dict)

    def __post_init__(self):
        rows_used = [m.row for m in self.members]
        if not self.members or len(set(rows_used)) != len(rows_used):
            raise ValueError(f"a wave needs at least one member and at most one per row (rows {rows_used})")
        if any(m.n_tokens > self.t_pad for m in self.members):
            raise ValueError(f"a member's chunk exceeds t_pad {self.t_pad}")
        if self.user is not None and any(m.user != self.user for m in self.members):
            raise ValueError(f"wave slot index {self.user} differs from a member's")

    @property
    def idle_rows(self) -> list[int]:
        return [r for r in range(self.rows) if r not in {m.row for m in self.members}]

    def member_of_row(self, row: int) -> WaveMember | None:
        for m in self.members:
            if m.row == row:
                return m
        return None


@dataclass(frozen=True)
class SPStep:
    """One request through today's sequence-parallel prefill (``PrefillPlan``): exact only while the request is
    ``SlotTable.sp_eligible``."""

    logical: int
    plan: PrefillPlan


@dataclass(frozen=True)
class WaveStep:
    call: WaveCall


class WavePlanner:
    """The prefill rows of one vLLM step -> ordered device steps under per-row prefill ownership.

    ``plan(pending)`` with ``pending = [(logical, start, end), ...]`` (every logical slot already placed in the
    table, chunks contiguous with the slot's ``prefilled_to``): requests that may still take the sequence-parallel
    path (``SlotTable.sp_eligible``) become ``SPStep``s in order; the rest are grouped into waves, every member's
    range walked in ``wave_rows`` device chunks across consecutive waves (a member drops out when its range is
    done). ``wave_affinity="user"``: one group per slot index (the rows are distinct by construction: one request
    per ``(row, user)``); ``"row"``: one group, every row serving its pending requests one after the other (the
    k-th wave takes the head of every row's queue). A wave whose members all start at 0 and fit a captured
    per-row traced length is marked for the traced replay.
    """

    def __init__(self, geometry: ServingGeometry, slots: SlotTable, prefill_planner: PrefillPlanner):
        self.g = geometry
        self.slots = slots
        self.pp = prefill_planner

    def plan(self, pending: Sequence[tuple[int, int, int]], *, enable_trace: bool = True) -> list:
        items = []
        for logical, start, end in pending:
            start, end = self.pp.check_range(start, end)
            s = self.slots.physical(int(logical))
            if s is None:
                raise RuntimeError(f"logical slot {logical} holds no request: allocate it before planning its chunk")
            items.append((int(logical), s, start, end))
        if len({i[0] for i in items}) != len(items):
            raise ValueError("a request appears twice in one prefill step")
        steps: list = []
        wave_items = []
        for logical, s, start, end in items:
            if self.slots.sp_eligible(logical):
                steps.append(SPStep(logical, self.pp.plan(start, end, enable_trace=enable_trace)))
            else:
                wave_items.append((logical, s, start, end))
        if not wave_items:
            return steps
        if self.g.wave_affinity == "user":
            groups: dict[int, list] = {}
            for it in wave_items:
                groups.setdefault(it[1].user, []).append(it)
            for user, group in groups.items():
                steps.extend(self._waves(group, user, enable_trace))
        else:
            steps.extend(self._waves(wave_items, None, enable_trace))
        return steps

    def _waves(self, group: list, user: int | None, enable_trace: bool) -> list:
        """Waves of one group: per row a queue of (logical, user, remaining chunks); every wave takes the next
        chunk of every row's current request."""
        queues: dict[int, list] = {}
        for logical, s, start, end in group:
            chunks = list(self.pp.walk(start, end, self.g.wave_rows))
            queues.setdefault(s.row, []).append([logical, s.user, chunks])
        out = []
        while any(queues.values()):
            members = []
            for row in sorted(queues):
                q = queues[row]
                while q and not q[0][2]:
                    q.pop(0)
                if not q:
                    continue
                logical, u, chunks = q[0]
                c0, c1 = chunks.pop(0)
                members.append(WaveMember(logical=logical, row=row, user=u, start=c0, end=c1))
            if not members:
                break
            n = max(m.n_tokens for m in members)
            t_pad = -(-n // TILE) * TILE
            traced = None
            if enable_trace and all(m.start == 0 for m in members):
                traced = self.pp.captured_length(n)
            out.append(
                WaveStep(
                    WaveCall(rows=self.g.rows, t_pad=t_pad, members=tuple(members), user=user, traced_length=traced)
                )
            )
        return out


# ---------------------------------------------------------------------------------- sampling / sizing
def prefill_buckets(env=None) -> tuple[int, ...]:
    """``GLM53_PREFILL_BUCKETS``: the eager prefill's padded chunk lengths (default 256,512,1024,2048,16384; ``0`` =
    pad to the row unit only, the pre-2026-10-27 behaviour)."""
    env = os.environ if env is None else env
    raw = env.get(PREFILL_BUCKETS_ENV, "")
    if raw.strip() == "0":
        return ()
    if not raw.strip():
        return DEFAULT_PREFILL_BUCKETS
    return tuple(sorted(int(v) for v in raw.split(",") if v.strip()))


def eager_pad_length(seq_len: int, unit: int, buckets: tuple[int, ...] | None = None) -> int:
    """Padded length of an eager prefill chunk of ``seq_len`` tokens: the row unit's ceiling, raised to the smallest
    bucket that holds it (a bucket must be a multiple of ``unit``; above the largest bucket the ceiling stands).
    Every bucket is warmed at boot (the serving warm-up ladder, ``GLM53FlashForCausalLM.warmup_model_prefill``), so
    a prompt's odd tail chunk never builds a new per-length ttKDA prefill layer or compiles its programs after the
    trace captures (a program's kernel binaries and any buffer allocated then are clobbered by replays: the
    2026-10-27 served hang) and never compiles at request time. The kernels clip to the valid ``seq_len``."""
    T = -(-int(seq_len) // unit) * unit
    for b in prefill_buckets() if buckets is None else buckets:
        if b % unit:
            raise ValueError(f"prefill bucket {b} is not a multiple of the row unit {unit}")
        if T <= b:
            return b
    return T


def greedy_rows(sampling_params, rows: Sequence[int]) -> None:
    """Device sampling is the head's argmax: every sampled row must be greedy (temperature 0, or top_k 1).
    ``sampling_params`` carries per-row lists (``TTSamplingParams`` on device-sampling calls)."""
    temperature = getattr(sampling_params, "temperature", None)
    top_k = getattr(sampling_params, "top_k", None)
    for r in rows:
        t = float(temperature[r]) if temperature is not None and r < len(temperature) else 0.0
        k = int(top_k[r]) if top_k is not None and r < len(top_k) else 1
        if t != 0.0 and k != 1:
            raise ValueError(
                f"row {r}: temperature {t} top_k {k} cannot be sampled on device (greedy argmax only; the plugin "
                "routes such rows to host sampling when max_device_top_k=1 is declared)"
            )


def max_tokens_all_users(max_num_seqs: int, max_model_len: int) -> int:
    """The all-user KV budget vLLM accounts for: every slot owns a full ``max_model_len`` of latent pages."""
    return int(max_num_seqs) * int(max_model_len)


@dataclass
class DecodeRowMap:
    """One decode submission's row -> device index map (what ``read_decode_output`` needs to un-permute)."""

    wire_rows: int
    read_index: list[int] = field(default_factory=list)  # per wire row: the flat device index or -1 (padding)

    def live_rows(self) -> list[int]:
        return [i for i, v in enumerate(self.read_index) if v >= 0]


# ------------------------------------------------------------------------------------ prefix cache
SNAPSHOT_KINDS = ("decode", "prefill")
APC_ENV = (
    "GLM53_VLLM_PREFIX_CACHING"  # "1": declare supports_prefix_caching (slot-level APC, doc/serving/prefix_caching.md)
)
APC_SNAPSHOTS_ENV = "GLM53_VLLM_APC_SNAPSHOTS"  # KDA snapshot entries per user slot (0 = planner's fit, capped)
APC_SNAPSHOT_STORE_ENV = "GLM53_VLLM_APC_SNAPSHOT_STORE"  # device (DRAM ring) | host (torch copies)
APC_TAIL_SPLIT_ENV = "GLM53_VLLM_APC_TAIL_SPLIT"  # "1": eager prefills snapshot the prompt's last 32-boundary
APC_MAX_SNAPSHOTS = 8
# "1": the served adapter records the DRAM footprint after the decode capture and refuses the first APC call site
# (snapshot save, hit restore, prefill row, release) after which it differs (doc/serving/prefix_caching.md s.6)
APC_ALLOC_CHECK_ENV = "GLM53_VLLM_APC_ALLOC_CHECK"


@dataclass(frozen=True)
class MMSpan:
    """One multimodal item's placeholder run in a prompt: ``[offset, offset + length)`` of ``kind`` (image / video /
    audio) and vLLM's content hash ``identifier`` when the plugin forwards it (``mm_identifiers``; the proposal in
    doc/serving/prefix_caching.md), else None. Two spans match only with equal, known identifiers: the token ids of
    an image run are all ``image_token_id``, so the ids alone cannot tell two images apart."""

    offset: int
    length: int
    kind: str = "image"
    identifier: str | None = None

    @property
    def end(self) -> int:
        return self.offset + self.length

    def intersects(self, n: int) -> bool:
        """Whether any placeholder of the span lies in ``[0, n)``."""
        return self.offset < n and self.length > 0

    def same_content(self, other: "MMSpan") -> bool:
        return (
            self.offset == other.offset
            and self.length == other.length
            and self.kind == other.kind
            and self.identifier is not None
            and self.identifier == other.identifier
        )


@dataclass(frozen=True)
class PrefixHit:
    """A served prefix-cache hit: continue in physical slot ``(row, user)`` from ``position`` (the KDA snapshot
    ``entry`` restored there); the request's tokens ``[position, end)`` are prefilled."""

    row: int
    user: int
    position: int
    entry: int


class SlotContent:
    """What one physical slot ``(row, user)`` holds on device: the token ids it consumed (prefilled or decoded),
    the multimodal spans among them, the positions at which a KDA state snapshot exists (``snapshots``: position
    -> ring entry), whether a request is live in it, and when it was last used (the retention LRU). The latent
    pages / pooled index are valid for ``[0, length)``; the live KDA state is at ``length`` only while the slot is
    live or parked (the idle decode step advances every slot's KDA state, which is why the snapshots exist)."""

    def __init__(self, row: int, user: int, max_seq_len: int):
        self.row, self.user = row, user
        self.tokens = torch.zeros(max_seq_len, dtype=torch.int32)
        self.length = 0
        self.mm: list[MMSpan] = []
        self.snapshots: dict[int, int] = {}
        self.live = False
        self.last_used = 0

    @property
    def key(self) -> tuple[int, int]:
        return (self.row, self.user)

    @property
    def retained(self) -> bool:
        return not self.live and self.length > 0

    def clear(self) -> None:
        self.length = 0
        self.mm = []
        self.snapshots = {}

    def truncate(self, n: int) -> None:
        """Keep ``[0, n)``: the content a prefix-cache hit continues from (the snapshot at the hit position stays;
        later snapshots and the spans past ``n`` are dropped; a span straddling ``n`` is dropped too: the resumed
        prefill scatters its features again)."""
        n = int(n)
        if n >= self.length:
            return
        self.length = n
        self.mm = [sp for sp in self.mm if sp.end <= n]
        self.snapshots = {pos: e for pos, e in self.snapshots.items() if pos <= n}

    def write(self, start: int, ids: torch.Tensor) -> None:
        ids = torch.as_tensor(ids).reshape(-1).to(torch.int32)
        n = int(ids.numel())
        if n == 0:
            return
        if start + n > self.tokens.numel():
            raise ValueError(f"slot {self.key}: content [{start}, {start + n}) exceeds {self.tokens.numel()} positions")
        self.tokens[start : start + n] = ids
        self.length = max(self.length, start + n) if start <= self.length else start + n

    def matches(self, ids: torch.Tensor, spans: Sequence[MMSpan], n: int) -> bool:
        """Whether this slot's content equals ``ids[:n]`` with compatible multimodal spans inside ``[0, n)``."""
        n = int(n)
        if n <= 0 or self.length < n:
            return False
        req = torch.as_tensor(ids).reshape(-1)[:n].to(torch.int32)
        if int(req.numel()) < n or not torch.equal(self.tokens[:n], req):
            return False
        mine = [sp for sp in self.mm if sp.intersects(n)]
        theirs = [sp for sp in spans if sp.intersects(n)]
        if len(mine) != len(theirs):
            return False
        return all(
            a.same_content(b)
            for a, b in zip(sorted(mine, key=lambda s: s.offset), sorted(theirs, key=lambda s: s.offset))
        )

    def best_snapshot(self, n: int) -> tuple[int, int] | None:
        """``(position, entry)`` of the highest snapshot at or below ``n`` (positive positions only)."""
        best = None
        for pos, e in self.snapshots.items():
            if 0 < pos <= n and (best is None or pos > best[0]):
                best = (pos, e)
        return best


class SlotPrefixCache:
    """Slot-level automatic prefix caching (doc/serving/prefix_caching.md): the host record of every physical
    slot's resident content and KDA snapshots, the hit decision for a request vLLM admitted with
    ``num_computed_tokens = n > 0``, the retention LRU, and the snapshot-ring policy.

    Ring policy (``entries`` per user slot, ``plan_snapshot``): entry 0 rolls with the decode (overwritten at every
    32-aligned crossing: the multi-turn hit lands exactly there), entries 1.. hold prefill-time positions (device
    chunk ends, the tail split), evicting the lowest position when full; with one entry everything shares it.
    A hit continues from the highest snapshot at or below ``n`` (``PrefixHit.position``); the tokens between it
    and ``n`` are prefilled again (their latent rows are rewritten; the result is what a prefill from 0 gives).
    """

    def __init__(self, geometry: ServingGeometry, entries: int):
        self.g = geometry
        self.entries = int(entries)
        if self.entries < 0:
            raise ValueError(f"snapshot entries {entries} < 0")
        self._content: dict[tuple[int, int], SlotContent] = {}
        self._tick = 0
        self.stats = {"hits": 0, "downgraded": 0, "refused_mm": 0, "snapshots": 0, "retained_evicted": 0}

    # ------------------------------------------------------------------ records
    def content(self, row: int, user: int) -> SlotContent:
        key = (int(row), int(user))
        c = self._content.get(key)
        if c is None:
            c = self._content[key] = SlotContent(key[0], key[1], self.g.max_seq_len)
        return c

    def _touch(self, c: SlotContent) -> None:
        self._tick += 1
        c.last_used = self._tick

    def begin(self, row: int, user: int, start: int) -> SlotContent:
        """A request takes the slot: a fresh prompt (``start == 0``: the resident content is dropped) or a served
        hit (``start > 0``: the content is kept up to ``start``). Pins the slot."""
        c = self.content(row, user)
        if start == 0:
            if c.retained:
                self.stats["retained_evicted"] += 1
            c.clear()
        else:
            c.truncate(start)
        c.live = True
        self._touch(c)
        return c

    def record_prefill(self, row: int, user: int, start: int, ids: torch.Tensor, spans: Sequence[MMSpan] = ()) -> None:
        c = self.content(row, user)
        c.write(start, ids)
        end = start + int(torch.as_tensor(ids).reshape(-1).numel())
        for sp in spans:
            if sp.offset < end and sp.end > start and all(sp.offset != m.offset for m in c.mm):
                c.mm.append(sp)

    def record_decode(self, row: int, user: int, position: int, token: int) -> None:
        """The decode step consumed ``token`` at ``position``: the slot's content grows by one."""
        c = self.content(row, user)
        if position < c.length:  # the host position is authoritative (a replay / reset): the tail is rewritten
            c.truncate(position)
        c.write(position, torch.tensor([int(token)], dtype=torch.int32))

    def release(self, row: int, user: int) -> None:
        """The request finished: the slot keeps its content (retained) until a prompt takes the slot."""
        c = self.content(row, user)
        c.live = False
        self._touch(c)

    def forget(self, row: int, user: int) -> None:
        c = self.content(row, user)
        c.clear()
        c.live = False

    # ------------------------------------------------------------------ policy
    def slot_rank(self, row: int, user: int) -> int:
        """``SlotTable.allocate`` tie-break: 0 for a slot without retained content, then the retained slots from the
        least to the most recently used (the LRU resident prefix is the first dropped)."""
        c = self._content.get((int(row), int(user)))
        if c is None or not c.retained:
            return 0
        order = sorted(x.last_used for x in self._content.values() if x.retained)
        return 1 + order.index(c.last_used)

    def park_positions(self) -> dict[tuple[int, int], int]:
        """Idle slots with resident content park at their content end (``SlotTable.device_feed``); content that fills
        the context cannot park past itself and is dropped."""
        out = {}
        for c in self._content.values():
            if c.retained:
                if c.length < self.g.max_seq_len:
                    out[c.key] = c.length
                else:
                    c.clear()
        return out

    def lookup(self, ids: torch.Tensor, spans: Sequence[MMSpan], n: int) -> PrefixHit | None:
        """The slot to continue a request whose first ``n`` tokens vLLM found cached: a retained slot whose content
        equals ``ids[:n]`` (multimodal spans by identifier) with a snapshot at or below ``n``; the highest snapshot
        position wins, then the most recently used slot. None = the hit cannot be served (prefill from 0)."""
        best = None
        for c in self._content.values():
            if not c.retained or not c.matches(ids, spans, n):
                continue
            snap = c.best_snapshot(n)
            if snap is None:
                continue
            key = (snap[0], c.last_used)
            if best is None or key > best[0]:
                best = (key, PrefixHit(c.row, c.user, snap[0], snap[1]))
        return None if best is None else best[1]

    def plan_snapshot(self, row: int, user: int, position: int, kind: str) -> int | None:
        """Which ring entry a KDA snapshot of the slot at ``position`` goes to (None: no ring, position 0, not a
        32-multiple, or already held). Records it: the caller then copies the device state into that entry."""
        if kind not in SNAPSHOT_KINDS:
            raise ValueError(f"snapshot kind {kind!r}")
        if self.entries == 0 or position <= 0 or position % TILE:
            return None
        c = self.content(row, user)
        if position in c.snapshots:
            return None
        if kind == "decode" or self.entries == 1:
            entry = 0
        else:
            used = {e for e in c.snapshots.values()}
            free = [e for e in range(1, self.entries) if e not in used]
            if free:
                entry = free[0]
            else:
                pos_min = min(p for p, e in c.snapshots.items() if e != 0)
                entry = c.snapshots[pos_min]
        c.snapshots = {p: e for p, e in c.snapshots.items() if e != entry}
        c.snapshots[position] = entry
        self.stats["snapshots"] += 1
        return entry

    def retained_slots(self) -> list[tuple[int, int]]:
        return sorted(c.key for c in self._content.values() if c.retained)


def mm_spans_of_row(kwargs: dict, row: int) -> list[MMSpan]:
    """The multimodal spans of prefill row ``row`` from the plugin's per-field lists (``mm_placeholder_offsets`` /
    ``mm_placeholder_lengths``, the kind from which pixel field is set, ``mm_identifiers`` when forwarded). An item
    without a placeholder position yields no span (its content cannot be located)."""
    offs = kwargs.get("mm_placeholder_offsets")
    lens = kwargs.get("mm_placeholder_lengths")
    if not offs or row >= len(offs) or offs[row] is None:
        return []
    idents = kwargs.get("mm_identifiers")
    row_ids = idents[row] if idents and row < len(idents) and idents[row] is not None else None
    videos = kwargs.get("pixel_values_videos")
    row_videos = videos[row] if videos and row < len(videos) and videos[row] is not None else None
    out = []
    for k, (o, n) in enumerate(zip(offs[row], lens[row])):
        if o is None or n is None:
            continue
        kind = "video" if row_videos is not None and k < len(row_videos) and row_videos[k] is not None else "image"
        ident = row_ids[k] if row_ids is not None and k < len(row_ids) else None
        out.append(MMSpan(int(o), int(n), kind, None if ident is None else str(ident)))
    return out


def mm_payload_missing(kwargs: dict, row: int, n: int) -> list[int]:
    """Items of prefill row ``row`` whose placeholders lie inside ``[0, n)`` and whose tensor payload the plugin
    stripped (vLLM ``strip_covered_mm_data``: fully prefix-cached items carry no pixels), i.e. the items a prefill
    from 0 could not scatter. Empty for a text row."""
    offs = kwargs.get("mm_placeholder_offsets")
    lens = kwargs.get("mm_placeholder_lengths")
    if not offs or row >= len(offs) or offs[row] is None:
        return []
    out = []
    for k, (o, ln) in enumerate(zip(offs[row], lens[row])):
        if o is None or ln is None or o + ln > n:
            continue
        has = False
        for field in ("pixel_values", "pixel_values_videos", "input_features"):
            col = kwargs.get(field)
            if col and row < len(col) and col[row] is not None and k < len(col[row]) and col[row][k] is not None:
                has = True
        if not has:
            out.append(k)
    return out
