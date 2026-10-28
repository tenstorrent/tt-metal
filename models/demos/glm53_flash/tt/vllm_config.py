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

# The sparse prefill's prefix read (``TtLatentCache.user_rows_rm``: the user's latent rows [0, start + T)) rounds its
# row count up to a multiple of this (0 = exact): the kv shape of a continued chunk then depends on the prefix bucket,
# not on the exact start, and the served warm-up builds every bucket's programs (``_warmup_prefix_starts``). The
# served adapter sets it to max(chunk, max_seq_len / 8 rounded to the chunk) before the caches are built.
PREFIX_BUCKET_ENV = "GLM53_PREFIX_BUCKET"
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

TILE = 32
DENSE_PREFILL_ROWS = 2048  # the traced per-length prefill covers prompts up to this many tokens
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
    def allocate(self, logical: int) -> SlotState:
        """Give ``logical`` a physical slot for a new prompt. A logical slot the plugin hands out again
        (``empty_slots`` of a prefill) is released first: the plugin reuses a slot only after its previous
        request finished, and a resumed request re-prefills from position 0."""
        self._check_logical(logical)
        if self._phys[logical] is not None:
            self.release(logical)
        col_loads, row_loads = self.column_loads(), self.row_loads()
        best = None  # (key, row, user)
        if self.g.rows_independent and self.g.wave_affinity == "user":
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
                            key = (col_loads[col], row_loads[row], col, row, user)
                            if best is None or key < best[0]:
                                best = (key, row, user)
                            break
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
    def device_feed(self, tokens: dict[int, int], positions: dict[int, int], park_position: int = 0):
        """Flat ``[rows * users_per_row]`` token and position lists for ``load_ids`` / ``set_positions``:
        live logical slots at their physical indices (every row's copy when the rows replicate), idle
        physical slots parked at ``(pad_token_id, park_position)``. ``tokens`` / ``positions`` are keyed by
        logical slot and must agree."""
        if set(tokens) != set(positions):
            raise ValueError(f"tokens for slots {sorted(tokens)} but positions for {sorted(positions)}")
        ids = [self.g.pad_token_id] * self.g.physical_slots
        pos = [park_position] * self.g.physical_slots
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
    def walk(start: int, end: int, chunk: int) -> tuple[tuple[int, int], ...]:
        """``[start, end)`` in consecutive device chunks of at most ``chunk`` tokens."""
        out = []
        s0 = start
        while s0 < end:
            e = min(end, s0 + chunk)
            out.append((s0, e))
            s0 = e
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

    def plan(self, start: int, end: int, *, enable_trace: bool = True) -> PrefillPlan:
        start, end = self.check_range(start, end)
        n = end - start
        if start == 0 and enable_trace:
            T = self.captured_length(n)
            if T is not None:
                return PrefillPlan("traced", ((0, end),), T)
        return PrefillPlan("eager", self.walk(start, end, self.g.device_prefill_chunk))


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
