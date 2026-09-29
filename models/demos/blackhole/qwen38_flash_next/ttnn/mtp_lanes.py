# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP lanes: B lanes verify R = k + 1 rows each in ONE 32-row tile (speculative drafting with exact acceptance on
the batched lane body).

Row layout, lane-major: row ``u*R + j`` is lane u's row j (j = 0 the pending token, j = 1 .. k the drafts); rows
``B*R .. 31`` are pad rows (token ``ZERO_EMBEDDING_TOKEN``, lane 0's geometry, never read into or written from a
cache).  Every dense op of the B=1 verify (``mtp_v2.forward_verify``) runs unchanged on the tile: the embedding, the
GR reads and writes, the projections, the MoE on the ``B*R`` real rows, the final mixer, the LM head and the per-row
greedy resolve are row-independent.  The per-lane state work runs on a batch axis behind exact 0/1 selects: the GDN
rows path with lane u's own FIR history and recurrent state and ONE batched chunk-kernel call
(``gdn.forward_rows_lanes`` / ``commit_rows_lanes``), the QSA verify with lane u's KV region of the flat cache, its
compressed blocks and its raw-key history (``qsa.forward_verify_lanes`` / ``commit_verify_lanes``), the PLE rows path
with lane u's dilated history (``ple.inject_rows_lanes`` / ``commit_rows_lanes``), the MTP alignment layer's lane
state and its per-lane residual row.  Positions live in one UINT32 row (lane u = P_u, any residues; the bodies bind
no GDN ring phase) and every position-dependent input is derived from it on device, so one trace serves every
combination of lane positions.

Stage 1 (this module's first form): the verify + alignment + commit body with host-written accept counts and active
mask (the teacher-forced measurement: every lane commits ``R`` rows per pass, ``accepted_u := R - 1``); the accept
arithmetic runs and is read back but drives nothing.  The per-lane accept driving the pass, the seed / import and
the draft body follow in the next stages on the same states.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch
from ttnn.tools.trace_allocation_tracker import corruptible_allocation_scope

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    BLOCK_START_LANE_MASK,
    CHUNK_ROWS,
    MESH_SHAPE,
    POSITION_INDEX_ROW_SHAPE,
    UINT32_LIMIT,
    Qwen38MeshContract,
    TensorPlacement,
    replicate_tensor_2d_mesh_mapper,
    require_lane_count,
    require_rope_shift,
    tensor_metadata,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import TOKEN_ROW_SHAPE, ZERO_EMBEDDING_TOKEN
from models.demos.blackhole.qwen38_flash_next.ttnn.final_mixer import Qwen38TTNNFinalMixer
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDN
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import (
    KV_STAGINGS,
    KV_WIDTH,
    RECURRENT_ROWS,
    Qwen38LaneHostSlot,
    Qwen38TTNNLanePager,
    slice_owned,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    BACKBONE_LAYERS,
    HIDDEN_SIZE,
    LOCAL_HIDDEN_SIZE,
    PLE_CHECKPOINT_LAYER,
    RESIDUAL_BRANCHES,
    Qwen38TTNNDecoderLayer,
    Qwen38TTNNLayerNamespace,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNTextModel
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import SUPPORTED_ROWS, Qwen38TTNNMoE
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp import Qwen38TTNNMTPInput
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_v2 import (
    BLOCK_ROWS_SHAPE,
    RESIDUAL_ROWS_SHAPE,
    SUPPORTED_DRAFTS,
    Qwen38TTNNVerifyOutput,
    _allocate_hidden_sharded_zeros,
    _deallocate,
    _embed_rows,
    _land,
    _lane,
    _pad_rows,
    _resolve_rows,
    _run_cleanup,
    _shape,
    _tensor_key,
    _upload_replicated,
    _validate_mtp_components,
    resolve_moe_rows,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import CONV_STATE_LENGTH, Qwen38TTNNPLERowsPreparedInput

# The KV slab import writes ``update_padded_kv_cache`` chunks of this many rows (a lane's C-row slab in C / chunk
# writes): the op needs the cache rows to be a multiple of the rows one call writes (its block-cyclic invariant,
# ``cache_seq % written_seq == 0``; a 4-lane 32k cache with 64 scratch rows refused a 32k slab, mtp-b4-s2-lanes4), so
# the scratch region past the last lane is exactly one chunk.  An inactive lane's two block writes land in its first
# 64 rows.  4096 rows = 4 MiB per QSA layer per device; 8 writes per layer at 32k.
KV_IMPORT_CHUNK_ROWS = 4096


def kv_scratch_rows(allocated_context: int) -> int:
    """The scratch rows past the last lane of a lane KV cache at ``allocated_context`` rows per lane: one import chunk
    (the context is a whole number of chunks; a context below the chunk is one chunk itself)."""

    chunk = min(KV_IMPORT_CHUNK_ROWS, int(allocated_context))
    if allocated_context % chunk or chunk % (2 * qsa_module.CACHE_WRITE_ROWS):
        raise ValueError(
            f"lane KV context {allocated_context} is not a whole number of {chunk}-row import chunks of two blocks"
        )
    return chunk


PAD_DRAFT_COUNT = 64.0  # a running accept count no pad column reaches: its prefix flag is 0
READBACK_FIXED_LANES = ("accepted", "next_token", "first_draft")


def lane_of_rows(lanes: int, rows: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``lane_of(r)`` and ``draft_of(r)`` for the 32 lane-major rows (pad rows: lane 0, draft ``R - 1``)."""

    real = lanes * rows
    lane_of = tuple(row // rows if row < real else 0 for row in range(CHUNK_ROWS))
    draft_of = tuple(row % rows if row < real else rows - 1 for row in range(CHUNK_ROWS))
    return lane_of, draft_of


def _validate_lane_geometry(lanes: int, drafts: int) -> tuple[int, int]:
    lanes = require_lane_count(lanes, label="MTP lanes")
    if drafts not in SUPPORTED_DRAFTS:
        raise ValueError(f"MTP lanes admit k in {SUPPORTED_DRAFTS}, got {drafts!r}")
    rows = drafts + 1
    if lanes * rows > CHUNK_ROWS:
        raise ValueError(f"MTP lanes admit B x R <= {CHUNK_ROWS} rows in one tile, got {lanes} x {rows}")
    return lanes, rows


def mtp_lane_constant_rows(lanes: int, drafts: int) -> dict[str, torch.Tensor]:
    """Host images of the accept and alignment constants of ``lanes`` lanes at ``drafts`` drafts (R = k + 1).

    ``prefix_upper_block`` ``[32, 32]`` (``U[i, c] = 1`` iff ``lane_of(i) == lane_of(c)`` and ``i <= c``, both real):
    the match flags' running counts per lane in one matmul; ``draft_plus_one_row`` ``[1, 32]`` (``draft_of(c) + 1``,
    64 on the pad columns, whose flags are 1 -- the sentinel argmax equals the sentinel draft -- so no pad prefix flag
    is ever 1); ``block_sum`` ``[32, B]`` (``[lane_of(c) == u]``, zero pad columns: the accept count per lane);
    ``block_sum_t`` ``[B, 32]`` its transpose (the alignment residual row select's lane rows); ``draft_match_row``
    ``[1, 32]`` (``draft_of(c)``, -1 on the pad columns: the one-hot of column ``u*R + a_u`` against the gathered
    counts); ``lane_base_row`` ``[1, B]`` (``u*R``: the gather index base).
    """

    lanes, rows = _validate_lane_geometry(lanes, drafts)
    real = lanes * rows
    lane_of, draft_of = lane_of_rows(lanes, rows)
    prefix = torch.zeros(CHUNK_ROWS, CHUNK_ROWS)
    block_sum = torch.zeros(CHUNK_ROWS, lanes)
    for row in range(real):
        block_sum[row, lane_of[row]] = 1.0
        for column in range(row, real):
            if lane_of[column] == lane_of[row]:
                prefix[row, column] = 1.0
    as_row = lambda values: torch.tensor(values, dtype=torch.float32).reshape(1, CHUNK_ROWS)  # noqa: E731
    return {
        "prefix_upper_block": prefix,
        "draft_plus_one_row": as_row([draft_of[c] + 1 if c < real else PAD_DRAFT_COUNT for c in range(CHUNK_ROWS)]),
        "block_sum": block_sum,
        "block_sum_t": block_sum.t().contiguous(),
        "draft_match_row": as_row([draft_of[c] if c < real else ZERO_EMBEDDING_TOKEN for c in range(CHUNK_ROWS)]),
        "lane_base_row": torch.tensor([lane * rows for lane in range(lanes)], dtype=torch.int64).reshape(1, lanes),
    }


@dataclass(frozen=True)
class Qwen38TTNNMTPLaneConstants:
    """Replicated accept / alignment constants of one ``(lanes, drafts)`` (see :func:`mtp_lane_constant_rows`).

    FP32 TILE: ``prefix_upper_block`` ``[1,1,32,32]``, ``draft_plus_one_row`` ``[1,1,1,32]``, ``block_sum``
    ``[1,1,32,B]``, ``draft_match_row`` ``[1,1,1,32]``; BF16 TILE ``block_sum_t`` ``[1,1,B,32]``; UINT32 ROW_MAJOR
    ``lane_base_row`` ``[1,1,1,B]``; FP32 ROW_MAJOR ``sentinel_tail`` ``[1,1,1,32-B*R]`` (``ZERO_EMBEDDING_TOKEN``,
    the resolve's pad lanes; None at 32 rows).
    """

    lanes: int
    drafts: int
    rows: int
    lane_of: tuple[int, ...]
    draft_of: tuple[int, ...]
    prefix_upper_block: Any
    draft_plus_one_row: Any
    block_sum: Any
    block_sum_t: Any
    draft_match_row: Any
    lane_base_row: Any
    sentinel_tail: Any
    compute_config: Any

    @classmethod
    def build(cls, mesh_device, mesh_contract: Qwen38MeshContract, *, lanes: int, drafts: int):
        mesh_contract.validate_mesh(mesh_device)
        lanes, rows = _validate_lane_geometry(lanes, drafts)
        host = mtp_lane_constant_rows(lanes, drafts)
        lane_of, draft_of = lane_of_rows(lanes, rows)
        uploaded: list[Any] = []

        def upload(values: torch.Tensor, dtype, layout, label: str):
            tensor = _upload_replicated(mesh_device, mesh_contract, values, dtype, layout, label=label)
            uploaded.append(tensor)
            return tensor

        tile, row_major = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
        tail = CHUNK_ROWS - lanes * rows
        try:
            return cls(
                lanes=lanes,
                drafts=drafts,
                rows=rows,
                lane_of=lane_of,
                draft_of=draft_of,
                prefix_upper_block=upload(
                    host["prefix_upper_block"].reshape(1, 1, CHUNK_ROWS, CHUNK_ROWS),
                    ttnn.float32,
                    tile,
                    "lane accept prefix matrix",
                ),
                draft_plus_one_row=upload(
                    host["draft_plus_one_row"].reshape(1, 1, 1, CHUNK_ROWS), ttnn.float32, tile, "lane accept counts"
                ),
                block_sum=upload(
                    host["block_sum"].reshape(1, 1, CHUNK_ROWS, lanes), ttnn.float32, tile, "lane block sum"
                ),
                block_sum_t=upload(
                    host["block_sum_t"].reshape(1, 1, lanes, CHUNK_ROWS).to(torch.bfloat16),
                    ttnn.bfloat16,
                    tile,
                    "lane block sum rows",
                ),
                draft_match_row=upload(
                    host["draft_match_row"].reshape(1, 1, 1, CHUNK_ROWS), ttnn.float32, tile, "lane draft index row"
                ),
                lane_base_row=upload(
                    host["lane_base_row"].reshape(1, 1, 1, lanes).to(torch.uint32),
                    ttnn.uint32,
                    row_major,
                    "lane row base",
                ),
                sentinel_tail=(
                    None
                    if tail == 0
                    else upload(
                        torch.full((1, 1, 1, tail), float(ZERO_EMBEDDING_TOKEN)),
                        ttnn.float32,
                        row_major,
                        "lane sentinel tail",
                    )
                ),
                compute_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
            )
        except BaseException:
            _deallocate(*uploaded)
            raise

    def deallocate(self) -> None:
        _deallocate(
            self.prefix_upper_block,
            self.draft_plus_one_row,
            self.block_sum,
            self.block_sum_t,
            self.draft_match_row,
            self.lane_base_row,
            *(() if self.sentinel_tail is None else (self.sentinel_tail,)),
        )


# --------------------------------------------------------------------------- the lane positions (any residues)


def _uint32_row(values: Sequence[int]) -> torch.Tensor:
    return (
        torch.tensor([int(value) for value in values], dtype=torch.int64)
        .reshape(POSITION_INDEX_ROW_SHAPE)
        .to(torch.uint32)
    )


def _host_row_tensor(mesh_device, host: torch.Tensor, dtype, layout):
    return ttnn.from_torch(host, dtype=dtype, layout=layout, mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device))


@dataclass
class Qwen38TTNNMTPLanePositions:
    """The lanes' positions: one UINT32 ROW_MAJOR ``[1,1,1,32]`` row (lane u = P_u for u < B, the pad lanes lane 0's
    P) and its host mirror, plus the ``P & ~3`` mask row.  Unlike ``Qwen38TTNNDevicePositionRow`` no residue is
    shared: the MTP lane bodies bind no GDN ring phase.  The body advances lane u by ``(a_u + 1) * active_u`` as its
    last op; the host mirror follows from the readback (:meth:`advance_mirror`)."""

    row: Any
    block_start_mask_row: Any
    lanes: int
    positions: list[int]
    mesh_device: Any = field(repr=False, compare=False)
    mesh_contract: Qwen38MeshContract = field(repr=False, compare=False)
    # Lane u's rotary shift ``S_u`` (UINT32 ROW_MAJOR ``[1,1,1,32]``, the pad lanes 0; ``S_u <= P_u & ~3``) and its
    # host mirror: the RoPE table row of lane u's token at P_u is ``P_u - S_u`` (``mrope.Qwen38MRoPEPositions.shift``;
    # 0 for text); the caches and masks stay on P_u.  Written with the position at an admission; a rewind keeps it.
    # None: a row built without one (a text-only stand-in): the rotary rows are the plain rows.
    shift_row: Any = None
    shifts: list[int] = field(default_factory=list)

    @staticmethod
    def _lane_values(positions: Sequence[int], lanes: int) -> list[int]:
        values = [int(value) for value in positions]
        if len(values) != lanes or any(not 0 <= value < UINT32_LIMIT for value in values):
            raise ValueError(f"lane positions need {lanes} ints in [0, 2^32), got {positions!r}")
        return values

    @staticmethod
    def _lane_shifts(shifts: Sequence[int] | None, positions: Sequence[int], lanes: int) -> list[int]:
        """The lanes' rotary shifts (None: all 0), each an int in ``[0, P_u]`` so ``P_u - S_u`` never wraps."""

        values = [0] * lanes if shifts is None else [int(value) for value in shifts]
        if len(values) != lanes:
            raise ValueError(f"lane shifts need {lanes} ints, got {shifts!r}")
        for shift, position in zip(values, positions):
            require_rope_shift(shift, int(position))  # the lane's block-start row (P_u & ~3) - S_u stays >= 0
        return values

    def _host_row(self, values: Sequence[int], pad: int) -> torch.Tensor:
        return _uint32_row(list(values) + [pad] * (CHUNK_ROWS - self.lanes))

    @classmethod
    def allocate(
        cls,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        positions: Sequence[int],
        *,
        lanes: int,
        shifts: Sequence[int] | None = None,
    ):
        mesh_contract.validate_mesh(mesh_device)
        lanes = require_lane_count(lanes, label="MTP lane positions lanes")
        values = cls._lane_values(positions, lanes)
        shift_values = cls._lane_shifts(shifts, values, lanes)
        row = _upload_replicated(
            mesh_device,
            mesh_contract,
            _uint32_row(values + [values[0]] * (CHUNK_ROWS - lanes)),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            label="lane positions",
        )
        owned = [row]
        try:
            mask = _upload_replicated(
                mesh_device,
                mesh_contract,
                _uint32_row([BLOCK_START_LANE_MASK] * CHUNK_ROWS),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                label="lane block-start mask row",
            )
            owned.append(mask)
            shift_row = _upload_replicated(
                mesh_device,
                mesh_contract,
                _uint32_row(shift_values + [0] * (CHUNK_ROWS - lanes)),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                label="lane rotary shifts",
            )
        except BaseException:
            _deallocate(*owned)
            raise
        return cls(row, mask, lanes, values, mesh_device, mesh_contract, shift_row, shift_values)

    def write(self, positions: Sequence[int], shifts: Sequence[int] | None = None) -> None:
        """Host write of every lane's position, and of every lane's rotary shift when ``shifts`` is given (outside
        any trace); a rewind of the positions alone keeps the shifts (a request's shift is fixed at its admission)."""

        values = self._lane_values(positions, self.lanes)
        current = list(self.shifts) if self.shifts else [0] * self.lanes
        shift_values = current if shifts is None else self._lane_shifts(shifts, values, self.lanes)
        self._lane_shifts(shift_values, values, self.lanes)
        ttnn.copy_host_to_device_tensor(
            _host_row_tensor(self.mesh_device, self._host_row(values, values[0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            self.row,
        )
        if self.shift_row is None:
            if any(shift_values):
                raise RuntimeError("these lane positions carry no rotary shift row")
        elif shifts is not None or shift_values != current:
            ttnn.copy_host_to_device_tensor(
                _host_row_tensor(self.mesh_device, self._host_row(shift_values, 0), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                self.shift_row,
            )
        self.positions = values
        self.shifts = shift_values

    def write_lane(self, lane: int, position: int, rope_shift: int = 0) -> None:
        """Host write of one lane's position and rotary shift (an admission; outside any trace): the rows from the
        updated mirrors."""

        if isinstance(lane, bool) or type(lane) is not int or not 0 <= lane < self.lanes:
            raise ValueError(f"lane must be an int in [0, {self.lanes}), got {lane!r}")
        values = list(self.positions)
        values[lane] = int(position)
        shifts = list(self.shifts)
        shifts[lane] = int(rope_shift)
        self.write(values, shifts)

    def advance_mirror(self, committed: Sequence[int]) -> None:
        """The mirror after a body advanced lane u by ``committed[u]`` rows."""

        if len(committed) != self.lanes or any(int(count) < 0 for count in committed):
            raise ValueError(f"lane advance needs {self.lanes} non-negative counts, got {committed!r}")
        self.positions = [value + int(count) for value, count in zip(self.positions, committed)]

    def read(self) -> list[int]:
        """Diagnostic readback of the B lanes from coordinate 0 (outside any trace)."""

        values = ttnn.to_torch(ttnn.get_device_tensors(self.row)[0]).reshape(-1).to(torch.int64) & (UINT32_LIMIT - 1)
        return [int(value) for value in values.tolist()[: self.lanes]]

    def read_shifts(self) -> list[int]:
        """Diagnostic readback of the B lanes' rotary shifts from coordinate 0 (outside any trace)."""

        values = ttnn.to_torch(ttnn.get_device_tensors(self.shift_row)[0]).reshape(-1).to(torch.int64)
        return [int(value) & (UINT32_LIMIT - 1) for value in values.tolist()[: self.lanes]]

    def rope_rows(self, row_positions, block_start_rows, lane_of_row, block_lane_of_row):
        """The RoPE index rows of the verify rows: ``row_positions`` (UINT32 ``[1,1,1,32]``, row r = P_u + d_r) and
        ``block_start_rows`` (row r = (P_u & ~3) + 4 i) less the shift of the row's lane (gathered like the
        positions were, ``lane_of_row`` / ``block_lane_of_row``): exact UINT32 subtracts.  Returns the two rows;
        the caller deallocates them."""

        dram = ttnn.DRAM_MEMORY_CONFIG
        if self.shift_row is None:  # no shift row: the plain rows, as fresh tensors (the caller releases them)
            return ttnn.add(row_positions, 0, memory_config=dram), ttnn.add(block_start_rows, 0, memory_config=dram)
        row_shifts = ttnn.gather(self.shift_row, 3, lane_of_row, memory_config=dram)
        block_shifts = ttnn.gather(self.shift_row, 3, block_lane_of_row, memory_config=dram)
        rope_rows = ttnn.subtract(row_positions, row_shifts, memory_config=dram)
        rope_block_rows = ttnn.subtract(block_start_rows, block_shifts, memory_config=dram)
        _deallocate(row_shifts, block_shifts)
        return rope_rows, rope_block_rows

    def deallocate(self) -> None:
        _deallocate(self.row, self.block_start_mask_row, *(() if self.shift_row is None else (self.shift_row,)))


# --------------------------------------------------------------------------- states


@dataclass(frozen=True)
class Qwen38TTNNMTPLaneLayerState:
    """One layer's lane verify buffers: the B-lane attention state (the GDN lane state with ``recurrent
    [B,12,128,128]`` or the QSA lane state with its scratch rows) and its rows / verify companion (the GDN lane rows
    state or the QSA lane verify state), the PLE lane rows state (layer 1), the MoE instance at the verify's row
    count with its persistent input slice (None when it takes the whole tile)."""

    namespace: Qwen38TTNNLayerNamespace
    layer_index: int
    attention_state: Any
    attention_rows: Any
    ple: Any
    moe: Qwen38TTNNMoE
    moe_input: Any


@dataclass(frozen=True)
class Qwen38TTNNMTPLaneAlignment:
    """The MTP alignment layer over the lanes: its lane layer state (its own lane QSA cache and raw history) and
    ``residual`` ``[1,4,B,640]`` = row ``u*R + a_u`` of the alignment residuals per lane (the draft body's start)."""

    layer: Qwen38TTNNDecoderLayer
    input_mixer: Qwen38TTNNMTPInput
    final_mixer: Qwen38TTNNFinalMixer
    layer_state: Qwen38TTNNMTPLaneLayerState
    residual: Any


@dataclass(frozen=True)
class Qwen38TTNNMTPLaneVerifyState:
    """Everything a lane verify pass reads or writes, allocated before any capture (the traces bake every address).

    Host-written per pass: ``token_row`` FP32 TILE ``[1,1,1,32]`` (lane-major ``[t_u, d_1 .. d_k]`` blocks, then
    ``ZERO_EMBEDDING_TOKEN``), ``draft_lanes`` FP32 ROW_MAJOR ``[1,1,1,32]`` (``d_{u, j+1}`` at ``u*R + j`` for j < k,
    -1 at j = k and past B*R), ``ple_rows`` (the ``[1,1,B*R,640]`` upload of every lane's n-gram rows), and, when
    they change, the per-lane accept counts (``accepted_lanes`` FP32 TILE ``[1,B,1,1]`` and ``accepted_row`` FP32
    ROW_MAJOR ``[1,1,1,32]``, -1 on the pad lanes: stage 1 pins them at ``R - 1``), the active mask of this pass
    (``active_lanes`` FP32 TILE ``[1,B,1,1]``; ``active_row`` / ``inactive_row`` UINT32 ROW_MAJOR ``[1,1,1,32]``: an
    inactive lane's KV writes go to the scratch rows and its position does not advance), the commit mask
    (``commit_lanes`` FP32 TILE ``[1,B,1,1]`` = the active mask of the pass being committed: a lane parked at this
    pass still commits its last real pass, and a lane just un-parked commits nothing of its parked junk rows) and
    ``k_eff_mask_row`` FP32 ROW_MAJOR ``[1,1,1,32]`` (1.0: every draft counts).  ``pass_contexts[u]`` holds lane u's R + 1 n-gram
    contexts of the current pass (``[c]`` after committing c of its rows); the committed context per lane lives in
    the checkpoint layer's PLE lane rows state.
    """

    lanes: int
    drafts: int
    rows: int
    moe_rows: int
    constants: Qwen38TTNNMTPLaneConstants
    gdn_rows_constants: gdn_module.Qwen38TTNNGDNRowsConstants
    gdn_lane_constants: gdn_module.Qwen38TTNNGDNLaneRowsConstants
    qsa_chunk_constants: qsa_module.Qwen38TTNNQSAChunkConstants
    qsa_verify_constants: qsa_module.Qwen38TTNNQSAVerifyConstants
    qsa_lane_constants: qsa_module.Qwen38TTNNQSALaneConstants
    qsa_lane_verify_constants: qsa_module.Qwen38TTNNQSALaneVerifyConstants
    layers: tuple[Qwen38TTNNMTPLaneLayerState, ...]
    alignment: Qwen38TTNNMTPLaneAlignment
    positions: Qwen38TTNNMTPLanePositions
    token_row: Any
    draft_lanes: Any
    ple_rows: Qwen38TTNNPLERowsPreparedInput
    accepted_lanes: Any
    accepted_row: Any
    active_lanes: Any
    active_row: Any
    inactive_row: Any
    commit_lanes: Any
    k_eff_mask_row: Any
    pass_contexts: list[tuple[tuple[int, int] | None, ...]]
    _owner: object = field(repr=False, compare=False)


def _allocate_layer_lane_state(
    layer: Qwen38TTNNDecoderLayer, lanes: int, rows: int, gdn_rows_constants, gdn_lane_constants, *, moe_rows: int
) -> Qwen38TTNNMTPLaneLayerState:
    actions: list[tuple[str, Callable[[], Any]]] = []
    try:
        if isinstance(layer.attention, Qwen38TTNNGDN):
            attention_state = layer.attention.allocate_lane_state(lanes)
            actions.append(("GDN lane state", attention_state.deallocate))
            attention_rows = layer.attention.allocate_lane_rows_state(gdn_rows_constants, gdn_lane_constants)
            actions.append(("GDN lane rows state", attention_rows.deallocate))
        else:
            attention_state = layer.attention.allocate_lane_state(
                lanes, kv_scratch_rows=kv_scratch_rows(layer.attention.allocated_context)
            )
            actions.append(("QSA lane state", lambda: layer.attention.release_lane_state(attention_state)))
            attention_rows = layer.attention.allocate_lane_verify_state(lanes)
            actions.append(("QSA lane verify state", lambda: layer.attention.release_lane_verify_state(attention_rows)))
        ple = None
        if layer.ple is not None:
            ple = layer.ple.allocate_lane_rows_state(lanes, rows)
            actions.append(("PLE lane rows state", ple.deallocate))
        moe = Qwen38TTNNMoE(
            layer.mlp.mesh_device,
            layer.mlp.mesh_contract,
            layer.mlp.weights,
            tt_ccl=layer.mlp.tt_ccl,
            collective_topology=layer.mlp.collective_topology,
            rows=moe_rows,
            synchronization_policy=layer.mlp.synchronization_policy,
            admitted_rows=SUPPORTED_ROWS if moe_rows in SUPPORTED_ROWS else (moe_rows,),
        )
        actions.append(("lane MoE buffers", moe.release_owned_buffers))
        moe_input = None
        if moe_rows != CHUNK_ROWS:
            moe_input = _allocate_hidden_sharded_zeros(
                layer.mlp.mesh_device,
                layer.mlp.mesh_contract,
                (1, 1, moe_rows, LOCAL_HIDDEN_SIZE),
                label=f"layer {layer.layer_index} lane MoE input",
            )
        return Qwen38TTNNMTPLaneLayerState(
            layer.namespace, layer.layer_index, attention_state, attention_rows, ple, moe, moe_input
        )
    except BaseException as error:
        _run_cleanup("decoder-layer lane verify state allocation", list(reversed(actions)), primary=error)
        raise


def _release_layer_lane_state(layer: Qwen38TTNNDecoderLayer, state: Qwen38TTNNMTPLaneLayerState) -> None:
    actions: list[tuple[str, Callable[[], Any]]] = [("lane MoE buffers", state.moe.release_owned_buffers)]
    if state.moe_input is not None:
        actions.append(("lane MoE input", lambda: _deallocate(state.moe_input)))
    if state.ple is not None:
        actions.append(("PLE lane rows state", state.ple.deallocate))
    if isinstance(layer.attention, Qwen38TTNNGDN):
        actions.append(("GDN lane rows state", state.attention_rows.deallocate))
        actions.append(("GDN lane state", state.attention_state.deallocate))
    else:
        actions.append(
            ("QSA lane verify state", lambda: layer.attention.release_lane_verify_state(state.attention_rows))
        )
        actions.append(("QSA lane state", lambda: layer.attention.release_lane_state(state.attention_state)))
    _run_cleanup("decoder-layer lane verify state", actions)


def _mask_row_host(values: Sequence[int], lanes: int) -> torch.Tensor:
    """``[1,1,1,32]`` int64 0/1 row; the pad lanes take 1 (their positions never advance: their counts are -1)."""

    if len(values) != lanes or any(value not in (0, 1) for value in values):
        raise ValueError(f"lane mask needs {lanes} 0/1 values, got {values!r}")
    return torch.tensor(list(values) + [1] * (CHUNK_ROWS - lanes), dtype=torch.int64).reshape(POSITION_INDEX_ROW_SHAPE)


def _accepted_row_host(values: Sequence[int], lanes: int) -> torch.Tensor:
    if len(values) != lanes:
        raise ValueError(f"lane accept counts need {lanes} values, got {values!r}")
    return torch.tensor(
        [float(value) for value in values] + [float(ZERO_EMBEDDING_TOKEN)] * (CHUNK_ROWS - lanes), dtype=torch.float32
    ).reshape(TOKEN_ROW_SHAPE)


def _allocate_lane_ple_rows(
    mesh_device, mesh_contract: Qwen38MeshContract, rows: int
) -> Qwen38TTNNPLERowsPreparedInput:
    """The persistent ROW_MAJOR BF16 ``[1,1,B*R,640]`` per device the lanes' n-gram rows are written into (the
    rows form's upload at B*R rows; the per-lane contexts live on the lane verify state)."""

    tensor = ttnn.from_torch(
        torch.zeros((1, 1, rows, HIDDEN_SIZE), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
    )
    if _shape(tensor) != (1, 1, rows, LOCAL_HIDDEN_SIZE):
        raise RuntimeError(
            f"lane PLE rows must be [1,1,{rows},{LOCAL_HIDDEN_SIZE}] local, got {tensor_metadata(tensor)}"
        )
    mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    return Qwen38TTNNPLERowsPreparedInput(tensor, (0,) * rows, ())


def allocate_lane_verify_state(
    model: Qwen38TTNNTextModel,
    *,
    lanes: int,
    drafts: int,
    mtp_components,
    moe_rows: int | None = None,
    positions: Sequence[int] | None = None,
) -> Qwen38TTNNMTPLaneVerifyState:
    """Allocate every constant and buffer of a B-lane verify at k drafts beside the resident weights (before any
    capture).  ``moe_rows`` (default the exact ``B*R`` instance; 32 = the proven 32-row form) sets every layer's
    verify MoE row count.  The MTP components (the builder's) give the alignment layer its lane state.  Lanes start
    at ``positions`` (default 0), active, with ``accepted_u = R - 1``."""

    model._require_healthy()
    lanes, rows = _validate_lane_geometry(lanes, drafts)
    total = lanes * rows
    moe_rows = resolve_moe_rows(total, total if moe_rows is None else moe_rows)
    if model.rope_table is None or model.qsa_position_constants is None:
        raise RuntimeError("generic constants are missing; allocate a generic or lane state through this owner first")
    mtp_layer, input_mixer, final_mixer = _validate_mtp_components(model, mtp_components)
    gdn = next(layer.attention for layer in model.layers if isinstance(layer.attention, Qwen38TTNNGDN))
    qsa = next(layer.attention for layer in model.layers if not isinstance(layer.attention, Qwen38TTNNGDN))
    mesh_device, mesh_contract = model.mesh_device, model.mesh_contract
    actions: list[tuple[str, Callable[[], Any]]] = []

    def keep(label: str, value, release: Callable[[], Any]):
        actions.append((label, release))
        return value

    def upload(values: torch.Tensor, dtype, layout, label: str):
        tensor = _upload_replicated(mesh_device, mesh_contract, values, dtype, layout, label=label)
        return keep(label, tensor, lambda: _deallocate(tensor))

    try:
        constants = Qwen38TTNNMTPLaneConstants.build(mesh_device, mesh_contract, lanes=lanes, drafts=drafts)
        keep("lane accept constants", constants, constants.deallocate)
        gdn_rows_constants = gdn.allocate_rows_constants(rows)
        keep("GDN rows constants", gdn_rows_constants, gdn_rows_constants.deallocate)
        gdn_lane_constants = gdn.allocate_lane_rows_constants(lanes, rows)
        keep("GDN lane rows constants", gdn_lane_constants, gdn_lane_constants.deallocate)
        chunk_constants = qsa_module.Qwen38TTNNQSAChunkConstants.build(
            mesh_device, mesh_contract, qsa.allocated_compressed_blocks
        )
        keep("QSA chunk constants", chunk_constants, chunk_constants.deallocate)
        verify_constants = qsa_module.Qwen38TTNNQSAVerifyConstants.build(
            mesh_device, mesh_contract, chunk_constants, rows=rows
        )
        keep("QSA verify constants", verify_constants, verify_constants.deallocate)
        lane_constants = qsa_module.Qwen38TTNNQSALaneConstants.build(
            mesh_device, mesh_contract, lanes=lanes, allocated_context=model.allocated_context
        )
        keep("QSA lane constants", lane_constants, lane_constants.deallocate)
        lane_verify_constants = qsa_module.Qwen38TTNNQSALaneVerifyConstants.build(
            mesh_device, mesh_contract, lanes=lanes, rows=rows, allocated_context=model.allocated_context
        )
        keep("QSA lane verify constants", lane_verify_constants, lane_verify_constants.deallocate)
        layers: list[Qwen38TTNNMTPLaneLayerState] = []
        for layer in (*model.layers, mtp_layer):
            layer_state = _allocate_layer_lane_state(
                layer, lanes, rows, gdn_rows_constants, gdn_lane_constants, moe_rows=moe_rows
            )
            layers.append(layer_state)
            keep(
                f"{layer.namespace.value} layer {layer.layer_index} lane verify state",
                layer_state,
                lambda layer=layer, layer_state=layer_state: _release_layer_lane_state(layer, layer_state),
            )
        residual = _allocate_hidden_sharded_zeros(
            mesh_device, mesh_contract, (1, RESIDUAL_BRANCHES, lanes, LOCAL_HIDDEN_SIZE), label="MTP lane residual"
        )
        keep("MTP lane residual", residual, lambda: _deallocate(residual))
        alignment = Qwen38TTNNMTPLaneAlignment(mtp_layer, input_mixer, final_mixer, layers.pop(), residual)
        position_row = Qwen38TTNNMTPLanePositions.allocate(
            mesh_device, mesh_contract, [0] * lanes if positions is None else positions, lanes=lanes
        )
        keep("lane positions", position_row, position_row.deallocate)
        token_row = model.model_io.embedding.upload_token_row(0)
        keep("lane token row", token_row, lambda: _deallocate(token_row))
        tile, row_major = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
        draft_lanes = upload(
            torch.full(TOKEN_ROW_SHAPE, float(ZERO_EMBEDDING_TOKEN)), ttnn.float32, row_major, "lane drafts"
        )
        accepted_lanes = upload(torch.full((1, lanes, 1, 1), float(rows - 1)), ttnn.float32, tile, "lane accept counts")
        accepted_row = upload(_accepted_row_host([rows - 1] * lanes, lanes), ttnn.float32, row_major, "lane accept row")
        active_lanes = upload(torch.ones(1, lanes, 1, 1), ttnn.float32, tile, "lane active mask")
        commit_lanes = upload(torch.ones(1, lanes, 1, 1), ttnn.float32, tile, "lane commit mask")
        active_row = upload(
            _mask_row_host([1] * lanes, lanes).to(torch.uint32), ttnn.uint32, row_major, "lane active row"
        )
        inactive_row = upload(
            _mask_row_host([0] * lanes, lanes).to(torch.uint32), ttnn.uint32, row_major, "lane inactive row"
        )
        k_eff_mask_row = upload(torch.ones(TOKEN_ROW_SHAPE), ttnn.float32, row_major, "lane k_eff mask row")
        if model.layers[PLE_CHECKPOINT_LAYER].ple is None or layers[PLE_CHECKPOINT_LAYER].ple is None:
            raise RuntimeError("checkpoint layer 1 PLE owner/lane rows state is unavailable")
        ple_rows = _allocate_lane_ple_rows(mesh_device, mesh_contract, total)
        keep("lane PLE rows", ple_rows, ple_rows.release)
        return Qwen38TTNNMTPLaneVerifyState(
            lanes=lanes,
            drafts=drafts,
            rows=rows,
            moe_rows=moe_rows,
            constants=constants,
            gdn_rows_constants=gdn_rows_constants,
            gdn_lane_constants=gdn_lane_constants,
            qsa_chunk_constants=chunk_constants,
            qsa_verify_constants=verify_constants,
            qsa_lane_constants=lane_constants,
            qsa_lane_verify_constants=lane_verify_constants,
            layers=tuple(layers),
            alignment=alignment,
            positions=position_row,
            token_row=token_row,
            draft_lanes=draft_lanes,
            ple_rows=ple_rows,
            accepted_lanes=accepted_lanes,
            accepted_row=accepted_row,
            active_lanes=active_lanes,
            active_row=active_row,
            inactive_row=inactive_row,
            commit_lanes=commit_lanes,
            k_eff_mask_row=k_eff_mask_row,
            pass_contexts=[(None,) * (rows + 1) for _ in range(lanes)],
            _owner=model._state_owner,
        )
    except BaseException as error:
        _run_cleanup("lane verify state allocation", list(reversed(actions)), primary=error)
        raise


def release_lane_verify_state(model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState) -> None:
    _validate_lane_verify_state(model, verify)
    actions: list[tuple[str, Callable[[], Any]]] = [("lane PLE rows", verify.ple_rows.release)]
    for label, tensor in (
        ("lane k_eff mask row", verify.k_eff_mask_row),
        ("lane commit mask", verify.commit_lanes),
        ("lane inactive row", verify.inactive_row),
        ("lane active row", verify.active_row),
        ("lane active mask", verify.active_lanes),
        ("lane accept row", verify.accepted_row),
        ("lane accept counts", verify.accepted_lanes),
        ("lane drafts", verify.draft_lanes),
        ("lane token row", verify.token_row),
    ):
        actions.append((label, lambda tensor=tensor: _deallocate(tensor)))
    actions.append(("lane positions", verify.positions.deallocate))
    alignment = verify.alignment
    actions.append(("MTP lane residual", lambda: _deallocate(alignment.residual)))
    for layer, layer_state in reversed((*zip(model.layers, verify.layers), (alignment.layer, alignment.layer_state))):
        actions.append(
            (
                f"{layer.namespace.value} layer {layer.layer_index} lane verify state",
                lambda layer=layer, layer_state=layer_state: _release_layer_lane_state(layer, layer_state),
            )
        )
    for label, constants in (
        ("QSA lane verify constants", verify.qsa_lane_verify_constants),
        ("QSA lane constants", verify.qsa_lane_constants),
        ("QSA verify constants", verify.qsa_verify_constants),
        ("QSA chunk constants", verify.qsa_chunk_constants),
        ("GDN lane rows constants", verify.gdn_lane_constants),
        ("GDN rows constants", verify.gdn_rows_constants),
        ("lane accept constants", verify.constants),
    ):
        actions.append((label, constants.deallocate))
    _run_cleanup("lane verify state", actions)


def _validate_lane_verify_state(model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState) -> None:
    if not isinstance(verify, Qwen38TTNNMTPLaneVerifyState) or verify._owner is not model._state_owner:
        raise ValueError("lane verify state was not allocated by this model owner")
    if verify.rows != verify.drafts + 1 or len(verify.layers) != BACKBONE_LAYERS:
        raise ValueError("lane verify state is incomplete")
    if (verify.constants.lanes, verify.constants.drafts) != (verify.lanes, verify.drafts):
        raise ValueError("lane accept constants were built for another lane count or draft count")
    if verify.qsa_lane_verify_constants.rows != verify.rows or verify.gdn_lane_constants.rows != verify.rows:
        raise ValueError("lane constants were built for another row count")
    if verify.positions.lanes != verify.lanes or len(verify.pass_contexts) != verify.lanes:
        raise ValueError("lane positions or contexts do not match the lane count")
    if not verify.ple_rows.active:
        raise RuntimeError("lane PLE rows were released")


# --------------------------------------------------------------------------- host writes (outside any trace)


def _copy_host(mesh_device, host: torch.Tensor, dtype, layout, target) -> None:
    ttnn.copy_host_to_device_tensor(_host_row_tensor(mesh_device, host, dtype, layout), target)


def write_lane_accepted(model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, accepted: Sequence[int]):
    """Host write of the per-lane accept counts (``-1 .. R - 1``; -1 = a seeded lane that commits nothing)."""

    _validate_lane_verify_state(model, verify)
    values = [int(value) for value in accepted]
    if len(values) != verify.lanes or any(not -1 <= value < verify.rows for value in values):
        raise ValueError(f"lane accept counts need {verify.lanes} ints in [-1, {verify.rows}), got {accepted!r}")
    column = torch.tensor(values, dtype=torch.float32).reshape(1, verify.lanes, 1, 1)
    _copy_host(model.mesh_device, column, ttnn.float32, ttnn.TILE_LAYOUT, verify.accepted_lanes)
    _copy_host(
        model.mesh_device,
        _accepted_row_host(values, verify.lanes),
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        verify.accepted_row,
    )


def write_lane_active(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, active: Sequence[int], *, commit: Sequence[int]
):
    """Host write of this pass's active mask (0/1 per lane: an inactive lane's KV writes go to the scratch rows and
    its position does not advance) and of the commit mask (0/1 per lane: the active mask of the pass whose rows the
    next commit takes -- the previous pass in the fused form, this pass for the split form's commit body)."""

    _validate_lane_verify_state(model, verify)
    values = [int(value) for value in active]
    mask = _mask_row_host(values, verify.lanes)
    commits = [int(value) for value in commit]
    _mask_row_host(commits, verify.lanes)
    column = torch.tensor(values, dtype=torch.float32).reshape(1, verify.lanes, 1, 1)
    _copy_host(model.mesh_device, column, ttnn.float32, ttnn.TILE_LAYOUT, verify.active_lanes)
    _copy_host(model.mesh_device, mask.to(torch.uint32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, verify.active_row)
    _copy_host(model.mesh_device, (1 - mask).to(torch.uint32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, verify.inactive_row)
    commit_column = torch.tensor(commits, dtype=torch.float32).reshape(1, verify.lanes, 1, 1)
    _copy_host(model.mesh_device, commit_column, ttnn.float32, ttnn.TILE_LAYOUT, verify.commit_lanes)


def _lane_tokens(verify: Qwen38TTNNMTPLaneVerifyState, tokens_by_lane: Sequence[Sequence[int]]) -> list[list[int]]:
    lanes, rows = verify.lanes, verify.rows
    if len(tokens_by_lane) != lanes or any(len(tokens) != rows for tokens in tokens_by_lane):
        raise ValueError(f"lane verify inputs need {lanes} lanes of {rows} tokens, got {tokens_by_lane!r}")
    return [[int(token) for token in tokens] for tokens in tokens_by_lane]


def write_lane_tokens(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, tokens_by_lane: Sequence[Sequence[int]]
) -> None:
    """Host writes of one pass's token row (lane u's R tokens ``[t_u, d_1 .. d_k]`` in its block) and draft lanes
    (its k drafts at ``u*R .. u*R + k - 1``): the ``draft_inputs`` segment of a pass on recorded drafts."""

    _validate_lane_verify_state(model, verify)
    tokens_by_lane = _lane_tokens(verify, tokens_by_lane)
    rows = verify.rows
    token_row = model.model_io.embedding.host_verify_token_rows(
        [token for tokens in tokens_by_lane for token in tokens]
    )
    drafts = torch.full(TOKEN_ROW_SHAPE, float(ZERO_EMBEDDING_TOKEN))
    for lane, tokens in enumerate(tokens_by_lane):
        drafts[..., lane * rows : lane * rows + verify.drafts] = torch.tensor(tokens[1:], dtype=torch.float32)
    _copy_host(model.mesh_device, token_row, ttnn.float32, ttnn.TILE_LAYOUT, verify.token_row)
    _copy_host(model.mesh_device, drafts, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, verify.draft_lanes)


def write_lane_ple_rows(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, tokens_by_lane: Sequence[Sequence[int]]
) -> None:
    """Host writes of one pass's PLE rows: every lane's R n-gram rows looked up from its committed context in ONE
    table read (``host_rows_lanes``) and uploaded as one ``[1,1,B*R,640]`` per device; ``verify.pass_contexts[u]``
    takes lane u's R + 1 contexts (the ``ple_rows`` segment)."""

    _validate_lane_verify_state(model, verify)
    tokens_by_lane = _lane_tokens(verify, tokens_by_lane)
    ple, ple_state = model.layers[PLE_CHECKPOINT_LAYER].ple, verify.layers[PLE_CHECKPOINT_LAYER].ple
    host, lane_contexts = ple.host_rows_lanes(tokens_by_lane, ple_state.token_contexts)
    for lane, contexts in enumerate(lane_contexts):
        verify.pass_contexts[lane] = tuple(contexts)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            host.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(model.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
        ),
        verify.ple_rows.embedding_rows,
    )


def write_lane_verify_inputs(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, tokens_by_lane: Sequence[Sequence[int]]
) -> None:
    """Host writes of one pass's inputs: the token row and draft lanes, then the PLE rows (both segments)."""

    write_lane_tokens(model, verify, tokens_by_lane)
    write_lane_ple_rows(model, verify, tokens_by_lane)


def commit_lane_verify_host(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, committed: Sequence[int]
) -> None:
    """Host bookkeeping after a pass committed ``committed[u]`` rows per lane (``(a_u + 1) * active_u``): the
    checkpoint layer's per-lane n-gram context moves to ``pass_contexts[u][committed[u]]`` and the position mirror
    advances."""

    _validate_lane_verify_state(model, verify)
    counts = [int(count) for count in committed]
    if len(counts) != verify.lanes or any(not 0 <= count <= verify.rows for count in counts):
        raise ValueError(f"lane commits need {verify.lanes} counts in [0, {verify.rows}], got {committed!r}")
    ple_state = verify.layers[PLE_CHECKPOINT_LAYER].ple
    ple_state.token_contexts = tuple(verify.pass_contexts[lane][count] for lane, count in enumerate(counts))
    ple_state.validate()
    verify.positions.advance_mirror(counts)


# --------------------------------------------------------------------------- admission: the 1-lane image into lane u


def _slice0(tensor, index: int):
    """Entry ``index`` of a pack stacked on dim 0, owned (a 1-layer pack's only entry is a clone, not the pack)."""

    end = list(_shape(tensor))
    start = [0, 0, 0, 0]
    start[0], end[0] = index, index + 1
    return slice_owned(tensor, start, end)


def _fill_lane(target_view, part, lane: int, *, label: str) -> None:
    try:
        result = ttnn.fill_cache(target_view, part, batch_idx=lane)
    except Exception as error:
        raise RuntimeError(
            f"{label}: fill_cache into lane {lane} failed (target {tensor_metadata(target_view)}, part "
            f"{tensor_metadata(part)}, allocated {part.is_allocated()}): {error}"
        ) from error
    if result is not None and _tensor_key(result) != _tensor_key(target_view):
        raise RuntimeError(f"{label} did not land in lane {lane}")


def _history_tile(rows: Sequence[Any], *, width: int, label: str):
    """Three ``[1,1,1,W]`` TILE rows into a ``[1,1,32,W]`` tile (rows 3..31 zero): the concat, then ``ttnn.pad`` inside
    the concat's own tile, which returns a view of it on this runtime (the ``sync_*_history`` forms).  Returns the tile
    and the owner to release after the tile was consumed (never the view)."""

    combined = ttnn.concat(list(rows), dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _deallocate(*rows)
    if combined.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"{label} rows must be TILE (tilize the pack once), got {combined.layout}")
    padded = ttnn.pad(
        combined, [(0, 0), (0, 0), (0, CHUNK_ROWS - len(rows)), (0, 0)], 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    if _shape(padded) != (1, 1, CHUNK_ROWS, width):
        raise RuntimeError(f"{label} tile has shape {_shape(padded)}, expected [1,1,32,{width}]")
    return padded, combined


def load_lane_image(pager: Qwen38TTNNLanePager, slot: Qwen38LaneHostSlot) -> None:
    """The host slot's small families into the pager's device pack buffers (the KV slabs go per layer)."""

    for name, pack in pager.packs.items():
        ttnn.copy_host_to_device_tensor(slot.tensors[name][0], pack)


def import_gdn_lane(
    state, rows_state, pager: Qwen38TTNNLanePager, conv_tiles, index: int, *, phase: int, lane: int
) -> None:
    """Lane ``lane`` of a GDN layer from image ``index`` of the loaded packs: the recurrent state (``fill_cache`` at
    batch u through the pager's rows view) and the FIR history rows 0..2 <- the ring's three history slots at
    ``phase`` (``sync_rows_history_from_state`` landing in batch u).  ``conv_tiles`` is the conv pack tilized once
    (``[1, 4 * layers, 1, W]``: layer l's ring slot s at row 4l + s)."""

    part = _slice0(pager.packs["recurrent"], index)
    filled = ttnn.experimental.view(part, (1, 1, RECURRENT_ROWS, gdn_module.HEAD_DIM))
    lanes = state.batch_size
    _fill_lane(
        ttnn.experimental.view(state.recurrent, (lanes, 1, RECURRENT_ROWS, gdn_module.HEAD_DIM)),
        filled,
        lane,
        label=f"GDN layer image {index} recurrent",
    )
    _deallocate(part)
    width = _shape(rows_state.history)[3]
    kernel = gdn_module.CONV_KERNEL_SIZE
    rows = []
    for i in range(gdn_module.CONV_HISTORY_ROWS):
        row = kernel * index + (phase + 1 + i) % kernel
        rows.append(
            ttnn.slice(conv_tiles, (0, row, 0, 0), (1, row + 1, 1, width), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
    tile, owner = _history_tile(rows, width=width, label=f"GDN layer image {index} history")
    _fill_lane(
        ttnn.experimental.view(rows_state.history, (rows_state.lanes, 1, CHUNK_ROWS, width)),
        tile,
        lane,
        label=f"GDN layer image {index} history",
    )
    _deallocate(owner)


def write_kv_slab_lane(cache, staging, lane: int, allocated_context: int, *, label: str) -> None:
    """``staging`` (a lane's ``[1, 1, C, 2*HEAD_DIM]`` KV slab on device) into lane ``lane`` of the lane cache
    (``[1, 1, B*C + scratch, 2*HEAD_DIM]``) as ``update_padded_kv_cache`` writes of :func:`kv_scratch_rows` rows each
    at row ``u*C + start`` (the pager's re-admission write, chunked to the cache's block-cyclic invariant)."""

    chunk = kv_scratch_rows(allocated_context)
    for start in range(0, allocated_context, chunk):
        part = staging
        if chunk != allocated_context:
            part = ttnn.slice(
                staging, (0, 0, start, 0), (1, 1, start + chunk, KV_WIDTH), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        result = ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache, part, 0, 0, 1, lane * allocated_context + start, qsa_module.STAGING_AXIS
        )
        if result is not None and _tensor_key(result) != _tensor_key(cache):
            raise RuntimeError(f"{label}: KV slab rows {start}..{start + chunk} did not land in lane {lane}")
        if part is not staging:
            _deallocate(part)


def import_qsa_lane(
    state,
    verify_state,
    pager: Qwen38TTNNLanePager,
    slot: Qwen38LaneHostSlot,
    index: int,
    *,
    position: int,
    lane: int,
    allocated_context: int,
) -> None:
    """Lane ``lane`` of a QSA layer from image ``index``: the KV slab at row ``u*C`` (the pager's re-admission write
    through its staging), the compressed blocks (``fill_cache`` at batch u) and the raw history rows 0..2 <- ring slots
    ``(P - 3 .. P - 1) % 4`` (``sync_verify_raw_history_from_ring`` landing in batch u)."""

    staging = pager.kv_stagings[index % KV_STAGINGS]
    ttnn.copy_host_to_device_tensor(slot.tensors["kv"][index], staging)
    write_kv_slab_lane(state.packed_kv_cache, staging, lane, allocated_context, label=f"QSA layer image {index}")
    part = _slice0(pager.packs["compressed"], index)
    _fill_lane(state.compressed_index_cache, part, lane, label=f"QSA layer image {index} compressed blocks")
    _deallocate(part)
    ring = _slice0(pager.packs["ring"], index)  # [1,1,32,128]: slot s at row s
    rows = []
    for back in range(qsa_module.RAW_HISTORY_ROWS, 0, -1):  # positions P - 3, P - 2, P - 1
        slot_row = (position - back) % qsa_module.COMPRESS_RATIO
        rows.append(
            ttnn.slice(
                ring,
                (0, 0, slot_row, 0),
                (1, 1, slot_row + 1, qsa_module.INDEX_HEAD_DIM),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    _deallocate(ring)
    tile, owner = _history_tile(rows, width=qsa_module.INDEX_HEAD_DIM, label=f"QSA layer image {index} raw history")
    _fill_lane(
        ttnn.experimental.view(
            verify_state.raw_history, (verify_state.lanes, 1, CHUNK_ROWS, qsa_module.INDEX_HEAD_DIM)
        ),
        tile,
        lane,
        label=f"QSA layer image {index} raw history",
    )
    _deallocate(owner)


def import_ple_lane(lanes_state, pager: Qwen38TTNNLanePager, *, lane: int, context: tuple[int, int] | None) -> None:
    """Lane ``lane`` of the PLE lane rows state: the nine slots of the image into its history (``load_from_state``
    landing in batch u) and its n-gram context."""

    lanes = lanes_state.lanes
    history_view = ttnn.experimental.view(
        lanes_state.history, (lanes * CONV_STATE_LENGTH, 1, *tuple(_shape(lanes_state.history))[2:])
    )
    for slot_index in range(CONV_STATE_LENGTH):
        part = _slice0(pager.packs["ple"], slot_index)
        _fill_lane(history_view, part, lane * CONV_STATE_LENGTH + slot_index, label=f"PLE slot {slot_index}")
        _deallocate(part)
    contexts = list(lanes_state.token_contexts)
    contexts[lane] = None if context is None else tuple(int(value) for value in context)
    lanes_state.token_contexts = tuple(contexts)
    lanes_state.validate()


def import_lane_state(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNMTPLaneVerifyState,
    lane: int,
    *,
    position: int,
    ple_context: tuple[int, int] | None,
    gdn_phases: Sequence[int],
    backbone_slot: Qwen38LaneHostSlot,
    backbone_pager: Qwen38TTNNLanePager,
    alignment_slot: Qwen38LaneHostSlot,
    alignment_pager: Qwen38TTNNLanePager,
    rope_shift: int = 0,
) -> None:
    """Admit a prefilled sequence into lane ``lane`` from its 1-lane images: the pager's host slot of the generic
    state (``model.generic_lane_layout``) and of the alignment layer's generic state, evicted after the prompt's
    prefill at host-known ``position``; ``gdn_phases`` are the generic GDN states' ring phases at that moment (in
    GDN-layer order).  Device -> device through the pagers' pack buffers, every family landing in lane u alone; the
    phase-bound families are converted at P into the MTP-lane families (the B=1 seed forms of ``mtp_v2`` per lane).
    Then the lane's position, rotary shift (``rope_shift``: the prompt's ``mrope`` shift, 0 for text) and n-gram
    context.  Eager, never traced; the caller writes the lane's accept count (-1: the next commit keeps everything)
    and active flag, and warms every (lane) variant before the captures."""

    _validate_lane_verify_state(model, verify)
    if isinstance(lane, bool) or type(lane) is not int or not 0 <= lane < verify.lanes:
        raise ValueError(f"lane must be an int in [0, {verify.lanes}), got {lane!r}")
    if isinstance(position, bool) or type(position) is not int or position < 0:
        raise ValueError(f"position must be a non-negative int, got {position!r}")
    require_rope_shift(rope_shift, position)
    gdn_layers = [layer for layer in model.layers if isinstance(layer.attention, Qwen38TTNNGDN)]
    if len(gdn_phases) != len(gdn_layers):
        raise ValueError(f"gdn_phases needs one ring phase per GDN layer ({len(gdn_layers)}), got {len(gdn_phases)}")
    context = model.allocated_context
    load_lane_image(backbone_pager, backbone_slot)
    conv_tiles = ttnn.to_layout(backbone_pager.packs["conv"], ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gdn_index = qsa_index = 0
    for layer, layer_state in zip(model.layers, verify.layers):
        if isinstance(layer.attention, Qwen38TTNNGDN):
            import_gdn_lane(
                layer_state.attention_state,
                layer_state.attention_rows,
                backbone_pager,
                conv_tiles,
                gdn_index,
                phase=int(gdn_phases[gdn_index]),
                lane=lane,
            )
            gdn_index += 1
        else:
            import_qsa_lane(
                layer_state.attention_state,
                layer_state.attention_rows,
                backbone_pager,
                backbone_slot,
                qsa_index,
                position=position,
                lane=lane,
                allocated_context=context,
            )
            qsa_index += 1
        if layer_state.ple is not None:
            import_ple_lane(layer_state.ple, backbone_pager, lane=lane, context=ple_context)
    _deallocate(conv_tiles)
    load_lane_image(alignment_pager, alignment_slot)
    alignment = verify.alignment
    import_qsa_lane(
        alignment.layer_state.attention_state,
        alignment.layer_state.attention_rows,
        alignment_pager,
        alignment_slot,
        0,
        position=position,
        lane=lane,
        allocated_context=context,
    )
    verify.positions.write_lane(lane, position, rope_shift)
    verify.pass_contexts[lane] = (ple_context,) * (verify.rows + 1)


# --------------------------------------------------------------------------- the body


def _forward_layer_verify_lanes(
    layer: Qwen38TTNNDecoderLayer,
    residual,
    layer_state: Qwen38TTNNMTPLaneLayerState,
    verify: Qwen38TTNNMTPLaneVerifyState,
    *,
    prepared_ple_rows: Qwen38TTNNPLERowsPreparedInput | None,
    rope_rows,
    qsa_inputs: qsa_module.Qwen38TTNNQSALaneVerifyInputs,
    selectors: gdn_module.Qwen38TTNNRowsSelectorsLanes | None,
):
    """One layer of the lane verify on the lane-major 32-row tile (``mtp_v2._forward_layer_verify`` with the lane
    forms); the input rows are consumed.  Commit first (with ``selectors``: every lane's previous pass out of its own
    persistent buffers), then the new rows: the PLE lane rows on the ``B*R`` real rows, the GDN lane rows or the QSA
    lane verify, the MoE on its admitted row count; both slices come back to the tile through the in-place pad."""

    rows = verify.lanes * verify.rows
    if _shape(residual) != RESIDUAL_ROWS_SHAPE or residual.dtype != ttnn.bfloat16:
        raise RuntimeError(
            f"lane verify residual rows must be BF16 {RESIDUAL_ROWS_SHAPE}, got {tensor_metadata(residual)}"
        )
    dram = ttnn.DRAM_MEMORY_CONFIG
    if layer.ple is not None:
        if prepared_ple_rows is None:
            raise ValueError("the PLE layer's lane verify pass needs its prepared persistent PLE rows")
        if selectors is not None:
            layer.ple.commit_rows_lanes(layer_state.ple, selectors)
        head = ttnn.slice(residual, (0, 0, 0, 0), (1, RESIDUAL_BRANCHES, rows, LOCAL_HIDDEN_SIZE), memory_config=dram)
        head.update_tensor_topology(residual.tensor_topology())
        injected = layer.ple.inject_rows_lanes(head, prepared_ple_rows, layer_state.ple)  # consumes head
        _deallocate(residual)
        residual = _pad_rows(injected, rows, label="PLE-injected lane verify rows")
        residual_owner = injected
    else:
        if prepared_ple_rows is not None:
            raise ValueError("prepared PLE rows were supplied outside checkpoint layer 1")
        residual_owner = residual

    attention_input, attention_gr_state = layer.attention_gr.read_rows(residual, flat_views=True)
    if isinstance(layer.attention, Qwen38TTNNGDN):
        if selectors is not None:
            layer.attention.commit_rows_lanes(layer_state.attention_state, layer_state.attention_rows, selectors)
        result = layer.attention.forward_rows_lanes(
            attention_input, layer_state.attention_state, layer_state.attention_rows
        )
        _deallocate(result.final_state)
        attention_hidden = result.hidden_rows
    else:
        if selectors is not None:
            layer.attention.commit_verify_lanes(layer_state.attention_rows, selectors)
        attention_hidden = layer.attention.forward_verify_lanes(
            attention_input,
            layer_state.attention_state,
            layer_state.attention_rows,
            cos=rope_rows.cos,
            sin=rope_rows.sin,
            block_start_cos=rope_rows.block_start_cos,
            block_start_sin=rope_rows.block_start_sin,
            inputs=qsa_inputs,
            constants=verify.qsa_chunk_constants,
            lane_constants=verify.qsa_lane_constants,
            lane_verify=verify.qsa_lane_verify_constants,
        )
    _deallocate(attention_input)
    if _shape(attention_hidden) != BLOCK_ROWS_SHAPE:
        raise RuntimeError(
            f"lane verify attention output has shape {_shape(attention_hidden)}, expected {BLOCK_ROWS_SHAPE}"
        )
    residual = layer.attention_gr.write_rows(attention_hidden, attention_gr_state)
    _deallocate(attention_hidden, residual_owner, attention_gr_state.injection)

    mlp_input, mlp_gr_state = layer.mlp_gr.read_rows(residual, flat_views=True)
    if layer_state.moe_input is None:
        moe_input = mlp_input
    else:
        moe_rows = layer_state.moe.rows
        landed = ttnn.slice(
            mlp_input, (0, 0, 0, 0), (1, 1, moe_rows, LOCAL_HIDDEN_SIZE), output_tensor=layer_state.moe_input
        )
        if landed is not None and _tensor_key(landed) != _tensor_key(layer_state.moe_input):
            raise RuntimeError("lane verify MoE input slice did not land in its persistent buffer")
        moe_input = layer_state.moe_input
        moe_input.update_tensor_topology(mlp_input.tensor_topology())
    with layer.expert_streamer.layer(layer.layer_index, namespace=layer.namespace.value) as packed_experts:
        if not isinstance(packed_experts, tuple) or len(packed_experts) != 2:
            raise RuntimeError("BF4 streamer must yield exactly (packed_w0_w1, packed_w2)")
        mlp_result = layer_state.moe.forward(moe_input, packed_experts[0], packed_experts[1])
    _deallocate(mlp_input)
    moe_hidden = mlp_result.hidden_sharded
    if layer_state.moe_input is not None:
        moe_hidden = _pad_rows(moe_hidden, layer_state.moe.rows, label="lane verify MoE output rows")
    if _shape(moe_hidden) != BLOCK_ROWS_SHAPE:
        raise RuntimeError(f"lane verify MoE output has shape {_shape(moe_hidden)}, expected {BLOCK_ROWS_SHAPE}")
    residual = layer.mlp_gr.write_rows(moe_hidden, mlp_gr_state)
    _deallocate(mlp_result.hidden_sharded, mlp_gr_state.residual, mlp_gr_state.injection)
    if _shape(residual) != RESIDUAL_ROWS_SHAPE:
        raise RuntimeError(f"lane verify layer output has shape {_shape(residual)}, expected {RESIDUAL_ROWS_SHAPE}")
    return residual


def accept_rows_lanes(argmax_row, draft_lanes, k_eff_mask_row, constants: Qwen38TTNNMTPLaneConstants):
    """The per-lane accept counts from the 32 argmaxes and the draft lanes, exact integers (``mtp_v2.accept_rows``
    per lane): the match flags (the pad columns' flags are 1: sentinel == sentinel) times the per-lane draft mask,
    the running counts per lane through the block-diagonal prefix matmul, the prefix flags (0 on the pad columns by
    their count 64), summed per lane by the block sum whose pad columns are zero.  Returns the FP32 ROW_MAJOR
    ``[1,1,1,B]`` row; no id enters an FPU."""

    for name, row in (
        ("verify argmax row", argmax_row),
        ("draft lanes", draft_lanes),
        ("k_eff mask row", k_eff_mask_row),
    ):
        if _shape(row) != TOKEN_ROW_SHAPE or row.dtype != ttnn.float32 or row.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"{name} must be FP32 ROW_MAJOR {TOKEN_ROW_SHAPE}, got {tensor_metadata(row)}")
    dram = ttnn.DRAM_MEMORY_CONFIG
    flags = ttnn.eq(argmax_row, draft_lanes, dtype=ttnn.float32, memory_config=dram)
    masked = ttnn.multiply(flags, k_eff_mask_row, memory_config=dram)
    flags_tile = ttnn.to_layout(masked, ttnn.TILE_LAYOUT, memory_config=dram)
    _deallocate(flags, masked)
    running = ttnn.matmul(
        flags_tile, constants.prefix_upper_block, memory_config=dram, compute_kernel_config=constants.compute_config
    )
    _deallocate(flags_tile)
    prefix = ttnn.eq(running, constants.draft_plus_one_row, dtype=ttnn.float32, memory_config=dram)
    _deallocate(running)
    accepted_tile = ttnn.matmul(
        prefix, constants.block_sum, memory_config=dram, compute_kernel_config=constants.compute_config
    )
    _deallocate(prefix)
    if _shape(accepted_tile) != (1, 1, 1, constants.lanes) or accepted_tile.dtype != ttnn.float32:
        raise RuntimeError(
            f"lane accept counts must be FP32 [1,1,1,{constants.lanes}], got {tensor_metadata(accepted_tile)}"
        )
    accepted_row = ttnn.to_layout(accepted_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    _deallocate(accepted_tile)
    return accepted_row


def _lane_gather_index(verify: Qwen38TTNNMTPLaneVerifyState):
    """``u*R + a_u`` per lane as a UINT32 ROW_MAJOR ``[1,1,1,B]`` row from the persistent accept row (small exact
    fp32 integers; the typecast and the add move no id through an FPU)."""

    dram = ttnn.DRAM_MEMORY_CONFIG
    counts = ttnn.slice(verify.accepted_row, (0, 0, 0, 0), (1, 1, 1, verify.lanes), memory_config=dram)
    counts_u32 = ttnn.typecast(counts, ttnn.uint32, memory_config=dram)
    index = ttnn.add(counts_u32, verify.constants.lane_base_row, memory_config=dram)
    _deallocate(counts, counts_u32)
    return index


def _select_lane_residual_rows(verify: Qwen38TTNNMTPLaneVerifyState, residual) -> None:
    """``alignment.residual[:, :, u] <- residual[:, :, u*R + a_u]`` for every lane: the accept counts gathered per
    column against the column's draft index give the one-hot row (one 1 per lane, 0 on the pad columns), times the
    block-sum lane rows the per-lane select ``[1,1,B,32]``, repeated per branch and applied as one equal-batch
    matmul whose rows are copied into the persistent buffer.  Exact: every output element is one bf16 value times 1.0 plus exact
    zeros under the fp32-accumulating HiFi4 config."""

    constants = verify.constants
    dram = ttnn.DRAM_MEMORY_CONFIG
    matched = ttnn.gather(verify.accepted_row, 3, verify.qsa_lane_verify_constants.lane_of_row, memory_config=dram)
    matched_tile = ttnn.to_layout(matched, ttnn.TILE_LAYOUT, memory_config=dram)
    onehot_row = ttnn.eq(constants.draft_match_row, matched_tile, dtype=ttnn.bfloat16, memory_config=dram)
    _deallocate(matched, matched_tile)
    select = ttnn.multiply(constants.block_sum_t, onehot_row, memory_config=dram)
    _deallocate(onehot_row)
    select_branches = ttnn.repeat_interleave(select, repeats=RESIDUAL_BRANCHES, dim=1, memory_config=dram)
    _deallocate(select)
    if _shape(select_branches) != (1, RESIDUAL_BRANCHES, verify.lanes, CHUNK_ROWS):
        raise RuntimeError(
            f"lane residual select has shape {_shape(select_branches)}, expected [1,4,{verify.lanes},32]"
        )
    # The matmul's own output, then the copy into the persistent rows (the B=1 select's form): at M = B rows the
    # matmul's ``optional_output_tensor`` came back under another tensor id on the pinned runtime (warm-up hold 2).
    selected = ttnn.matmul(
        select_branches, residual, memory_config=dram, compute_kernel_config=constants.compute_config
    )
    target = verify.alignment.residual
    if _shape(selected) != _shape(target):
        raise RuntimeError(f"selected lane residual rows have shape {_shape(selected)}, expected {_shape(target)}")
    landed = ttnn.copy(selected, target)
    if landed is not None and _tensor_key(landed) != _tensor_key(target):
        raise RuntimeError("selected lane residual rows did not land in their persistent buffer")
    _deallocate(select_branches, selected)


def _forward_alignment_lanes(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNMTPLaneVerifyState,
    roots,
    argmax_row,
    gather_index,
    *,
    rope_rows,
    qsa_inputs,
    selectors,
):
    """The alignment rows of every lane at ``P_u .. P_u + k`` (``mtp_v2._forward_alignment`` on the lane tile): row
    ``u*R + j`` takes lane u's root at ``P_u + j`` and the target's own prediction there (``argmax_{u*R + j}``)
    through the MTP layer's lane state; returns ``d_1'`` per lane (the alignment argmax at row ``u*R + a_u``, a
    32-bit gather) with the whole alignment argmax row, and lands row ``u*R + a_u`` of the alignment residuals per
    lane in ``alignment.residual``.  The pad rows carry the sentinel token and never reach a cache."""

    alignment = verify.alignment
    dram = ttnn.DRAM_MEMORY_CONFIG
    shifted_tile = ttnn.to_layout(argmax_row, ttnn.TILE_LAYOUT, memory_config=dram)
    embedding_rows = model.model_io.embedding.embed_device_token_rows(shifted_tile)
    _deallocate(shifted_tile)
    mixed = alignment.input_mixer.rows(embedding_rows, roots)
    _deallocate(embedding_rows)
    residual = _forward_layer_verify_lanes(
        alignment.layer,
        mixed,
        alignment.layer_state,
        verify,
        prepared_ple_rows=None,
        rope_rows=rope_rows,
        qsa_inputs=qsa_inputs,
        selectors=selectors,
    )
    hidden = alignment.final_mixer.rows(residual, flat_views=True)
    lanes_row = _resolve_rows(
        model, hidden, rows=verify.lanes * verify.rows, sentinel_tail=verify.constants.sentinel_tail
    )
    _deallocate(hidden)
    first_draft = ttnn.gather(lanes_row, 3, gather_index, memory_config=dram)
    _select_lane_residual_rows(verify, residual)
    _deallocate(residual)
    return first_draft, lanes_row


def _land_accept_rows(verify: Qwen38TTNNMTPLaneVerifyState, accepted_row) -> None:
    """The device accept counts into the two persistent forms the commit selects, the gather index and the position
    advance read: the ``[1,1,1,32]`` ROW_MAJOR row (pad lanes keep -1) and the ``[1,B,1,1]`` TILE counts."""

    dram = ttnn.DRAM_MEMORY_CONFIG
    tail = ttnn.slice(verify.accepted_row, (0, 0, 0, verify.lanes), (1, 1, 1, CHUNK_ROWS), memory_config=dram)
    padded = ttnn.concat([accepted_row, tail], dim=3, memory_config=dram)
    landed = ttnn.copy(padded, verify.accepted_row)
    if landed is not None and _tensor_key(landed) != _tensor_key(verify.accepted_row):
        raise RuntimeError("lane accept row did not land in its persistent buffer")
    column = ttnn.reshape(accepted_row, (1, verify.lanes, 1, 1))
    column_tile = ttnn.to_layout(column, ttnn.TILE_LAYOUT, memory_config=dram)
    landed = ttnn.copy(column_tile, verify.accepted_lanes)
    if landed is not None and _tensor_key(landed) != _tensor_key(verify.accepted_lanes):
        raise RuntimeError("lane accept counts did not land in their persistent buffer")
    _deallocate(tail, padded, column, column_tile)


def _advance_lane_positions(verify: Qwen38TTNNMTPLaneVerifyState) -> None:
    """``P_u <- P_u + (a_u + 1) * active_u``: exact UINT32 arithmetic into the resident row (the pad lanes' -1
    counts give 0)."""

    dram = ttnn.DRAM_MEMORY_CONFIG
    row = verify.positions.row
    plus_one = ttnn.add(verify.accepted_row, 1.0, memory_config=dram)
    increments = ttnn.typecast(plus_one, ttnn.uint32, memory_config=dram)
    active_increments = ttnn.multiply(increments, verify.active_row, memory_config=dram)
    key = _tensor_key(row)
    advanced = ttnn.add(row, active_increments, memory_config=dram)
    copied = ttnn.copy(advanced, row)
    if _tensor_key(row) != key or (copied is not None and _tensor_key(copied) != key):
        raise RuntimeError("lane verify position advance did not write the resident row in place")
    _deallocate(plus_one, increments, active_increments, advanced)


def forward_verify_lanes(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNMTPLaneVerifyState,
    *,
    catch_up: bool,
    land_accept: bool = False,
    observer: Callable[[str], Any] | None = None,
    hidden_observer: Callable[[Any], Any] | None = None,
) -> Qwen38TTNNVerifyOutput:
    """One lane verify pass at the lanes' positions: fixed op sequence, fixed shapes, no host ints, no host I/O.

    ``catch_up=True`` commits every lane's previous pass first (``c_u = (a_u + 1) * commit_u`` rows out of the
    persistent buffers, at the start of each layer; ``commit_u`` = that pass's active flag); the first pass after a
    seed runs with ``catch_up=False``.  The accept counts are computed and read back; with ``land_accept`` they also
    replace the per-lane accept rows the next commit, the gather index and the position advance read (stage 1
    leaves them host-pinned at ``R - 1``).  ``P_u <- P_u + (a_u + 1) * active_u`` is the body's last op.  Any
    failure poisons the model owner.  ``observer`` (eager diagnostics only) is called with a stage label after the
    prologue, every layer, the head, the accept, the alignment and the epilogue; ``hidden_observer`` (eager only)
    with the final mixer's ``[1,1,32,640]`` rows before the head.
    """

    _validate_lane_verify_state(model, verify)
    if not isinstance(catch_up, bool) or not isinstance(land_accept, bool):
        raise TypeError("catch_up and land_accept must be bools")
    if model.poisoned:
        raise RuntimeError("the text model is poisoned")
    dram = ttnn.DRAM_MEMORY_CONFIG
    lane_verify = verify.qsa_lane_verify_constants
    positions = verify.positions
    processed_layers = 0

    def stage(name: str) -> None:
        if observer is not None:
            observer(f"verify_lanes:{name}")

    try:
        selectors = (
            gdn_module.build_rows_selectors_lanes(verify.accepted_lanes, verify.commit_lanes, verify.gdn_lane_constants)
            if catch_up
            else None
        )
        # Row r's position P_{lane_of(r)} + draft_of(r) and the pooled compressed tile's block starts (rows 2u + i =
        # (P_u & ~3) + 4i): 32-bit gathers and UINT32 adds on the resident row, then the RoPE lookups.
        lane_rows = ttnn.gather(positions.row, 3, lane_verify.lane_of_row, memory_config=dram)
        row_positions = ttnn.add(lane_rows, lane_verify.draft_of_row, memory_config=dram)
        block_start_lanes = ttnn.bitwise_and(positions.row, positions.block_start_mask_row, memory_config=dram)
        block_lane_starts = ttnn.gather(block_start_lanes, 3, lane_verify.block_lane_of_row, memory_config=dram)
        block_start_rows = ttnn.add(block_lane_starts, lane_verify.block_offset_row, memory_config=dram)
        # The RoPE rows read the table at the rows' positions less their lanes' rotary shifts (0 for text).
        rope_rows, rope_block_rows = positions.rope_rows(
            row_positions, block_start_rows, lane_verify.lane_of_row, lane_verify.block_lane_of_row
        )
        rope = model.rope_table.rows_chunk(rope_rows, rope_block_rows)
        _deallocate(lane_rows, block_start_lanes, block_lane_starts, block_start_rows, rope_rows, rope_block_rows)
        qsa_inputs = qsa_module.derive_qsa_lane_verify_inputs(
            positions.row,
            verify.active_row,
            verify.inactive_row,
            row_positions,
            model.qsa_position_constants,
            verify.qsa_chunk_constants,
            verify.qsa_verify_constants,
            verify.qsa_lane_constants,
            lane_verify,
        )
        _deallocate(row_positions)
        residual = _embed_rows(model, verify.token_row)
        stage("prologue")
        for layer_index in range(BACKBONE_LAYERS):
            residual = _forward_layer_verify_lanes(
                model.layers[layer_index],
                residual,
                verify.layers[layer_index],
                verify,
                prepared_ple_rows=verify.ple_rows if layer_index == PLE_CHECKPOINT_LAYER else None,
                rope_rows=rope,
                qsa_inputs=qsa_inputs,
                selectors=selectors,
            )
            processed_layers += 1
            stage(f"layer-{layer_index}")
        hidden = model.final_mixer.rows(residual, flat_views=True)
        if hidden_observer is not None:
            hidden_observer(hidden)
        argmax_row = _resolve_rows(
            model, hidden, rows=verify.lanes * verify.rows, sentinel_tail=verify.constants.sentinel_tail
        )
        _deallocate(hidden)
        stage("head")
        accepted_row = accept_rows_lanes(argmax_row, verify.draft_lanes, verify.k_eff_mask_row, verify.constants)
        if land_accept:
            _land_accept_rows(verify, accepted_row)
        stage("accept")
        gather_index = _lane_gather_index(verify)
        next_token = ttnn.gather(argmax_row, 3, gather_index, memory_config=dram)
        first_draft, alignment_row = _forward_alignment_lanes(
            model,
            verify,
            residual,
            argmax_row,
            gather_index,
            rope_rows=rope,
            qsa_inputs=qsa_inputs,
            selectors=selectors,
        )
        _deallocate(gather_index, residual)
        stage("alignment")
        readback = ttnn.concat(
            [accepted_row, next_token, first_draft, argmax_row, alignment_row], dim=3, memory_config=dram
        )
        width = len(READBACK_FIXED_LANES) * verify.lanes + 2 * CHUNK_ROWS
        if _shape(readback) != (1, 1, 1, width) or readback.dtype != ttnn.float32:
            raise RuntimeError(f"lane verify readback must be FP32 [1,1,1,{width}], got {tensor_metadata(readback)}")
        _deallocate(accepted_row, next_token, first_draft, argmax_row, alignment_row)
        _advance_lane_positions(verify)  # the body's last device op
        qsa_inputs.deallocate()
        rope.deallocate()
        if selectors is not None:
            selectors.deallocate()
        stage("epilogue")
        return Qwen38TTNNVerifyOutput(readback)
    except BaseException as error:
        model._mark_poisoned("forward_verify_lanes", processed_layers, error)


def forward_commit_lanes(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, *, observer: Callable[[str], Any] | None = None
) -> None:
    """The commits of one pass as their own body (the split form, ``mtp_v2.forward_commit``): every layer's PLE /
    GDN / QSA lane commit of the last pass's rows per lane (``(a_u + 1) * commit_u`` of them), then the alignment
    layer's raw-history commit.  No host tensor, no position change."""

    _validate_lane_verify_state(model, verify)
    if model.poisoned:
        raise RuntimeError("the text model is poisoned")
    processed_layers = 0

    def stage(name: str) -> None:
        if observer is not None:
            observer(f"commit_lanes:{name}")

    try:
        selectors = gdn_module.build_rows_selectors_lanes(
            verify.accepted_lanes, verify.commit_lanes, verify.gdn_lane_constants
        )
        stage("selectors")
        for layer, layer_state in zip(model.layers, verify.layers):
            if layer.ple is not None:
                layer.ple.commit_rows_lanes(layer_state.ple, selectors)
            if isinstance(layer.attention, Qwen38TTNNGDN):
                layer.attention.commit_rows_lanes(layer_state.attention_state, layer_state.attention_rows, selectors)
            else:
                layer.attention.commit_verify_lanes(layer_state.attention_rows, selectors)
            processed_layers += 1
            stage(f"layer-{layer.layer_index}")
        alignment = verify.alignment
        alignment.layer.attention.commit_verify_lanes(alignment.layer_state.attention_rows, selectors)
        stage("alignment")
        selectors.deallocate()
    except BaseException as error:
        model._mark_poisoned("forward_commit_lanes", processed_layers, error)


def capture_verify_lanes(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNMTPLaneVerifyState,
    *,
    catch_up: bool,
    guard: Callable[[str], AbstractContextManager[Any]],
    land_accept: bool = False,
    cq_id: int = 0,
) -> tuple[int, Qwen38TTNNVerifyOutput]:
    """Capture one lane verify body (the model's capture discipline: ``corruptible_allocation_scope`` and the
    caller's no-host-I/O ``guard``); returns the trace id and the output whose readback address every replay
    rewrites.  Capture records without executing, so the device state is unchanged."""

    with corruptible_allocation_scope(model.mesh_device):
        trace_id = ttnn.begin_trace_capture(model.mesh_device, cq_id=cq_id)
        with guard(f"lane verify capture catch_up={catch_up}"):
            output = forward_verify_lanes(model, verify, catch_up=catch_up, land_accept=land_accept)
        ttnn.end_trace_capture(model.mesh_device, trace_id, cq_id=cq_id)
    return trace_id, output


def capture_commit_lanes(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNMTPLaneVerifyState,
    *,
    guard: Callable[[str], AbstractContextManager[Any]],
    cq_id: int = 0,
) -> int:
    with corruptible_allocation_scope(model.mesh_device):
        trace_id = ttnn.begin_trace_capture(model.mesh_device, cq_id=cq_id)
        with guard("lane commit capture"):
            forward_commit_lanes(model, verify)
        ttnn.end_trace_capture(model.mesh_device, trace_id, cq_id=cq_id)
    return trace_id


# --------------------------------------------------------------------------- the readback


@dataclass(frozen=True)
class Qwen38TTNNLaneVerifyReadback:
    """One pass's readback per lane: ``accepted[u]``, ``next_token[u]`` (= argmax ``u*R + a_u`` at the accept row
    the body read), ``first_draft[u]`` (the alignment argmax there), ``argmaxes[u]`` (lane u's R per-row argmaxes) and
    ``alignment_argmaxes[u]`` (lane u's R alignment argmaxes: row j = the draft after the token at ``P_u + j + 1``)."""

    accepted: tuple[int, ...]
    next_token: tuple[int, ...]
    first_draft: tuple[int, ...]
    argmaxes: tuple[tuple[int, ...], ...]
    alignment_argmaxes: tuple[tuple[int, ...], ...]


def parse_lane_readback(values: torch.Tensor, *, lanes: int, rows: int) -> Qwen38TTNNLaneVerifyReadback:
    width = len(READBACK_FIXED_LANES) * lanes + 2 * CHUNK_ROWS
    if values.numel() != width:
        raise RuntimeError(f"lane verify readback has {values.numel()} lanes, expected {width}")
    ints = [int(value) for value in values.reshape(-1).tolist()]
    fixed = [tuple(ints[index * lanes : (index + 1) * lanes]) for index in range(len(READBACK_FIXED_LANES))]
    rows_start = len(READBACK_FIXED_LANES) * lanes
    argmax = ints[rows_start : rows_start + CHUNK_ROWS]
    alignment = ints[rows_start + CHUNK_ROWS :]
    per_lane = lambda row: tuple(tuple(row[lane * rows : (lane + 1) * rows]) for lane in range(lanes))  # noqa: E731
    return Qwen38TTNNLaneVerifyReadback(
        accepted=fixed[0],
        next_token=fixed[1],
        first_draft=fixed[2],
        argmaxes=per_lane(argmax),
        alignment_argmaxes=per_lane(alignment),
    )


def read_lane_verify_output(output: Qwen38TTNNVerifyOutput, *, lanes: int, rows: int) -> Qwen38TTNNLaneVerifyReadback:
    """Host readback (outside any trace) of the lane verify row from coordinate 0."""

    if not output.active:
        raise RuntimeError("lane verify output tensors were released")
    return parse_lane_readback(ttnn.to_torch(ttnn.get_device_tensors(output.readback)[0]), lanes=lanes, rows=rows)


def verify_pass_fits_lanes(positions: Sequence[int], active: Sequence[int], allocated_context: int) -> bool:
    """Every active lane's verify pass writes its KV block at ``P_u & ~31`` and the next one; both must exist."""

    return all(
        not flag or mtp_v2.verify_pass_fits(int(position), allocated_context)
        for position, flag in zip(positions, active)
    )


# --------------------------------------------------------------------------- the draft body (k - 1 rows per lane)
# forward_draft_lanes is mtp_v2.forward_draft on the lane tile: after a lane verify landed every lane's accept count,
# the MTP layer runs k - 1 more rows per lane -- row u of a B-row tile is lane u's next draft at MTP position P_u' +
# i (P_u' the position the verify advanced to) -- on the alignment layer's lane caches with the draft's own raw
# history per lane, the rows-B MoE, the final mixer and the LM head; the resolved ids chain on the device (lane u's
# id at lane u of the resolved row is the next row's token), and the next verify pass's token row / draft lanes and
# the pass row are assembled by slices and concats (no id through an FPU, no host tensor).

# Placeholder drafts of the bootstrap pass (the chain's ``[t_P, 0 .. 0]``; any exact ids would do).
BOOTSTRAP_DRAFT_TOKEN = 0


def lane_readback_width(lanes: int) -> int:
    """Lanes of the lane verify readback row: ``[accepted(B), next_token(B), first_draft(B), argmaxes(32),
    alignment(32)]``."""

    return len(READBACK_FIXED_LANES) * lanes + 2 * CHUNK_ROWS


def lane_pass_row_width(lanes: int) -> int:
    """The pass row: the readback row, then the assembled token lanes of the next pass."""

    return lane_readback_width(lanes) + CHUNK_ROWS


@dataclass
class Qwen38TTNNMTPLaneDraftState:
    """Fixed-address buffers of the lane draft body: the MTP layer at one real row per lane (row u = lane u) on the
    lane verify path's 32-row operands.  ``qsa_state`` is the MTP layer's DRAFT raw history / raw rows per lane (the
    alignment lane state keeps its windows for the pass's real commit); ``qsa_verify_constants`` and
    ``qsa_lane_verify_constants`` the rows = 1 forms, ``qsa_chunk_constants`` / ``qsa_lane_constants`` the verify
    state's (the geometry :func:`_forward_layer_verify_lanes` reads: ``lanes``, ``rows``, the four constants);
    ``layer_state`` runs the alignment layer's lane QSA state (the shared caches) through the draft raw rows and the
    layer's own rows-B MoE; ``advance_selectors`` the constant a = 0 lane selectors (one committed row per lane: the
    previous draft row joins its history); ``step_offset_rows[i]`` UINT32 ROW_MAJOR ``[1,1,1,32]`` (lane u = i, the
    pad lanes 0: draft step i runs at ``P_u' + i``); ``sentinel_tail`` FP32 ROW_MAJOR ``[1,1,1,32-B]`` and
    ``sentinel_lane`` ``[1,1,1,1]`` of ``ZERO_EMBEDDING_TOKEN``; ``pass_row`` FP32 ROW_MAJOR ``[1,1,1,W]`` = the lane
    verify readback row this draft followed, then the next verify pass's token lanes (device-written: the pass's one
    host readback)."""

    lanes: int
    drafts: int
    rows: int
    qsa_chunk_constants: qsa_module.Qwen38TTNNQSAChunkConstants
    qsa_verify_constants: qsa_module.Qwen38TTNNQSAVerifyConstants
    qsa_lane_constants: qsa_module.Qwen38TTNNQSALaneConstants
    qsa_lane_verify_constants: qsa_module.Qwen38TTNNQSALaneVerifyConstants
    qsa_state: Any
    layer_state: Qwen38TTNNMTPLaneLayerState
    zero_accepted_lanes: Any
    ones_lanes: Any
    advance_selectors: gdn_module.Qwen38TTNNRowsSelectorsLanes
    step_offset_rows: tuple[Any, ...]
    sentinel_tail: Any
    sentinel_lane: Any
    pass_row: Any
    _owner: object = field(repr=False, compare=False)


def _validate_lane_draft_state(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, draft: Qwen38TTNNMTPLaneDraftState
) -> None:
    _validate_lane_verify_state(model, verify)
    if not isinstance(draft, Qwen38TTNNMTPLaneDraftState) or draft._owner is not model._state_owner:
        raise ValueError("lane draft state was not allocated by this model owner")
    if draft.lanes != verify.lanes or draft.drafts != verify.drafts or draft.rows != 1:
        raise ValueError(
            f"lane draft state ({draft.lanes} lanes, k={draft.drafts}, rows {draft.rows}) does not match the verify "
            f"state ({verify.lanes} lanes, k={verify.drafts})"
        )
    if draft.qsa_verify_constants.rows != 1 or draft.qsa_lane_verify_constants.rows != 1:
        raise ValueError("lane draft constants must be the rows = 1 forms")
    if (
        draft.qsa_chunk_constants is not verify.qsa_chunk_constants
        or draft.qsa_lane_constants is not verify.qsa_lane_constants
    ):
        raise ValueError("lane draft state must share the verify state's chunk and lane constants")
    alignment = verify.alignment
    if draft.layer_state.attention_state is not alignment.layer_state.attention_state:
        raise ValueError("lane draft rows must run on the alignment layer's lane QSA state (the shared caches)")
    if draft.layer_state.attention_rows is not draft.qsa_state or draft.layer_state.ple is not None:
        raise ValueError("lane draft layer state must run the draft QSA lane state through the MTP layer, no PLE")
    if len(draft.step_offset_rows) != draft.drafts - 1:
        raise ValueError(
            f"lane draft state needs {draft.drafts - 1} step offset rows, got {len(draft.step_offset_rows)}"
        )
    for name, tensor, shape in (
        ("lane draft sentinel tail", draft.sentinel_tail, (1, 1, 1, CHUNK_ROWS - draft.lanes)),
        ("lane draft sentinel lane", draft.sentinel_lane, (1, 1, 1, 1)),
        ("lane draft pass row", draft.pass_row, (1, 1, 1, lane_pass_row_width(draft.lanes))),
    ):
        if _shape(tensor) != shape or tensor.dtype != ttnn.float32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"{name} must be FP32 ROW_MAJOR {shape}, got {tensor_metadata(tensor)}")


def allocate_lane_draft_state(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState
) -> Qwen38TTNNMTPLaneDraftState:
    """Allocate the lane draft body's buffers beside ``verify`` (before any capture): the rows = 1 QSA verify and
    lane verify constants, the draft raw history per lane, the MTP layer's rows-B MoE instance, the a = 0 lane
    selectors, the step offset rows, the sentinels and the pass row."""

    _validate_lane_verify_state(model, verify)
    alignment = verify.alignment
    layer = alignment.layer
    qsa = layer.attention
    mesh_device, mesh_contract = model.mesh_device, model.mesh_contract
    lanes, drafts = verify.lanes, verify.drafts
    actions: list[tuple[str, Callable[[], Any]]] = []

    def keep(label: str, value, release: Callable[[], Any]):
        actions.append((label, release))
        return value

    def upload(values: torch.Tensor, dtype, layout, label: str):
        tensor = _upload_replicated(mesh_device, mesh_contract, values, dtype, layout, label=label)
        return keep(label, tensor, lambda: _deallocate(tensor))

    try:
        verify_constants = qsa_module.Qwen38TTNNQSAVerifyConstants.build(
            mesh_device, mesh_contract, verify.qsa_chunk_constants, rows=1
        )
        keep("lane draft QSA verify constants", verify_constants, verify_constants.deallocate)
        lane_verify_constants = qsa_module.Qwen38TTNNQSALaneVerifyConstants.build(
            mesh_device, mesh_contract, lanes=lanes, rows=1, allocated_context=model.allocated_context
        )
        keep("lane draft QSA lane verify constants", lane_verify_constants, lane_verify_constants.deallocate)
        qsa_state = qsa.allocate_lane_verify_state(lanes)
        keep("lane draft QSA lane state", qsa_state, lambda: qsa.release_lane_verify_state(qsa_state))
        moe_rows = resolve_moe_rows(lanes, lanes)  # the exact rows-B instance (the lane rule)
        moe = Qwen38TTNNMoE(
            layer.mlp.mesh_device,
            layer.mlp.mesh_contract,
            layer.mlp.weights,
            tt_ccl=layer.mlp.tt_ccl,
            collective_topology=layer.mlp.collective_topology,
            rows=moe_rows,
            synchronization_policy=layer.mlp.synchronization_policy,
            admitted_rows=SUPPORTED_ROWS if moe_rows in SUPPORTED_ROWS else (moe_rows,),
        )
        keep("lane draft MoE buffers", moe, moe.release_owned_buffers)
        moe_input = _allocate_hidden_sharded_zeros(
            mesh_device, mesh_contract, (1, 1, moe_rows, LOCAL_HIDDEN_SIZE), label="lane draft MoE input"
        )
        keep("lane draft MoE input", moe_input, lambda: _deallocate(moe_input))
        layer_state = Qwen38TTNNMTPLaneLayerState(
            layer.namespace, layer.layer_index, alignment.layer_state.attention_state, qsa_state, None, moe, moe_input
        )
        tile, row_major = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
        zero_accepted = upload(torch.zeros(1, lanes, 1, 1), ttnn.float32, tile, "lane draft zero accept counts")
        ones = upload(torch.ones(1, lanes, 1, 1), ttnn.float32, tile, "lane draft commit mask")
        advance_selectors = gdn_module.build_rows_selectors_lanes(zero_accepted, ones, verify.gdn_lane_constants)
        keep("lane draft advance selectors", advance_selectors, advance_selectors.deallocate)
        step_offset_rows = tuple(
            upload(
                _uint32_row([step] * lanes + [0] * (CHUNK_ROWS - lanes)),
                ttnn.uint32,
                row_major,
                f"lane draft step {step} offsets",
            )
            for step in range(drafts - 1)
        )
        sentinel_tail = upload(
            torch.full((1, 1, 1, CHUNK_ROWS - lanes), float(ZERO_EMBEDDING_TOKEN)),
            ttnn.float32,
            row_major,
            "lane draft sentinel tail",
        )
        sentinel_lane = upload(
            torch.full((1, 1, 1, 1), float(ZERO_EMBEDDING_TOKEN)), ttnn.float32, row_major, "lane draft sentinel lane"
        )
        pass_row = upload(
            torch.full((1, 1, 1, lane_pass_row_width(lanes)), float(ZERO_EMBEDDING_TOKEN)),
            ttnn.float32,
            row_major,
            "lane draft pass row",
        )
        draft = Qwen38TTNNMTPLaneDraftState(
            lanes=lanes,
            drafts=drafts,
            rows=1,
            qsa_chunk_constants=verify.qsa_chunk_constants,
            qsa_verify_constants=verify_constants,
            qsa_lane_constants=verify.qsa_lane_constants,
            qsa_lane_verify_constants=lane_verify_constants,
            qsa_state=qsa_state,
            layer_state=layer_state,
            zero_accepted_lanes=zero_accepted,
            ones_lanes=ones,
            advance_selectors=advance_selectors,
            step_offset_rows=step_offset_rows,
            sentinel_tail=sentinel_tail,
            sentinel_lane=sentinel_lane,
            pass_row=pass_row,
            _owner=model._state_owner,
        )
        _validate_lane_draft_state(model, verify, draft)
        return draft
    except BaseException as error:
        _run_cleanup("lane draft state allocation", list(reversed(actions)), primary=error)
        raise


def release_lane_draft_state(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, draft: Qwen38TTNNMTPLaneDraftState
) -> None:
    """Release the lane draft buffers; the alignment layer's lane caches and its MoE weights are not owned here."""

    _validate_lane_draft_state(model, verify, draft)
    qsa = verify.alignment.layer.attention
    _run_cleanup(
        "lane draft state",
        [
            ("lane draft pass row", lambda: _deallocate(draft.pass_row)),
            ("lane draft sentinels", lambda: _deallocate(draft.sentinel_tail, draft.sentinel_lane)),
            ("lane draft step offsets", lambda: _deallocate(*draft.step_offset_rows)),
            ("lane draft advance selectors", draft.advance_selectors.deallocate),
            ("lane draft masks", lambda: _deallocate(draft.zero_accepted_lanes, draft.ones_lanes)),
            ("lane draft MoE input", lambda: _deallocate(draft.layer_state.moe_input)),
            ("lane draft MoE buffers", draft.layer_state.moe.release_owned_buffers),
            ("lane draft QSA lane state", lambda: qsa.release_lane_verify_state(draft.qsa_state)),
            ("lane draft QSA lane verify constants", draft.qsa_lane_verify_constants.deallocate),
            ("lane draft QSA verify constants", draft.qsa_verify_constants.deallocate),
        ],
    )


def forward_draft_history_lanes(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, draft: Qwen38TTNNMTPLaneDraftState
) -> None:
    """The lane draft body's first op: every lane's draft raw history <- its alignment window selected with this
    pass's accept count and active flag (``c_u = (a_u + 1) * active_u``; the alignment state is read only)."""

    _validate_lane_draft_state(model, verify, draft)
    if model.poisoned:
        raise RuntimeError("the text model is poisoned")
    alignment = verify.alignment
    try:
        selectors = gdn_module.build_rows_selectors_lanes(
            verify.accepted_lanes, verify.active_lanes, verify.gdn_lane_constants
        )
        alignment.layer.attention.commit_verify_lanes(
            alignment.layer_state.attention_rows, selectors, target=draft.qsa_state
        )
        selectors.deallocate()
    except BaseException as error:
        model._mark_poisoned("forward_draft_history_lanes", 0, error)


def forward_draft_lanes(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNMTPLaneVerifyState,
    draft: Qwen38TTNNMTPLaneDraftState,
    verify_output: Qwen38TTNNVerifyOutput,
    *,
    derive_history: bool = True,
    observer: Callable[[str], Any] | None = None,
) -> None:
    """The k - 1 draft rows of every lane after a lane verify (``P_u'`` = the position it advanced to).

    Step i (i = 0 .. k - 2) embeds lane u's current draft (``d_{u,1}`` = the readback's first draft, then the
    previous step's resolved id) at row u of a B-row tile, mixes it with the lane's alignment residual (the row
    the verify selected at its accept), runs the MTP QSA layer at MTP position ``P_u' + i`` on the lane verify path
    with one real row per lane (KV row and compressed block written by absolute position into the alignment layer's
    lane caches, the draft raw history per lane, advanced by one committed row between steps; an inactive lane's KV
    write is redirected like the verify's), the rows-B MoE, the final mixer, the LM head and the on-device resolve,
    and emits ``d_{u,i+2}``.  Before the rows the draft raw histories are derived from the alignment windows with
    this pass's accept counts (``derive_history``).  Afterwards the next verify pass's inputs are assembled on
    device: ``verify.token_row`` = ``[t'_u, d_{u,1} .. d_{u,k}]`` per lane block then the sentinel,
    ``verify.draft_lanes`` = ``[d_{u,1} .. d_{u,k}, -1]`` per lane block then the sentinel, and ``draft.pass_row``
    = the verify readback row followed by the token lanes (the host's one readback per pass).  No host tensor, no
    position change, no id through an FPU or reduce stage; ``observer`` (eager diagnostics only) is called after the
    history derivation, every step's inputs, embedding + mixer, layer and head, and after the assembly.
    """

    def stage(name: str) -> None:
        if observer is not None:
            observer(f"draft_lanes:{name}")

    _validate_lane_draft_state(model, verify, draft)
    if not isinstance(derive_history, bool):
        raise TypeError(f"derive_history must be a bool, got {derive_history!r}")
    if not isinstance(verify_output, Qwen38TTNNVerifyOutput) or not verify_output.active:
        raise ValueError("the lane draft body needs the live lane verify output whose readback row it reads")
    lanes, drafts = verify.lanes, verify.drafts
    width = lane_readback_width(lanes)
    if _shape(verify_output.readback) != (1, 1, 1, width) or verify_output.readback.dtype != ttnn.float32:
        raise RuntimeError(
            f"lane verify readback must be FP32 [1,1,1,{width}], got {tensor_metadata(verify_output.readback)}"
        )
    if model.poisoned:
        raise RuntimeError("the text model is poisoned")
    alignment = verify.alignment
    qsa = alignment.layer.attention
    positions = verify.positions
    lane_verify = draft.qsa_lane_verify_constants
    dram = ttnn.DRAM_MEMORY_CONFIG
    processed_rows = 0
    try:
        if derive_history:
            forward_draft_history_lanes(model, verify, draft)
        readback = verify_output.readback
        next_token = ttnn.slice(readback, (0, 0, 0, lanes), (1, 1, 1, 2 * lanes), memory_config=dram)
        first_draft = ttnn.slice(readback, (0, 0, 0, 2 * lanes), (1, 1, 1, 3 * lanes), memory_config=dram)
        first_tokens = ttnn.concat([first_draft, draft.sentinel_tail], dim=3, memory_config=dram)
        token_lanes_row = first_tokens
        step_rows: list[Any] = []  # step i's resolved row: lane u = d_{u, i+2}, the pad lanes -1
        # The roots of step 0: the persistent alignment residual rows padded to the tile (a view: never released).
        roots = _pad_rows(alignment.residual, lanes, label="MTP lane draft roots")
        owned_roots = None
        stage("history")
        for step in range(drafts - 1):
            if step:
                qsa.commit_verify_lanes(draft.qsa_state, draft.advance_selectors)  # the previous row joins the history
            position_row = ttnn.add(positions.row, draft.step_offset_rows[step], memory_config=dram)
            lane_rows = ttnn.gather(position_row, 3, lane_verify.lane_of_row, memory_config=dram)
            row_positions = ttnn.add(lane_rows, lane_verify.draft_of_row, memory_config=dram)
            block_start_lanes = ttnn.bitwise_and(position_row, positions.block_start_mask_row, memory_config=dram)
            block_lane_starts = ttnn.gather(block_start_lanes, 3, lane_verify.block_lane_of_row, memory_config=dram)
            block_start_rows = ttnn.add(block_lane_starts, lane_verify.block_offset_row, memory_config=dram)
            rope_rows, rope_block_rows = positions.rope_rows(
                row_positions, block_start_rows, lane_verify.lane_of_row, lane_verify.block_lane_of_row
            )
            rope = model.rope_table.rows_chunk(rope_rows, rope_block_rows)
            _deallocate(lane_rows, block_start_lanes, block_lane_starts, block_start_rows, rope_rows, rope_block_rows)
            qsa_inputs = qsa_module.derive_qsa_lane_verify_inputs(
                position_row,
                verify.active_row,
                verify.inactive_row,
                row_positions,
                model.qsa_position_constants,
                draft.qsa_chunk_constants,
                draft.qsa_verify_constants,
                draft.qsa_lane_constants,
                lane_verify,
                single_row=True,
            )
            _deallocate(row_positions, position_row)
            stage(f"row-{step}:inputs")
            token_tile = ttnn.to_layout(token_lanes_row, ttnn.TILE_LAYOUT, memory_config=dram)
            embedding_rows = model.model_io.embedding.embed_device_token_rows(token_tile)
            _deallocate(token_tile)
            if token_lanes_row is first_tokens:
                _deallocate(first_tokens)
            mixed = alignment.input_mixer.rows(embedding_rows, roots)
            _deallocate(embedding_rows)
            if owned_roots is not None:
                _deallocate(owned_roots)
                owned_roots = None
            stage(f"row-{step}:embed-mixer")
            residual = _forward_layer_verify_lanes(
                alignment.layer,
                mixed,
                draft.layer_state,
                draft,
                prepared_ple_rows=None,
                rope_rows=rope,
                qsa_inputs=qsa_inputs,
                selectors=None,
            )
            stage(f"row-{step}:layer")
            hidden = alignment.final_mixer.rows(residual, flat_views=True)
            resolved = _resolve_rows(model, hidden, rows=lanes, sentinel_tail=draft.sentinel_tail)
            _deallocate(hidden)
            step_rows.append(resolved)
            token_lanes_row = resolved
            roots = owned_roots = residual
            qsa_inputs.deallocate()
            rope.deallocate()
            processed_rows += 1
            stage(f"row-{step}:head")
        if owned_roots is not None:
            _deallocate(owned_roots)
        # Assembly: lane u's block ``[t'_u, d_{u,1} .. d_{u,k}]`` and ``[d_{u,1} .. d_{u,k}, -1]`` from 32-bit lane
        # copies, one concat each, the sentinel tail past B*R; the pass row = the readback then the token lanes.
        token_pieces: list[Any] = []
        draft_pieces: list[Any] = []
        for lane in range(lanes):
            ids = [_lane(first_draft, lane), *(_lane(row, lane) for row in step_rows)]
            token_pieces += [_lane(next_token, lane), *ids]
            draft_pieces += [*ids, draft.sentinel_lane]
        # the tail past B*R is absent when the lanes fill the tile (B * R == 32: 8 lanes at k = 3)
        tail = [] if verify.constants.sentinel_tail is None else [verify.constants.sentinel_tail]
        token_lanes = ttnn.concat([*token_pieces, *tail], dim=3, memory_config=dram)
        draft_lanes = ttnn.concat([*draft_pieces, *tail], dim=3, memory_config=dram)
        pass_row = ttnn.concat([readback, token_lanes], dim=3, memory_config=dram)
        for name, row, shape in (
            ("assembled lane verify token lanes", token_lanes, TOKEN_ROW_SHAPE),
            ("assembled lane draft lanes", draft_lanes, TOKEN_ROW_SHAPE),
            ("assembled lane pass row", pass_row, (1, 1, 1, lane_pass_row_width(lanes))),
        ):
            if _shape(row) != shape or row.dtype != ttnn.float32 or row.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise RuntimeError(f"{name} must be FP32 ROW_MAJOR {shape}, got {tensor_metadata(row)}")
        _land(pass_row, draft.pass_row, label="assembled lane pass row")
        token_tile = ttnn.to_layout(token_lanes, ttnn.TILE_LAYOUT, memory_config=dram)
        model.model_io.embedding.validate_token_row(token_tile, label="assembled lane verify token row")
        _land(token_tile, verify.token_row, label="assembled lane verify token row")
        _land(draft_lanes, verify.draft_lanes, label="assembled lane draft lanes")
        pieces = {_tensor_key(piece): piece for piece in (*token_pieces, *draft_pieces)}
        pieces.pop(_tensor_key(draft.sentinel_lane), None)
        _deallocate(
            token_lanes, token_tile, draft_lanes, pass_row, next_token, first_draft, *pieces.values(), *step_rows
        )
        stage("assemble")
    except BaseException as error:
        model._mark_poisoned("forward_draft_lanes", processed_rows, error)


def capture_draft_lanes(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNMTPLaneVerifyState,
    draft: Qwen38TTNNMTPLaneDraftState,
    verify_output: Qwen38TTNNVerifyOutput,
    *,
    guard: Callable[[str], AbstractContextManager[Any]],
    cq_id: int = 0,
    derive_history: bool = True,
) -> int:
    """Capture the lane draft body (after the lane verify whose readback address it reads); returns the trace id."""

    with corruptible_allocation_scope(model.mesh_device):
        trace_id = ttnn.begin_trace_capture(model.mesh_device, cq_id=cq_id)
        with guard(f"lane draft capture k={verify.drafts}"):
            forward_draft_lanes(model, verify, draft, verify_output, derive_history=derive_history)
        ttnn.end_trace_capture(model.mesh_device, trace_id, cq_id=cq_id)
    return trace_id


def read_lane_pass_row(
    verify: Qwen38TTNNMTPLaneVerifyState, draft: Qwen38TTNNMTPLaneDraftState
) -> tuple[Qwen38TTNNLaneVerifyReadback, tuple[tuple[int, ...], ...]]:
    """Host readback (outside any trace, after a lane verify and the draft that followed it) of the pass row from
    coordinate 0: the verify's per-lane ``[a_u, t'_u, d_1'_u, argmaxes, alignment]`` and every lane's assembled next
    verify tokens ``[t'_u, d_{u,1} .. d_{u,k}]`` (whose first two must be the verify's ``t'_u`` and ``d_1'_u``)."""

    lanes, rows = verify.lanes, verify.rows
    values = ttnn.to_torch(ttnn.get_device_tensors(draft.pass_row)[0]).reshape(-1)
    width = lane_pass_row_width(lanes)
    if values.numel() != width:
        raise RuntimeError(f"lane pass row readback has {values.numel()} lanes, expected {width}")
    split = lane_readback_width(lanes)
    readback = parse_lane_readback(values[:split], lanes=lanes, rows=rows)
    ids = [int(value) for value in values[split:].tolist()]
    tokens = tuple(tuple(ids[lane * rows : (lane + 1) * rows]) for lane in range(lanes))
    if any(token < 0 for block in tokens for token in block) or any(
        value != ZERO_EMBEDDING_TOKEN for value in ids[lanes * rows :]
    ):
        raise RuntimeError(f"lane pass row token lanes hold {ids}, expected {lanes} x {rows} ids then the sentinel")
    for lane, block in enumerate(tokens):
        expected = (readback.next_token[lane], readback.first_draft[lane])
        if block[:2] != expected:
            raise RuntimeError(f"lane {lane} pass row token lanes start {block[:2]}, its verify row says {expected}")
    return readback, tokens


# --------------------------------------------------------------------------- the pass loop


@dataclass(frozen=True)
class Qwen38TTNNMTPLaneTraces:
    """The three captured traces of a lane chain (the B=1 split form): ``verify_first`` (``catch_up=False``,
    ``land_accept=True``), ``commit``, ``draft``."""

    verify_first: int
    commit: int
    draft: int


@dataclass(frozen=True)
class Qwen38TTNNMTPLanePassRecord:
    """One pass of every lane: the host-mirrored start positions, the R verify tokens per lane, the accept counts,
    the ``a_u + 1`` committed ids, the R argmaxes and alignment argmaxes per lane, the next tokens the draft
    assembled, and the wall of every segment in ns (``commit_enqueue`` or ``commit_replay``, ``ple_rows``,
    ``verify_enqueue`` / ``draft_enqueue`` or the ``_replay`` forms, ``readback``; the bootstrap pass has
    ``host_inputs`` instead of the first two)."""

    index: int
    positions: tuple[int, ...]
    tokens: tuple[tuple[int, ...], ...]
    accepted: tuple[int, ...]
    committed: tuple[tuple[int, ...], ...]
    argmaxes: tuple[tuple[int, ...], ...]
    alignment_argmaxes: tuple[tuple[int, ...], ...]
    next_token: tuple[int, ...]
    first_draft: tuple[int, ...]
    next_tokens: tuple[tuple[int, ...], ...]
    segments_ns: dict[str, int]


class Qwen38TTNNMTPLaneChain:
    """The lane pass loop over the three traces (``mtp_v2.Qwen38TTNNMTPChain`` with per-lane bookkeeping): per pass
    (1) the commit trace enqueued non-blocking (every lane's previous pass out of its persistent buffers), (2) the
    host looks up every lane's R n-gram rows from its committed context (one table read) and uploads them, (3) the
    verify and, right behind it, the draft of the next pass enqueued, (4) ONE blocking readback of the pass row:
    per lane the accept, the committed ids and the next pass's tokens; the PLE contexts commit and the position
    mirror advances.  ``replay`` is the caller's raw blocking replay (``verify_before_replay`` called once outside
    the loop), ``enqueue`` its non-blocking form (without it every trace replays blocking: the segment-measuring
    form).  Every lane is active (stage 3: no lane parks; the section-9 mechanisms stay off: a pass that would not
    fit any lane is refused, never truncated)."""

    def __init__(
        self,
        model: Qwen38TTNNTextModel,
        verify: Qwen38TTNNMTPLaneVerifyState,
        draft: Qwen38TTNNMTPLaneDraftState,
        traces: Qwen38TTNNMTPLaneTraces,
        *,
        replay: Callable[[int], Any],
        enqueue: Callable[[int], Any] | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        observer: Callable[[str], Any] | None = None,
        next_tokens: Sequence[Sequence[int]] | None = None,
    ) -> None:
        _validate_lane_draft_state(model, verify, draft)
        if not isinstance(traces, Qwen38TTNNMTPLaneTraces) or not callable(replay):
            raise TypeError("the lane chain needs the captured traces and a replay callable")
        if (enqueue is not None and not callable(enqueue)) or (observer is not None and not callable(observer)):
            raise TypeError("enqueue and observer must be callables or None")
        self.model, self.verify, self.draft, self.traces = model, verify, draft, traces
        self.replay, self.enqueue, self.clock_ns, self.observer = replay, enqueue, clock_ns, observer
        self.records: list[Qwen38TTNNMTPLanePassRecord] = []
        self.active: list[int] = [1] * verify.lanes  # the host's copy of the device masks (:meth:`set_active`)
        self.commit_mask: list[int] = [
            1
        ] * verify.lanes  # the device's commit mask: the mask of the pass being committed
        # ``next_tokens``: the tokens an earlier (eager) pass assembled; the chain then continues with ``step``
        # (its commit replay commits that pass) instead of a bootstrap.
        self.next_tokens: tuple[tuple[int, ...], ...] | None = (
            None if next_tokens is None else tuple(tuple(int(t) for t in block) for block in next_tokens)
        )
        if self.next_tokens is not None and (
            len(self.next_tokens) != verify.lanes or any(len(block) != verify.rows for block in self.next_tokens)
        ):
            raise ValueError(f"next_tokens need {verify.lanes} lanes of {verify.rows} tokens, got {next_tokens!r}")

    def _require_room(self) -> None:
        positions = self.verify.positions.positions
        if not verify_pass_fits_lanes(positions, self.active, self.model.allocated_context):
            raise RuntimeError(
                f"lane positions {positions} have no room for a verify pass in {self.model.allocated_context} rows"
            )

    def set_active(self, active: Sequence[int]) -> None:
        """The active mask of the coming pass (host write between passes): the commit mask stays the mask of the pass
        being committed (the previous one), so a lane parked now still commits its last real pass and a lane admitted
        now commits nothing (its accept count is -1).  A parked lane's state is bitwise held by the selects."""

        flags = [int(flag) for flag in active]
        if len(flags) != self.verify.lanes or any(flag not in (0, 1) for flag in flags):
            raise ValueError(f"active needs {self.verify.lanes} 0/1 flags, got {active!r}")
        write_lane_active(self.model, self.verify, flags, commit=self.active)
        self.commit_mask = list(self.active)
        self.active = flags

    def set_next_tokens(self, tokens_by_lane: Sequence[Sequence[int]]) -> None:
        """The host rewrites the next pass's token row and draft lanes (an admission's bootstrap block, a parked
        lane's held block) and takes them as the tokens the next ``step`` runs on."""

        write_lane_tokens(self.model, self.verify, tokens_by_lane)
        self.next_tokens = tuple(tuple(int(token) for token in block) for block in tokens_by_lane)

    def _timed(self, segments: dict[str, int], name: str, action: Callable[[], Any]):
        if self.observer is not None:
            self.observer(name)
        started = self.clock_ns()
        result = action()
        segments[name] = self.clock_ns() - started
        return result

    def _finish_pass(self, tokens, segments: dict[str, int]) -> Qwen38TTNNMTPLanePassRecord:
        lanes = self.verify.lanes
        positions = tuple(self.verify.positions.positions)
        launch, form = (self.replay, "replay") if self.enqueue is None else (self.enqueue, "enqueue")
        self._timed(segments, f"verify_{form}", lambda: launch(self.traces.verify_first))
        self._timed(segments, f"draft_{form}", lambda: launch(self.traces.draft))
        readback, self.next_tokens = self._timed(
            segments, "readback", lambda: read_lane_pass_row(self.verify, self.draft)
        )
        # An inactive lane's rows are junk: it commits nothing and its position holds (the device advance is
        # (a_u + 1) * active_u); its readback fields are recorded as read.
        committed = tuple(
            tuple(readback.argmaxes[u][: readback.accepted[u] + 1]) if self.active[u] else () for u in range(lanes)
        )
        for lane, block in enumerate(committed):
            if self.active[lane] and (not block or block[-1] != readback.next_token[lane]):
                raise RuntimeError(
                    f"lane {lane} readback next token {readback.next_token[lane]} is not argmax {readback.accepted[lane]}"
                )
        commit_lane_verify_host(self.model, self.verify, [len(block) for block in committed])
        record = Qwen38TTNNMTPLanePassRecord(
            index=len(self.records),
            positions=positions,
            tokens=tuple(tuple(int(t) for t in block) for block in tokens),
            accepted=tuple(readback.accepted),
            committed=committed,
            argmaxes=readback.argmaxes,
            alignment_argmaxes=readback.alignment_argmaxes,
            next_token=tuple(readback.next_token),
            first_draft=tuple(readback.first_draft),
            next_tokens=self.next_tokens,
            segments_ns=segments,
        )
        self.records.append(record)
        return record

    def bootstrap(self, tokens_by_lane: Sequence[Sequence[int]]) -> Qwen38TTNNMTPLanePassRecord:
        """The first traced pass after the imports: host-written ``[t_u, d_1 .. d_k]`` per lane (the caller's drafts,
        any exact ids) and the PLE rows, the verify (``catch_up=False`` form: the accept rows hold -1, nothing
        commits), then the draft of the next pass."""

        if self.records or self.next_tokens is not None:
            raise RuntimeError("the lane chain was already bootstrapped (or continues an eager pass)")
        self._require_room()
        segments: dict[str, int] = {}
        self._timed(segments, "host_inputs", lambda: write_lane_verify_inputs(self.model, self.verify, tokens_by_lane))
        return self._finish_pass(tokens_by_lane, segments)

    def step(self) -> Qwen38TTNNMTPLanePassRecord:
        """One traced pass on the tokens the previous pass assembled: commit, PLE rows, verify, draft, readback."""

        if self.next_tokens is None:
            raise RuntimeError("bootstrap the lane chain first")
        self._require_room()
        segments: dict[str, int] = {}
        tokens = self.next_tokens
        if self.enqueue is not None:
            self._timed(segments, "commit_enqueue", lambda: self.enqueue(self.traces.commit))
        else:
            self._timed(segments, "commit_replay", lambda: self.replay(self.traces.commit))
        if self.commit_mask != self.active:
            # The commit just launched used the mask of the pass it commits; from the next commit on the mask is this
            # pass's (the host write queues behind the commit on the same command queue).  Left stale, a parked lane's
            # junk pass would be committed and an admitted lane's first pass never (mtp-b4-s4-lifecycle, 1b81ca8b).
            self._timed(
                segments,
                "mask_refresh",
                lambda: write_lane_active(self.model, self.verify, self.active, commit=self.active),
            )
            self.commit_mask = list(self.active)
        self._timed(segments, "ple_rows", lambda: write_lane_ple_rows(self.model, self.verify, tokens))
        return self._finish_pass(tokens, segments)


# --------------------------------------------------------------------------- lifecycle: the MTP lane image
# An MTP lane's parked image is the lane's slice of every MTP-lane family, read to the host as per-device torch shards
# and written back through the import's landing forms (the chunked KV slab write, ``fill_cache`` at the lane's batch
# index).  Unlike the generic lane image (the pager's) it holds the DERIVED histories -- the GDN FIR history rows, the
# QSA raw history rows, the PLE history slots -- because the MTP lane state carries no phase-bound ring; no residue
# conversion happens on either leg.  Only the used prefix of a lane's KV region and its whole compressed cache are
# carried: rows at or past the committed position are rewritten by the next pass before any sparse row can name
# them (the verify's write rule), so the image is exact for the stream that continues from it.  Eager, never traced;
# the caller warms every lane's eviction and import before the captures (a program compiled after them is the
# replay-hang class) and checks ``num_program_cache_entries()`` across every lifecycle op.


@dataclass
class Qwen38MTPLaneImage:
    """One lane's image on the host: per family the per-device torch shards (``kv[i][d]`` = QSA layer i's slab rows
    ``[0, kv_rows)`` on device d; ``compressed[i][d]`` its whole compressed cache; ``recurrent[j][d]`` GDN layer j's
    ``[H, 128, 128]`` fp32; ``gdn_history[j][d]`` its three FIR rows; ``raw_history[i][d]`` QSA layer i's three raw
    history rows; ``alignment_*`` the MTP layer's; ``ple_history[d]`` the nine slots ``[9, 4, 640]``), the position,
    the PLE n-gram context and the accept count the lane parks with."""

    position: int
    kv_rows: int
    ple_context: tuple[int, int] | None
    accepted: int
    kv: list[list[torch.Tensor]]
    compressed: list[list[torch.Tensor]]
    recurrent: list[list[torch.Tensor]]
    gdn_history: list[list[torch.Tensor]]
    raw_history: list[list[torch.Tensor]]
    alignment_kv: list[torch.Tensor]
    alignment_compressed: list[torch.Tensor]
    alignment_raw_history: list[torch.Tensor]
    ple_history: list[torch.Tensor]
    rope_shift: int = 0  # the lane's rotary shift (mrope; 0 for text)

    def digest(self) -> str:
        """sha256 over the committed state's bytes and the host record: the KV rows below the position, the compressed
        rows of the complete blocks below it (``position // 4``; the open block and the ones past it are rewritten by
        the lane's next pass, and a parked lane's redirected verify pools junk into them), the recurrent states and the
        history rows whole.  Two images of one committed state are equal iff equal here."""

        digest = hashlib.sha256(
            f"{self.position}:{self.kv_rows}:{self.ple_context}:{self.accepted}:{self.rope_shift}".encode()
        )
        blocks = self.position // qsa_module.COMPRESS_RATIO

        def update(shard: torch.Tensor) -> None:
            digest.update(shard.contiguous().view(torch.uint8).numpy().tobytes())

        for shards in (*self.kv, self.alignment_kv):
            for shard in shards:
                update(shard[:, :, : self.position])
        for shards in (*self.compressed, self.alignment_compressed):
            for shard in shards:
                update(shard[:, :, :blocks])
        for family in (
            self.recurrent,
            self.gdn_history,
            self.raw_history,
            [self.alignment_raw_history],
            [self.ple_history],
        ):
            for shards in family:
                for shard in shards:
                    update(shard)
        return digest.hexdigest()


def _read_shards(tensor) -> list[torch.Tensor]:
    """A small device tensor's per-device torch shards; the tensor is released."""

    shards = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]
    _deallocate(tensor)
    return shards


def _lane_slice(tensor, lane_axis: int, lane: int, *, rows: int | None = None, row_axis: int = 2):
    """Lane ``lane`` of ``tensor`` along ``lane_axis`` (rows ``[0, rows)`` along ``row_axis`` when given), owned."""

    start, end = [0] * 4, list(_shape(tensor))
    start[lane_axis], end[lane_axis] = lane, lane + 1
    if rows is not None:
        end[row_axis] = rows
    return slice_owned(tensor, start, end)


def kv_image_rows(position: int, allocated_context: int) -> int:
    """The KV rows an image carries: the committed prefix rounded up to whole import chunks (rows past the
    position are junk the next pass rewrites; whole chunks are what the slab write lands)."""

    chunk = kv_scratch_rows(allocated_context)
    rows = -(-max(1, int(position)) // chunk) * chunk
    return min(rows, allocated_context)


def _upload_shards(mesh_device, shards: Sequence[torch.Tensor], dtype, layout, shard_dim: int | None):
    """The per-device shards back on the mesh as one device tensor (sharded along ``shard_dim`` or replicated)."""

    if shard_dim is None:
        host = shards[0]
        mapper = replicate_tensor_2d_mesh_mapper(mesh_device)
    else:
        host = torch.cat(list(shards), dim=shard_dim)
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, shard_dim))
    return ttnn.from_torch(
        host, dtype=dtype, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )


def _qsa_lane_layers(verify: Qwen38TTNNMTPLaneVerifyState) -> list[Qwen38TTNNMTPLaneLayerState]:
    return [s for s in verify.layers if isinstance(s.attention_state, qsa_module.Qwen38TTNNQSALaneState)]


def _gdn_lane_layers(verify: Qwen38TTNNMTPLaneVerifyState) -> list[Qwen38TTNNMTPLaneLayerState]:
    return [s for s in verify.layers if not isinstance(s.attention_state, qsa_module.Qwen38TTNNQSALaneState)]


def evict_lane(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, lane: int, *, accepted: int
) -> Qwen38MTPLaneImage:
    """Lane ``lane``'s image to the host (outside any trace, between passes, after the host commit of the last pass
    the lane ran): its KV prefix rows, compressed caches, recurrent states, FIR / raw / PLE history rows, position,
    n-gram context and ``accepted`` (the accept count the lane parks with: -1 for a lane whose last pass is already
    committed).  The device state is left as it is."""

    _validate_lane_verify_state(model, verify)
    if isinstance(lane, bool) or type(lane) is not int or not 0 <= lane < verify.lanes:
        raise ValueError(f"lane must be an int in [0, {verify.lanes}), got {lane!r}")
    context = model.allocated_context
    position = int(verify.positions.positions[lane])
    kv_rows = kv_image_rows(position, context)
    width = 2 * qsa_module.HEAD_DIM
    history_rows = gdn_module.CONV_HISTORY_ROWS

    def kv_rows_of(state):
        base = lane * context
        return _read_shards(slice_owned(state.packed_kv_cache, (0, 0, base, 0), (1, 1, base + kv_rows, width)))

    def raw_history_of(rows_state):
        return _read_shards(_lane_slice(rows_state.raw_history, 1, lane, rows=qsa_module.RAW_HISTORY_ROWS))

    qsa_layers, gdn_layers = _qsa_lane_layers(verify), _gdn_lane_layers(verify)
    alignment = verify.alignment.layer_state
    ple_state = verify.layers[PLE_CHECKPOINT_LAYER].ple
    image = Qwen38MTPLaneImage(
        position=position,
        kv_rows=kv_rows,
        ple_context=ple_state.token_contexts[lane],
        accepted=int(accepted),
        rope_shift=int(verify.positions.shifts[lane]),
        kv=[kv_rows_of(s.attention_state) for s in qsa_layers],
        compressed=[_read_shards(_lane_slice(s.attention_state.compressed_index_cache, 0, lane)) for s in qsa_layers],
        recurrent=[_read_shards(_lane_slice(s.attention_state.recurrent, 0, lane)) for s in gdn_layers],
        gdn_history=[
            _read_shards(_lane_slice(s.attention_rows.history, 1, lane, rows=history_rows)) for s in gdn_layers
        ],
        raw_history=[raw_history_of(s.attention_rows) for s in qsa_layers],
        alignment_kv=kv_rows_of(alignment.attention_state),
        alignment_compressed=_read_shards(_lane_slice(alignment.attention_state.compressed_index_cache, 0, lane)),
        alignment_raw_history=raw_history_of(alignment.attention_rows),
        ple_history=_read_shards(_lane_slice(ple_state.history, 0, lane)),
    )
    return image


def readmit_lane(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNMTPLaneVerifyState, lane: int, image: Qwen38MTPLaneImage
) -> None:
    """``image`` into lane ``lane`` (outside any trace, between passes): every family through the import's landing
    forms, then the lane's position and n-gram context.  The caller writes the accept count (``image.accepted``, -1
    for a seeded lane), the active mask and the lane's next tokens before the next pass."""

    _validate_lane_verify_state(model, verify)
    if isinstance(lane, bool) or type(lane) is not int or not 0 <= lane < verify.lanes:
        raise ValueError(f"lane must be an int in [0, {verify.lanes}), got {lane!r}")
    if not isinstance(image, Qwen38MTPLaneImage):
        raise TypeError("readmit_lane takes a Qwen38MTPLaneImage")
    mesh, context = model.mesh_device, model.allocated_context
    lanes = verify.lanes
    chunk = kv_scratch_rows(context)
    if image.kv_rows % chunk or image.kv_rows > context:
        raise ValueError(f"image KV rows {image.kv_rows} are not whole {chunk}-row chunks of the {context}-row lane")
    width = 2 * qsa_module.HEAD_DIM
    dram = ttnn.DRAM_MEMORY_CONFIG
    bf16, tile, row_major = ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT

    def land_kv(state, shards):
        for start in range(0, image.kv_rows, chunk):
            part = _upload_shards(mesh, [s[:, :, start : start + chunk] for s in shards], bf16, row_major, 1)
            result = ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                state.packed_kv_cache, part, 0, 0, 1, lane * context + start, qsa_module.STAGING_AXIS
            )
            if result is not None and _tensor_key(result) != _tensor_key(state.packed_kv_cache):
                raise RuntimeError(f"lane {lane} image KV rows {start}..{start + chunk} did not land")
            _deallocate(part)

    def land_rows(target, shards, rows: int, width_: int, shard_dim: int | None, label: str):
        # ``rows`` real rows padded to the tile with zeros, landed at the lane's batch index through the tile view.
        padded = [torch.zeros((1, 1, CHUNK_ROWS, s.shape[-1]), dtype=s.dtype) for s in shards]
        for pad, s in zip(padded, shards):
            pad[0, 0, :rows] = s.reshape(rows, s.shape[-1])
        part = _upload_shards(mesh, padded, bf16, tile, shard_dim)
        _fill_lane(ttnn.experimental.view(target, (lanes, 1, CHUNK_ROWS, width_)), part, lane, label=label)
        _deallocate(part)

    qsa_layers, gdn_layers = _qsa_lane_layers(verify), _gdn_lane_layers(verify)
    for index, layer_state in enumerate(qsa_layers):
        state = layer_state.attention_state
        land_kv(state, image.kv[index])
        part = _upload_shards(mesh, image.compressed[index], bf16, tile, None)
        _fill_lane(state.compressed_index_cache, part, lane, label=f"lane {lane} image compressed {index}")
        _deallocate(part)
        land_rows(
            layer_state.attention_rows.raw_history,
            image.raw_history[index],
            qsa_module.RAW_HISTORY_ROWS,
            qsa_module.INDEX_HEAD_DIM,
            None,
            f"lane {lane} image raw history {index}",
        )
    for index, layer_state in enumerate(gdn_layers):
        state = layer_state.attention_state
        part = _upload_shards(mesh, image.recurrent[index], ttnn.float32, tile, 1)
        filled = ttnn.experimental.view(part, (1, 1, RECURRENT_ROWS, gdn_module.HEAD_DIM))
        _fill_lane(
            ttnn.experimental.view(state.recurrent, (lanes, 1, RECURRENT_ROWS, gdn_module.HEAD_DIM)),
            filled,
            lane,
            label=f"lane {lane} image recurrent {index}",
        )
        _deallocate(part)
        rows_state = layer_state.attention_rows
        land_rows(
            rows_state.history,
            image.gdn_history[index],
            gdn_module.CONV_HISTORY_ROWS,
            _shape(rows_state.history)[3],
            3,
            f"lane {lane} image FIR history {index}",
        )
    alignment = verify.alignment.layer_state
    land_kv(alignment.attention_state, image.alignment_kv)
    part = _upload_shards(mesh, image.alignment_compressed, bf16, tile, None)
    _fill_lane(
        alignment.attention_state.compressed_index_cache, part, lane, label=f"lane {lane} image alignment compressed"
    )
    _deallocate(part)
    land_rows(
        alignment.attention_rows.raw_history,
        image.alignment_raw_history,
        qsa_module.RAW_HISTORY_ROWS,
        qsa_module.INDEX_HEAD_DIM,
        None,
        f"lane {lane} image alignment raw history",
    )
    ple_state = verify.layers[PLE_CHECKPOINT_LAYER].ple
    history_view = ttnn.experimental.view(
        ple_state.history, (lanes * CONV_STATE_LENGTH, 1, *tuple(_shape(ple_state.history))[2:])
    )
    for slot in range(CONV_STATE_LENGTH):
        part = _upload_shards(mesh, [s[:, slot : slot + 1] for s in image.ple_history], bf16, tile, 3)
        _fill_lane(history_view, part, lane * CONV_STATE_LENGTH + slot, label=f"lane {lane} image PLE slot {slot}")
        _deallocate(part)
    contexts = list(ple_state.token_contexts)
    contexts[lane] = None if image.ple_context is None else tuple(int(v) for v in image.ple_context)
    ple_state.token_contexts = tuple(contexts)
    ple_state.validate()
    verify.positions.write_lane(lane, image.position, image.rope_shift)
    verify.pass_contexts[lane] = (contexts[lane],) * (verify.rows + 1)
