"""``mtp_accept``: the MTP pass's point-mass acceptance on the device, one program on one core.

The split sampled pass today reads the verify head (the rows' candidate rows), decides on the host
(``accept_point_mass`` over each row's distribution) and writes four decision buffers back before the tail.  This
program decides on the device under the device sampler's law (``fused/sampler_tail``: the temperature table's integer
weights, exact prefix sums, one fp32 multiply per decision, no division anywhere): row ``j`` accepts ``d_{j+1}`` iff
``fl32(u_j * S_j) < w_j(d_{j+1})`` (``S_j`` the row's kept total, ``w_j(d)`` the kept weight of ``d``, 0 when ``d`` is
not kept; a tie rejects); the first rejection draws ``x*`` from that row with ``d`` skipped in the prefix sums, using
the second uniform ``v``; with every draft accepted ``x*`` is the bonus row's draw with ``v``.  The outputs are the
split verify's decision buffers exactly as ``write_verify_decision`` writes them (the accept tile and index, the next
token, the alignment lanes ``[d_1 .. d_a*, x*, sentinel ...]``) plus a 16-lane statistics row for the ledger.

The host mirror is :func:`accept_reference` (the arithmetic of ``device_sampler_reference`` row by row); the device
test proves the program bitwise against it.  The constants are the sampler tail's (policy row, uniforms row, the
temperature table) built with ``rows = k + 1``: the uniforms row holds ``[u_0 .. u_{k-1}, v]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "mtp_accept"
KERNEL = fp.kernel_source(NAME, "accept.cpp")
CB_STAGE = 0
LANES = 128  # device_sampler.LANES
SHARD_LANES = 32
MAX_TOP_K = 32  # device_sampler.MAX_TOP_K
TABLE_SIZE = 65536  # device_sampler.TABLE_SIZE
DELTA_GRID = 1024  # device_sampler.DELTA_GRID
ROW_LANES = 2 * LANES  # SAMPLING_CANDIDATE_ROW_SHAPE[-1]
TOKEN_ROW_SHAPE = (1, 1, 1, 32)  # embedding.TOKEN_ROW_SHAPE
MIN_DRAFTS, MAX_DRAFTS = 1, 5
STATS_LANES = 16
STATS_SHAPE = (1, 1, 1, STATS_LANES)
STAT_WEIGHT, STAT_TOTAL, STAT_GUARD, STAT_RESAMPLED, STAT_ACCEPTED, STAT_TOKEN, STAT_THETA, STAT_KEPT = (
    0,
    5,
    10,
    11,
    12,
    13,
    14,
    15,
)
UNIFORM_BITS = 24  # sampling.UNIFORM_BITS: the host writes n * 2**-24
GRAIN = 64
TILE_BYTES = 4096


def stage_bytes(rows: int) -> int:
    """The kernel's L1 stage (accept.cpp's STAGE_BYTES): the tile, the rows, the three 32-lane rows, the policy, index
    and next-token grains, the statistics, the table chunks, the working arrays."""

    work_words = 3 * LANES + 5 * MAX_TOP_K
    return (
        TILE_BYTES + rows * ROW_LANES * 4 + 3 * 128 + 3 * GRAIN + STATS_LANES * 4 + MAX_TOP_K * GRAIN + work_words * 4
    )


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


def _metadata(tensor) -> str:
    return f"{tensor.dtype} {tensor.layout} {_shape(tensor)}"


def _one_core(mesh):
    core = ttnn.CoreCoord(0, 0)
    return core, ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _f32(value) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32)


# --- the host mirror ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RowKeptSet:
    """One row under the device law: the kept lanes in the filter's order with their integer weights."""

    ids: tuple[int, ...]  # the top_k lanes' global ids, value descending then id ascending
    weights: tuple[int, ...]  # their table weights (0 beyond top_k)
    kept: int  # the kept prefix (top-p and min-p over the top_k lanes, at least 1)
    total: int  # the kept total S
    guard_failed: bool  # the kept minimum did not clear the shard floor (the largest shard minimum)


def _order_key(values: torch.Tensor) -> torch.Tensor:
    """The kernel's key_of: a uint32 order where -0.0 sorts as +0.0."""

    bits = values.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits = torch.where((bits & 0x7FFFFFFF) == 0, torch.zeros_like(bits), bits)
    negative = (bits & 0x80000000) != 0
    return torch.where(negative, (~bits) & 0xFFFFFFFF, bits | 0x80000000)


def row_kept_set(values: torch.Tensor, ids: torch.Tensor, policy, table: torch.Tensor) -> RowKeptSet:
    """The device sampler's kept set of one candidate row (``device_sampler_reference`` up to its draw), plus the
    kernel's guard: ``values`` fp32 ``[128]``, ``ids`` int64 ``[128]`` in lane order, ``policy`` a
    ``Qwen38DeviceSamplerPolicy`` with no presence penalty, ``table`` the policy's temperature table."""

    if tuple(values.shape) != (LANES,) or values.dtype != torch.float32:
        raise ValueError(f"values must be fp32 [{LANES}], got {values.dtype} {tuple(values.shape)}")
    if tuple(ids.shape) != (LANES,) or ids.dtype != torch.int64:
        raise ValueError(f"ids must be int64 [{LANES}], got {ids.dtype} {tuple(ids.shape)}")
    if policy.presence_penalty != 0:
        raise ValueError("mtp_accept v1 takes no presence penalty (the admission keeps penalised requests on the host)")
    table = table.reshape(-1)
    keys = _order_key(values)
    by_id = torch.argsort(ids)  # distinct ids: ascending id first, then the stable sort by key descending
    order = by_id[torch.sort(keys[by_id], descending=True, stable=True).indices]
    s_sorted = values[order]
    g_sorted = ids[order]
    delta = (s_sorted - s_sorted[0]) * -float(DELTA_GRID)
    index = torch.clamp(torch.floor(delta), 0.0, float(TABLE_SIZE - 1)).to(torch.int64)
    weights = table[index].to(torch.int64)
    weights[policy.top_k :] = 0
    inclusive = torch.cumsum(weights, dim=0)
    exclusive = inclusive - weights
    total_top_k = inclusive[policy.top_k - 1]
    tau = (_f32(policy.top_p) * total_top_k.to(torch.float32)).to(torch.float32)
    k_p = int((exclusive.to(torch.float32) <= tau).sum())
    k_m = int((weights.to(torch.float32) >= _f32(policy.min_weight)).sum())
    kept = max(1, min(k_p, k_m))
    total = int(inclusive[kept - 1])
    floor_key = int(keys.reshape(LANES // SHARD_LANES, SHARD_LANES).min(dim=1).values.max())
    guard_failed = int(keys[order[kept - 1]]) <= floor_key
    top = policy.top_k
    return RowKeptSet(
        tuple(int(g) for g in g_sorted[:top]), tuple(int(w) for w in weights[:top]), kept, total, guard_failed
    )


def theta_draw(row: RowKeptSet, uniform: float, *, skip: int | None = None) -> tuple[int, float]:
    """The theta rule: ``fl32(u * total)`` against the exact prefix sums of the kept lanes (lane ``skip`` left out of
    the sums and the choice), the lane whose prefix first exceeds theta, capped at the last lane.  Returns the
    token and theta."""

    lanes = [i for i in range(row.kept) if i != skip]
    total = sum(row.weights[i] for i in lanes)
    theta = float((_f32(uniform) * _f32(total)).to(torch.float32))
    running, below = 0, 0
    for i in lanes:
        running += row.weights[i]
        if float(_f32(running)) <= theta:
            below += 1
    lane = min(below, len(lanes) - 1)
    return row.ids[lanes[lane]], theta


@dataclass(frozen=True)
class AcceptReference:
    accepted: int
    token: int
    alignment: tuple[int, ...]  # 32 lanes: [d_1 .. d_a*, x*, sentinel ...]
    weights: tuple[int, ...]  # w_j(d_{j+1}) per evaluated row
    totals: tuple[int, ...]  # S_j per evaluated row
    guard_mask: int
    resampled: bool
    theta: float
    kept: int

    def statistics_row(self) -> torch.Tensor:
        row = torch.full((STATS_LANES,), -1.0, dtype=torch.float32)
        row[STAT_GUARD:] = 0.0
        for j, (w, s) in enumerate(zip(self.weights, self.totals)):
            row[STAT_WEIGHT + j] = float(w)
            row[STAT_TOTAL + j] = float(s)
        row[STAT_GUARD] = float(self.guard_mask)
        row[STAT_RESAMPLED] = 1.0 if self.resampled else 0.0
        row[STAT_ACCEPTED] = float(self.accepted)
        row[STAT_TOKEN] = float(self.token)
        row[STAT_THETA] = self.theta
        row[STAT_KEPT] = float(self.kept)
        return row


def _check_uniform(uniform: float) -> float:
    uniform = float(uniform)
    if not 0 <= uniform < 1 or uniform * (1 << UNIFORM_BITS) != int(uniform * (1 << UNIFORM_BITS)):
        raise ValueError(f"uniform must be a multiple of 2**-{UNIFORM_BITS} in [0, 1), got {uniform!r}")
    return uniform


def accept_reference(
    candidate_rows: torch.Tensor,
    drafts: Sequence[int],
    policy,
    uniforms: Sequence[float],
    *,
    table=None,
    sentinel: int,
) -> AcceptReference:
    """The kernel's decision on the host: ``candidate_rows`` fp32 ``[k + 1, 256]`` (per shard 32 values then 32 ids),
    ``drafts`` ``[d_1 .. d_k]``, ``uniforms`` ``[u_0 .. u_{k-1}, v]`` (multiples of ``2**-24``), ``sentinel`` the
    zero-embedding token the alignment lanes past ``a*`` hold."""

    k = len(drafts)
    if not MIN_DRAFTS <= k <= MAX_DRAFTS:
        raise ValueError(f"a pass proposes {MIN_DRAFTS}..{MAX_DRAFTS} drafts, got {k}")
    if tuple(candidate_rows.shape) != (k + 1, ROW_LANES) or candidate_rows.dtype != torch.float32:
        raise ValueError(f"candidate rows must be fp32 [{k + 1}, {ROW_LANES}], got {tuple(candidate_rows.shape)}")
    if len(uniforms) != k + 1:
        raise ValueError(f"{k} drafts take {k + 1} uniforms (u_0 .. u_{k - 1}, v), got {len(uniforms)}")
    uniforms = [_check_uniform(u) for u in uniforms]
    if table is None:
        from models.demos.blackhole.qwen38_flash_next.ttnn.device_sampler import weight_table

        table = weight_table(policy.temperature)
    packs = candidate_rows.reshape(k + 1, LANES // SHARD_LANES, 2, SHARD_LANES)
    values = packs[:, :, 0].reshape(k + 1, LANES).to(torch.float32)
    ids = packs[:, :, 1].reshape(k + 1, LANES).to(torch.int64)
    weights: list[int] = []
    totals: list[int] = []
    guard_mask = 0
    alignment = [int(sentinel)] * 32
    for j, draft in enumerate(drafts):
        row = row_kept_set(values[j], ids[j], policy, table)
        guard_mask |= int(row.guard_failed) << j
        lane_d = next((i for i in range(row.kept) if row.ids[i] == int(draft)), None)
        w_d = row.weights[lane_d] if lane_d is not None else 0
        weights.append(w_d)
        totals.append(row.total)
        theta_j = float((_f32(uniforms[j]) * _f32(row.total)).to(torch.float32))
        if theta_j < float(_f32(w_d)):
            alignment[j] = int(draft)
            continue
        token, theta = theta_draw(row, uniforms[k], skip=lane_d)
        alignment[j] = token
        return AcceptReference(
            j, token, tuple(alignment), tuple(weights), tuple(totals), guard_mask, True, theta, row.kept
        )
    row = row_kept_set(values[k], ids[k], policy, table)
    guard_mask |= int(row.guard_failed) << k
    token, theta = theta_draw(row, uniforms[k])
    alignment[k] = token
    return AcceptReference(
        k, token, tuple(alignment), tuple(weights), tuple(totals), guard_mask, False, theta, row.kept
    )


# --- the device side --------------------------------------------------------------------------------------------


def _expect(tensor, shape, dtype, layout, label: str) -> None:
    if _shape(tensor) != tuple(shape) or tensor.dtype != dtype or tensor.layout != layout:
        raise ValueError(f"{label} must be {dtype} {layout} {tuple(shape)}, got {_metadata(tensor)}")


def allocate_statistics(mesh, memory_config=None):
    """The program's statistics row (fp32 ROW_MAJOR ``[1,1,1,16]``), one per device."""

    return fp.allocate(STATS_SHAPE, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config or ttnn.DRAM_MEMORY_CONFIG)


def mtp_accept(
    candidate_rows: Any,
    draft_lanes: Any,
    constants: Any,
    *,
    accept_tile: Any,
    accept_index: Any,
    next_token: Any,
    alignment_tokens: Any,
    statistics: Any,
):
    """The decision program: ``candidate_rows`` fp32 ROW_MAJOR ``[1,1,k+1,256]`` (the split's candidates readback),
    ``draft_lanes`` fp32 ROW_MAJOR ``[1,1,1,32]`` (``[d_1 .. d_k, sentinel ...]``), ``constants`` the sampler tail's
    (policy row, uniforms row ``[u_0 .. u_{k-1}, v, ...]``, temperature table); the four decision buffers are written
    in place (the split's persistent tensors) and ``statistics`` (:func:`allocate_statistics`) is returned."""

    rows = _shape(candidate_rows)[2] if len(_shape(candidate_rows)) == 4 else 0
    if not MIN_DRAFTS + 1 <= rows <= MAX_DRAFTS + 1:
        raise ValueError(f"candidate rows must hold k + 1 in {MIN_DRAFTS + 1}..{MAX_DRAFTS + 1} rows, got {rows}")
    _expect(candidate_rows, (1, 1, rows, ROW_LANES), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, "candidate rows")
    _expect(draft_lanes, TOKEN_ROW_SHAPE, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, "draft lanes")
    _expect(accept_tile, (1, 1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT, "accept tile")
    _expect(accept_index, (1, 1, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, "accept index")
    _expect(next_token, (1, 1, 1, 1), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, "next token")
    _expect(alignment_tokens, TOKEN_ROW_SHAPE, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, "alignment tokens")
    _expect(statistics, STATS_SHAPE, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, "statistics")
    for name in ("policy_row", "uniforms", "weight_table"):
        if not hasattr(constants, name):
            raise TypeError(f"constants need the sampler tail's {name}")
    if _shape(constants.uniforms)[3] < rows:
        raise ValueError(f"the uniforms row holds {_shape(constants.uniforms)[3]} lanes, the pass needs {rows}")
    mesh = candidate_rows.device()
    core, one = _one_core(mesh)
    inputs = [candidate_rows, draft_lanes, constants.policy_row, constants.uniforms, constants.weight_table]
    outputs = [accept_tile, accept_index, next_token, alignment_tokens, statistics]
    accessed = inputs + outputs
    kernel = fp.reader_kernel(
        KERNEL,
        one,
        [a for tensor in accessed for a in fp.accessor_args(tensor)],
        [(core, [tensor.buffer_address() for tensor in accessed])],
        named={"cb_stage": CB_STAGE, "rows": rows, "lanes": LANES, "table_size": TABLE_SIZE},
    )
    pages = -(-stage_bytes(rows) // TILE_BYTES)
    # the candidate rows and the draft lanes in, the policy grain, the uniforms row and the table's kept-weight chunks
    # (64-byte grains, top_k per evaluated row at its bound) at the bytes the kernel reads, the accept tile, index,
    # next token, alignment lanes and statistics row out; per evaluated row (at its bound, k + 1) the sampler tail's
    # six passes over the 128 lanes (the key, the order, the weights, the prefix sums, the two kept counts) and one
    # more for the decision's product and the draw's prefix compares
    meta = fp.program_meta(
        NAME,
        "accept",
        rows,
        reads=(candidate_rows, draft_lanes),
        writes=(accept_tile, accept_index, next_token, alignment_tokens, statistics),
        partial=(
            (constants.policy_row, GRAIN),
            (constants.uniforms, TOKEN_ROW_SHAPE[3] * 4),
            (constants.weight_table, rows * MAX_TOP_K * GRAIN),
        ),
        flops=rows * LANES * 7,
        cores=1,
    )
    fp.run_program(
        accessed,
        fp.program_descriptor([kernel], cbs=[fp.cb_descriptor(CB_STAGE, ttnn.float32, TILE_BYTES, pages, one)]),
        meta=meta,
    )
    return fp.stamp_topology(statistics, candidate_rows)


def decide_on_host_composed(*args, **kwargs):
    """There is no ttnn composite: the composed form is the host pass decision (``qwen38_sampling_step.accept_pass``),
    which the chain runs when this program is off."""

    raise RuntimeError("mtp_accept has no composed device form: the chain decides the pass on the host when it is off")


register(
    FusedKernel(
        name=NAME,
        replaces="the sampled pass's host decision (read_verify_head, accept_point_mass, write_verify_decision)",
        tolerance=BITWISE,
        fused=mtp_accept,
        composed=decide_on_host_composed,
        gate=None,  # the single-chip device test against accept_reference is the component gate
    )
)

__all__ = [
    "NAME",
    "STATS_LANES",
    "AcceptReference",
    "RowKeptSet",
    "accept_reference",
    "allocate_statistics",
    "mtp_accept",
    "row_kept_set",
    "stage_bytes",
    "theta_draw",
]
