# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 MoE long-context accuracy against the FP64 chain: the device routing readback, the per-chunk scoring and
the bars of ``test_kimi_k3_1_moe_long_context.py``.

Every row is scored (``common/prefill/long_context_accuracy``): the MoE output's absolute and relative error
(``windowed_errors.errors``: RMS error, the reference's RMS, pooled NRMSE, PCC) and the median per-row NRMSE, and the
routing against the FP64 top-16 (``routing_error.routing_rows``: recall, the FP64 weight on missed experts, missed
experts ranked in the FP64 selection).

Bars (``BARS``), each held in every window: the study's MoE teacher-feed bars (``moe_teacher_feed.BARS``, branch
``mvasilijevic/k31/accuracy-harness-consumers``), proposed there against the vLLM trace over 9 layers at 128K. Its
growth bar is left out: the MoE carries no state across chunks (a reverse-order run is bit-identical), so its error
versus position follows the input statistics, not accumulation; it is gated by the absolute per-window bars instead.
Measured by this test at 1M (commit 9bf2a291b5b, one 4x4 tray half), layer 1 / layer 2: pooled NRMSE 0.088 / 0.078,
median row NRMSE 0.083 / 0.071, mean recall 0.994 / 0.992 (worst 16K window 0.9939 / 0.9916), mean dropped weight
0.0038 / 0.0053 (worst window 0.0041 / 0.0059), no missed expert in the top half; the RMS error's worst window is
1.02x / 1.03x its first.
"""

from __future__ import annotations

import torch

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar
from models.demos.common.prefill.long_context_accuracy.moe.routing_error import routing_rows
from models.demos.common.prefill.long_context_accuracy.windowed_errors import (
    SUM_YY,
    Window,
    WindowedErrors,
    errors,
    quantile,
    row_nrmse,
    row_sums,
)
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import golden_floor

TOP_K = KimiK31Config.NUM_EXPERTS_PER_TOKEN  # 16
NUM_EXPERTS = KimiK31Config.NUM_ROUTED_EXPERTS  # 896

BARS = [
    AcceptanceBar("moe_row_nrmse_median", "max", 0.20),
    AcceptanceBar("recall_mean", "min", 0.98),
    AcceptanceBar("miss_top_half_fraction", "max", 0.01),
    AcceptanceBar("dropped_weight_mean", "max", 0.01),
]


def owner_positions(
    sp_row: int, tp_col: int, *, rows_per_sp_row: int, pad_rows: int, row_blocks: int, tp: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the MoE block chip (``sp_row``, ``tp_col``)'s owner rows (its routing rows, in order) sit in the chunk:
    (index into the chip's owner rows, chunk row) for every real row it owns. SP row r holds chunk rows
    ``[r * rows_per_sp_row, (r + 1) * rows_per_sp_row)`` plus ``pad_rows`` block padding rows at its end; in each of
    the ``row_blocks`` blocks of that SP row, TP column p owns the p-th of ``tp`` equal pieces."""
    sub = (rows_per_sp_row + pad_rows) // row_blocks
    piece = sub // tp
    pos = torch.cat([j * sub + tp_col * piece + torch.arange(piece) for j in range(row_blocks)])
    owned = torch.arange(pos.numel())
    keep = pos < rows_per_sp_row
    return owned[keep], sp_row * rows_per_sp_row + pos[keep]


def device_routing(ffn, block, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The routing of the block's last call as (checkpoint expert ids ``[rows, 16]`` int64, weights ``[rows, 16]``
    bf16) in the chunk's token order.

    ``block.last_head.routing_rows`` holds, on each chip, its owner rows: per token 16 expert ids (block order) then
    16 bf16 weights as uint16 words (``router_common.hpp routing_rows``). Mesh chip i = (mr, mc) is the block's chip
    (SP row, TP column) = (mr, mc) with SP on axis 0, (mc, mr) with SP on axis 1; ``owner_positions`` places its owner
    rows in the chunk. Block expert b is checkpoint expert ``block_order_perm[b]``."""
    import ttnn
    from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import block_order_perm

    mesh = ffn.mesh
    R, C = tuple(mesh.shape)
    sp_axis = ffn.options.sp_axis
    sp, tp = (R, C) if sp_axis == 0 else (C, R)
    chunk = block.resources.config.chunk_tokens
    assert block.splits == 1, f"the routing rows hold the last of {block.splits} splits only"
    words = torch.full((chunk, 2 * TOP_K), -1, dtype=torch.int64)
    for i, d in enumerate(ttnn.get_device_tensors(block.last_head.routing_rows)):
        mr, mc = divmod(i, C)
        r, p = (mr, mc) if sp_axis == 0 else (mc, mr)
        owned, pos = owner_positions(
            r, p, rows_per_sp_row=chunk // sp, pad_rows=block.pad_rows, row_blocks=block.row_blocks, tp=tp
        )
        words[pos] = (ttnn.to_torch(d).to(torch.int64).reshape(-1, 2 * TOP_K) & 0xFFFF)[owned]
    assert (words[:rows] >= 0).all(), "a chunk row no chip's routing rows hold"
    perm = block_order_perm(mesh, NUM_EXPERTS // (R * C), sp_axis)
    return perm[words[:rows, :TOP_K]], words[:rows, TOP_K:].to(torch.int16).view(torch.bfloat16)


def append_chunk(
    series: WindowedErrors,
    start: int,
    *,
    output: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    want_output: torch.Tensor,
    want_ids: torch.Tensor,
    want_weights: torch.Tensor,
) -> None:
    """Score one chunk's rows ``[start, start + n)``: the device ``output`` / routing (``ids``, ``weights``) against
    the FP64 chain's (``want_*``; ``want_ids`` best first, so a miss's rank is its FP64 rank)."""
    want_ids = want_ids.long()
    order = -torch.arange(want_ids.shape[1], dtype=torch.float32).expand(want_ids.shape)  # (best first)
    r = routing_rows(ids.long(), weights, want_ids, want_weights, order)
    series.append(
        start,
        sums=row_sums(want_output, output),
        recall=r["recall"],
        dropped=r["dropped"],
        weight_err=r["weight_err"],
    )
    series.add_events("misses", **{**r["misses"], "row": r["misses"]["row"] + start})


def row_floor(series: WindowedErrors) -> float:
    """The per-row NRMSE floor (``golden_floor``: 1% of the median reference row norm) over the whole series."""
    return golden_floor(series.column("sums")[:, SUM_YY].sqrt())


def measure(floor: float):
    """A window's measures (``WindowedErrors.windows`` / ``whole``) with per-row NRMSE floored at ``floor``: the
    output's absolute and relative error (``moe_*``), the median / p99 / max per-row NRMSE, and the routing: mean recall
    of the FP64 top-16, the fraction of rows with the FP64 set, the FP64 weight on missed experts (mean, p99), the
    misses and the fraction of them in the top half of the FP64 selection, the largest weight error on a shared expert
    (median)."""

    def of(w: Window) -> dict:
        nr, recall, rank = row_nrmse(w["sums"], floor), w["recall"], w["misses"]["rank"]
        return {
            **errors(w["sums"], "moe_"),
            "moe_row_nrmse_median": float(nr.median()),
            "moe_row_nrmse_p99": quantile(nr, 0.99),
            "moe_row_nrmse_max": float(nr.max()),
            "recall_mean": float(recall.mean()),
            "set_equal_fraction": float((recall == 1).float().mean()),
            "dropped_weight_mean": float(w["dropped"].mean()),
            "dropped_weight_p99": quantile(w["dropped"], 0.99),
            "misses": int(rank.numel()),
            # (no miss: none in the top half, not undefined)
            "miss_top_half_fraction": (
                float((2 * rank < w["misses"]["selected"]).float().mean()) if rank.numel() else 0.0
            ),
            "weight_err_median": float(w["weight_err"].median()),
        }

    return of
