# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA accuracy against an FP64 golden: the metric tiers of a run (``kda_tiers``) and the CI scorer (``KdaScorer``).

``kda_tiers`` is the one definition of the KDA tiers. It reads reduced sums only, so a CI run and a dev run directory
(``deepseek_v3_d_p/tests/kda/dev/layer0_run_analysis.py``, which stores these sums) score the same way:

* per output row: Σe² (squared error), Σyy (squared golden), Σxy (device times golden) over the hidden;
* per state snapshot (one after each device call) and head: the same three sums over the head's recurrent state.

With NRMSE ``sqrt(Σe² / Σyy)`` and gain ``Σxy / Σyy`` (the least-squares scale of the golden in the device result,
below 1 when the result is too small; PCC is blind to it), the tiers are:

* tier 0: output NRMSE over the run; pooled state NRMSE after the last call;
* tier 1: output per row (its golden norm floored at the golden's token floor) and per 1,024-row window; final state
  per head; pooled state per call. Each as ``<name>_p50/_p95/_p99`` (linear interpolation), ``<name>_worst`` and
  ``<name>_worst_at`` (row, window start, head, or snapshot end row);
* tier 2: output gain; final pooled state gain and each head's distance from gain 1 (``state_head_gain_final_*``,
  with the worst head's signed ``_worst_gain``); state per head per call (worst at ``_worst_row``, ``_worst_head``).

Each tier is one measurement over rows ``[0, rows)``. The output sums are added in torch over the caller's column
views, so the CI scorer's whole-run and window NRMSE equal ``windowed_errors.nrmse`` over the same rows bit for bit.

``KdaScorer`` scores a device run chunk by chunk (``block.BlockScorer``): the device output against the golden
``output`` rows and the device recurrent state, read after every call, against the golden's ``state.recurrent`` at
that call's end. Its run tiers are ``kda_tiers`` plus the output's absolute error (``output_rms_err`` beside the
golden's ``output_rms_ref``) and PCC, the per-1,024-row-window and per-call errors, and the final state's worst
per-head PCC.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar, check
from models.demos.common.prefill.long_context_accuracy.block import BlockOutputs, MetricTiers
from models.demos.common.prefill.long_context_accuracy.fp64_golden import Fp64Golden
from models.demos.common.prefill.long_context_accuracy.windowed_errors import (
    C_XY,
    COUNT,
    MEAN_X,
    MEAN_Y,
    SUM_EE,
    SUM_YY,
    WindowedErrors,
    errors,
    pcc,
    row_sums,
)

WINDOW = 1024  # rows per output window
TAIL_PERCENTILES = (50, 95, 99)
STATE_PRODUCT = "state.recurrent"  # the golden's recurrent state after each segment, [1, heads, k, v]


def _sqrt_ratio(num: float, den: float) -> float:
    """``sqrt(num / den)``: 0 for no error (also on an all-zero golden), inf for an error on an all-zero golden."""
    if num == 0:
        return 0.0
    return math.sqrt(num / den) if den > 0 else math.inf


def _nrmse(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(num / den)


def _tail(name: str, values: np.ndarray, positions: Sequence) -> dict:
    """``<name>_p50/_p95/_p99`` and the worst (largest; NaN when any is NaN) value with ``<name>_worst_at``."""
    worst = int(np.argmax(values))
    out = {f"{name}_p{p}": float(np.percentile(values, p)) for p in TAIL_PERCENTILES}
    out.update({f"{name}_worst": float(values[worst]), f"{name}_worst_at": positions[worst]})
    return out


def kda_tiers(
    e2: torch.Tensor,
    yy: torch.Tensor,
    xy: torch.Tensor,
    state_sums: np.ndarray,
    state_ends: Sequence[int],
    floor: float,
) -> MetricTiers:
    """The KDA metric tiers (module docstring) of ``rows`` output rows from row 0 and their state snapshots.

    ``e2``, ``yy``, ``xy``: per output row ``[rows]`` FP64 Σe², Σyy, Σxy. ``state_sums``: ``[snapshots, heads, 3]``
    FP64 (Σe², Σyy, Σxy) per state snapshot and head, the snapshots taken at rows ``state_ends``. ``floor``: the
    golden's per-row NRMSE floor (``windowed_errors.golden_floor`` over every row of the golden)."""
    rows = e2.shape[0]
    if not (rows and yy.shape == xy.shape == (rows,)):
        raise ValueError(f"expected per-row sums of equal length, got {[tuple(t.shape) for t in (e2, yy, xy)]}")
    if state_sums.ndim != 3 or state_sums.shape[-1] != 3 or len(state_sums) != len(state_ends) or not len(state_ends):
        raise ValueError(f"state sums {state_sums.shape} do not match {len(state_ends)} snapshots of [heads, 3]")
    span = {"start": 0, "end": rows, "rows": rows}
    whole_e2, whole_yy = float(e2.sum()), float(yy.sum())
    window_starts = list(range(0, rows, WINDOW))
    window_nrmse = np.array(
        [_sqrt_ratio(float(e2[s : s + WINDOW].sum()), float(yy[s : s + WINDOW].sum())) for s in window_starts]
    )
    error = e2.sqrt()
    token_nrmse = torch.where(error == 0, torch.zeros_like(error), error / yy.sqrt().clamp_min(floor)).numpy()

    s = state_sums
    head = _nrmse(s[..., 0], s[..., 1])  # [snapshots, heads]
    pooled = _nrmse(s[..., 0].sum(-1), s[..., 1].sum(-1))
    final_gain = s[-1, :, 2] / s[-1, :, 1]
    heads = list(range(s.shape[1]))
    head_gain = _tail("state_head_gain_final", np.abs(final_gain - 1), heads)
    head_gain["state_head_gain_final_worst_gain"] = float(final_gain[head_gain["state_head_gain_final_worst_at"]])
    per_head_call = _tail("state_head_window", head.ravel(), range(head.size))
    worst_call, worst_head = divmod(per_head_call.pop("state_head_window_worst_at"), s.shape[1])
    per_head_call.update(state_head_window_worst_row=state_ends[worst_call], state_head_window_worst_head=worst_head)
    return {
        0: [{**span, "output_nrmse": _sqrt_ratio(whole_e2, whole_yy), "state_final_nrmse": float(pooled[-1])}],
        1: [
            {
                **span,
                **_tail("output_token", token_nrmse, range(rows)),
                **_tail("output_window", window_nrmse, window_starts),
                **_tail("state_head_final", head[-1], heads),
                **_tail("state_window", pooled, list(state_ends)),
            }
        ],
        2: [
            {
                **span,
                "output_gain": float(xy.sum()) / whole_yy,
                "state_gain_final": float(s[-1, :, 2].sum() / s[-1, :, 1].sum()),
                **head_gain,
                **per_head_call,
            }
        ],
    }


def cross_sums(sums: torch.Tensor) -> torch.Tensor:
    """Σxy per row of ``windowed_errors.row_sums``: Σ(x − x̄)(y − ȳ) + n·x̄·ȳ."""
    return sums[:, C_XY] + sums[:, COUNT] * sums[:, MEAN_X] * sums[:, MEAN_Y]


class KdaScorer:
    """``block.BlockScorer`` of a KDA layer: ``outputs.output`` ``[rows, hidden]`` against the golden ``output``
    rows, and ``outputs.extras["recurrent"]`` ``[heads, k, v]``, the device state after the call, against the golden
    ``state.recurrent`` at the call's end (so every call ends a golden segment).

    Per call (``check_chunk``): finite output and state, and the per-call bars on the call's output errors
    (``call_nrmse``, ``call_pcc``, ``call_rms_err``, ``call_rms_ref``). Over the run (``score_run``): ``kda_tiers``
    with the golden's per-row floor ``token_floor``, plus in tier 0 ``output_rms_err`` / ``output_rms_ref`` /
    ``output_pcc``, in tier 1 one measurement per 1,024-row window (``window_nrmse`` / ``_rms_err`` / ``_rms_ref`` /
    ``_pcc``) and per call (``call_*``), and the final state's worst per-head PCC ``state_head_final_pcc_min`` (head
    ``_at``). No carried state: the state arrives with every call."""

    golden_tensors = ("output", STATE_PRODUCT)

    def __init__(self, token_floor: float) -> None:
        self.token_floor = token_floor
        self._rows = WindowedErrors()  # column "sums": windowed_errors.row_sums of every output row
        self._calls: list[dict] = []  # per call: span, call_nrmse, call_pcc
        self._state_sums: list[np.ndarray] = []  # per call: [heads, 3] Σe², Σyy, Σxy
        self._state_ends: list[int] = []
        self._final_head_sums: torch.Tensor | None = None  # row_sums per head of the last call's state

    def score_chunk(self, lo: int, outputs: BlockOutputs, rows: Mapping[str, torch.Tensor]) -> None:
        sums = row_sums(rows["output"], outputs.output)
        self._rows.append(lo, sums=sums)
        hi = lo + sums.shape[0]
        self._calls.append({"start": lo, "end": hi, "rows": hi - lo, **errors(sums, "call_")})
        golden_state = rows[STATE_PRODUCT][0]
        head_sums = row_sums(golden_state.flatten(1), outputs.extras["recurrent"].flatten(1))
        self._state_sums.append(
            torch.stack([head_sums[:, SUM_EE], head_sums[:, SUM_YY], cross_sums(head_sums)], 1).numpy()
        )
        self._state_ends.append(hi)
        self._final_head_sums = head_sums

    def check_chunk(
        self, lo: int, outputs: BlockOutputs, rows: Mapping[str, torch.Tensor], bars: Sequence[AcceptanceBar]
    ) -> list[str]:
        call = self._calls[-1]
        if call["start"] != lo:
            raise ValueError(f"check_chunk at row {lo} does not follow score_chunk at row {call['start']}")
        violations = [
            f"{name} has {int((~torch.isfinite(t)).sum())} non-finite values"
            for name, t in (("output", outputs.output), ("recurrent state", outputs.extras["recurrent"]))
            if not torch.isfinite(t).all()
        ]
        return violations + check([call], bars)

    def score_run(self, carried_state: Mapping[str, torch.Tensor], golden: Fp64Golden) -> MetricTiers:
        sums = self._rows.column("sums")
        tiers = kda_tiers(
            sums[:, SUM_EE],
            sums[:, SUM_YY],
            cross_sums(sums),
            np.stack(self._state_sums),
            self._state_ends,
            self.token_floor,
        )
        whole = self._rows.whole(lambda w: errors(w["sums"], "output_"))
        tiers[0][0].update({k: v for k, v in whole.items() if k in ("output_rms_err", "output_rms_ref", "output_pcc")})
        head_pcc = torch.tensor([pcc(self._final_head_sums[h : h + 1]) for h in range(len(self._final_head_sums))])
        worst = int(head_pcc.argmin()) if not head_pcc.isnan().any() else int(head_pcc.isnan().nonzero()[0])
        tiers[1][0].update(state_head_final_pcc_min=float(head_pcc[worst]), state_head_final_pcc_min_at=worst)
        tiers[1] += self._rows.windows(WINDOW, lambda w: errors(w["sums"], "window_")) + self._calls
        return tiers
