# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers and floors (skill §3). ``FLOORS`` / ``DRIFT`` change only through a DECISIONS.md entry."""

from __future__ import annotations

import torch

from models.experimental.tafsut_forecast.reference import dataflow as df
from models.experimental.tafsut_forecast.tests.pcc import metrics as M

# CONTRACT §3 / D0 Q3: pcc_normalized 0.999 provisional (accuracy), pcc_final 0.99; performance 0.995 / 0.98.
# Band, row, inactive and abs_err_max floors: proposed at G1 from round 0.
FLOORS = {
    "accuracy": {"pcc_normalized": 0.999, "pcc_final": 0.99},
    "performance": {"pcc_normalized": 0.995, "pcc_final": 0.98},
}
MODULE_FLOORS = {"pcc_normalized": 0.99, "pcc_final": 0.98}  # tt-metal tiers for committed tests (skill §3)
DRIFT: dict = {}


def pcc(a, b) -> float:
    return M.pearson(a, b)


def case_metrics(heads: list, g: dict, q: torch.Tensor, forward: int = 0, teacher_head=None) -> dict:
    """Head metrics of forward ``forward`` plus final-quantile metrics (``q`` ``[B, H, 9]``)."""
    f = g["forwards"][forward]
    head = heads[forward] if teacher_head is None else teacher_head
    n = head.shape[1] if forward else min(head.shape[1], -(-g["horizon"] // df.PATCH))
    n = min(n, f["head"].shape[1])
    if forward == 0:
        n = min(-(-min(g["horizon"], 1024) // df.PATCH), n)
    gh = f["head"][:, :n]
    prep = df.prepare_inputs(g["x"])
    return M.gate_metrics(head[:, :n].float(), gh, q.transpose(1, 2), g["quantiles"].transpose(1, 2), prep.scale)


def check_floors(met: dict, prec: str, module: bool = False):
    floors = dict(FLOORS[prec])
    if module:
        floors = {k: min(v, MODULE_FLOORS[k]) for k, v in floors.items()}
    bad = {k: met[k] for k, v in floors.items() if met[k] is None or met[k] < v}
    assert not bad, f"below floor {floors}: {bad}"
