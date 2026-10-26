# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU check (python_env, no device) that the gate metrics can fail (PLAN section 9, v1.1).

For each golden, the reference head output with every quantile collapsed to its mean over the 9 quantiles is
scored like a device output (``goldens.gate_metrics``), after the same host post-processing. ``pcc_spread``,
``pcc_row_p01_active``, ``spread_ratio_p01`` and ``pcc_final_spread`` must score below their accuracy floors (``goldens.GATE_FLOORS``) on every
golden, and every metric but ``pcc_final`` must on the realistic-mix goldens. ``pcc_final`` is recorded but cannot
catch a collapse on smooth inputs: raw-space per-series levels carry it (0.9998-0.99997 there).

    TOTO2_COLLAPSED_JSON=<path>  appends one record per golden
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from models.experimental.toto2_forecast.tests.golden.goldens import (
    CEILINGS,
    below_floors,
    collapsed,
    floors_for,
    gate_metrics,
    have_golden,
    load_golden,
)
from models.experimental.toto2_forecast.tt.preprocessing import prepare_inputs
from models.experimental.toto2_forecast.tt.readout import postprocess

CASES = [
    ("4m", "uni_4x2048"),
    ("22m", "long_2x8192"),
    ("22m", "single_rw_1x2048"),
    ("313m", "uni_16x2048"),
    ("313m", "paper_1024x2048"),
    ("313m", "paper_mix_1024x2048"),
    ("313m", "single_rw_1x2048"),
    ("2.5B-FT", "uni_16x2048"),
    ("2.5B-FT", "missing_16x2048"),
    ("2.5B-FT", "mv_4x8x2048"),
    ("2.5B-FT", "paper_1024x2048"),
    ("2.5B-FT", "long_context_32x8192"),
    ("2.5B-FT", "paper_mix_1024x2048"),
    ("2.5B-FT", "long_context_mix_32x8192"),
    ("2.5B-FT", "multivariate_mix_128x8x2048"),
    ("2.5B-FT", "single_rw_1x2048"),
    ("2.5B-FT", "single_sparse_1x2048"),
    ("2.5B-FT", "single_shift_1x2048"),
    ("2.5B-FT", "single_long_rw_1x8192"),
]
UNCATCHABLE = {"pcc_final"}


@pytest.mark.parametrize("ckpt,case", CASES, ids=[f"{c}-{n}" for c, n in CASES])
def test_collapsed_output_fails(ckpt, case):
    if not have_golden(ckpt, case):
        pytest.skip(f"golden {ckpt}/{case} not generated yet")
    g = load_golden(ckpt, case)
    i, c = g["inputs"], g["captures"]
    host = prepare_inputs(i["target"], i["target_mask"], i["series_ids"], i["horizon"])
    head = collapsed(c["head_out"])
    q = postprocess(head, host.loc, host.scale, host.caps, host.horizon)
    metrics = gate_metrics(head, q, c["head_out"], c["q"])
    path = os.environ.get("TOTO2_COLLAPSED_JSON")
    if path:
        out = Path(path)
        data = json.loads(out.read_text()) if out.is_file() else []
        out.write_text(json.dumps([*data, {"ckpt": ckpt, "case": case, **metrics}], indent=2) + "\n")
    print(f"\n[COLLAPSED] {ckpt}/{case}: " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
    floors = floors_for(ckpt, "accuracy")
    caught = below_floors(metrics, floors)
    if metrics["rows_degenerate"] == metrics["rows"]:
        # D2 item 3: the degenerate spread_ratio_p50 range catches the collapse (p50 ~ 0) on an all-degenerate case.
        assert "spread_ratio_p50" in caught, f"a collapsed output passes spread_ratio_p50 {metrics['spread_ratio_p50']:.5f}"
        return
    # The spread and per-row metrics must catch it everywhere; on the realistic-mix goldens every metric but
    # pcc_final must (on the smooth unit-test goldens per-row offsets carry pcc_normalized to 0.990-0.9986).
    required = {"pcc_spread", "pcc_row_p01_active", "spread_ratio_p01", "pcc_final_spread"}
    if g["case"].get("mix") == "realistic":
        required |= {
            k for k in set(floors) - UNCATCHABLE - CEILINGS if not k.startswith("spread_ratio_inactive")
        } - {"pcc_row_min_active_many", "spread_ratio_p50_degenerate"}
    missed = sorted(required - set(caught))
    assert not missed, f"a collapsed output passes these floors: { {k: round(metrics[k], 5) for k in missed} }"
