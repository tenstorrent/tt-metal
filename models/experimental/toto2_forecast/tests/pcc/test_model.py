# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Full-model forecast PCC on device against the reference goldens (``TtToto2.forecast``, one forward).

Gate metrics (``goldens.gate_metrics``; PLAN section 9 v1.1):
    pcc_normalized      head output (9, B, V, h, 32) in asinh-normalized space vs the hook on ``m.output_head``
    pcc_spread          the same minus its mean over the 9 quantiles (a quantile collapse scores 0)
    pcc_row_p01 / min   per-series PCC of the head output, 1st percentile / minimum
    pcc_final           quantiles (9, B, V, H) after sinh, rescale, clamp and sort vs the reference ``forecast``
    pcc_final_spread    the same minus its quantile mean
    abs_err_max         max abs error of the head output (judges near-constant series, ``goldens.degenerate_rows``)

Floors: ``goldens.GATE_FLOORS`` (contract floors for pcc_normalized / pcc_final, proposed ones for the others).
Cases take every path: clean (causal SDPA), missing patches (group mask), V = 8 sharing one series id (variate
SDPA without a mask), V = 4 with two series ids (variate mask), context 8192 (S = 288, 32 read-out tokens), and
single series in the realistic mix at contexts 2048 and 8192.

    TOTO2_PCC_JSON=<path>  appends one record per case
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest

from models.experimental.toto2_forecast.tests.golden.goldens import (
    tree_state,
    below_floors,
    floors_for,
    info_misses,
    gate_metrics,
    have_golden,
    load_golden,
)

# One device per module, so a loaded model (2.5B: about a minute) serves every case that uses it.
pytestmark = pytest.mark.use_module_device

SMALL = ("uni_4x2048", "missing_5x2048", "mv_2x4x2048", "long_2x8192")
FULL = ("uni_16x2048", "missing_16x2048", "mv_4x8x2048")
SINGLE = ("single_rw_1x2048", "single_sparse_1x2048", "single_shift_1x2048", "single_long_rw_1x8192")
# Ordered by checkpoint and precision, so each process holds one model at a time.
CASES = [
    *(("4m", n, "accuracy") for n in SMALL),
    *(("22m", n, "accuracy") for n in SMALL + SINGLE),
    *(("22m", n, "performance") for n in SMALL),
    *(("313m", n, "accuracy") for n in FULL + SINGLE),
    *(("313m", n, "performance") for n in FULL + SINGLE),  # D4 condition 2: every case in both precisions
    *(("2.5B-FT", n, "accuracy") for n in FULL + SINGLE),
    *(("2.5B-FT", n, "performance") for n in FULL + SINGLE),
]
_MODEL: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _free_models_before_device_close(_device_module_impl):
    """The cached model's device tensors must go before the module's device closes."""
    yield
    _MODEL.clear()
    gc.collect()


def tt_model(device, ckpt: str, precision_name: str):
    """One cached model per process (2.5B: one precision's weights at a time)."""
    from models.experimental.toto2_forecast.tt.common import Toto2Precision
    from models.experimental.toto2_forecast.tt.model import TtToto2

    key = (ckpt, precision_name)
    if key not in _MODEL:
        _MODEL.clear()
        gc.collect()
        _MODEL[key] = TtToto2.from_checkpoint(device, ckpt, Toto2Precision.named(precision_name))
    return _MODEL[key]


def run_forecast(model, golden):
    """``(q, head)`` on device for the golden's inputs."""
    import ttnn

    i = golden["inputs"]
    host = model.prepare_inputs(i["target"], i["target_mask"], i["series_ids"], i["horizon"])
    inputs = model.upload_inputs(host)
    try:
        out = model.forward_device(inputs)
        q, head = model.postprocess(out, host, normalized=True)
        ttnn.deallocate(out)
    finally:
        model.deallocate_inputs(inputs)
    return q, head


def _record(entry: dict) -> None:
    path = os.environ.get("TOTO2_PCC_JSON")
    if path:
        out = Path(path)
        data = json.loads(out.read_text()) if out.is_file() else []
        out.write_text(json.dumps([*data, {**entry, **tree_state()}], indent=2) + "\n")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("ckpt,case,precision_name", CASES, ids=[f"{c}-{n}-{p}" for c, n, p in CASES])
def test_model(device, ckpt, case, precision_name):
    if not have_golden(ckpt, case):
        pytest.skip(f"golden {ckpt}/{case} not generated yet")
    golden = load_golden(ckpt, case)
    model = tt_model(device, ckpt, precision_name)
    q, head = run_forecast(model, golden)
    cap = golden["captures"]
    metrics = gate_metrics(head, q, cap["head_out"], cap["q"])
    _record({"ckpt": ckpt, "case": case, "precision": precision_name, **metrics})
    print(f"\n[MODEL] {ckpt}/{case}/{precision_name}: " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
    info = info_misses(metrics, ckpt, precision_name)
    if info:
        print(f"[MODEL] info (not gated, D3): {info}")
    below = below_floors(metrics, floors_for(ckpt, precision_name, f"{ckpt}/{case}"))
    assert not below, f"below floor: {below}"
