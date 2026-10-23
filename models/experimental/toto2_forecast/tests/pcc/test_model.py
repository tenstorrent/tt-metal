# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Full-model forecast PCC on device against the reference goldens (``TtToto2.forecast``, one forward).

    pcc_normalized  head output (9, B, V, h, 32) in asinh-normalized space vs the hook on ``m.output_head``
    pcc_final       quantiles (9, B, V, H) after sinh, rescale, clamp and sort vs the reference ``forecast``

Floors (CONTRACT section 3): accuracy 0.999 / 0.99; performance 0.995 / 0.98.
Cases take every path: clean (causal SDPA), missing patches (group mask), V = 8 sharing one series id (variate
SDPA without a mask), V = 4 with two series ids (variate mask), and context 8192 (S = 288, 32 read-out tokens).
"""

from __future__ import annotations

import gc
import pytest

from models.experimental.toto2_forecast.tests.golden.goldens import have_golden, load_golden, pcc

# One device per module, so a loaded model (2.5B: about a minute) serves every case that uses it.
pytestmark = pytest.mark.use_module_device

FLOORS = {"accuracy": (0.999, 0.99), "performance": (0.995, 0.98)}
# The unit-test checkpoints (no contract floor): at context 8192 (32 read-out tokens, whose head outputs vary
# 3-10x less than at the last context token) the bf16 stream error of these small, high-tau models shows up as
# 4m 0.9975 / 22m 0.9989 normalized PCC; fp32 accumulation everywhere lifts 22m to 0.99956 (ttq job 448).
SMALL_FLOORS = {"accuracy": (0.995, 0.99), "performance": (0.99, 0.98)}
SMALL = ("uni_4x2048", "missing_5x2048", "mv_2x4x2048", "long_2x8192")
FULL = ("uni_16x2048", "missing_16x2048", "mv_4x8x2048")
# Ordered by checkpoint and precision, so each process holds one model at a time.
CASES = [
    *((c, n, "accuracy") for c in ("4m", "22m") for n in SMALL),
    *(("22m", n, "performance") for n in SMALL),
    *(("313m", n, "accuracy") for n in FULL),
    *(("313m", n, "performance") for n in FULL[:1]),
    *(("2.5B-FT", n, "accuracy") for n in FULL),
    *(("2.5B-FT", n, "performance") for n in FULL[:1]),
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


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("ckpt,case,precision_name", CASES, ids=[f"{c}-{n}-{p}" for c, n, p in CASES])
def test_model(device, ckpt, case, precision_name):
    if not have_golden(ckpt, case):
        pytest.skip(f"golden {ckpt}/{case} not generated yet")
    golden = load_golden(ckpt, case)
    model = tt_model(device, ckpt, precision_name)
    q, head = run_forecast(model, golden)
    cap = golden["captures"]
    pcc_normalized, pcc_final = pcc(head, cap["head_out"]), pcc(q, cap["q"])
    floor_n, floor_f = (SMALL_FLOORS if ckpt in ("4m", "22m") else FLOORS)[precision_name]
    print(f"\n[MODEL] {ckpt}/{case}/{precision_name}: pcc_normalized={pcc_normalized:.6f} pcc_final={pcc_final:.6f}")
    assert pcc_normalized >= floor_n and pcc_final >= floor_f, (pcc_normalized, pcc_final)
