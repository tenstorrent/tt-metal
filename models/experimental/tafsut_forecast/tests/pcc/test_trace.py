# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Fresh inputs through captured traces (skill §1 C): new values through an existing trace and cached constants,
two shapes in one forecast (median feedback), and a shape captured after another trace already exists."""

import pytest
import torch

from models.experimental.tafsut_forecast.tests.golden.goldens import load_golden
from models.experimental.tafsut_forecast.tests.helpers import pcc
from models.experimental.tafsut_forecast.tt.model import TtTafsut
from models.experimental.tafsut_forecast.tt.trace import TracedForward


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 256 * 1024**2}], indirect=True)
def test_trace_fresh_inputs(device):
    tt = TtTafsut(device)
    traced = TracedForward(tt)
    a, b = load_golden("model16"), load_golden("fallback")  # same shape (16 x 2048), different values
    h = load_golden("horizon4")  # 4 x 2048 -> 1100: two shapes, the second captured after the first
    for g in (a, b, h, a, h, b):  # every capture replayed on values it was not captured with
        heads = []
        q = traced.forecast(g["x"], g["horizon"], heads=heads)
        for k, f in enumerate(g["forwards"]):
            p = pcc(heads[k], f["head"])
            print(f"[trace] rows {g['x'].shape[0]} forward {k}: head pcc {p:.6f}")
            assert p > 0.9995
        assert pcc(q, g["quantiles"]) > 0.999
        assert torch.isfinite(q).all()
    traced.release()
