# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Fused sLSTM kernel (``ops/slstm_step``) against the op-level loop (``tt/layers.py: slstm_recurrence``).

Same WxT for both (golden block input through the layer's W_in, rows tiled with noise up to B); per-step PCC
of y, c, n and m must be >= 0.9999 (D2), and no worse than the loop's own PCC against the fp32 upstream goldens.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.tirex2_forecast.tests.golden.goldens import golden, have_goldens
from models.experimental.tirex2_forecast.tests.pcc.helpers import host_weights, pcc

pytestmark = [
    pytest.mark.skipif(not have_goldens(), reason="goldens not generated"),
    pytest.mark.use_module_device({"l1_small_size": 16384}),
]

FLOOR = 0.9999


@pytest.fixture(scope="module")
def tt_model(request):
    device = request.getfixturevalue("_device_module_impl")
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2

    model = TtTiRex2(device, host_weights())
    yield model
    model.release()


def _wx(device, tt_model, b: int, B: int):
    import ttnn

    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.common import upload

    g = golden("slstm_short")
    w = host_weights().blocks[b]
    x = g[f"block{b}.x"] / w.w_norm_lstm
    rows, steps, _ = x.shape
    gen = torch.Generator().manual_seed(b)
    reps = -(-B // rows)
    xb = x.repeat(reps, 1, 1)[:B]
    xb[rows:] += 0.3 * torch.randn(xb[rows:].shape, generator=gen)
    cell = tt_model.weights.blocks[b].cell
    xT = upload(device, xb.transpose(0, 1).reshape(steps, 1, B, 512))
    WxT = ttnn.add(layers.linear(xT, cell.w_in, tt_model.precision, dtype=ttnn.float32), cell.bias)
    return WxT, cell, steps


@pytest.mark.parametrize("B", [32, 256])
@pytest.mark.parametrize("b", [1, 11])
def test_slstm_step_op(device, tt_model, b, B):
    import ttnn

    from models.experimental.tirex2_forecast.ops.slstm_step.op import slstm_steps
    from models.experimental.tirex2_forecast.tt import layers

    WxT, cell, steps = _wx(device, tt_model, b, B)
    record = []
    layers.slstm_recurrence(device, WxT, cell.r, steps, tt_model.consts, record=record)
    ys, cs, ns, ms = slstm_steps(WxT, cell.r, steps, record=True)
    got = [ttnn.to_torch(t).float()[:, 0] for t in (ys, cs, ns, ms)]  # [T, B, 512] each
    worst = 1.0
    for j, nm in enumerate("ycnm"):
        per_step = [pcc(record[t][j], got[j][t]) for t in range(steps)]
        lo = min(per_step)
        print(f"block{b} B={B} {nm}: min step pcc {lo:.7f} (step {per_step.index(lo)}), last {per_step[-1]:.7f}")
        worst = min(worst, lo)
    assert worst >= FLOOR
