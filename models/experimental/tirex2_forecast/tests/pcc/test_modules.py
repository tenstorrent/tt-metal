# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC of the TTNN TiRex-2 layers against the upstream fp32 CPU goldens (real weights).

Every module gets the golden input and is compared with the golden output. Floors (CONTRACT section 3):
embedding, FFN, norms, head and each mLSTM block >= 0.999; each sLSTM block >= 0.995; sLSTM per-step
states over 32 steps >= 0.9999.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tests.golden.goldens import golden, have_goldens
from models.experimental.tirex2_forecast.tests.pcc.helpers import (
    from_device_seq,
    host_weights,
    report,
    to_device_seq,
)

pytestmark = [
    pytest.mark.skipif(not have_goldens(), reason="goldens not generated"),
    pytest.mark.use_module_device({"l1_small_size": 16384}),
]

T = df.NUM_TOKENS


@pytest.fixture(scope="module")
def tt_model(request):
    device = request.getfixturevalue("_device_module_impl")
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2

    model = TtTiRex2(device, host_weights())
    yield model
    model.release()


def _block_input(g, b):
    return g["embed"] if b == 0 else g[f"block{b - 1}.out"]


def test_embed(device, tt_model):
    from models.experimental.tirex2_forecast.tt.model import SEQ

    g = golden("model16")
    x = to_device_seq(device, g["tokens"], SEQ)
    out = tt_model.embed(type("I", (), {"tokens": x})())
    got = from_device_seq(out, 16, T)
    assert report("embed", g["embed"], got, 0.999) >= 0.999


@pytest.mark.parametrize("b", [0, 5, 11])
def test_ffn(device, tt_model, b):
    """FFN residual on the golden time-mixer output (x + cell_out) of the single-series run."""
    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.model import SEQ

    g = golden("single")
    x_mid = _block_input(g, b) + g[f"block{b}.cell_out"]
    out = layers.ffn_residual(to_device_seq(device, x_mid, SEQ), tt_model.weights.blocks[b].ffn, tt_model.precision)
    got = from_device_seq(out, 1, T)
    assert report(f"ffn{b}", g[f"block{b}.out"], got, 0.999) >= 0.999


def test_rms_norm(device):
    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.model import SEQ

    g = golden("model16")
    w = host_weights()
    x = g["block6.out"]
    got = from_device_seq(layers.rms_norm(to_device_seq(device, x, SEQ), df.NORM_EPS), 16, T)
    exp = df.rms_norm(x, torch.ones(512))
    assert report("rms_norm", exp, got, 0.999) >= 0.999
    got = from_device_seq(layers.rms_norm(to_device_seq(device, x, SEQ), df.OUT_NORM_EPS), 16, T) * w.w_out_norm
    assert report("stack_out_norm", df.t5_norm(x, w.w_out_norm), got, 0.999) >= 0.999


@pytest.mark.parametrize("b", range(0, df.NUM_BLOCKS, 2))
def test_mlstm_cell(device, tt_model, b):
    """mLSTM time mixer (norm + cell + residual) on the single series: ``x + cell_out``."""
    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.model import SEQ

    g = golden("single")
    x = _block_input(g, b)
    dbg = {}
    layers.mlstm_layer(
        device,
        to_device_seq(device, x, SEQ),
        tt_model.weights.blocks[b].cell,
        tt_model.precision,
        tt_model.consts,
        debug=dbg,
    )
    got = dbg["y"][:1, 0, :T]  # the cell output before the (bf16) residual add
    assert report(f"mlstm{b}.cell", g[f"block{b}.cell_out"], got, 0.999) >= 0.999


@pytest.mark.parametrize("b", [0, 10])
def test_mlstm_block16(device, tt_model, b):
    """Full mLSTM block (time mixer + FFN) at 16 series."""
    from models.experimental.tirex2_forecast.tt.model import SEQ

    g = golden("model16")
    x = to_device_seq(device, _block_input(g, b), SEQ)
    inputs = type("I", (), {"vmask": None, "keep": None, "cut": 0})()
    got = from_device_seq(tt_model.run_block(tt_model.weights.blocks[b], x, inputs), 16, T)
    assert report(f"block{b}[16]", g[f"block{b}.out"], got, 0.999) >= 0.999


@pytest.mark.parametrize("b", range(1, df.NUM_BLOCKS, 2))
def test_slstm_short_states(device, tt_model, b):
    """32 sLSTM steps from a zero state: per-step y, c, n, m against the upstream recurrence."""
    import ttnn

    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.common import upload

    g = golden("slstm_short")
    w = host_weights().blocks[b]
    cell = tt_model.weights.blocks[b].cell
    x = g[f"block{b}.x"] / w.w_norm_lstm  # gain-free normed input (the gain is folded into W_in)
    rows, steps, _ = x.shape
    xT = upload(device, torch.cat([x, torch.zeros(32 - rows, steps, 512)]).transpose(0, 1).reshape(steps, 1, 32, 512))
    WxT = ttnn.add(layers.linear(xT, cell.w_in, tt_model.precision, dtype=ttnn.float32), cell.bias)
    record = []
    layers.slstm_recurrence(device, WxT, cell.r, steps, tt_model.consts, record=record)
    gs = g[f"block{b}.states"]  # [4, rows, steps, 4, 128]
    worst = 1.0
    for j, nm in enumerate("ycnm"):
        got = torch.stack([r[j][:rows] for r in record], dim=1)
        p = report(f"slstm{b}.{nm}[32 steps]", gs[j].reshape(rows, steps, -1), got, 0.9999)
        worst = min(worst, p)
    assert worst >= 0.9999


@pytest.mark.parametrize("b", range(1, df.NUM_BLOCKS, 2))
def test_slstm_cell(device, tt_model, b):
    """sLSTM time mixer over all 285 steps on the single series: ``x + cell_out``."""
    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.model import SEQ

    g = golden("single")
    x = _block_input(g, b)
    dbg = {}
    layers.slstm_layer(
        device,
        to_device_seq(device, x, SEQ),
        tt_model.weights.blocks[b].cell,
        tt_model.precision,
        tt_model.weights.head_mean,
        T,
        tt_model.consts,
        debug=dbg,
    )
    got = dbg["y"][:1, 0, :T]  # the cell output (group-normed) before the residual add
    assert report(f"slstm{b}.cell", g[f"block{b}.cell_out"], got, 0.995) >= 0.995


@pytest.mark.parametrize("b", [1, 11])
def test_slstm_block16(device, tt_model, b):
    """Full sLSTM block at 16 series; block 11 is the deepest."""
    from models.experimental.tirex2_forecast.tt.model import SEQ

    g = golden("model16")
    x = to_device_seq(device, _block_input(g, b), SEQ)
    inputs = type("I", (), {"vmask": None, "keep": None, "cut": 0})()
    got = from_device_seq(tt_model.run_block(tt_model.weights.blocks[b], x, inputs), 16, T)
    assert report(f"block{b}[16]", g[f"block{b}.out"], got, 0.995) >= 0.995


def test_head(device, tt_model):
    from models.experimental.tirex2_forecast.tt.model import SEQ, WINDOW

    g = golden("model16")
    out = tt_model.head(to_device_seq(device, g["block11.out"], SEQ))
    import ttnn

    got = ttnn.to_torch(out).float()[:16, 0]
    exp = g["head_out"][:, WINDOW[0] : min(WINDOW[1], T)]
    assert report("head", exp, got[:, : exp.shape[1]], 0.999) >= 0.999


@pytest.mark.parametrize("b", [0, 11])
def test_variate(device, tt_model, b):
    """Variate attention across 4 groups x 8 rows (1 target + 7 past covariates) on a token range."""
    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.common import pad_to, upload
    from models.experimental.tirex2_forecast.tt.model import pack_rows, variate_masks

    import ttnn

    g = golden("multivar4")
    groups, per, _ = g["context"].shape
    gid = torch.arange(groups).repeat_interleave(per)
    tm = (torch.arange(groups * per) % per) == 0
    x = g[f"block{b}.variate_in"].transpose(0, 1)  # [rows, tokens, 512]
    rows, tokens, _ = x.shape
    order = pack_rows(gid)
    assert torch.equal(order, torch.arange(rows))
    vmask, keep = variate_masks(order, gid, tm)
    xd = to_device_seq(device, x, pad_to(tokens))
    dbg = {}
    out = layers.variate_block(
        device,
        xd,
        tt_model.weights.blocks[b].variate,
        tt_model.precision,
        upload(device, vmask, dtype=ttnn.float32),
        upload(device, keep),
        debug=dbg,
    )
    got = from_device_seq(out, rows, tokens)
    exp = g[f"block{b}.variate_out"].transpose(0, 1)
    assert report(f"variate{b}", exp, got, 0.999) >= 0.999
    # the attention update alone, against the dataflow on the golden input (validated against upstream to 1e-7)
    w = host_weights().blocks[b].variate
    xt = x.transpose(0, 1)
    q, k, v = (df.linear(df.rms_norm(xt, w.w_norm_attn), m) for m in (w.w_q, w.w_k, w.w_v))
    heads = lambda t: t.reshape(tokens, rows, 4, -1).transpose(1, 2)
    _, allowed = df.group_layout(gid, tm)
    s = (heads(q) @ heads(k).transpose(-1, -2)) / 128**0.5
    a = torch.softmax(s.masked_fill(~allowed, float("-inf")), -1) @ heads(v)
    attn = df.linear(a.transpose(1, 2).reshape(tokens, rows, 512), w.w_o)
    got_attn = dbg["attn"][:tokens, 0, :rows]
    assert report(f"variate{b}.attn", attn, got_attn, 0.999) >= 0.999


def test_log_sigmoid_fp32(device):
    """The gates' log-sigmoid runs in fp32 as ``log(sigmoid(max(x, -80)))`` (``ttnn.log_sigmoid`` and a bf16
    softplus are inaccurate; PLAN section 9). Check it over |x| <= 40, past the gate ranges seen in the goldens."""
    import ttnn

    from models.experimental.tirex2_forecast.tt import layers
    from models.experimental.tirex2_forecast.tt.common import upload

    x = torch.linspace(-40, 40, 32 * 4096).reshape(1, 1, 128, 1024)
    got = ttnn.to_torch(layers.log_sigmoid(upload(device, x, dtype=ttnn.float32))).double()
    exp = torch.nn.functional.logsigmoid(x.double())
    err = float((got - exp).abs().max())
    print(f"[PCC] log_sigmoid fp32: max abs err {err:.3e} over [-40, 40]")
    assert err <= 1e-5
