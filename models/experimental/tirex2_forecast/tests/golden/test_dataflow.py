# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU check: the torch dataflow re-implementation (``reference/dataflow.py``) against the upstream fp32 goldens.

No device. Every module is fed the golden input and compared with the golden output, so errors do not
compound; the end-to-end tests then run the whole dataflow from the golden inputs. The bound is
``max|got - golden| <= RTOL * max|golden|`` per tensor.
"""

from __future__ import annotations

import json

import pytest
import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tests.golden.goldens import golden, golden_path, have_goldens

RTOL = 1e-5

pytestmark = pytest.mark.skipif(not have_goldens(), reason="goldens not generated (tests/golden/make_goldens.py)")


def rel_err(got: torch.Tensor, exp: torch.Tensor) -> float:
    got, exp = got.double(), exp.double()
    return float((got - exp).abs().max() / exp.abs().max().clamp_min(1e-30))


def check(name: str, got: torch.Tensor, exp: torch.Tensor, rtol: float = RTOL):
    assert got.shape == exp.shape, f"{name}: shape {tuple(got.shape)} != {tuple(exp.shape)}"
    err = rel_err(got, exp)
    print(f"[dataflow] {name}: rel err {err:.2e}")
    assert err <= rtol, f"{name}: rel err {err:.3e} > {rtol}"


@pytest.fixture(scope="module")
def weights():
    return df.build_weights(df.load_state_dict(df.default_checkpoint()))


@pytest.fixture(scope="module")
def single():
    return golden("single")


@pytest.mark.parametrize(
    "name", ["single", "model16", "multivar4", "bench_paper", "bench_long_context", "bench_multivariate"]
)
def test_prepare(name):
    g = golden(name)
    ctx = g["context"]
    ctx = ctx.reshape(-1, ctx.shape[-1])
    x = df.pad_context(ctx, g["horizon"])
    if "x_padded" in g:
        assert torch.equal(torch.isnan(x), torch.isnan(g["x_padded"]))
        check(f"{name}.x_padded", torch.nan_to_num(x), torch.nan_to_num(g["x_padded"]), 0)
    loc, scale = df.scaler(x)
    check(f"{name}.loc", loc, g["loc"], 1e-6)
    check(f"{name}.scale", scale, g["scale"], 1e-6)
    if "tokens" in g:
        check(f"{name}.tokens", df.tokenize(x, loc, scale), g["tokens"])


def test_embed(weights, single):
    check("embed", df.residual_block(single["tokens"], weights.embed), single["embed"])


@pytest.mark.parametrize("b", range(df.NUM_BLOCKS))
def test_block(weights, single, b):
    w = weights.blocks[b]
    x_in = single["embed"] if b == 0 else single[f"block{b - 1}.out"]
    xn = df.rms_norm(x_in, w.w_norm_lstm)
    check(f"block{b}.norm_lstm", xn, single[f"block{b}.norm_lstm"])
    cell = df.mlstm_cell if df.is_mlstm(b) else df.slstm_cell
    check(f"block{b}.cell", cell(single[f"block{b}.norm_lstm"], w.cell), single[f"block{b}.cell_out"])
    check(f"block{b}.out", df.block(x_in, w), single[f"block{b}.out"])


@pytest.mark.parametrize("b", range(0, df.NUM_BLOCKS, 2))
def test_mlstm_internals(weights, single, b):
    w = weights.blocks[b].cell
    cap = {}
    df.mlstm_cell(single[f"block{b}.norm_lstm"], w, capture=cap)
    for k in "qkvif":
        check(f"block{b}.{k}", cap[k], single[f"block{b}.mlstm.{k}"])
    q, k, v, i, f = (single[f"block{b}.mlstm.{n}"] for n in "qkvif")
    h, states = df.mlstm_chunkwise(q, k, v, i, f, chunk=64, return_states=True)
    check(f"block{b}.h", h, single[f"block{b}.mlstm.h"])
    chunks = golden("mlstm_chunks")
    for c in range(len(states) - 1):  # the state after the padded last chunk depends on the pad values only
        C, n, m = states[c]
        check(f"block{b}.C[{c}]", C, chunks[f"block{b}.C"][:, :, c], 1e-5)
        check(f"block{b}.n[{c}]", n, chunks[f"block{b}.n"][:, :, c], 1e-5)
        check(f"block{b}.m[{c}]", m, chunks[f"block{b}.m"][:, :, c], 1e-5)


@pytest.mark.parametrize("chunk", [32, 96])
def test_mlstm_chunk_size_invariance(single, chunk):
    """The running max (and so the eps term) does not depend on the chunk size: any L reproduces L = 64.

    The identity is exact in real arithmetic; in fp32 the per-chunk exponent bookkeeping rounds differently
    (up to ~2e-5 relative at L = 32), hence the looser bound.
    """
    for b in range(0, df.NUM_BLOCKS, 2):
        q, k, v, i, f = (single[f"block{b}.mlstm.{n}"] for n in "qkvif")
        got = df.mlstm_chunkwise(q, k, v, i, f, chunk=chunk)
        check(f"block{b}.h[L={chunk}]", got, single[f"block{b}.mlstm.h"], 5e-5)


@pytest.mark.parametrize("b", range(1, df.NUM_BLOCKS, 2))
def test_slstm_short(weights, b):
    g = golden("slstm_short")
    w = weights.blocks[b].cell
    Wx = df.slstm_input(g[f"block{b}.x"], w)
    gWx = g[f"block{b}.Wx"]  # [B, T, 4 gates, 4 heads, 128]
    bias = w.bias.reshape(4, 4, 128)
    check(f"block{b}.Wx", Wx, (gWx + bias).reshape(*gWx.shape[:2], -1))
    _, states = df.slstm_recurrence(Wx, w.r)
    gs = g[f"block{b}.states"]  # [4 (y, c, n, m), B, T, 4, 128]
    for j, nm in enumerate("ycnm"):
        got = torch.stack([s[j] for s in states], dim=1)
        check(f"block{b}.{nm}", got, gs[j].reshape(*gs.shape[1:3], -1))
    check(f"block{b}.out", df.slstm_cell(g[f"block{b}.x"], w), g[f"block{b}.out"])


@pytest.mark.parametrize("b", range(1, df.NUM_BLOCKS, 2))
def test_slstm_full_states(weights, single, b):
    w = weights.blocks[b].cell
    Wx = single[f"block{b}.slstm.Wx"]
    bias = w.bias.reshape(4, 4, 128)
    _, states = df.slstm_recurrence((Wx + bias).reshape(*Wx.shape[:2], -1), w.r)
    gs = single[f"block{b}.slstm.states"]
    for j, nm in enumerate("ycnm"):
        got = torch.stack([s[j] for s in states], dim=1)
        check(f"block{b}.{nm}[285]", got, gs[j].reshape(*gs.shape[1:3], -1))


def test_head(weights, single):
    x = df.t5_norm(single["block11.out"], weights.w_out_norm)
    check("stack_out_norm", x, single["stack_out_norm"])
    check("head", df.residual_block(single["stack_out_norm"], weights.head), single["head_out"])


@pytest.mark.parametrize("name", ["single", "model16"])
def test_forward(weights, name):
    g = golden(name)
    prep = df.prepare(g["context"].reshape(-1, g["context"].shape[-1]), g["horizon"])
    cap = {}
    head = df.forward(weights, prep, capture=cap)
    for b in range(df.NUM_BLOCKS):
        check(f"{name}.block{b}.out", cap[f"block{b}"]["out"], g[f"block{b}.out"], 1e-4)
    check(f"{name}.head", head, g["head_out"], 1e-4)
    q = df.postprocess(head[:, df.readout_tokens(g["horizon"])], prep)
    check(f"{name}.q_one", q, g["q_one"], 1e-4)
    q_full = df.predict(weights, g["context"].reshape(-1, g["context"].shape[-1]), g["horizon"], sign_flip=True)
    check(f"{name}.q_full", q_full, g["q_full"], 1e-4)


def test_postprocess_bench():
    """Host post-processing alone: golden head output at the read-out tokens -> golden q_one."""
    for name in ("bench_paper", "bench_long_context", "bench_multivariate"):
        g = golden(name)
        ctx = g["context"].reshape(-1, g["context"].shape[-1])
        rows = ctx.shape[0]
        if name == "bench_multivariate":
            groups = g["context"].shape[0]
            per = g["context"].shape[1]
            gid = torch.arange(groups).repeat_interleave(per)
            tmask = (torch.arange(rows) % per) == 0
        else:
            gid, tmask = None, None
        prep = df.prepare(ctx, g["horizon"], gid, tmask)
        assert g["readout_tokens"] == df.readout_tokens(g["horizon"])
        check(f"{name}.q_one(post)", df.postprocess(g["head_out"], prep), g["q_one"], 1e-6)


def test_multivariate(weights):
    g = golden("multivar4")
    groups, per, T = g["context"].shape
    ctx = g["context"].reshape(-1, T)
    gid = torch.arange(groups).repeat_interleave(per)
    tmask = (torch.arange(groups * per) % per) == 0
    assert torch.equal(g["target_mask"], tmask)
    assert torch.equal(g["group_vector"].long() - 1, gid)
    t0, t1 = g["variate_tokens"]
    for b in (0, 11):
        x_in = g[f"block{b}.variate_in"].transpose(0, 1)  # [rows, tokens, 512]
        got = df.variate_mixer(x_in, weights.blocks[b].variate, gid, tmask)
        check(f"block{b}.variate", got.transpose(0, 1), g[f"block{b}.variate_out"])
    prep = df.prepare(ctx, g["horizon"], gid, tmask)
    cap = {}
    head = df.forward(weights, prep, capture=cap)
    for b in (0, 11):
        check(f"multivar4.block{b}.out", cap[f"block{b}"]["out"], g[f"block{b}.out"], 1e-4)
    check("multivar4.head", head, g["head_out"], 1e-4)
    check("multivar4.q_one", df.postprocess(head[:, df.readout_tokens(g["horizon"])], prep), g["q_one"], 1e-4)


def test_pad_hypothesis():
    report = json.loads(golden_path("pad_check.json").read_text())
    assert report["confirmed_1e-6"], report["worst_max_abs_diff"]
