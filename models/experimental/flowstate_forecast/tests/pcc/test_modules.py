# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Device module tests that assert, both precisions (CONTRACT §3, PLAN v1.24, D3 item 8).

Each module runs alone on its golden input and is compared with its golden output: embedding >= 0.9999, every
other module >= 0.999 (CONTRACT §3, the doc's numbers).
    modules.pt  random inputs: embedding; layers 0 and 5 (S5 block, layer) at L = 96 (s = 1) and L = 70 (s = 6,
                non-aligned), the last layer at its read rows; MLP 0; the folded decoder (s = 1, P = 6)
    single.pt   a real series (L = 2106, s = 1): every layer on the golden output of the layer before (the stacked
                PCCs of the full forward are in test_model.py, report-only), and the S5 block and MLP of layers 0, 5
Every layer test also runs on the forward's own path (``_fused``: residual_input_tensor, D5 item 4).
Results go to FLOWSTATE_PCC_OUT/modules_<precision>.json.
"""

import json
import os

import pytest
import torch
import ttnn

from models.experimental.flowstate_forecast.reference import weights
from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS
from models.experimental.flowstate_forecast.tests.pcc import metrics
from models.experimental.flowstate_forecast.tt.model import TILE, FlowStateTT

OUT = os.environ.get("FLOWSTATE_PCC_OUT", "/tmp/flowstate_pcc")
FLOOR, EMBED_FLOOR = 0.999, 0.9999


def up(device, t, dtype=ttnn.bfloat16):
    """(L, B, C) -> (1, 1, L * Bp, C) on device, B padded to a tile."""
    L, B, C = t.shape
    Bp = -(-B // TILE) * TILE
    z = torch.zeros(L, Bp, C)
    z[:, :B] = t.float()
    return ttnn.from_torch(z.view(1, 1, L * Bp, C), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def down(t, B, C=512):
    x = ttnn.to_torch(t).float()
    return x.reshape(-1, -(-B // TILE) * TILE, x.shape[-1])[:, :B, :C]


def run_layer(tt, i, u, s, rows, P=6, fused=False):
    """Layer i on u (L, B, 512): the S5 block, MLP and layer outputs over the last `rows` rows. fused: the forward's
    own path (the MLP residual in the layer norm's residual_input_tensor, R2-2; D5 item 4), the layer output only."""
    L, B, _ = u.shape
    Bp = -(-B // TILE) * TILE
    consts = tt.constants(s, L, P)[0][i]
    parts = None if fused else {}
    out = tt.layer(i, up(tt.device, u), consts, L, Bp, L - rows if i == len(tt.layers) - 1 else 0, parts=parts)
    if fused:
        return {"layer": down(out, B)}
    return {"ssm": down(parts["ssm"], B), "mlp": down(parts["mlp"], B), "layer": down(out, B)}


@pytest.fixture(scope="module")
def ref():
    return weights.load()


@pytest.fixture(params=["accuracy", "performance"])
def tt(request, device, ref):
    return FlowStateTT(ref, device, request.param)


@pytest.fixture
def results(tt):
    res = {}
    yield res
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/modules_{tt.p.name}.json"
    old = json.load(open(path)) if os.path.isfile(path) else {}
    with open(path, "w") as f:
        json.dump({**old, **res}, f, indent=1)


def _assert(results, tt, name, got, exp, floor=FLOOR):
    p = metrics.pearson(exp, got)
    results[name] = p
    print(f"module {tt.p.name} {name}: pcc {p:.6f} (floor {floor})")
    assert p >= floor, (tt.p.name, name, p)


def test_embedding(tt, results):
    g = torch.load(GOLDENS / "modules.pt", weights_only=True)
    tok = g["embed_in"]  # (96, 2, 2)
    t = torch.zeros(*tok.shape[:2], TILE)
    t[..., :2] = tok
    p = tt.p
    h = ttnn.linear(
        up(tt.device, t, ttnn.float32), tt.embed_w, bias=tt.embed_b, dtype=ttnn.bfloat16, compute_kernel_config=p.embed
    )
    _assert(results, tt, "embed_L96", down(h, 2), g["embed_out"], EMBED_FLOOR)


@pytest.mark.parametrize("L,s", [(96, 1.0), (70, 6.0)])
@pytest.mark.parametrize("i", [0, 5])
def test_random_layer(tt, results, L, s, i):
    g = torch.load(GOLDENS / "modules.pt", weights_only=True)
    rows = g[f"ssm{i}_L{L}"].shape[0]
    o = run_layer(tt, i, g[f"u_L{L}"], s, rows)
    _assert(results, tt, f"ssm{i}_L{L}", o["ssm"], g[f"ssm{i}_L{L}"])
    _assert(results, tt, f"layer{i}_L{L}", o["layer"], g[f"layer{i}_L{L}"])


@pytest.mark.parametrize("L,s", [(96, 1.0), (70, 6.0)])
@pytest.mark.parametrize("i", [0, 5])
def test_random_layer_fused(tt, results, L, s, i):
    """D5 item 4: the forward's residual_input_tensor path, which the parts= tests above do not run."""
    g = torch.load(GOLDENS / "modules.pt", weights_only=True)
    o = run_layer(tt, i, g[f"u_L{L}"], s, g[f"layer{i}_L{L}"].shape[0], fused=True)
    _assert(results, tt, f"layer{i}_L{L}_fused", o["layer"], g[f"layer{i}_L{L}"])


@pytest.mark.parametrize("L", [96, 70])
def test_random_mlp(tt, results, L):
    g = torch.load(GOLDENS / "modules.pt", weights_only=True)
    z = tt.mlp_block(0, up(tt.device, g[f"mlp_in_L{L}"]))
    _assert(results, tt, f"mlp0_L{L}", down(z, 2), g[f"mlp0_L{L}"])


def test_decoder(tt, results):
    g = torch.load(GOLDENS / "modules.pt", weights_only=True)
    o = g["dec_in"]  # (13, 2, 512), s = 1, P = 6
    _, (dw, db) = tt.constants(1.0, o.shape[0], 6)
    dec = ttnn.linear(up(tt.device, o), dw, bias=db, dtype=ttnn.float32, compute_kernel_config=tt.p.decoder)
    got = down(dec, 2, 9 * 6).reshape(o.shape[0], 2, 9, 6, 1)
    _assert(results, tt, "decoder", got, g["dec_out"])


@pytest.mark.parametrize("i", range(6))
def test_single_layer(tt, results, i):
    """Layer i of the real `single` series on the golden output of layer i - 1 (the embedding for i = 0)."""
    g = torch.load(GOLDENS / "single.pt", weights_only=True)
    u = g["embed"] if i == 0 else g[f"layer{i - 1}"]
    rows = g[f"layer{i}"].shape[0]
    o = run_layer(tt, i, u, g["s"], rows, g["fl"])
    _assert(results, tt, f"single_layer{i}", o["layer"], g[f"layer{i}"])
    if f"ssm{i}" in g:
        _assert(results, tt, f"single_ssm{i}", o["ssm"], g[f"ssm{i}"])
        _assert(results, tt, f"single_mlp{i}", o["mlp"], g[f"mlp{i}"])


@pytest.mark.parametrize("i", range(6))
def test_single_layer_fused(tt, results, i):
    """test_single_layer on the forward's residual_input_tensor path (D5 item 4)."""
    g = torch.load(GOLDENS / "single.pt", weights_only=True)
    u = g["embed"] if i == 0 else g[f"layer{i - 1}"]
    o = run_layer(tt, i, u, g["s"], g[f"layer{i}"].shape[0], g["fl"], fused=True)
    _assert(results, tt, f"single_layer{i}_fused", o["layer"], g[f"layer{i}"])
