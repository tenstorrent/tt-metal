# One-token GatedDeltaNet step (tt/gdn_step) against the fp32 torch reference, with layer 0's conv / gate / norm
# weights. Activation and conv-state padding rows hold garbage on purpose: the op must read row 0 (rows 0-2) only.
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import ttnn
from models.demos.blackhole.bonsai2.tt.gdn_step.gdn_step import GDNStep, reference_step

HF = Path("/local/ttuser/jerrywang/bonsai/out")
P = "model.language_model.layers.0.linear_attn."


def _w(name):
    wm = json.load(open(HF / "model.safetensors.index.json"))["weight_map"]
    with safe_open(str(HF / wm[P + name]), "pt") as f:
        return f.get_tensor(P + name).float()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _row0(v, device, garbage=True):
    """[N] -> [1, 32, N] bf16 TILE with v in row 0 and garbage (or zeros) below."""
    t = torch.randn(1, 32, v.shape[-1]) * 50 if garbage else torch.zeros(1, 32, v.shape[-1])
    t[0, 0] = v
    return ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("steps", [1, 3])
def test_gdn_step(device, steps):
    torch.manual_seed(0)
    nk, hk, hd, nv = 16, 128, 128, 48
    dq = 2 * nk * hk + nv * hd
    taps = _w("conv1d.weight")[:, 0, :].T.contiguous()  # [4, dq]
    taps = taps.bfloat16().float()
    dt_bias, a_neg = _w("dt_bias"), -torch.exp(_w("A_log"))
    norm_w = _w("norm.weight").bfloat16().float()
    step = GDNStep(device, taps, dt_bias, a_neg, norm_w, nk, hk, hd, eps_o=1e-6)
    if os.environ.get("STEP_ZONES"):
        step.compute_defines = (("STEP_ZONES", "1"),)

    conv = (torch.randn(3, dq) * 0.5).bfloat16().float()
    S = (torch.randn(nv, hk, hd) * 0.05).bfloat16().float()
    conv_t = torch.randn(1, 32, dq) * 50
    conv_t[0, :3] = conv
    conv_dev = ttnn.from_torch(conv_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    S_dev = ttnn.from_torch(
        S.reshape(1, nv, hk, hd).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    for s in range(steps):
        qkv = (torch.randn(dq) * 0.5).bfloat16().float()
        z = (torch.randn(nv * hd) * 0.5).bfloat16().float()
        ab = (torch.randn(2 * nv) * 1.0).bfloat16().float()
        out = step(_row0(qkv, device), _row0(ab, device), _row0(z, device), conv_dev, S_dev)
        r_out, S, conv = reference_step(qkv, ab, z, conv, S, taps, dt_bias, a_neg, norm_w, nk, hk, hd, 1e-6)
        S = S.bfloat16().float()  # the device keeps the state in bf16
        d_out = ttnn.to_torch(out).float().reshape(-1, nv * hd)[0]
        d_S = ttnn.to_torch(S_dev).float().reshape(nv, hk, hd)
        d_conv = ttnn.to_torch(conv_dev).float()[0, :3]
        print(
            f"step {s}: out rel {_rel(d_out, r_out):.5f}  S rel {_rel(d_S, S):.5f}  conv rel {_rel(d_conv, conv):.6f}"
            f"  conv row3 max {ttnn.to_torch(conv_dev).float()[0, 3].abs().max().item():.3g}"
        )
        assert _rel(d_conv, conv) == 0.0
        assert _rel(d_S, S) < 0.01 and _rel(d_out, r_out) < 0.02
        S = d_S  # continue from the device state (bf16 rounding)
