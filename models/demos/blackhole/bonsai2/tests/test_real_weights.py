# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ternary decode linear on Bonsai's real layer-0 MLP weights: y = W'(H D x) on the device against the dense
converted checkpoint's y = W_dense x (bf16 weights, which rounded W' H D)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3
from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import BonsaiGGUF

GGUF = Path(os.environ.get("BONSAI_GGUF", "/local/ttuser/jerrywang/bonsai/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
HF = Path(os.environ.get("HF_MODEL", "/local/ttuser/jerrywang/bonsai/out"))


def _dense(name):
    wm = json.load(open(HF / "model.safetensors.index.json"))["weight_map"]
    with safe_open(str(HF / wm[name]), "pt") as f:
        return f.get_tensor(name).float()


@pytest.mark.parametrize(
    "gname, hname",
    [
        ("blk.0.ffn_gate.weight", "model.language_model.layers.0.mlp.gate_proj.weight"),
        ("blk.0.ffn_down.weight", "model.language_model.layers.0.mlp.down_proj.weight"),
    ],
)
def test_real_layer0(device, gname, hname):
    g = BonsaiGGUF(GGUF)
    t, s = g.ternary(gname)
    K, N = t.shape
    x = torch.randn(K) * 0.5
    xr = torch.from_numpy(g.rotate(x.numpy().astype(np.float64)).astype(np.float32))
    # host check of the fold: W' (H D x) == W_dense x
    w_prime = torch.from_numpy(t.T.astype(np.float32)) * torch.from_numpy(np.repeat(s.astype(np.float32), 128, axis=1))
    y_host = w_prime @ xr
    wd = _dense(hname)
    y_dense = wd @ x
    print(f"{gname}: host fold rel {((y_host - y_dense).norm() / y_dense.norm()).item():.5f}")
    lin = TernaryDecodeLinear3.from_host(device, t, s, inline_scales=True)
    act = ttnn.from_torch(
        xr.bfloat16().reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    y = ttnn.to_torch(lin(act)).float()[0]
    rel_host = ((y - y_host).norm() / y_host.norm()).item()
    rel_dense = ((y - y_dense).norm() / y_dense.norm()).item()
    print(f"{gname}: device vs ternary host rel {rel_host:.5f}, vs dense bf16 rel {rel_dense:.5f}")
    assert rel_host < 0.02
