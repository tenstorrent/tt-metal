# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Decode-mode PCC test for MLP. Random weights vs torch reference.

Run with the DRAM-sharded flag to verify the new path:
    QWEN3_TTS_MLP_DRAM_SHARD_DOWN=1 python -m pytest -q \
        models/demos/qwen3_tts/tests/test_mlp_decode_pcc.py
"""
import pytest
import torch

import ttnn
from models.demos.qwen3_tts.reference.functional import swiglu_mlp as torch_swiglu_mlp
from models.demos.qwen3_tts.tt.mlp import MLP


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    a = x.flatten().float()
    b = y.flatten().float()
    a_c = a - a.mean()
    b_c = b - b.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-12)).item()


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


@pytest.mark.parametrize("seq_len,mode", [(1, "decode"), (128, "prefill")])
def test_mlp_pcc_modes(device, seq_len, mode):
    torch.manual_seed(0)
    hidden = 2048
    intermediate = 6144
    x_torch = torch.randn(1, 1, seq_len, hidden, dtype=torch.bfloat16)
    g = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    u = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    d = torch.randn(hidden, intermediate, dtype=torch.bfloat16)

    ref = torch_swiglu_mlp(x_torch.squeeze(1), g, u, d)

    sd = {
        "test_layer.mlp.gate_proj.weight": g,
        "test_layer.mlp.up_proj.weight": u,
        "test_layer.mlp.down_proj.weight": d,
    }
    mlp = MLP(device, hidden, intermediate, sd, "test_layer")

    x_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_tt = mlp(x_tt, mode=mode)
    y = ttnn.to_torch(y_tt).squeeze(1)
    pcc = pearson(ref, y)
    print(f"[seq_len={seq_len}, mode={mode}] MLP PCC = {pcc:.6f}")
    assert pcc > 0.99
