# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Channel ("temporal") rounds on the DRAM path: the VAE decoder shapes, with and without fused SiLU.
GN_TEMPORAL_ROUNDS=0 runs the same cells on the single streaming program (the perf baseline)."""
import os
import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C_sharded import (
    torch_groupnorm_n_1_hw_c,
    compute_pcc,
)
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C.groupnorm_sc_N_1_HW_C import plan_channel_rounds

SHAPES = [
    pytest.param((1, 1, 1048576, 128), id="vae_1048576x128"),
    pytest.param((1, 1, 262144, 256), id="vae_262144x256"),
    pytest.param((1, 1, 262144, 512), id="vae_262144x512"),
    pytest.param((1, 1, 65536, 512), id="vae_65536x512"),
]


@pytest.mark.parametrize("activation", [None, "silu"], ids=["plain", "silu"])
@pytest.mark.parametrize("shape", SHAPES)
def test_rounds(device, shape, activation):
    torch.manual_seed(0)
    N, _, HW, C = shape
    x = (torch.randn(shape) * 0.5 + 0.2).to(torch.bfloat16)
    gamma = torch.randn(1, 1, 1, C).to(torch.bfloat16)
    beta = torch.randn(1, 1, 1, C).to(torch.bfloat16)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rounds = plan_channel_rounds(tt_x, 32)
    print(f"\nROUNDS {shape} -> {rounds} rounds (GN_TEMPORAL_ROUNDS={os.environ.get('GN_TEMPORAL_ROUNDS', '1')})")
    y = groupnorm_sc_N_1_HW_C(
        tt_x, 32, gamma=ttnn.from_torch(gamma, **kw), beta=ttnn.from_torch(beta, **kw), activation=activation
    )
    expected = torch_groupnorm_n_1_hw_c(x, 32, gamma=gamma, beta=beta).float()
    if activation == "silu":
        expected = torch.nn.functional.silu(expected)
    actual = ttnn.to_torch(y).float()
    pcc = compute_pcc(actual, expected)
    rms = float(((actual - expected) ** 2).mean().sqrt())
    print(f"ROUNDS_RESULT {shape} {activation} pcc={pcc:.6f} rms={rms:.4f}")
    assert torch.isfinite(actual).all()
    assert pcc >= 0.9995 and rms <= 0.05, f"pcc={pcc} rms={rms}"
