# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for ttnn.practice_routed_expert: one Kimi K3 routed expert as a single fused op.

    out = (4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)) @ w_down,   g = x @ w_gate,   u = x @ w_up

Unlike the production unified_routed_expert_moe, the op gets one expert's tokens only: no dispatch
buffer, token counts, region offsets or expert-id table.

    x               (T, K)  TILE, DRAM interleaved, T a multiple of 32
    w_gate, w_up    (K, N)  TILE, DRAM interleaved, transposed from the HF (out, in) layout
    w_down          (N, K)  same
    returns         (T, K)  TILE, x's dtype
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SITU, TorchExpert
from tests.ttnn.utils_for_testing import comp_pcc

SITU_BETA = KimiK3Config.ACTIVATION_SITU_BETA
SITU_LINEAR_BETA = KimiK3Config.ACTIVATION_SITU_LINEAR_BETA
# K3's routed experts run after the LatentMoE down-projection, so K is 3584, not the 7168 model width.
K3_EMB = KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE
K3_HIDDEN = KimiK3Config.MOE_INTERMEDIATE_SIZE

# "tiny" is for debugging: every dim spans more than one tile and K != N, so a swapped loop bound
# or tile index fails instead of passing by symmetry.
SHAPES = [
    pytest.param(64, 64, 96, id="tiny"),
    pytest.param(32, K3_EMB, K3_HIDDEN, id="k3-t32"),
    pytest.param(512, K3_EMB, K3_HIDDEN, id="k3-t512"),
]

# (std of the gate/up matmul outputs, minimum fraction of gate/up values past their caps).
# At std 1.2, what production's k3_sweep weights give, both caps are near-linear and a kernel
# missing them still passes; only the saturated case checks the activation.
REGIMES = [
    pytest.param(1.2, None, id="normal"),
    pytest.param(24.0, (0.80, 0.25), id="saturated"),
]

# bf16's 0.99 bar is the one that catches a missing up cap (PCC ~0.97 when saturated).
# bf8 activations with bf4 weights are the production formats, held to production's bf4 bar.
DTYPES = [
    pytest.param(ttnn.bfloat16, ttnn.bfloat16, 0.99, id="bf16"),
    pytest.param(ttnn.bfloat8_b, ttnn.bfloat4_b, 0.96, id="bf8_bf4"),
]


def make_inputs(tokens, emb_dim, hidden_dim, gate_up_std):
    """Random x and HF-layout weights. A gate/up output sums emb_dim products, so its std is the
    weight std times sqrt(emb_dim); dividing that out keeps each regime the same at every shape."""
    torch.manual_seed(0)
    scale = gate_up_std / math.sqrt(emb_dim)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim) * scale,
        "up_proj": torch.randn(hidden_dim, emb_dim) * scale,
        "down_proj": torch.randn(emb_dim, hidden_dim) / math.sqrt(hidden_dim),
    }
    return torch.randn(tokens, emb_dim), weights


def torch_reference(x, weights):
    expert = TorchExpert(
        x.shape[-1],
        weights["gate_proj"].shape[0],
        weights,
        activation=ACTIVATION_SITU,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    )
    with torch.no_grad():
        return expert(x)


def assert_caps_reached(x, weights, min_cap_frac):
    """Guards the test itself: a change to the seed, scale or dims must not quietly move the
    saturated case back into the caps' linear middle."""
    gate = torch.nn.functional.linear(x, weights["gate_proj"])
    up = torch.nn.functional.linear(x, weights["up_proj"])
    gate_frac = (gate.abs() > SITU_BETA).float().mean().item()
    up_frac = (up.abs() > SITU_LINEAR_BETA).float().mean().item()
    assert gate_frac >= min_cap_frac[0], f"only {gate_frac:.1%} of gate values past the cap"
    assert up_frac >= min_cap_frac[1], f"only {up_frac:.1%} of up values past the cap"


def to_device(x, weights, x_dtype, w_dtype, device):
    def put(tensor, dtype):
        return ttnn.from_torch(
            tensor.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return (
        put(x, x_dtype),
        put(weights["gate_proj"].T, w_dtype),
        put(weights["up_proj"].T, w_dtype),
        put(weights["down_proj"].T, w_dtype),
    )


@pytest.mark.parametrize("x_dtype, w_dtype, pcc_threshold", DTYPES)
@pytest.mark.parametrize("gate_up_std, min_cap_frac", REGIMES)
@pytest.mark.parametrize("tokens, emb_dim, hidden_dim", SHAPES)
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU's SFPU op is Blackhole-only")
def test_practice_routed_expert(
    device, tokens, emb_dim, hidden_dim, gate_up_std, min_cap_frac, x_dtype, w_dtype, pcc_threshold
):
    x, weights = make_inputs(tokens, emb_dim, hidden_dim, gate_up_std)
    if min_cap_frac is not None:
        assert_caps_reached(x, weights, min_cap_frac)
    expected = torch_reference(x, weights)

    tt_out = ttnn.practice_routed_expert(*to_device(x, weights, x_dtype, w_dtype, device))

    assert tuple(tt_out.shape) == (tokens, emb_dim)
    assert tt_out.layout == ttnn.TILE_LAYOUT
    assert tt_out.dtype == x_dtype
    out = ttnn.to_torch(tt_out)
    assert torch.isfinite(out).all(), "output contains NaN or Inf"
    _, pcc = comp_pcc(expected, out)
    assert pcc >= pcc_threshold, f"PCC {pcc:.5f} below {pcc_threshold}"
