# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Single-device PCC test for the GeGLU (tanh GELU) activation path of the fused
``unified_routed_expert_ffn`` / ``unified_routed_expert_moe`` op.

Gemma 4's routed experts use ``down(gelu_tanh(gate(x)) * up(x))``. The kernel keeps the SiLU
structure (unary SFPU pass on the gate accumulators, then the multiply with up) and swaps the
SFPU op; it is selected by ``activation=RoutedExpertActivation.GeluTanhGlu`` (compile-time
``GELU_TANH_GLU`` define). Blackhole-only (the op assumes the BH compute grid).
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc

SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip"),
]


def _torch_geglu_expert(x, w):
    """GeGLU FFN with the tanh GELU (Gemma 4), fp32. Weights in HF (out, in)."""
    gate = F.linear(x, w["gate_proj"])
    up = F.linear(x, w["up_proj"])
    return F.linear(F.gelu(gate, approximate="tanh") * up, w["down_proj"])


def _torch_silu_expert(x, w):
    gate = F.linear(x, w["gate_proj"])
    up = F.linear(x, w["up_proj"])
    return F.linear(F.silu(gate) * up, w["down_proj"])


def _weights(emb_dim, hidden_dim):
    torch.manual_seed(42)
    return {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.08,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.08,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.05,
    }


def run_routed_expert(mesh_device, num_tokens, emb_dim, hidden_dim, activation):
    """1 chip, 1 expert: the fused op with the selected activation vs the torch reference."""
    weights = _weights(emb_dim, hidden_dim)
    torch.manual_seed(7)
    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)
    ref = _torch_geglu_expert if activation == ttnn.RoutedExpertActivation.GeluTanhGlu else _torch_silu_expert
    with torch.no_grad():
        torch_output = ref(torch_input, weights)

    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint32
        )

    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=1,
        global_expert_idx_table=_make_idx_tensor([0]),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=num_tokens,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=activation,
    )
    inp = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    out = tt_expert(inp, _make_idx_tensor([num_tokens]), _make_idx_tensor([0]))
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:num_tokens]
    passing, pcc = comp_pcc(torch_output, tt_output_torch, 0.97)
    logger.info(f"activation={activation} num_tokens={num_tokens}: PCC={pcc}")
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"
    assert passing, f"PCC below threshold: {pcc}"
    return tt_output_torch, torch_input, weights


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("num_tokens", [128, 1024], ids=["t128", "t1k"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=True)
def test_gelu_tanh_routed_expert(mesh_device, device_params, num_tokens):
    run_routed_expert(mesh_device, num_tokens, 256, 512, ttnn.RoutedExpertActivation.GeluTanhGlu)


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("num_tokens", [128, 4096], ids=["t128", "t4k"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=True)
def test_gelu_tanh_routed_expert_gemma_shapes(mesh_device, device_params, num_tokens):
    # Gemma 4 26B-A4B routed expert: hidden 2816, intermediate 704
    run_routed_expert(mesh_device, num_tokens, 2816, 704, ttnn.RoutedExpertActivation.GeluTanhGlu)


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=True)
def test_gelu_tanh_differs_from_silu(mesh_device, device_params):
    """The define really changes the kernel: the GeGLU output must not match the SiLU reference."""
    tt_out, torch_input, weights = run_routed_expert(
        mesh_device, 128, 256, 512, ttnn.RoutedExpertActivation.GeluTanhGlu
    )
    silu_ref = _torch_silu_expert(torch_input, weights)
    _, pcc_vs_silu = comp_pcc(silu_ref, tt_out, 0.99)
    assert pcc_vs_silu < 0.99, f"GeGLU output matches the SiLU reference (PCC {pcc_vs_silu}): the define did not take"


def test_routed_expert_activation_enum_exposed():
    assert hasattr(ttnn.RoutedExpertActivation, "GeluTanhGlu")
