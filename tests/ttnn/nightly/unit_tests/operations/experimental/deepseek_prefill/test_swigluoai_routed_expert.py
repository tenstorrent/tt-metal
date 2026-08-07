# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Single-device PCC test for the SwiGLU-OAI activation path of the fused
``unified_routed_expert_ffn`` / ``unified_routed_expert_moe`` op.

MiniMax-M3 (and gpt-oss) use the clamped "swigluoai" activation instead of plain
SiLU SwiGLU:

    gate = clamp(gate, max=limit)            # limit = 7.0
    up   = clamp(up, -limit, limit)
    out  = (up + 1) * gate * sigmoid(alpha * gate)   # alpha = 1.702

The fused kernel bakes alpha/limit (``SwiGLUConfigGPTOSS``) and is selected by
``activation=RoutedExpertActivation.SwiGluOai`` (compile-time ``SWIGLU_OAI`` define).
This test runs the real op (via ``TtRoutedExpert``, single chip / single expert) and compares against
a hand-written torch reference. Inputs/weights are scaled so the ±limit clamp is
actually exercised. Blackhole-only (the op assumes the BH compute grid).
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc


SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip"),
]

SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0


def _torch_silu_expert(x, w):
    """Plain SiLU SwiGLU FFN (DeepSeek default), fp32. Weights in HF (out, in)."""
    gate = F.linear(x, w["gate_proj"])
    up = F.linear(x, w["up_proj"])
    activated = F.silu(gate) * up
    return F.linear(activated, w["down_proj"])


def _torch_swigluoai_expert(x, w, alpha=SWIGLU_ALPHA, limit=SWIGLU_LIMIT):
    """Clamped swigluoai FFN (MiniMax-M3 / gpt-oss), fp32. Weights in HF (out, in)."""
    gate = F.linear(x, w["gate_proj"])
    up = F.linear(x, w["up_proj"])
    gate_c = gate.clamp(max=limit)
    up_c = up.clamp(min=-limit, max=limit)
    glu = gate_c * torch.sigmoid(alpha * gate_c)
    activated = (up_c + 1.0) * glu
    # Report how often the clamp actually fired so a "passing" test can't silently
    # skip the clamp branches.
    gate_clamp_frac = (gate > limit).float().mean().item()
    up_clamp_frac = ((up > limit) | (up < -limit)).float().mean().item()
    logger.info(f"clamp coverage: gate>{limit}: {gate_clamp_frac:.1%}, |up|>{limit}: {up_clamp_frac:.1%}")
    return F.linear(activated, w["down_proj"])


def run_swigluoai_routed_expert(mesh_device, num_tokens, emb_dim, hidden_dim, activation):
    """1 chip, 1 expert. Compares the fused op (selected RoutedExpertActivation) vs torch reference."""
    torch.manual_seed(42)
    is_swigluoai = activation == ttnn.RoutedExpertActivation.SwiGluOai

    # Weights scaled so the gate/up matmul outputs span past ±limit (so both clamp
    # branches are exercised). gate_out std ~= sqrt(emb_dim) * std_w ~= 55 * 0.08 ~= 4.4.
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.08,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.08,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.05,
    }

    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)

    with torch.no_grad():
        if is_swigluoai:
            torch_output = _torch_swigluoai_expert(torch_input, weights)
        else:
            torch_output = _torch_silu_expert(torch_input, weights)

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )

    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    global_expert_idx_tt = _make_idx_tensor([0])
    expert_token_counts_tt = _make_idx_tensor([num_tokens])
    expert_region_offsets_tt = _make_idx_tensor([0])

    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=1,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=num_tokens,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=activation,
    )

    tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[:num_tokens]

    passing, pcc = comp_pcc(torch_output, tt_output_torch, 0.97)
    logger.info(f"activation={activation} num_tokens={num_tokens}: PCC={pcc}")
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"
    assert passing, f"PCC below threshold: {pcc}"


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("num_tokens", [128, 1024], ids=["t128", "t1k"])
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_swigluoai_routed_expert(mesh_device, device_params, num_tokens):
    """MiniMax-M3 clamped swigluoai (RoutedExpertActivation.SwiGluOai) vs torch reference."""
    run_swigluoai_routed_expert(
        mesh_device,
        num_tokens=num_tokens,
        emb_dim=MiniMaxM27Config.EMB_SIZE,
        hidden_dim=MiniMaxM27Config.MOE_INTERMEDIATE_SIZE,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
    )


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("num_tokens", [128, 1024, 4096], ids=["t128", "t1k", "t4k"])
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_swigluoai_routed_expert_m3_shapes(mesh_device, device_params, num_tokens):
    """MiniMax-M3 real MoE dims (emb 6144 / hidden 3072 — 2x M2.7 on both axes).

    Regression for the adaptive L1-budget sizing: at these dims the previous fixed
    chunk_M_tiles=64 / in0_block_w_gu=16 CB layout overflows Blackhole L1. The program
    factory now shrinks in0_block_w_gu / per_core_M to fit; this asserts the op builds
    (no L1 overflow) AND stays numerically correct at M3 shapes. Dims from
    models/demos/minimax_m3/configs/MiniMax-M3/config.json (hidden_size / intermediate_size).

    Token counts cover both program-factory M paths: t128/t1k (<= 1024 tokens =
    <= 32 M-tiles) take the short-seq path; t4k (4096 tokens = 128 M-tiles) takes
    the general 2D path — the exact prefill dispatch-buffer shape that overflowed
    L1 before the adaptive guard.
    """
    run_swigluoai_routed_expert(
        mesh_device,
        num_tokens=num_tokens,
        emb_dim=6144,
        hidden_dim=3072,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
    )


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_silu_routed_expert_unchanged(mesh_device, device_params):
    """Sanity: RoutedExpertActivation.Silu (default) still matches plain SiLU SwiGLU (DeepSeek path unchanged)."""
    run_swigluoai_routed_expert(
        mesh_device,
        num_tokens=128,
        emb_dim=MiniMaxM27Config.EMB_SIZE,
        hidden_dim=MiniMaxM27Config.MOE_INTERMEDIATE_SIZE,
        activation=ttnn.RoutedExpertActivation.Silu,
    )


def test_routed_expert_activation_enum_exposed():
    """Host-only (no device): the RoutedExpertActivation enum is reachable via the
    public ``ttnn`` namespace. Regression guard — the C++ enum is registered on the
    experimental ops module and must be re-exported as ``ttnn.RoutedExpertActivation``
    in ttnn/__init__.py; without that, the SwiGluOai selection is unreachable from Python.
    """
    assert hasattr(ttnn, "RoutedExpertActivation"), "ttnn.RoutedExpertActivation is not exposed"
    activation = ttnn.RoutedExpertActivation
    assert hasattr(activation, "Silu") and hasattr(activation, "SwiGluOai"), "enum is missing Silu/SwiGluOai variants"
    # Distinct variants so SiLU and SwiGLU-OAI cache as separate device programs.
    assert activation.Silu != activation.SwiGluOai
