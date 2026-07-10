# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
[DRAFT / WIP — gpt-oss] Single-device PCC *spec* test for **expert biases** in the fused
``unified_routed_expert_ffn`` op.

gpt-oss MoE experts add a learned bias to the gate, up, AND down projections — unlike
DeepSeek-V3 / MiniMax-M3, which are bias-free (so the fused op was never given bias support).
Target behaviour (clamped SwiGLU-OAI, limit=7.0, alpha=1.702):

    gate = x @ gate_proj.T + gate_bias
    up   = x @ up_proj.T   + up_bias
    gate = clamp(gate, max=limit)
    up   = clamp(up, -limit, limit)
    act  = (up + 1) * gate * sigmoid(alpha * gate)
    out  = act @ down_proj.T + down_bias

The fused op currently has NO bias support. The device test below *specifies* the target so
the kernel work has a golden to hit; it is SKIPPED until bias fusion (and the ``TtRoutedExpert``
bias API shown here) land — tracked in this PR. The host-only reference smoke test is NOT
skipped, so the reference math (and the exact gpt-oss expert formula) is guarded meanwhile.
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

SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0

# gpt-oss-120b / 20b MoE expert dims (hidden_size / intermediate_size).
GPT_OSS_EMB = 2880
GPT_OSS_HIDDEN = 2880


def _torch_swigluoai_expert_with_bias(x, w, b, alpha=SWIGLU_ALPHA, limit=SWIGLU_LIMIT):
    """Clamped SwiGLU-OAI FFN WITH gate/up/down biases (gpt-oss), fp32. Weights HF (out, in)."""
    gate = F.linear(x, w["gate_proj"], b["gate_proj_bias"])
    up = F.linear(x, w["up_proj"], b["up_proj_bias"])
    gate_c = gate.clamp(max=limit)
    up_c = up.clamp(min=-limit, max=limit)
    glu = gate_c * torch.sigmoid(alpha * gate_c)
    activated = (up_c + 1.0) * glu
    return F.linear(activated, w["down_proj"], b["down_proj_bias"])


def run_bias_routed_expert(mesh_device, num_tokens, emb_dim, hidden_dim):
    """1 chip, 1 expert. Compares the fused op (SwiGLU-OAI + biases) vs a torch reference."""
    torch.manual_seed(42)

    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.08,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.08,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.05,
    }
    # Biases scaled to a meaningful magnitude relative to the pre-activation std so they
    # actually shift the clamp/activation (a no-op bias would make this test vacuous).
    biases = {
        "gate_proj_bias": torch.randn(hidden_dim, dtype=torch.float32) * 0.5,
        "up_proj_bias": torch.randn(hidden_dim, dtype=torch.float32) * 0.5,
        "down_proj_bias": torch.randn(emb_dim, dtype=torch.float32) * 0.5,
    }

    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)
    with torch.no_grad():
        torch_output = _torch_swigluoai_expert_with_bias(torch_input, weights, biases)

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )

    def _idx(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    # PROPOSED bias API: TtRoutedExpert takes a per-expert list of bias dicts, mirroring
    # `torch_weights`. Kernel contract: gate/up bias added BEFORE the SwiGLU-OAI clamp, down
    # bias added AFTER the down matmul. (Does not exist yet — this is the spec.)
    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=1,
        global_expert_idx_table=_idx([0]),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=num_tokens,
        torch_weights=[weights],
        torch_biases=[biases],  # PROPOSED — bias support is the subject of this PR
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
    )

    tt_output = tt_expert(tt_input, _idx([num_tokens]), _idx([0]))
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[:num_tokens]

    passing, pcc = comp_pcc(torch_output, tt_output_torch, 0.97)
    logger.info(f"bias routed expert num_tokens={num_tokens}: PCC={pcc}")
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"
    assert passing, f"PCC below threshold: {pcc}"


@pytest.mark.skip(
    reason="WIP: unified_routed_expert_ffn has no expert-bias support yet (needed for gpt-oss). "
    "Spec test — enable once bias fusion + the TtRoutedExpert bias API land (tracked in this PR)."
)
@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("num_tokens", [128, 1024], ids=["t128", "t1k"])
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_gptoss_bias_routed_expert(mesh_device, device_params, num_tokens):
    """gpt-oss expert biases (gate/up/down) + SwiGLU-OAI vs torch. SPEC — skipped until kernel support lands."""
    run_bias_routed_expert(mesh_device, num_tokens=num_tokens, emb_dim=GPT_OSS_EMB, hidden_dim=GPT_OSS_HIDDEN)


def test_gptoss_bias_torch_reference_smoke():
    """Host-only (no device): guards the bias SwiGLU-OAI reference math and documents the gpt-oss
    expert formula. Not skipped — proves bias actually changes the output vs the bias-free path, so
    the skipped device spec above is meaningful even before the kernel lands."""
    torch.manual_seed(0)
    emb, hidden, n = 256, 512, 32
    x = torch.randn(n, emb)
    w = {
        "gate_proj": torch.randn(hidden, emb) * 0.08,
        "up_proj": torch.randn(hidden, emb) * 0.08,
        "down_proj": torch.randn(emb, hidden) * 0.05,
    }
    b = {
        "gate_proj_bias": torch.randn(hidden) * 0.5,
        "up_proj_bias": torch.randn(hidden) * 0.5,
        "down_proj_bias": torch.randn(emb) * 0.5,
    }
    zero_b = {k: torch.zeros_like(v) for k, v in b.items()}
    with torch.no_grad():
        out_bias = _torch_swigluoai_expert_with_bias(x, w, b)
        out_nobias = _torch_swigluoai_expert_with_bias(x, w, zero_b)
    assert out_bias.shape == (n, emb)
    assert not torch.allclose(out_bias, out_nobias), "bias must change the expert output"
