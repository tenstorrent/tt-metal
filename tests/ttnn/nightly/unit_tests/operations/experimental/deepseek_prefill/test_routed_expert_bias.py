# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
[gpt-oss] Single-device PCC tests for **expert biases** in the fused
``unified_routed_expert_ffn`` op.

gpt-oss MoE experts add a learned bias to the gate, up, AND down projections — unlike
DeepSeek-V3 / MiniMax-M3, which are bias-free. The fused kernel adds gate/up bias BEFORE the
SwiGLU-OAI clamp and down bias AFTER the down matmul. Target behaviour (clamped SwiGLU-OAI,
limit=7.0, alpha=1.702):

    gate = x @ gate_proj.T + gate_bias
    up   = x @ up_proj.T   + up_bias
    gate = clamp(gate, max=limit)
    up   = clamp(up, -limit, limit)
    act  = (up + 1) * gate * sigmoid(alpha * gate)
    out  = act @ down_proj.T + down_bias

Coverage (all single-chip; no multi-device mesh required):
  - ``test_gptoss_bias_routed_expert``   — 1 expert, gate/up/down bias vs a torch reference.
  - ``test_gptoss_bias_multi_expert``    — several experts on one chip, each with its own
    weights+biases and a *different* token count, laid out ``max_tokens`` apart like the real
    dispatch buffer. Exercises the per-expert bias-list wiring (``gate_biases[i]`` <-> expert i)
    and the device-side per-expert count bounding.
  - ``test_gptoss_bias_torch_reference_smoke`` — host-only; guards the reference math.

Cross-device EP (mesh > 1) is intentionally not covered here (needs a Galaxy / multi-Blackhole
box); the kernel is per-core / EP-agnostic and the bias mesh-distribution reuses the proven
weight gather + mapper.
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


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("num_tokens", [128, 1024], ids=["t128", "t1k"])
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_gptoss_bias_routed_expert(mesh_device, device_params, num_tokens):
    """gpt-oss expert biases (gate/up/down) + SwiGLU-OAI vs torch. SPEC — skipped until kernel support lands."""
    run_bias_routed_expert(mesh_device, num_tokens=num_tokens, emb_dim=GPT_OSS_EMB, hidden_dim=GPT_OSS_HIDDEN)


# Multiple experts on ONE chip. Different real token count per expert (all multiples of TILE=32,
# all < max) so the device-side count bounding is exercised alongside the per-expert bias wiring.
MULTI_EXPERT_PARAMS = [
    pytest.param([128, 256, 64, 160], id="e4"),
]


def run_multi_expert_bias(mesh_device, counts, emb_dim, hidden_dim):
    """N experts on 1 chip, each with its own weights+biases and token count.

    Regions are laid out ``max_tokens`` apart (the production dispatch-buffer layout): expert e's
    real tokens occupy the first ``counts[e]`` rows of its region, the rest is slack (random here,
    to prove it is never read). The op mutates the shared buffer in place; each expert's output
    lands back in ``[offset_e, offset_e + counts[e])`` and is compared to its own torch reference.
    """
    torch.manual_seed(123)
    num_experts = len(counts)
    max_tokens = max(counts)
    offsets = [e * max_tokens for e in range(num_experts)]
    total_rows = num_experts * max_tokens

    weights_list, biases_list, inputs = [], [], []
    for _ in range(num_experts):
        weights_list.append(
            {
                "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.08,
                "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.08,
                "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.05,
            }
        )
        biases_list.append(
            {
                "gate_proj_bias": torch.randn(hidden_dim, dtype=torch.float32) * 0.5,
                "up_proj_bias": torch.randn(hidden_dim, dtype=torch.float32) * 0.5,
                "down_proj_bias": torch.randn(emb_dim, dtype=torch.float32) * 0.5,
            }
        )

    # Shared dispatched buffer: random slack (must be ignored) + each expert's input tokens.
    buf = torch.randn(total_rows, emb_dim, dtype=torch.float32)
    for e in range(num_experts):
        x_e = torch.randn(counts[e], emb_dim, dtype=torch.float32)
        inputs.append(x_e)
        buf[offsets[e] : offsets[e] + counts[e]] = x_e

    with torch.no_grad():
        expected = [
            _torch_swigluoai_expert_with_bias(inputs[e], weights_list[e], biases_list[e]) for e in range(num_experts)
        ]

    tt_buf = ttnn.from_torch(
        buf,
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

    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=num_experts,
        global_expert_idx_table=_idx(list(range(num_experts))),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_tokens,
        torch_weights=weights_list,
        torch_biases=biases_list,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
    )

    tt_out = tt_expert(tt_buf, _idx(counts), _idx(offsets))
    out_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    for e in range(num_experts):
        region = out_torch[offsets[e] : offsets[e] + counts[e]]
        passing, pcc = comp_pcc(expected[e], region, 0.97)
        logger.info(f"multi-expert bias: expert {e} count={counts[e]} PCC={pcc}")
        assert not torch.isnan(region).any(), f"expert {e} output contains NaN"
        assert not torch.isinf(region).any(), f"expert {e} output contains Inf"
        assert passing, f"expert {e} PCC below threshold: {pcc}"


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("counts", MULTI_EXPERT_PARAMS)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_gptoss_bias_multi_expert(mesh_device, device_params, counts):
    """Several experts on one chip (per-expert bias wiring + count bounding), gpt-oss dims."""
    run_multi_expert_bias(mesh_device, counts=counts, emb_dim=GPT_OSS_EMB, hidden_dim=GPT_OSS_HIDDEN)


def run_bias_cache_hit(mesh_device, emb_dim, hidden_dim, num_tokens=256):
    """Two invocations with IDENTICAL op attributes (same expert id, dims, token count,
    and weights) but DIFFERENT bias tensors. The second call is a program-cache HIT, so
    the reused program must patch the bias buffer addresses via runtime args; a stale
    address would silently re-apply the first call's bias. Each output is checked against
    its own reference, and the two outputs must differ (proving the second bias took effect)."""
    torch.manual_seed(7)

    # Shared weights across both calls — only the biases change, so the compare isolates
    # bias-address patching rather than weight-address patching.
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.08,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.08,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.05,
    }

    def _biases(scale):
        return {
            "gate_proj_bias": torch.randn(hidden_dim, dtype=torch.float32) * scale,
            "up_proj_bias": torch.randn(hidden_dim, dtype=torch.float32) * scale,
            "down_proj_bias": torch.randn(emb_dim, dtype=torch.float32) * scale,
        }

    biases_a = _biases(0.5)
    biases_b = _biases(0.9)  # different magnitude + different values

    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)
    with torch.no_grad():
        ref_a = _torch_swigluoai_expert_with_bias(torch_input, weights, biases_a)
        ref_b = _torch_swigluoai_expert_with_bias(torch_input, weights, biases_b)

    def _idx(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    def run_once(biases):
        # A fresh TtRoutedExpert each call → new bias buffers, but IDENTICAL op attributes
        # (expert id / dims / max_tokens / counts / dtypes / activation / fuse_bias), so the
        # second invocation reuses the cached program and exercises the address override.
        tt_expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=1,
            global_expert_idx_table=_idx([0]),
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            max_tokens=num_tokens,
            torch_weights=[weights],
            torch_biases=[biases],
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat4_b,
            activation=ttnn.RoutedExpertActivation.SwiGluOai,
        )
        tt_input = ttnn.from_torch(
            torch_input,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
        )
        out = tt_expert(tt_input, _idx([num_tokens]), _idx([0]))
        return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:num_tokens]

    out_a = run_once(biases_a)
    out_b = run_once(biases_b)  # program-cache hit on identical attrs → must patch bias addrs

    pass_a, pcc_a = comp_pcc(ref_a, out_a, 0.97)
    pass_b, pcc_b = comp_pcc(ref_b, out_b, 0.97)
    logger.info(f"cache-hit bias: call A PCC={pcc_a}, call B (cache hit) PCC={pcc_b}")
    assert pass_a, f"call A PCC below threshold: {pcc_a}"
    assert pass_b, f"call B (cache hit) PCC below threshold: {pcc_b} — stale bias address on program-cache hit?"
    # Sanity: the two biases genuinely produce different outputs, so B's pass is not a
    # vacuous match against A's (still-resident) bias.
    assert not torch.allclose(out_a, out_b, atol=1e-2), "outputs identical — second bias did not take effect"


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_gptoss_bias_cache_hit(mesh_device, device_params):
    """Program-cache hit must patch bias buffer addresses (no stale bias), gpt-oss dims."""
    run_bias_cache_hit(mesh_device, emb_dim=GPT_OSS_EMB, hidden_dim=GPT_OSS_HIDDEN)


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
