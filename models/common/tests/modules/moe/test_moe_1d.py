# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the MoE1D module (1D mesh topology: N150, N300, T3K), TP only.

M2 coverage (this file):
1. MoEConfig dataclass construction / defaults / power-user overrides (no device).
2. Torch golden self-consistency (CPU, no device) for both families.

M3 will add the vs-reference PCC device tests (parametrized over mesh shape x mode x family).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.tests.modules.moe.reference_moe import (
    CLAMPED_SWIGLU,
    GEGLU,
    SOFTMAX_TOPK_RENORM,
    SWIGLU,
    TOPK_SIGMOID,
    TOPK_SOFTMAX,
    MoEReferenceWeights,
    gemma_preset,
    gpt_oss_preset,
    granite_preset,
    north_mini_preset,
    qwen3_preset,
    reference_moe_forward,
)
from models.common.utility_functions import comp_allclose, comp_pcc

# ============================================================================
# Unit tests — MoEConfig dataclass (no device)
# ============================================================================


def test_moe_config_creation():
    """MoEConfig can be created with explicit values and preserves them."""
    from unittest.mock import MagicMock

    from models.common.modules.moe.moe_1d import ExpertActivation, MoEConfig, RoutingNorm

    rw, gp, up, dp = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    mock_dev, mock_ccl = MagicMock(), MagicMock()

    config = MoEConfig(
        router_weight=rw,
        gate_proj=gp,
        up_proj=up,
        down_proj=dp,
        top_k=4,
        hidden_size=2880,
        intermediate_size=2880,
        num_experts=128,
        mesh_device=mock_dev,
        tt_ccl=mock_ccl,
        expert_activation=ExpertActivation.CLAMPED_SWIGLU,
        routing_norm=RoutingNorm.TOPK_SOFTMAX,
        fuse_gate_up=True,
        topology=ttnn.Topology.Ring,
    )

    assert config.router_weight is rw
    assert config.gate_proj is gp and config.up_proj is up and config.down_proj is dp
    assert config.top_k == 4
    assert config.hidden_size == 2880
    assert config.num_experts == 128
    assert config.mesh_device is mock_dev
    assert config.tt_ccl is mock_ccl
    assert config.expert_activation == ExpertActivation.CLAMPED_SWIGLU
    assert config.routing_norm == RoutingNorm.TOPK_SOFTMAX
    assert config.fuse_gate_up is True
    assert config.topology == ttnn.Topology.Ring


def test_moe_config_defaults():
    """MoEConfig has Gemma-shaped defaults and Nones for optional fields."""
    from unittest.mock import MagicMock

    from models.common.modules.moe.moe_1d import ExpertActivation, MoEConfig, RoutingNorm

    config = MoEConfig(
        router_weight=MagicMock(), gate_proj=MagicMock(), up_proj=MagicMock(), down_proj=MagicMock(), top_k=8
    )

    # Strategy defaults = Gemma family
    assert config.expert_activation == ExpertActivation.GEGLU
    assert config.routing_norm == RoutingNorm.SOFTMAX_TOPK_RENORM
    assert config.fuse_gate_up is False
    assert config.swiglu_alpha == pytest.approx(1.702)
    assert config.swiglu_limit == pytest.approx(7.0)

    # Optional fields default to None
    assert config.router_bias is None
    assert config.gate_bias is None
    assert config.router_norm_weight is None
    assert config.expert_norm_weight is None
    assert config.per_expert_scale is None
    assert config.mesh_device is None
    assert config.tt_ccl is None
    assert config.hidden_size is None
    assert config.num_experts is None
    assert config.expert_weight_dtype is None
    assert config.router_dtype is None


def test_moe_config_power_user_overrides():
    """MoEConfig accepts power-user program/memory/dtype overrides and biases."""
    from unittest.mock import MagicMock

    from models.common.modules.moe.moe_1d import MoEConfig

    mock_prg, mock_mem = MagicMock(), MagicMock()

    config = MoEConfig(
        router_weight=MagicMock(),
        gate_proj=MagicMock(),
        up_proj=MagicMock(),
        down_proj=MagicMock(),
        top_k=4,
        router_bias=MagicMock(),
        gate_bias=MagicMock(),
        up_bias=MagicMock(),
        down_bias=MagicMock(),
        gate_up_program_config=mock_prg,
        down_program_config=mock_prg,
        decode_memory_config=mock_mem,
        expert_weight_dtype=ttnn.bfloat4_b,
        router_dtype=ttnn.bfloat16,
    )

    assert config.router_bias is not None and config.gate_bias is not None
    assert config.gate_up_program_config is mock_prg
    assert config.down_program_config is mock_prg
    assert config.decode_memory_config is mock_mem
    assert config.expert_weight_dtype == ttnn.bfloat4_b
    assert config.router_dtype == ttnn.bfloat16


def test_moe_config_swiglu_sigmoid_strategies():
    """New-model strategy enums (plain SwiGLU activation, topk->sigmoid routing) are accepted."""
    from unittest.mock import MagicMock

    from models.common.modules.moe.moe_1d import ExpertActivation, MoEConfig, RoutingNorm

    # New enum members exist.
    assert ExpertActivation.SWIGLU.value == "swiglu"
    assert RoutingNorm.TOPK_SIGMOID.value == "topk_sigmoid"

    config = MoEConfig(
        router_weight=MagicMock(),
        gate_proj=MagicMock(),
        up_proj=MagicMock(),
        down_proj=MagicMock(),
        top_k=8,
        expert_activation=ExpertActivation.SWIGLU,
        routing_norm=RoutingNorm.TOPK_SIGMOID,
    )
    assert config.expert_activation == ExpertActivation.SWIGLU
    assert config.routing_norm == RoutingNorm.TOPK_SIGMOID
    # No biases / norms / scales for the new (Qwen3/Granite/North-Mini) families.
    assert config.router_bias is None and config.gate_bias is None
    assert config.router_norm_weight is None and config.expert_norm_weight is None
    assert config.per_expert_scale is None


def test_moe_config_is_resolved_requires_dims_and_device():
    """is_resolved() is False until the foundational fields are filled."""
    from unittest.mock import MagicMock

    from models.common.modules.moe.moe_1d import MoEConfig

    config = MoEConfig(
        router_weight=MagicMock(), gate_proj=MagicMock(), up_proj=MagicMock(), down_proj=MagicMock(), top_k=8
    )
    # Unresolved: hidden_size/num_experts/mesh_device/dtypes are None.
    assert config.is_resolved() is False


# ============================================================================
# Torch golden self-consistency (CPU, no device)
# ============================================================================


def _mk_ref_weights(H, I, E, *, bias=False, router_norm=False, expert_norm=False, per_expert=False):
    w = MoEReferenceWeights(
        router_weight=torch.randn(E, H),
        gate_proj=torch.randn(E, I, H) * 0.1,
        up_proj=torch.randn(E, I, H) * 0.1,
        down_proj=torch.randn(E, H, I) * 0.1,
    )
    if bias:
        w.router_bias = torch.randn(E)
        w.gate_bias = torch.randn(E, I) * 0.1
        w.up_bias = torch.randn(E, I) * 0.1
        w.down_bias = torch.randn(E, H) * 0.1
    if router_norm:
        w.router_norm_weight = torch.randn(H) * 0.1 + 1.0
        w.router_input_scale = torch.randn(H) * 0.1 + 1.0
    if expert_norm:
        w.expert_norm_weight = torch.randn(H) * 0.1 + 1.0
    if per_expert:
        w.per_expert_scale = torch.rand(E) + 0.5
    return w


@pytest.mark.parametrize("family", ["gemma", "gpt_oss", "qwen3", "granite", "north_mini"])
def test_reference_moe_runs_and_is_finite(family):
    torch.manual_seed(0)
    H, I, E, K = 64, 128, 8, 2
    x = torch.randn(1, 16, H)
    if family == "gemma":
        w = _mk_ref_weights(H, I, E, router_norm=True, expert_norm=True, per_expert=True)
        cfg = gemma_preset(E, K, H)
        assert cfg.expert_activation == GEGLU and cfg.routing_norm == SOFTMAX_TOPK_RENORM
    elif family == "gpt_oss":
        w = _mk_ref_weights(H, I, E, bias=True)
        cfg = gpt_oss_preset(E, K)
        assert cfg.expert_activation == CLAMPED_SWIGLU and cfg.routing_norm == TOPK_SOFTMAX
    else:
        # Qwen3 / Granite / North-Mini: plain SwiGLU, no bias/norm; differ only in routing.
        w = _mk_ref_weights(H, I, E)
        cfg = {"qwen3": qwen3_preset, "granite": granite_preset, "north_mini": north_mini_preset}[family](E, K)
        expected_routing = {
            "qwen3": SOFTMAX_TOPK_RENORM,
            "granite": TOPK_SOFTMAX,
            "north_mini": TOPK_SIGMOID,
        }[family]
        assert cfg.expert_activation == SWIGLU and cfg.routing_norm == expected_routing

    out = reference_moe_forward(x, w, cfg)
    assert tuple(out.shape) == (1, 16, H)
    assert torch.isfinite(out).all()


def test_reference_routing_sigmoid_is_independent_no_renorm():
    """North-Mini topk->sigmoid routing: exactly top_k nonzeros, each a sigmoid in (0,1),
    NOT renormalized to sum 1 (independent per-expert gates)."""
    from models.common.tests.modules.moe.reference_moe import _compute_dense_routing

    torch.manual_seed(2)
    H, I, E, K = 64, 128, 8, 3
    x = torch.randn(1, 16, H)
    w = _mk_ref_weights(H, I, E)
    dense = _compute_dense_routing(x.reshape(-1, H), w, north_mini_preset(E, K))
    nz = dense > 0
    assert int(nz.sum(-1).float().mean().item()) == K  # exactly top_k gates per token
    vals = dense[nz]
    # sigmoid outputs in (0,1); well-separated logits saturate to exactly 1.0 in fp32.
    assert (vals > 0).all() and (vals <= 1.0).all()
    # Independent gates do not sum to 1 (would be a near-impossible coincidence for K>1).
    assert not torch.allclose(dense.sum(-1), torch.ones(16), atol=1e-2)


def test_reference_routing_weights_sum_to_one_gemma():
    """Gemma topk routing weights renormalize to sum 1 per token."""
    from models.common.tests.modules.moe.reference_moe import _compute_dense_routing

    torch.manual_seed(1)
    H, I, E, K = 64, 128, 8, 2
    x = torch.randn(1, 16, H)
    w = _mk_ref_weights(H, I, E)
    dense = _compute_dense_routing(x.reshape(-1, H), w, gemma_preset(E, K, H))
    assert torch.allclose(dense.sum(-1), torch.ones(16), atol=1e-5)
    # exactly top_k nonzeros per token
    assert int((dense > 0).sum(-1).float().mean().item()) == K


# ============================================================================
# vs-reference PCC (device) — Gemma family, single-device first
# ============================================================================


def _build_gemma_weights(H, I, E, seed=1234):
    """Random weights shared by the golden (HF layout) and the MoE1D LazyWeights (TT layout)."""
    torch.manual_seed(seed)
    # Scale the router weight so expert logits are well-separated: random near-tie logits
    # would flip top-k selection under bf16 (device) vs fp32 (golden). Real trained routers
    # are well-separated; this avoids the pathological random near-tie regime.
    router_hf = torch.randn(E, H) * (H**-0.5) * 4.0
    gate_hf = torch.randn(E, I, H) * (H**-0.5)
    up_hf = torch.randn(E, I, H) * (H**-0.5)
    down_hf = torch.randn(E, H, I) * (I**-0.5)
    router_input_scale = torch.randn(H) * 0.1 + 1.0
    expert_gamma = torch.randn(H) * 0.1 + 1.0
    per_expert = torch.rand(E) + 0.5
    router_gamma = torch.ones(H)  # Gemma router RMSNorm has no learned gamma (pure normalize)

    golden_w = MoEReferenceWeights(
        router_weight=router_hf,
        gate_proj=gate_hf,
        up_proj=up_hf,
        down_proj=down_hf,
        router_norm_weight=router_gamma,
        router_input_scale=router_input_scale,
        per_expert_scale=per_expert,
        expert_norm_weight=expert_gamma,
    )
    return dict(
        router_hf=router_hf,
        gate_hf=gate_hf,
        up_hf=up_hf,
        down_hf=down_hf,
        router_input_scale=router_input_scale,
        expert_gamma=expert_gamma,
        per_expert=per_expert,
        router_gamma=router_gamma,
        golden_w=golden_w,
    )


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 4)], ids=["1x1", "1x2", "1x4"], indirect=True)
@pytest.mark.parametrize(
    "mode,seq_len",
    [("prefill", 32), ("prefill", 64), ("prefill", 128), ("decode", 1)],
    ids=["prefill-32", "prefill-64", "prefill-128", "decode-1"],
)
def test_moe_1d_vs_reference_gemma(ttnn_mesh_device, mode, seq_len):
    """MoE1D (Gemma config) matches the torch golden at PCC >= 0.99, single device."""
    from models.common.modules.moe.moe_1d import ExpertActivation, MoE1D, MoEConfig, RoutingNorm

    H, I, E, K = 128, 256, 8, 2
    w = _build_gemma_weights(H, I, E)

    # Golden output. Feed the golden the SAME bf16-rounded input the device sees
    # (the device input is bf16), so routing decisions align — comparing an fp32-input
    # golden against a bf16-input module conflates input quantization with module error.
    torch.manual_seed(7)
    torch_input = torch.randn(1, 1, seq_len, H, dtype=torch.float32).to(torch.bfloat16)
    cfg_ref = gemma_preset(E, K, H)
    cfg_ref.route_topk_in_bf16 = True  # device topk runs in bf16; match selection regime
    with torch.no_grad():
        golden_out = reference_moe_forward(torch_input.float(), w["golden_w"], cfg_ref)

    # TT layout LazyWeights from the SAME tensors
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    lw = lambda src, dt=ttnn.bfloat16: LazyWeight(source=src, dtype=dt)
    # Router weight in fp32 so logits match the golden's routing (small tensor, cheap).
    router_weight = LazyWeight(source=w["router_hf"].t().contiguous().reshape(1, 1, H, E), dtype=ttnn.float32)
    gate_proj = lw(w["gate_hf"].transpose(-2, -1).unsqueeze(0).contiguous())  # [1,E,H,I]
    up_proj = lw(w["up_hf"].transpose(-2, -1).unsqueeze(0).contiguous())
    down_proj = lw(w["down_hf"].transpose(-2, -1).unsqueeze(0).contiguous())  # [1,E,I,H]
    router_norm_weight = lw(w["router_gamma"].reshape(1, 1, H // 32, 32))
    expert_norm_weight = lw(w["expert_gamma"].reshape(1, 1, H // 32, 32))
    router_input_scale = lw(w["router_input_scale"].reshape(1, 1, 1, H))
    per_expert_scale = lw(w["per_expert"].reshape(1, 1, 1, E))

    config = MoEConfig(
        router_weight=router_weight,
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
        top_k=K,
        router_norm_weight=router_norm_weight,
        router_input_scale=router_input_scale,
        router_logit_scale=H**-0.5,
        per_expert_scale=per_expert_scale,
        expert_norm_weight=expert_norm_weight,
        expert_activation=ExpertActivation.GEGLU,
        routing_norm=RoutingNorm.SOFTMAX_TOPK_RENORM,
        expert_weight_dtype=ttnn.bfloat16,
        router_dtype=ttnn.float32,
    )
    model = MoE1D.from_config(config)

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16),
        device=ttnn_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
    )
    tt_out = model.forward(tt_input, mode)
    tt_out_torch = to_torch_auto_compose(tt_out)[..., :seq_len, :H].reshape(1, 1, seq_len, H)
    ttnn.SetDefaultDevice(None)

    passing, pcc_message = comp_pcc(golden_out, tt_out_torch.float(), 0.99)
    logger.info(comp_allclose(golden_out, tt_out_torch.float()))
    logger.info(f"MoE1D (Gemma) {mode} seq={seq_len}: {pcc_message}")
    assert passing, f"MoE1D Gemma {mode} seq={seq_len} below PCC 0.99: {pcc_message}"


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 4)], ids=["1x1", "1x2", "1x4"], indirect=True)
@pytest.mark.parametrize(
    "mode,seq_len",
    [("prefill", 32), ("prefill", 64), ("prefill", 128), ("decode", 1)],
    ids=["prefill-32", "prefill-64", "prefill-128", "decode-1"],
)
def test_moe_1d_vs_reference_gpt_oss(ttnn_mesh_device, mode, seq_len):
    """MoE1D (GPT-OSS config: clamped-SwiGLU, biases, topk->softmax) matches golden, PCC >= 0.99."""
    from models.common.modules.moe.moe_1d import ExpertActivation, MoE1D, MoEConfig, RoutingNorm

    H, I, E, K = 128, 256, 8, 2
    torch.manual_seed(4321)
    router_hf = torch.randn(E, H) * (H**-0.5) * 4.0  # well-separated (avoid near-tie flips)
    router_bias = torch.randn(E) * 0.1
    gate_hf = torch.randn(E, I, H) * (H**-0.5)
    up_hf = torch.randn(E, I, H) * (H**-0.5)
    down_hf = torch.randn(E, H, I) * (I**-0.5)
    gate_bias = torch.randn(E, I) * 0.1
    up_bias = torch.randn(E, I) * 0.1
    down_bias = torch.randn(E, H) * 0.1

    golden_w = MoEReferenceWeights(
        router_weight=router_hf,
        gate_proj=gate_hf,
        up_proj=up_hf,
        down_proj=down_hf,
        router_bias=router_bias,
        gate_bias=gate_bias,
        up_bias=up_bias,
        down_bias=down_bias,
    )
    cfg_ref = gpt_oss_preset(E, K)
    cfg_ref.route_topk_in_bf16 = True

    torch.manual_seed(7)
    torch_input = torch.randn(1, 1, seq_len, H, dtype=torch.float32).to(torch.bfloat16)
    with torch.no_grad():
        golden_out = reference_moe_forward(torch_input.float(), golden_w, cfg_ref)

    ttnn.SetDefaultDevice(ttnn_mesh_device)
    lw = lambda src, dt=ttnn.bfloat16: LazyWeight(source=src, dtype=dt)
    config = MoEConfig(
        router_weight=LazyWeight(source=router_hf.t().contiguous().reshape(1, 1, H, E), dtype=ttnn.float32),
        gate_proj=lw(gate_hf.transpose(-2, -1).unsqueeze(0).contiguous()),  # [1,E,H,I]
        up_proj=lw(up_hf.transpose(-2, -1).unsqueeze(0).contiguous()),
        down_proj=lw(down_hf.transpose(-2, -1).unsqueeze(0).contiguous()),  # [1,E,I,H]
        top_k=K,
        router_bias=LazyWeight(source=router_bias.reshape(1, 1, 1, E), dtype=ttnn.float32),
        gate_bias=lw(gate_bias.reshape(1, E, I)),
        up_bias=lw(up_bias.reshape(1, E, I)),
        down_bias=lw(down_bias.reshape(1, E, H)),
        expert_activation=ExpertActivation.CLAMPED_SWIGLU,
        routing_norm=RoutingNorm.TOPK_SOFTMAX,
        expert_weight_dtype=ttnn.bfloat16,
        router_dtype=ttnn.float32,
    )
    model = MoE1D.from_config(config)

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16),
        device=ttnn_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
    )
    tt_out = model.forward(tt_input, mode)
    tt_out_torch = to_torch_auto_compose(tt_out)[..., :seq_len, :H].reshape(1, 1, seq_len, H)
    ttnn.SetDefaultDevice(None)

    passing, pcc_message = comp_pcc(golden_out, tt_out_torch.float(), 0.99)
    logger.info(comp_allclose(golden_out, tt_out_torch.float()))
    logger.info(f"MoE1D (GPT-OSS) {mode} seq={seq_len}: {pcc_message}")
    assert passing, f"MoE1D GPT-OSS {mode} seq={seq_len} below PCC 0.99: {pcc_message}"


# ============================================================================
# vs-reference PCC (device) — new SwiGLU families (Qwen3 / Granite / North-Mini)
#
# All three are plain-SwiGLU, bias-free, norm-free routed-expert blocks that differ
# only in the routing strategy:
#   qwen3      -> softmax->topk->renorm   (norm_topk_prob=True)
#   granite    -> topk->softmax
#   north_mini -> topk->sigmoid (no renorm)
# Structurally identical to the existing families; this exercises the two additive
# strategy values (ExpertActivation.SWIGLU, RoutingNorm.TOPK_SIGMOID).
# ============================================================================


def _build_plain_swiglu_weights(H, I, E, seed=2468):
    """Random no-bias/no-norm MoE weights (HF layout) shared by golden + MoE1D LazyWeights."""
    torch.manual_seed(seed)
    # Well-separated router logits so bf16 topk selection matches fp32 golden (real routers
    # are separated; random near-ties flip selection — see MODULE_NOTES M3 learnings).
    router_hf = torch.randn(E, H) * (H**-0.5) * 4.0
    gate_hf = torch.randn(E, I, H) * (H**-0.5)
    up_hf = torch.randn(E, I, H) * (H**-0.5)
    down_hf = torch.randn(E, H, I) * (I**-0.5)
    golden_w = MoEReferenceWeights(router_weight=router_hf, gate_proj=gate_hf, up_proj=up_hf, down_proj=down_hf)
    return dict(router_hf=router_hf, gate_hf=gate_hf, up_hf=up_hf, down_hf=down_hf, golden_w=golden_w)


def _swiglu_family_strategy(family):
    """(MoE1D ExpertActivation, MoE1D RoutingNorm, reference preset builder) per new family."""
    from models.common.modules.moe.moe_1d import ExpertActivation, RoutingNorm

    table = {
        "qwen3": (ExpertActivation.SWIGLU, RoutingNorm.SOFTMAX_TOPK_RENORM, qwen3_preset),
        "granite": (ExpertActivation.SWIGLU, RoutingNorm.TOPK_SOFTMAX, granite_preset),
        "north_mini": (ExpertActivation.SWIGLU, RoutingNorm.TOPK_SIGMOID, north_mini_preset),
    }
    return table[family]


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 4)], ids=["1x1", "1x2", "1x4"], indirect=True)
@pytest.mark.parametrize(
    "mode,seq_len",
    [("prefill", 32), ("prefill", 64), ("prefill", 128), ("decode", 1)],
    ids=["prefill-32", "prefill-64", "prefill-128", "decode-1"],
)
@pytest.mark.parametrize("family", ["qwen3", "granite", "north_mini"], ids=["qwen3", "granite", "north_mini"])
def test_moe_1d_vs_reference_swiglu_families(ttnn_mesh_device, mode, seq_len, family):
    """MoE1D (plain SwiGLU; softmax-renorm / softmax / sigmoid routing) matches golden, PCC >= 0.99."""
    from models.common.modules.moe.moe_1d import MoE1D, MoEConfig

    H, I, E, K = 128, 256, 8, 2
    w = _build_plain_swiglu_weights(H, I, E)
    activation, routing_norm, preset = _swiglu_family_strategy(family)

    # Golden: feed the same bf16-rounded input the device sees; match the bf16 topk regime.
    torch.manual_seed(7)
    torch_input = torch.randn(1, 1, seq_len, H, dtype=torch.float32).to(torch.bfloat16)
    cfg_ref = preset(E, K)
    cfg_ref.route_topk_in_bf16 = True
    with torch.no_grad():
        golden_out = reference_moe_forward(torch_input.float(), w["golden_w"], cfg_ref)

    ttnn.SetDefaultDevice(ttnn_mesh_device)
    lw = lambda src, dt=ttnn.bfloat16: LazyWeight(source=src, dtype=dt)
    config = MoEConfig(
        # Router weight fp32 so logits match the golden's routing (small tensor, cheap).
        router_weight=LazyWeight(source=w["router_hf"].t().contiguous().reshape(1, 1, H, E), dtype=ttnn.float32),
        gate_proj=lw(w["gate_hf"].transpose(-2, -1).unsqueeze(0).contiguous()),  # [1,E,H,I]
        up_proj=lw(w["up_hf"].transpose(-2, -1).unsqueeze(0).contiguous()),
        down_proj=lw(w["down_hf"].transpose(-2, -1).unsqueeze(0).contiguous()),  # [1,E,I,H]
        top_k=K,
        expert_activation=activation,
        routing_norm=routing_norm,
        expert_weight_dtype=ttnn.bfloat16,
        router_dtype=ttnn.float32,
    )
    model = MoE1D.from_config(config)

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16),
        device=ttnn_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
    )
    tt_out = model.forward(tt_input, mode)
    tt_out_torch = to_torch_auto_compose(tt_out)[..., :seq_len, :H].reshape(1, 1, seq_len, H)
    ttnn.SetDefaultDevice(None)

    passing, pcc_message = comp_pcc(golden_out, tt_out_torch.float(), 0.99)
    logger.info(comp_allclose(golden_out, tt_out_torch.float()))
    logger.info(f"MoE1D ({family}) {mode} seq={seq_len}: {pcc_message}")
    assert passing, f"MoE1D {family} {mode} seq={seq_len} below PCC 0.99: {pcc_message}"


# ============================================================================
# Power-user override (device) — overridden MoEConfig field honored end-to-end
# ============================================================================


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_moe_1d_power_user_override_prefill_chunk(ttnn_mesh_device):
    """Overriding prefill_chunk_size forces the multi-chunk prefill loop (vs the E-aware
    single-pass default) and the result still matches golden — the override is honored."""
    from models.common.modules.moe.moe_1d import ExpertActivation, MoE1D, MoEConfig, RoutingNorm

    H, I, E, K, seq_len = 128, 256, 8, 2, 128
    w = _build_plain_swiglu_weights(H, I, E)

    torch.manual_seed(7)
    torch_input = torch.randn(1, 1, seq_len, H, dtype=torch.float32).to(torch.bfloat16)
    cfg_ref = qwen3_preset(E, K)
    cfg_ref.route_topk_in_bf16 = True
    with torch.no_grad():
        golden_out = reference_moe_forward(torch_input.float(), w["golden_w"], cfg_ref)

    ttnn.SetDefaultDevice(ttnn_mesh_device)
    lw = lambda src, dt=ttnn.bfloat16: LazyWeight(source=src, dtype=dt)
    config = MoEConfig(
        router_weight=LazyWeight(source=w["router_hf"].t().contiguous().reshape(1, 1, H, E), dtype=ttnn.float32),
        gate_proj=lw(w["gate_hf"].transpose(-2, -1).unsqueeze(0).contiguous()),
        up_proj=lw(w["up_hf"].transpose(-2, -1).unsqueeze(0).contiguous()),
        down_proj=lw(w["down_hf"].transpose(-2, -1).unsqueeze(0).contiguous()),
        top_k=K,
        expert_activation=ExpertActivation.SWIGLU,
        routing_norm=RoutingNorm.SOFTMAX_TOPK_RENORM,
        expert_weight_dtype=ttnn.bfloat16,
        router_dtype=ttnn.float32,
        prefill_chunk_size=32,  # override: force 4 chunks of 32 (default E=8 -> 256 single-pass)
    )
    model = MoE1D.from_config(config)
    # Override survives resolution (not clobbered by the E-aware default).
    assert model.config.prefill_chunk_size == 32

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16),
        device=ttnn_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
    )
    tt_out = model.forward(tt_input, "prefill")
    tt_out_torch = to_torch_auto_compose(tt_out)[..., :seq_len, :H].reshape(1, 1, seq_len, H)
    ttnn.SetDefaultDevice(None)

    passing, pcc_message = comp_pcc(golden_out, tt_out_torch.float(), 0.99)
    logger.info(f"MoE1D power-user prefill_chunk_size=32: {pcc_message}")
    assert passing, f"MoE1D override prefill_chunk_size below PCC 0.99: {pcc_message}"
