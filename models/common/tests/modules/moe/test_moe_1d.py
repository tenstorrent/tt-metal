# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the MoE1D module (1D mesh topology: N150, N300, T3K).

1. Config-dataclass unit tests (no device).
2. MoE1D vs the parameterized torch golden (reference_moe.py), per (config-family x mode x mesh).
3. Power-user override test.

The golden is fed the *dequantized device weights* so weight quantization is not an error source —
the residual gap is matmul fidelity + activation precision + discrete top-k selection.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.moe.moe_1d import ExpertActivation, MoE1D, MoE1DConfig, RoutingStrategy
from models.common.tests.modules.moe.reference_moe import RefMoEConfig, reference_experts, reference_router
from models.common.utility_functions import comp_allclose, comp_pcc

# ============================================================================
# Unit tests — no device required
# ============================================================================


def test_moe_1d_config_creation():
    from unittest.mock import MagicMock

    cfg = MoE1DConfig(
        gate_proj=MagicMock(),
        up_proj=MagicMock(),
        down_proj=MagicMock(),
        router_weight=MagicMock(),
        top_k=8,
        num_experts=128,
        hidden_size=2816,
        intermediate_size=704,
    )
    assert cfg.top_k == 8
    assert cfg.num_experts == 128
    assert cfg.hidden_size == 2816
    assert cfg.intermediate_size == 704
    # defaults
    assert cfg.routing_strategy == RoutingStrategy.SOFTMAX_TOPK_SUMNORM
    assert cfg.activation_strategy == ExpertActivation.GEGLU


def test_moe_1d_config_defaults():
    from unittest.mock import MagicMock

    cfg = MoE1DConfig(gate_proj=MagicMock(), up_proj=MagicMock(), down_proj=MagicMock(), router_weight=MagicMock())
    assert cfg.top_k is None
    assert cfg.routing_strategy == RoutingStrategy.SOFTMAX_TOPK_SUMNORM
    assert cfg.activation_strategy == ExpertActivation.GEGLU
    assert cfg.num_links == 1
    assert cfg.prefill_chunk_size == 32
    # optional knobs default None
    assert cfg.router_prenorm_eps is None
    assert cfg.router_bias is None
    assert cfg.gate_bias is None
    assert cfg.swiglu_limit is None
    assert cfg.mesh_device is None


def test_moe_1d_config_gptoss_overrides():
    from unittest.mock import MagicMock

    cfg = MoE1DConfig(
        gate_proj=MagicMock(),
        up_proj=MagicMock(),
        down_proj=MagicMock(),
        router_weight=MagicMock(),
        top_k=4,
        routing_strategy=RoutingStrategy.TOPK_SOFTMAX,
        activation_strategy=ExpertActivation.SWIGLU_CLAMP,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        router_bias=MagicMock(),
        gate_bias=MagicMock(),
        up_bias=MagicMock(),
        down_bias=MagicMock(),
        expert_weight_dtype=ttnn.bfloat4_b,
    )
    assert cfg.routing_strategy == RoutingStrategy.TOPK_SOFTMAX
    assert cfg.activation_strategy == ExpertActivation.SWIGLU_CLAMP
    assert cfg.swiglu_limit == 7.0
    assert cfg.swiglu_alpha == 1.702
    assert cfg.router_bias is not None
    assert cfg.gate_bias is not None
    assert cfg.expert_weight_dtype == ttnn.bfloat4_b


def test_moe_1d_config_power_user_prg_overrides():
    from unittest.mock import MagicMock

    mock_prg = MagicMock()
    cfg = MoE1DConfig(
        gate_proj=MagicMock(),
        up_proj=MagicMock(),
        down_proj=MagicMock(),
        router_weight=MagicMock(),
        top_k=2,
        decode_gate_up_prg_config=mock_prg,
        decode_down_prg_config=mock_prg,
        gate_up_in0_block_w=8,
        down_in0_block_w=3,
    )
    assert cfg.decode_gate_up_prg_config is mock_prg
    assert cfg.gate_up_in0_block_w == 8
    assert cfg.down_in0_block_w == 3


def test_moe_1d_enums():
    assert RoutingStrategy.SOFTMAX_TOPK_SUMNORM.value == "softmax_topk_sumnorm"
    assert RoutingStrategy.TOPK_SOFTMAX.value == "topk_softmax"
    assert ExpertActivation.GEGLU.value == "geglu"
    assert ExpertActivation.SWIGLU_CLAMP.value == "swiglu_clamp"


# ============================================================================
# Device accuracy tests — MoE1D vs torch golden
# ============================================================================

# Small dims for fast iteration; mirrors the reference unit tests' factory sizes.
H = 256
I = 128
E = 8
TOP_K = 2


def _gemma4_family(router_weight):
    """Gemma4-style config dict + matching golden RefMoEConfig."""
    cfg_kwargs = dict(
        routing_strategy=RoutingStrategy.SOFTMAX_TOPK_SUMNORM,
        activation_strategy=ExpertActivation.GEGLU,
        router_prenorm_eps=1e-6,
        router_input_scalar=H**-0.5,
    )
    ref = RefMoEConfig(
        top_k=TOP_K,
        routing_strategy="softmax_topk_sumnorm",
        activation="geglu",
        router_prenorm_eps=1e-6,
        router_input_scalar=H**-0.5,
    )
    return cfg_kwargs, ref


def _gptoss_family():
    cfg_kwargs = dict(
        routing_strategy=RoutingStrategy.TOPK_SOFTMAX,
        activation_strategy=ExpertActivation.SWIGLU_CLAMP,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
    )
    ref = RefMoEConfig(
        top_k=TOP_K,
        routing_strategy="topk_softmax",
        activation="swiglu_clamp",
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
    )
    return cfg_kwargs, ref


_slow = pytest.mark.slow


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        pytest.param("decode", 1, id="decode"),
        pytest.param("prefill", 32, id="prefill-32"),
        pytest.param("prefill", 128, id="prefill-128"),
        pytest.param("prefill", 256, id="prefill-256", marks=_slow),
        pytest.param("prefill", 1024, id="prefill-1024", marks=_slow),
    ],
)
@pytest.mark.parametrize("family", ["gemma4", "gptoss"])
def test_moe_1d_vs_reference(ttnn_mesh_device: ttnn.MeshDevice, family, mode, seq_len):
    torch.manual_seed(1234)
    mesh_device = ttnn_mesh_device
    ttnn.SetDefaultDevice(mesh_device)

    # --- weights (router proj boosted N(0,1) for peaked, selection-stable routing) ---
    gate_w = torch.randn(1, E, H, I, dtype=torch.float32) * 0.1
    up_w = torch.randn(1, E, H, I, dtype=torch.float32) * 0.1
    down_w = torch.randn(1, E, I, H, dtype=torch.float32) * 0.1
    router_w = torch.randn(1, 1, H, E, dtype=torch.float32)  # peaked logits

    use_bias = family == "gptoss"
    gate_b = torch.randn(1, E, I) * 0.05 if use_bias else None
    up_b = torch.randn(1, E, I) * 0.05 if use_bias else None
    down_b = torch.randn(1, E, H) * 0.05 if use_bias else None
    router_b = torch.randn(1, 1, 1, E) * 0.1 if use_bias else None

    if family == "gemma4":
        cfg_kwargs, ref_cfg = _gemma4_family(router_w)
        router_scale = torch.randn(1, 1, 1, H) * 0.1 + 1.0
        per_expert_scale = torch.rand(1, 1, 1, E) * 0.5 + 0.75
    else:
        cfg_kwargs, ref_cfg = _gptoss_family()
        router_scale = None
        per_expert_scale = None

    def lazy(src, dtype=ttnn.bfloat8_b):
        return LazyWeight(source=src, dtype=dtype) if src is not None else None

    moe_cfg = MoE1DConfig(
        gate_proj=lazy(gate_w),
        up_proj=lazy(up_w),
        down_proj=lazy(down_w),
        router_weight=lazy(router_w, ttnn.bfloat16),
        top_k=TOP_K,
        router_scale=lazy(router_scale, ttnn.bfloat16),
        router_bias=lazy(router_b, ttnn.bfloat16),
        per_expert_scale=lazy(per_expert_scale, ttnn.bfloat16),
        gate_bias=lazy(gate_b, ttnn.bfloat16),
        up_bias=lazy(up_b, ttnn.bfloat16),
        down_bias=lazy(down_b, ttnn.bfloat16),
        **cfg_kwargs,
    )
    moe = MoE1D.from_config(moe_cfg)
    moe.load_device_weights()

    # Dequantize device weights for the golden (removes weight-quant as an error source).
    def deq(t):
        return to_torch_auto_compose(t).float()

    gate_d = deq(moe.gate_proj).reshape(E, H, I)
    up_d = deq(moe.up_proj).reshape(E, H, I)
    down_d = deq(moe.down_proj).reshape(E, I, H)
    router_d = deq(moe.router_weight).reshape(H, E)
    router_scale_d = deq(moe.router_scale).reshape(1, 1, 1, H) if moe.router_scale is not None else None
    per_expert_scale_d = deq(moe.per_expert_scale).reshape(E) if moe.per_expert_scale is not None else None
    router_b_d = deq(moe.router_bias).reshape(E) if moe.router_bias is not None else None
    gate_b_d = deq(moe.gate_bias).reshape(E, I) if moe.gate_bias is not None else None
    up_b_d = deq(moe.up_bias).reshape(E, I) if moe.up_bias is not None else None
    down_b_d = deq(moe.down_bias).reshape(E, H) if moe.down_bias is not None else None

    # --- inputs ---
    router_input = torch.randn(1, 1, seq_len, H, dtype=torch.float32) * 0.5
    expert_input = torch.randn(1, 1, seq_len, H, dtype=torch.float32) * 0.5

    tt_router_in = ttnn.from_torch(
        router_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_expert_in = ttnn.from_torch(
        expert_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Device routing (top-k is a DISCRETE selection: validate it separately from the smooth expert
    # compute, per the playbook's discrete-selection gotcha — bf16 vs fp32 rounding can flip a near-tie
    # rank-k pick, which is a selection difference, not a math error).
    dense_tt = moe._route(tt_router_in)
    dense_tt_torch = to_torch_auto_compose(dense_tt)[:, :, :seq_len, :].float()

    tt_out = moe.forward(tt_router_in, tt_expert_in, mode)
    tt_out_torch = to_torch_auto_compose(tt_out)[:, :, :seq_len, :].float()
    ttnn.SetDefaultDevice(None)

    # --- router selection agreement vs the fp32 golden router ---
    dense_golden = reference_router(
        router_input.to(torch.bfloat16).float(),
        router_d,
        ref_cfg,
        router_scale=router_scale_d,
        router_bias=router_b_d,
        per_expert_scale=per_expert_scale_d,
    ).reshape(1, 1, seq_len, E)
    sel_agree = ((dense_tt_torch != 0) == (dense_golden != 0)).float().mean().item()
    logger.info(f"MoE1D[{family}/{mode}] router selection agreement: {sel_agree:.4f}")
    assert sel_agree >= 0.95, f"router selected different experts than golden ({sel_agree:.4f} < 0.95)"

    # --- expert-compute parity: feed the golden the DEVICE routing (matched selection regime) ---
    ref_out = reference_experts(
        expert_input.to(torch.bfloat16).float(),
        gate_d,
        up_d,
        down_d,
        dense_tt_torch,
        ref_cfg,
        gate_bias=gate_b_d,
        up_bias=up_b_d,
        down_bias=down_b_d,
    ).reshape(1, 1, seq_len, H)

    pcc_threshold = 0.99
    passing, pcc_msg = comp_pcc(ref_out, tt_out_torch, pcc_threshold)
    logger.info(comp_allclose(ref_out, tt_out_torch))
    logger.info(f"MoE1D[{family}/{mode}] experts vs golden: {pcc_msg}")
    assert passing, f"MoE1D[{family}/{mode}] PCC < {pcc_threshold}: {pcc_msg}"


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_moe_1d_power_user_overrides(ttnn_mesh_device: ttnn.MeshDevice):
    """Power-user path: override expert weight dtype (bf4) + the autoport in0_block_w tuning (8/3).

    Validates that the override knobs are honored end-to-end and the module still matches the golden
    (looser threshold since bfloat4_b weights are intentionally lossy).
    """
    torch.manual_seed(7)
    mesh_device = ttnn_mesh_device
    ttnn.SetDefaultDevice(mesh_device)

    gate_w = torch.randn(1, E, H, I) * 0.1
    up_w = torch.randn(1, E, H, I) * 0.1
    down_w = torch.randn(1, E, I, H) * 0.1
    router_w = torch.randn(1, 1, H, E)

    cfg = MoE1DConfig(
        gate_proj=LazyWeight(source=gate_w, dtype=ttnn.bfloat4_b),
        up_proj=LazyWeight(source=up_w, dtype=ttnn.bfloat4_b),
        down_proj=LazyWeight(source=down_w, dtype=ttnn.bfloat4_b),
        router_weight=LazyWeight(source=router_w, dtype=ttnn.bfloat16),
        top_k=TOP_K,
        routing_strategy=RoutingStrategy.SOFTMAX_TOPK_SUMNORM,
        activation_strategy=ExpertActivation.GEGLU,
        router_prenorm_eps=1e-6,
        router_input_scalar=H**-0.5,
        # power-user overrides under test:
        expert_weight_dtype=ttnn.bfloat4_b,
        gate_up_in0_block_w=8,
        down_in0_block_w=3,
    )
    moe = MoE1D.from_config(cfg)
    # the override knobs landed in the resolved config
    assert moe.config.expert_weight_dtype == ttnn.bfloat4_b
    assert moe.config.gate_up_in0_block_w == 8
    assert moe.config.down_in0_block_w == 3
    moe.load_device_weights()

    def deq(t):
        return to_torch_auto_compose(t).float()

    ref_cfg = RefMoEConfig(
        top_k=TOP_K,
        routing_strategy="softmax_topk_sumnorm",
        activation="geglu",
        router_prenorm_eps=1e-6,
        router_input_scalar=H**-0.5,
    )
    ri = torch.randn(1, 1, 1, H) * 0.5
    ei = torch.randn(1, 1, 1, H) * 0.5
    tt_ri = ttnn.from_torch(ri, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_ei = ttnn.from_torch(ei, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    dense_tt = to_torch_auto_compose(moe._route(tt_ri))[:, :, :1, :].float()
    tt_out = moe.forward(tt_ri, tt_ei, "decode")
    tt_out_torch = to_torch_auto_compose(tt_out)[:, :, :1, :].float()
    ttnn.SetDefaultDevice(None)

    ref_out = reference_experts(
        ei.to(torch.bfloat16).float(),
        deq(moe.gate_proj).reshape(E, H, I),
        deq(moe.up_proj).reshape(E, H, I),
        deq(moe.down_proj).reshape(E, I, H),
        dense_tt,
        ref_cfg,
    ).reshape(1, 1, 1, H)
    passing, pcc_msg = comp_pcc(ref_out, tt_out_torch, 0.97)
    logger.info(f"MoE1D[override bf4 + in0_block_w 8/3] vs golden: {pcc_msg}")
    assert passing, f"override test PCC < 0.97: {pcc_msg}"
