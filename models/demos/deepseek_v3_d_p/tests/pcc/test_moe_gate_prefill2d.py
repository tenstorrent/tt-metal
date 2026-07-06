# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    create_fabric_router_config,
    create_gate_weights,
    get_max_payload_size,
    get_sp_mesh_composer,
    load_gate_weights_from_hf,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    ValidationResult,
    compare_pcc,
    compare_recall,
    validate_composed,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_validation_results
from models.demos.deepseek_v3_d_p.utils.test_utils import adjust_shapes_for_testing, get_input_mem_config
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import GOLDEN_LONGBOOK_TRACE, load_trace_gate_input

# First MoE layer in DeepSeek-V3 (metadata moe_layer_offset == 3); the golden
# trace stores its gate input as post_attn_norm_layer_3.
_MOE_LAYER_IDX = 3

_DEFAULT_HF_REPO = "deepseek-ai/DeepSeek-V3"
_LOCAL_FALLBACKS = (
    "models/demos/deepseek_v3/reference",
    "/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528",
)


def _resolve_model_id() -> str:
    """Resolve a model identifier (local dir or HF repo ID) for gate weight loading.

    Checks DEEPSEEK_V3_HF_MODEL and standard local paths first; falls back to the
    HF repo ID so that ``load_hf_state_dict_filtered`` can resolve from the HF cache.
    """
    env_path = os.getenv("DEEPSEEK_V3_HF_MODEL")
    if env_path and (Path(env_path) / "model.safetensors.index.json").exists():
        return env_path
    for fallback in _LOCAL_FALLBACKS:
        if (Path(fallback) / "model.safetensors.index.json").exists():
            return fallback
    return _DEFAULT_HF_REPO


def _try_load_real_gate_weights(n_routed_experts: int, dim: int) -> dict | None:
    """Try to load real gate weights from HF; return None on failure."""
    model_id = _resolve_model_id()
    try:
        gate_w = load_gate_weights_from_hf(model_id, layer_idx=3, dtype=torch.bfloat16)
        gate_w["weight"] = gate_w["weight"][:n_routed_experts, :dim]
        gate_w["e_score_correction_bias"] = gate_w["e_score_correction_bias"][:n_routed_experts]
        return gate_w
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Could not load real gate weights ({model_id}): {e}. Using random weights.")
        return None


def _try_load_real_gate_input(max_seq_len: int, dim: int) -> torch.Tensor | None:
    """Try to load the gate input from the golden GPU prefill trace; return None on failure.

    Mirrors test_ttnn_moe / test_prefill_transformer: the gate input is the
    post-attention RMSNorm output (``post_attn_norm_layer_{i}``) of the first MoE
    layer in the bit_sculpt golden trace. ``DEEPSEEK_V3_GATE_INPUT_CACHE`` may
    override the trace directory. Returns ``None`` when the trace is unavailable
    so the caller can fall back to synthetic input.
    """
    trace_dir_env = os.environ.get("DEEPSEEK_V3_GATE_INPUT_CACHE")
    trace_dir = Path(trace_dir_env) if trace_dir_env else GOLDEN_LONGBOOK_TRACE
    return load_trace_gate_input(trace_dir, layer_idx=_MOE_LAYER_IDX, max_seq_len=max_seq_len, dim=dim)


@pytest.mark.parametrize(
    "gate_fallback_mode",
    [GateComputeMode.HOST_ALL, GateComputeMode.DEVICE_FP32],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="mesh-2x2",
        ),
        pytest.param(
            (2, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="fabric2d-mesh-2x2",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="fabric2d-mesh-4x2",
        ),
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p", "kimi_k2_6"], indirect=True, ids=["deepseek_v3", "kimi"])
def test_forward_pass(
    variant,
    mesh_device,
    num_links,
    topology,
    gate_fallback_mode,
):
    """Gate PCC for both model variants. The gate itself picks grouped vs plain
    top-k routing from n_expert_groups (Kimi has one group), so the test just
    passes the model config and the compute mode."""
    random.seed(42)
    torch.manual_seed(42)

    # Build the gate config from the variant's HF dimension constants.
    config = TtMoEGateConfig.from_model_cfg(variant.model_config)
    config.ccl_config["NUM_LINKS"] = num_links
    adjust_shapes_for_testing(config, mesh_device)

    ref_config = SimpleNamespace(
        num_experts_per_tok=config.n_activated_experts,
        n_routed_experts=config.n_routed_experts,
        routed_scaling_factor=config.route_scale,
        scoring_func=config.score_func,
        topk_method="noaux_tc",
        n_group=config.n_expert_groups,
        topk_group=config.n_limited_groups,
        norm_topk_prob=True,
        hidden_size=config.dim,
    )
    reference_model = ReferenceMoEGate(ref_config, use_bitonic_sort=True)

    # Real DeepSeek gate weights (256 experts) can't be reshaped to other expert
    # counts, so only attempt the real-weight/input load for the 256-expert path.
    gate_w = (
        _try_load_real_gate_weights(config.n_routed_experts, config.dim) if config.n_routed_experts == 256 else None
    )
    if gate_w is None:
        gate_w = create_gate_weights(config.n_routed_experts, config.dim)
    reference_model.weight.data = gate_w["weight"]
    reference_model.e_score_correction_bias.data = gate_w["e_score_correction_bias"]

    n_sp_devices = mesh_device.shape[0]
    total_seq_len = config.sp_dim * n_sp_devices
    torch_input = _try_load_real_gate_input(total_seq_len, config.dim) if config.n_routed_experts == 256 else None
    if torch_input is None:
        torch_input = (
            torch.randn(total_seq_len, config.dim, dtype=torch.bfloat16) * 0.1147 * (7168 / config.dim)
        )  # 0.1147 is the std of the real gate input and we need scale it to adjust for smaller dims

    # Reference forward pass
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    reference_topk_indices, reference_topk_scores = reference_model.grouped_forward(torch_input.unsqueeze(0))
    reference_logits = torch_input @ gate_w["weight"].T

    # Create TT input tensor
    sharded_mem_config = get_input_mem_config(config, mesh_device.shape)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(0, -1),  # tensor parallel
            mesh_shape=mesh_device.shape,
        ),
    )

    # Create TT gate
    n_sp_devices = mesh_device.shape[0]
    n_tp_devices = mesh_device.shape[1]

    tt_model = TtMoEGatePrefill(
        config,
        mesh_device,
        weight=gate_w["weight"],
        bias=gate_w["e_score_correction_bias"],
        fallback_mode=gate_fallback_mode,
    )
    # TT forward pass
    tt_topk_scores, tt_topk_indices, tt_logits = tt_model(tt_input)

    # Validation thresholds depend on gate compute mode
    if gate_fallback_mode == GateComputeMode.HOST_ALL:
        recall_threshold = 0.997
        logits_pcc_threshold = 0.997
        scores_pcc_threshold = 0.99
    else:
        recall_threshold = 0.95
        logits_pcc_threshold = 0.997
        scores_pcc_threshold = 0.93

    seq_len_per_device = reference_logits.shape[0] // mesh_device.shape[0]
    sp_composer = get_sp_mesh_composer(mesh_device)

    # SP-replicated checks: compose into [1, n_sp_devices, ...] for validate_composed
    host_tt_topk_indices = ttnn.to_torch(tt_topk_indices, mesh_composer=sp_composer)
    host_tt_topk_indices = host_tt_topk_indices.view(1, n_sp_devices, seq_len_per_device, -1).sort(dim=-1).values
    reference_topk_indices = reference_topk_indices.view(1, n_sp_devices, seq_len_per_device, -1).sort(dim=-1).values

    recall_topk_indices = validate_composed(
        host_tt_topk_indices,
        reference_topk_indices,
        1,
        n_sp_devices,
        compare_recall(recall_threshold),
        name="recall_topk_indices",
        broadcast_groups=n_tp_devices,
    )

    host_tt_logits = ttnn.to_torch(tt_logits, mesh_composer=sp_composer)
    host_tt_logits = host_tt_logits.view(1, n_sp_devices, seq_len_per_device, -1)
    reference_logits = reference_logits.view(1, n_sp_devices, seq_len_per_device, -1)

    pcc_logits = validate_composed(
        host_tt_logits,
        reference_logits,
        1,
        n_sp_devices,
        compare_pcc(logits_pcc_threshold),
        name="pcc_logits",
        broadcast_groups=n_tp_devices,
    )

    host_tt_topk_scores = ttnn.to_torch(tt_topk_scores, mesh_composer=sp_composer)
    host_tt_topk_scores = host_tt_topk_scores.view(1, n_sp_devices, seq_len_per_device, -1)
    reference_topk_scores = reference_topk_scores.view(1, n_sp_devices, seq_len_per_device, -1)

    pcc_scores = validate_composed(
        host_tt_topk_scores,
        reference_topk_scores,
        1,
        n_sp_devices,
        compare_pcc(scores_pcc_threshold),
        name="pcc_scores",
        broadcast_groups=n_tp_devices,
    )

    all_results = [recall_topk_indices, pcc_logits, pcc_scores]

    for res in all_results:
        res.log_mismatches()
    log_validation_results(
        results=all_results,
        num_dispatch_groups=n_tp_devices,
        dispatch_group_size=n_sp_devices,
        title="Gate Prefill2D Validation",
    )
    merged = ValidationResult.merge(all_results, name="gate_prefill2d")
    merged.assert_passed("Gate prefill2d validation failed")
