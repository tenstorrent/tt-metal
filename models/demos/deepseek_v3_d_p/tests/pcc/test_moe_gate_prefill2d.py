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
    ExpertMapping,
    create_fabric_router_config,
    create_gate_weights,
    get_ep_mesh_composer,
    get_gate_outputs,
    get_max_payload_size,
    get_sp_mesh_composer,
    load_gate_weights_from_hf,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    ValidationResult,
    compare_exact,
    compare_pcc,
    compare_recall,
    validate_composed,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_validation_results
from models.demos.deepseek_v3_d_p.utils.test_utils import adjust_shapes_for_testing, get_input_mem_config

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
    """Try to load a saved gate input tensor; return None on failure."""
    gate_input_cache = os.environ.get("DEEPSEEK_V3_GATE_INPUT_CACHE")
    moe_dir = Path(gate_input_cache) if gate_input_cache else Path(__file__).parent.parent.parent / "tt" / "moe"

    for name in ("gate_input_layer3_seq100000.pt", "gate_input_layer3.pt"):
        path = moe_dir / name
        if path.exists():
            saved = torch.load(path, weights_only=True)
            real_input = saved["gate_input"].squeeze(0).to(torch.bfloat16)
            if real_input.shape[0] >= max_seq_len:
                result = real_input[:max_seq_len, :dim]
            else:
                repeats = (max_seq_len + real_input.shape[0] - 1) // real_input.shape[0]
                result = real_input.repeat(repeats, 1)[:max_seq_len, :dim]
            logger.info(f"Loaded real gate input from {path} (raw {real_input.shape}, sliced to {result.shape})")
            return result

    return None


@pytest.mark.parametrize(
    "gate_fallback_mode",
    [GateComputeMode.DEVICE, GateComputeMode.DEVICE_FP32],
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
def test_forward_pass(
    mesh_device,
    num_links,
    topology,
    gate_fallback_mode,
):
    random.seed(42)
    torch.manual_seed(42)

    # Create reference gate
    config = TtMoEGateConfig()
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

    gate_w = _try_load_real_gate_weights(config.n_routed_experts, config.dim)
    if gate_w is None:
        gate_w = create_gate_weights(config.n_routed_experts, config.dim)
    reference_model.weight.data = gate_w["weight"]
    reference_model.e_score_correction_bias.data = gate_w["e_score_correction_bias"]

    n_sp_devices = mesh_device.shape[0]
    total_seq_len = config.sp_dim * n_sp_devices
    torch_input = _try_load_real_gate_input(total_seq_len, config.dim)
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
    n_routed_experts = config.n_routed_experts

    dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=n_routed_experts,
        dispatch_group_size=n_sp_devices,
        num_dispatch_groups=n_tp_devices,
    )
    experts_per_chip = n_routed_experts // (n_sp_devices * n_tp_devices)
    tt_model = TtMoEGatePrefill(
        config,
        mesh_device,
        dispatch_table=dispatch_table,
        experts_per_chip=experts_per_chip,
        weight=gate_w["weight"],
        bias=gate_w["e_score_correction_bias"],
        fallback_mode=gate_fallback_mode,
    )
    # TT forward pass
    (
        tt_topk_scores,
        tt_topk_indices,
        tt_logits,
        tt_dispatch_offsets,
        tt_total_counts_per_expert,
        _,
    ) = tt_model(tt_input)

    # Validation thresholds depend on gate compute mode
    if gate_fallback_mode == GateComputeMode.HOST_ALL:
        recall_threshold = 0.997
        logits_pcc_threshold = 0.997
        scores_pcc_threshold = 0.99
    else:
        recall_threshold = 0.70
        logits_pcc_threshold = 0.70
        scores_pcc_threshold = 0.70

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

    # EP-sharded exact checks only make sense when the gate runs on host
    if gate_fallback_mode == GateComputeMode.HOST_ALL:
        experts_per_chip = n_routed_experts // (n_sp_devices * n_tp_devices)
        ref_dispatch_offsets, ref_expert_token_counts, _, _ = get_gate_outputs(
            indices=reference_topk_indices.view(n_sp_devices, seq_len_per_device, -1).int(),
            dispatch_group_size=n_sp_devices,
            num_routed_experts=n_routed_experts,
            experts_per_chip=experts_per_chip,
            seq_len_per_chip=seq_len_per_device,
            num_experts_per_tok=config.n_activated_experts,
            expert_dispatch_table=dispatch_table,
        )

        ep_composer = get_ep_mesh_composer(mesh_device)

        tt_dispatch_offsets = ttnn.unsqueeze_to_4D(tt_dispatch_offsets)
        host_dispatch_offsets = ttnn.to_torch(tt_dispatch_offsets, mesh_composer=ep_composer).squeeze(2).long()

        exact_dispatch_offsets = validate_composed(
            host_dispatch_offsets,
            ref_dispatch_offsets.long(),
            n_tp_devices,
            n_sp_devices,
            compare_exact,
            name="dispatch_offsets",
        )

        tt_total_counts_per_expert = ttnn.unsqueeze_to_4D(tt_total_counts_per_expert)
        host_tt_total_counts_per_expert = (
            ttnn.to_torch(tt_total_counts_per_expert, mesh_composer=ep_composer).squeeze(2).long()
        )
        ref_expert_token_counts = ref_expert_token_counts[:, 0, :].unsqueeze(1).expand(-1, n_sp_devices, -1).long()

        expert_token_counts = validate_composed(
            host_tt_total_counts_per_expert,
            ref_expert_token_counts,
            n_tp_devices,
            n_sp_devices,
            compare_exact,
            name="expert_token_counts",
        )

        all_results.extend([exact_dispatch_offsets, expert_token_counts])
    else:
        logger.info("Skipping dispatch_offsets and expert_token_counts exact checks (device gate mode)")
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
