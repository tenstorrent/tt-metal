# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for TtMoe configured with Kimi K2.6 hyperparameters.

Kimi K2.6 reuses the DeepSeek V3 architecture; only hyperparameters
differ (see KimiK26Config). For PCC we drive TtMoe and TorchMoe with
Kimi's MoE dimensions (384 routed experts, 8 experts/tok) on a Galaxy
8x4 mesh at total sequence lengths 1k and ~25k.

Notes:
- experts_per_chip = 384 / 32 = 12 on the 8x4 mesh.
- TtMoe and TorchMoe are constructed with their DSv3 defaults, then
  the gate's grouping (n_group, topk_group) and route_scale are
  overridden in-place to Kimi's values (1, 1, 2.827). Only HOST_ALL gate
  fallback is supported for Kimi values today (the device kernels TT_FATAL
  on n_groups != 8).
"""

import random

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.experimental.kimi_k26.reference.configuration_deepseek import DeepseekV3Config
from models.experimental.kimi_k26.kimi_k26_config import KimiK26Config
from models.experimental.kimi_k26.reference.modeling_deepseek import DeepseekV3MoE as KimiBundledMoE
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    create_gate_weights,
    create_shared_expert_weights,
    create_torch_expert_weights,
    extract_mesh_config,
    get_sp_mesh_composer,
    get_tp_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import compare_recall, validate_composed
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_validation_results
from tests.ttnn.utils_for_testing import comp_pcc


def _override_gate_to_kimi_tt(tt_moe):
    """
    Patch a freshly constructed TtMoe so host gate matches Kimi K2.6.
    """
    cfg = tt_moe.gate.config
    cfg.n_expert_groups = KimiK26Config.NUM_EXPERT_GROUPS
    cfg.n_limited_groups = KimiK26Config.NUM_LIMITED_GROUPS
    cfg.route_scale = KimiK26Config.ROUTE_SCALE


def _override_gate_to_kimi_torch(torch_moe):
    """
    Patch TorchMoe.gate (a ReferenceMoEGate) attributes to Kimi K2.6 values.
    """
    g = torch_moe.gate
    g.n_group = KimiK26Config.NUM_EXPERT_GROUPS
    g.topk_group = KimiK26Config.NUM_LIMITED_GROUPS
    g.routed_scaling_factor = KimiK26Config.ROUTE_SCALE


def _build_kimi_config() -> DeepseekV3Config:
    """HF DeepseekV3Config populated with Kimi K2.6 text-config values."""
    return DeepseekV3Config(
        vocab_size=KimiK26Config.VOCAB_SIZE,
        hidden_size=KimiK26Config.EMB_SIZE,
        intermediate_size=KimiK26Config.INTERMEDIATE_SIZE,
        moe_intermediate_size=KimiK26Config.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=KimiK26Config.NUM_LAYERS,
        num_attention_heads=KimiK26Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=KimiK26Config.NUM_KEY_VALUE_HEADS,
        q_lora_rank=KimiK26Config.Q_LORA_RANK,
        kv_lora_rank=KimiK26Config.KV_LORA_RANK,
        qk_nope_head_dim=KimiK26Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=KimiK26Config.QK_ROPE_HEAD_DIM,
        v_head_dim=KimiK26Config.V_HEAD_DIM,
        max_position_embeddings=KimiK26Config.MAX_POSITION_EMBEDDINGS,
        rope_theta=KimiK26Config.ROPE_THETA,
        rope_scaling={
            "type": "yarn",
            "factor": KimiK26Config.ROPE_SCALING_FACTOR,
            "original_max_position_embeddings": KimiK26Config.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "beta_fast": KimiK26Config.ROPE_SCALING_BETA_FAST,
            "beta_slow": KimiK26Config.ROPE_SCALING_BETA_SLOW,
            "mscale": KimiK26Config.ROPE_SCALING_MSCALE,
            "mscale_all_dim": KimiK26Config.ROPE_SCALING_MSCALE_ALL_DIM,
        },
        rms_norm_eps=KimiK26Config.RMS_NORM_EPS,
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=KimiK26Config.NUM_DENSE_LAYERS,
        n_routed_experts=KimiK26Config.NUM_ROUTED_EXPERTS,
        n_shared_experts=KimiK26Config.NUM_SHARED_EXPERTS,
        num_experts_per_tok=KimiK26Config.NUM_EXPERTS_PER_TOKEN,
        n_group=KimiK26Config.NUM_EXPERT_GROUPS,
        topk_group=KimiK26Config.NUM_LIMITED_GROUPS,
        routed_scaling_factor=KimiK26Config.ROUTE_SCALE,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )


def _run_kimi_moe_cpu(
    config: DeepseekV3Config,
    gate_weights: dict,
    routed_expert_weights: list[dict],
    shared_expert_weights: dict,
    x: torch.Tensor,
) -> torch.Tensor:
    """Forward Kimi's bundled DeepseekV3MoE on CPU.

    Reuses the same random gate / routed-expert / shared-expert weight tensors
    that drive the existing TorchMoe reference, repackaged into Kimi's
    state_dict layout. Returns shape matching `x`.
    """
    moe = KimiBundledMoE(config)
    state_dict = {
        "gate.weight": gate_weights["weight"].to(torch.bfloat16),
        "gate.e_score_correction_bias": gate_weights["e_score_correction_bias"].to(torch.bfloat16),
        "shared_experts.gate_proj.weight": shared_expert_weights["gate_proj"].to(torch.bfloat16),
        "shared_experts.up_proj.weight": shared_expert_weights["up_proj"].to(torch.bfloat16),
        "shared_experts.down_proj.weight": shared_expert_weights["down_proj"].to(torch.bfloat16),
    }
    for i, w in enumerate(routed_expert_weights):
        state_dict[f"experts.{i}.gate_proj.weight"] = w["gate_proj"].to(torch.bfloat16)
        state_dict[f"experts.{i}.up_proj.weight"] = w["up_proj"].to(torch.bfloat16)
        state_dict[f"experts.{i}.down_proj.weight"] = w["down_proj"].to(torch.bfloat16)

    moe.load_state_dict(state_dict, strict=False)
    moe = moe.eval().to(torch.bfloat16)

    with torch.no_grad():
        out = moe(x.to(torch.bfloat16))
    return out


@pytest.mark.parametrize(
    (
        "seq_len_per_chip, num_routed_experts, num_experts_per_tok, "
        "dispatch_buffer_capacity_factor, gate_fallback_mode, run_pcc_check"
    ),
    [
        pytest.param(
            128,
            KimiK26Config.NUM_ROUTED_EXPERTS,
            KimiK26Config.NUM_EXPERTS_PER_TOKEN,
            5,
            GateComputeMode.HOST_ALL,
            True,
            id="kimi-1k",
            marks=pytest.mark.timeout(0),
        ),
        pytest.param(
            640,
            KimiK26Config.NUM_ROUTED_EXPERTS,
            KimiK26Config.NUM_EXPERTS_PER_TOKEN,
            5,
            GateComputeMode.HOST_ALL,
            True,
            id="kimi-5k",
            marks=pytest.mark.timeout(0),
        ),
        pytest.param(
            3200,
            KimiK26Config.NUM_ROUTED_EXPERTS,
            KimiK26Config.NUM_EXPERTS_PER_TOKEN,
            5,
            GateComputeMode.HOST_ALL,
            True,
            id="kimi-25k",
            marks=pytest.mark.timeout(0),
        ),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            marks=[
                pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"),
                pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            ],
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_kimi_k26_moe(
    mesh_device,
    device_params,
    seq_len_per_chip,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    run_pcc_check,
    num_links,
    topology,
    gate_fallback_mode,
):
    mesh_device.disable_and_clear_program_cache()

    profiler.clear()
    profiler.start("test_ttnn_kimi_k26_moe")

    random.seed(42)
    torch.manual_seed(42)

    emb_dim = KimiK26Config.EMB_SIZE
    hidden_dim = KimiK26Config.MOE_INTERMEDIATE_SIZE

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups
    n_sp_devices, n_tp_devices = mesh_device.shape

    logger.info(
        f"KimiK26 MoE PCC: mesh={tuple(mesh_device.shape)}, "
        f"total_seq={dispatch_group_size * seq_len_per_chip} "
        f"(seq_len_per_chip={seq_len_per_chip}), "
        f"experts={num_routed_experts}, topk={num_experts_per_tok}"
    )

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
    )
    logger.info(
        f"experts_per_chip={experts_per_chip}, metadata_len={metadata_len}, "
        f"max_dispatch_buffer_token_size={max_dispatch_buffer_token_size}, "
        f"max_dispatched_tokens_per_expert={max_dispatched_tokens_per_expert}"
    )

    if run_pcc_check:
        all_routed_weights = create_torch_expert_weights(num_routed_experts, emb_dim, hidden_dim)
        shared_expert_weights = create_shared_expert_weights(emb_dim, hidden_dim)
    else:
        all_routed_weights = None
        shared_expert_weights = None

    gate_weights = create_gate_weights(num_routed_experts, emb_dim)

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    x = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(
        x,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    if run_pcc_check:
        torch_moe = TorchMoe(
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            expert_dispatch_table=expert_dispatch_table,
            num_dispatch_groups=num_dispatch_groups,
            routed_expert_weights=all_routed_weights,
            shared_expert_weights=shared_expert_weights,
            gate_weights=gate_weights,
        )
        _override_gate_to_kimi_torch(torch_moe)
        torch_output, torch_intermediates = torch_moe(x, return_intermediates=True)

    tt_moe = TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_links=num_links,
        topology=topology,
        routed_expert_weights=all_routed_weights,
        shared_expert_weights=shared_expert_weights,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        gate_weights=gate_weights,
        gate_fallback_mode=gate_fallback_mode,
    )
    assert (
        gate_fallback_mode == GateComputeMode.HOST_ALL
    ), "Kimi gate (n_group=1, topk_group=1) is only supported via host fallback"
    _override_gate_to_kimi_tt(tt_moe)
    ttnn.synchronize_device(mesh_device)

    tt_output, tt_intermediates = tt_moe(tt_x, return_intermediates=True)
    ttnn.synchronize_device(mesh_device)

    if not run_pcc_check:
        return

    all_passed = True
    tt_indices = ttnn.to_torch(
        tt_intermediates.gate_indices,
        mesh_composer=get_sp_mesh_composer(mesh_device),
        dtype=torch.int32,
    )
    target_recall = 0.99 if gate_fallback_mode == GateComputeMode.HOST_ALL else 0.90
    recall_result = validate_composed(
        tt_indices.view(1, n_sp_devices, seq_len_per_chip, -1),
        torch_intermediates.gate_indices.view(1, n_sp_devices, seq_len_per_chip, -1),
        1,
        n_sp_devices,
        compare_recall(target_recall),
        name="gate_indices_recall",
        broadcast_groups=n_tp_devices,
    )
    log_validation_results(
        results=[recall_result],
        num_dispatch_groups=n_tp_devices,
        dispatch_group_size=n_sp_devices,
        title="Gate Recall Validation",
    )
    if recall_result.passed:
        logger.info("[gate_indices_recall] PASSED")
    else:
        logger.error(
            f"[gate_indices_recall] FAILED {len(recall_result.mismatches)}/{recall_result.total} "
            f"below threshold {target_recall}"
        )
        all_passed = False

    dense_checks = [
        (
            "shared_output",
            tt_intermediates.shared_output,
            torch_intermediates.shared_output,
            get_tp_mesh_composer(mesh_device),
            0.997,
        ),
        (
            "routed_output",
            tt_intermediates.routed_output,
            torch_intermediates.routed_output,
            get_tp_mesh_composer(mesh_device),
            0.90,
        ),
        ("final_output", tt_output, torch_output, get_tp_mesh_composer(mesh_device), 0.96),
    ]
    for name, tt_tensor, torch_tensor, composer, threshold in dense_checks:
        if tt_tensor is None:
            logger.warning(f"[{name}] SKIPPED")
            continue
        tt_host = ttnn.to_torch(tt_tensor, mesh_composer=composer, dtype=torch.bfloat16)
        _, pcc = comp_pcc(torch_tensor.float(), tt_host.float())
        if pcc >= threshold:
            logger.info(f"[{name}] PASSED - PCC: {pcc:.6f} (threshold: {threshold})")
        else:
            logger.error(f"[{name}] FAILED - PCC: {pcc:.6f} below threshold {threshold}")
            all_passed = False

    if tt_output is not None:
        logger.info("Running Kimi-bundled DeepseekV3MoE reference")
        kimi_out = _run_kimi_moe_cpu(
            config=_build_kimi_config(),
            gate_weights=gate_weights,
            routed_expert_weights=all_routed_weights,
            shared_expert_weights=shared_expert_weights,
            x=x,
        )
        tt_final_host = ttnn.to_torch(tt_output, mesh_composer=get_tp_mesh_composer(mesh_device), dtype=torch.bfloat16)
        kimi_threshold = 0.96
        _, kimi_pcc = comp_pcc(kimi_out.float(), tt_final_host.float())
        if kimi_pcc >= kimi_threshold:
            logger.info(f"[final_output_vs_kimi_cpu] PASSED - PCC: {kimi_pcc:.6f} (threshold: {kimi_threshold})")
        else:
            logger.error(f"[final_output_vs_kimi_cpu] FAILED - PCC: {kimi_pcc:.6f} below threshold {kimi_threshold}")
            all_passed = False
        del kimi_out

    profiler.end("test_ttnn_kimi_k26_moe")
    for key in profiler.times:
        logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")

    assert all_passed, "One or more PCC checks failed; see logs."
