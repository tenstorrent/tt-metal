# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN DecoderBlock Test
Tests decoder fused operation with full pipeline:
- CCL Broadcast -> RMSNorm -> Matmul -> Gather -> RMSNorm2 -> Matmul2 (shuffled) -> Matmul3 (Qnope only) & RoPE (Qrope only) -> Interleaved Pre-SDPA Output
- Qnope output: [64, 1, 512] after matmul3
- Qrope output: [64, 1, 64] after RoPE
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.demo.decoder_stage import create_decoder_block_tensors
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock
from models.demos.deepseek_v3_b1.fused_ops.decoder_block.op import DecoderBlock
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata
from models.demos.deepseek_v3_b1.model_dimensions import RoutedExpert
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import create_fabric_router_config
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import (
    DENSE_LAYER_IDX,
    DENSE_SHARED_N,
    ROUTED_EXPERT_LAYER_IDX,
    extract_routed_expert_output,
)
from models.demos.deepseek_v3_b1.weights.prepare import (
    get_layer_raw_tensors,
    prepare_dense_layer_weights,
    prepare_moe_layer_weights,
)

MTP_LAYER_IDX = 61


def _decode_expert_upload_mode(expert_upload_mode: str) -> tuple[int, int | None]:
    """Map mode string to (num_routed_experts_to_upload, rigged_group_count).

    Supported modes:
      - unrigged_all_experts
      - rigged_groups1 ... rigged_groups8

    For rigged_groupsN we upload contiguous groups [0..N-1] and rig
    pseudo-random experts within those uploaded groups.
    """
    if expert_upload_mode == "unrigged_all_experts":
        return 256, None
    if expert_upload_mode.startswith("rigged_groups"):
        suffix = expert_upload_mode.removeprefix("rigged_groups")
        if suffix.isdigit():
            rigged_group_count = int(suffix)
            if not (1 <= rigged_group_count <= 8):
                raise ValueError(f"Invalid expert_upload_mode: {expert_upload_mode}")
            return rigged_group_count * 32, rigged_group_count
    raise ValueError(f"Invalid expert_upload_mode: {expert_upload_mode}")


# ============================================================================
# Test helpers: expert rigging + golden reference tensors
# ============================================================================


def rig_experts(state_dict, layer_idx, rigged_group_count):
    """Rig expert routing bias in state_dict for deterministic test routing.

    Generates RMS-normalized input for stability with rigged routing,
    modifies state_dict bias entries, and returns the rigged configuration.

    Returns (rigged_group_ids, rigged_expert_ids, torch_input).
    """
    K = 7168
    shape = (1, K)
    torch_input_f32 = torch.randn(shape, dtype=torch.float32)
    torch_input_f32 = torch_input_f32 / torch.sqrt(torch_input_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    torch_input = torch_input_f32.to(torch.bfloat16)

    g = torch.Generator()
    g.manual_seed(2026)
    if not (1 <= rigged_group_count <= 8):
        raise ValueError(f"rigged_group_count must be in [1, 8], got {rigged_group_count}")
    num_selected_groups = min(4, rigged_group_count)
    rigged_group_ids = torch.randperm(rigged_group_count, generator=g)[:num_selected_groups].tolist()

    total_rigged_experts = 8
    experts_per_group = [1] * num_selected_groups
    remaining = total_rigged_experts - num_selected_groups
    for _ in range(remaining):
        chosen_group = int(torch.randint(0, num_selected_groups, (1,), generator=g).item())
        experts_per_group[chosen_group] += 1

    rigged_expert_ids = {
        grp: torch.randperm(32, generator=g)[:num_experts].tolist()
        for grp, num_experts in zip(rigged_group_ids, experts_per_group, strict=True)
    }

    rigged_bias = torch.full((8, 32), -10.0, dtype=torch.bfloat16)
    for grp in rigged_group_ids:
        for exp in rigged_expert_ids[grp]:
            rigged_bias[grp, exp] = 10.0
    state_dict[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = rigged_bias.reshape(-1).contiguous()
    logger.info(
        f"Rigged experts enabled: groups={rigged_group_ids}, "
        f"experts={[(grp, rigged_expert_ids[grp]) for grp in rigged_group_ids]}"
    )

    return rigged_group_ids, rigged_expert_ids, torch_input


def create_decoder_golden_tensors(
    d,
    submesh,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    state_dict,
    layer_idx,
    reduce_root_coord=ttnn.MeshCoordinate(1, 1),
    *,
    metadata: DeepseekMetadata | None = None,
    max_seq_len: int = 128 * 1024,
    num_slots: int = 1,
    is_moe: bool = True,
    num_routed_experts: int = 0,
    rigged_group_ids=None,
    rigged_expert_ids=None,
):
    """Build golden PyTorch reference tensors for decoder validation.

    Reads intermediate CPU tensors and the on-device KV cache from d
    (the output of create_decoder_block_tensors).
    """
    if metadata is None:
        metadata = DeepseekMetadata()

    QNOPE_HEAD_DIM = 128
    K = 7168
    (
        golden_torch_matmul_weights,
        golden_torch_matmul2_weights,
        golden_torch_dkv_matmul_weights,
        golden_kv_b1,
        golden_kv_b2,
        golden_torch_o_proj_weights,
        golden_torch_gamma,
        golden_torch_rmsnorm2_gamma,
        golden_torch_dkv_rmsnorm_gamma,
        ffn_norm,
    ) = get_layer_raw_tensors(state_dict, layer_idx)

    total_kv_heads = golden_kv_b1.shape[0] // QNOPE_HEAD_DIM
    kv_lora_rank = golden_kv_b1.shape[1]
    golden_torch_matmul3_weights = golden_kv_b1.reshape(total_kv_heads, QNOPE_HEAD_DIM, kv_lora_rank)

    golden_total_qnope_heads = total_kv_heads
    golden_total_qrope_heads = total_kv_heads
    golden_moe_rmsnorm_gamma = ffn_norm.to(torch.bfloat16).float()

    def _sd_key(suffix):
        return f"model.layers.{layer_idx}.{suffix}"

    if is_moe:
        golden_moe_shared_gate = state_dict[_sd_key("mlp.shared_experts.gate_proj.weight")].T.contiguous()
        golden_moe_shared_up = state_dict[_sd_key("mlp.shared_experts.up_proj.weight")].T.contiguous()
        golden_moe_shared_down = state_dict[_sd_key("mlp.shared_experts.down_proj.weight")].T.contiguous()
        golden_moe_routing_weights = state_dict[_sd_key("mlp.gate.weight")].T.contiguous()
        golden_moe_bias = (
            state_dict[_sd_key("mlp.gate.e_score_correction_bias")].reshape(1, 8, 32).contiguous().to(torch.bfloat16)
        )
        golden_moe_gate_proj_dict = {}
        golden_moe_up_proj_dict = {}
        golden_moe_down_proj_dict = {}
        for e in range(num_routed_experts):
            w_g = state_dict[_sd_key(f"mlp.experts.{e}.gate_proj.weight")].T.contiguous()
            golden_moe_gate_proj_dict[e] = w_g.reshape(1, 1, K, -1)
            w_u = state_dict[_sd_key(f"mlp.experts.{e}.up_proj.weight")].T.contiguous()
            golden_moe_up_proj_dict[e] = w_u.reshape(1, 1, K, -1)
            w_d = state_dict[_sd_key(f"mlp.experts.{e}.down_proj.weight")].T.contiguous()
            golden_moe_down_proj_dict[e] = w_d.reshape(1, 1, -1, K)
    else:
        gate_full = state_dict[_sd_key("mlp.gate_proj.weight")].T.contiguous()
        up_full = state_dict[_sd_key("mlp.up_proj.weight")].T.contiguous()
        down_full = state_dict[_sd_key("mlp.down_proj.weight")].T.contiguous()
        golden_moe_shared_gate = gate_full[:, :DENSE_SHARED_N].contiguous()
        golden_moe_shared_up = up_full[:, :DENSE_SHARED_N].contiguous()
        golden_moe_shared_down = down_full[:DENSE_SHARED_N, :].contiguous()
        golden_moe_routing_weights = None
        golden_moe_bias = None
        golden_moe_gate_proj_dict = {}
        golden_moe_up_proj_dict = {}
        golden_moe_down_proj_dict = {}
        for e in range(8):
            start = DENSE_SHARED_N + e * RoutedExpert.GATE_PROJ_N
            end = start + RoutedExpert.GATE_PROJ_N
            golden_moe_gate_proj_dict[e] = gate_full[:, start:end].reshape(1, 1, K, -1)
            golden_moe_up_proj_dict[e] = up_full[:, start:end].reshape(1, 1, K, -1)
            golden_moe_down_proj_dict[e] = down_full[start:end, :].reshape(1, 1, -1, K)

    num_devices = mesh_rows * mesh_cols
    per_device_max_seq_len = max_seq_len // mesh_rows
    kvpe_dim = d["torch_kv_cache"].shape[-1]
    kv_cache_bfp8_before_op = ttnn.to_torch(
        d["ttnn_kv_cache"], mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
    ).reshape(num_devices, num_slots, 1, per_device_max_seq_len, kvpe_dim)

    return {
        "golden_torch_input": d["torch_input"],
        "golden_torch_gamma": golden_torch_gamma,
        "golden_torch_matmul_weights": golden_torch_matmul_weights,
        "golden_torch_rmsnorm2_gamma": golden_torch_rmsnorm2_gamma,
        "golden_torch_matmul2_weights": golden_torch_matmul2_weights,
        "golden_torch_matmul3_weights": golden_torch_matmul3_weights,
        "golden_torch_sin": d["torch_sin"],
        "golden_torch_cos": d["torch_cos"],
        "golden_metadata": metadata,
        "golden_torch_dkv_matmul_weights": golden_torch_dkv_matmul_weights,
        "golden_torch_dkv_rmsnorm_gamma": golden_torch_dkv_rmsnorm_gamma,
        "golden_torch_kv_cache": d["torch_kv_cache"],
        "golden_scale": d["scale"],
        "golden_torch_kv_b2_proj_weights": golden_kv_b2,
        "golden_torch_o_proj_weights": golden_torch_o_proj_weights,
        "golden_total_qnope_heads": golden_total_qnope_heads,
        "golden_total_qrope_heads": golden_total_qrope_heads,
        "golden_kv_cache_bfp8_before_op": kv_cache_bfp8_before_op,
        "num_slots": num_slots,
        "per_device_max_seq_len": per_device_max_seq_len,
        "golden_moe_rmsnorm_gamma": golden_moe_rmsnorm_gamma,
        "golden_moe_shared_gate": golden_moe_shared_gate,
        "golden_moe_shared_up": golden_moe_shared_up,
        "golden_moe_shared_down": golden_moe_shared_down,
        "golden_moe_routing_weights": golden_moe_routing_weights,
        "golden_moe_bias": golden_moe_bias,
        "golden_moe_gate_proj_dict": golden_moe_gate_proj_dict,
        "golden_moe_up_proj_dict": golden_moe_up_proj_dict,
        "golden_moe_down_proj_dict": golden_moe_down_proj_dict,
        "rigged_group_ids": rigged_group_ids,
        "rigged_expert_ids": rigged_expert_ids,
    }


@pytest.mark.parametrize(
    "sender_row, sender_col",
    [
        (1, 0),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [False])
@pytest.mark.parametrize("reduce_cluster_axis", [1])
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
@pytest.mark.parametrize("num_iters", [(1)])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])
@pytest.mark.parametrize(
    "position_id",
    [
        0,
        127,
        511,
        1023,
        2047,
        4096,  # (1 + partial,1,1,1): partial into dev0 (if SP = 4)
        pytest.param(6644, marks=pytest.mark.skip_post_commit),  # (2,2,1 + partial,1): partial into dev2 (if SP = 4)
        pytest.param(9916, marks=pytest.mark.skip_post_commit),  # (3,2 + partial,2,2): partial into dev1 (if SP = 4)
        pytest.param(11664, marks=pytest.mark.skip_post_commit),  # (3,3,3,2 + partial): partial into dev3 (if SP = 4)
    ],
)  # Must test 128 chunk aligned decode positions, add other tests when causal masks are in for SDPA
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
            "worker_l1_size": 1431568,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.parametrize("num_internal_iterations", [1])
@pytest.mark.parametrize(
    "expert_upload_mode",
    [
        "unrigged_all_experts",
        pytest.param("rigged_groups1", marks=pytest.mark.skip_post_commit),
        pytest.param("rigged_groups2", marks=pytest.mark.skip_post_commit),
        pytest.param("rigged_groups3", marks=pytest.mark.skip_post_commit),
        pytest.param("rigged_groups4", marks=pytest.mark.skip_post_commit),
        pytest.param("rigged_groups5", marks=pytest.mark.skip_post_commit),
        pytest.param("rigged_groups6", marks=pytest.mark.skip_post_commit),
        pytest.param("rigged_groups7", marks=pytest.mark.skip_post_commit),
        "rigged_groups8",
    ],
)
@pytest.mark.parametrize(
    "enable_routing, use_hardcoded_expert_index, num_routed_experts",
    [
        # (True, True, 8),
        pytest.param(True, False, 256, marks=pytest.mark.skip_post_commit),
    ],
    ids=[
        # "hardcoded_experts",
        "full_routing",
    ],
)
@pytest.mark.parametrize(
    "decoder_layer_idx",
    [
        ROUTED_EXPERT_LAYER_IDX,
        pytest.param(MTP_LAYER_IDX, id="mtp_layer_61"),
    ],
)
@pytest.mark.parametrize("use_real_weights", [False, True], ids=["random_weights", "real_weights"])
@pytest.mark.parametrize(
    "validate_standalone_mla",
    [pytest.param(True, marks=pytest.mark.skip_post_commit), False],
    ids=["validate_standalone_mla", "just_decoder_mla"],
)
@pytest.mark.parametrize(
    "validate_standalone_moe",
    [pytest.param(True, marks=pytest.mark.skip_post_commit), False],
    ids=["validate_standalone_moe", "just_decoder_moe"],
)
@pytest.mark.parametrize("slot_id, num_slots", [(0, 1)])
@pytest.mark.requires_grid_size((13, 10))
def test_decoder(
    bh_2d_mesh_device,
    device_params,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    epsilon,
    use_fp32,
    reduce_cluster_axis,
    num_iters,
    max_seq_len,
    position_id,
    noc_mode,
    num_internal_iterations,
    expert_upload_mode,
    enable_routing,
    use_hardcoded_expert_index,
    num_routed_experts,
    decoder_layer_idx,
    use_real_weights,
    validate_standalone_mla,
    validate_standalone_moe,
    slot_id,
    num_slots,
    get_reference_model_state_dict,
):
    """Test TTNN decoder fused operation with CCL broadcast, kv cache, mla, reduce, residual add"""
    torch.manual_seed(0)
    num_devices = mesh_rows * mesh_cols
    logger.info(f"Number of devices: {num_devices}")
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than available")

    if use_real_weights and expert_upload_mode != "unrigged_all_experts":
        pytest.skip("Real-weight decoder tests require unrigged_all_experts")
    if use_real_weights and not os.getenv("DEEPSEEK_V3_HF_MODEL"):
        pytest.skip("DEEPSEEK_V3_HF_MODEL must be set to run real MTP layer tests")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()

    effective_num_routed_experts, rigged_group_count = _decode_expert_upload_mode(expert_upload_mode)
    logger.info(f"Expert upload mode: {expert_upload_mode}")
    if effective_num_routed_experts != num_routed_experts:
        logger.info(
            "Mode {}: uploading {} routed experts instead of {}",
            expert_upload_mode,
            effective_num_routed_experts,
            num_routed_experts,
        )

    logger.info("Preparing model state dict...")
    state_dict = get_reference_model_state_dict(
        layer_idx=decoder_layer_idx,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=effective_num_routed_experts,
        random_weights=not use_real_weights,
    )

    rigged_group_ids = None
    rigged_expert_ids = None
    torch_input = None
    if rigged_group_count is not None:
        rigged_group_ids, rigged_expert_ids, torch_input = rig_experts(
            state_dict, ROUTED_EXPERT_LAYER_IDX, rigged_group_count
        )

    logger.info("Preparing layer weights on device...")
    layer_weights = prepare_moe_layer_weights(
        submesh,
        state_dict,
        ROUTED_EXPERT_LAYER_IDX,
        num_routed_experts=effective_num_routed_experts,
        move_to_device=True,
    )

    logger.info("Creating decoder block tensors...")
    d = create_decoder_block_tensors(
        submesh,
        mesh_rows,
        mesh_cols,
        sender_row,
        sender_col,
        position_id,
        max_seq_len=max_seq_len,
        weights=layer_weights,
        metadata=DeepseekMetadata(position_id=position_id, slot_id=slot_id),
        num_slots=num_slots,
        is_moe=True,
        validate_debug_tensors=validate_standalone_mla or validate_standalone_moe,
        torch_input=torch_input,
    )

    logger.info("Creating golden reference tensors...")
    golden = create_decoder_golden_tensors(
        d,
        submesh,
        mesh_rows,
        mesh_cols,
        sender_row,
        sender_col,
        state_dict,
        ROUTED_EXPERT_LAYER_IDX,
        metadata=DeepseekMetadata(position_id=position_id, slot_id=slot_id),
        max_seq_len=max_seq_len,
        num_slots=num_slots,
        is_moe=True,
        num_routed_experts=effective_num_routed_experts,
        rigged_group_ids=rigged_group_ids,
        rigged_expert_ids=rigged_expert_ids,
    )
    d.update(golden)

    num_cores = device_grid_size.x * device_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, device_grid_size, row_wise=True)
    ttnn.synchronize_device(submesh)
    reduce_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
    persistent_next_iter_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 1)
    ttnn.synchronize_device(submesh)

    num_links_bcast = 1
    num_links_allreduce = 2
    attn_semaphores = AttentionBlock.create_semaphores(
        submesh, num_links_bcast=num_links_bcast, num_links_allreduce=num_links_allreduce
    )
    moe_semaphores = MoeOp.create_semaphores(submesh)

    logger.info("Done setup")

    # ========================================================================
    # Run standalone AttentionBlock.op as sanity reference (uses cloned KV cache)
    # ========================================================================
    ttnn_attn_ref_output_torch = None
    if validate_standalone_mla:
        logger.info(f"Running standalone AttentionBlock.op with position_id={position_id}...")
        attn_ref_semaphores = AttentionBlock.create_semaphores(
            submesh, num_links_bcast=num_links_bcast, num_links_allreduce=num_links_allreduce
        )
        ttnn_attn_ref_result = AttentionBlock.op(
            d["input_tensor_mesh"],
            d["gamma_overlapped"],
            d["matmul_weights_overlapped"],
            d["rmsnorm2_gamma_overlapped"],
            d["matmul2_weights_overlapped"],
            d["matmul3_weights_overlapped"],
            d["ttnn_qrope_sin"],
            d["ttnn_qrope_cos"],
            d["ttnn_trans_mat"],
            d["ttnn_krope_cos"],
            d["ttnn_krope_sin"],
            d["dkv_matmul_weights_overlapped"],
            d["dkv_rmsnorm_gamma_overlapped"],
            d["ttnn_kv_cache_attn_ref"],
            d["scale"],
            d["ttnn_sdpa_output"],
            d["sdpa_kv_cache_buffer"],
            d["sdpa_out_interm_buffer"],
            d["sender_coord"],
            d["kv_b2_overlapped"],
            d["o_proj_overlapped"],
            d["ttnn_sdpa_input_l"],
            d["ttnn_sdpa_input_ms"],
            d["ttnn_sdpa_output_l"],
            d["ttnn_sdpa_intermediate_recv"],
            d["ttnn_sdpa_forwarder_scratch"],
            d["device_chunk_size"],
            d["ttnn_attn_ref_output"],
            attn_ref_semaphores,
            reduce_cluster_axis,
            0,  # sdpa_cluster_axis
            num_links_bcast,
            num_links_allreduce,
            epsilon,
            use_fp32,
            False,  # skip_ccl
            noc_mode,
            num_iterations=1,
            fabric_config=device_params["fabric_config"],
        )
        ttnn.synchronize_device(submesh)
        ttnn_attn_ref_output_torch = ttnn.to_torch(
            d["ttnn_attn_ref_output"], mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
        )
        logger.info("Standalone AttentionBlock.op completed.")

    # ========================================================================
    # Run decoder operation
    # ========================================================================
    logger.info(f"Running decoder operation with position_id={position_id}...")
    decoder_program_context = DecoderBlock.get_program_context(
        # AttentionBlock parameters
        d["input_tensor_mesh"],
        d["gamma_overlapped"],
        d["matmul_weights_overlapped"],
        d["rmsnorm2_gamma_overlapped"],
        d["matmul2_weights_overlapped"],
        d["matmul3_weights_overlapped"],
        d["ttnn_qrope_sin"],
        d["ttnn_qrope_cos"],
        d["ttnn_trans_mat"],
        d["ttnn_krope_cos"],
        d["ttnn_krope_sin"],
        d["dkv_matmul_weights_overlapped"],
        d["dkv_rmsnorm_gamma_overlapped"],
        d["ttnn_kv_cache"],
        d["scale"],
        d["sdpa_kv_cache_buffer"],
        d["sdpa_out_interm_buffer"],
        d["sender_coord"],
        # Post-SDPA parameters
        # Post-SDPA
        d["kv_b2_overlapped"],
        d["o_proj_overlapped"],
        d["ttnn_sdpa_input_l"],
        d["ttnn_sdpa_input_ms"],
        d["ttnn_sdpa_output_l"],
        d["ttnn_sdpa_intermediate_recv"],
        d["ttnn_sdpa_forwarder_scratch"],
        d["device_chunk_size"],
        d["ttnn_attention_block_output"],
        attention_block_semaphores=attn_semaphores,
        # MoE parameters
        shared_residual_mcast_src_tensor=d["ttnn_residual_mcast_src"],
        gate_mm_weights_tensor=d["gate_mm_overlapped"],
        gate_bias_tensor=d["ttnn_gate_bias"],
        gate_indices_tensor=d["ttnn_gate_indices"],
        gate_output_scores_tensor=d["gate_output_scores_tensor"],
        gate_output_indices_tensor=d["gate_output_indices_tensor"],
        gate_proj_weights_tensor=d["gate_proj_weights"],
        up_proj_weights_tensor=d["up_proj_weights"],
        down_proj_weights_tensor=d["down_proj_weights"],
        moe_final_output_tensor=None,
        rmsnorm_gamma_tensor=d["ffn_norm_overlapped"],
        shared_gate_weights_overlapped=d["shared_gate_weights_overlapped"],
        shared_up_weights_overlapped=d["shared_up_weights_overlapped"],
        shared_down_weights_tensor=d["shared_down_weights_tensor"],
        shared_k_parallel=d["shared_k_parallel"],
        shared_n_parallel=d["shared_n_parallel"],
        moe_semaphores=moe_semaphores,
        reduce_intermediate_tensors=d["reduce_intermediate_tensors"],
        reduce_output_tensor=d["reduce_output_tensor"],
        reduce_semaphores=reduce_semaphores,
        reduce_root_coord=d["reduce_root_coord"],
        # Shared parameters
        enable_routing=True,
        reduce_cluster_axis=reduce_cluster_axis,
        sdpa_cluster_axis=0,  # sdpa_cluster_axis
        num_links_bcast=num_links_bcast,
        num_links_allreduce=num_links_allreduce,
        epsilon=epsilon,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        noc_mode=noc_mode,
        num_iterations=num_internal_iterations,
        upstream_socket=None,
        downstream_sockets=None,
        fabric_config=device_params["fabric_config"],
        persistent_next_iter_semaphore=persistent_next_iter_semaphore,
        persistent_mode=False,
    )
    for i in range(num_iters):
        moe_final_output_tensor, attention_block_output_tensor = DecoderBlock.execute(*decoder_program_context)
    ttnn.synchronize_device(submesh)

    kv_cache_output_torch_flat = ttnn.to_torch(
        d["ttnn_kv_cache"], mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
    )
    kv_cache_output_torch = kv_cache_output_torch_flat.reshape(
        num_devices, d["num_slots"], 1, d["per_device_max_seq_len"], -1
    )

    ttnn_attention_output = ttnn.to_torch(
        attention_block_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
    )

    # ========================================================================
    # Extract decoder MoE output and reduce root info (always needed for golden)
    # ========================================================================
    decoder_moe_output = ttnn.to_torch(moe_final_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    root_coord_tuple = d["reduce_root_coord"]
    root_device_idx = root_coord_tuple[0] * mesh_cols + root_coord_tuple[1]
    decoder_moe_output_root = decoder_moe_output[root_device_idx]
    decoder_moe_output_valid = extract_routed_expert_output(
        decoder_moe_output_root.unsqueeze(0),
        d["num_gate_proj_cores"],
        RoutedExpert.FINAL_OUTPUT_WIDTH_PER_CORE,
        d["per_core_down_proj_N"],
    )

    # ========================================================================
    # Run standalone MoeOp.op on MLA output (validate golden MoE path)
    # ========================================================================
    moe_device_output_valid = None
    if validate_standalone_moe:
        logger.info(
            f"Running standalone MoeOp.op (enable_routing={enable_routing}, hardcoded={use_hardcoded_expert_index})..."
        )
        moe_ref_semaphores = MoeOp.create_semaphores(submesh)
        moe_ref_reduce_sems = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
        ttnn.synchronize_device(submesh)

        moe_ref_result = MoeOp.op(
            attention_block_output_tensor,
            gate_mm_weights_tensor=d["gate_mm_overlapped"],
            gate_bias_tensor=d["ttnn_gate_bias"],
            gate_indices_tensor=d["ttnn_gate_indices"],
            gate_output_scores_tensor=d["moe_ref_gate_output_scores"],
            gate_output_indices_tensor=d["moe_ref_gate_output_indices"],
            gate_proj_weights_tensor=d["gate_proj_weights"],
            up_proj_weights_tensor=d["up_proj_weights"],
            down_proj_weights_tensor=d["down_proj_weights"],
            rmsnorm_gamma_tensor=d["ffn_norm_overlapped"],
            shared_gate_weights_overlapped=d["shared_gate_weights_overlapped"],
            shared_up_weights_overlapped=d["shared_up_weights_overlapped"],
            shared_down_weights_tensor=d["shared_down_weights_tensor"],
            shared_k_parallel=d["shared_k_parallel"],
            shared_n_parallel=d["shared_n_parallel"],
            enable_routing=enable_routing,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            sdpa_kv_cache_buffer=d["sdpa_kv_cache_buffer"],
            sdpa_out_interm_buffer=d["sdpa_out_interm_buffer"],
            num_iterations=1,
            reduce_intermediate_tensors=d["moe_ref_reduce_intermediate"],
            reduce_output_tensor=d["moe_ref_reduce_output"],
            reduce_semaphores=moe_ref_reduce_sems,
            reduce_root_coord=ttnn.MeshCoordinate(d["reduce_root_coord"]),
            semaphores=moe_ref_semaphores,
            noc_mode=noc_mode,
        )
        ttnn.synchronize_device(submesh)
        logger.info("Standalone MoeOp.op completed.")

        if enable_routing:
            moe_ref_scores_tensor, moe_ref_indices_tensor, moe_ref_result = moe_ref_result
            moe_ref_scores_torch = ttnn.to_torch(
                moe_ref_scores_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
            )
            moe_ref_indices_torch = ttnn.to_torch(
                moe_ref_indices_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
            )
        else:
            moe_ref_scores_torch = None
            moe_ref_indices_torch = None

        moe_reduce_torch = ttnn.to_torch(moe_ref_result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
        moe_reduce_root = moe_reduce_torch[root_device_idx]
        moe_device_output_valid = extract_routed_expert_output(
            moe_reduce_root.unsqueeze(0),
            d["num_gate_proj_cores"],
            RoutedExpert.FINAL_OUTPUT_WIDTH_PER_CORE,
            d["per_core_down_proj_N"],
        )

    # ========================================================================
    # Compute golden reference
    # ========================================================================
    logger.info("Computing golden reference...")

    device_chunk_size = d["device_chunk_size"]
    num_sp = mesh_rows
    owning_sp_device = (position_id // device_chunk_size) % num_sp

    QNOPE_HEAD_DIM = 128
    QROPE_HEAD_DIM = 64
    KNOPE_DIM = 512
    KROPE_DIM = 64
    HEADS_PER_ROW = 8

    full_q, golden_new_kv, mla_output, moe_scores, moe_indices, moe_output = DecoderBlock.golden(
        d["golden_torch_input"],
        d["golden_torch_gamma"],
        d["golden_torch_matmul_weights"],
        d["golden_torch_rmsnorm2_gamma"],
        d["golden_torch_matmul2_weights"],
        d["golden_torch_matmul3_weights"],
        d["golden_torch_sin"],
        d["golden_torch_cos"],
        d["golden_metadata"],
        d["golden_torch_dkv_matmul_weights"],
        d["golden_torch_dkv_rmsnorm_gamma"],
        d["golden_torch_kv_cache"],
        d["golden_scale"],
        d["golden_torch_kv_b2_proj_weights"],
        d["golden_torch_o_proj_weights"],
        epsilon=epsilon,
        num_qnope_heads=d["golden_total_qnope_heads"],
        num_qrope_heads=d["golden_total_qrope_heads"],
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
        nope_dim=KNOPE_DIM,
        rope_dim=KROPE_DIM,
        # MoE golden parameters
        moe_shared_gate_weights=d["golden_moe_shared_gate"],
        moe_shared_up_weights=d["golden_moe_shared_up"],
        moe_shared_down_weights=d["golden_moe_shared_down"],
        moe_gate_proj_weights_dict=d["golden_moe_gate_proj_dict"],
        moe_up_proj_weights_dict=d["golden_moe_up_proj_dict"],
        moe_down_proj_weights_dict=d["golden_moe_down_proj_dict"],
        moe_rmsnorm_gamma=d["golden_moe_rmsnorm_gamma"],
        moe_rmsnorm_epsilon=epsilon,
        moe_routing_weights=d["golden_moe_routing_weights"],
        moe_bias=d["golden_moe_bias"],
        moe_gate_eps=RoutedExpert.GATE_EPS,
        moe_gate_scaling_factor=RoutedExpert.GATE_SCALING_FACTOR,
        moe_enable_routing=enable_routing,
    )

    logger.info(f"MLA output: {mla_output}")

    logger.info(f"Golden computed (owning_sp_device={owning_sp_device}, device_chunk_size={device_chunk_size})")

    def get_local_seq_len(sp_idx):
        """Return how many KV positions SP device sp_idx holds for the current global position_id."""
        sp_block = device_chunk_size * num_sp
        num_full_blocks = position_id // sp_block
        remainder = position_id % sp_block
        dev_start = sp_idx * device_chunk_size
        dev_end = dev_start + device_chunk_size
        dev_contrib = max(0, min(remainder, dev_end) - dev_start)
        return num_full_blocks * device_chunk_size + dev_contrib

    # ========================================================================
    # Validate KV cache outputs (per SP device)
    # ========================================================================
    for device_idx in range(mesh_rows * mesh_cols):
        sp_group = device_idx // mesh_cols
        local_seq_len = get_local_seq_len(sp_group)

        if local_seq_len == 0 and sp_group != owning_sp_device:
            logger.info(f"Device {device_idx} (SP={sp_group}) no data yet, skipped")
            continue

        assert torch.equal(
            d["golden_kv_cache_bfp8_before_op"][device_idx, slot_id, ..., :local_seq_len, :],
            kv_cache_output_torch[device_idx, slot_id, ..., :local_seq_len, :],
        ), f"Device {device_idx} (SP={sp_group}) KV Cache before and after op mismatch"
        logger.info(f"Device {device_idx} (SP={sp_group}) old cache validation passed")

        if sp_group == owning_sp_device:
            compare_kv_cache = kv_cache_output_torch[device_idx, slot_id, ..., local_seq_len, :]
            expected_nope = golden_new_kv[..., :KNOPE_DIM]
            expected_rope = golden_new_kv[..., KNOPE_DIM:]
            compare_nope = compare_kv_cache[..., :KNOPE_DIM]
            compare_rope = compare_kv_cache[..., KNOPE_DIM:]

            nope_passing, nope_pcc = comp_pcc(compare_nope, expected_nope, 0.98)
            logger.info(f"Device {device_idx} (SP={sp_group}) KV Cache NOPE PCC: {nope_pcc}")
            assert nope_passing, f"Device {device_idx} (SP={sp_group}) KV Cache NOPE PCC check failed: {nope_pcc}"

            rope_passing, rope_pcc = comp_pcc(compare_rope, expected_rope, 0.98)
            logger.info(f"Device {device_idx} (SP={sp_group}) KV Cache ROPE PCC: {rope_pcc}")
            assert rope_passing, f"Device {device_idx} (SP={sp_group}) KV Cache ROPE PCC check failed: {rope_pcc}"

        # Other slots must be completely unchanged
        for other_slot in range(num_slots):
            if other_slot == slot_id:
                continue
            assert torch.equal(
                d["golden_kv_cache_bfp8_before_op"][device_idx, other_slot],
                kv_cache_output_torch[device_idx, other_slot],
            ), f"Device {device_idx} (SP={sp_group}) KV Cache slot {other_slot} was modified but should be untouched"
        if num_slots > 1:
            logger.info(f"Device {device_idx} (SP={sp_group}) other slots unchanged validation passed")

    if moe_scores is not None:
        logger.info(f"Golden MoE scores: {moe_scores}")
        logger.info(f"Golden MoE indices: {moe_indices}")

    # ========================================================================
    # Validate decoder MLA output (full pipeline)
    # ========================================================================
    for device_idx in range(mesh_rows * mesh_cols):
        received = ttnn_attention_output[device_idx : device_idx + 1, :]
        passing, pcc = comp_pcc(mla_output, received, 0.98)
        logger.info(f"Device {device_idx} DecoderBlock Output PCC: {pcc}")
        logger.info(f"Golden MLA output: {mla_output}")
        logger.info(f"DecoderBlock MLA output: {received}")
        if validate_standalone_mla:
            pure_mla = ttnn_attn_ref_output_torch[device_idx : device_idx + 1, :]
            pure_mla_passing, pure_mla_pcc = comp_pcc(mla_output, pure_mla, 0.98)
            logger.info(f"Pure MLA PCC: {pure_mla_pcc}")
            logger.info(f"Pure MLA output: {pure_mla}")
        assert passing, f"Device {device_idx} DecoderBlock Output PCC check failed: {pcc}"

    # ========================================================================
    # Validate MoE output vs DecoderBlock golden MoE output
    # ========================================================================
    # Golden with moe_num_devices>1 computes per-device golden with TP-sharded shared
    # weights and per-device expert indices, then sums — matching reduce-to-one exactly.
    passing, pcc = comp_pcc(moe_output.flatten(), decoder_moe_output_valid.flatten(), 0.97)
    logger.info(f"MoE PCC (decoder vs golden): {pcc}")
    logger.info(f"Golden MoE output: {moe_output.flatten()[:8]}")
    logger.info(f"DecoderBlock MoE output: {decoder_moe_output_valid.flatten()[:8]}")

    if validate_standalone_moe:
        pure_moe_passing, pure_moe_pcc = comp_pcc(moe_output.flatten(), moe_device_output_valid.flatten(), 0.97)
        logger.info(f"Pure MoE PCC (standalone vs golden): {pure_moe_pcc}")
        logger.info(f"Pure MoE output: {moe_device_output_valid.flatten()[:8]}")

        device_passing, device_pcc = comp_pcc(
            decoder_moe_output_valid.flatten(), moe_device_output_valid.flatten(), 0.996
        )
        logger.info(f"Pure MoE vs Decoder MoE PCC: {device_pcc}")

        if use_hardcoded_expert_index:
            assert pure_moe_passing, f"Standalone MoE PCC check failed: {pure_moe_pcc}"
        assert device_passing, f"Pure MoE vs Decoder MoE PCC check failed: {device_pcc}"
    assert passing, f"DecoderBlock MoE PCC check failed: {pcc}"
    logger.info("✓ DecoderBlock mesh test passed!")
    ttnn.synchronize_device(submesh)


@pytest.mark.parametrize(
    "sender_row, sender_col",
    [
        (1, 0),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [False])
@pytest.mark.parametrize("reduce_cluster_axis", [1])
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
@pytest.mark.parametrize("num_iters", [(1)])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])
@pytest.mark.parametrize(
    "position_id",
    [
        0,
        127,
        pytest.param(511, marks=pytest.mark.skip_post_commit),
        pytest.param(1023, marks=pytest.mark.skip_post_commit),
        pytest.param(11664, marks=pytest.mark.skip_post_commit),  # (3,3,3,2 + partial): partial into dev3 (if SP = 4)
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
            "worker_l1_size": 1374544,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.parametrize("num_internal_iterations", [1])
@pytest.mark.parametrize("slot_id, num_slots", [(0, 2), (1, 2)])
@pytest.mark.requires_grid_size((13, 10))
def test_decoder_mlp(
    bh_2d_mesh_device,
    device_params,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    epsilon,
    use_fp32,
    reduce_cluster_axis,
    num_iters,
    max_seq_len,
    position_id,
    noc_mode,
    num_internal_iterations,
    slot_id,
    num_slots,
    get_reference_model_state_dict,
):
    """Test TTNN decoder fused operation for a dense (MLP) layer with enable_routing=False."""
    torch.manual_seed(0)
    num_devices = mesh_rows * mesh_cols
    logger.info(f"Number of devices: {num_devices}")
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than available")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()

    logger.info("Preparing dense MLP model state dict...")
    state_dict = get_reference_model_state_dict(
        layer_idx=DENSE_LAYER_IDX,
        is_moe=False,
        seed=RoutedExpert.SEED,
    )

    logger.info("Preparing dense layer weights on device...")
    layer_weights = prepare_dense_layer_weights(submesh, state_dict, DENSE_LAYER_IDX, move_to_device=True)

    logger.info("Creating dense decoder block tensors...")
    d = create_decoder_block_tensors(
        submesh,
        mesh_rows,
        mesh_cols,
        sender_row,
        sender_col,
        position_id,
        max_seq_len=max_seq_len,
        weights=layer_weights,
        metadata=DeepseekMetadata(position_id=position_id, slot_id=slot_id),
        num_slots=num_slots,
        is_moe=False,
    )

    logger.info("Creating golden reference tensors...")
    golden = create_decoder_golden_tensors(
        d,
        submesh,
        mesh_rows,
        mesh_cols,
        sender_row,
        sender_col,
        state_dict,
        DENSE_LAYER_IDX,
        metadata=DeepseekMetadata(position_id=position_id, slot_id=slot_id),
        max_seq_len=max_seq_len,
        num_slots=num_slots,
        is_moe=False,
    )
    d.update(golden)

    num_cores = device_grid_size.x * device_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, device_grid_size, row_wise=True)
    ttnn.synchronize_device(submesh)
    reduce_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
    persistent_next_iter_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 1)
    ttnn.synchronize_device(submesh)

    num_links_bcast = 1
    num_links_allreduce = 2
    attn_semaphores = AttentionBlock.create_semaphores(
        submesh, num_links_bcast=num_links_bcast, num_links_allreduce=num_links_allreduce
    )
    moe_semaphores = MoeOp.create_semaphores(submesh)

    logger.info(f"Running dense decoder operation with position_id={position_id}...")
    decoder_program_context = DecoderBlock.get_program_context(
        # AttentionBlock parameters
        d["input_tensor_mesh"],
        d["gamma_overlapped"],
        d["matmul_weights_overlapped"],
        d["rmsnorm2_gamma_overlapped"],
        d["matmul2_weights_overlapped"],
        d["matmul3_weights_overlapped"],
        d["ttnn_qrope_sin"],
        d["ttnn_qrope_cos"],
        d["ttnn_trans_mat"],
        d["ttnn_krope_cos"],
        d["ttnn_krope_sin"],
        d["dkv_matmul_weights_overlapped"],
        d["dkv_rmsnorm_gamma_overlapped"],
        d["ttnn_kv_cache"],
        d["scale"],
        d["sdpa_kv_cache_buffer"],
        d["sdpa_out_interm_buffer"],
        d["sender_coord"],
        # Post-SDPA parameters
        d["kv_b2_overlapped"],
        d["o_proj_overlapped"],
        d["ttnn_sdpa_input_l"],
        d["ttnn_sdpa_input_ms"],
        d["ttnn_sdpa_output_l"],
        d["ttnn_sdpa_intermediate_recv"],
        d["ttnn_sdpa_forwarder_scratch"],
        d["device_chunk_size"],
        d["ttnn_attention_block_output"],
        attention_block_semaphores=attn_semaphores,
        # MoE parameters (no gate_mm / routing tensors for dense MLP)
        shared_residual_mcast_src_tensor=d["ttnn_residual_mcast_src"],
        gate_mm_weights_tensor=None,
        gate_bias_tensor=None,
        gate_indices_tensor=None,
        gate_output_scores_tensor=None,
        gate_output_indices_tensor=None,
        gate_proj_weights_tensor=d["gate_proj_weights"],
        up_proj_weights_tensor=d["up_proj_weights"],
        down_proj_weights_tensor=d["down_proj_weights"],
        moe_final_output_tensor=None,
        rmsnorm_gamma_tensor=d["ffn_norm_overlapped"],
        shared_gate_weights_overlapped=d["shared_gate_weights_overlapped"],
        shared_up_weights_overlapped=d["shared_up_weights_overlapped"],
        shared_down_weights_tensor=d["shared_down_weights_tensor"],
        shared_k_parallel=d["shared_k_parallel"],
        shared_n_parallel=d["shared_n_parallel"],
        moe_semaphores=moe_semaphores,
        reduce_intermediate_tensors=d["reduce_intermediate_tensors"],
        reduce_output_tensor=d["reduce_output_tensor"],
        reduce_semaphores=reduce_semaphores,
        reduce_root_coord=ttnn.MeshCoordinate(d["reduce_root_coord"]),
        # Shared parameters
        enable_routing=False,
        reduce_cluster_axis=reduce_cluster_axis,
        sdpa_cluster_axis=0,
        num_links_bcast=num_links_bcast,
        num_links_allreduce=num_links_allreduce,
        epsilon=epsilon,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        use_hardcoded_expert_index=False,
        noc_mode=noc_mode,
        num_iterations=num_internal_iterations,
        upstream_socket=None,
        downstream_sockets=None,
        fabric_config=device_params["fabric_config"],
        persistent_next_iter_semaphore=persistent_next_iter_semaphore,
        persistent_mode=False,
    )
    for i in range(num_iters):
        moe_final_output_tensor, attention_block_output_tensor = DecoderBlock.execute(*decoder_program_context)
    ttnn.synchronize_device(submesh)

    # ========================================================================
    # Extract decoder MLP output from reduce root
    # ========================================================================
    root_coord_tuple = d["reduce_root_coord"]
    root_device_idx = root_coord_tuple[0] * mesh_cols + root_coord_tuple[1]
    root_device_tensor = ttnn.get_device_tensors(moe_final_output_tensor)[root_device_idx]
    decoder_mlp_output = ttnn.to_torch(root_device_tensor)
    decoder_mlp_output_valid = extract_routed_expert_output(
        decoder_mlp_output.unsqueeze(0),
        d["num_gate_proj_cores"],
        RoutedExpert.FINAL_OUTPUT_WIDTH_PER_CORE,
        d["per_core_down_proj_N"],
    )

    # ========================================================================
    # Compute golden reference via DecoderBlock.golden (MLA → MLP full pipeline)
    # ========================================================================
    logger.info("Computing golden reference...")

    QNOPE_HEAD_DIM = 128
    QROPE_HEAD_DIM = 64
    KNOPE_DIM = 512
    KROPE_DIM = 64
    HEADS_PER_ROW = 8

    _full_q, golden_new_kv, _mla_output, _scores, _indices, moe_output = DecoderBlock.golden(
        d["golden_torch_input"],
        d["golden_torch_gamma"],
        d["golden_torch_matmul_weights"],
        d["golden_torch_rmsnorm2_gamma"],
        d["golden_torch_matmul2_weights"],
        d["golden_torch_matmul3_weights"],
        d["golden_torch_sin"],
        d["golden_torch_cos"],
        d["golden_metadata"],
        d["golden_torch_dkv_matmul_weights"],
        d["golden_torch_dkv_rmsnorm_gamma"],
        d["golden_torch_kv_cache"],
        d["golden_scale"],
        d["golden_torch_kv_b2_proj_weights"],
        d["golden_torch_o_proj_weights"],
        epsilon=epsilon,
        num_qnope_heads=d["golden_total_qnope_heads"],
        num_qrope_heads=d["golden_total_qrope_heads"],
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
        nope_dim=KNOPE_DIM,
        rope_dim=KROPE_DIM,
        moe_shared_gate_weights=d["golden_moe_shared_gate"],
        moe_shared_up_weights=d["golden_moe_shared_up"],
        moe_shared_down_weights=d["golden_moe_shared_down"],
        moe_gate_proj_weights_dict=d["golden_moe_gate_proj_dict"],
        moe_up_proj_weights_dict=d["golden_moe_up_proj_dict"],
        moe_down_proj_weights_dict=d["golden_moe_down_proj_dict"],
        moe_rmsnorm_gamma=d["golden_moe_rmsnorm_gamma"],
        moe_rmsnorm_epsilon=epsilon,
        moe_enable_routing=False,
    )

    # ========================================================================
    # Validate KV cache outputs (per SP device)
    # ========================================================================
    device_chunk_size = d["device_chunk_size"]
    num_sp = mesh_rows
    owning_sp_device = (position_id // device_chunk_size) % num_sp

    kv_cache_output_torch_flat = ttnn.to_torch(
        d["ttnn_kv_cache"], mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
    )
    kv_cache_output_torch = kv_cache_output_torch_flat.reshape(
        num_devices, num_slots, 1, d["per_device_max_seq_len"], -1
    )

    def get_local_seq_len(sp_idx):
        sp_block = device_chunk_size * num_sp
        num_full_blocks = position_id // sp_block
        remainder = position_id % sp_block
        dev_start = sp_idx * device_chunk_size
        dev_end = dev_start + device_chunk_size
        dev_contrib = max(0, min(remainder, dev_end) - dev_start)
        return num_full_blocks * device_chunk_size + dev_contrib

    for device_idx in range(num_devices):
        sp_group = device_idx // mesh_cols
        local_seq_len = get_local_seq_len(sp_group)

        if local_seq_len == 0 and sp_group != owning_sp_device:
            logger.info(f"Device {device_idx} (SP={sp_group}) no data yet, skipped")
            continue

        assert torch.equal(
            d["golden_kv_cache_bfp8_before_op"][device_idx, slot_id, ..., :local_seq_len, :],
            kv_cache_output_torch[device_idx, slot_id, ..., :local_seq_len, :],
        ), f"Device {device_idx} (SP={sp_group}) KV Cache before and after op mismatch"
        logger.info(f"Device {device_idx} (SP={sp_group}) old cache validation passed")

        if sp_group == owning_sp_device:
            compare_kv_cache = kv_cache_output_torch[device_idx, slot_id, ..., local_seq_len, :]
            expected_nope = golden_new_kv[..., :KNOPE_DIM]
            expected_rope = golden_new_kv[..., KNOPE_DIM:]
            compare_nope = compare_kv_cache[..., :KNOPE_DIM]
            compare_rope = compare_kv_cache[..., KNOPE_DIM:]

            nope_passing, nope_pcc = comp_pcc(compare_nope, expected_nope, 0.98)
            logger.info(f"Device {device_idx} (SP={sp_group}) KV Cache NOPE PCC: {nope_pcc}")
            assert nope_passing, f"Device {device_idx} (SP={sp_group}) KV Cache NOPE PCC check failed: {nope_pcc}"

            rope_passing, rope_pcc = comp_pcc(compare_rope, expected_rope, 0.98)
            logger.info(f"Device {device_idx} (SP={sp_group}) KV Cache ROPE PCC: {rope_pcc}")
            assert rope_passing, f"Device {device_idx} (SP={sp_group}) KV Cache ROPE PCC check failed: {rope_pcc}"

        for other_slot in range(num_slots):
            if other_slot == slot_id:
                continue
            assert torch.equal(
                d["golden_kv_cache_bfp8_before_op"][device_idx, other_slot],
                kv_cache_output_torch[device_idx, other_slot],
            ), f"Device {device_idx} (SP={sp_group}) KV Cache slot {other_slot} was modified but should be untouched"
        if num_slots > 1:
            logger.info(f"Device {device_idx} (SP={sp_group}) other slots unchanged validation passed")

    # ========================================================================
    # Validate MLP output vs DecoderBlock golden
    # ========================================================================
    passing, pcc = comp_pcc(moe_output.flatten(), decoder_mlp_output_valid.float().flatten(), 0.975)
    logger.info(f"MLP PCC (decoder vs golden): {pcc}")
    logger.info(f"Golden MLP output: {moe_output.flatten()[:8]}")
    logger.info(f"DecoderBlock MLP output: {decoder_mlp_output_valid.flatten()[:8]}")
    assert passing, f"DecoderBlock MLP Output PCC check failed: {pcc}"

    logger.info("✓ DecoderBlock MLP mesh test passed!")

    ttnn.synchronize_device(submesh)
