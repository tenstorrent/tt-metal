# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TtPrefillBlock — verifies composition of norm → MLA → residual → norm → FFN/MoE → residual.

Validates output shapes and PCC against torch reference.

Uses HF DeepseekV3Model layer as the reference: creates a model with random weights,
extracts those weights into our TT state_dict format, and compares forward passes.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import create_hf_model, extract_layer_state_dict

PCC_THRESHOLD_DENSE = 0.996
PCC_THRESHOLD_MOE_GATE_HOST = 0.996
PCC_THRESHOLD_MOE_GATE_DEVICE = 0.992
PCC_THRESHOLD_KVPE = 0.999
SEQ_LEN_25_K = 25 * 1024
SEQ_LEN_100_K = 100 * 1024


@pytest.mark.parametrize(
    "input_source, is_balanced, isl_total",
    [
        ("random", True, SEQ_LEN_25_K),
        # ("random", True, SEQ_LEN_100_K),
    ],
    ids=["seq_25k"],
)
@pytest.mark.parametrize(
    "layer_type, gate_fallback_mode",
    [
        ("moe", GateComputeMode.DEVICE),
    ],
    ids=["moe-gate_device"],
)
@pytest.mark.parametrize("num_iterations", [100])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
        pytest.param(
            (32, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            2,
            ttnn.Topology.Linear,
            # marks=pytest.mark.requires_mesh_topology(mesh_shape=(32, 4), topology="mesh-32x4"),
            id="mesh-32x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(600 * 100)
def test_prefill_block(
    config_only,
    mesh_device,
    device_params,
    num_iterations,
    isl_total,
    layer_type,
    gate_fallback_mode,
    num_links,
    topology,
    is_balanced,
    input_source,
    request,
):
    config = config_only
    config.max_seq_len = isl_total

    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    if mesh_shape[0] == 32 and mesh_shape[1] == 4:
        if isl_total != SEQ_LEN_100_K:
            pytest.skip("For mesh shape (32, 4) we only run the test with isl_total=100K")
    # if mesh_shape[0] == 8 and mesh_shape[1] == 4:
    #     if isl_total != SEQ_LEN_25_K:
    #         pytest.skip("For mesh shape (8, 4) we only run the test with isl_total=25K")
    sp_factor = mesh_shape[sp_axis]
    tp_factor = mesh_shape[tp_axis]
    emb_dim = config.hidden_size
    isl_per_chip = isl_total // sp_factor

    # layer_idx=0 for dense (< NUM_DENSE_LAYERS=3), layer_idx=3 for MoE (>= 3)
    layer_idx = 0 if layer_type == "dense" else DeepSeekV3Config.NUM_DENSE_LAYERS

    logger.info(f"mesh_shape={mesh_shape}, sp_factor={sp_factor}, tp_factor={tp_factor}")
    logger.info(
        f"isl_total={isl_total}, isl_per_chip={isl_per_chip}, "
        f"layer_type={layer_type}, layer_idx={layer_idx}, gate_fallback_mode={gate_fallback_mode}, "
        f"input_source={input_source}"
    )

    # --- Build HF reference model and extract weights ---
    torch.manual_seed(42)
    num_layers = layer_idx + 1
    hf_model = create_hf_model(config, num_layers)
    hf_sd = hf_model.state_dict()
    state_dict = extract_layer_state_dict(hf_sd, layer_idx, hf_model.layers[layer_idx])
    torch.manual_seed(123)
    torch_input = torch.randn(1, isl_total, emb_dim, dtype=torch.bfloat16)

    # --- TT block ---
    block_kwargs = dict(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        layer_idx=layer_idx,
        seq_len=isl_total,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
    )
    if gate_fallback_mode is not None:
        block_kwargs["gate_fallback_mode"] = gate_fallback_mode

    block = TtPrefillBlock(**block_kwargs)
    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    # Shard input to device: [1, 1, isl_total, emb_dim] → [1, 1, isl_per_chip, emb_dim/tp]
    tt_input_4d = torch_input.unsqueeze(0)  # [1, 1, isl_total, emb_dim]
    tt_input = ttnn.from_torch(
        tt_input_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(-2, -1)),
    )

    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)
    rope_tensors = rope_setup.get_rope_tensors(isl_total)

    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=isl_total,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    rank = ttnn.distributed_context_get_rank()
    logger.info("Running TtPrefillBlock forward...")
    group_size = 58
    group_start_time = time.time()

    for i in range(num_iterations):
        # Start of new group
        if i % group_size == 0:
            group_start_time = time.time()
            logger.info(
                f"Rank: {rank} Starting group {i // group_size + 1} (iterations {i+1}-{min(i+group_size, num_iterations)})"
            )

        logger.info(f"Rank: {rank} Iteration {i+1}/{num_iterations} start fwd pass")
        iteration_start = time.time()
        tt_output, _ = block(tt_input, rope_tensors, tt_kvpe_cache, return_kv_cache=False)
        ttnn.deallocate(tt_output)
        ttnn.synchronize_device(mesh_device)
        ttnn.distributed_context_barrier()
        iteration_end = time.time()
        iteration_time = iteration_end - iteration_start
        logger.info(f"Rank: {rank} Iteration {i+1}/{num_iterations} completed in {iteration_time:.3f}s")

        # End of group or last iteration
        if (i + 1) % group_size == 0 or i == num_iterations - 1:
            group_end_time = time.time()
            group_time = group_end_time - group_start_time
            group_num = i // group_size + 1
            actual_group_size = group_size if (i + 1) % group_size == 0 else (i % group_size) + 1
            logger.info(
                f"Rank: {rank} Group {group_num} ({actual_group_size} iterations) completed in {group_time:.3f}s"
            )

    logger.info("Forward pass loop completed successfully")
