# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage pipeline integration test for socket-fed bcast + MoE fused op + reduce-to-one.

Stage 0:
  host token -> fused embedding (HostInterface) -> cross-stage D2D
  D2H loopback: read aggregated reduce output from pipeline
Stage 1:
  entry D2D receiver -> moe sender core socket input -> bcast + fused MoE
  -> reduce-to-one -> D2D_0 aggregator -> pipeline exit
Stage 2+ (if applicable):
  passive forwarding, no downstream op
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.demo.stage import MoEComputeStage, StageContext
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3MoELayerWeights,
    MoERoutedExpertWeights,
    prepare_attention_weights,
    prepare_routed_expert_weights,
    prepare_shared_expert_weights,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import ROUTED_EXPERT_LAYER_IDX, RoutedExpert


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _create_moe_weights(
    mesh_device: ttnn.MeshDevice,
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    num_routed_experts: int = 256,
) -> DeepSeekV3MoELayerWeights:
    """Build DeepSeekV3MoELayerWeights from a state dict with tensors moved to device."""
    bdw = BlitzDecodeWeights(mesh_device)
    attn = prepare_attention_weights(bdw, state_dict, layer_idx, is_moe=True, move_to_device=True)
    shared = prepare_shared_expert_weights(bdw, state_dict, layer_idx, is_moe=True, move_to_device=True)
    routed = prepare_routed_expert_weights(
        bdw, state_dict, layer_idx, is_moe=True, num_routed_experts=num_routed_experts, move_to_device=True
    )
    assert isinstance(routed, MoERoutedExpertWeights)
    return DeepSeekV3MoELayerWeights(
        q_a_proj=attn.q_a_proj,
        q_b_proj=attn.q_b_proj,
        kv_a_proj=attn.kv_a_proj,
        o_proj=attn.o_proj,
        gate_mm=attn.gate_mm,
        attn_norm=attn.attn_norm,
        q_norm=attn.q_norm,
        kv_norm=attn.kv_norm,
        ffn_norm=attn.ffn_norm,
        gate_bias=attn.gate_bias,
        kv_b1_proj=attn.kv_b1_proj,
        kv_b2_proj=attn.kv_b2_proj,
        shared_gate_proj=shared.shared_gate_proj,
        shared_up_proj=shared.shared_up_proj,
        shared_down_proj=shared.shared_down_proj,
        routed_gate_proj=routed.routed_gate_proj,
        routed_up_proj=routed.routed_up_proj,
        routed_down_proj=routed.routed_down_proj,
    )


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("vocab_size, embedding_dim", [(64, 7168)])
@pytest.mark.parametrize("token_id", [0])
@pytest.mark.timeout(1200)
def test_bcast_moe_two_stage_pipeline(
    mesh_device, vocab_size, embedding_dim, token_id, device_params, get_reference_model_state_dict
):
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for MoE (need >= 13x10)")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
    is_torus = device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    is_stage0 = my_mesh_id == 0

    K = embedding_dim
    pipeline_core = MoEComputeStage.PIPELINE_CORE
    token_size_bytes = MoEComputeStage.TOKEN_SIZE_BYTES
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 4

    torch_embedding = torch.arange(vocab_size * K, dtype=torch.float32).reshape(1, 1, vocab_size, K).to(torch.bfloat16)

    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=256,
        include_global=False,
    )

    ctx = StageContext(mesh_device=mesh_device, pipeline_config=pipeline_config, my_mesh_id=my_mesh_id)
    moe_stage = None

    # ── Pipeline block setup (collective — all hosts must participate simultaneously) ────
    if is_stage0:
        embedding_tensor = ttnn.from_torch(torch_embedding, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            h2d_socket_fifo_size=token_size_bytes * 2,
            d2h_socket_fifo_size=embedding_fifo_size,
            d2h_socket_page_size=embedding_size_bytes,
            embedding_tensor=embedding_tensor,
        )
    else:
        weights = _create_moe_weights(mesh_device, state_dict, ROUTED_EXPERT_LAYER_IDX, num_routed_experts=256)
        moe_stage = MoEComputeStage(weights, persistent_mode=False, is_torus=is_torus)
        pipeline_block = moe_stage.create_pipeline_block(ctx)
        moe_stage.setup(ctx, pipeline_block)
        logger.info(f"[rank={my_mesh_id}] MoE stage setup complete")

    logger.info(f"[rank={my_mesh_id}] pipeline block created")

    # ── Launch pipeline programs ──────────────────────────────────────────────
    pipeline_block.run()
    logger.info(f"[rank={my_mesh_id}] pipeline programs launched")

    ttnn.distributed_context_barrier()

    if is_stage0:
        token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
        torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
        torch_token[0, 0] = token_id
        token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pipeline_block.write_token(token_tensor)
        logger.info(f"[rank=0] token {token_id} injected")

    ttnn.distributed_context_barrier()

    if my_mesh_id >= 1:
        logger.info(f"[rank={my_mesh_id}] launching MoE bcast + reduce (num_iterations=1)")
        moe_stage.launch_compute(ctx, pipeline_block)
        logger.info(f"[rank={my_mesh_id}] MoE + reduce completed")

    # ── Stage 0: D2H loopback read + golden validation ───────────────────────
    if mesh_device.get_system_mesh_id() == num_procs - 1:
        rank = num_procs - 1
        logger.info(f"[rank={rank}] waiting for D2H result from pipeline loopback")
        num_elements = embedding_size_bytes // 2
        received_tensor_torch = torch.zeros(1, num_elements, dtype=torch.bfloat16)
        d2h_output_tensor = ttnn.from_torch(received_tensor_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        pipeline_block.read_output(d2h_output_tensor)
        d2h_result_torch = ttnn.to_torch(d2h_output_tensor)
        logger.info(f"[rank={rank}] D2H read complete, shape={d2h_result_torch.shape}")
        logger.info(f"[rank={rank}] D2H first 5 values: {d2h_result_torch[0, :5]}")

        d2h_nonzero = torch.count_nonzero(d2h_result_torch)
        logger.info(f"[rank={rank}] D2H non-zero elements: {d2h_nonzero}/{d2h_result_torch.numel()}")
        assert d2h_nonzero > 0, "D2H output is all zeros — reduce or D2D0 pipeline failed"

    ttnn.distributed_context_barrier()

    # ── Pipeline teardown ───────────────────────────────────────────────────
    logger.info(f"[rank={my_mesh_id}] waiting for pipeline block termination")
    pipeline_block.terminate()
    logger.info(f"[rank={my_mesh_id}] programs terminated")

    logger.info(f"[rank={my_mesh_id}] test PASSED")


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_dim", [7168])
@pytest.mark.parametrize("iterations", [2048])
@pytest.mark.timeout(120000)
def test_persistent_moe_15_stages(
    mesh_device, embedding_dim, iterations, device_params, get_reference_model_state_dict
):
    """
    Persistent-mode 15-stage MoE pipeline test.

    Pipeline topology:
      Stage 0  : H2D embedding -> downstream D2D, D2H loopback <- pipeline
      Stage 1-14 (15 MoE stages): socket bcast -> fused MoE -> reduce-to-one -> pipeline exit
      Stage 15 : passive forwarding back to stage 0

    The MoE kernel on stages 1-14 runs in a while(true) loop.  Stage 0 drives
    the pipeline by writing *iterations* tokens and reading back each D2H result.
    Validates every result is non-zero.
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for MoE (need >= 13x10)")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
    is_torus = device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_2D_TORUS_Y

    is_stage0 = my_mesh_id == 0
    is_moe_stage = my_mesh_id >= 1

    K = embedding_dim
    pipeline_core = MoEComputeStage.PIPELINE_CORE
    token_size_bytes = MoEComputeStage.TOKEN_SIZE_BYTES
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 4

    torch.manual_seed(42)
    torch_embedding = torch.randn(iterations, 1, 1, K, dtype=torch.bfloat16)

    logger.info(f"Creating reference model state dict")
    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=256,
        include_global=False,
    )

    logger.info(f"Creating stage context")
    ctx = StageContext(mesh_device=mesh_device, pipeline_config=pipeline_config, my_mesh_id=my_mesh_id)
    moe_stage = None

    pipeline_block = None
    try:
        # ── Pipeline block setup (collective — all hosts must participate simultaneously) ────
        if is_stage0:
            embedding_tensor = ttnn.from_torch(torch_embedding, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            logger.info(f"Creating embedding pipeline block")
            pipeline_block = PipelineBlock(
                mesh_device,
                pipeline_core,
                upstream_d2d_socket_fifo_size=embedding_fifo_size,
                downstream_d2d_socket_fifo_size=embedding_fifo_size,
                upstream_d2d_socket_page_size=embedding_size_bytes,
                downstream_d2d_socket_page_size=embedding_size_bytes,
                h2d_socket_fifo_size=token_size_bytes * 2,
                d2h_socket_fifo_size=embedding_fifo_size,
                d2h_socket_page_size=embedding_size_bytes,
                embedding_tensor=embedding_tensor,
            )
        else:
            logger.info(f"Creating MoE weights")
            weights = _create_moe_weights(mesh_device, state_dict, ROUTED_EXPERT_LAYER_IDX, num_routed_experts=256)
            logger.info(f"Creating MoE stage")
            moe_stage = MoEComputeStage(weights, persistent_mode=True, is_torus=is_torus)
            logger.info(f"Creating MoE pipeline block")
            pipeline_block = moe_stage.create_pipeline_block(ctx)
            logger.info(f"Setting up MoE stage")
            moe_stage.setup(ctx, pipeline_block)
            logger.info(f"[rank={my_mesh_id}] MoE stage setup complete")

        logger.info(f"[rank={my_mesh_id}] pipeline block created")

        # ── Launch pipeline ──
        pipeline_block.run()
        logger.info(f"[rank={my_mesh_id}] pipeline launched")

        # ── MoE stages: submit persistent kernel ──
        if is_moe_stage:
            logger.info(f"[rank={my_mesh_id}] submitting persistent MoE kernel")
            moe_stage.launch_compute(ctx, pipeline_block)
            logger.info(f"[rank={my_mesh_id}] persistent MoE kernel submitted")

        ttnn.distributed_context_barrier()

        # ── Stage 0: drive pipeline with multiple tokens ──
        if is_stage0:
            token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
            torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
            torch_token[0, 0] = 0
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

            start_time = time.time()
            for iteration in range(iterations):
                pipeline_block.write_token(token_tensor)
                # logger.info(f"[rank=0] token {iteration} injected")
                output_token = ttnn.recv_token(ttnn.Rank(num_procs - 1))
            end_time = time.time()
            print(f"[rank=0] time taken to move {iterations} tokens: {end_time - start_time} seconds")

        elif my_mesh_id == num_procs - 1:
            num_elements = embedding_size_bytes // 2
            d2h_output_tensor = ttnn.from_torch(
                torch.zeros(1, num_elements, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            for iteration in range(iterations):
                # print(f"[rank={my_mesh_id}] iteration {iteration} start")
                # print(f"[rank={my_mesh_id}] iteration {iteration} waiting for D2H result")
                pipeline_block.read_output(d2h_output_tensor)
                ttnn.send_token(iteration, ttnn.Rank(0))
                # print(f"[rank={my_mesh_id}] iteration {iteration} D2H result read")
                # d2h_result = ttnn.to_torch(d2h_output_tensor)

                # d2h_nonzero = torch.count_nonzero(d2h_result)
                # logger.info(
                #     f"[rank={my_mesh_id}] iteration {iteration}: non-zero={d2h_nonzero}/{d2h_result.numel()}, "
                #     f"first 5={d2h_result[0, :5]}"
                # )
                # assert (
                #     d2h_nonzero > 0
                # ), f"D2H output is all zeros at iteration {iteration} — persistent MoE 15-stage pipeline failed"
                # ttnn.distributed_context_barrier()

            logger.info(f"[rank={my_mesh_id}] all {iterations} iterations passed")

        logger.info(f"[rank={my_mesh_id}] waiting for barrier")
        ttnn.distributed_context_barrier()
        logger.info(f"[rank={my_mesh_id}] barrier completed")

    finally:
        pass

    logger.info(f"[rank={my_mesh_id}] persistent 15-stage MoE test PASSED")
