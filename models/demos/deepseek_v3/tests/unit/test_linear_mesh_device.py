# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "op_name, mode, input_shape, weight_shape, input_memory_config, weight_memory_config, output_memory_config",
    [
        (
            "test_on_dram",
            "decode",
            [1, 1, 32, 7168],  # Input shape
            [1, 1, 7168, 256],  # Weight shape
            "dram",
            "dram",
            "dram",
        ),
        (
            "test_on_l1",
            "decode",
            [1, 1, 32, 7168],  # Input shape
            [1, 1, 7168, 256],  # Weight shape
            "L1",
            "L1",
            "L1",
        ),
        (
            "test_on_l1_dram",
            "decode",
            [1, 1, 32, 7168],  # Input shape
            [1, 1, 7168, 256],  # Weight shape
            "L1",
            "DRAM",
            "L1",
        ),
        (
            "test_on_dram",
            "prefill",
            [1, 1, 128, 7168],  # Input shape
            [1, 1, 7168, 256],  # Weight shape
            "dram",
            "dram",
            "dram",
        ),
        (
            "test_on_l1",
            "prefill",
            [1, 1, 128, 7168],  # Input shape
            [1, 1, 7168, 256],  # Weight shape
            "L1",
            "L1",
            "L1",
        ),
        (
            "test_on_l1_dram",
            "prefill",
            [1, 1, 128, 7168],  # Input shape
            [1, 1, 7168, 256],  # Weight shape
            "L1",
            "DRAM",
            "L1",
        ),
    ],
    ids=["dram_dram_decode", "l1_l1_decode", "l1_dram_decode", "dram_dram_prefill", "l1_l1_prefill", "l1_dram_prefill"],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {"trace_region_size": 7000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
def test_deepseek_v3_moe_gate_linear_trace_mode(
    mesh_device,
    op_name,
    mode,
    input_shape,
    weight_shape,
    input_memory_config,
    weight_memory_config,
    output_memory_config,
    warmup_iters,
    num_iters,
):
    """
    Test all decode linear operations from MoEGate with trace mode.

    Linear operations tested:
    1. MoEGate proj on DRAM
    2. MoEGate proj on L1

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    """
    torch.manual_seed(0)

    # Create random tensors for golden reference
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_weight_tensor = torch.randn(weight_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor @ torch_weight_tensor
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG if input_memory_config == "dram" else ttnn.L1_MEMORY_CONFIG
    weight_memory_config = ttnn.DRAM_MEMORY_CONFIG if weight_memory_config == "dram" else ttnn.L1_MEMORY_CONFIG
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG if output_memory_config == "dram" else ttnn.L1_MEMORY_CONFIG

    # Create ttnn tensors
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_memory_config,
    )

    grid = mesh_device.compute_with_storage_grid_size()
    linear_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 10),
        in0_block_w=32,
        out_subblock_h=1,
        out_subblock_w=2,
        out_block_h=1,
        out_block_w=2,
        per_core_M=1,
        per_core_N=2,
        transpose_mcast=False,
        fused_activation=None,
    )
    linear_program_config = None

    # Compile run
    logger.info(f"Compiling linear operation: {op_name}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Weight shape: {weight_shape}")
    logger.info(f"  Output shape: {list(torch_output_tensor.shape)}")

    tt_output_tensor = ttnn.linear(
        tt_input_tensor,
        tt_weight_tensor,
        memory_config=output_memory_config,
        dtype=ttnn.bfloat16,
        program_config=linear_program_config,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.linear(
            tt_input_tensor,
            tt_weight_tensor,
            memory_config=output_memory_config,
            dtype=ttnn.bfloat16,
            program_config=linear_program_config,
        )
    ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.linear(
            tt_input_tensor,
            tt_weight_tensor,
            memory_config=output_memory_config,
            dtype=ttnn.bfloat16,
            program_config=linear_program_config,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler = BenchmarkProfiler()
    profiler.start("warmup")
    ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_id_warmup)
    profiler.end("warmup")
    ttnn.synchronize_device(mesh_device)

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("main")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    profiler.end("main")
    signpost("stop")
    ttnn.synchronize_device(mesh_device)

    # Verify correctness
    seq_len = 32 if mode == "decode" else 128
    torch_output_from_tt = ttnn.to_torch(
        tt_output_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0, :, :seq_len, :].unsqueeze(0)

    assert torch_output_from_tt.shape == torch_output_tensor.shape
    pcc_passed, pcc_output = comp_pcc(torch_output_tensor, torch_output_from_tt, 0.99)

    logger.info(f"✓ Trace mode {op_name} test passed with correct output with pcc {pcc_output}")
    assert pcc_passed, f"Trace mode {op_name} test failed with pcc {pcc_output}"
