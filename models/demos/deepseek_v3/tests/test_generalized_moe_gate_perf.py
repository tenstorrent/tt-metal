# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-side perf test for the fused ``generalized_moe_gate`` op (global top-8 over 256 experts).

Captures a trace of repeated ``GeneralizedMoeGateOp.op`` calls, bracketed by signposts, so the perf
harness (``perf_generalized_moe_gate.py``) can isolate the device-kernel time. Tensor setup mirrors
``test_generalized_moe_gate.py`` exactly (sharded 32x32 tile, bfloat16). 256 experts, sigmoid ON.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3.tt.generalized_moe_gate.op import GeneralizedMoeGateOp


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 7000000}],
    indirect=True,
)
@pytest.mark.parametrize("enable_sigmoid", [True])
@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
def test_generalized_moe_gate_perf(device, enable_sigmoid, warmup_iters, num_iters, device_params):
    """Capture and execute a trace of the gate op for perf measurement (256 experts)."""
    batch_size = 1
    input_shape = (batch_size, 8, 32)  # 256 experts
    reshaped_input_shape = (batch_size, 16, 16)
    input_shard_shape = (32, 32)
    input_tile = ttnn.Tile(input_shard_shape)
    output_shape = (batch_size, 1, 16)
    output_shard_shape = (32, 32)
    output_tile = ttnn.Tile(output_shard_shape)
    eps = 1e-20
    scaling_factor = 2.5

    torch.manual_seed(42)
    torch_input = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1
    if not enable_sigmoid:
        torch_input = torch.sigmoid(torch_input)
    torch_bias = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    def upload(t, dtype):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mem_config, tile=input_tile
        )

    ttnn_input = upload(torch.reshape(torch_input, reshaped_input_shape), ttnn.bfloat16)
    ttnn_bias = upload(torch.transpose(torch.reshape(torch_bias, reshaped_input_shape), -2, -1), ttnn.bfloat16)

    torch_idx = torch.arange(reshaped_input_shape[1] * reshaped_input_shape[2], dtype=torch.int32)
    torch_idx = torch_idx.unsqueeze(0).expand(reshaped_input_shape[0], -1).reshape(reshaped_input_shape)
    torch_idx = torch.transpose(torch_idx, -2, -1).to(torch.uint16)
    ttnn_input_indices = upload(torch_idx, ttnn.uint16)

    ttnn_output = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    def run_gate():
        return GeneralizedMoeGateOp.op(
            ttnn_input,
            ttnn_bias,
            ttnn_output,
            ttnn_input_indices,
            ttnn_output_indices,
            eps,
            scaling_factor,
            enable_sigmoid,
        )

    # Compile the op
    run_gate()
    ttnn.synchronize_device(device)

    # Capture warmup trace
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(warmup_iters):
        run_gate()
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace (256 experts, sigmoid={enable_sigmoid}, {num_iters} iters)")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(num_iters):
        run_gate()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)

    # Execute main trace, bracketed by signposts for the perf harness
    logger.info("Executing main trace")
    signpost("start")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    signpost("stop")

    # One more eager call + light sanity (indices in range)
    res, res_idx = run_gate()
    ttnn.synchronize_device(device)
    idx = ttnn.to_torch(res_idx)[:, 0, :8]
    assert int(idx.to(torch.int32).min()) >= 0 and int(idx.to(torch.int32).max()) < 256, f"indices out of range: {idx}"

    ttnn.deallocate(ttnn_input)


if __name__ == "__main__":
    pytest.main([__file__])
