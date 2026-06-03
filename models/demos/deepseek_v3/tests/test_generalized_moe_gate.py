# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone unit test for the C++ op ``ttnn.experimental.deepseek.moe.generalized_moe_gate``.

Exercises the device op directly (via ``GeneralizedMoeGateOp.op``) against the PyTorch
``golden`` reference, so the op can be validated in isolation without running the full
``MoEGate`` module. Modeled on
``models/demos/deepseek_v3_b1/tests/unit_tests/test_deepseek_moe_gate.py``.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.generalized_moe_gate.op import GeneralizedMoeGateOp


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("enable_sigmoid", [True, False])
@pytest.mark.parametrize("seed", [42, 201, 512])
def test_generalized_moe_gate(device, batch_size, enable_sigmoid, seed):
    """Test the generalized MoE gate C++ op on a 32x32 tile against the golden reference."""

    # Tensor dimensions — full 32x32 tile, logical 32x32 per shard.
    input_shape = (batch_size, 8, 32)
    reshaped_input_shape = (batch_size, 16, 16)
    input_shard_shape = (32, 32)
    input_tile = ttnn.Tile(input_shard_shape)
    output_shape = (batch_size, 1, 16)
    output_shard_shape = (32, 32)
    output_tile = ttnn.Tile(output_shard_shape)

    logger.info(f"Testing generalized MoE gate with input shape {input_shape}")

    # Create input PyTorch tensor with random values.
    torch.manual_seed(seed)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    if not enable_sigmoid:
        torch_input = torch.sigmoid(torch_input)
    torch_bias = torch.randn(input_shape, dtype=torch.bfloat16)
    eps = 1e-20
    scaling_factor = 2.5

    # Reference output.
    top8_scores, top8_indices = GeneralizedMoeGateOp.golden(
        torch_input, torch_bias, eps, scaling_factor, enable_sigmoid
    )

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(
        batch_size,
        ttnn.CoreCoord(grid.x, grid.y),
        row_wise=True,
    )
    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Input values — sharded on a single core per batch.
    reshaped_input = torch.reshape(torch_input, reshaped_input_shape)
    ttnn_input = ttnn.from_torch(
        reshaped_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Bias is transposed before upload (the kernel expects the transposed layout).
    reshaped_bias = torch.transpose(torch.reshape(torch_bias, reshaped_input_shape), -2, -1)
    ttnn_bias = ttnn.from_torch(
        reshaped_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Transposed routing indices: 0..255 laid out as (16,16) then transposed.
    torch_input_indices = torch.arange(reshaped_input_shape[1] * reshaped_input_shape[2], dtype=torch.int32)
    torch_input_indices = torch_input_indices.unsqueeze(0).expand(reshaped_input_shape[0], -1)
    torch_input_indices = torch_input_indices.reshape(reshaped_input_shape)
    torch_input_indices = torch.transpose(torch_input_indices, -2, -1).to(torch.uint16)
    ttnn_input_indices = ttnn.from_torch(
        torch_input_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Preallocated output buffers (filled in place by the op).
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

    logger.info("Running generalized MoE gate operation...")
    ttnn_result, ttnn_result_indices = GeneralizedMoeGateOp.op(
        ttnn_input,
        ttnn_bias,
        ttnn_output,
        ttnn_input_indices,
        ttnn_output_indices,
        eps,
        scaling_factor,
        enable_sigmoid,
    )

    # Convert back to torch and keep the top-8 slots.
    output_torch = ttnn.to_torch(ttnn_result)[:, 0, :8]
    output_indices_torch = ttnn.to_torch(ttnn_result_indices)[:, 0, :8]

    # The op does not guarantee a stable order across ties, so sort both by index
    # before comparing (same approach as the reference unit test).
    sorted_output_indices_torch, i = torch.sort(output_indices_torch, dim=-1)
    sorted_output_torch = torch.gather(output_torch, dim=-1, index=i)

    top8_indices, i = torch.sort(top8_indices, dim=-1)
    top8_scores = torch.gather(top8_scores, dim=-1, index=i)
    breakpoint()

    assert torch.equal(
        sorted_output_indices_torch.to(top8_indices.dtype), top8_indices
    ), "Output indices do not match golden"
    assert torch.allclose(sorted_output_torch, top8_scores, atol=1e-2, rtol=1e-4), "Output scores do not match golden"


@pytest.mark.parametrize("batch_size", [1])
def test_dump_sum_top2_layout(device, batch_size):
    """DEBUG PROBE: dump the per-group top-8 DEST layout right after sum_top2.

    Requires the kernel built with ``GMG_DUMP_AFTER_SUM_TOP2`` enabled (see
    generalized_moe_gate_kernel.cpp). With that macro on, the op stops after sum_top2 and
    returns the *bias region* (ranking keys) in the score output and the *indices region*
    (expert ids) in the index output, both as a full 16x16 face.

    Inputs are rigged so the layout is readable:
      - input = 0.5 everywhere (scores irrelevant to layout)
      - bias[g, j] = j  -> each group's top-8 are its experts j in {24..31}
      - indices = global arange(256) (so each cell's value = global expert id; group = id // 32)

    Run it, then read the printed grids: in the INDEX grid, find where the ids of each
    group (id // 32 == g) land -> that's group g's top-8 column slot / lane layout.
    """
    n_group, group_size = 8, 32  # 256 experts
    reshaped = (batch_size, 16, 16)
    shard_shape = (32, 32)
    tile = ttnn.Tile(shard_shape)

    torch_input = torch.full((batch_size, n_group, group_size), 0.5, dtype=torch.bfloat16)
    # bias[g, j] = j + g  -> within every group the top-8 is still j in {24..31}, but the
    # per-group top-2 sum (= 61 + 2g) is distinct, so sort_top4 deterministically keeps
    # groups {4,5,6,7}. That makes the AFTER_STEP1 dump readable (you can tell which group
    # is which from the expert ids), while AFTER_SUM_TOP2 still shows all 8 groups.
    j = torch.arange(group_size, dtype=torch.bfloat16).view(1, 1, group_size)
    g = torch.arange(n_group, dtype=torch.bfloat16).view(1, n_group, 1)
    torch_bias = (j + g).expand(batch_size, n_group, group_size).contiguous()

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    def upload(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg, tile=tile)

    ttnn_input = upload(torch.reshape(torch_input, reshaped), ttnn.bfloat16)
    ttnn_bias = upload(torch.transpose(torch.reshape(torch_bias, reshaped), -2, -1), ttnn.bfloat16)

    idx = torch.arange(n_group * group_size, dtype=torch.int32).view(1, n_group, group_size)
    idx = idx.expand(batch_size, n_group, group_size).reshape(reshaped)
    idx = torch.transpose(idx, -2, -1).to(torch.uint16)
    ttnn_idx = upload(idx, ttnn.uint16)

    # FULL 32x32 output (logical) so the dump shows all 4 faces of the tile, incl. dst rows 16-31
    # (step1 reads offsets 16/28 = the bottom half, which a 16x16 face dump never showed).
    out_shaped = (batch_size, 32, 32)
    ttnn_out = upload(torch.zeros(out_shaped, dtype=torch.bfloat16), ttnn.bfloat16)
    ttnn_out_idx = upload(torch.zeros(out_shaped, dtype=torch.uint16), ttnn.uint16)

    GeneralizedMoeGateOp.op(ttnn_input, ttnn_bias, ttnn_out, ttnn_idx, ttnn_out_idx, 1e-20, 1.0, False)

    keys = ttnn.to_torch(ttnn_out)[0].float()
    ids = ttnn.to_torch(ttnn_out_idx)[0].to(torch.int32)

    torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=200, precision=1, sci_mode=False)
    logger.info(f"\n=== bias region (ranking keys), 16x16 ===\n{keys}")
    logger.info(f"\n=== indices region (expert ids), 16x16 ===\n{ids}")
    logger.info(f"\n=== group id (= expert_id // 32), 16x16 ===\n{ids // group_size}")


if __name__ == "__main__":
    pytest.main([__file__])
