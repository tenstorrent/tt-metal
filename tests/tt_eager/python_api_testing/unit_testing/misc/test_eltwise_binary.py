# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc
from models.utility_functions import is_grayskull, run_for_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "bfloat16",
    ],
)
@pytest.mark.parametrize(
    "test_func_name, torch_func_name",
    [(ttnn.add, torch.add), (ttnn.sub, torch.sub), (ttnn.mul, torch.mul)],
)
@pytest.mark.parametrize(
    "pre_in0_silu",
    [True, False],
    ids=["silu", "no-silu"],
)
@pytest.mark.parametrize(
    "shard",
    [False, True],
    ids=["interleaved", "sharded"],
)
def test_run_elt_binary(dtype, test_func_name, torch_func_name, pre_in0_silu, device, shard):
    shape = (1, 1, 32, 1024)

    torch.manual_seed(10)

    if shard:
        mem_config = ttnn.create_sharded_memory_config(
            shape=shape,
            core_grid=ttnn.CoreGrid(y=1, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    in0 = torch.randn(shape).bfloat16().float()
    in1 = torch.randn(shape).bfloat16().float()
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=mem_config, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=mem_config, tt_dtype=dtype)

    if pre_in0_silu:
        torch_silu = torch.nn.SiLU()
        out_t = test_func_name(
            in0_t, in1_t, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mem_config
        )
    else:
        out_t = test_func_name(in0_t, in1_t)
    out = tt2torch_tensor(out_t)

    if pre_in0_silu:
        passing, output = comp_pcc(out, torch_func_name(torch_silu(in0), in1), 0.9999)
    else:
        passing, output = comp_pcc(out, torch_func_name(in0, in1), 0.9999)
    logger.info(output)
    assert passing


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_run_elt_binary_add_with_sub_devices(device):
    unharvested_grid_size = (7, 10)
    compute_grid_size = device.compute_with_storage_grid_size()
    if unharvested_grid_size[0] > compute_grid_size.x or unharvested_grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {unharvested_grid_size} grid size to run this test but core grid is {compute_grid_size}")

    shape = (1, 1, 32, 2048)
    torch.manual_seed(10)

    start_core = ttnn.CoreCoord(1, 0)
    core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(start_core, 32, core_grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    in0 = torch.randn(shape).bfloat16().float()
    in1 = torch.randn(shape).bfloat16().float()
    in0_t = ttnn.from_torch(
        in0,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    in1_t = ttnn.from_torch(
        in0,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_mem_config,
    )
    in1 = ttnn.to_torch(in1_t)

    out_t = ttnn.add(in0_t, in1_t, dtype=ttnn.bfloat16, memory_config=output_mem_config)

    out = ttnn.to_torch(out_t)

    passing, output = comp_pcc(out, torch.add(in0, in1), 0.9999)
    logger.info(output)
    assert passing


def run_elt_binary_mul_with_sub_devices(
    batch, num_heads, seq_len, dim, dtype, in_mem_config, out_mem_config, device, mesh_mapper, mesh_composer, pcc
):
    input_shape = [batch, num_heads, seq_len, dim]
    in0 = torch.randn(input_shape)
    in1 = torch.randn(input_shape)
    in0_ref = in0.clone()
    in1_ref = in1.clone()
    in0_t = ttnn.from_torch(
        in0,
        device=device,
        mesh_mapper=mesh_mapper,
        dtype=dtype,
        memory_config=in_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )
    in1_t = ttnn.from_torch(
        in1,
        device=device,
        mesh_mapper=mesh_mapper,
        dtype=dtype,
        memory_config=in_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )
    out_t = ttnn.mul(
        in0_t,
        in1_t,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat8_b,
        memory_config=out_mem_config,
    )
    tt_out = ttnn.to_torch(out_t, mesh_composer=mesh_composer)
    torch_silu = torch.nn.SiLU()
    passing, output = comp_pcc(tt_out, torch.mul(torch_silu(in0_ref), in1_ref), pcc)
    logger.info(output)
    assert passing


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_run_elt_binary_mul_with_sub_devices(device):
    unharvested_grid_size = (7, 10)
    compute_grid_size = device.compute_with_storage_grid_size()
    if unharvested_grid_size[0] > compute_grid_size.x or unharvested_grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {unharvested_grid_size} grid size to run this test but core grid is {compute_grid_size}")

    pcc = 0.999
    shape = (1, 1, 32, 896)
    dtype = ttnn.bfloat8_b
    mesh_mapper = None
    mesh_composer = None
    torch.manual_seed(10)
    start_core = ttnn.CoreCoord(1, 0)
    core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(start_core, 28, core_grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
    in_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    out_mem_config = in_mem_config
    run_elt_binary_mul_with_sub_devices(
        *shape, dtype, in_mem_config, out_mem_config, device, mesh_mapper, mesh_composer, pcc
    )
