#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor, run_test
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape, shard_type, cores, out_mem_config",
    [
        # kv_nope -> L1 interleaved (mla1d.py _fwd_decode_norm_and_rope): width sharded 2x8 [32,32]
        ([1, 1, 32, 512], "W", (2, 8), ttnn.L1_MEMORY_CONFIG),
        # q_rope -> L1 interleaved (mla1d.py _fwd_decode_q_rope_nope): height sharded 4x8 [32,64]
        ([1, 32, 16, 64], "H", (4, 8), ttnn.L1_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_sharded_to_interleaved(mesh_device, shape, shard_type, cores, out_mem_config, dtype, layout, enable_trace):
    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch_input

    # Shard spec
    num_cores = cores[0] * cores[1]
    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores[0] - 1, cores[1] - 1)),
        }
    )
    padded_shape = list(shape)
    if layout == ttnn.TILE_LAYOUT:
        padded_shape[-1] = ttnn.core.roundup(padded_shape[-1], ttnn.TILE_SIZE)
        padded_shape[-2] = ttnn.core.roundup(padded_shape[-2], ttnn.TILE_SIZE)
    if shard_type == "H":
        mem_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        shard_shape = ((padded_shape[0] * padded_shape[1] * padded_shape[2]) // num_cores, padded_shape[3])
    else:
        mem_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        shard_shape = (padded_shape[0] * padded_shape[1] * padded_shape[2], padded_shape[3] // num_cores)
    mem_config = ttnn.MemoryConfig(
        mem_layout, ttnn.BufferType.L1, ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.to_memory_config(tt_input, out_mem_config)

    def check_op(tt_output):
        tt_output = tt_output[tuple(slice(s) for s in shape)]
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op, enable_trace)
