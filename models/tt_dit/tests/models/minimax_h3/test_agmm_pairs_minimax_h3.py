# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""A projection's persistent all-gather pair on the dynamic-length path: sized at the row capacity, one pair serves
every row count up to it, and the fused all-gather matmul reads only the live rows."""

import pytest
import torch

import ttnn

from ....layers.linear import ColParallelLinear
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor_2dshard
from .common import MESH_4X8_RING

TP_AXIS, SP_AXIS = 0, 1  # the Galaxy preset's axes
K, N = 3072, 3072
ROW_CAPACITY = 1024  # per-device rows the pair is sized at


@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_persistent_agmm_pair_at_row_capacity(mesh_device, reset_seeds):
    tp, sp = mesh_device.shape[TP_AXIS], mesh_device.shape[SP_AXIS]
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=TP_AXIS, factor=tp),
        sequence_parallel=ParallelFactor(mesh_axis=SP_AXIS, factor=sp),
        cfg_parallel=None,
    )
    torch_model = torch.nn.Linear(K, N, bias=False).to(dtype=torch.bfloat16).eval()
    tt_model = ColParallelLinear(K, N, bias=False, mesh_device=mesh_device, mesh_axis=TP_AXIS, ccl_manager=ccl_manager)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    def ag_pairs():
        return {key for key in ccl_manager._ping_pong_buffer_cache if key[0] == "ag"}

    for rows in (ROW_CAPACITY // 2, ROW_CAPACITY, ROW_CAPACITY // 4):
        torch_input = torch.randn((1, 1, rows * sp, K), dtype=torch.bfloat16)
        tt_input = bf16_tensor_2dshard(torch_input, device=mesh_device, shard_mapping={SP_AXIS: 2, TP_AXIS: 3})
        with torch.no_grad():
            torch_output = torch_model(torch_input)
        tt_output = tt_model(
            tt_input, parallel_config=parallel_config, use_persistent_buffer=True, persistent_buffer_rows=ROW_CAPACITY
        )
        tt_output = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 2), mesh_shape=tuple(mesh_device.shape)),
        )
        assert_quality(torch_output, tt_output, pcc=0.999_5)
        assert len(ag_pairs()) == 1, f"{rows} rows/device did not reuse the capacity-sized pair: {ag_pairs()}"
    (key,) = ag_pairs()
    assert key[1][-2] == ROW_CAPACITY
