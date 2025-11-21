# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from .....utils.tensor import bf16_tensor
from .....utils.check import assert_quality
from .....layers.linear import ColParallelLinear
from .....parallel.manager import CCLManager


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 32)],
    indirect=True,
)
@pytest.mark.parametrize("tp_mesh_axis", [0])
@pytest.mark.parametrize("is_fsdp", [True, False], ids=["yes_fsdp", "no_fsdp"])
@pytest.mark.parametrize(
    ("B, M, K, N, bias, sp_mesh_axis"),
    [
        (1, 81920, 5120, 5120, False, 1),  # sequence padding for chunk size 256
    ],
    ids=[
        "wan_spatial_qkv",
    ],
)
@pytest.mark.parametrize("num_links", [1], ids=["num_links_1"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT}],
    indirect=True,
)
def test_wan_col_parallel_linear(
    mesh_device: ttnn.MeshDevice,
    B: int,
    M: int,
    K: int,
    N: int,
    bias: bool,
    sp_mesh_axis: int,
    tp_mesh_axis: int,
    is_fsdp: bool,
    num_links: int,
    reset_seeds,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = torch.nn.Linear(K, N, bias=bias).to(dtype=torch_dtype)
    torch_model.eval()

    if is_fsdp:
        fsdp_mesh_axis = 1 - tp_mesh_axis
        ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        tt_model = ColParallelLinear(
            K,
            N,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=tp_mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
    else:
        tt_model = ColParallelLinear(K, N, bias=bias, mesh_device=mesh_device, mesh_axis=tp_mesh_axis)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((1, B, M, K), dtype=torch_dtype)

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device, mesh_axis=sp_mesh_axis, shard_dim=-2)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    shard_dims = [None, None]
    shard_dims[tp_mesh_axis] = -1
    shard_dims[sp_mesh_axis] = -2
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )

    assert_quality(torch_output.squeeze(), tt_output.squeeze(), pcc=0.999_500)
