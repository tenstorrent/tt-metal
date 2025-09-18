# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn

from ...utils.tensor import bf16_tensor
from ...utils.check import assert_quality
from ...layers.linear import Linear, ColParallelLinear, RowParallelLinear
from ...parallel.manager import CCLManager


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1), (1, 2), (2, 1)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("B, M, K, N"),
    [
        (1, 1, 256, 2432),  # SD3.5 timestep_embedder
        (1, 1, 2432, 2432),  # SD3.5 timestep_embedder
        (1, 333, 2048, 2432),  # SD3.5 text_embedder
        (1, 333, 2432, 2432),  # SD3.5 text_embedder
        (1, 4096, 2432, 2432 * 3),  # SD3.5 spatial qkv
        (1, 333, 2432, 2432 * 3),  # SD3.5 text qkv
        (1, 4096, 2432, 2432),  # SD3.5 spatial out_proj
        (1, 333, 2432, 2432),  # SD3.5 text out_proj
        (1, 4096, 2432, 9728),  # SD3.5 spatial FF
        (1, 4096, 9728, 2432),  # SD3.5 spatial FF
        (1, 333, 2432, 9728),  # SD3.5 text FF
        (1, 333, 9728, 2432),  # SD3.5 text FF
        (1, 1, 2432, 14592),  # SD3.5 context
        (1, 1, 2432, 4864),  # SD3.5 final context
        (1, 4096, 2432, 64),  # SD3.5 proj_out
    ],
)
@pytest.mark.parametrize(
    ("bias"),
    [
        True,
        # False,
    ],
)
def test_linear(
    mesh_device: ttnn.MeshDevice,
    B: int,
    M: int,
    K: int,
    N: int,
    bias: bool,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = torch.nn.Linear(K, N, bias=bias).to(dtype=torch_dtype)
    torch_model.eval()

    tt_model = Linear(K, N, bias=bias, mesh_device=mesh_device)
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((1, B, M, K), dtype=torch_dtype)

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    for t in ttnn.get_device_tensors(tt_output):
        t = ttnn.to_torch(t)
        assert_quality(torch_output, t, pcc=0.999_500)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("tp_mesh_axis"),
    [
        0,
        1,
    ],
)
@pytest.mark.parametrize("is_fsdp", [True, False], ids=["yes_fsdp", "no_fsdp"])
@pytest.mark.parametrize(
    ("B, M, K, N"),
    [
        (1, 1, 256, 2432),  # SD3.5 timestep_embedder
        (1, 1, 2432, 2432),  # SD3.5 timestep_embedder
        (1, 333, 2048, 2432),  # SD3.5 text_embedder
        (1, 333, 2432, 2432),  # SD3.5 text_embedder
        (1, 4096, 2432, 2432 * 3),  # SD3.5 spatial qkv
        (1, 333, 2432, 2432 * 3),  # SD3.5 text qkv
        (1, 4096, 2432, 2432),  # SD3.5 spatial out_proj
        (1, 333, 2432, 2432),  # SD3.5 text out_proj
        (1, 4096, 2432, 9728),  # SD3.5 spatial FF
        (1, 4096, 9728, 2432),  # SD3.5 spatial FF
        (1, 333, 2432, 9728),  # SD3.5 text FF
        (1, 333, 9728, 2432),  # SD3.5 text FF
        (1, 1, 2432, 14592),  # SD3.5 context
        (1, 1, 2432, 4864),  # SD3.5 final context
        (1, 1, 3072, 12288),  # Mochi timestep norm1 linear
        (1, 1, 3072, 6144),  # Mochi timestep norm1_context linear
        (1, 44520, 3072, 3072),  # Mochi spatial Q, K, V and out_proj
        (1, 118, 1536, 3072),  # Mochi text Q, K, V
        (1, 118, 3072, 1536),  # Mochi text out_proj
    ],
    ids=[
        "sd35_timestep_embedder",
        "sd35_timestep_embedder_2",
        "sd35_text_embedder",
        "sd35_text_embedder_2",
        "sd35_spatial_qkv",
        "sd35_text_qkv",
        "sd35_spatial_out_proj",
        "sd35_text_out_proj",
        "sd35_spatial_FF",
        "sd35_spatial_FF_2",
        "sd35_text_FF",
        "sd35_text_FF_2",
        "sd35_context",
        "sd35_final_context",
        "mochi_timestep_norm1_linear",
        "mochi_timestep_norm1_context_linear",
        "mochi_spatial_qkv_and_out_proj",
        "mochi_text_qkv",
        "mochi_text_out_proj",
    ],
)
@pytest.mark.parametrize(
    ("bias"),
    [
        True,
        # False,
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_col_parallel_linear(
    mesh_device: ttnn.MeshDevice,
    B: int,
    M: int,
    K: int,
    N: int,
    bias: bool,
    tp_mesh_axis: int,
    is_fsdp: bool,
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
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((1, B, M, K), dtype=torch_dtype)

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    shard_dims = [None, None]
    shard_dims[tp_mesh_axis] = -1
    shard_dims[1 - tp_mesh_axis] = 0
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    for i in range(tt_output.shape[0]):
        assert_quality(torch_output.squeeze(), tt_output[i].squeeze(), pcc=0.999_500)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2), (2, 1), (2, 2), (2, 4), (4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("tp_mesh_axis"),
    [
        0,
        1,
    ],
)
@pytest.mark.parametrize("is_fsdp", [True, False], ids=["yes_fsdp", "no_fsdp"])
@pytest.mark.parametrize(
    ("B, M, K, N"),
    [
        (1, 1, 256, 2432),  # SD3.5 timestep_embedder
        (1, 1, 2432, 2432),  # SD3.5 timestep_embedder
        (1, 333, 2048, 2432),  # SD3.5 text_embedder
        (1, 333, 2432, 2432),  # SD3.5 text_embedder
        (1, 4096, 2432, 2432 * 3),  # SD3.5 spatial qkv
        (1, 333, 2432, 2432 * 3),  # SD3.5 text qkv
        (1, 4096, 2432, 2432),  # SD3.5 spatial out_proj
        (1, 333, 2432, 2432),  # SD3.5 text out_proj
        (1, 4096, 2432, 9728),  # SD3.5 spatial FF
        (1, 4096, 9728, 2432),  # SD3.5 spatial FF
        (1, 333, 2432, 9728),  # SD3.5 text FF
        (1, 333, 9728, 2432),  # SD3.5 text FF
        (1, 1, 2432, 14592),  # SD3.5 context
        (1, 1, 2432, 4864),  # SD3.5 final context
    ],
)
@pytest.mark.parametrize(
    ("bias"),
    [
        True,
        # False,
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_row_parallel_linear(
    mesh_device: ttnn.MeshDevice,
    B: int,
    M: int,
    K: int,
    N: int,
    bias: bool,
    tp_mesh_axis: int,
    is_fsdp: bool,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = torch.nn.Linear(K, N, bias=bias).to(dtype=torch_dtype)
    torch_model.eval()

    fsdp_mesh_axis = 1 - tp_mesh_axis if is_fsdp else None
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    tt_model = RowParallelLinear(
        K,
        N,
        bias=bias,
        mesh_device=mesh_device,
        mesh_axis=tp_mesh_axis,
        fsdp_mesh_axis=fsdp_mesh_axis,
        ccl_manager=ccl_manager,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((1, B, M, K), dtype=torch_dtype)

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device, mesh_axis=tp_mesh_axis, shard_dim=-1)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    shard_dims = [None, None]
    shard_dims[tp_mesh_axis] = -1
    shard_dims[1 - tp_mesh_axis] = 0
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    for i in range(tt_output.shape[0]):
        assert_quality(torch_output.squeeze(), tt_output[i].squeeze(), pcc=0.999_500)
