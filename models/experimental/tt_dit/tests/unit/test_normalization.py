# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn

from ...utils.tensor import bf16_tensor
from ...utils.check import assert_quality
from ...layers.normalization import RMSNorm, LayerNorm, DistributedLayerNorm
from ...parallel.manager import CCLManager


class TorchRMSNorm(torch.nn.Module):
    def __init__(self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True):
        super().__init__()
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.use_bias = bias
        self.weight = torch.nn.Parameter(torch.randn(embedding_dim))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(embedding_dim))

    def forward(self, x):
        x = x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.norm_eps)
        if self.norm_elementwise_affine:
            x = x * self.weight
            if self.use_bias:
                x = x + self.bias
        return x


class TorchLayerNorm(torch.nn.Module):
    def __init__(self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True):
        super().__init__()
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.use_bias = bias
        self.weight = torch.nn.Parameter(torch.randn(embedding_dim))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(embedding_dim))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.norm_eps)
        if self.norm_elementwise_affine:
            x = x * self.weight
            if self.use_bias:
                x = x + self.bias
        return x


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1), (1, 2), (2, 1)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("input_shape"),
    [
        (1, 38, 4096, 64),  # Q, K spatial norm
        (1, 38, 333, 64),  # Q, K prompt norm
    ],
)
@pytest.mark.parametrize(
    ("norm_eltwise_affine, bias"),
    [
        (True, False),
    ],
)
def test_rmsnorm(
    mesh_device: ttnn.MeshDevice,
    input_shape: tuple[int, int, int, int],
    norm_eltwise_affine: bool,
    bias: bool,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = TorchRMSNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias
    ).to(dtype=torch_dtype)
    torch_model.eval()

    tt_model = RMSNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias, mesh_device=mesh_device
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype) * 2 + 4

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    for t in ttnn.get_device_tensors(tt_output):
        t = ttnn.to_torch(t)
        assert_quality(torch_output, t, pcc=0.999_500)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1), (1, 2), (2, 1)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("input_shape"),
    [
        (1, 1, 4096, 2432),  # spatial norm
        (1, 1, 333, 2432),  # prompt norm
    ],
)
@pytest.mark.parametrize(
    ("norm_eltwise_affine, bias"),
    [
        (True, False),
    ],
)
def test_layernorm(
    mesh_device: ttnn.MeshDevice,
    input_shape: tuple[int, int, int, int],
    norm_eltwise_affine: bool,
    bias: bool,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = TorchLayerNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias
    ).to(dtype=torch_dtype)
    torch_model.eval()

    tt_model = LayerNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias, mesh_device=mesh_device
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype) * 2 + 4

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    for t in ttnn.get_device_tensors(tt_output):
        t = ttnn.to_torch(t)
        assert_quality(torch_output, t, pcc=0.982_000)


@pytest.mark.parametrize(
    "mesh_device, mesh_axis",
    [
        [(1, 2), 1],
        [(2, 1), 0],
        [(2, 2), 0],
        [(2, 2), 1],
        [(2, 4), 0],
        [(4, 2), 1],
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("input_shape"),
    [
        (1, 1, 4096, 2432),  # spatial norm
        (1, 1, 333, 2432),  # prompt norm
    ],
)
@pytest.mark.parametrize(
    ("norm_eltwise_affine, bias"),
    [
        (True, True),
        (False, False),
        (True, False),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_distributed_layernorm(
    mesh_device: ttnn.MeshDevice,
    mesh_axis: int,
    input_shape: tuple[int, int, int, int],
    norm_eltwise_affine: bool,
    bias: bool,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = TorchLayerNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias
    ).to(dtype=torch_dtype)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)

    tt_model = DistributedLayerNorm(
        embedding_dim=input_shape[-1],
        norm_elementwise_affine=norm_eltwise_affine,
        bias=bias,
        mesh_device=mesh_device,
        mesh_axis=mesh_axis,
        ccl_manager=ccl_manager,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype) * 2 + 4

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device, mesh_axis=mesh_axis, shard_dim=-1)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    shard_dims = [None, None]
    shard_dims[mesh_axis] = -1
    shard_dims[1 - mesh_axis] = 0
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    for i in range(tt_output.shape[0]):
        assert_quality(torch_output.squeeze(), tt_output[i].squeeze(), pcc=0.999_300)
