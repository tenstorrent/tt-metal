# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from loguru import logger

from ...utils.tensor import bf16_tensor
from ...utils.check import assert_quality
from ...layers.normalization import RMSNorm, LayerNorm, DistributedLayerNorm, GroupNorm, DistributedRMSNorm
from ...parallel.manager import CCLManager


class TorchRMSNorm(torch.nn.Module):
    def __init__(self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True):
        super().__init__()
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.use_bias = bias
        if norm_elementwise_affine:
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
        if norm_elementwise_affine:
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
    tt_model.load_torch_state_dict(torch_model.state_dict())

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
    ("input_shape", "use_row_major_workaround"),
    [
        ((1, 1, 4096, 2432), False),  # spatial norm
        ((1, 1, 333, 2432), False),  # prompt norm
        ((1, 1, 22528, 3072), True),  # Mochi large layernorm
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
    use_row_major_workaround: bool,
) -> None:
    MIN_PCC = 0.982_000 if input_shape[-2] < 20000 else 0.961_000
    torch_dtype = torch.bfloat16
    torch_model = TorchLayerNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias
    ).to(dtype=torch_dtype)
    torch_model.eval()

    tt_model = LayerNorm(
        embedding_dim=input_shape[-1],
        norm_elementwise_affine=norm_eltwise_affine,
        bias=bias,
        mesh_device=mesh_device,
        use_row_major_workaround=use_row_major_workaround,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype) * 2 + 4

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    logger.info(f"Running torch model with input shape {torch_input_tensor.shape}")
    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    logger.info(f"Running TT model with input shape {tt_input_tensor.shape}")
    tt_output = tt_model(tt_input_tensor)

    for t in ttnn.get_device_tensors(tt_output):
        t = ttnn.to_torch(t)
        assert_quality(torch_output, t, pcc=MIN_PCC)


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
    ids=[
        "1x2_1",
        "2x1_0",
        "2x2_0",
        "2x2_1",
        "2x4_0",
        "4x2_1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("input_shape"),
    [
        (1, 1, 4096, 2432),  # spatial norm
        (1, 1, 333, 2432),  # prompt norm
        (1, 1, 32768, 384),
    ],
    ids=["shape1", "shape2", "shape3"],
)
@pytest.mark.parametrize(
    ("norm_eltwise_affine, bias"),
    [
        (True, False),
        (False, False),
    ],
    ids=["yes_eltwise_no_bias", "no_eltwise_no_bias"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_distributed_rms_norm(
    mesh_device: ttnn.MeshDevice,
    mesh_axis: int,
    input_shape: tuple[int, int, int, int],
    norm_eltwise_affine: bool,
    bias: bool,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = TorchRMSNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias
    ).to(dtype=torch_dtype)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)

    tt_model = DistributedRMSNorm(
        embedding_dim=input_shape[-1],
        norm_elementwise_affine=norm_eltwise_affine,
        bias=bias,
        mesh_device=mesh_device,
        mesh_axis=mesh_axis,
        ccl_manager=ccl_manager,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

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


TP_SWEEP = [
    pytest.param((1, 1), 0, id="tp1_axis0"),
    pytest.param((1, 2), 1, id="tp2_axis1"),
    pytest.param((1, 4), 1, id="tp4_axis1"),
]


@pytest.mark.parametrize(
    "embedding_dim",
    [2048, 2432, 3072, 5120],
    ids=["dim0", "dim1", "dim2", "dim3"],
)
@pytest.mark.parametrize(
    "seq_len",
    [512, 2048, 4096, 9472],
    ids=["len0", "len1", "len2", "len3"],
)
@pytest.mark.parametrize(
    "affine_parameters, affine_dynamic",
    [
        (False, False),
        (True, False),
        (False, True),
    ],
    ids=["no_affine", "static_affine", "dynamic_affine"],
)
@pytest.mark.parametrize("mesh_device, mesh_axis", TP_SWEEP, indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_distributed_layernorm(
    mesh_device: ttnn.MeshDevice,
    mesh_axis: int,
    embedding_dim: int,
    seq_len: int,
    affine_parameters: bool,
    affine_dynamic: bool,
) -> None:
    """Covers all DistributedLayerNorm instantiations seen in tt_dit models."""
    torch_dtype = torch.bfloat16

    torch_model = TorchLayerNorm(
        embedding_dim=embedding_dim, norm_elementwise_affine=affine_parameters, bias=affine_parameters
    ).to(dtype=torch_dtype)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)

    tt_model = DistributedLayerNorm(
        embedding_dim=embedding_dim,
        norm_elementwise_affine=affine_parameters,
        bias=affine_parameters,
        mesh_device=mesh_device,
        mesh_axis=mesh_axis,
        ccl_manager=ccl_manager,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((1, 1, seq_len, embedding_dim), dtype=torch_dtype) * 2 + 4
    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device, mesh_axis=mesh_axis, shard_dim=-1)

    if affine_dynamic:
        torch_model.norm_elementwise_affine = True
        torch_model.use_bias = True
        torch_model.weight = torch.nn.Parameter(torch.randn((embedding_dim), dtype=torch_dtype))
        torch_model.bias = torch.nn.Parameter(torch.randn((embedding_dim), dtype=torch_dtype))
        # Tilized weights and bias for dynamic affine
        tt_dynamic_weight_tensor = bf16_tensor(
            torch_model.weight.data.unsqueeze(0), device=mesh_device, mesh_axis=mesh_axis, shard_dim=-1
        )
        tt_dynamic_bias_tensor = bf16_tensor(
            torch_model.bias.data.unsqueeze(0), device=mesh_device, mesh_axis=mesh_axis, shard_dim=-1
        )
    else:
        tt_dynamic_weight_tensor = None
        tt_dynamic_bias_tensor = None

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor, dynamic_weight=tt_dynamic_weight_tensor, dynamic_bias=tt_dynamic_bias_tensor)

    shard_dims = [None, None]
    shard_dims[mesh_axis] = -1
    shard_dims[1 - mesh_axis] = 0
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )

    for i in range(tt_output.shape[0]):
        assert_quality(torch_output.squeeze(), tt_output[i].squeeze(), pcc=0.999_300)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("group_count", [32])
@pytest.mark.parametrize(
    "mesh_axis",
    [1, None],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 512, 128, 128),
        (1, 512, 256, 256),
        (1, 512, 512, 512),
        (1, 256, 512, 512),
        (1, 256, 1024, 1024),
    ],
)
def test_group_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    input_shape: tuple[int, int, int, int],
    group_count: int,
    mesh_axis: int,
) -> None:
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)

    torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=input_shape[1])
    torch.nn.init.normal_(torch_model.weight)
    torch.nn.init.normal_(torch_model.bias)
    torch_model.eval()

    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype)

    tt_model = GroupNorm.from_torch(
        torch_ref=torch_model,
        mesh_device=mesh_device,
        mesh_axis=mesh_axis,
        core_grid=ttnn.CoreGrid(x=8, y=8),
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1),
        dtype=ttnn_dtype,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1) if mesh_axis is not None else None,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model(tt_input_tensor)

    tt_torch = ttnn.to_torch(
        tt_output if mesh_axis is not None else ttnn.get_device_tensors(tt_output)[0],
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1) if mesh_axis is not None else None,
    )

    tt_torch = tt_torch.permute(0, 3, 1, 2)

    assert_quality(torch_output, tt_torch, pcc=0.999_300)
