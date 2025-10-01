# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn

from ...utils.tensor import bf16_tensor
from ...utils.check import assert_quality
from ...layers.feedforward import FeedForward, ParallelFeedForward
from ...parallel.manager import CCLManager


class TorchFeedForward(torch.nn.Module):
    def __init__(self, dim, dim_out, bias=True, activation_fn=None, inner_dim=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.bias = bias
        self.activation_fn = activation_fn
        if activation_fn == "swiglu":
            ff1_inner_dim = inner_dim * 2
        else:
            ff1_inner_dim = inner_dim
        self.ff1 = torch.nn.Linear(dim, ff1_inner_dim, bias=bias)
        self.ff2 = torch.nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x):
        x = self.ff1(x)
        if self.activation_fn == "gelu":
            x = torch.nn.functional.gelu(x)
        elif self.activation_fn == "swiglu":
            x, gate = torch.chunk(x, 2, -1)
            x = x * torch.nn.functional.silu(gate)
        return self.ff2(x)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1), (1, 2), (2, 1)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("B, seq, dim, inner_dim, dim_out"),
    [
        (1, 1, 256, 2432, 2432),  # SD3.5 timestep_embedder
        (1, 333, 2048, 2432, 2432),  # SD3.5 text_embedder
        (1, 4096, 2432, 9728, 2432),  # SD3.5 spatial FF
        (1, 333, 2432, 9728, 2432),  # SD3.5 text FF
    ],
)
@pytest.mark.parametrize("activation_fn", [None, "gelu"])
@pytest.mark.parametrize(
    ("bias"),
    [
        True,
        # False,
    ],
)
def test_feedforward(
    mesh_device: ttnn.MeshDevice,
    B: int,
    seq: int,
    dim: int,
    inner_dim: int,
    dim_out: int,
    bias: bool,
    activation_fn: str,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = TorchFeedForward(dim, dim_out, bias=bias, activation_fn=activation_fn, inner_dim=inner_dim).to(
        dtype=torch_dtype
    )
    torch_model.eval()

    tt_model = FeedForward(
        dim, dim_out, inner_dim=inner_dim, bias=bias, activation_fn=activation_fn, mesh_device=mesh_device
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((1, B, seq, dim), dtype=torch_dtype)

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    device_grid = mesh_device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    tt_output = tt_model(tt_input_tensor, core_grid=core_grid)

    for t in ttnn.get_device_tensors(tt_output):
        t = ttnn.to_torch(t)
        assert_quality(torch_output, t, pcc=0.999_400)


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
    ("B, seq, dim, inner_dim, dim_out, bias, activation_fn"),
    [
        (1, 1, 256, 2432, 2432, True, "gelu"),  # SD3.5 timestep_embedder
        (1, 333, 2048, 2432, 2432, True, "gelu"),  # SD3.5 text_embedder
        (1, 4096, 2432, 9728, 2432, True, "gelu"),  # SD3.5 spatial FF
        (1, 333, 2432, 9728, 2432, True, "gelu"),  # SD3.5 text FF
        (1, 44520, 3072, 8192, 3072, False, "swiglu"),  # Mochi spatial FF
        (1, 118, 1536, 4096, 1536, False, "swiglu"),  # Mochi prompt FF
    ],
    ids=[
        "sd35_timestep_embedder",
        "sd35_text_embedder",
        "sd35_spatial_ff",
        "sd35_text_ff",
        "mochi_spatial_ff",
        "mochi_prompt_ff",
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_parallel_feedforward(
    mesh_device: ttnn.MeshDevice,
    B: int,
    seq: int,
    dim: int,
    inner_dim: int,
    dim_out: int,
    bias: bool,
    activation_fn: str,
    tp_mesh_axis: int,
    is_fsdp: bool,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = TorchFeedForward(dim, dim_out, bias=bias, activation_fn=activation_fn, inner_dim=inner_dim).to(
        dtype=torch_dtype
    )
    torch_model.eval()

    fsdp_mesh_axis = 1 - tp_mesh_axis if is_fsdp else None

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    tt_model = ParallelFeedForward(
        dim,
        dim_out,
        inner_dim=inner_dim,
        bias=bias,
        activation_fn=activation_fn,
        mesh_device=mesh_device,
        mesh_axis=tp_mesh_axis,
        fsdp_mesh_axis=fsdp_mesh_axis,
        ccl_manager=ccl_manager,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((1, B, seq, dim), dtype=torch_dtype)

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    device_grid = mesh_device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    tt_output = tt_model(tt_input_tensor, core_grid=core_grid)

    shard_dims = [None, None]
    shard_dims[tp_mesh_axis] = -1
    shard_dims[1 - tp_mesh_axis] = 0
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    for i in range(tt_output.shape[0]):
        assert_quality(torch_output.squeeze(), tt_output[i].squeeze(), pcc=0.999_400)
