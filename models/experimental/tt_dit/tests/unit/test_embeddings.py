# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
import math

from ...utils.tensor import bf16_tensor
from ...utils.check import assert_quality
from ...layers.embeddings import (
    TimestepEmbedding,
    PixartAlphaTextProjection,
    SD35CombinedTimestepTextProjEmbeddings,
    PatchEmbed,
)
from ....stable_diffusion_35_large.reference import SD3Transformer2DModel as TorchSD3Transformer2DModel


class TorchCombinedTimestepTextProjEmbeddings(torch.nn.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()

        self.timestep_embedder = TorchTimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TorchPixArtAlphaTextProjection(
            in_features=pooled_projection_dim, hidden_size=embedding_dim
        )

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = _time_proj(num_channels=256, timesteps=timestep)

        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))

        return timesteps_emb + self.text_embedder(pooled_projection)


class TorchTimestepEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim)
        self.act = torch.nn.SiLU()
        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


def _time_proj(num_channels: int, timesteps: torch.Tensor) -> torch.Tensor:
    assert num_channels % 2 == 0
    half_dim = num_channels // 2

    max_period = 10000

    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TorchPixArtAlphaTextProjection(torch.nn.Module):
    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_size)
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.act_1 = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act_1(x)
        return self.linear_2(x)


class TorchPatchEmbed(torch.nn.Module):
    def __init__(
        self,
        *,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        pos_embed_max_size: int,
    ) -> None:
        super().__init__()

        self.pos_embed_max_size = pos_embed_max_size
        self.patch_size = patch_size

        self.proj = torch.nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
        )

        self.register_buffer("pos_embed", torch.zeros((1, pos_embed_max_size**2, embed_dim)))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        height, width = latent.shape[-2:]

        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC

        pos_embed = self._cropped_pos_embed(height, width)

        return (latent + pos_embed).to(latent.dtype)

    def _cropped_pos_embed(self, height: int, width: int) -> torch.Tensor:
        height = height // self.patch_size
        width = width // self.patch_size

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2

        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        return spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("B, in_channels, time_embed_dim"),
    [
        (1, 256, 2432),  # SD3.5 timestep embedding
        (2, 256, 1536),  # SD3.5 medium
        (1, 128, 1024),  # Smaller test case
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_timestep_embedding(
    mesh_device: ttnn.MeshDevice,
    B: int,
    in_channels: int,
    time_embed_dim: int,
) -> None:
    torch_dtype = torch.bfloat16

    # Create Torch model
    torch_model = TorchTimestepEmbedding(in_channels, time_embed_dim).to(torch_dtype)
    torch_model.eval()

    # Create TT model
    tt_model = TimestepEmbedding(
        in_channels=in_channels,
        time_embed_dim=time_embed_dim,
        mesh_device=mesh_device,
        init=True,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Create input tensors
    torch.manual_seed(0)
    input_tensor = torch.randn((B, in_channels), dtype=torch_dtype)

    # Run torch model
    torch_output = torch_model(input_tensor)

    # Convert to TT tensor
    input_4d = input_tensor.unsqueeze(0).unsqueeze(0)
    tt_input = bf16_tensor(input_4d, device=mesh_device)

    # Run TT model
    tt_output = tt_model(tt_input)

    # Convert back to torch and compare
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0).squeeze(0)
    assert_quality(torch_output, tt_output_torch, pcc=0.9998, relative_rmse=0.017)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("B, in_features, hidden_size"),
    [
        (1, 2048, 2432),  # SD3.5 text projection
        (2, 1152, 1536),  # SD3.5 medium text projection
        (1, 512, 1024),  # Smaller test case
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_pixart_alpha_text_projection(
    mesh_device: ttnn.MeshDevice,
    B: int,
    in_features: int,
    hidden_size: int,
) -> None:
    torch_dtype = torch.bfloat16

    # Create Torch model
    torch_model = TorchPixArtAlphaTextProjection(in_features, hidden_size).to(torch_dtype)
    torch_model.eval()

    # Create TT model
    tt_model = PixartAlphaTextProjection(
        in_features=in_features,
        hidden_size=hidden_size,
        mesh_device=mesh_device,
        init=True,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Create input tensors
    torch.manual_seed(0)
    input_tensor = torch.randn((B, in_features), dtype=torch_dtype)

    # Run torch model
    torch_output = torch_model(input_tensor)

    # Convert to TT tensor
    input_4d = input_tensor.unsqueeze(0).unsqueeze(0)
    tt_input = bf16_tensor(input_4d, device=mesh_device)

    # Run TT model
    tt_output = tt_model(tt_input)

    # Convert back to torch and compare
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0).squeeze(0)
    assert_quality(torch_output, tt_output_torch, pcc=0.99984, relative_rmse=0.019)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("B, embedding_dim, pooled_projection_dim"),
    [
        (1, 2432, 2048),  # SD3.5 large config
        (2, 1536, 1152),  # SD3.5 medium config
        (1, 1024, 768),  # Smaller test case
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_combined_timestep_text_proj_embeddings(
    mesh_device: ttnn.MeshDevice,
    B: int,
    embedding_dim: int,
    pooled_projection_dim: int,
) -> None:
    torch_dtype = torch.bfloat16

    # Create Torch model
    torch_model = TorchCombinedTimestepTextProjEmbeddings(embedding_dim, pooled_projection_dim).to(torch_dtype)
    torch_model.eval()

    # Create TT model
    tt_model = SD35CombinedTimestepTextProjEmbeddings(
        embedding_dim=embedding_dim,
        pooled_projection_dim=pooled_projection_dim,
        mesh_device=mesh_device,
        init=True,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Create input tensors
    torch.manual_seed(0)
    timestep_tensor = torch.randn((B,), dtype=torch_dtype)
    pooled_projection_tensor = torch.randn((B, pooled_projection_dim), dtype=torch_dtype)

    # Run torch model
    torch_output = torch_model(timestep_tensor, pooled_projection_tensor)

    # Convert to TT tensors
    timestep_4d = timestep_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    pooled_4d = pooled_projection_tensor.unsqueeze(0).unsqueeze(0)
    tt_timestep = bf16_tensor(timestep_4d, device=mesh_device)
    tt_pooled = bf16_tensor(pooled_4d, device=mesh_device)

    # Run TT model
    tt_output = tt_model(tt_timestep, tt_pooled)

    # Convert back to torch and compare
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0).squeeze(0)
    assert_quality(
        torch_output, tt_output_torch, pcc=0.99984, relative_rmse=0.019
    )  # Slightly lower PCC due to sinusoidal ops


@pytest.mark.parametrize(
    "mesh_device, tp_mesh_axis, sp_mesh_axis",
    [
        [(1, 1), 0, 1],
        [(1, 2), 0, 1],
        [(1, 2), 1, 0],
        [(2, 1), 0, 1],
        [(2, 1), 1, 0],
        [(2, 2), 0, 1],
        [(2, 2), 1, 0],
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, H, W, patch_size, in_channels, embed_dim, pos_embed_max_size"),
    [
        (1, 128, 128, 2, 16, 2432, 192),  # SD3.5 large config
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_patch_embed(
    mesh_device: ttnn.MeshDevice,
    tp_mesh_axis: int,
    sp_mesh_axis: int,
    B: int,
    H: int,
    W: int,
    patch_size: int,
    in_channels: int,
    embed_dim: int,
    pos_embed_max_size: int,
) -> None:
    torch_dtype = torch.bfloat16

    # Create Torch model
    parent_torch_model = TorchSD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-large", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model = parent_torch_model.pos_embed
    torch_model.eval()

    assert patch_size == torch_model.patch_size
    assert in_channels == torch_model.proj.in_channels
    assert embed_dim == torch_model.proj.out_channels
    assert pos_embed_max_size == torch_model.pos_embed_max_size

    # Create TT model
    tt_model = PatchEmbed(
        height=H,
        width=W,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        pos_embed_max_size=pos_embed_max_size,
        mesh_device=mesh_device,
        tp_mesh_axis=tp_mesh_axis,
        sp_mesh_axis=sp_mesh_axis,
        init=False,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    # Create input tensors - NHWC format for TT model
    torch.manual_seed(0)
    input_tensor_nchw = torch.randn((B, in_channels, H, W), dtype=torch_dtype)
    input_tensor_nhwc = input_tensor_nchw.permute(0, 2, 3, 1).clone()

    # Run torch model (expects NHWC input)
    torch_output = torch_model(input_tensor_nchw)

    # Convert to TT tensor
    tt_input = bf16_tensor(input_tensor_nhwc, device=mesh_device, mesh_axis=sp_mesh_axis, shard_dim=1)

    # Run TT model
    tt_output = tt_model(tt_input)

    # Convert back to torch and compare
    # Handle sharded output
    shard_dims = [None, None]
    shard_dims[sp_mesh_axis] = 2  # Sequence dimension sharding
    shard_dims[tp_mesh_axis] = 3
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    assert_quality(
        torch_output, tt_output_torch, pcc=0.999_995, relative_rmse=0.05
    )  # Lower PCC due to conv2d approximation
