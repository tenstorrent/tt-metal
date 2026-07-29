# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel as TorchSD3Transformer2DModel

import ttnn

from ....layers.embeddings import PatchEmbed
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor


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
# @pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_patch_embed_sd35(
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
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

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
