# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers.models.transformers.transformer_mochi import MochiTransformer3DModel

import ttnn

from ....layers.embeddings import MochiPatchEmbed
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
    ("B, T, H, W, patch_size, in_channels, embed_dim"),
    [
        (1, 28, 60, 106, 2, 12, 3072),  # Mochi config
    ],
)
# @pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_patch_embed_mochi(
    mesh_device: ttnn.MeshDevice,
    tp_mesh_axis: int,
    sp_mesh_axis: int,
    B: int,
    T: int,
    H: int,
    W: int,
    patch_size: int,
    in_channels: int,
    embed_dim: int,
) -> None:
    torch_dtype = torch.bfloat16

    # Create Torch model
    parent_torch_model = MochiTransformer3DModel.from_pretrained(
        f"genmo/mochi-1-preview", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model = parent_torch_model.patch_embed
    torch_model.eval()

    assert patch_size == torch_model.patch_size
    assert in_channels == torch_model.proj.in_channels
    assert embed_dim == torch_model.proj.out_channels

    # Create TT model
    tt_model = MochiPatchEmbed(
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        mesh_device=mesh_device,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # Create input tensors - NHWC format for TT model
    torch.manual_seed(0)
    input_tensor_ncthw = torch.randn((B, in_channels, T, H, W), dtype=torch_dtype)
    input_tensor_ntchw = input_tensor_ncthw.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B*T, in_channels, H, W)

    input_tensor_patched = input_tensor_ncthw.clone().reshape(
        B, in_channels, T, H // patch_size, patch_size, W // patch_size, patch_size
    )
    input_tensor_patched = input_tensor_patched.permute(
        0, 2, 3, 5, 4, 6, 1
    )  # (B, T, H//patch_size, W//patch_size, patch_size, patch_size, in_channels)
    input_tensor_patched = input_tensor_patched.reshape(
        1, B, T * H // patch_size * W // patch_size, patch_size * patch_size * in_channels
    )

    # Run torch model (expects NHWC input)
    torch_output_tsd = torch_model(input_tensor_ntchw)
    torch_output_flattened = torch_output_tsd.reshape(B, T * H // patch_size * W // patch_size, embed_dim)

    # Convert to TT tensor
    tt_input = bf16_tensor(input_tensor_patched, device=mesh_device, mesh_axis=sp_mesh_axis, shard_dim=2)

    # Run TT model
    tt_output = tt_model(tt_input)

    # Convert back to torch and compare
    # Handle sharded output
    shard_dims = [None, None]
    shard_dims[sp_mesh_axis] = 2  # Sequence dimension sharding
    shard_dims[tp_mesh_axis] = 0
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    for i in range(tt_output_torch.shape[0]):
        assert_quality(
            torch_output_flattened, tt_output_torch[i], pcc=0.999_994, relative_rmse=0.05
        )  # Lower PCC due to conv2d approximation
