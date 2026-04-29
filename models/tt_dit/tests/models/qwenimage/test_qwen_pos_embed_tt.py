# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for ``QwenPosEmbedTT`` device-side RoPE builder.

Validates that the device assembly reproduces the host-side
``diffusers.QwenEmbedRope.forward`` outputs (within bf16 tolerance) on
replicated meshes (sp_factor == 1).
"""

import pytest
import torch
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope

import ttnn

from ....pipelines.qwenimage.qwen_pos_embed_tt import QwenPosEmbedTT
from ....utils import tensor as tensor_utils


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 1), id="1x1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("axes_dim", "scale_rope"),
    [
        pytest.param([16, 56, 56], True, id="qwen_image_scale_rope"),
    ],
)
@pytest.mark.parametrize(
    ("img_shapes", "max_txt_seq_len"),
    [
        pytest.param([(1, 32, 32)], 128, id="single_32x32_txt128"),
        pytest.param([(1, 64, 64)], 256, id="single_64x64_txt256"),
        pytest.param([(1, 32, 32), (1, 16, 16)], 128, id="edit_two_images"),
    ],
)
def test_qwen_pos_embed_tt(
    mesh_device: ttnn.MeshDevice,
    axes_dim: list[int],
    scale_rope: bool,
    img_shapes: list[tuple[int, int, int]],
    max_txt_seq_len: int,
) -> None:
    torch.manual_seed(0)

    torch_pos_embed = QwenEmbedRope(theta=10000, axes_dim=axes_dim, scale_rope=scale_rope)

    ref_img_shapes = [img_shapes]
    ref_txt_seq_lens = [max_txt_seq_len]
    spatial_rope_ref, prompt_rope_ref = torch_pos_embed.forward(
        ref_img_shapes, ref_txt_seq_lens, device=None, max_txt_seq_len=max_txt_seq_len
    )

    spatial_cos_ref = spatial_rope_ref.real
    spatial_sin_ref = spatial_rope_ref.imag
    prompt_cos_ref = prompt_rope_ref.real
    prompt_sin_ref = prompt_rope_ref.imag

    tt_pos_embed = QwenPosEmbedTT(torch_pos_embed, [mesh_device])
    tt_spatial_cos, tt_spatial_sin, tt_prompt_cos, tt_prompt_sin = tt_pos_embed.build(
        submesh_index=0,
        img_shapes_per_batch=img_shapes,
        max_txt_seq_len=max_txt_seq_len,
    )

    spatial_cos_dev = tensor_utils.to_torch(tt_spatial_cos)
    spatial_sin_dev = tensor_utils.to_torch(tt_spatial_sin)
    prompt_cos_dev = tensor_utils.to_torch(tt_prompt_cos)
    prompt_sin_dev = tensor_utils.to_torch(tt_prompt_sin)

    atol = 2e-2
    rtol = 2e-2

    def _check(name, ref, dev):
        assert ref.shape == dev.shape, f"{name}: shape mismatch ref={tuple(ref.shape)} dev={tuple(dev.shape)}"
        diff = (ref.to(torch.float32) - dev.to(torch.float32)).abs()
        max_abs = diff.max().item()
        assert torch.allclose(
            dev.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol
        ), f"{name} mismatch: max_abs={max_abs}"

    _check("spatial_cos", spatial_cos_ref, spatial_cos_dev)
    _check("spatial_sin", spatial_sin_ref, spatial_sin_dev)
    _check("prompt_cos", prompt_cos_ref, prompt_cos_dev)
    _check("prompt_sin", prompt_sin_ref, prompt_sin_dev)
