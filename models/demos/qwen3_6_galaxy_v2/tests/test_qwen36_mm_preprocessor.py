# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-PREPROC tests — CPU sanity that the multimodal preprocessor produces the
right shapes + types from a real PIL image input.
"""

import torch
from PIL import Image

from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_preprocessor import Qwen36MMInputs, Qwen36MMPreprocessor


@torch.no_grad()
def test_preprocessor_text_only():
    """Pure-text prompt — no pixel_values, no image_grid_thw, 1D-equivalent positions."""
    proc = Qwen36MMPreprocessor("Qwen/Qwen3.6-27B")
    out = proc("Hello world")
    assert isinstance(out, Qwen36MMInputs)
    assert out.pixel_values is None
    assert out.image_grid_thw is None
    assert out.input_ids.ndim == 2
    assert out.attention_mask.shape == out.input_ids.shape
    assert out.position_ids_3d.shape == (3, *out.input_ids.shape)
    # Text-only: all 3 axes equal a 1D ramp
    ramp = torch.arange(out.input_ids.shape[1], dtype=torch.long)
    for axis in range(3):
        torch.testing.assert_close(out.position_ids_3d[axis, 0], ramp)
    print(f"text-only OK: input_ids shape {tuple(out.input_ids.shape)}")


@torch.no_grad()
def test_preprocessor_with_image():
    """Prompt with one PIL image — produces pixel_values + image_grid_thw + 3D positions."""
    proc = Qwen36MMPreprocessor("Qwen/Qwen3.6-27B")

    img = Image.new("RGB", (224, 224), color=(123, 45, 67))
    out = proc("<|vision_start|><|image_pad|><|vision_end|>Describe this", images=[img])

    assert out.pixel_values is not None
    # patch_feat_dim = in_channels * temporal_patch_size * patch_size**2 = 3*2*16*16 = 1536
    assert out.pixel_values.shape[-1] == 1536
    assert out.image_grid_thw is not None
    assert out.image_grid_thw.shape == (1, 3)
    assert out.image_grid_thw[0, 0].item() == 1  # T = 1 for image

    # Verify divergent axes for image positions
    # Find the first image_token_id position
    image_token_id = 248056
    img_pos = (out.input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    assert img_pos.numel() > 0, "expected image_pad tokens in input_ids"

    # At image positions, axes should NOT all be equal (3D grid coords)
    p = out.position_ids_3d[:, 0, img_pos]  # [3, num_image_tokens]
    print(f"image position_ids T-axis: {p[0].tolist()[:10]}")
    print(f"image position_ids H-axis: {p[1].tolist()[:10]}")
    print(f"image position_ids W-axis: {p[2].tolist()[:10]}")
    # At least one image position should have different H/W axes (a 2D grid coord)
    assert not (p[1] == p[2]).all(), "expected H/W axes to diverge at image positions"

    print(
        f"image-input OK: input_ids shape {tuple(out.input_ids.shape)}, "
        f"pixel_values shape {tuple(out.pixel_values.shape)}, "
        f"image_grid_thw {out.image_grid_thw.tolist()}"
    )
