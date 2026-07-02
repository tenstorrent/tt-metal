# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision-RoPE parity: TT ``Gemma4VisionRotaryEmbedding`` vs HF
``Gemma4VisionRotaryEmbedding``.

HF returns full-dim cos/sin shape ``[B, num_patches, head_dim]`` with layout
``[cos_x_full(36), cos_y_full(36)]``. We return half-dim cos/sin shape
``[B, 1, num_patches, head_dim_padded/2]`` with layout
``[cos_x_half(18), cos_y_half(18), pad(head_dim_padded/2 - 36)=1]``.

We assert:
  * Our ``cos[..., :18]`` matches HF ``cos[..., :18]`` (unique x-half).
  * Our ``cos[..., 18:36]`` matches HF ``cos[..., 36:54]`` (unique y-half).
  * Our ``cos[..., 36:]`` is all 1, ``sin[..., 36:]`` all 0 (identity padding).

    pytest models/tt_dit/tests/encoders/gemma4/test_vision_rope.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....encoders.gemma4.vision_rope import Gemma4VisionRotaryEmbedding
from ....utils.check import assert_quality
from ....utils.test import line_params

PCC_THRESHOLD = 0.9999
ALLCLOSE_ATOL = 1e-2
ALLCLOSE_RTOL = 1e-2


@pytest.mark.parametrize(
    ("mesh_device", "num_links", "device_params"),
    [pytest.param((1, 1), 1, line_params, id="single")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_patches", [64])
def test_vision_rope(mesh_device: ttnn.MeshDevice, num_links: int, num_patches: int) -> None:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding as HFVisionRotary

    torch.manual_seed(0)

    head_dim = 72
    head_dim_padded = 96
    position_embedding_size = 256
    rope_theta = 100.0

    hf_config = Gemma4VisionConfig(
        hidden_size=1152,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=head_dim,
        rope_parameters={"rope_theta": rope_theta, "rope_type": "default"},
        position_embedding_size=position_embedding_size,
        patch_size=16,
    )

    # Build random 2-D position ids in range [0, position_embedding_size).
    B = 1
    px = torch.randint(0, position_embedding_size, (B, num_patches), dtype=torch.long)
    py = torch.randint(0, position_embedding_size, (B, num_patches), dtype=torch.long)
    position_ids = torch.stack([px, py], dim=-1)

    # HF reference.
    hf_rope = HFVisionRotary(hf_config).eval()
    dummy_x = torch.zeros(B, num_patches, head_dim)
    with torch.no_grad():
        hf_cos_full, hf_sin_full = hf_rope(dummy_x, position_ids)
    # hf_*_full shape: (B, num_patches, head_dim=72)
    assert hf_cos_full.shape == (B, num_patches, head_dim)

    # Our module.
    tt_rope = Gemma4VisionRotaryEmbedding(
        head_dim=head_dim,
        head_dim_padded=head_dim_padded,
        position_embedding_size=position_embedding_size,
        rope_theta=rope_theta,
        mesh_device=mesh_device,
    )
    tt_cos, tt_sin = tt_rope.get_cos_sin(position_ids)
    tt_cos_torch = ttnn.to_torch(tt_cos).float().squeeze(1)  # (B, P, head_dim_padded/2)
    tt_sin_torch = ttnn.to_torch(tt_sin).float().squeeze(1)
    assert tt_cos_torch.shape == (B, num_patches, head_dim_padded // 2)

    # Slices: first 18 = x-half, next 18 = y-half, rest = padding.
    unique_per_dim = head_dim // 4  # 18
    our_cos_x = tt_cos_torch[..., :unique_per_dim]
    our_cos_y = tt_cos_torch[..., unique_per_dim : 2 * unique_per_dim]
    our_cos_pad = tt_cos_torch[..., 2 * unique_per_dim :]
    our_sin_pad = tt_sin_torch[..., 2 * unique_per_dim :]

    # HF: cos[..., :18] = unique x-half; cos[..., 36:54] = unique y-half.
    hf_cos_x = hf_cos_full[..., :unique_per_dim]
    hf_cos_y = hf_cos_full[..., 2 * unique_per_dim : 3 * unique_per_dim]
    hf_sin_x = hf_sin_full[..., :unique_per_dim]
    hf_sin_y = hf_sin_full[..., 2 * unique_per_dim : 3 * unique_per_dim]

    logger.info(f"shapes: our_cos {tt_cos_torch.shape}, hf_cos {hf_cos_full.shape}, " f"slice tests on cos_x/y and pad")

    assert_quality(hf_cos_x, our_cos_x, pcc=PCC_THRESHOLD)
    assert_quality(hf_cos_y, our_cos_y, pcc=PCC_THRESHOLD)
    assert_quality(hf_sin_x, tt_sin_torch[..., :unique_per_dim], pcc=PCC_THRESHOLD)
    assert_quality(hf_sin_y, tt_sin_torch[..., unique_per_dim : 2 * unique_per_dim], pcc=PCC_THRESHOLD)

    assert torch.allclose(hf_cos_x, our_cos_x, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
    assert torch.allclose(hf_cos_y, our_cos_y, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
    # Padding entries: cos = 1, sin = 0.
    assert torch.allclose(our_cos_pad, torch.ones_like(our_cos_pad), atol=ALLCLOSE_ATOL)
    assert torch.allclose(our_sin_pad, torch.zeros_like(our_sin_pad), atol=ALLCLOSE_ATOL)
