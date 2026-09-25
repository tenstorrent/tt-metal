# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Encode-only smoke of the pipeline's sharded (TP + SP) vision tower with SP-alignment padding.

Keyframe grids are deliberately SP-misaligned so `pad_patches_for_sp` is live. Gates shape + finiteness only.
"""

import pytest
import torch
from PIL import Image

import ttnn

from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from .common import GALAXY_MESHES
from .common_av import weights_dir

HEIGHT, WIDTH = 768, 1344
PROMPT = "a fox jumps over a fence"


def _noise_image(seed: int) -> Image.Image:
    generator = torch.Generator().manual_seed(seed)
    return Image.fromarray((torch.rand(HEIGHT, WIDTH, 3, generator=generator) * 255).to(torch.uint8).numpy())


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("num_keyframes", [1, 2], ids=["one_keyframe", "two_keyframes"])
def test_encode_prompt_vision_sp_tower(mesh_device, num_keyframes):
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights_dir(), warmup=False)
    assert pipeline.sp_factor > 1, "this smoke exists to exercise the SP tower; the mesh has no SP axis"

    keyframes = [_noise_image(seed) for seed in range(num_keyframes)]
    embeds, tags = pipeline.encode_prompt(PROMPT, keyframes=keyframes)
    # encode_prompt returns the device tensor, replicated on every device; check one replica on host.
    embeds = ttnn.to_torch(ttnn.get_device_tensors(embeds)[0])

    assert embeds.ndim == 3 and embeds.shape[-1] == 5120, f"unexpected embeds shape {tuple(embeds.shape)}"
    # The embeds are padded to a whole SP * TILE multiple for the ring; the tags cover the real presentation only.
    sp_alignment = pipeline.sp_factor * ttnn.TILE_SIZE
    padded = -(-tags.shape[0] // sp_alignment) * sp_alignment
    assert embeds.shape[1] == padded, f"embeds seq {embeds.shape[1]} != tags {tags.shape[0]} padded to {padded}"
    embeds = embeds[:, : tags.shape[0]]
    assert torch.isfinite(embeds).all(), "prompt embeds contain NaN or Inf"
    assert embeds.shape[1] > num_keyframes * 1008, "presentation is missing the vision rows"
