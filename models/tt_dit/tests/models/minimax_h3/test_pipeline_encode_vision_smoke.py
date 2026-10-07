# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Encode-only smoke of the pipeline's sharded (TP + SP) vision tower with SP-alignment padding.

Keyframe grids are deliberately SP-misaligned so `pad_patches_for_sp` is live. Gates shape + finiteness only.
"""

import pytest
import torch
from PIL import Image

from ....pipelines.minimax_h3.packing import MINIMAX_H3_CANVAS_MULTIPLE
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.policy import decodable_canvases
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

    assert embeds.ndim == 3 and embeds.shape[-1] == 5120, f"unexpected embeds shape {tuple(embeds.shape)}"
    assert embeds.shape[1] == tags.shape[0], f"embeds seq {embeds.shape[1]} != tags {tags.shape[0]}"
    assert torch.isfinite(embeds).all(), "prompt embeds contain NaN or Inf"
    assert embeds.shape[1] > num_keyframes * 1008, "presentation is missing the vision rows"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_encode_prompt_vision_ring_shares_kv_pair(mesh_device):
    """Two single-keyframe canvases with distinct SP-aligned patch counts both take the tower's ring SDPA."""
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights_dir(), warmup=False)
    assert pipeline.sp_factor > 1, "this smoke exists to exercise the SP tower; the mesh has no SP axis"

    alignment = pipeline.sp_factor * 32
    by_patches = {}
    for height, width in decodable_canvases():
        patches = 4 * (height // MINIMAX_H3_CANVAS_MULTIPLE) * (width // MINIMAX_H3_CANVAS_MULTIPLE)
        if patches % alignment == 0:
            by_patches.setdefault(patches, (height, width))
    assert len(by_patches) >= 2, f"need two SP-aligned keyframe canvases at alignment {alignment}"
    canvases = [by_patches[patches] for patches in sorted(by_patches)[:2]]

    for height, width in canvases:
        pipeline.encode_prompt(PROMPT, keyframes=[Image.new("RGB", (width, height), (127, 127, 127))])

    head_dim = pipeline._vision_tower.blocks[0].attn.padded_head_dim
    tower_pairs = [
        key for key in pipeline.encoder_ccl_manager._ping_pong_buffer_cache if key[0] == "ag" and key[1][-1] == head_dim
    ]
    assert len(tower_pairs) == 1, f"canvases {canvases} pinned {len(tower_pairs)} tower K/V pairs: {tower_pairs}"
