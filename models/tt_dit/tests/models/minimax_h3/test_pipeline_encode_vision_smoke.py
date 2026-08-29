# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Encode-only smoke of the PIPELINE's vision path -- the sharded (TP + SP) tower with
SP-alignment padding, through `MiniMaxH3Pipeline.encode_prompt` itself.

The conditioner tests validate the same composition but build the stages directly; nothing else
exercises the pipeline class's own tower construction and its padded, sharded input prep without
paying for a full generation. Keyframe grids here (48x84 -> 4,032 patches; two keyframes -> 8,064)
are deliberately NOT multiples of sp8's 256-row alignment, so `pad_patches_for_sp` is live: the
single-keyframe case routes single-block + phantom-pad -> windowed-SP, and the two-keyframe case
is multi-block windowed-SP with a pad window. Gates shape + finiteness (the conditioner tests own
PCC); this is a wiring gate, not a fidelity one.
"""

import pytest
import torch
from PIL import Image

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
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights_dir())
    assert pipeline.sp_factor > 1, "this smoke exists to exercise the SP tower; the mesh has no SP axis"

    keyframes = [_noise_image(seed) for seed in range(num_keyframes)]
    embeds, tags = pipeline.encode_prompt(PROMPT, keyframes=keyframes)

    assert embeds.ndim == 3 and embeds.shape[-1] == 5120, f"unexpected embeds shape {tuple(embeds.shape)}"
    assert embeds.shape[1] == tags.shape[0], f"embeds seq {embeds.shape[1]} != tags {tags.shape[0]}"
    assert torch.isfinite(embeds).all(), "prompt embeds contain NaN or Inf"
    # The vision rows must be present: each 48x84 keyframe grid contributes 1,008 merged tokens.
    assert embeds.shape[1] > num_keyframes * 1008, "presentation is missing the vision rows"
