# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Encode-only smoke of the pipeline's sharded (TP + SP) vision tower with SP-alignment padding.

Keyframe grids are deliberately SP-misaligned so `pad_patches_for_sp` is live. Gates shape + finiteness only.
"""

import pytest
import torch
from PIL import Image

import ttnn

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

    # The embeds stay on device as [1, seq_len, 5120] with seq_len padded up to the SP alignment bucket the
    # presentation lands in; every device's shard must be finite.
    alignment = pipeline.sp_factor * ttnn.TILE_SIZE
    shape = tuple(embeds.shape)
    assert len(shape) == 3 and shape[-1] == 5120, f"unexpected embeds shape {shape}"
    expected_seq = -(-tags.shape[0] // alignment) * alignment
    assert shape[1] == expected_seq, f"embeds seq {shape[1]} != {tags.shape[0]} tags padded to {expected_seq}"
    for index, shard in enumerate(ttnn.get_device_tensors(embeds)):
        assert torch.isfinite(ttnn.to_torch(shard)).all(), f"prompt embeds contain NaN or Inf on device {index}"
    assert shape[1] > num_keyframes * 1008, "presentation is missing the vision rows"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_encode_prompt_vision_ring_shares_kv_pair(mesh_device):
    """Two single-keyframe canvases with distinct SP-aligned patch counts both take the tower's ring SDPA and
    share one K/V ping-pong pair while they encode; `encode_prompt` then leaves no pair behind."""
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

    manager = pipeline.encoder_ccl_manager

    def tower_pairs():
        # The tower is built on the first encode, so read its head dim only once an encode has run.
        head_dim = pipeline._vision_tower.blocks[0].attn.padded_head_dim
        return [key for key in manager._ping_pong_buffer_cache if key[0] == "ag" and key[1][-1] == head_dim]

    # The pair is keyed by the tower's gather shape, not by the canvas, so both encodes reuse the one pair.
    # `encode_prompt` frees its pairs on return, so run the encode body under one scope of our own to look.
    with manager.transient_ping_pong_buffers():
        for height, width in canvases:
            pipeline._encode_prompt_device(PROMPT, keyframes=[Image.new("RGB", (width, height), (127, 127, 127))])
        assert (
            len(tower_pairs()) == 1
        ), f"canvases {canvases} pinned {len(tower_pairs())} tower K/V pairs: {tower_pairs()}"
    assert not tower_pairs(), f"the transient scope left tower K/V pairs behind: {tower_pairs()}"

    # The public entry point owns the same scope: the pair must not outlive the call.
    height, width = canvases[0]
    pipeline.encode_prompt(PROMPT, keyframes=[Image.new("RGB", (width, height), (127, 127, 127))])
    assert not tower_pairs(), f"encode_prompt left tower K/V pairs behind: {tower_pairs()}"
