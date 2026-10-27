# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Rotary position embeddings, against the transformers model each layout comes from.

Two halves, tested against the same prompts. `mrope_position_ids` assigns every token a temporal, a
height and a width position, and `RotaryEmbedding` turns those into cos and sin tables. A multimodal
layout gives each rotary pair to one of the three axes, and the layouts differ only in which pair
goes to which axis. Text positions agree on all three axes, so only an image tells them apart.
"""

from __future__ import annotations

import pytest
import torch
import transformers

import ttnn
from models.tt_dit.blocks.rope import RopeConfig, RotaryEmbedding
from models.tt_dit.encoders.qwen3vl.model_qwen3vl_v2 import mrope_position_ids, vision_position_ids
from models.tt_dit.utils import tensor

HEAD_SIZE = 128

CHUNKED_SECTION = [16, 24, 24]  # Qwen2.5-VL
CHUNKED_THETA = 1_000_000.0

INTERLEAVED_SECTION = [24, 20, 20]  # MiniMax-H3's conditioner
INTERLEAVED_THETA = 5_000_000.0

SPATIAL_MERGE_SIZE = 2

_TIMESTAMP_TOKENS = 2  # frames are separated by timestamp text in the presentation, so each frame is its own run


@pytest.fixture(scope="module")
def reference() -> transformers.Qwen3VLModel:
    """A Qwen3-VL to take the reference position grid from.

    Only `spatial_merge_size` affects that grid. The remaining sizes are arbitrary, and small so
    that building the model is cheap.
    """
    config = transformers.Qwen3VLConfig(
        text_config={
            "num_hidden_layers": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "num_heads": 2,
            "out_hidden_size": 64,
            "spatial_merge_size": SPATIAL_MERGE_SIZE,
        },
    )
    return transformers.Qwen3VLModel(config)


def _prompt(
    *runs: tuple[str, int | tuple[int, int, int]]
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """(mm_token_type_ids, image_grid_thw, video_grid_thw) from ("text", n) / ("image"|"video", (t, h, w)) runs."""
    type_ids, images, videos = [], [], []
    for kind, spec in runs:
        if kind == "text":
            assert isinstance(spec, int)
            type_ids += [0] * spec
            continue

        assert isinstance(spec, tuple)
        t, h, w = spec
        if kind == "image":
            type_ids += [1] * (t * h * w // (SPATIAL_MERGE_SIZE**2))
            images.append([t, h, w])
        else:
            videos.append([t, h, w])
            for _ in range(t):
                type_ids += [0] * _TIMESTAMP_TOKENS
                type_ids += [2] * (h * w // (SPATIAL_MERGE_SIZE**2))
    return (
        torch.tensor([type_ids], dtype=torch.long),
        torch.tensor(images, dtype=torch.long) if images else None,
        torch.tensor(videos, dtype=torch.long) if videos else None,
    )


PROMPTS = {
    "text_only": (("text", 24),),
    "one_image": (("text", 6), ("image", (1, 4, 6)), ("text", 5)),
    "two_images": (("text", 3), ("image", (1, 4, 4)), ("text", 2), ("image", (1, 6, 2)), ("text", 4)),
    "image_first": (("image", (1, 2, 2)), ("text", 8)),
    "one_video": (("text", 4), ("video", (2, 4, 4)), ("text", 3)),
    "video_and_image": (("text", 2), ("video", (2, 2, 4)), ("text", 3), ("image", (1, 4, 4)), ("text", 2)),
    "video_3_frames": (("text", 2), ("video", (3, 4, 2)), ("text", 2)),
    # production fl2va working point; rope tables only hit rounding boundaries at production length
    "keyframe_768x1344": (("text", 5), ("image", (1, 48, 84)), ("text", 41)),
    "two_keyframes_768x1344": (
        ("text", 5),
        ("image", (1, 48, 84)),
        ("text", 5),
        ("image", (1, 48, 84)),
        ("text", 41),
    ),
}


def _positions(name: str) -> torch.Tensor:
    """`(3, 1, sequence)` temporal, height and width positions of one entry of [`PROMPTS`]."""
    type_ids, image_grid, video_grid = _prompt(*PROMPTS[name])
    return mrope_position_ids(
        type_ids, image_grid_thw=image_grid, video_grid_thw=video_grid, spatial_merge_size=SPATIAL_MERGE_SIZE
    )


@pytest.mark.parametrize("name", list(PROMPTS))
def test_mrope_position_ids_matches_reference(reference: transformers.Qwen3VLModel, name: str) -> None:
    type_ids, image_grid, video_grid = _prompt(*PROMPTS[name])
    input_ids = torch.zeros_like(type_ids)

    expected, _ = reference.get_rope_index(
        input_ids,
        mm_token_type_ids=type_ids,
        image_grid_thw=None if image_grid is None else image_grid.clone(),
        video_grid_thw=None if video_grid is None else video_grid.clone(),
    )
    actual = mrope_position_ids(
        type_ids,
        image_grid_thw=image_grid,
        video_grid_thw=video_grid,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
    )

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)

    grids = ([] if image_grid is None else list(image_grid)) + ([] if video_grid is None else list(video_grid))
    for grid in grids:
        for start in (0, 7):
            block_expected = reference.get_vision_position_ids(start, grid, 1, SPATIAL_MERGE_SIZE, device=None)
            block_actual = vision_position_ids(start, grid, spatial_merge_size=SPATIAL_MERGE_SIZE)
            assert torch.equal(block_actual, block_expected)


def test_missing_grid_is_an_error(expect_error) -> None:
    type_ids, _, _ = _prompt(*PROMPTS["one_image"])
    with expect_error(ValueError, "no matching grid"):
        mrope_position_ids(type_ids, spatial_merge_size=SPATIAL_MERGE_SIZE)


def test_adjacent_same_modality_blocks_are_unsupported(expect_error) -> None:
    """The reference rejects this format too: two images need text between them."""
    type_ids = torch.tensor([[0, 0] + [1] * 4 + [1] * 4 + [0, 0]], dtype=torch.long)
    grids = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    with expect_error(ValueError, "does not match"):
        mrope_position_ids(type_ids, image_grid_thw=grids, spatial_merge_size=SPATIAL_MERGE_SIZE)


def _chunked_reference(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    config = transformers.Qwen2_5_VLTextConfig(
        hidden_size=HEAD_SIZE,
        num_attention_heads=1,
        num_key_value_heads=1,
        rope_parameters={"rope_type": "default", "rope_theta": CHUNKED_THETA, "mrope_section": CHUNKED_SECTION},
    )
    rotary = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding(config)

    def select_axes(x: torch.Tensor) -> torch.Tensor:
        # The reference carries all three axes through the cosine and picks the owner of each
        # section afterwards, where we zero the frequencies of the other two axes first.
        # https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L589
        return torch.cat([t[i % 3] for i, t in enumerate(x.split(CHUNKED_SECTION * 2, dim=-1))], dim=-1)

    cos, sin = rotary(torch.zeros(1), positions)
    return select_axes(cos), select_axes(sin)


def _interleaved_reference(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    config = transformers.Qwen3VLTextConfig(
        hidden_size=HEAD_SIZE,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=HEAD_SIZE,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": INTERLEAVED_THETA,
            "mrope_section": INTERLEAVED_SECTION,
            "mrope_interleaved": True,
        },
    )
    rotary = transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding(config)
    return rotary(torch.zeros(1), positions)


LAYOUTS = {
    "chunked": (RopeConfig(theta=CHUNKED_THETA, mrope_section=CHUNKED_SECTION), _chunked_reference),
    "interleaved": (
        RopeConfig(theta=INTERLEAVED_THETA, mrope_section=INTERLEAVED_SECTION, mrope_interleaved=True),
        _interleaved_reference,
    ),
}


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("name", list(PROMPTS))
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_multimodal_tensors_match_reference(mesh_device: ttnn.MeshDevice, layout: str, name: str) -> None:
    config, reference_tables = LAYOUTS[layout]

    positions = _positions(name)
    expected_cos, expected_sin = reference_tables(positions)

    rope = RotaryEmbedding(head_size=HEAD_SIZE, config=config)
    tt_positions = [tensor.from_torch(p.float(), device=mesh_device, dtype=ttnn.float32) for p in positions]
    cos, sin = rope.forward(tt_positions, dtype=ttnn.float32)

    for actual, expected in ((cos, expected_cos), (sin, expected_sin)):
        actual = tensor.to_torch(actual)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-2)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_agreeing_axes_are_one_position_tensor(mesh_device: ttnn.MeshDevice, layout: str) -> None:
    """Every rotary pair belongs to one axis, so agreeing axes sum back to the whole table."""
    config, _ = LAYOUTS[layout]

    rope = RotaryEmbedding(head_size=HEAD_SIZE, config=config)
    positions = tensor.from_torch(_positions("text_only")[0].float(), device=mesh_device, dtype=ttnn.float32)

    cos, sin = rope.forward([positions] * 3, dtype=ttnn.float32)
    cos_1d, sin_1d = rope.forward(positions, dtype=ttnn.float32)

    assert torch.equal(tensor.to_torch(cos_1d), tensor.to_torch(cos))
    assert torch.equal(tensor.to_torch(sin_1d), tensor.to_torch(sin))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_tensors_match_reference(mesh_device: ttnn.MeshDevice) -> None:
    """The plain layout of a text-only model, where there are no sections to assign."""
    config = transformers.LlamaConfig(
        hidden_size=HEAD_SIZE,
        num_attention_heads=1,
        num_key_value_heads=1,
        rope_parameters={"rope_type": "default", "rope_theta": CHUNKED_THETA},
    )
    rotary = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding(config)

    positions = _positions("text_only")[0]
    expected_cos, expected_sin = rotary(torch.zeros(1), positions)

    rope = RotaryEmbedding(head_size=HEAD_SIZE, config=RopeConfig(theta=CHUNKED_THETA))
    tt_positions = tensor.from_torch(positions.float(), device=mesh_device, dtype=ttnn.float32)
    cos, sin = rope.forward(tt_positions, dtype=ttnn.float32)

    for actual, expected in ((cos, expected_cos), (sin, expected_sin)):
        actual = tensor.to_torch(actual)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-2)
