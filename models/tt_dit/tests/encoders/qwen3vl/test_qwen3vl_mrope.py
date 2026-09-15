# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Qwen3-VL M-RoPE: the multimodal position grid and the interleaved layout,
# against the HF reference. Pure torch, no device. Text-only prompts make the
# chunked and interleaved layouts coincide; vision runs make the layout observable.

import pytest
import torch
import transformers

from ....encoders.qwen3vl.model_qwen3vl import create_rope_tensors, mrope_position_ids, vision_position_ids

# MiniMax-H3's conditioner: head_dim 128 (not hidden_size // num_heads), interleaved M-RoPE.
HEAD_DIM = 128
MROPE_SECTION = [24, 20, 20]
ROPE_THETA = 5_000_000.0
SPATIAL_MERGE_SIZE = 2


@pytest.fixture(scope="module")
def reference():
    config = transformers.Qwen3VLConfig(
        text_config={
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "num_heads": 2,
            "out_hidden_size": 64,
            "spatial_merge_size": SPATIAL_MERGE_SIZE,
        },
    )
    return transformers.AutoModel.from_config(config).eval()


_TIMESTAMP_TOKENS = 2  # frames are separated by timestamp text in the presentation, so each frame is its own run


def _prompt(*runs):
    """(mm_token_type_ids, image_grid_thw, video_grid_thw) from ("text", n) / ("image"|"video", (t, h, w)) runs."""
    type_ids, images, videos = [], [], []
    for kind, spec in runs:
        t, h, w = spec if kind != "text" else (0, 0, 0)
        if kind == "text":
            type_ids += [0] * spec
        elif kind == "image":
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


@pytest.mark.parametrize("name", list(PROMPTS))
def test_mrope_position_ids_matches_reference(reference, name):
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

    assert actual.shape == expected.shape, f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.equal(actual, expected), (
        f"{name}: position grid differs at "
        f"{(actual != expected).nonzero()[:5].tolist()}\n  ours={actual[:, 0, :12].tolist()}\n"
        f"  ref ={expected[:, 0, :12].tolist()}"
    )

    grids = ([] if image_grid is None else list(image_grid)) + ([] if video_grid is None else list(video_grid))
    for grid in grids:
        for start in (0, 7):
            block_expected = reference.get_vision_position_ids(start, grid, 1, SPATIAL_MERGE_SIZE, device=None)
            block_actual = vision_position_ids(start, grid, spatial_merge_size=SPATIAL_MERGE_SIZE)
            assert torch.equal(block_actual, block_expected), (
                f"{name} grid {grid.tolist()} start {start}: vision block grid differs\n"
                f"  ours={block_actual[:, :12].tolist()}\n  ref ={block_expected[:, :12].tolist()}"
            )


def test_text_only_layouts_are_identical():
    seq = 64
    chunked = create_rope_tensors(1, seq, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION, interleaved=False)
    interleaved = create_rope_tensors(1, seq, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION, interleaved=True)
    for a, b, which in zip(chunked, interleaved, ("cos", "sin")):
        assert torch.equal(a, b), f"text-only {which} differs between layouts"


@pytest.mark.parametrize("name", list(PROMPTS))
def test_interleaved_rope_tensors_match_reference(name):
    """Toleranced, not bitwise: theta**-x vs the reference's 1/theta**x inv_freq differ by one fp32 ulp."""
    text_config = transformers.Qwen3VLTextConfig(
        hidden_size=5120,
        num_attention_heads=64,
        head_dim=HEAD_DIM,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": ROPE_THETA,
            "mrope_section": MROPE_SECTION,
            "mrope_interleaved": True,
        },
    )
    rotary = transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding(text_config)

    type_ids, image_grid, video_grid = _prompt(*PROMPTS[name])
    seq = type_ids.shape[1]
    position_ids = mrope_position_ids(
        type_ids, image_grid_thw=image_grid, video_grid_thw=video_grid, spatial_merge_size=SPATIAL_MERGE_SIZE
    )

    expected_cos, expected_sin = rotary(torch.zeros(1, seq, text_config.hidden_size), position_ids)
    cos, sin = create_rope_tensors(
        1, seq, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION, position_ids=position_ids, interleaved=True
    )

    for actual, expected, which in ((cos, expected_cos, "cos"), (sin, expected_sin, "sin")):
        actual = actual.squeeze(1)
        assert actual.shape == expected.shape, f"{which}: {tuple(actual.shape)} != {tuple(expected.shape)}"
        max_diff = (actual - expected).abs().max().item()
        assert max_diff < 1e-5, f"{name} {which}: max abs diff {max_diff:.2e} exceeds the 1-ulp inv_freq budget"


def test_position_ids_default_is_the_shared_token_index():
    seq = 32
    implicit = create_rope_tensors(1, seq, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION)
    explicit = create_rope_tensors(
        1,
        seq,
        None,
        HEAD_DIM,
        ROPE_THETA,
        MROPE_SECTION,
        position_ids=torch.arange(seq).view(1, 1, -1).expand(3, 1, -1),
    )
    for a, b, which in zip(implicit, explicit, ("cos", "sin")):
        assert torch.equal(a, b), f"{which} differs between the implicit and explicit text grid"


def test_missing_grid_is_an_error(expect_error):
    type_ids, image_grid, _ = _prompt(*PROMPTS["one_image"])
    del image_grid
    with expect_error(ValueError, "no matching grid"):
        mrope_position_ids(type_ids, spatial_merge_size=SPATIAL_MERGE_SIZE)


def test_adjacent_same_modality_blocks_are_unsupported(expect_error):
    """The reference rejects this format too; do not "fix" ours to accept an input it cannot express."""
    type_ids = torch.tensor([[0, 0] + [1] * 4 + [1] * 4 + [0, 0]], dtype=torch.long)
    grids = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    with expect_error(RuntimeError, "must match the existing size"):
        mrope_position_ids(type_ids, image_grid_thw=grids, spatial_merge_size=SPATIAL_MERGE_SIZE)
