# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL M-RoPE: the multimodal position grid and the interleaved layout.
#
# A text-only prompt gives all three (t, h, w) axes the same position, which
# makes the chunked [TTT..HHH..WWW] and interleaved [THWTHW..] layouts coincide
# exactly -- the two differ only in which axis feeds each frequency slot, never
# in the frequency itself. Vision runs break that: an image block carries
# genuinely different t/h/w positions, so the layout becomes observable and the
# chunked path stops reproducing a checkpoint that declares
# `rope_scaling.mrope_interleaved` (MiniMax-H3's conditioner does).
#
# This is the groundwork for the vision tower: it fixes the position grid and the
# rotary layout against the HF reference before anything is built on top. Pure
# torch, no device -- it runs anywhere.
#
# Scope: synthetic `mm_token_type_ids` and grids, which is exactly what
# `get_rope_index` consumes (it reads nothing else). Wiring these up from a real
# `Qwen3VLProcessor` belongs with the fused-conditioner test.
# =============================================================================

import itertools

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
    """A minimal `Qwen3VLModel` for its `get_rope_index` / `get_vision_position_ids`.

    Both are pure index arithmetic -- they read only `config.vision_config.spatial_merge_size` and no
    weights -- so the stack is shrunk to keep the fixture cheap.
    """
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


# A video's frames are separated by timestamp text (`<0.2 seconds>`) in the presentation, so each
# frame is its own run rather than one contiguous block. Two tokens stand in for that text.
_TIMESTAMP_TOKENS = 2


def _prompt(*runs):
    """`(mm_token_type_ids, image_grid_thw, video_grid_thw)` for a sequence of runs.

    Each run is `("text", n)`, `("image", (t, h, w))` or `("video", (t, h, w))`.

    Two format rules are modelled, because the reference relies on both and raises without them:

    - A vision block occupies `h * w // spatial_merge_size**2` token slots per frame.
    - Same-modality blocks are never adjacent. `get_rope_index` groups the sequence with `groupby`, so
      two touching blocks merge into ONE run while still consuming TWO grid entries -- the positions
      then come up short. Real prompts always interpose text: MiniMax-H3 emits a `"<Picture i>: "`
      label before each image and a `"<t> seconds"` stamp before each video frame, and Qwen3-VL's own
      chat template does the same. A video is therefore emitted here as one run per frame.
    """
    type_ids, images, videos = [], [], []
    for kind, spec in runs:
        t, h, w = spec if kind != "text" else (0, 0, 0)
        if kind == "text":
            type_ids += [0] * spec
        elif kind == "image":
            type_ids += [1] * (t * h * w // (SPATIAL_MERGE_SIZE**2))
            images.append([t, h, w])
        else:
            # One grid for the whole video; the reference splits it per frame internally.
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
}


@pytest.mark.parametrize("name", list(PROMPTS))
def test_mrope_position_ids_matches_reference(reference, name):
    """The `(3, batch, seq)` position grid matches `Qwen3VLModel.get_rope_index` exactly."""
    type_ids, image_grid, video_grid = _prompt(*PROMPTS[name])
    input_ids = torch.zeros_like(type_ids)  # get_rope_index reads only shape and the type ids

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


@pytest.mark.parametrize(
    "grid", [(1, 4, 6), (1, 2, 2), (2, 4, 4), (3, 6, 2)], ids=["img_4x6", "img_2x2", "vid_2f", "vid_3f"]
)
@pytest.mark.parametrize("start", [0, 7])
def test_vision_position_ids_matches_reference(reference, grid, start):
    """One vision block's `(3, n)` grid matches `get_vision_position_ids`."""
    expected = reference.get_vision_position_ids(start, torch.tensor(grid), 1, SPATIAL_MERGE_SIZE, device=None)
    actual = vision_position_ids(start, torch.tensor(grid), spatial_merge_size=SPATIAL_MERGE_SIZE)
    assert torch.equal(actual, expected), f"ours={actual.tolist()}\nref ={expected.tolist()}"


def test_text_only_layouts_are_identical():
    """With all three axes on the same position the two layouts coincide bitwise.

    This is the invariant the MiniMax-H3 conditioner test relies on to use the chunked path for
    `t2va`; asserting it here pins it on our own implementation rather than on the reference's.
    """
    seq = 64
    chunked = create_rope_tensors(1, seq, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION, interleaved=False)
    interleaved = create_rope_tensors(1, seq, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION, interleaved=True)
    for a, b, which in zip(chunked, interleaved, ("cos", "sin")):
        assert torch.equal(a, b), f"text-only {which} differs between layouts"


def test_vision_makes_the_layout_observable(reference):
    """With a vision run the two layouts genuinely diverge.

    The complement of the test above: it is what makes `interleaved` load-bearing rather than
    cosmetic, and it is why a checkpoint declaring `mrope_interleaved` cannot use the chunked path
    once pixels are present.
    """
    type_ids, image_grid, _ = _prompt(*PROMPTS["one_image"])
    seq = type_ids.shape[1]
    position_ids = mrope_position_ids(type_ids, image_grid_thw=image_grid, spatial_merge_size=SPATIAL_MERGE_SIZE)
    # the axes really do disagree, or this test would prove nothing
    assert not torch.equal(position_ids[0], position_ids[1]) or not torch.equal(position_ids[1], position_ids[2])

    chunked = create_rope_tensors(
        1, seq, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION, position_ids=position_ids, interleaved=False
    )
    interleaved = create_rope_tensors(
        1, seq, None, HEAD_DIM, ROPE_THETA, MROPE_SECTION, position_ids=position_ids, interleaved=True
    )
    assert not torch.equal(chunked[0], interleaved[0]), "cos is identical despite a vision run"
    assert not torch.equal(chunked[1], interleaved[1]), "sin is identical despite a vision run"


@pytest.mark.parametrize("name", list(PROMPTS))
def test_interleaved_rope_tensors_match_reference(name):
    """`create_rope_tensors(interleaved=True)` reproduces HF's rotary embedding for the same grid.

    Compared with a tolerance rather than bitwise: `inv_freq` is computed as `theta ** -x` here and
    `1 / theta ** x` in the reference. Those are mathematically equal but one fp32 ulp apart, so the
    outputs agree to ~1e-7 and not exactly.
    """
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
    """Passing no `position_ids` is the same as passing the token index on all three axes."""
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
    """A vision run with no matching grid is a caller error, not a silent wrong answer."""
    type_ids, image_grid, _ = _prompt(*PROMPTS["one_image"])
    del image_grid
    with expect_error(ValueError, "no matching grid"):
        mrope_position_ids(type_ids, spatial_merge_size=SPATIAL_MERGE_SIZE)


def test_video_frames_are_separate_runs():
    """A multi-frame video is one grid but one *run per frame*, split by timestamp text.

    Pins the format rule the fixture models. `video_grid_thw` stays `[[t, h, w]]` -- the split into t
    grids of `t=1` happens inside [`mrope_position_ids`], mirroring the reference.
    """
    type_ids, _, video_grid = _prompt(*PROMPTS["video_3_frames"])
    runs = [k for k, _ in itertools.groupby(type_ids[0].tolist())]
    assert runs == [0, 2, 0, 2, 0, 2, 0], f"expected one run per frame, got {runs}"
    assert video_grid.tolist() == [[3, 4, 2]], "the video is still a single grid entry"


def test_adjacent_same_modality_blocks_are_unsupported(expect_error):
    """Two touching vision blocks are a format the reference cannot express either.

    `groupby` merges them into one run while two grid entries remain to be consumed, so the positions
    come up short and the reference raises. Recorded so that nobody "fixes" our implementation to
    accept an input the reference rejects -- real presentations always interpose a label or timestamp.
    """
    type_ids = torch.tensor([[0, 0] + [1] * 4 + [1] * 4 + [0, 0]], dtype=torch.long)
    grids = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    with expect_error(RuntimeError, "must match the existing size"):
        mrope_position_ids(type_ids, image_grid_thw=grids, spatial_merge_size=SPATIAL_MERGE_SIZE)
