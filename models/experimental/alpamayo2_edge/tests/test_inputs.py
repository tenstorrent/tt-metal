# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tokenizers import Tokenizer

from models.experimental.alpamayo2_edge.reference.chat import relative_time_labels
from models.experimental.alpamayo2_edge.reference.inputs import (
    normalize_patches,
    patchify,
    prepare_inputs,
    prepare_sample_inputs,
    synthetic_sample,
)

WIDE_CAMERAS = (1, 6)


@pytest.fixture(scope="module")
def sample():
    return synthetic_sample(0)


@pytest.fixture(scope="module")
def inputs(cfg, sample):
    return prepare_sample_inputs(cfg, sample)


@pytest.fixture(scope="module")
def prompt_text(model_dir, inputs):
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return tokenizer.decode(inputs.input_ids[0].tolist(), skip_special_tokens=False)


def test_synthetic_sample_layout(sample):
    assert sample["image_frames"].shape == (22, 1, 3, 1080, 1920)
    assert sample["image_frames"].dtype == torch.uint8
    assert sample["camera_indices"].tolist() == [0] * 3 + [1] * 5 + [2] * 3 + [3] * 3 + [5] * 3 + [6] * 5
    assert sample["absolute_timestamps"].dtype == torch.int64
    assert sample["ego_history_xyz"].shape == (1, 20, 3)
    assert sample["ego_history_rot"].shape == (1, 20, 3, 3)
    assert sample["nav_text"] == ["Continue straight."]
    assert torch.equal(synthetic_sample(0)["image_frames"], sample["image_frames"])


def test_grids_16x9(sample, inputs):
    expected = [(1, 20, 36) if cam in WIDE_CAMERAS else (1, 14, 26) for cam in sample["camera_indices"].tolist()]
    assert [tuple(g) for g in inputs.image_grid_thw.tolist()] == expected
    n = inputs.patches.shape[0]
    print(f"N = {n} patches, {n // 4} vision tokens, S = {inputs.input_ids.shape[1]}")
    assert inputs.patches.shape == (11568, 768)
    assert inputs.patches.dtype == torch.float32
    assert n // 4 == 2892


def test_image_tokens_match(cfg, inputs):
    image_tokens = inputs.input_ids == cfg.tokens.image
    assert int(image_tokens.sum()) == inputs.patches.shape[0] // 4
    assert torch.equal(inputs.mm_token_type_ids, image_tokens.long())
    assert int((inputs.input_ids == cfg.tokens.vision_start).sum()) == 22
    assert int((inputs.input_ids == cfg.tokens.vision_end).sum()) == 22


def test_traj_tokens_in_range(cfg, inputs):
    ids = inputs.input_ids[0]
    start = (ids == cfg.tokens.history_start).nonzero().flatten()
    end = (ids == cfg.tokens.history_end).nonzero().flatten()
    assert start.numel() == 1 and end.numel() == 1
    history = ids[start + 1 : end]
    assert history.numel() == 57
    assert bool(((history >= 131072) & (history < 132072)).all())
    assert int((ids == cfg.tokens.history_pad).sum()) == 0
    # 1 m forward per 0.1 s step, quantized over [-4, 4] x [-4, 4] x [-10, 10] m into 1000 bins.
    assert history.tolist() == [131072 + 624, 131072 + 500, 131072 + 500] * 19


def test_rope_deltas_shape(inputs):
    s = inputs.input_ids.shape[1]
    assert inputs.position_ids.shape == (3, 1, s)
    assert inputs.rope_deltas.shape == (1, 1)
    assert inputs.attention_mask.shape == (1, s)
    assert int(inputs.rope_deltas) == int(inputs.position_ids.max()) + 1 - s


def test_prompt_text(prompt_text):
    # The checkpoint template emits a newline after its macro definition; transformers renders it the same way.
    assert prompt_text.startswith(
        "\n<|im_start|>system\nYou are a driving assistant that generates safe and accurate actions.<|im_end|>\n"
        "<|im_start|>user\nFront left camera: <t=-1.0s><|vision_start|><|image_pad|>"
    )
    assert "<|vision_end|>Front camera: <t=-2.0s><|vision_start|>" in prompt_text
    assert "<|traj_history_end|>\nNavigation input is: Continue straight.\nYour task: " in prompt_text
    assert prompt_text.endswith("then output the future trajectory.<|im_end|>\n<|im_start|>assistant\n<think></think>")


def test_relative_time_labels_round_negative_zero():
    labels = relative_time_labels(torch.tensor([-2_000_000, -40_000, -1, 0]))
    assert labels == ["<t=-2.0s>", "<t=0.0s>", "<t=0.0s>", "<t=0.0s>"]


def test_patchify_block_major_hwc():
    image = torch.arange(3 * 64 * 96, dtype=torch.int32).reshape(3, 64, 96)
    patches, grid = patchify(image, patch=16, merge=2)
    assert grid == (4, 6)
    assert patches.shape == (24, 768)
    # Row order walks each 2x2 merge block before the next block; values are (y, x, channel) within a patch.
    for row, (py, px) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3)]):
        expected = image[:, py * 16 : py * 16 + 16, px * 16 : px * 16 + 16].permute(1, 2, 0).reshape(-1)
        assert torch.equal(patches[row], expected)
    assert torch.equal(patches[12], image[:, 32:48, :16].permute(1, 2, 0).reshape(-1))


def test_normalize_patches():
    raw = torch.tensor([[0, 255, 128] * 256], dtype=torch.uint8)
    out = normalize_patches(raw, mean=[0.5] * 3, std=[0.5] * 3, rescale_factor=1 / 255)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out[0, :3], torch.tensor([-1.0, 1.0, 0.5 / 127.5]))


def test_rejects_wrong_camera_profile(cfg, sample, expect_error):
    frames = [frame[0].permute(1, 2, 0).numpy() for frame in sample["image_frames"][1:]]
    with expect_error(ValueError, "frames per camera"):
        prepare_inputs(
            cfg,
            frames,
            sample["camera_indices"][1:].tolist(),
            sample["absolute_timestamps"][1:].tolist(),
            sample["ego_history_xyz"],
            sample["ego_history_rot"],
            "Continue straight.",
        )
