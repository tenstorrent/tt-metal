# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from models.experimental.audiox.demo.media import prepare_video_prompt


def test_prepare_video_prompt_resamples_and_resizes():
    frames = torch.arange(3 * 3 * 8 * 12, dtype=torch.float32).reshape(3, 3, 8, 12)
    prompt = prepare_video_prompt(frames, target_frames=5, image_size=16)
    assert prompt.shape == (1, 5, 3, 16, 16)


def test_prepare_video_prompt_repeats_single_frame():
    frame = torch.ones(1, 3, 10, 10)
    prompt = prepare_video_prompt(frame, target_frames=4, image_size=10)
    assert prompt.shape == (1, 4, 3, 10, 10)
    assert torch.all(prompt == 1.0)
