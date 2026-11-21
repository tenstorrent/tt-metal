# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from models.demos.qwen25_vl.tt.common import merge_vision_tokens, reject_video_inputs

HF_CONFIG = SimpleNamespace(image_token_id=7, video_token_id=8)


def test_reject_video_inputs(expect_error):
    reject_video_inputs(None)
    reject_video_inputs([])
    with expect_error(ValueError, "video inputs are not supported"):
        reject_video_inputs([torch.zeros(2, 3, 4, 4)])


def test_merge_vision_tokens_rejects_video_placeholders(expect_error):
    input_ids = torch.tensor([[1, 8, 8, 2]])
    input_embeds = torch.zeros(1, 4, 16)
    image_embeds = torch.zeros(0, 16)
    with expect_error(ValueError, "Video placeholder tokens"):
        merge_vision_tokens(input_ids, input_embeds, image_embeds, HF_CONFIG)


def test_merge_vision_tokens_images_still_merge():
    input_ids = torch.tensor([[1, 7, 7, 2]])
    input_embeds = torch.zeros(1, 4, 16)
    image_embeds = torch.ones(2, 16)
    out = merge_vision_tokens(input_ids, input_embeds, image_embeds, HF_CONFIG)
    assert out[0, 1].eq(1).all() and out[0, 0].eq(0).all()
