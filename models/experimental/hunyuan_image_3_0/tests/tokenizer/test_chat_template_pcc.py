# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer tests for T2I chat template and host preprocess bundle."""

from __future__ import annotations

import torch

from models.experimental.hunyuan_image_3_0.ref.tokenizer import prepare_gen_image_inputs


PROMPT = "a cat on a mat"
IMAGE_SIZE = 1024


def test_cfg_uncond_replaces_text_not_image(hunyuan_tokenizer):
    out = hunyuan_tokenizer.apply_chat_template(PROMPT, image_size=IMAGE_SIZE, cfg_factor=2)["output"]
    assert out.tokens.shape[0] == 2
    cfg_id = hunyuan_tokenizer.special.cfg_token_id
    boi_id = hunyuan_tokenizer.special.boi_token_id

    cond, uncond = out.tokens[0], out.tokens[1]
    boi_idx = (cond == boi_id).nonzero(as_tuple=False)[0].item()
    assert (uncond[:boi_idx] == cfg_id).any()
    assert torch.equal(cond[boi_idx:], uncond[boi_idx:])
    assert torch.equal(out.gen_image_mask[0], out.gen_image_mask[1])
    assert torch.equal(out.gen_timestep_scatter_index[0], out.gen_timestep_scatter_index[1])


def test_host_preprocess_bundle(hunyuan_tokenizer):
    bundle = prepare_gen_image_inputs(hunyuan_tokenizer, PROMPT, image_size=IMAGE_SIZE, cfg_factor=2)
    assert bundle.input_ids.shape == (2, bundle.seq_len)
    assert bundle.position_ids.shape == bundle.input_ids.shape
    assert bundle.rope_image_info is not None
    assert len(bundle.rope_image_info) == 2
    slice_i, (th, tw) = bundle.rope_image_info[0][0]
    assert th * tw == 4096
    assert slice_i.stop - slice_i.start == 4096
