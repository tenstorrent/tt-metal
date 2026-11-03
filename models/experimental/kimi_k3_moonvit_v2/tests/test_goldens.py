# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU: what `goldens.py` uses to generate the saved full-size outputs matches the reference on small
inputs: `encoder_steps` is the reference encoder with every block's output kept, and
`chunked_attention` (the fp32 outputs' attention) is upstream `eager_attention`."""

from __future__ import annotations

import copy

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.reference.modeling_kimi_k3 import eager_attention
from models.experimental.kimi_k3_moonvit_v2.tests.common import SHALLOW_REFERENCE, random_patches, random_tensor
from models.experimental.kimi_k3_moonvit_v2.tests.goldens import (
    chunked_attention,
    encoder_steps,
    to_fp32,
    use_chunked_attention,
)


@SHALLOW_REFERENCE
def test_encoder_steps_match_encoder(shallow_config, shallow_reference):
    tower = shallow_reference.vision_tower
    pixels, grid_thws = random_patches(shallow_config, [(1, 8, 12)])
    with torch.no_grad():
        x = tower.patch_embed(pixels.to(torch.bfloat16), grid_thws)
        expected = tower.encoder(x, grid_thws)
        steps = dict(encoder_steps(shallow_reference, x, grid_thws))
        # Resuming at block 1 from block 0's output gives the same result.
        resumed = dict(encoder_steps(shallow_reference, steps["block_00"], grid_thws, start=1))

    assert list(steps) == [f"block_{i:02d}" for i in range(shallow_config.num_hidden_layers)] + ["final_layernorm"]
    assert torch.equal(steps["final_layernorm"], expected)
    assert torch.equal(resumed["final_layernorm"], expected)


@pytest.mark.parametrize("length", [64, 300])
def test_chunked_attention_matches_eager(length):
    """fp32, chunks of 64 rows: one chunk exactly, and several with a partial last one."""
    q, k, v = (random_tensor(length, 12, 128, dtype=torch.float32, seed=s) for s in range(3))
    cu_seqlens = torch.tensor([0, length], dtype=torch.int32)
    expected = eager_attention(q, k, v, cu_seqlens, cu_seqlens)
    actual = chunked_attention(q, k, v, cu_seqlens, cu_seqlens, chunk_size=64)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@SHALLOW_REFERENCE
def test_fp32_chunked_encoder_matches_eager(shallow_config, shallow_reference):
    """The fp32 golden path (fp32 model, bf16 norm eps, chunked attention) against the same fp32 model
    with upstream eager attention, on an image of several chunks."""
    reference = to_fp32(copy.deepcopy(shallow_reference))
    tower = reference.vision_tower
    pixels, grid_thws = random_patches(shallow_config, [(1, 40, 52)])
    with torch.no_grad():
        x = tower.patch_embed(pixels, grid_thws)
        expected = tower.encoder(x, grid_thws)
        with use_chunked_attention():
            actual = dict(encoder_steps(reference, x, grid_thws))["final_layernorm"]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
