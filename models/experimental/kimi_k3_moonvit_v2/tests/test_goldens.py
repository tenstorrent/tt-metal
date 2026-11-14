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
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    SYNTHETIC,
    random_patches,
    random_tensor,
    reference_rope,
)
from models.experimental.kimi_k3_moonvit_v2.tests.goldens import (
    NORM_EPS,
    block_file,
    chunked_attention,
    encoder_steps,
    to_fp32,
    use_chunked_attention,
    use_sampled_attention,
)


@SYNTHETIC
def test_encoder_steps_match_encoder(config, reference):
    tower = reference.vision_tower
    pixels, grid_thws = random_patches(config, [(1, 8, 12)])
    with torch.no_grad():
        x = tower.patch_embed(pixels.to(torch.bfloat16), grid_thws)
        expected = tower.encoder(x, grid_thws)
        (expected_model,) = reference(pixels.to(torch.bfloat16), grid_thws)
        steps = dict(encoder_steps(reference, x, grid_thws))
        # Resuming at block 1 from block 0's output gives the same result.
        resumed = dict(encoder_steps(reference, steps["block_00"], grid_thws, start=1))

    blocks = [block_file(i)[:-3] for i in range(config.num_hidden_layers)]
    assert list(steps) == blocks + ["final_layernorm", "mm_projector"]
    assert torch.equal(steps["final_layernorm"], expected)
    assert torch.equal(resumed["final_layernorm"], expected)
    assert torch.equal(steps["mm_projector"], expected_model)


@SYNTHETIC
def test_to_fp32_keeps_explicit_norm_eps(config, reference):
    """`to_fp32` gives the encoder norms (`eps=None`) the bf16 eps and leaves the projector's
    `post_norm` eps (set explicitly) alone."""
    assert NORM_EPS == torch.finfo(torch.bfloat16).eps  # what upstream's eps=None gives bf16 input
    reference = to_fp32(copy.deepcopy(reference))
    assert reference.vision_tower.encoder.final_layernorm.eps == NORM_EPS
    assert reference.vision_tower.encoder.blocks[0].norm0.eps == NORM_EPS
    assert reference.mm_projector.post_norm.eps == config.projector_norm_eps


@pytest.mark.parametrize("length", [64, 300])
def test_chunked_attention_matches_eager(length):
    """fp32, chunks of 64 rows: one chunk exactly, and several with a partial last one."""
    q, k, v = (random_tensor(length, 12, 128, dtype=torch.float32, seed=s) for s in range(3))
    cu_seqlens = torch.tensor([0, length], dtype=torch.int32)
    expected = eager_attention(q, k, v, cu_seqlens, cu_seqlens)
    actual = chunked_attention(q, k, v, cu_seqlens, cu_seqlens, chunk_size=64)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@SYNTHETIC
def test_fp32_chunked_encoder_matches_eager(config, reference):
    """The fp32 golden path (fp32 model, bf16 norm eps, chunked attention) against the same fp32 model
    with upstream eager attention, on an image of several chunks."""
    reference = to_fp32(copy.deepcopy(reference))
    tower = reference.vision_tower
    pixels, grid_thws = random_patches(config, [(1, 40, 52)])
    with torch.no_grad():
        x = tower.patch_embed(pixels, grid_thws)
        expected = tower.encoder(x, grid_thws)
        with use_chunked_attention():
            actual = dict(encoder_steps(reference, x, grid_thws))["final_layernorm"]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@SYNTHETIC
def test_sampled_attention_matches_eager(config, reference):
    """A block run with `use_sampled_attention`, as the device tests with real rows on every chip
    build their golden, equals the same fp32 block with upstream eager attention on the sampled rows,
    over several query chunks; the rows not sampled are left out of the attention."""
    reference = to_fp32(copy.deepcopy(reference))
    block = reference.vision_tower.encoder.blocks[0]
    grid_thws = torch.tensor([(1, 40, 52)])
    n = 40 * 52
    x = random_tensor(n, config.hidden_size, dtype=torch.float32)
    cu_seqlens, max_seqlen, rope_freqs_cis = reference_rope(reference, grid_thws)
    rows = torch.cat([torch.arange(0, 300), torch.arange(1000, 1100), torch.arange(n - 50, n)])
    others = torch.ones(n, dtype=torch.bool)
    others[rows] = False
    with torch.no_grad():
        expected_attention = block.attention_qkvpacked(x, cu_seqlens, max_seqlen, rope_freqs_cis)
        expected = block(x, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis)
        with use_sampled_attention(rows):
            attention = block.attention_qkvpacked(x, cu_seqlens, max_seqlen, rope_freqs_cis)
            actual = block(x, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis)
    torch.testing.assert_close(attention[rows], expected_attention[rows], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[rows], expected[rows], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(attention[others], expected_attention[others])
