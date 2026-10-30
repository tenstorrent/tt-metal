# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Large inputs as chunks of whole sequences: tt/chunking.py's plans and the model's chunked paths.

A chunk runs the whole backbone on its own, so the chunked forward must return, for each sequence, what
a forward of that sequence's chunk alone returns, bit for bit; and embed(), which also sorts the texts and
cuts each chunk to its longest one, must return the unchunked embeddings to within the routing flips a
different batch shape brings.
"""

from dataclasses import replace

import pytest
import torch

import ttnn

from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.common import random_input_ids
from models.experimental.nomic_embed_text_v2_moe.tt import pooling
from models.experimental.nomic_embed_text_v2_moe.tt.chunking import plan_chunks, plan_sorted_chunks
from models.experimental.nomic_embed_text_v2_moe.tt.common import pooling_mask
from models.experimental.nomic_embed_text_v2_moe.tt.model import TtNomicBertModel, embed
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup

# embed() against the unchunked, untrimmed backbone: the chunks have other shapes, so near-tied tokens
# route differently (measured min cosine 0.99974 over a 24-sequence ragged batch at 2048-token chunks).
CHUNKED_EMBED_COSINE = 0.999


@pytest.mark.parametrize(
    "batch, seqlen, max_tokens", [(32, 512, 8192), (20, 512, 8192), (7, 512, 2048), (512, 32, 8192), (3, 2048, 1024)]
)
def test_plan_covers_the_batch(batch, seqlen, max_tokens):
    """Contiguous chunks, in order, covering every sequence once, none over max_tokens unless one sequence
    alone is (then it is a chunk of its own), every chunk but the last full."""
    chunks = plan_chunks(batch, seqlen, max_tokens)
    full = max(1, max_tokens // (ttnn.core.divup(seqlen, ttnn.TILE_SIZE) * ttnn.TILE_SIZE))
    start = 0
    for chunk in chunks:
        assert chunk.start == start and chunk.count > 0 and chunk.seqlen == seqlen
        assert chunk.tokens <= max_tokens or chunk.count == 1
        assert chunk.count == full or chunk is chunks[-1]
        start += chunk.count
    assert start == batch


def test_sorted_plan_covers_the_batch_and_cuts_each_chunk_to_its_longest(expect_error):
    generator = torch.Generator().manual_seed(0)
    lengths = sorted(torch.randint(1, 513, (96,), generator=generator).tolist(), reverse=True)
    chunks = plan_sorted_chunks(lengths, 8192)
    start = 0
    for chunk in chunks:
        assert chunk.start == start and chunk.seqlen == lengths[start]
        assert chunk.tokens <= 8192
        assert chunk is chunks[-1] or chunk.tokens + chunk.tokens // chunk.count > 8192, "a chunk not filled"
        start += chunk.count
    assert start == len(lengths)
    with expect_error(ValueError, "sorted longest first"):
        plan_sorted_chunks(lengths[::-1], 8192)


@pytest.fixture
def tt_model(device, config, tt_config, state_dict):
    return TtNomicBertModel(device, config, tt_config, state_dict)


def small_chunks(model, max_tokens):
    """The same model, planning chunks of at most max_tokens."""
    model.tt_config = replace(model.tt_config, max_chunk_tokens=max_tokens)
    return model


@run_for_blackhole()
@pytest.mark.use_module_device
@pytest.mark.needs_weights
@pytest.mark.parametrize("batch, seqlen, max_tokens", [(6, 512, 2048), (5, 100, 256)])
def test_chunked_forward_equals_its_chunks(config, tt_model, batch, seqlen, max_tokens):
    """Each sequence's output equals a backbone pass over its chunk alone, bit for bit, and the
    joined output keeps the batch's (B, 1, S, H) shape in DRAM."""
    model = small_chunks(tt_model, max_tokens)
    input_ids, attention_mask = random_input_ids(batch, seqlen, config, seed=4)
    chunks = plan_chunks(batch, seqlen, max_tokens)
    assert len(chunks) > 1, "the shape no longer splits; pick one that does"

    out = model(input_ids, attention_mask)
    assert tuple(out.shape) == (batch, 1, seqlen, config.hidden_size)
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    whole = ttnn.to_torch(out)
    ttnn.deallocate(out)
    for chunk in chunks:
        rows = slice(chunk.start, chunk.start + chunk.count)
        alone = model.backbone(input_ids[rows], attention_mask[rows])
        assert torch.equal(whole[rows], ttnn.to_torch(alone)), f"chunk {chunk} differs from its pass alone"
        ttnn.deallocate(alone)


def unchunked_embeddings(model, input_ids, attention_mask):
    kernel = model.tt_config.compute_kernel_config(OpGroup.REDUCE)
    hidden = model.backbone(input_ids, attention_mask)
    pooled = pooling.mean_pool(hidden, pooling_mask(attention_mask, model.device), compute_kernel_config=kernel)
    ttnn.deallocate(hidden)
    normalized = pooling.l2_normalize(pooled, compute_kernel_config=kernel)
    return ttnn.to_torch(normalized).float().reshape(input_ids.shape[0], -1)


@run_for_blackhole()
@pytest.mark.use_module_device
@pytest.mark.needs_weights
def test_embed_matches_the_unchunked_embeddings_in_input_order(config, tt_model):
    """embed() sorts by length, cuts each chunk to its longest text and pools per chunk; the rows come
    back in input order and match one unchunked, untrimmed pass of the whole ragged batch."""
    generator = torch.Generator().manual_seed(3)
    lengths = torch.randint(20, 513, (24,), generator=generator)
    input_ids, attention_mask = random_input_ids(24, 512, config, seed=3, pad_lengths=[512 - int(n) for n in lengths])
    want = unchunked_embeddings(tt_model, input_ids, attention_mask)

    got = embed(small_chunks(tt_model, 2048), input_ids, attention_mask)

    assert got.shape == want.shape
    cosine = (got * want).sum(-1) / (got.norm(dim=-1) * want.norm(dim=-1))
    assert bool((cosine > CHUNKED_EMBED_COSINE).all()), f"worst row cosine {float(cosine.min()):.6f}"
