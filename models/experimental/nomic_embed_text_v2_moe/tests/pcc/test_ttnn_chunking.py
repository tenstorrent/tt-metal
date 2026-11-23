# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Large inputs as chunks of whole sequences: tt/chunking.py's plans and the model's chunked paths.

A chunk runs the whole backbone on its own, so the chunked forward must return, for each sequence, what
a forward of that sequence's chunk alone returns, bit for bit. embed() also sorts the texts and cuts each
chunk to its longest one: in one chunk at the batch's own shape it must return the unchunked embeddings
exactly, and in several it must match the reference at the model's own tolerance.
"""

from dataclasses import replace

import pytest
import torch

import ttnn

from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.common import random_input_ids
from models.experimental.nomic_embed_text_v2_moe.reference.postprocessing import l2_normalize, mean_pool
from models.experimental.nomic_embed_text_v2_moe.tests.pcc.test_ttnn_model import COSINE_TOLERANCE
from models.experimental.nomic_embed_text_v2_moe.tt import pooling
from models.experimental.nomic_embed_text_v2_moe.tt.chunking import plan_chunks, plan_sorted_chunks
from models.experimental.nomic_embed_text_v2_moe.tt.common import pooling_mask
from models.experimental.nomic_embed_text_v2_moe.tt.model import TtNomicBertModel, embed
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup


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


def ragged_batch(config, seed, batch=24, seqlen=512):
    """Right-padded texts of 20 to seqlen tokens, the first one full length."""
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(20, seqlen + 1, (batch,), generator=generator)
    lengths[0] = seqlen
    return random_input_ids(batch, seqlen, config, seed=seed, pad_lengths=[seqlen - int(n) for n in lengths])


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
def test_one_chunk_embed_equals_the_unchunked_pass(config, tt_model):
    """With every text in one chunk and the longest at the full S, embed() runs the batch's own shape with
    its rows sorted by length, so each row must come back in input order holding exactly the embedding of
    one unchunked pass."""
    input_ids, attention_mask = ragged_batch(config, seed=3)
    batch, seqlen = input_ids.shape
    want = unchunked_embeddings(tt_model, input_ids, attention_mask)

    got = embed(small_chunks(tt_model, batch * seqlen), input_ids, attention_mask)

    assert torch.equal(got, want)


@run_for_blackhole()
@pytest.mark.use_module_device
@pytest.mark.needs_weights
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_chunked_embed_matches_the_reference(config, reference_model, tt_model, seed):
    """embed() over a ragged batch in several length-sorted chunks, each cut to its longest text, against
    the reference per row in input order.

    Gated on the reference rather than on an unchunked device pass: the chunks' shapes reroute other
    near-tied tokens than the unchunked pass does, so the two device runs differ by both errors (worst row
    cosine 0.9955 against each other on one draw, with each of them within 6e-03 of the reference). The short
    texts are the worst rows, as at the model's own gate, so this sweeps seeds too.
    """
    input_ids, attention_mask = ragged_batch(config, seed)
    max_tokens = 2048
    assert len(plan_sorted_chunks(sorted(attention_mask.sum(-1).tolist(), reverse=True), max_tokens)) > 1

    got = embed(small_chunks(tt_model, max_tokens), input_ids, attention_mask)

    with torch.no_grad():
        ref = l2_normalize(mean_pool(reference_model(input_ids, attention_mask=attention_mask), attention_mask))
    cosine = (got * ref).sum(-1)
    assert torch.allclose(got.norm(dim=-1), torch.ones(len(got)), atol=1e-2), "output is not unit norm"
    assert bool((cosine > 1.0 - COSINE_TOLERANCE).all()), f"worst row cosine {float(cosine.min()):.6f}"
