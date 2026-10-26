# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for how ``Qwen3ForEmbedding`` reads a padded batch.

A served batch is padded to its longest request. The embedding of each request must still be the hidden state
of its own last real token, as in the Qwen3-Embedding reference (``last_token_pool`` with the attention mask).
The pooling test checks that on a small randomly initialized Qwen3 model: the left-aligned rows, run causally
without a padding mask and read at ``prompt_len - 1`` (what the TT prefill does), give the reference embedding.
The forward test checks that ``forward`` hands the generator those rows and lengths.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import transformers

from models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm import Qwen3ForEmbedding, left_align_real_tokens

PAD = 7  # any token id that is not one of the real ones, which start at 100


def _padded_batch(lengths, max_len, side, pad=PAD, mask_dtype=torch.long):
    ids = torch.full((len(lengths), max_len), pad, dtype=torch.long)
    mask = torch.zeros((len(lengths), max_len), dtype=mask_dtype)
    for row, n in enumerate(lengths):
        tokens = torch.arange(100 + 10 * row, 100 + 10 * row + n)
        span = slice(0, n) if side == "right" else slice(max_len - n, max_len)
        ids[row, span] = tokens
        mask[row, span] = 1
    return ids, mask


@pytest.mark.parametrize(
    "side, pad, mask_dtype",
    [
        ("right", PAD, torch.long),
        ("left", PAD, torch.long),
        ("right", 0, torch.float32),  # the vLLM pooling runner pads with token 0 and passes a float mask
    ],
)
def test_rows_start_with_their_real_tokens(side, pad, mask_dtype):
    lengths = [5, 2, 8]
    ids, mask = _padded_batch(lengths, max_len=8, side=side, pad=pad, mask_dtype=mask_dtype)
    aligned, prompt_lens = left_align_real_tokens(ids, mask)
    assert prompt_lens == lengths, f"expected real lengths {lengths}, got {prompt_lens}"
    for row, n in enumerate(lengths):
        expected = torch.arange(100 + 10 * row, 100 + 10 * row + n)
        assert torch.equal(aligned[row, :n], expected), f"row {row} starts with {aligned[row, :n].tolist()}"


def test_row_without_real_tokens_is_rejected(expect_error):
    ids, mask = _padded_batch([3, 2], max_len=3, side="right")
    mask[1] = 0
    with expect_error(ValueError, "at least one real token"):
        left_align_real_tokens(ids, mask)


@pytest.mark.parametrize("side", ["right", "left"])
def test_last_real_token_matches_reference_pooling(side):
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    model = transformers.Qwen3Model(config).eval()
    lengths = [9, 3, 6, 1]
    ids, mask = _padded_batch(lengths, max_len=9, side=side)

    with torch.no_grad():
        # Reference: each text alone, last token
        reference = torch.stack(
            [model(input_ids=ids[row][mask[row].bool()][None]).last_hidden_state[0, -1] for row in range(len(lengths))]
        )
        # TT path: aligned rows, causal attention without a padding mask, hidden state at prompt_len - 1
        aligned, prompt_lens = left_align_real_tokens(ids, mask)
        hidden = model(input_ids=aligned).last_hidden_state
        tt = torch.stack([hidden[row, n - 1] for row, n in enumerate(prompt_lens)])
        # Every row read at the batch length, with the padding in place
        at_batch_length = model(input_ids=ids).last_hidden_state[:, -1]

    # Same real tokens through the same float32 model: only the reduction order differs, well under 1e-5
    torch.testing.assert_close(tt, reference, rtol=1e-5, atol=1e-5)
    # Reading through the padding moves the padded rows far more than this tolerance
    assert not torch.allclose(at_batch_length, reference, rtol=1e-3, atol=1e-3), "batch-length read should differ"


def test_forward_passes_each_row_its_real_length():
    lengths = [5, 2]
    ids, mask = _padded_batch(lengths, max_len=5, side="right")
    embedder = Qwen3ForEmbedding.__new__(Qwen3ForEmbedding)
    embedder.max_batch_size, embedder.max_seq_len = 2, 64
    embedder.paged_attention_config = SimpleNamespace(block_size=32)
    embedder._kv_cache = None
    embedder.generator = mock.Mock()
    embedder.generator.prefill_forward_text.return_value = torch.zeros(len(lengths), 8)

    with mock.patch.object(Qwen3ForEmbedding, "_initialize_model"):
        embedder.forward(ids, attention_mask=mask)

    call = embedder.generator.prefill_forward_text.call_args
    tokens, prompt_lens = call.args[0], call.kwargs["prompt_lens"]
    assert prompt_lens == lengths, f"expected prompt_lens {lengths}, got {prompt_lens}"
    for row, n in enumerate(lengths):
        assert torch.equal(tokens[row, :n], ids[row, :n]), f"row {row} starts with {tokens[row, :n].tolist()}"
