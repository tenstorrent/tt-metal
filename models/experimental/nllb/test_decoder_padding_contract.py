# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""CPU API/routing checks with local method substitutes; not TT numeric proof."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from . import backend


def sliced(value, start, end):
    return value[tuple(slice(a, b) for a, b in zip(start, end))].copy()


@pytest.mark.parametrize("final_token_only", [False, True])
def test_actual_decode_crops_only_trailing_masked_tiles_before_cross_mask(final_token_only):
    model = object.__new__(backend.Backend)
    model.pad = 1
    memory = np.arange(96 * 128).reshape(1, 1, 96, 128)
    valid = np.zeros(96, dtype=bool)
    valid[[2, 31]] = True
    prefix = np.array([[2, 9]], dtype=np.int64)
    originals = [value.copy() for value in (memory, valid, prefix)]
    observed = []
    cache = {}

    class BeforeAttention(Exception):
        pass

    def mask(keys, length, causal=False):
        observed.append((keys.copy(), length, causal))
        if not causal:
            # Stop before learned operations; exercise the real decode entry path.
            raise BeforeAttention()
        return object()

    model.mask = mask
    with patch.object(backend.ttnn, "slice", side_effect=sliced):
        with pytest.raises(BeforeAttention):
            model.decode(prefix, memory, valid, final_token_only=final_token_only, cross_kv=cache)
    assert len(observed) == 2
    assert observed[0][2] is True and observed[1][2] is False
    assert observed[1][1] == 32
    np.testing.assert_array_equal(observed[1][0], valid[:32])
    assert len(observed[1][0]) == 32
    assert np.flatnonzero(observed[1][0]).tolist() == [2, 31]
    assert cache == {}
    for value, original in zip((memory, valid, prefix), originals):
        np.testing.assert_array_equal(value, original)


def test_forward_keeps_full_public_encoder_extent_and_original_masks():
    model = object.__new__(backend.Backend)
    model.dim, model.vocab, model.pad = 4, 16, 1
    model.config = dict(vocab_size=16, max_position_embeddings=256)
    ids = np.full((2, 97), 5, dtype=np.int64)
    mask = np.zeros_like(ids)
    mask[0, [0, 31]] = 1
    mask[1, [2, 64]] = 1
    prefix = np.array([[2, 9], [2, 9]], dtype=np.int64)
    originals = [value.copy() for value in (ids, mask, prefix)]
    encoded, decoder_widths = [], []

    def encode(row_ids, row_mask):
        memory = np.arange(128 * model.dim).reshape(1, 1, 128, model.dim).astype(np.float32)
        memory += len(encoded) * 1000
        valid = np.pad(row_mask[0], (0, 128 - row_mask.shape[1]))
        encoded.append(memory.copy())
        return memory, valid

    def decode(row_prefix, memory, valid):
        # This substitute only isolates forward's public encoder-output contract.
        cropped, cropped_valid = model.decoder_memory(memory, valid)
        decoder_widths.append(cropped.shape[-2])
        assert cropped.shape[-2] == len(cropped_valid)
        np.testing.assert_array_equal(cropped, memory[..., : len(cropped_valid), :])
        return np.zeros((1, row_prefix.shape[1], model.vocab), dtype=np.float32)

    model.encode, model.decode = encode, decode
    with (
        patch.object(backend.ttnn, "slice", side_effect=sliced),
        patch.object(backend.ttnn, "to_torch", side_effect=torch.from_numpy),
    ):
        result = model.forward(ids, mask, prefix)
    assert decoder_widths == [32, 96]
    assert result["encoder"].shape == (2, 97, 4)
    assert result["logits"].shape == (2, 2, 16)
    np.testing.assert_array_equal(result["encoder"], np.concatenate(encoded).reshape(2, 128, 4)[:, :97])
    for value, original in zip((ids, mask, prefix), originals):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("method", ["forward", "generate"])
def test_one_fully_masked_row_rejected_before_any_encode(method):
    model = object.__new__(backend.Backend)
    model.config = dict(vocab_size=16, max_position_embeddings=256)
    model.vocab, model.pad = 16, 1
    model.encode = Mock(side_effect=AssertionError("invalid batch reached encoder"))
    ids = np.array([[5, 2, 1], [6, 2, 1]], dtype=np.int64)
    mask = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.int64)
    before = [value.copy() for value in (ids, mask)]
    with pytest.raises(ValueError, match="unmasked"):
        if method == "forward":
            model.forward(ids, mask, np.array([[2, 9], [2, 9]], dtype=np.int64))
        else:
            model.generate(ids, mask, 9, 1)
    model.encode.assert_not_called()
    for value, original in zip((ids, mask), before):
        np.testing.assert_array_equal(value, original)
