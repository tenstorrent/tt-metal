# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-sampled decode logits keep their vocab axis when the decode bucket is
narrower than the batch the caller indexes by."""

import torch

from models.demos.gemma4.tt.model import _decode_logits_rows


def test_narrow_bucket_is_padded_not_folded():
    vocab = 1024
    logits = torch.arange(4 * vocab, dtype=torch.float32).reshape(1, 1, 4, vocab)
    out = _decode_logits_rows(logits, B=32, S=1)
    assert tuple(out.shape) == (32, 1, vocab)
    assert torch.equal(out[:4, 0, :], logits[0, 0])
    assert out[4:].abs().sum() == 0
    assert int(out[0, 0].argmax()) == vocab - 1  # the full row, not its first 1/8th


def test_full_batch_passes_through():
    logits = torch.randn(1, 1, 32, 64)
    out = _decode_logits_rows(logits, B=32, S=1)
    assert torch.equal(out, logits.reshape(32, 1, 64))
