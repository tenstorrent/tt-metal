# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ttml.datasets and ttml.trainers (no device required)."""

import numpy as np
import pytest

import ttml
import ttnn

from ttml.datasets.dataloader import Batch, TTMLDataloader
from ttml.datasets.hf_dataloader import InMemoryDataloader, sft_collate_fn


# ---------------------------------------------------------------------------
# Patch ttml.autograd.Tensor.from_numpy to return a plain numpy wrapper so
# these tests do not require an initialized Tenstorrent device.
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal stand-in for ttml.autograd.Tensor used in dataloader tests."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    @staticmethod
    def from_numpy(arr, layout, dtype):
        return _FakeTensor(arr)

    def to_numpy(self, dtype=None, composer=None):
        return self._arr

    def __mul__(self, other):
        v = other._arr if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self._arr * v)

    __rmul__ = __mul__


@pytest.fixture(autouse=True)
def patch_tensor(monkeypatch):
    """Replace Tensor.from_numpy with a device-free stub for every test."""
    monkeypatch.setattr(
        ttml.autograd.Tensor, "from_numpy", staticmethod(_FakeTensor.from_numpy)
    )


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def test_batch_fields():
    b = Batch(input_ids="ids", labels="lbs", loss_mask="mask")
    assert b.input_ids == "ids"
    assert b.labels == "lbs"
    assert b.loss_mask == "mask"


# ---------------------------------------------------------------------------
# TTMLDataloader (abstract)
# ---------------------------------------------------------------------------


def test_ttml_dataloader_is_abstract():
    with pytest.raises(TypeError):
        TTMLDataloader(dataset=[], collate_fn=lambda x: x, batch_size=1)


def test_ttml_dataloader_concrete_subclass():
    class SimpleLoader(TTMLDataloader):
        def __iter__(self):
            for i in range(0, len(self.dataset), self.batch_size):
                yield self.collate_fn(self.dataset[i : i + self.batch_size])

        def __len__(self):
            return len(self.dataset) // self.batch_size

    loader = SimpleLoader(dataset=list(range(10)), collate_fn=list, batch_size=2)
    batches = list(loader)
    assert len(batches) == 5
    assert batches[0] == [0, 1]


# ---------------------------------------------------------------------------
# sft_collate_fn
# ---------------------------------------------------------------------------


def _make_examples():
    return [
        {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
        {"input_ids": [10, 20, 30], "labels": [-100, 20, 30]},
    ]


def test_sft_collate_fn_returns_batch():
    batch = sft_collate_fn(_make_examples(), max_seq_len=8, pad_token_id=0)
    assert isinstance(batch, Batch)


def test_sft_collate_fn_loss_mask_normalized():
    """mask.sum() == B * T so mean(loss * mask) == per-completion-token loss."""
    examples = [
        {"input_ids": [1, 2, 10, 11, 12], "labels": [-100, -100, 10, 11, 12]},
    ]
    batch = sft_collate_fn(examples, max_seq_len=8, pad_token_id=0)
    mask = batch.loss_mask._arr  # [B, 1, T, 1] via _FakeTensor
    B, _, T, _ = mask.shape
    assert abs(mask.sum() - B * T) < 1e-3, f"expected sum={B * T}, got {mask.sum()}"


def test_sft_collate_fn_prompt_zeroed_completion_nonzero():
    # 4-token sequence (2 prompt + 2 completion), padded to max_seq_len=6
    # Batch of 2 so shorter sequence gets padded
    examples = [
        {"input_ids": [1, 2, 10, 11], "labels": [-100, -100, 10, 11]},
        {"input_ids": [1, 2, 3, 10, 11, 12], "labels": [-100, -100, -100, 10, 11, 12]},
    ]
    batch = sft_collate_fn(examples, max_seq_len=8, pad_token_id=0)
    mask = batch.loss_mask._arr  # [2, 1, 6, 1]

    # First example: positions 0-1 are prompt, 2-3 are completion, 4-5 are padding
    assert mask[0, 0, 0, 0] == 0.0, "prompt token should be masked"
    assert mask[0, 0, 1, 0] == 0.0, "prompt token should be masked"
    assert mask[0, 0, 2, 0] > 0.0, "completion token should be unmasked"
    assert mask[0, 0, 3, 0] > 0.0, "completion token should be unmasked"
    assert mask[0, 0, 4, 0] == 0.0, "padding should be masked"
    assert mask[0, 0, 5, 0] == 0.0, "padding should be masked"


# ---------------------------------------------------------------------------
# InMemoryDataloader
# ---------------------------------------------------------------------------


def _sft_examples(n):
    return [
        {"input_ids": list(range(i, i + 4)), "labels": [-100, i + 1, i + 2, i + 3]}
        for i in range(n)
    ]


def _collate(examples):
    return sft_collate_fn(examples, max_seq_len=8, pad_token_id=0)


def test_in_memory_dataloader_len():
    loader = InMemoryDataloader(_sft_examples(10), _collate, batch_size=3)
    assert len(loader) == 3  # drop_last=True


def test_in_memory_dataloader_yields_batch_objects():
    loader = InMemoryDataloader(_sft_examples(6), _collate, batch_size=2)
    batches = list(loader)
    assert len(batches) == 3
    for b in batches:
        assert isinstance(b, Batch)


def test_in_memory_dataloader_shuffle_changes_order():
    data = _sft_examples(20)

    np.random.seed(0)
    loader = InMemoryDataloader(data, lambda x: x, batch_size=1, shuffle=True)
    order_a = [b[0]["input_ids"][0] for b in loader]

    np.random.seed(99)
    order_b = [b[0]["input_ids"][0] for b in loader]

    assert order_a != order_b
