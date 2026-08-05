# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ttml.datasets and ttml.trainers (no device required)."""

import numpy as np
import pytest

import ttml

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
    monkeypatch.setattr(ttml.autograd.Tensor, "from_numpy", staticmethod(_FakeTensor.from_numpy))


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


def test_ttml_dataloader_is_abstract(expect_error):
    with expect_error(TypeError, "abstract class TTMLDataloader"):
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
    return [{"input_ids": list(range(i, i + 4)), "labels": [-100, i + 1, i + 2, i + 3]} for i in range(n)]


def _collate(examples):
    return sft_collate_fn(examples, max_seq_len=8, pad_token_id=0)


def _lead_token(batch):
    """First token of a batch's lead example — a per-batch identifier (assumes the identity `lambda x: x` collate)."""
    return batch[0]["input_ids"][0]


def _batch_order(loader):
    """Sequence of per-batch lead tokens — a fingerprint of the loader's iteration order."""
    return [_lead_token(batch) for batch in loader]


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
    """Order is a pure function of the loader's seed — independent of the global np.random."""
    data = _sft_examples(20)

    order_a = _batch_order(InMemoryDataloader(data, lambda x: x, batch_size=1, shuffle=True, seed=0))
    order_b = _batch_order(InMemoryDataloader(data, lambda x: x, batch_size=1, shuffle=True, seed=99))
    order_a_again = _batch_order(InMemoryDataloader(data, lambda x: x, batch_size=1, shuffle=True, seed=0))

    assert order_a != order_b  # different seed -> different order
    assert order_a == order_a_again  # same seed -> reproducible epoch-0 order


def test_in_memory_dataloader_resume_matches_uninterrupted():
    """Saving (epoch, position) mid-epoch and restoring into a fresh loader reproduces the exact
    remaining batch sequence — first_k + resumed_remainder == one uninterrupted epoch."""
    data = _sft_examples(20)

    uninterrupted = _batch_order(InMemoryDataloader(data, lambda x: x, batch_size=2, shuffle=True, seed=7))

    k = 3
    partial_loader = InMemoryDataloader(data, lambda x: x, batch_size=2, shuffle=True, seed=7)
    it = iter(partial_loader)
    first = [_lead_token(next(it)) for _ in range(k)]
    state = partial_loader.get_state_dict()

    # Built with a deliberately wrong seed: set_state_dict must restore seed=7 from the state, or the
    # resumed order wouldn't line up with the uninterrupted run.
    resumed = InMemoryDataloader(data, lambda x: x, batch_size=2, shuffle=True, seed=999)
    resumed.set_state_dict(state)
    remainder = _batch_order(resumed)

    assert first + remainder == uninterrupted


def test_dataloader_state_dict_roundtrip():
    """get_state_dict -> set_state_dict carries (seed, epoch, position) onto a fresh loader."""
    loader = InMemoryDataloader(_sft_examples(8), lambda x: x, batch_size=2, shuffle=True, seed=5)
    it = iter(loader)
    next(it)
    state = loader.get_state_dict()
    assert state == {"seed": 5, "epoch": 0, "position": 1}

    restored = InMemoryDataloader(_sft_examples(8), lambda x: x, batch_size=2, seed=0)
    restored.set_state_dict(state)
    assert restored.get_state_dict() == state


# ---------------------------------------------------------------------------
# LoraConfig re-export
# ---------------------------------------------------------------------------


def test_lora_config_import():
    """LoraConfig is accessible from ttml.trainers."""
    from ttml.trainers import LoraConfig

    cfg = LoraConfig(rank=4, alpha=8.0, target_modules=[".*q_proj.*"])
    assert cfg.rank == 4
    assert cfg.alpha == 8.0
    assert cfg.target_modules == [".*q_proj.*"]


# ---------------------------------------------------------------------------
# TrainerCallback
# ---------------------------------------------------------------------------


def test_trainer_callback_import():
    """TrainerCallback is accessible from ttml.trainers."""
    from ttml.trainers import TrainerCallback

    cb = TrainerCallback()
    # All hooks are no-ops by default — just verify they're callable.
    cb.on_train_begin(None)
    cb.on_step_end(None, 0, 0.0, 1e-5)
    cb.on_eval_end(None, 0, 0.0)
    cb.on_save(None, 0, "/tmp/ckpt")
    cb.on_train_end(None)


# ---------------------------------------------------------------------------
# SFTConfig new fields
# ---------------------------------------------------------------------------


def test_sft_config_defaults():
    """New SFTConfig fields have sensible defaults."""
    from ttml.trainers import SFTConfig

    cfg = SFTConfig()
    assert cfg.max_grad_norm == 0.0
    assert cfg.log_interval == 1
    assert cfg.max_steps == 1000
    assert cfg.learning_rate == 2e-5
    assert cfg.warmup_steps == 0


def test_sft_config_custom_values():
    """SFTConfig accepts custom values for new fields."""
    from ttml.trainers import SFTConfig

    cfg = SFTConfig(max_grad_norm=1.0, log_interval=10)
    assert cfg.max_grad_norm == 1.0
    assert cfg.log_interval == 10
