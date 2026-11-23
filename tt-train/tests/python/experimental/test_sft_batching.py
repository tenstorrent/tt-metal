# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""How SFT dataset providers batch a split: train drops a last partial batch, validation fills it with padding rows
that don't count in the loss; and how a validation split carved out of the train split stays held out. Device-free."""

from types import SimpleNamespace

import numpy as np
from datasets import Dataset

from ttml.experimental.datasets.provider.sft.base import DatasetProvider
from ttml.experimental.datasets.utils.sft import collate_causal_lm

BATCH_SIZE = 4
MAX_LENGTH = 32
PAD = 0
NUM_EXAMPLES = 10  # two full batches and one of two examples


class _Tokenizer:
    """One token per character, its code point; no special tokens."""

    eos_token = "\x01"
    pad_token_id = PAD

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


class _Provider(DatasetProvider):
    def _load_raw_dataset(self):
        return Dataset.from_dict({"n": list(range(NUM_EXAMPLES))})

    def _prompt_and_response(self, example):
        return f"q{example['n']}:", "a" * (1 + example["n"] % 3)


class _CarvedProvider(_Provider):
    train_val_split_ratio = 0.8


def _provider(split):
    config = SimpleNamespace(batch_size=BATCH_SIZE, max_length=MAX_LENGTH, seed=0)
    return _Provider(config, split, _Tokenizer())


def _carved_ns(split, seed, max_length):
    """The ``n`` of each example of ``split`` of ``_CarvedProvider``."""
    config = SimpleNamespace(batch_size=BATCH_SIZE, max_length=max_length, seed=seed)
    examples = _CarvedProvider(config, split, _Tokenizer()).examples
    return {int("".join(chr(t) for t in example["input_ids"]).split(":")[0][1:]) for example in examples}


def _masked_mean(batch, per_token):
    """The loss ``SFTTrainer._forward`` computes, for a per-token loss of ``per_token``."""
    return float(np.mean(per_token[: len(batch["loss_mask"])] * batch["loss_mask"][:, 0, :, 0]))


def test_train_drops_the_last_partial_batch():
    train = _provider("train")
    batches = list(train)
    assert len(train) == len(batches) == NUM_EXAMPLES // BATCH_SIZE
    assert all(batch["input_ids"].shape[0] == BATCH_SIZE for batch in batches)


def test_validation_pads_the_last_batch_and_runs_every_example():
    validation = _provider("validation")
    batches = list(validation)
    assert len(validation) == len(batches) == -(-NUM_EXAMPLES // BATCH_SIZE)
    assert all(batch["input_ids"].shape == (BATCH_SIZE, 1, 1, MAX_LENGTH) for batch in batches)

    real_rows = NUM_EXAMPLES % BATCH_SIZE
    last = batches[-1]
    assert (last["input_ids"][real_rows:] == PAD).all()
    assert (last["loss_mask"][real_rows:] == 0).all()
    assert (last["loss_mask"][:real_rows].sum(axis=(1, 2, 3)) > 0).all()

    # Every example is in one batch, once, unshuffled.
    rows = [row for batch in batches for row in batch["input_ids"][:, 0, 0] if row.any()]
    prompts = ["".join(chr(t) for t in row if t != PAD).split(":")[0] for row in rows]
    assert prompts == [f"q{n}" for n in range(NUM_EXAMPLES)]


def test_a_carved_validation_split_is_held_out_whatever_the_seed_and_max_length():
    validation = _carved_ns("validation", seed=0, max_length=MAX_LENGTH)
    assert len(validation) == NUM_EXAMPLES - int(_CarvedProvider.train_val_split_ratio * NUM_EXAMPLES)
    assert validation.isdisjoint(_carved_ns("train", seed=0, max_length=MAX_LENGTH))

    assert _carved_ns("validation", seed=1, max_length=MAX_LENGTH) == validation
    assert _carved_ns("train", seed=1, max_length=MAX_LENGTH).isdisjoint(validation)

    # "q{n}:" and 1 + n % 3 response tokens and EOS: 6 tokens drops the examples with n % 3 == 2.
    short = 6
    assert _carved_ns("validation", seed=0, max_length=short) == {n for n in validation if n % 3 != 2}
    assert _carved_ns("train", seed=0, max_length=short).isdisjoint(validation)


def test_padding_rows_leave_the_loss_unchanged():
    examples = [_provider("validation").examples[i] for i in range(2)]
    padded = collate_causal_lm(examples, MAX_LENGTH, PAD, batch_size=BATCH_SIZE)
    unpadded = collate_causal_lm(examples, MAX_LENGTH, PAD)
    assert padded["input_ids"].shape[0] == BATCH_SIZE and unpadded["input_ids"].shape[0] == len(examples)

    per_token = np.random.default_rng(0).uniform(0.5, 5.0, size=(BATCH_SIZE, MAX_LENGTH))
    np.testing.assert_allclose(_masked_mean(padded, per_token), _masked_mean(unpadded, per_token), rtol=1e-6)
