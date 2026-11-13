# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dataset providers: prompt templates, tokenization with the prompt masked, filtering, splits, and the batches a
trainer gets from them. Device-free."""

import json

import numpy as np
import pytest

from ttml.datasets.alpaca import AlpacaProvider
from ttml.datasets.custom import CustomProvider
from ttml.datasets.provider import IGNORED_INDEX, DatasetProvider, clear_cache, get_provider, providers
from ttml.datasets.sst2 import SST2Provider
from ttml.datasets.templates import custom_prompt_and_response, resolve_column_mapping
from ttml.unified_trainer import SFTTrainerConfig

datasets = pytest.importorskip("datasets")

BOS, EOS, PAD = 1, 2, 0


class CharTokenizer:
    """One token per character, a BOS first, and "</s>" as EOS: enough to check what gets masked."""

    eos_token = "</s>"
    pad_token_id = PAD

    def __call__(self, text, truncation=False, padding=False):
        ids = [BOS]
        for i, part in enumerate(text.split(self.eos_token)):
            if i:
                ids.append(EOS)
            ids += [3 + ord(c) % 200 for c in part]
        return {"input_ids": ids}


def config(**overrides):
    fields = {
        "dataset": {"id": "sst2"},
        "model": {"name": "m"},
        "optimizer": {"lr": 1e-4},
        "batch_size": 2,
        "max_length": 32,
    }
    return SFTTrainerConfig.model_validate({**fields, **overrides})


class Pairs(DatasetProvider):
    """Prompt/response pairs given in the test; counts how often the source is loaded."""

    rows: list = []
    loads = 0

    def _load_raw_dataset(self):
        type(self).loads += 1
        return datasets.Dataset.from_list(self.rows)

    def _prompt_and_response(self, example):
        return example["prompt"], example["response"]


@pytest.fixture(autouse=True)
def fresh_cache():
    clear_cache()
    yield
    clear_cache()


def test_the_prompt_is_masked_and_eos_appended():
    Pairs.rows = [{"prompt": "ab", "response": "cd"}]
    provider = Pairs(config(), "train", CharTokenizer())
    sample = provider.examples[0]
    assert sample["input_ids"][0] == BOS and sample["input_ids"][-1] == EOS
    assert len(sample["input_ids"]) == 1 + 4 + 1
    # BOS and the 2 prompt characters are masked; the response and EOS are trained on.
    assert sample["labels"][:3] == [IGNORED_INDEX] * 3
    assert sample["labels"][3:] == sample["input_ids"][3:]


def test_examples_longer_than_max_length_are_dropped():
    Pairs.rows = [{"prompt": "a", "response": "b" * 29}, {"prompt": "a", "response": "b" * 30}]
    provider = Pairs(config(), "train", CharTokenizer())
    assert provider.num_examples == 1 and len(provider.examples[0]["input_ids"]) == 32


class Carved(Pairs):
    train_val_split_ratio = 0.75


def test_a_single_split_source_is_carved_once_into_disjoint_splits(expect_error):
    Carved.rows = [{"prompt": "p", "response": str(i)} for i in range(8)]
    Carved.loads = 0
    train = Carved(config(), "train", CharTokenizer())
    validation = Carved(config(), "validation", CharTokenizer())
    assert (train.num_examples, validation.num_examples, Carved.loads) == (6, 2, 1)
    rows = [tuple(example["input_ids"]) for split in (train, validation) for example in split.examples]
    assert len(set(rows)) == 8
    with expect_error(ValueError, "'test'"):
        Carved(config(), "test", CharTokenizer())


def test_the_carve_out_is_redone_after_clear_cache():
    Carved.rows = [{"prompt": "p", "response": str(i)} for i in range(4)]
    Carved.loads = 0
    Carved(config(), "train", CharTokenizer())
    clear_cache()
    Carved(config(), "train", CharTokenizer())
    assert Carved.loads == 2


def numbered(count):
    """``count`` examples whose response, and so whose last token before EOS, tells them apart."""
    return [{"prompt": "p", "response": chr(ord("a") + i)} for i in range(count)]


def batch_examples(batch):
    """The examples of a batch, each as its response token."""
    return [int(row[2]) for row in batch["input_ids"].reshape(len(batch["input_ids"]), -1)]


def test_a_provider_serves_an_epoch_batch_by_batch_and_then_the_next_one():
    Pairs.rows = numbered(5)
    provider = Pairs(config(), "validation", CharTokenizer())
    # 5 examples make 2 full batches of 2; the partial one is dropped.
    assert (provider.num_examples, len(provider)) == (5, 2)

    first, second = provider.next_batch(), provider.next_batch()
    assert set(first) == {"input_ids", "labels", "loss_mask"} and first["input_ids"].shape == (2, 1, 1, 32)
    assert first["input_ids"].dtype == np.uint32 and first["loss_mask"].dtype == np.float32
    # Validation is served in the order of the examples.
    tokens = [example["input_ids"][2] for example in provider.examples]
    assert batch_examples(first) + batch_examples(second) == tokens[:4]
    # None ends the epoch, and the next call starts the next one.
    assert provider.next_batch() is None
    assert batch_examples(provider.next_batch()) == tokens[:2]


def test_iterating_a_provider_takes_the_batches_left_in_the_epoch():
    Pairs.rows = numbered(6)
    provider = Pairs(config(), "validation", CharTokenizer())
    provider.next_batch()
    assert len(list(provider)) == 2
    assert len(list(provider)) == 3


def test_the_train_split_is_shuffled_by_seed_and_epoch_and_seek_continues_it():
    Pairs.rows = numbered(8)

    def epoch(provider):
        return [tuple(batch_examples(batch)) for batch in provider]

    provider = Pairs(config(), "train", CharTokenizer())
    first, second = epoch(provider), epoch(provider)
    assert sorted(token for batch in first for token in batch) == sorted(
        example["input_ids"][2] for example in provider.examples
    )
    assert first != second

    # Another provider of the same run makes the same epochs, from any batch of them.
    other = Pairs(config(), "train", CharTokenizer())
    other.seek(1, 2)
    assert epoch(other) == second[2:]
    other.seek(0)
    assert epoch(other) == first
    assert epoch(Pairs(config(seed=1), "train", CharTokenizer())) != first


def test_sst2_prompts():
    prompt, response = SST2Provider._prompt_and_response(None, {"sentence": "a gripping film", "label": 1})
    assert prompt == "Review: a gripping film\nOutput:"
    assert response == '{"label": "positive"}'
    assert SST2Provider._prompt_and_response(None, {"sentence": "x", "label": 0})[1] == '{"label": "negative"}'


def test_alpaca_prompts():
    intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    with_input = AlpacaProvider._prompt_and_response(None, {"instruction": "Add.", "input": "1 2", "output": "3"})
    assert with_input == (f"\n{intro}\n\n### Instruction:\nAdd.\n\n### Input:\n1 2\n\n### Response:\n", "3")
    without = AlpacaProvider._prompt_and_response(None, {"instruction": "Hi.", "input": " ", "output": "Hello"})
    assert without == (f"\n{intro}\n\n### Instruction:\nHi.\n\n### Response:\n", "Hello")


def test_custom_alpaca_prompts():
    mapping = {"instruction": "q", "input": "ctx", "output": "a"}
    prompt, response = custom_prompt_and_response({"q": "Add.", "ctx": "1 2", "a": "3"}, "alpaca", mapping)
    assert prompt.endswith("### Instruction:\nAdd.\n\n### Input:\n1 2\n\n\n### Response:\n") and response == "3"
    prompt, _ = custom_prompt_and_response({"q": "Hi.", "ctx": "", "a": "Hello"}, "alpaca", mapping)
    assert prompt.endswith("### Instruction:\nHi.\n\n\n\n### Response:\n")


def test_column_mapping_fills_in_columns_named_like_the_fields():
    assert resolve_column_mapping("alpaca", None, {"instruction", "input", "output", "id"}) == {
        "instruction": "instruction",
        "input": "input",
        "output": "output",
    }
    assert resolve_column_mapping("alpaca", {"output": "answer"}, {"instruction", "answer"}) == {
        "instruction": "instruction",
        "output": "answer",
    }


@pytest.mark.parametrize(
    "mapping, columns, error",
    [
        ({"output": "answer"}, {"instruction", "output"}, "no dataset columns \\['answer'\\]"),
        ({"context": "c"}, {"instruction", "output", "c"}, "reads no \\['context'\\]"),
        (None, {"instruction", "text"}, "needs \\['output'\\]"),
    ],
)
def test_column_mapping_errors(mapping, columns, error, expect_error):
    with expect_error(ValueError, error):
        resolve_column_mapping("alpaca", mapping, columns)


def test_an_unknown_template_is_an_error(expect_error):
    with expect_error(ValueError, "dataset.custom_options.template"):
        resolve_column_mapping("chatml", None, {"instruction", "output"})


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def test_a_custom_dataset_from_jsonl_files(tmp_path):
    train = write_jsonl(tmp_path / "train.jsonl", [{"question": "Hi?", "answer": "Yes"}] * 3)
    custom = {
        "file_type": "jsonl",
        "train_dataset_path": train,
        "column_mapping": {"instruction": "question", "output": "answer"},
    }
    run_config = config(dataset={"id": "custom", "custom_options": custom}, max_length=256)
    provider = get_provider(run_config, "train", CharTokenizer())
    assert isinstance(provider, CustomProvider) and provider.num_examples == 3
    assert len(provider) == 1 and provider.next_batch()["input_ids"].shape == (2, 1, 1, 256)
    # Without val_dataset_path there is no validation split, and no provider for it.
    assert get_provider(run_config, "validation", CharTokenizer()) is None


def test_a_custom_dataset_needs_a_train_file(expect_error):
    run_config = config(dataset={"id": "custom", "custom_options": {"file_type": "json"}})
    with expect_error(ValueError, "dataset.custom_options.train_dataset_path is required"):
        get_provider(run_config, "train", CharTokenizer())


def test_dataset_ids(expect_error):
    assert sorted(providers()) == ["alpaca", "custom", "sst2"]
    with expect_error(ValueError, "dataset.id: 'imdb' is not one of"):
        get_provider(config(dataset={"id": "imdb"}), "train", CharTokenizer())
