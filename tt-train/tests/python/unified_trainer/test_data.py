# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Datasets: prompt templates, tokenization with the prompt masked, filtering and splits. Device-free."""

import json

import pytest

from ttml.unified_trainer import SFTTrainerConfig
from ttml.unified_trainer.data import (
    DATASETS,
    IGNORED_INDEX,
    AlpacaDataset,
    CustomDataset,
    LLMDataset,
    SST2Dataset,
    build_dataloader,
    clear_cache,
    get_dataset,
)
from ttml.unified_trainer.data.templates import custom_prompt_and_response, resolve_column_mapping

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


class Pairs(LLMDataset):
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
    dataset = Pairs(config(), "train", CharTokenizer())
    sample = dataset[0]
    assert sample["input_ids"][0] == BOS and sample["input_ids"][-1] == EOS
    assert len(sample["input_ids"]) == 1 + 4 + 1
    # BOS and the 2 prompt characters are masked; the response and EOS are trained on.
    assert sample["labels"][:3] == [IGNORED_INDEX] * 3
    assert sample["labels"][3:] == sample["input_ids"][3:]


def test_examples_longer_than_max_length_are_dropped():
    Pairs.rows = [{"prompt": "a", "response": "b" * 29}, {"prompt": "a", "response": "b" * 30}]
    dataset = Pairs(config(), "train", CharTokenizer())
    assert len(dataset) == 1 and len(dataset[0]["input_ids"]) == 32


class Carved(Pairs):
    train_val_split_ratio = 0.75


def test_a_single_split_source_is_carved_once_into_disjoint_splits(expect_error):
    Carved.rows = [{"prompt": "p", "response": str(i)} for i in range(8)]
    Carved.loads = 0
    train = Carved(config(), "train", CharTokenizer())
    validation = Carved(config(), "validation", CharTokenizer())
    assert (len(train), len(validation), Carved.loads) == (6, 2, 1)
    rows = [tuple(train[i]["input_ids"]) for i in range(6)] + [tuple(validation[i]["input_ids"]) for i in range(2)]
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


def test_sst2_prompts():
    prompt, response = SST2Dataset._prompt_and_response(None, {"sentence": "a gripping film", "label": 1})
    assert prompt == "Review: a gripping film\nOutput:"
    assert response == '{"label": "positive"}'
    assert SST2Dataset._prompt_and_response(None, {"sentence": "x", "label": 0})[1] == '{"label": "negative"}'


def test_alpaca_prompts():
    intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    with_input = AlpacaDataset._prompt_and_response(None, {"instruction": "Add.", "input": "1 2", "output": "3"})
    assert with_input == (f"\n{intro}\n\n### Instruction:\nAdd.\n\n### Input:\n1 2\n\n### Response:\n", "3")
    without = AlpacaDataset._prompt_and_response(None, {"instruction": "Hi.", "input": " ", "output": "Hello"})
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
    dataset = get_dataset(run_config, "train", CharTokenizer())
    assert isinstance(dataset, CustomDataset) and len(dataset) == 3
    # Without val_dataset_path there is no validation split, and no dataloader for it.
    validation = get_dataset(run_config, "validation", CharTokenizer())
    assert validation.dataset is None and len(validation) == 0
    assert build_dataloader(validation, run_config, shuffle=False) is None


def test_a_custom_dataset_needs_a_train_file(expect_error):
    run_config = config(dataset={"id": "custom", "custom_options": {"file_type": "json"}})
    with expect_error(ValueError, "dataset.custom_options.train_dataset_path is required"):
        get_dataset(run_config, "train", CharTokenizer())


def test_dataset_ids(expect_error):
    assert sorted(DATASETS) == ["alpaca", "custom", "sst2"]
    with expect_error(ValueError, "dataset.id: 'imdb' is not one of"):
        get_dataset(config(dataset={"id": "imdb"}), "train", CharTokenizer())
