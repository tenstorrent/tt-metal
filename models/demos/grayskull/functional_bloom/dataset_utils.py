# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset

from models.datasets.dataset_squadv2 import squad_divide_chunks


class InputExample(object):
    def __init__(self, input_sentence, endings=None, label=None):
        self.input_sentence = input_sentence
        self.endings = endings
        self.label = label


def get_input(item):
    if "ctx_a" not in item:
        return item["ctx"]
    if "ctx" not in item:
        return item["ctx_a"]
    if len(item["ctx"]) == len(item["ctx_a"]):
        return item["ctx"]
    return item["ctx_a"]


def get_endings(item):
    if ("ctx_b" not in item) or len(item["ctx_b"]) == 0:
        return item["endings"]
    return ["{} {}".format(item["ctx_b"], x) for x in item["endings"]]


def get_data(input_loc):
    examples = []
    with open(input_loc, "r") as file:
        for data in file:
            item = json.loads(data)
            examples.append(
                InputExample(
                    input_sentence=get_input(item),
                    endings=get_endings(item),
                    label=item["label"],
                )
            )

    return examples


class SQUADV2Dataset(Dataset):
    def __init__(
        self,
        dataset_question: Any,
        dataset_context: Any,
        dataset_reference: Any,
        tokenizer: Any,
        seq_len: int,
        attention_mask: bool,
        token_type_ids: bool,
    ):
        self.data = []
        for i in range(len(dataset_question)):
            self.data.append(
                (
                    tokenizer.batch_encode_plus(
                        list(zip(dataset_question[i], dataset_context[i])),
                        max_length=seq_len,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=attention_mask,
                        return_token_type_ids=token_type_ids,
                        return_tensors="pt",
                    ),
                    dataset_reference[i],
                    dataset_question[i],
                    dataset_context[i],
                )
            )

    def __getitem__(self, index: int):
        X = self.data[index]
        return X


def squadv2_1K_samples_input(tokenizer, seq_len, attention_mask, token_type_ids, microbatch=8):
    squadv2_dataset = load_dataset("squad_v2", use_auth_token=False, streaming=True)["validation"]
    dataset_iter = iter(squadv2_dataset)
    dataset_question = []
    dataset_context = []
    dataset_reference = []

    for _ in range(2048):
        dataset_sgl = next(dataset_iter)
        if len(dataset_sgl["answers"]["text"]) > 0:
            dataset_question.append(dataset_sgl["question"])
            dataset_context.append(dataset_sgl["context"])
            dataset_reference.append({"answers": dataset_sgl["answers"], "id": dataset_sgl["id"]})
        if len(dataset_question) == 1024:
            logger.info("SQuADv2 1024 samples load ..done")
            break

    dataset_question, dataset_context, dataset_reference = squad_divide_chunks(
        dataset_question, dataset_context, dataset_reference, microbatch
    )
    dataset_processed = SQUADV2Dataset(
        dataset_question,
        dataset_context,
        dataset_reference,
        tokenizer=tokenizer,
        seq_len=seq_len,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    return dataset_processed
