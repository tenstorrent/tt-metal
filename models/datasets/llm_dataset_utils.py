# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import re
import datasets
from sklearn.metrics import top_k_accuracy_score
import numpy as np
from loguru import logger


def wikitext_detokenizer(string):
    """From Megatron-DeepSpeed/tasks/zeroshot_gpt/detokenizer.py"""
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def prepare_textgen_dataset(dataset_name, dataset_config, split):
    dataset = datasets.load_dataset(dataset_name, dataset_config, split=split, ignore_verifications=True)
    if dataset_name == "wikitext":
        dataset = wikitext_detokenizer("\n".join(dataset["text"]))
    else:
        assert False, f"Dataset {dataset_name} is not currently supported"
    return dataset


def prepare_textgen_dataloader(
    encodings,  # Tokenized dataset (1D torch.tensor)
    batch_size,  # Number of prompts per batch
    seq_len,  # Seq len of each prompt
    num_samples=None,  # Total number of prompts
    stride=None,
):  # Number of tokens between prompts
    num_tokens_dataset = encodings.shape[0]
    if stride is None:
        stride = seq_len
    if num_samples is None:
        num_samples = (num_tokens_dataset - seq_len) // stride

    assert stride <= seq_len, "Stride must be less than or equal to seq_len"
    assert (
        num_samples - 1
    ) * stride + seq_len < num_tokens_dataset, f"The dataset ({num_tokens_dataset} tokens) is too small to generate {num_samples} samples with stride {stride}."

    inputs = []
    labels = []
    for begin_loc in range(0, num_samples * stride, stride):
        end_loc = begin_loc + seq_len
        inputs.append(encodings[begin_loc:end_loc])
        labels.append(encodings[begin_loc + 1 : end_loc + 1])

    dataset = torch.utils.data.TensorDataset(torch.stack(inputs), torch.stack(labels))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return dataloader


def calculate_acc_metrics(
    logits: torch.tensor, labels: torch.tensor  # [batch_size * seq_len, vocab_size]  # [batch_size * seq_len]
):
    # Calculate negative log-likelihood
    nll = torch.nn.functional.cross_entropy(logits, labels).item()

    # Calculate top-1/5 accuracy
    logits = logits.float().detach().numpy()
    labels = labels.float().detach().numpy()
    top1 = top_k_accuracy_score(labels, logits, k=1, labels=np.arange(logits.shape[-1]))
    top5 = top_k_accuracy_score(labels, logits, k=5, labels=np.arange(logits.shape[-1]))

    return nll, top1, top5


def verify_acc_metrics(calculated_metrics: dict, expected_metrics: dict):
    if (
        calculated_metrics["ppl"] > expected_metrics["ppl"]
        or calculated_metrics["top1_acc"] < expected_metrics["top1_acc"]
        or calculated_metrics["top5_acc"] < expected_metrics["top5_acc"]
    ):
        assert (
            False
        ), f"At least one of Perplexity {calculated_metrics['ppl']}, Top1-Acc {calculated_metrics['top1_acc']}, or Top5-Acc {calculated_metrics['top5_acc']} is worse (higher for perplexity or lower for acc) than {expected_metrics}"
    elif (
        calculated_metrics["ppl"] < expected_metrics["ppl"] * 0.96
        or calculated_metrics["top1_acc"] > expected_metrics["top1_acc"] * 1.04
        or calculated_metrics["top5_acc"] > expected_metrics["top5_acc"] * 1.04
    ):
        assert (
            False
        ), f"At least one of Perplexity {calculated_metrics['ppl']}, Top1-Acc {calculated_metrics['top1_acc']}, or Top5-Acc {calculated_metrics['top5_acc']} is better (lower for perplexity or higher for acc) than {expected_metrics}. Please update the expected targets."
    logger.info("Perplexity/Accuracy Check Passed!")
