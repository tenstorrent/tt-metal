# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import re
import datasets


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
