# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
import ttnn

from models.generation_utils import get_logits_processor

mem_config = ttnn.L1_MEMORY_CONFIG


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = ttnn.Tensor(py_tensor.reshape(size), ttnn.bfloat16).to(tt_device)

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu()
    if tt_output.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        tt_output = tt_output.to(ttnn.ROW_MAJOR_LAYOUT)
    return tt_output.to_torch()


def tt_const_tensor(value, shape, device):
    pytorch_const = torch.full(shape, value)
    tt_const = torch2tt_tensor(pytorch_const, device)
    return tt_const


def pad_input_tensor(tensor, value, multiple):
    len = tensor.shape[1]

    if len % multiple == 0:
        return tensor

    padded_len = ((len // multiple) + 1) * multiple

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def tt_matmul(t1, t2, device, on_torch=False):
    if on_torch:
        t1 = tt2torch_tensor(t1)
        t2 = tt2torch_tensor(t2)

        res = torch.matmul(t1, t2, output_mem_config=mem_config)
        return torch2tt_tensor(res, device)
    else:
        return ttnn.matmul(t1, t2, mem_config)


def tt_bmm(t1, t2, device, on_torch=False):
    if on_torch:
        return tt_matmul(t1, t2, device)
    else:
        return ttnn.matmul(t1, t2, mem_config)


def read_model_config(json_file):
    # read file
    with open(json_file, "r") as myfile:
        data = myfile.read()

    # parse file
    obj = json.loads(data)
    return obj


def pad_input_32(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def run_generate(hf_reference_model, tt_model, tokenizer, input_sentance, max_tokens, device):
    # Prepare input
    tokenized = tokenizer(input_sentance, return_tensors="pt")  # Batch size 1
    generation_config = hf_reference_model.generation_config

    input_ids = tokenized.input_ids

    input_ids = pad_input_tensor(tokenized.input_ids, generation_config.pad_token_id, 2)

    # Start to generate i'th token
    i = input_ids.shape[1]

    logits_processor = get_logits_processor(input_ids, hf_reference_model.config)

    # Input IDs expansion
    input_ids_expansion = generation_config.pad_token_id * torch.ones(1, 2).to(torch.long)

    while i < max_tokens:
        tt_out = tt_model.forward(device, input_ids=input_ids, return_dict=False)
        next_token_logits = tt_out[0]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if next_tokens[0][i - 1] == generation_config.eos_token_id:
            break

        # We need to expand decoder_input_ids
        if i % 2 == 0:
            input_ids = torch.cat([input_ids, input_ids_expansion], dim=1)

        # Append predicted token
        input_ids[0][i] = next_tokens[0][i - 1]
        i += 1

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
