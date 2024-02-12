# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import List
from sentencepiece import SentencePieceProcessor
from pathlib import Path
import tt_lib
import ttnn


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path

        self._model = SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)


def generate(config, prompts: List[str], model, tokenizer: Tokenizer, parameters, max_tokens: int, device):
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="cpu")
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != tokenizer.pad_id

    positions = torch.arange(0, min_prompt_len)
    output_mem_config = tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
    )
    ttnn_input = ttnn.from_torch(input_tokens, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    ttnn_input = ttnn.to_device(ttnn_input, device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.ROW_MAJOR_LAYOUT)
    logits = model(config, ttnn_input[:, :min_prompt_len], positions, parameters, device, output_mem_config)
    logits = ttnn.to_torch(logits)
    logprobs = nn.functional.log_softmax(logits, dim=-1)

    generated = []
    all_logprobs = [
        logprobs[:, :-1, :].gather(2, input_tokens[:, 1:min_prompt_len, None]).squeeze(-1),
    ]
    cur_pos = min_prompt_len
    for _ in range(max_tokens):
        next_token = torch.argmax(logprobs[:, -1, :], dim=-1)
        if cur_pos < input_mask.shape[1]:
            next_token = torch.where(input_mask[:, cur_pos], input_tokens[:, cur_pos], next_token)
        all_logprobs.append(
            logprobs[:, -1, :].gather(1, next_token[:, None]),
        )
        generated.append(next_token[:, None])

        input = ttnn.from_torch(next_token[:, None], dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
        input = ttnn.to_layout(input, layout=ttnn.ROW_MAJOR_LAYOUT)

        logits = model(config, input, torch.LongTensor([cur_pos]).to(next_token), parameters, device, output_mem_config)
        logits = ttnn.to_torch(logits)
        logprobs = nn.functional.log_softmax(logits, dim=-1)
        cur_pos += 1

    all_logprobs = torch.cat(all_logprobs, 1)
    res = []
    if max_tokens > 0:
        generated = torch.cat(generated, 1)

        for i, x in enumerate(encoded_prompts):
            res.append(tokenizer.decode(x[:min_prompt_len] + generated[i].tolist()))
    return res
