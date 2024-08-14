# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

#################################################################################################################
# Link: https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py
# NOTE: changed the device from CUDA to CPU and dtype to float32
#################################################################################################################

# coding=utf-8
# Copyright 2023 Mistral AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sentencepiece import SentencePieceProcessor
from pathlib import Path
<<<<<<< HEAD:models/demos/wormhole/mistral7b/reference/tokenizer.py
from typing import List
import torch
from torch import nn

from models.demos.wormhole.mistral7b.reference.model import Transformer
=======
from models.utility_functions import tt_to_torch_tensor
import json
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from tt_lib.utils import pad_weight
>>>>>>> #11433: Replace tt_lib in mistral:models/experimental/mistral/mistral_utils.py


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def n_words(self) -> int:
        return self._model.vocab_size()

    @property
    def bos_id(self) -> int:
        return self._model.bos_id()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str, bos: bool = True) -> List[int]:
        assert isinstance(s, str)
        t = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)


def generate(prompts: List[str], model: Transformer, tokenizer: Tokenizer, max_tokens: int):
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="cpu")
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != tokenizer.pad_id

    positions = torch.arange(0, min_prompt_len).to("cpu")
    logits = model.forward(input_tokens[:, :min_prompt_len], positions)
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
        logits = model.forward(next_token[:, None], torch.LongTensor([cur_pos]).to(next_token))
        logprobs = nn.functional.log_softmax(logits, dim=-1)
        cur_pos += 1

    all_logprobs = torch.cat(all_logprobs, 1)
    res = []
    if max_tokens > 0:
        generated = torch.cat(generated, 1)

        for i, x in enumerate(encoded_prompts):
            res.append(tokenizer.decode(x[:min_prompt_len] + generated[i].tolist()))
<<<<<<< HEAD:models/demos/wormhole/mistral7b/reference/tokenizer.py
    return res, all_logprobs
=======
    return res


def cache_weights_in_weka(model_location_generator, device, dtype, reset_seeds):
    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))
    weights_dtype = dtype

    # initial weights are stored in "models/experimental/mistral/weights/" and moved to weka path
    file_name = "models/experimental/mistral/weights/"
    for key, value in state_dict.items():
        if "tok_embeddings" in key:
            torch.save(value, file_name + str(key) + ".pt")
            continue
        if len(value.shape) == 1:
            value = value.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            value = value.unsqueeze(0).unsqueeze(0)
        if value.shape[-2] % 32 == 0 and value.shape[-1] % 32 == 0:
            value = ttnn.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                weights_dtype,
                ttnn.ROW_MAJOR_LAYOUT,
            ).to(ttnn.TILE_LAYOUT)
        else:
            value = pad_weight(value)
            value = ttnn.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                weights_dtype,
                ttnn.ROW_MAJOR_LAYOUT,
            ).to(ttnn.TILE_LAYOUT)
        ttnn.experimental.tensor.dump_tensor(file_name + str(key) + str(weights_dtype) + ".bin", value)
>>>>>>> #11433: Replace tt_lib in mistral:models/experimental/mistral/mistral_utils.py
