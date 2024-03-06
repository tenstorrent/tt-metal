# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 Tri Dao, Albert Gu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from transformers import AutoTokenizer

import tt_lib


def generate_through_selective_scan(
    model, tokenizer, prompt: str, n_tokens_to_gen: int = 30, sample: bool = False, top_k: Optional[int] = None
):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, n_tokens_to_gen, sample=sample, top_k=top_k)
    return [tokenizer.decode(out.tolist()) for out in output][0]


def generate_through_decode(model, tokenizer, prompt: str, n_tokens_to_gen: int = 51):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, n_tokens_to_gen)
    return tokenizer.batch_decode(output)[0]


def test_decode():
    from model import Mamba

    # One of:
    #     'state-spaces/mamba-2.8b-slimpj'
    #     'state-spaces/mamba-2.8b'
    #     'state-spaces/mamba-1.4b'
    #     'state-spaces/mamba-790m'
    #     'state-spaces/mamba-370m'
    #     'state-spaces/mamba-130m'
    pretrained_model_name = "state-spaces/mamba-370m"
    prompt = "Who is the captain of indian cricket team?"
    model = Mamba.from_pretrained(pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print("Output from selective scan:")
    print(generate_through_selective_scan(model, tokenizer, prompt))

    from decode_model import MambaDecode

    model_decode = MambaDecode.from_pretrained(pretrained_model_name)
    print("Output from decode only mode:")
    print(generate_through_decode(model_decode, tokenizer, prompt))

    from models.experimental.mamba.tt.full_model import MambaTT

    tt_device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(tt_device)
    tt_device = tt_lib.device.GetDefaultDevice()
    hf_reference_model = MambaDecode.from_pretrained(pretrained_model_name)
    tt_model_decode = MambaTT(hf_reference_model, tt_device)
    print("Output from decode only mode through metal:")
    print(generate_through_decode(tt_model_decode, tokenizer, prompt))


if __name__ == "__main__":
    test_decode()
