# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
import torch
from models.demos.wormhole.mistral7b.reference.model import Transformer, precompute_freqs_cis
from models.demos.wormhole.mistral7b.reference.tokenizer import Tokenizer
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs


def main():
    model_args = TtModelArgs()
    state_dict = torch.load(model_args.consolidated_weights_path)
    tokenizer = Tokenizer(model_args.tokenizer_path)
    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)

    freqs_cis = precompute_freqs_cis(model_args.head_dim, model_args.max_seq_len * 2)
    res = generate(["1, 2, 3, 4, "], reference_model, freqs_cis, tokenizer, 20)
    print(res)


def generate(prompts, model, freqs_cis, tokenizer, max_tokens):
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="cpu")
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != tokenizer.pad_id

    # pre-fill
    positions = torch.arange(0, min_prompt_len)
    freqs_cis_i = freqs_cis[0, :].unsqueeze(0)
    logits = model.forward(input_tokens[:, :min_prompt_len], freqs_cis_i, positions)
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    # decode
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
        pos = torch.LongTensor([cur_pos]).to(next_token)
        freqs_cis_i = freqs_cis[pos, :].unsqueeze(0)
        logits = model.forward(next_token[:, None], freqs_cis_i, pos)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        cur_pos += 1

    all_logprobs = torch.cat(all_logprobs, 1)
    res = []
    if max_tokens > 0:
        generated = torch.cat(generated, 1)

        for i, x in enumerate(encoded_prompts):
            res.append(tokenizer.decode(x[:min_prompt_len] + generated[i].tolist()))
    return res, all_logprobs


if __name__ == "__main__":
    main()
