# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import json
from pathlib import Path
import os

# import ttnn
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    precompute_freqs,
)
from models.demos.wormhole.mistral7b.reference.model import Transformer
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.demos.wormhole.mistral7b.reference.tokenizer import Tokenizer

# from transformers.generation.utils import top_k_top_p_filtering


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


def test_mistral_torch_inference():
    iterations = 20

    model_args = TtModelArgs(device=None)
    state_dict = torch.load(model_args.consolidated_weights_path)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    prompts = ["1 2 3 4 "] * 32
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    generation_start_pos = 0
    generation_length = iterations

    seqlen = 1  # Generating one token per user at a time
    batch = 32

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    all_outputs_ref = []

    # After loading the model weights, wait for an input to start the generation
    # print("Waiting for an input to start...")
    # input()

    for i in range(generation_length):
        print(f"[Decode] Generating token {i}")

        start_pos = generation_start_pos + i

        freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
        positions = torch.tensor([start_pos])
        ref_output = reference_model(pt_decode_input, freqs_cis_i, positions)

        # While in "prefill" mode, use the prompt tokens as the output
        if i in range(len(encoded_prompts[0])):
            all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs
            pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # pt_out_tok = torch.argmax(torch.nn.functional.log_softmax(ref_output, dim=-1), dim=-1)
            pt_out_tok = torch.argmax(ref_output, dim=-1)
            # pt_out_tok_logscores = top_k_top_p_filtering(ref_output.squeeze(1), top_k=0, top_p=0.9)
            # probs = torch.nn.functional.softmax(pt_out_tok_logscores, dim=-1)
            # pt_out_tok = torch.multinomial(probs, num_samples=1)#.squeeze(1)

            pt_decode_input = embd(pt_out_tok)

            all_outputs_ref.append(pt_out_tok.squeeze(1).tolist()[0])  # Update generated token to list of ref outputs

        # TODO print all 32 users
        print("[User 0] Ref generation: ", "".join(tokenizer.decode(all_outputs_ref)))
