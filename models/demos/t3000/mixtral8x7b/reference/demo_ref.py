# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
import json

from models.demos.t3000.mixtral8x7b.tt.mixtral_common import load_inputs
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.demos.t3000.mixtral8x7b.reference.model import Transformer
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


def preprocess_inputs(input_prompts, tokenizer, model_args, embd):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]

    # Pad the inputs to the max length prompt
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(input_prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long)

    # TODO Change padding to be left padding instead of right padding
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != tokenizer.pad_id

    num_users = len(encoded_prompts)
    logger.info(f"# of users: {num_users}")
    return input_tokens, max_prompt_len, input_mask


@torch.no_grad()
def run_mixtral_demo(user_input, batch_size):
    assert batch_size == 32, "Batch size must be 32"

    logger.info(f"Reading inputs...")
    if len(user_input) == 1:
        input_prompts = user_input * 32  # Always process 32 users
    else:
        input_prompts = load_inputs(user_input, 32)

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs()
    tokenizer = Tokenizer(model_args.tokenizer_path)

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.state_dict_path)

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    pt_encoded_input, max_prompt_len, input_mask = preprocess_inputs(
        input_prompts,
        tokenizer,
        model_args,
        embd,
    )

    # Load TTNN mixtral model
    logger.info("Loading weights to device...")
    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)
    reference_model.eval()

    logger.info("Finished loading weights to device. Starting inference...")

    generation_start_pos = 0
    max_generated_tokens = 150
    users_decoding = True

    # Keep track of generated outputs to print out every iteration
    all_outputs = [[] for _ in range(batch_size)]
    user_done = [False] * batch_size  # Keeps track when a user reaches EoD token
    pt_decode_input = embd(pt_encoded_input[:, 0]).view(batch_size, 1, -1)

    iteration = 0
    # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
    while users_decoding:
        start_pos = generation_start_pos + iteration

        positions = torch.LongTensor([start_pos])
        # mask = tt2torch_tensor(attn_mask[0])
        ref_output = reference_model(pt_decode_input, positions)  # mask)
        # If temperature is 0, does greedy decoding (top-1)
        pt_out_tok = torch.argmax(torch.nn.functional.log_softmax(ref_output, dim=-1), dim=-1).squeeze(1)

        if iteration < input_mask.shape[1]:  # If prefill
            # If token is pad token, start generating new token, otherwise, push the next prompt token to the model
            pt_out_tok = torch.where(input_mask[:, iteration], pt_encoded_input[:, iteration], pt_out_tok[:])

        # Save output token to print out later
        for user in range(batch_size):
            user_tok = pt_out_tok[user].item()
            if user_tok != tokenizer.eos_id:  # Stop saving the ouput after hitting the EOS token
                all_outputs[user].append(user_tok)
            else:
                if (
                    iteration < input_mask.shape[1]
                ):  # Still in prefill, so ignore EOS token and save the generated token
                    all_outputs[user].append(user_tok)
                else:
                    logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                    user_done[user] = True
                    if all(user_done):
                        users_decoding = False

        pt_decode_input = embd(pt_out_tok).view(batch_size, 1, -1)

        # Print out generated outputs for each user at the end of every iteration
        if len(user_input) == 1:
            logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
        else:
            for user in range(batch_size):
                logger.info("[User {}] {}".format(user, "".join(tokenizer.decode(all_outputs[user]))))

        iteration += 1

        # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
        if iteration >= max_generated_tokens:
            users_decoding = False


def test_ref_demo(user_input="models/demos/mixtral8x7b/reference/input_data.json", batch_size=32):
    return run_mixtral_demo(user_input=user_input, batch_size=batch_size)
