# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch

# import ttnn
from models.demos.qwen.tt.qwen_common import HostEmbedding, precompute_freqs
from models.demos.qwen.tt.model_config import TtModelArgs
from models.demos.qwen.reference.model import Transformer
from models.demos.qwen.reference.tokenizer import Tokenizer

from loguru import logger


@torch.no_grad()
def test_qwen_torch_inference(ensure_gc):
    iterations = 200

    model_args = TtModelArgs(mesh_device=None)
    state_dict = model_args.load_state_dict()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    prompts = ["What is life?"] * model_args.max_batch_size
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]
    input_prompts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]
    encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]
    state_dict_prefix = model_args.get_state_dict_prefix("", None)

    reference_state_dict = {
        k[len(state_dict_prefix) :]: v
        for k, v in state_dict.items()
        if (
            any([f"{state_dict_prefix}layers.{i}." in k for i in range(model_args.n_layers)])
            or any(
                [f"{state_dict_prefix}{name}" in k for name in ["embed_tokens.weight", "norm.weight", "lm_head.weight"]]
            )
        )
    }
    del reference_state_dict["embed_tokens.weight"]
    reference_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]
    reference_model = Transformer(model_args)
    reference_model.load_state_dict(reference_state_dict)

    # Embedding on host
    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}embed_tokens.weight"]})

    generation_start_pos = 0
    generation_length = iterations

    seqlen = 1  # Generating one token per user at a time

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(model_args.max_batch_size, seqlen, -1)
    logger.info(pt_decode_input.shape)

    all_outputs_ref = []

    for i in range(generation_length):
        logger.info(f"[Decode] Generating token {i}")

        start_pos = generation_start_pos + i
        cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
        freqs_cis = torch.complex(cos, sin)
        freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
        positions = torch.tensor([start_pos])
        ref_output = reference_model(pt_decode_input, freqs_cis_i, positions)

        # While in "prefill" mode, use the prompt tokens as the output
        if i in range(len(encoded_prompts[0])):
            all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs
            pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(model_args.max_batch_size, seqlen, -1)
        else:
            pt_out_tok = torch.argmax(ref_output, dim=-1)
            pt_decode_input = embd(pt_out_tok)

            all_outputs_ref.append(pt_out_tok.squeeze(1).tolist()[0])  # Update generated token to list of ref outputs
        # TODO print all 32 users
        logger.info("[User 0] Ref generation: " + tokenizer.decode(all_outputs_ref))
