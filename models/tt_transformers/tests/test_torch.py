# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger

# import ttnn
from models.tt_transformers.tt.model_config import ModelArgs


@torch.no_grad()
def test_torch_inference(ensure_gc):
    iterations = 20

    model_args = ModelArgs(mesh_device=None)
    state_dict = model_args.load_state_dict()
    tokenizer = model_args.tokenizer

    prompts = ["1 2 3 4 "] * model_args.max_batch_size
    encoded_prompts = [model_args.encode_prompt(prompt, instruct=False) for prompt in prompts]

    reference_model = model_args.reference_transformer()
    reference_model.load_state_dict(state_dict, fuse_qkv=model_args.fuse_qkv, fuse_mlp=model_args.fuse_mlp)

    # Embedding on host
    embd = model_args.reference_embedding()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

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

        ref_output = reference_model(pt_decode_input, start_pos)

        # While in "prefill" mode, use the prompt tokens as the output
        if i in range(len(encoded_prompts[0])):
            all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs
            pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(model_args.max_batch_size, seqlen, -1)
        else:
            # pt_out_tok = torch.argmax(torch.nn.functional.log_softmax(ref_output, dim=-1), dim=-1)
            pt_out_tok = torch.argmax(ref_output, dim=-1)
            # pt_out_tok_logscores = top_k_top_p_filtering(ref_output.squeeze(1), top_k=0, top_p=0.9)
            # probs = torch.nn.functional.softmax(pt_out_tok_logscores, dim=-1)
            # pt_out_tok = torch.multinomial(probs, num_samples=1)#.squeeze(1)

            pt_decode_input = embd(pt_out_tok)

            all_outputs_ref.append(pt_out_tok.squeeze(1).tolist()[0])  # Update generated token to list of ref outputs

        # TODO print all 32 users
        logger.info("[User 0] Ref generation: '" + "".join(tokenizer.decode(all_outputs_ref)) + "'")
