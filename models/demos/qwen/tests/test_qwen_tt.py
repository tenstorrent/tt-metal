# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import ttnn
from models.demos.qwen.tt.qwen_common import HostEmbedding, get_single_rot_mat
from models.demos.qwen.tt.model_config import TtModelArgs
from models.demos.qwen.tt.qwen_model import TtTransformer

from loguru import logger


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 1),
    ],
    indirect=True,
)
def test_qwen_torch_inference(mesh_device, ensure_gc):
    mesh_device.enable_async(True)
    iterations = 200

    model_args = TtModelArgs(mesh_device, instruct=True)
    state_dict = model_args.load_state_dict()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    prompts = [
        "Do you have mayonnaise recipes? Mayonnaise is a versatile ingredient that can be used in countless recipes beyond just a sandwich spread. What are some of your favorite ways to use mayonnaise in cooking or baking? Do you have a special recipe for a creamy potato salad, a tangy coleslaw, or perhaps a savory dip for vegetables and chips? Mayonnaise can also be used as a base for homemade dressings and sauces, adding richness and flavor to your dishes. Have you tried baking with mayonnaise to keep cakes moist and tender? Share any recipes, tips, or creative uses you have for mayonnaise. How did you discover these recipes, and do you have any variations that you particularly enjoy?"
    ] * model_args.max_batch_size
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

    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
    )
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
        current_rot_mat, rot_matrix = get_single_rot_mat(
            model_args.head_dim,
            mesh_device,
            model_args.num_devices,
            start_pos=start_pos,
        )
        pt_decode_input = model_args.prepare_inputs_ttnn_decode(
            pt_decode_input,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        current_pos_tensor = ttnn.from_torch(
            torch.tensor([start_pos]),
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_out = tt_model(pt_decode_input, current_pos_tensor, rot_mat=current_rot_mat)
        ref_output = ttnn.to_torch(tt_out)
        # While in "prefill" mode, use the prompt tokens as the output
        if i in range(len(encoded_prompts[0])):
            all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs
            pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(model_args.max_batch_size, seqlen, -1)
        else:
            pt_out_tok = torch.argmax(ref_output, dim=-1)[:, :, :1]
            pt_decode_input = embd(pt_out_tok)

            all_outputs_ref.append(
                pt_out_tok.squeeze(1).tolist()[0][0]
            )  # Update generated token to list of ref outputs

        # TODO print all 32 users
        logger.info("[User 0] Ref generation: " + tokenizer.decode(all_outputs_ref))
