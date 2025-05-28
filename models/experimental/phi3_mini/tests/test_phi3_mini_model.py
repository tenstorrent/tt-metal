# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.tt_transformers.tt.common import (
    encode_prompt_hf,
    sample_host,
    PagedAttentionConfig,
)
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.experimental.phi3_mini.tt.model_config import Phi3MiniModelArgs
from models.experimental.phi3_mini.tt.phi3_mini_model import Phi3Transformer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull, skip_for_blackhole
from models.tt_transformers.tt.model_config import HfModelWrapper

import re


def parse_chat_output(text):
    pattern = r"<\|(?P<role>user|assistant)\|>\s*(?P<message>.*?)(?=<\|(?:user|assistant|end)\|>|$)"
    matches = re.finditer(pattern, text, re.DOTALL)
    return [(match.group("role"), match.group("message").strip()) for match in matches]


def display_chat(logger, conversation):
    for role, message in conversation:
        if role == "user":
            logger.info(f"👤 User: {message}")
        elif role == "assistant":
            logger.info(f"🤖 Assistant: {message}")


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@skip_for_blackhole("Failing on DRAM harvested P100a, see #21419")
@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "weights, layers",
    [
        ("instruct", None),
    ],
    ids=["full"],
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
    ],
    ids=["performance", "accuracy"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_model_inference(
    weights,
    layers,
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
    request,
    parse_chat=False,
):
    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    dtype = ttnn.bfloat8_b

    test_id = request.node.callspec.id

    mode_accuracy = "accuracy" in test_id
    instruct = True if weights == "instruct" else False

    model_args = Phi3MiniModelArgs(
        mesh_device,
        instruct=instruct,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )

    # Expected PCC for the model
    pcc = 0.94 if mode_accuracy else 0.86

    # Number of decode iterations to run for the model
    iterations = 10

    if layers is not None:
        model_args.n_layers = layers
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)

    prompts = ["This is a test"] * model_args.max_batch_size

    tokenizer = model_args.tokenizer
    if instruct:
        encoded_prompts = encode_prompt_hf(tokenizer=tokenizer, prompt_text=prompts[0])
    else:
        encoded_prompts = [model_args.encode_prompt(prompt, instruct=False) for prompt in prompts]

    if run_ref_pt:
        reference_transformer_model = model_args.reference_transformer(wrap=False)
        reference_model = HfModelWrapper(reference_transformer_model, model_args.head_dim)
        logger.info("Finished loading reference model.")

        # Embedding on host
        embd = model_args.reference_embedding(reference_transformer_model)
    else:
        # Embedding on host
        embd = model_args.reference_embedding()

    generation_start_pos = 0
    generation_length = iterations

    page_table_tt = None
    paged_attention_config = None

    # Prepare page table for paged attention
    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if batch_size > 1 else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Load TTNN model
    tt_model = Phi3Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    logger.info("Model and caches loaded.")

    if run_ref_pt:
        all_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = model_args.max_batch_size

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts).unsqueeze(0)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
    tt_decode_input = pt_decode_input

    # Keep track of generated outputs to print out later
    all_outputs = []
    if run_ref_pt:
        all_outputs_ref = []

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    for i in range(generation_length):
        logger.info(f"[Model] Generating token {i}")

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        # Convert ttnn tensor to torch tensor
        mesh_composer = ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=(3, 1) if model_args.is_galaxy else (1, -1), mesh_shape=model_args.cluster_shape
        )
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=mesh_composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[: model_args.max_batch_size, 0:1, : model_args.vocab_size]
        )

        ttnn.deallocate(tt_out)

        if run_ref_pt:  # Run reference model
            # In this test all users have the same position
            ref_output = reference_model(pt_decode_input, current_pos[0])

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i + 1 for _ in range(batch)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        # Append the generated token to the list of outputs /prefill
        if i in range(len(encoded_prompts)):
            # While in "prefill" mode, use the prompt tokens as the output
            all_outputs.append(encoded_prompts[i])  # Update list of TT outputs
            if run_ref_pt:
                all_outputs_ref.append(encoded_prompts[i])  # Update list of ref outputs

            tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Greedy decode (temperature = 0) the generated token and save it to print out later
            if run_ref_pt:
                # Sample from reference model first
                _, pt_out_tok = sample_host(ref_output, temperature=0, top_p=0.8)
                pt_decode_input = embd(pt_out_tok)
                all_outputs_ref.append(pt_out_tok.squeeze(1).tolist()[0])

                # Use the same token for TT model (teacher forcing)
                tt_decode_input = pt_decode_input
                all_outputs.append(pt_out_tok.squeeze(1).tolist()[0])
            else:
                # If not running reference model, sample from TT model directly
                _, tt_out_tok = sample_host(tt_output_torch, temperature=0, top_p=0.8)
                tt_decode_input = embd(tt_out_tok)
                all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])

        # Measure PCC if also running reference model
        if run_ref_pt:
            passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

            logger.info(comp_allclose(ref_output, tt_output_torch))
            logger.info(f"PCC: {pcc_message}")

            if passing:
                logger.info("Model Passed!")
            else:
                logger.warning("Model Failed!")
            if not passing:
                all_tests_pass = False

        if parse_chat:
            conversation = parse_chat_output(tokenizer.decode(all_outputs).replace("\n", "\\n"))
            display_chat(logger, conversation)

    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All {generation_length} decode iterations Passed!")
        else:
            logger.warning("One or more iterations of decode had bad PCC")
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
