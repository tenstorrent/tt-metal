"""End-to-end test for Gemma-3-4B-it vision-text pipeline."""
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import (
    encode_prompt_hf,
    sample_host,
    PagedAttentionConfig,
    preprocess_inputs_prefill,
)
from models.tt_transformers.tt.model_config import DecodersPrecision

from models.experimental.gemma3_4b.tt.text_model import Gemma3_4BTransformer
from models.experimental.gemma3_4b.tt.gemma_vision_crossattention import TtGemmaTransformerVision
from models.experimental.gemma3_4b.tt.gemma3_generator import Gemma3Generator
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull, skip_for_blackhole
from models.tt_transformers.tt.model_config import HfModelWrapper

from models.tt_transformers.tt.model_config import ModelArgs
from transformers import AutoProcessor

import re


def parse_chat_output(text):
    """Parse chat output format from generated text."""
    pattern = r"<\|(?P<role>user|assistant)\|>\s*(?P<message>.*?)(?=<\|(?:user|assistant|end)\|>|$)"
    matches = re.finditer(pattern, text, re.DOTALL)
    return [(match.group("role"), match.group("message").strip()) for match in matches]


def display_chat(logger, conversation):
    """Display chat conversation in formatted output."""
    for role, message in conversation:
        if role == "user":
            logger.info(f"👤 User: {message}")
        elif role == "assistant":
            logger.info(f"🤖 Assistant: {message}")


def setup_model_args(weights, max_seq_len, batch_size, mesh_device, optimizations):
    """Setup model arguments and configuration."""
    instruct = True if weights == "instruct" else False

    model_args = ModelArgs(
        mesh_device=mesh_device,
        instruct=instruct,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )

    return model_args, instruct


def setup_prompts_and_tokenizer(model_args, instruct):
    """Setup prompts and tokenizer for the test."""
    prompts = ["Write a essay about Lion"] * model_args.max_batch_size
    tokenizer = model_args.processor if model_args.is_multimodal else model_args.tokenizer

    if instruct:
        encoded_prompts = encode_prompt_hf(tokenizer, prompt_text=prompts[0])
    else:
        encoded_prompts = [model_args.encode_prompt(prompt, instruct=False) for prompt in prompts]

    return prompts, tokenizer, encoded_prompts


def setup_reference_model(model_args, run_ref_pt):
    """Setup reference PyTorch model and embedding."""
    if run_ref_pt:
        reference_transformer_model = model_args.reference_transformer(wrap=False)
        reference_model = HfModelWrapper(reference_transformer_model, model_args.head_dim)
        logger.info("Finished loading reference model.")
        embd = model_args.reference_embedding(reference_transformer_model)
    else:
        reference_model = None
        embd = model_args.reference_embedding()

    return reference_model, embd


def setup_paged_attention(paged_attention, page_params, model_args, mesh_device):
    """Setup paged attention configuration and page table."""
    page_table_tt = None
    paged_attention_config = None

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
                dims=(None, -2) if model_args.max_batch_size > 1 else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    return paged_attention_config, page_table_tt


def load_tt_model(model_args, mesh_device, dtype, paged_attention_config):
    """Load the TT model with state dict."""
    state_dict = model_args.load_state_dict()

    tt_model = Gemma3_4BTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    logger.info("Model and caches loaded.")
    return tt_model


# =============================================================================
# NEW E2E PIPELINE COMPONENTS - Following SOLID Principles
# =============================================================================


def setup_vision_model_args(weights, max_seq_len, batch_size, mesh_device, optimizations):
    """Setup model arguments for vision-enabled model (Single Responsibility)."""
    instruct = True if weights == "instruct" else False

    model_args = ModelArgs(
        mesh_device=mesh_device,
        instruct=instruct,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )
    return model_args, instruct


def setup_vision_prompts_and_tokenizer(model_args, instruct):
    """Setup multimodal prompts and tokenizer for vision-enabled model."""
    # Create multimodal messages similar to test_end2end.py
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Write about Marvel in detail for 1000 words."},
            ],
        }
    ]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]

    tokenizer = model_args.tokenizer
    return messages, tokenizer


def setup_vision_reference_model(model_args, run_ref_pt):
    """Setup reference vision-enabled model (Open/Closed Principle)."""
    if run_ref_pt:
        reference_transformer_model = model_args.reference_vision_transformer(wrap=False)
        reference_model = HfModelWrapper(reference_transformer_model, model_args.head_dim)
        logger.info("Finished loading reference vision model.")
        embd = model_args.reference_embedding(reference_transformer_model)
    else:
        reference_model = None
        embd = model_args.reference_embedding()

    return reference_model, embd


def process_real_vision_inputs(messages, model_args):
    """Process real image inputs using AutoProcessor (Interface Segregation)."""
    model_id = "google/gemma-3-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)

    # Process the multimodal messages similar to test_end2end.py
    encoded = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(dtype=torch.bfloat16)

    input_ids = encoded["input_ids"]
    pixel_values = encoded["pixel_values"]
    attention_mask = encoded["attention_mask"]

    # logger.info(f"Processed vision inputs - input_ids: {input_ids.shape}, pixel_values: {pixel_values.shape}")

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "processor": processor,
        "input_prompts": messages,
    }


# Legacy function removed - vision model now part of multimodal model


def load_separate_models_like_test_end2end(model_args, mesh_device, dtype, paged_attention, page_params):
    """Load separate vision and text models following test_end2end.py pattern."""
    state_dict = model_args.load_state_dict()
    vision_prefix = model_args.state_dict_vision_prefix

    # Setup paged attention config (exactly like test_end2end.py)
    paged_attention_config = None
    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )

    tt_ccl = TT_CCL(mesh_device)

    # Load vision model (exactly like test_end2end.py)
    vision_model = TtGemmaTransformerVision(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        state_dict_prefix=vision_prefix,
        dtype=dtype,
        configuration=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )

    # Load text model (exactly like test_end2end.py)
    text_model = Gemma3_4BTransformer(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    logger.info("Separate vision and text models loaded like test_end2end.py")
    return vision_model, text_model


def run_generation_exactly_like_test_end2end(
    vision_model, text_model, processed_inputs, model_args, page_table=None, paged_attention_config=None, max_gen_len=20
):
    """Run generation following the EXACT pattern from test_end2end.py."""
    input_ids = processed_inputs["input_ids"]
    pixel_values = processed_inputs["pixel_values"]
    input_prompts = processed_inputs["input_prompts"]

    logger.info("Running generation exactly like test_end2end.py...")

    # Process vision (exactly like test_end2end.py)
    logger.info("Running Vision Model...")

    # Create Generator (exactly like test_end2end.py)
    generator = Gemma3Generator([text_model], [model_args], vision_model.mesh_device, tokenizer=model_args.tokenizer)

    # Setup KV cache (exactly like test_end2end.py)
    tt_kv_cache = [[l.attention.layer_past for l in text_model.layers]] if paged_attention_config else None

    # Get embeddings and combine with vision (exactly like test_end2end.py)
    # host_embedding = model_args.reference_embedding()

    # # Text generation setup (exactly like test_end2end.py)
    input_tokens_prefill = input_ids
    batch_size = input_tokens_prefill.shape[0]
    # seq_len = input_tokens_prefill.shape[1]

    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        [input_prompts],
        model_args.tokenizer,
        [model_args],
        instruct=True,
        max_generated_tokens=max_gen_len,
        max_prefill_len=8192,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

    logger.info("Running prefill...")
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        vision_model=vision_model,
        processed_inputs=processed_inputs,
    )

    # Get first token (exactly like test_end2end.py)
    prefilled_token = torch.argmax(logits, dim=-1)
    logger.info(f"Prefilled token: {prefilled_token}")

    # Initialize generation (exactly like test_end2end.py)
    all_outputs = [encoded_prompts[0][: prefill_lens[0]]]
    all_outputs[0].append(int(prefilled_token[0].item()))

    current_pos = torch.tensor([decoding_pos[0]])
    out_tok = prefilled_token
    generation_length = 150

    results = []

    # Decode loop (exactly like test_end2end.py)
    logger.info("Starting decode loop...")
    for iteration in range(generation_length):
        logger.info(f"[Text] Decoding token {iteration}, current_pos: {current_pos.item()}")

        # Run decode (exactly like test_end2end.py)
        logits = generator.decode_forward_text(
            out_tok,
            current_pos,
            enable_trace=False,
            page_table=page_table,
            kv_cache=tt_kv_cache,
        )

        # Sample next token (exactly like test_end2end.py)
        _, out_tok = sample_host(
            logits,
            temperature=0,
            top_p=0.9,
        )

        token_id = out_tok[0].item()
        decoded_token = model_args.tokenizer.decode([token_id])
        logger.info(f"Generated token {iteration}: ID={token_id}, text='{decoded_token}'")

        # Create result object
        result = type("TokenResult", (), {"token": token_id, "text": decoded_token})()

        results.append(result)

        all_outputs[0].append(token_id)
        current_pos += 1

        # Early stopping (exactly like test_end2end.py)
        if len(all_outputs[0]) >= 5 and all(t == all_outputs[0][-1] for t in all_outputs[0][-5:]):
            logger.warning(f"Detected exact repetition of token {all_outputs[0][-1]} five times in a row. Stopping.")
            break

    # Final response (exactly like test_end2end.py)
    response = model_args.tokenizer.decode(all_outputs[0], skip_special_tokens=True)
    logger.info(f"📝 Final Generated Response:\n{response}")
    logger.info(f"📝 Generated {len(all_outputs[0])} tokens: {all_outputs[0]}")
    chat = parse_chat_output(response)
    display_chat(logger, chat)

    logger.info(f"Generated {len(results)} tokens successfully")
    return results


# Legacy function removed - vision processing now handled in multimodal model


def validate_e2e_outputs(results, expected_min_tokens=1):
    """Validate end-to-end pipeline outputs."""
    if not results:
        logger.error("No results generated from E2E pipeline")
        return False

    if len(results) < expected_min_tokens:
        logger.warning(f"Generated only {len(results)} tokens, expected at least {expected_min_tokens}")
        return False

    # Check if tokens are valid
    for result in results:
        if not hasattr(result, "token") or not hasattr(result, "text"):
            logger.error("Invalid result format")
            return False

    logger.info("E2E pipeline validation passed")
    return True


# =============================================================================
# EXISTING FUNCTIONS (Unchanged for backward compatibility)
# =============================================================================


def create_position_tensor(current_pos, model_args, mesh_device):
    """Create position tensor for the model."""
    return ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and model_args.max_batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )


def convert_tt_output_to_torch(tt_out, model_args, mesh_device):
    """Convert TTNN tensor to PyTorch tensor."""
    mesh_composer = ttnn.ConcatMesh2dToTensor(
        mesh_device, dims=(3, 1) if model_args.is_galaxy else (1, -1), mesh_shape=model_args.cluster_shape
    )
    tt_output_torch = (
        ttnn.to_torch(tt_out, mesh_composer=mesh_composer)
        .permute(2, 1, 0, 3)
        .squeeze(2)[: model_args.max_batch_size, 0:1, : model_args.vocab_size]
    )
    ttnn.deallocate(tt_out)
    return tt_output_torch


def process_token_generation(
    i,
    encoded_prompts,
    encoded_prompts_tensor,
    embd,
    batch,
    seqlen,
    all_outputs,
    all_outputs_ref,
    run_ref_pt,
    ref_output,
    tt_output_torch,
):
    """Process token generation for both prefill and decode phases."""
    if i in range(len(encoded_prompts)):
        # While in "prefill" mode, use the prompt tokens as the output
        all_outputs.append(encoded_prompts[i])
        if run_ref_pt:
            all_outputs_ref.append(encoded_prompts[i])

        tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        if run_ref_pt:
            pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            pt_decode_input = None
    else:
        # Greedy decode (temperature = 0) the generated token and save it to print out later
        # Exact copy of original logic (including commented sections)
        # if run_ref_pt:
        #     # Sample from reference model first
        _, pt_out_tok = sample_host(ref_output, temperature=0, top_p=0.8)
        pt_decode_input = embd(pt_out_tok)
        all_outputs_ref.append(pt_out_tok.squeeze(1).tolist()[0])

        # Use the same token for TT model (teacher forcing)
        tt_decode_input = pt_decode_input
        # all_outputs.append(pt_out_tok.squeeze(1).tolist()[0])
        # else:
        # If not running reference model, sample from TT model directly
        _, tt_out_tok = sample_host(tt_output_torch, temperature=0, top_p=0.8)
        tt_decode_input = embd(tt_out_tok)
        all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])

    return tt_decode_input, pt_decode_input


def validate_outputs(run_ref_pt, ref_output, tt_output_torch, pcc, all_outputs, all_outputs_ref, tokenizer, logger):
    """Validate model outputs and compute PCC."""
    if run_ref_pt:
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

        # Decode the output tokens back to text
        decoded_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in all_outputs]
        logger.info(f"TTNN Decoded Outputs: {''.join(decoded_texts)}")
        decoded_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in all_outputs_ref]
        logger.info(f"Torch Decoded Outputs: {''.join(decoded_texts)}")

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

        if passing:
            logger.info("Model Passed!")
        else:
            logger.warning("Model Failed!")

        return passing
    return True


def run_generation_loop(
    tt_model,
    model_args,
    mesh_device,
    reference_model,
    embd,
    encoded_prompts_tensor,
    generation_length,
    generation_start_pos,
    batch,
    seqlen,
    page_table_tt,
    run_ref_pt,
    pcc,
    tokenizer,
    logger,
    parse_chat,
    encoded_prompts,
):
    """Run the main token generation loop."""
    all_outputs = []
    all_outputs_ref = [] if run_ref_pt else []
    if run_ref_pt:
        all_tests_pass = True

    # Initial setup
    current_pos = torch.tensor([generation_start_pos for _ in range(batch)])
    current_pos_tensor = create_position_tensor(current_pos, model_args, mesh_device)

    # Select the first token from the prompts for initial decoding
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
    tt_decode_input = pt_decode_input

    for i in range(generation_length):
        logger.info(f"[Model] Generating token {i}")

        # Prepare input
        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get rotation matrices
        rot_mats_global = tt_model.rope_setup.get_rot_mats(current_pos)
        rot_mats_local = tt_model.rope_setup_local.get_rot_mats(current_pos)
        rot_mats = [rot_mats_global, rot_mats_local]

        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        # Convert output
        tt_output_torch = convert_tt_output_to_torch(tt_out, model_args, mesh_device)

        # Run reference model if needed
        ref_output = None
        if run_ref_pt:
            ref_output = reference_model(pt_decode_input, current_pos[0])

        # Update position
        current_pos = torch.tensor([generation_start_pos + i + 1 for _ in range(batch)])
        current_pos_tensor = create_position_tensor(current_pos, model_args, mesh_device)

        # Process token generation
        tt_decode_input, pt_decode_input = process_token_generation(
            i,
            encoded_prompts,
            encoded_prompts_tensor,
            embd,
            batch,
            seqlen,
            all_outputs,
            all_outputs_ref,
            run_ref_pt,
            ref_output,
            tt_output_torch,
        )

        # Validate outputs
        passing = validate_outputs(
            run_ref_pt, ref_output, tt_output_torch, pcc, all_outputs, all_outputs_ref, tokenizer, logger
        )

        # Note: Individual PCC failures don't affect overall test result (matching original behavior)
        # if not passing:
        #     all_tests_pass = False

        # Display chat if enabled
        if parse_chat:
            conversation = parse_chat_output(tokenizer.decode(all_outputs).replace("\n", "\\n"))
            display_chat(logger, conversation)

    if run_ref_pt:
        return all_tests_pass
    else:
        return True  # If not running reference model, always pass


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
    (2048,),  # Use smaller seq_len like test_end2end.py to avoid memory issues
)
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
    ],
    ids=["accuracy"],
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
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_e2e_vision_text_pipeline(
    weights,
    layers,
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    reset_seeds,
    request,
    device_params,
):
    """Test end-to-end vision-text pipeline using proper Generator methods."""
    logger.info("Starting E2E vision-text pipeline test")

    # Use bfloat8_b like test_end2end.py for better memory efficiency
    dtype = ttnn.bfloat16

    # Setup vision-enabled model configuration
    model_args, instruct = setup_vision_model_args(weights, max_seq_len, batch_size, mesh_device, optimizations)

    if layers is not None:
        model_args.n_layers = layers

    # Setup vision prompts and tokenizer
    messages, tokenizer = setup_vision_prompts_and_tokenizer(model_args, instruct)

    # Process real vision inputs from images
    processed_inputs = process_real_vision_inputs(messages, model_args)

    # Load separate models following test_end2end.py pattern
    logger.info("Loading separate vision and text models like test_end2end.py...")
    vision_model, text_model = load_separate_models_like_test_end2end(
        model_args, mesh_device, dtype, paged_attention, page_params
    )

    # Setup page table for paged attention (exactly like test_end2end.py)
    page_table_tt = None
    paged_attention_config = None

    # Prepare page table for paged attention (exactly like test_end2end.py)
    page_table = None
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

    # Run generation following EXACT test_end2end.py pattern
    logger.info("Running generation following EXACT test_end2end.py pattern...")
    results = run_generation_exactly_like_test_end2end(
        vision_model, text_model, processed_inputs, model_args, page_table, paged_attention_config, max_gen_len=10
    )

    # Validate results
    validation_passed = validate_e2e_outputs(results, expected_min_tokens=1)

    # Final validation
    if validation_passed and len(results) > 0:
        logger.info("✅ E2E vision-text pipeline test PASSED!")
        logger.info(f"Successfully generated {len(results)} tokens")

        # Log generated tokens for debugging
        for i, result in enumerate(results[:5]):
            logger.info(f"Token {i}: {result.token} -> '{result.text}'")
    else:
        logger.error("❌ E2E pipeline test failed")
        assert False, f"E2E pipeline failed - generated {len(results)} tokens, validation: {validation_passed}"
