"""End-to-end test for Gemma-3-4B-it vision-text pipeline."""
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.tt_transformers.tt.common import (
    sample_host,
    PagedAttentionConfig,
    preprocess_inputs_prefill,
)
from models.tt_transformers.tt.model_config import DecodersPrecision

from models.experimental.gemma3_4b.tt.gemma_model import TtGemma3Model
from models.tt_transformers.tt.generator import Generator
from models.common.utility_functions import skip_for_grayskull, skip_for_blackhole

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


# =============================================================================
# NEW E2E PIPELINE COMPONENTS - Following SOLID Principles
# =============================================================================


def setup_model_args(weights, max_seq_len, batch_size, mesh_device, optimizations):
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


def create_tt_model(model_args, mesh_device, dtype, paged_attention, page_params):
    """Load separate vision and text models following test_end2end.py pattern."""
    state_dict = model_args.load_state_dict()

    # Setup paged attention config (exactly like test_end2end.py)
    paged_attention_config = None
    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )

    model = TtGemma3Model(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    logger.info("Loaded the entire Gemma3 model")
    return model


def generate(model, processed_inputs, model_args, page_table=None, paged_attention_config=None, max_gen_len=20):
    """Run generation following the EXACT pattern from test_end2end.py."""
    input_ids = processed_inputs["input_ids"]
    pixel_values = processed_inputs["pixel_values"]
    input_prompts = processed_inputs["input_prompts"]

    logger.info("Running generation...")

    # Create Generator (exactly like test_end2end.py)
    generator = Generator(
        [model], [model_args], model.mesh_device, processor=model_args.processor, tokenizer=model_args.tokenizer
    )

    # Setup KV cache (exactly like test_end2end.py)
    tt_kv_cache = [[l.attention.layer_past for l in model.layers]] if paged_attention_config else None

    # # Text generation setup (exactly like test_end2end.py)
    input_tokens_prefill = input_ids
    batch_size = input_tokens_prefill.shape[0]

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
        pixel_values=pixel_values,
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
    model_args, instruct = setup_model_args(weights, max_seq_len, batch_size, mesh_device, optimizations)

    if layers is not None:
        model_args.n_layers = layers

    # Setup vision prompts and tokenizer
    messages, tokenizer = setup_vision_prompts_and_tokenizer(model_args, instruct)

    # Process real vision inputs from images
    processed_inputs = process_real_vision_inputs(messages, model_args)

    # Load separate models following test_end2end.py pattern
    logger.info("Loading separate vision and text models like test_end2end.py...")
    model = create_tt_model(model_args, mesh_device, dtype, paged_attention, page_params)

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

    # Run generation
    results = generate(model, processed_inputs, model_args, page_table, paged_attention_config, max_gen_len=10)

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
