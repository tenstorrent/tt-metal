# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Full model integration test with accuracy testing
"""

import os
import pickle

import pytest
import torch
from loguru import logger

from ..test_factory import TestFactory, parametrize_mesh_with_fabric


class TokenAccuracy:
    """Token accuracy testing similar to tt_transformers"""

    def __init__(self, reference_tokens, top5_tokens):
        self.gt_pos = -1
        self.store_predicted_tokens = []
        self.reference_tokens = reference_tokens
        self.top5_tokens = top5_tokens
        self.maxindex = len(self.reference_tokens) - 1

    def collect_predicted_tokens(self, tokens):
        """Collect predicted tokens during generation"""
        self.store_predicted_tokens.append(tokens)
        self.gt_pos += 1
        return self.reference_tokens[min(self.gt_pos, self.maxindex)].unsqueeze(-1).unsqueeze(-1)

    def compute_accuracy(self):
        """Compute top-1 and top-5 accuracy"""
        count = 0
        count_t5 = 0
        matching_sz = min(len(self.reference_tokens), len(self.store_predicted_tokens))
        for i in range(matching_sz):
            # Compare against actual reference tokens, not top5_tokens[0]
            if self.reference_tokens[i].item() == self.store_predicted_tokens[i]:
                count += 1
            if self.store_predicted_tokens[i] in self.top5_tokens[i, :]:
                count_t5 += 1
        accuracy_top1 = count / matching_sz
        accuracy_top5 = count_t5 / matching_sz
        return accuracy_top1, accuracy_top5


def run_accuracy(
    generator, model_args, input_ids, reference_tokens, top5_tokens, tokenizer, mesh_device, tt_kv_cache, prefill_len
):
    """
    Test accuracy using teacher forcing method similar to tt_transformers
    """
    logger.info("Starting accuracy test with teacher forcing")

    # Initialize accuracy tracker
    accuracy_tracker = TokenAccuracy(reference_tokens, top5_tokens)

    # Prepare input prompt (use the preprocessed format)
    # Use the full prefill length, not just the reference tokens length
    input_prompt = input_ids[0, :prefill_len]
    prompt_text = tokenizer.decode(input_prompt.tolist())
    logger.info(f"Input prompt: {prompt_text[:100]}...")

    # Convert to the format expected by generator (use the preprocessed format)
    input_tokens_prefill_pt = [input_prompt.tolist()]

    # Generate tokens with teacher forcing using generator
    generated_tokens = []

    # First, do a single prefill like the demo
    logger.info(f"Running initial prefill with {prefill_len} tokens...")
    with torch.no_grad():
        # Use the same format as the demo
        # input_tokens_prefill_pt is already a list of lists, just convert to tensor
        input_tokens_prefill_pt_tensor = torch.tensor(input_tokens_prefill_pt)
        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt_tensor,
            page_table=None,  # No paged attention for simple test
            kv_cache=tt_kv_cache,  # Use the actual KV cache
            prompt_lens=[prefill_len],  # Use the correct prefill length
        )

    # Get the first predicted token
    predicted_token = torch.argmax(logits, dim=-1)[0].item()
    generated_tokens.append(predicted_token)

    # Print first token from TT model
    tt_token_text = tokenizer.decode([predicted_token])
    ref_token_text = tokenizer.decode([reference_tokens[0].item()])
    logger.info(f"Token 0 - TT model: '{tt_token_text}' (id: {predicted_token})")
    logger.info(f"Token 0 - Ref model: '{ref_token_text}' (id: {reference_tokens[0].item()})")

    # For subsequent tokens, use decode (not prefill) like the demo
    # Initialize like the demo
    all_outputs = [input_tokens_prefill_pt[0][:prefill_len]]  # Start with prefill tokens
    all_outputs[0].append(predicted_token)  # Add first generated token

    # Initialize decode state like the demo
    current_pos = torch.tensor([prefill_len])  # Start position after prefill
    out_tok = torch.tensor([[predicted_token]])  # Current token to decode

    for i in range(1, len(reference_tokens)):
        # Use decode for subsequent tokens with teacher forcing (use reference tokens)
        with torch.no_grad():
            logits, _ = generator.decode_forward(
                out_tok,  # out_tok (current token)
                current_pos,  # current_pos
                enable_trace=False,  # enable_trace
                page_table=None,  # page_table
                kv_cache=tt_kv_cache,  # kv_cache
            )

        # Get predicted token
        predicted_token = torch.argmax(logits, dim=-1)[0].item()
        generated_tokens.append(predicted_token)

        # Update for next iteration using reference token (teacher forcing)
        reference_token = reference_tokens[i].item()
        out_tok = torch.tensor([[reference_token]])  # Use reference token for next decode
        current_pos += 1

        # Log token comparison for debugging
        if i < 50:  # Only log first 5 tokens to avoid spam
            tt_token_text = tokenizer.decode([predicted_token])
            ref_token_text = tokenizer.decode([reference_tokens[i].item()])
            logger.info(f"Token {i} - TT: '{tt_token_text}' (id: {predicted_token})")
            logger.info(f"Token {i} - Ref: '{ref_token_text}' (id: {reference_tokens[i].item()})")

    # Compute accuracy
    accuracy_tracker.store_predicted_tokens = generated_tokens
    top1_acc, top5_acc = accuracy_tracker.compute_accuracy()

    logger.info(f"Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    logger.info(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")

    return top1_acc, top5_acc


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize(
    "mesh_shape",
    [
        (1, 8),
        (4, 8),
    ],
    ids=[
        "mesh_1x8",
        "mesh_4x8",
    ],
)
def test_full_model_accuracy(mesh_device, mesh_shape, device_params, reset_seeds, state_dict):
    """Test full model with accuracy testing using new abstractions"""

    # Cache file for reference tokens
    cache_dir = "models/demos/gpt_oss/tests/accuracy/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "reference_tokens.pkl")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    mesh_config = setup["mesh_config"]

    # Try to load cached reference tokens first
    reference_tokens = None
    top5_tokens = None

    if os.path.exists(cache_file):
        logger.info("Loading cached reference tokens...")
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                reference_tokens = cached_data["reference_tokens"]
                top5_tokens = cached_data["top5_tokens"]
                logger.info(f"Loaded {len(reference_tokens)} cached reference tokens")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            reference_tokens = None
            top5_tokens = None

    # Generate reference tokens if not cached
    if reference_tokens is None:
        logger.info("Generating reference tokens (this may take a while)...")

        # Create reference model for comparison (use full model like demo)
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

        # Load the same weights that the TTNN model uses
        logger.info("Loading reference model weights...")
        reference_weights = setup["model_args"].load_state_dict(
            weights_path=setup["model_args"].model_path,
            dummy_weights=setup["model_args"].dummy_weights,
            convert_to_meta_format=True,
        )

        # Create reference model and load the real weights
        reference_model = GptOssForCausalLM(config)
        reference_model.load_state_dict(reference_weights, strict=False)
        reference_model.eval()

        # Use the same tokenizer setup as simple_text_demo
        tokenizer = setup["model_args"].tokenizer

        # Use the same setup as simple_text_demo
        from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args

        # Use the same parameters as the demo
        num_devices = setup["mesh_device"].get_num_devices()
        data_parallel = 1
        global_batch_size = 1
        max_seq_len = 1024
        paged_attention = False
        page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": 16}

        (
            model_args,
            model,
            page_table,
            tt_kv_cache,
            tokenizer,
            processor,
            paged_attention_config,
        ) = prepare_gpt_oss_generator_args(
            num_devices=num_devices,
            data_parallel=data_parallel,
            mesh_device=setup["mesh_device"],
            global_batch_size=global_batch_size,
            optimizations=None,
            max_seq_len=max_seq_len,
            page_params=page_params,
            paged_attention=paged_attention,
            mesh_config=mesh_config,
            state_dict=state_dict,
            users_row_sharded=False,
        )

        # Create test input using the same prompt as the demo
        input_prompts = ["What are the prime factors of 1?"]

        # Use the same preprocessing as the demo
        from models.tt_transformers.tt.common import preprocess_inputs_prefill

        max_generated_tokens = 30  # Generate 30 tokens for testing

        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts,
            tokenizer,
            model_args,
            instruct=False,
            max_generated_tokens=max_generated_tokens,
            max_prefill_len=1024,
        )

        # Use the correct prefill length from the preprocessing
        actual_prefill_len = prefill_lens[0]

        # Use the preprocessed input
        input_ids = torch.stack(input_tokens_prefill_pt).view(1, -1)

        # Generate reference tokens using the reference model (autonomous generation like demo)
        with torch.no_grad():
            # Use the same preprocessed input as TTNN model
            reference_tokens = []
            top5_tokens = []

            # Start with the preprocessed input (90 tokens)
            current_input = input_ids.clone()

            # Generate tokens step by step (autonomous generation)
            for i in range(max_generated_tokens):
                # Get prediction from reference model
                reference_output = reference_model(current_input)
                reference_logits = reference_output.logits

                # Get the next token from the last position
                next_token = torch.argmax(reference_logits[0, -1, :]).item()
                reference_tokens.append(next_token)

                # Get top-5 for this token
                top5_next = torch.topk(reference_logits[0, -1, :], k=5).indices
                top5_tokens.append(top5_next)

                # Add the predicted token to input for next iteration
                current_input = torch.cat([current_input, torch.tensor([[next_token]])], dim=1)
            # Convert to tensors
            reference_tokens = torch.tensor(reference_tokens)
            top5_tokens = torch.stack(top5_tokens)

        # Save reference tokens to cache
        logger.info("Saving reference tokens to cache...")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"reference_tokens": reference_tokens, "top5_tokens": top5_tokens}, f)
            logger.info(f"Saved {len(reference_tokens)} reference tokens to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    # If we have cached tokens, we still need to set up the TTNN model
    if reference_tokens is not None:
        # Use the same tokenizer setup as simple_text_demo
        tokenizer = setup["model_args"].tokenizer

        # Use the same setup as simple_text_demo
        from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args

        # Use the same parameters as the demo
        num_devices = setup["mesh_device"].get_num_devices()
        data_parallel = 1
        global_batch_size = 1
        max_seq_len = 1024  # TODO: extend this to 128k
        paged_attention = False
        page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": 16}

        (
            model_args,
            model,
            page_table,
            tt_kv_cache,
            tokenizer,
            processor,
            paged_attention_config,
        ) = prepare_gpt_oss_generator_args(
            num_devices=num_devices,
            data_parallel=data_parallel,
            mesh_device=setup["mesh_device"],
            global_batch_size=global_batch_size,
            optimizations=None,
            max_seq_len=max_seq_len,
            page_params=page_params,
            paged_attention=paged_attention,
            mesh_config=mesh_config,
            state_dict=state_dict,
            users_row_sharded=False,
        )

        # Create test input using the same prompt as the demo
        input_prompts = ["What are the prime factors of 1?"]

        # Use the same preprocessing as the demo
        from models.tt_transformers.tt.common import preprocess_inputs_prefill

        max_generated_tokens = 30  # Generate 30 tokens for testing

        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts,
            tokenizer,
            model_args,
            instruct=False,
            max_generated_tokens=max_generated_tokens,
            max_prefill_len=1024,
        )

        # Use the correct prefill length from the preprocessing
        actual_prefill_len = prefill_lens[0]

        # Use the preprocessed input
        input_ids = torch.stack(input_tokens_prefill_pt).view(1, -1)

    # Create generator like the demo
    from models.tt_transformers.tt.generator import Generator

    generator = Generator(model, model_args, setup["mesh_device"], processor=processor, tokenizer=tokenizer)

    # Test accuracy
    top1_acc, top5_acc = run_accuracy(
        generator,
        model_args,
        input_ids,
        reference_tokens,
        top5_tokens,
        tokenizer,
        setup["mesh_device"],
        tt_kv_cache,
        actual_prefill_len,
    )

    # Assert minimum accuracy thresholds (realistic for teacher forcing)
    min_top1_acc = 0.83  # 83% minimum top-1 accuracy with teacher forcing
    min_top5_acc = 0.96  # 96% minimum top-5 accuracy with teacher forcing

    assert top1_acc >= min_top1_acc, f"Top-1 accuracy {top1_acc:.4f} below threshold {min_top1_acc}"
    assert top5_acc >= min_top5_acc, f"Top-5 accuracy {top5_acc:.4f} below threshold {min_top5_acc}"

    logger.info(f"✅ Full model accuracy test passed!")
    logger.info(f"Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    logger.info(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
