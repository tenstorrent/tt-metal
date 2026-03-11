# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test to compare TTNN vs Reference generation token by token.
This helps identify where KV cache corruption or other issues occur.
"""

import torch
from loguru import logger

import ttnn


def comp_pcc(a, b, pcc_threshold=0.99):
    """Calculate Pearson Correlation Coefficient between two tensors."""
    a = a.flatten().float()
    b = b.flatten().float()

    if a.shape != b.shape:
        return 0.0, False

    a_mean = a.mean()
    b_mean = b.mean()
    a_centered = a - a_mean
    b_centered = b - b_mean

    numerator = (a_centered * b_centered).sum()
    denominator = torch.sqrt((a_centered**2).sum() * (b_centered**2).sum())

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0, True

    pcc = (numerator / denominator).item()
    return pcc, pcc >= pcc_threshold


def _run_generation_comparison_impl(num_tokens: int = 20):
    """Compare TTNN vs reference generation token by token. Returns (hf_tokens, ttnn_tokens, prefill_pcc)."""

    from transformers import AutoProcessor

    from models.demos.molmo2.demo.demo import (
        DEFAULT_IMAGE,
        Molmo2Generator,
        create_model,
        get_image_tokens,
        load_model_weights,
        load_processor,
        preprocess_image_molmo2,
    )

    logger.info("=" * 60)
    logger.info("TTNN vs Reference Generation Comparison")
    logger.info("=" * 60)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_processor()

    # Load HuggingFace reference model using the custom model class
    logger.info("Loading HuggingFace reference model...")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)

    # Import the custom model class
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_class = get_class_from_dynamic_module(
        "modeling_molmo2.Molmo2ForConditionalGeneration",
        "allenai/Molmo2-8B",
        trust_remote_code=True,
    )
    hf_model = model_class.from_pretrained(
        "allenai/Molmo2-8B",
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.eval()
    hf_processor = AutoProcessor.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)

    # Load TTNN model
    logger.info("Loading TTNN model weights...")
    state_dict = load_model_weights()

    logger.info("Opening TTNN device...")
    device = ttnn.open_device(device_id=0)

    logger.info("Creating TTNN model...")
    model = create_model(device, state_dict, num_layers=36)

    # Create TTNN generator
    generator = Molmo2Generator(
        mesh_device=device,
        model=model,
        tokenizer=tokenizer,
        num_layers=36,
        batch_size=1,
        max_seq_len=2048,
    )

    # Preprocess image
    logger.info("Preprocessing image...")
    from PIL import Image

    image = Image.open(DEFAULT_IMAGE).convert("RGB")

    # Process for HuggingFace
    hf_inputs = hf_processor.process(
        images=[image],
        text="Describe this image.",
    )
    hf_input_ids = hf_inputs["input_ids"]
    hf_images = hf_inputs.get("images")
    hf_image_input_idx = hf_inputs.get("image_input_idx")
    hf_image_masks = hf_inputs.get("image_masks")

    logger.info(f"HF input shape: {hf_input_ids.shape}")

    # Process for TTNN
    image_inputs = preprocess_image_molmo2(str(DEFAULT_IMAGE))
    image_grid = image_inputs["image_grids"][0]
    image_tokens_str = get_image_tokens(image_grid)
    prompt = "Describe this image."
    full_prompt = f"{image_tokens_str} {prompt}"
    ttnn_input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

    logger.info(f"TTNN input shape: {ttnn_input_ids.shape}")

    # Initialize KV cache for TTNN
    generator.init_kv_cache()

    # ========== PREFILL COMPARISON ==========
    logger.info("\n" + "=" * 60)
    logger.info("PREFILL COMPARISON")
    logger.info("=" * 60)

    # HuggingFace prefill
    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=hf_input_ids,
            images=hf_images,
            image_input_idx=hf_image_input_idx,
            image_masks=hf_image_masks,
            use_cache=True,
        )
        hf_logits = hf_outputs.logits
        hf_past_kv = hf_outputs.past_key_values

    logger.info(f"HF prefill logits shape: {hf_logits.shape}")

    # TTNN prefill
    pixel_values = image_inputs["pixel_values"]
    pooled_patches_idx = image_inputs["image_token_pooling"]

    hidden_states_ttnn, _ = generator._prepare_text_inputs(ttnn_input_ids, pixel_values, pooled_patches_idx)

    ttnn_logits, _ = generator.model.text_model.forward(
        hidden_states=hidden_states_ttnn,
        start_pos=0,
        attn_mask=None,
        kv_caches=generator.kv_caches,
    )
    ttnn.synchronize_device(device)

    ttnn_logits_torch = ttnn.to_torch(ttnn_logits)
    logger.info(f"TTNN prefill logits shape: {ttnn_logits_torch.shape}")

    # Compare last token logits (used for first generated token)
    hf_last_logits = hf_logits[0, -1, :].unsqueeze(0).unsqueeze(0)
    ttnn_last_logits = ttnn_logits_torch[0, 0, -1, :].unsqueeze(0).unsqueeze(0)

    pcc, _ = comp_pcc(hf_last_logits, ttnn_last_logits)
    prefill_pcc = pcc
    logger.info(f"Prefill last token logits PCC: {pcc:.6f}")

    # Get first predicted tokens
    hf_next_token = hf_logits[0, -1, :].argmax().item()
    ttnn_next_token = ttnn_logits_torch[0, 0, -1, :].argmax().item()

    logger.info(f"HF first predicted token: {hf_next_token} = '{tokenizer.decode([hf_next_token])}'")
    logger.info(f"TTNN first predicted token: {ttnn_next_token} = '{tokenizer.decode([ttnn_next_token])}'")

    # Set up for decode
    seq_len = ttnn_input_ids.shape[1]
    generator.reset_kv_cache(seq_len)

    # ========== DECODE COMPARISON ==========
    logger.info("\n" + "=" * 60)
    logger.info("DECODE COMPARISON (token by token)")
    logger.info("=" * 60)

    hf_generated = [hf_next_token]
    ttnn_generated = [ttnn_next_token]

    current_token_hf = hf_next_token
    current_token_ttnn = ttnn_next_token

    for step in range(num_tokens - 1):
        # HuggingFace decode step
        with torch.no_grad():
            hf_step_input = torch.tensor([[current_token_hf]])
            hf_outputs = hf_model(
                input_ids=hf_step_input,
                past_key_values=hf_past_kv,
                use_cache=True,
            )
            hf_logits = hf_outputs.logits
            hf_past_kv = hf_outputs.past_key_values

        # TTNN decode step
        token_tensor = torch.tensor([[current_token_ttnn]], dtype=torch.long)
        input_ids_ttnn = ttnn.from_torch(
            token_tensor,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=generator.mesh_mapper,
        )
        hidden_states = generator.model.text_model.embed_tokens(input_ids_ttnn)

        ttnn_logits = generator.model.text_model.forward_decode(
            hidden_states=hidden_states,
            kv_caches=generator.kv_caches,
            current_pos=generator.current_pos,
        )
        ttnn.synchronize_device(device)

        # Update position
        generator.decode_position += 1
        new_pos_ttnn = ttnn.from_torch(
            torch.tensor([generator.decode_position], dtype=torch.int32),
            dtype=ttnn.int32,
            device=device,
            mesh_mapper=generator.mesh_mapper,
        )
        ttnn.copy(new_pos_ttnn, generator.current_pos)
        ttnn.deallocate(new_pos_ttnn)
        ttnn.deallocate(hidden_states)

        ttnn_logits_torch = ttnn.to_torch(ttnn_logits)

        # Compare logits
        hf_step_logits = hf_logits[0, -1, :].unsqueeze(0).unsqueeze(0)
        ttnn_step_logits = ttnn_logits_torch[0, 0, -1, :].unsqueeze(0).unsqueeze(0)

        pcc, _ = comp_pcc(hf_step_logits, ttnn_step_logits)

        # Get next tokens
        hf_next = hf_logits[0, -1, :].argmax().item()
        ttnn_next = ttnn_logits_torch[0, 0, -1, :].argmax().item()

        hf_generated.append(hf_next)
        ttnn_generated.append(ttnn_next)

        match_str = "✓" if hf_next == ttnn_next else "✗"
        logger.info(
            f"Step {step + 1}: PCC={pcc:.4f} {match_str} | "
            f"HF={hf_next} ('{tokenizer.decode([hf_next])}') | "
            f"TTNN={ttnn_next} ('{tokenizer.decode([ttnn_next])}')"
        )

        current_token_hf = hf_next
        current_token_ttnn = ttnn_next

        ttnn.deallocate(ttnn_logits)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 60)

    hf_text = tokenizer.decode(hf_generated)
    ttnn_text = tokenizer.decode(ttnn_generated)

    logger.info(f"HF output: {hf_text}")
    logger.info(f"TTNN output: {ttnn_text}")

    matching_tokens = sum(1 for a, b in zip(hf_generated, ttnn_generated) if a == b)
    logger.info(f"Matching tokens: {matching_tokens}/{len(hf_generated)}")

    ttnn.close_device(device)

    return hf_generated, ttnn_generated, prefill_pcc


def run_generation_comparison(num_tokens: int = 20):
    hf_generated, ttnn_generated, prefill_pcc = _run_generation_comparison_impl(num_tokens)
    return hf_generated, ttnn_generated


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


def test_generation_prefill_pcc(device):
    """
    Prefill logits PCC >= 0.95 against HuggingFace reference.

    Cumulative threshold for full 36-layer model is 0.95.
    Individual block PCC >= 0.99 is enforced in test_text_block.py.
    """
    _, _, prefill_pcc = _run_generation_comparison_impl(num_tokens=1)
    assert prefill_pcc >= 0.95, (
        f"Prefill last-token logits PCC {prefill_pcc:.6f} < 0.95. " "Check text_model, embedding, or vision splicing."
    )


def test_generation_token_match(device):
    """
    First 20 greedy-decoded tokens match HF reference.

    Requires prefill to produce correct first token, then decode to follow.
    Even a single token mismatch can diverge the sequence — so we allow
    up to 1 mismatch in 20 tokens (95% match rate).
    """
    hf_tokens, ttnn_tokens, _ = _run_generation_comparison_impl(num_tokens=20)
    matching = sum(1 for a, b in zip(hf_tokens, ttnn_tokens) if a == b)
    match_rate = matching / len(hf_tokens)
    assert match_rate >= 0.95, (
        f"Token match rate {match_rate:.2%} ({matching}/{len(hf_tokens)}) < 95%. "
        "TTNN generation is diverging from HF reference."
    )


if __name__ == "__main__":
    run_generation_comparison(num_tokens=30)
