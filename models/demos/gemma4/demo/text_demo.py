# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 text generation demo.

Simple prefill + decode loop following gpt-oss text_demo.py pattern.

Usage:
    pytest models/demos/gemma4/demo/text_demo.py -v --timeout=600

    # With fewer layers for testing:
    pytest models/demos/gemma4/demo/text_demo.py -v --timeout=600 -k "test_demo"
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.common import create_tt_model


def run_generation(
    mesh_device,
    model_path,
    prompts,
    max_new_tokens=32,
    num_layers=None,
):
    """
    Run text generation with Gemma4.

    Args:
        mesh_device: TT device
        model_path: Path to model weights
        prompts: List of prompt strings
        max_new_tokens: Number of tokens to generate per prompt
        num_layers: Override layer count (for quick testing)

    Returns:
        List of generated text strings
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"Tokenizer loaded from {model_path}")

    # Create model
    max_seq_len = 128  # Keep very small for initial testing
    logger.info(f"Creating model with {num_layers or 'all'} layers, max_seq_len={max_seq_len}...")
    t0 = time.time()
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        model_path=model_path,
        create_kv_cache=True,
    )
    logger.info(f"Model created in {time.time() - t0:.1f}s")

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    generated_texts = []

    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt {prompt_idx}: {prompt}")

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)  # [seq_len]
        prompt_len = input_ids.shape[0]
        # Pad to tile alignment for prefill
        padded_len = ((prompt_len + 31) // 32) * 32
        if padded_len > prompt_len:
            input_ids_padded = torch.nn.functional.pad(input_ids, (0, padded_len - prompt_len), value=0)
        else:
            input_ids_padded = input_ids
        logger.info(f"Prompt tokens: {prompt_len} (padded to {padded_len})")

        # Prefill
        logger.info("Prefilling...")
        t_prefill = time.time()

        import traceback as tb

        tokens_tt = ttnn.from_torch(
            input_ids_padded.unsqueeze(0).to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = model.embed_tokens(tokens_tt)
        embeds = ttnn.reshape(embeds, (1, 1, padded_len, model_args.hidden_size))
        embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

        # CPU tensors for per-layer input computation (E2B/E4B)
        import torch.nn.functional as F

        embeds_torch = (
            F.embedding(
                input_ids_padded.unsqueeze(0).long(),
                state_dict.get(
                    "model.language_model.embed_tokens.weight",
                    state_dict.get("model.embed_tokens.weight", torch.zeros(1)),
                ),
            )
            * model.embed_scale
        ).float()

        # Get last token tile for first decode token
        get_last_token = ((prompt_len - 1) // 32) * 32
        try:
            logits = model.ttnn_prefill_forward(
                embeds,
                page_table=None,
                kv_cache=tt_kv_cache,
                get_last_token=get_last_token,
                input_ids_torch=input_ids_padded.unsqueeze(0),
                embeds_torch=embeds_torch,
            )
        except Exception as e:
            logger.error(f"Prefill failed: {e}")
            tb.print_exc()
            raise

        # Sample first token (argmax from last position)
        if is_mesh:
            logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
        else:
            logits_cpu = ttnn.to_torch(logits)
        logits.deallocate(True)

        # Get logits at the actual last prompt position within the tile
        pos_in_tile = (prompt_len - 1) - get_last_token
        next_token = logits_cpu[0, 0, pos_in_tile, :].argmax().item()

        prefill_time = time.time() - t_prefill
        logger.info(
            f"Prefill done in {prefill_time:.2f}s, first token: {next_token} = '{tokenizer.decode([next_token])}'"
        )

        # Decode loop
        generated_tokens = [next_token]
        current_pos = prompt_len

        logger.info("Decoding...")
        t_decode = time.time()

        for step in range(max_new_tokens - 1):
            # Prepare decode input
            token_tensor = torch.tensor([[next_token]], dtype=torch.int32)
            token_tt = ttnn.from_torch(
                token_tensor,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
            )

            position_idx = torch.tensor([[current_pos]], dtype=torch.int32)
            position_tt = ttnn.from_torch(
                position_idx,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                mesh_mapper=replicate,
            )

            decode_logits, _ = model.ttnn_decode_forward(
                token_tt,
                current_pos=position_tt,
                kv_cache=tt_kv_cache,
                input_ids_torch=token_tensor,
            )

            if is_mesh:
                logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(decode_logits)[0])
            else:
                logits_cpu = ttnn.to_torch(decode_logits)
            decode_logits.deallocate(True)

            next_token = logits_cpu.squeeze().argmax().item()
            generated_tokens.append(next_token)
            current_pos += 1

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break

        decode_time = time.time() - t_decode
        tokens_per_sec = len(generated_tokens) / decode_time if decode_time > 0 else 0

        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = prompt + generated_text
        generated_texts.append(full_text)

        logger.info(f"Generated {len(generated_tokens)} tokens in {decode_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
        logger.info(f"Output: {full_text}")

    return generated_texts


# ── Pytest entry points ──────────────────────────────────────────────────


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "/proj_sw/user_dev/gemma4/gemma-4-26B-A4B-it")


def test_demo_single_layer(device, model_path):
    """Quick demo with 1 layer — verifies the pipeline works on single device."""
    prompts = ["The capital of France is"]
    results = run_generation(
        mesh_device=device,
        model_path=model_path,
        prompts=prompts,
        max_new_tokens=8,
        num_layers=1,
    )
    assert len(results) == 1
    assert len(results[0]) > len(prompts[0])


def test_demo_full_model(device, model_path):
    """Full model demo — requires sufficient DRAM for all layers."""
    prompts = ["Explain quantum computing in simple terms:"]
    results = run_generation(
        mesh_device=device,
        model_path=model_path,
        prompts=prompts,
        max_new_tokens=64,
    )
    assert len(results) == 1
    logger.info(f"Full model output: {results[0]}")
