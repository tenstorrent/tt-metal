# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple
import torch
from loguru import logger

import ttnn

from .sampling import sample_with_temperature, SamplingParams


def generate_text_tokens(
    model,
    input_tokens: torch.Tensor,
    max_gen_len: int = 512,
    sampling_params: SamplingParams = None,
    encoder_hidden_states: Optional[ttnn.Tensor] = None,
    stop_token_ids: Optional[List[int]] = None,
    mesh_device: Optional[ttnn.MeshDevice] = None,
) -> List[int]:
    """
    Generate text tokens using the model.

    Adapted from generation patterns in models/tt_transformers/demo/simple_text_demo.py

    Args:
        model: TTNN model (Qwen LLM)
        input_tokens: Input token IDs [batch, seq_len]
        max_gen_len: Maximum number of tokens to generate
        sampling_params: Sampling parameters
        encoder_hidden_states: Encoder features for cross-attention (multimodal)
        stop_token_ids: Token IDs that should stop generation

    Returns:
        List of generated token IDs
    """
    if sampling_params is None:
        sampling_params = SamplingParams()

    batch_size, seq_len = input_tokens.shape
    assert batch_size == 1, "Only batch size 1 supported for now"

    # Convert input tokens to TTNN format
    input_tokens_ttnn = ttnn.from_torch(
        input_tokens,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=model.mesh_device,
    )

    # Prefill phase - process initial tokens
    logits, kv_cache = prefill_forward(model, input_tokens_ttnn, encoder_hidden_states=encoder_hidden_states)

    # Get first token logits and convert to torch for sampling
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=2) if mesh_device else None
    logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer).float()

    # Handle Qwen LLM output shape: [1, 1, seq_len, vocab_size] -> [batch, seq_len, vocab_size]
    logits_torch = logits_torch.squeeze(
        0
    )  # Remove the first dimension [1, 1, seq_len, vocab_size] -> [1, seq_len, vocab_size]
    if logits_torch.dim() == 3 and logits_torch.shape[0] == 1:
        logits_torch = logits_torch.squeeze(0)  # [1, seq_len, vocab_size] -> [seq_len, vocab_size]

    # Initialize generation
    generated_tokens = []
    current_pos = seq_len
    stop_generation = False

    # Get the last token's logits for first generation step
    if logits_torch.dim() == 2:  # [seq_len, vocab_size]
        next_token_logits = logits_torch[-1, :]  # Last position logits
    else:  # [batch, seq_len, vocab_size]
        next_token_logits = logits_torch[:, -1, :]  # Last position logits

    # Generation loop
    for gen_step in range(max_gen_len):
        if stop_generation:
            break

        # Sample next token (skip for first iteration as we already have next_token_logits)
        if gen_step > 0:
            if logits_torch.dim() == 2:  # [seq_len, vocab_size]
                next_token_logits = logits_torch[-1, :]  # Last position logits
            else:  # [batch, seq_len, vocab_size]
                next_token_logits = logits_torch[:, -1, :]  # Last position logits
        next_token = sample_with_temperature(
            next_token_logits,
            temperature=sampling_params.get_temperature(),
            top_p=sampling_params.get_top_p(),
            top_k=sampling_params.get_top_k() if sampling_params.get_top_k() > 0 else None,
        )

        next_token_id = next_token.item()
        generated_tokens.append(next_token_id)

        # Check stop conditions
        if stop_token_ids and next_token_id in stop_token_ids:
            logger.debug(f"Stopping generation at token {next_token_id}")
            stop_generation = True
            break

        # Decode phase - generate next token
        next_token_ttnn = ttnn.from_torch(
            next_token.unsqueeze(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=model.mesh_device,
        )

        logits, kv_cache = decode_forward(
            model, next_token_ttnn, current_pos, kv_cache, encoder_hidden_states=encoder_hidden_states
        )

        # Convert logits for next iteration - handle Qwen LLM output format
        logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer).float()

        # Handle Qwen LLM decode output shape: [1, 1, 1, vocab_size] -> [vocab_size]
        logits_torch = logits_torch.squeeze()  # Remove all singleton dimensions
        if logits_torch.dim() > 1:
            logits_torch = logits_torch.squeeze(0)  # Ensure we have [vocab_size] shape

        current_pos += 1

    return generated_tokens


def prefill_forward(
    model,
    input_tokens: ttnn.Tensor,
    encoder_hidden_states: Optional[ttnn.Tensor] = None,
) -> Tuple[ttnn.Tensor, dict]:
    """
    Prefill forward pass - process initial prompt tokens.

    Adapted from prefill_forward_text in models/tt_transformers/demo/simple_text_demo.py:875-894

    Args:
        model: TTNN model (Qwen LLM)
        input_tokens: Input token IDs [batch, seq_len]
        encoder_hidden_states: Encoder features for cross-attention

    Returns:
        Tuple of (logits, kv_cache)
    """
    logger.debug("Starting prefill forward pass")

    # Call model's forward method for prefill
    logger.debug(
        f"Calling model.forward with input_ids shape: {input_tokens.shape if hasattr(input_tokens, 'shape') else 'No shape'}"
    )
    logger.debug(f"Model type: {type(model)}, model is None: {model is None}")
    if hasattr(model, "qwen_llm"):
        logger.debug(f"Model has qwen_llm: {model.qwen_llm is not None}")
    else:
        logger.debug("Model does not have qwen_llm attribute")

    logits = model.forward(
        tokens=input_tokens,
        start_pos=0,
        encoder_hidden_states=encoder_hidden_states,
        mode="prefill",
    )

    logger.debug(
        f"Model forward returned: {type(logits)}, shape: {logits.shape if hasattr(logits, 'shape') else 'No shape'}"
    )

    # Extract KV cache from model (this is a placeholder - actual implementation
    # would depend on how KV cache is managed in the model)
    kv_cache = getattr(model, "kv_cache", None)

    logger.debug("Prefill forward pass completed")
    return logits, kv_cache


def decode_forward(
    model,
    next_token: ttnn.Tensor,
    position: int,
    kv_cache: dict,
    encoder_hidden_states: Optional[ttnn.Tensor] = None,
) -> Tuple[ttnn.Tensor, dict]:
    """
    Decode forward pass - generate next token.

    Adapted from decode_forward_text in models/tt_transformers/demo/simple_text_demo.py:933-954

    Args:
        model: TTNN model (Qwen LLM)
        next_token: Next token to process [batch, 1]
        position: Current position in sequence
        kv_cache: Key-value cache from previous steps
        encoder_hidden_states: Encoder features for cross-attention

    Returns:
        Tuple of (logits, updated_kv_cache)
    """
    logger.debug(f"Decode step at position {position}")

    # Call model's forward method for decode (single token)
    # Note: This is a simplified implementation. Actual decode would use
    # optimized decode forward with KV cache management
    logits = model.forward(
        tokens=next_token,
        start_pos=position,
        encoder_hidden_states=encoder_hidden_states,
        mode="decode",
    )

    # Update KV cache (placeholder)
    updated_kv_cache = kv_cache

    logger.debug(f"Decode step completed at position {position}")
    return logits, updated_kv_cache


def generate_text_with_callback(
    model,
    input_tokens: torch.Tensor,
    max_gen_len: int = 512,
    sampling_params: SamplingParams = None,
    encoder_hidden_states: Optional[ttnn.Tensor] = None,
    stop_token_ids: Optional[List[int]] = None,
    callback: Optional[callable] = None,
    mesh_device: Optional[ttnn.MeshDevice] = None,
) -> Tuple[List[int], List[str]]:
    """
    Generate text tokens with callback for real-time output.

    Similar to Generator.generate method in models/tt_transformers/tt/generator.py:1371-1456

    Args:
        model: TTNN model
        input_tokens: Input token IDs
        max_gen_len: Maximum generation length
        sampling_params: Sampling parameters
        encoder_hidden_states: Encoder features for cross-attention
        stop_token_ids: Stop token IDs
        callback: Callback function called for each generated token

    Returns:
        Tuple of (generated_token_ids, generated_texts)
    """
    if sampling_params is None:
        sampling_params = SamplingParams()

    batch_size, seq_len = input_tokens.shape
    assert batch_size == 1, "Only batch size 1 supported"

    # Convert input tokens to TTNN format
    input_tokens_ttnn = ttnn.from_torch(
        input_tokens,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=model.mesh_device,
    )

    # Prefill phase
    logits, kv_cache = prefill_forward(model, input_tokens_ttnn, encoder_hidden_states=encoder_hidden_states)

    # Get first token logits
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=2) if mesh_device else None
    logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer).float()
    next_token_logits = logits_torch[:, -1, :]

    # Sample first token
    next_token = sample_with_temperature(
        next_token_logits,
        temperature=sampling_params.get_temperature(),
        top_p=sampling_params.get_top_p(),
        top_k=sampling_params.get_top_k() if sampling_params.get_top_k() > 0 else None,
    )

    generated_tokens = []
    generated_texts = []

    # Yield first token
    first_token_id = next_token.item()
    generated_tokens.append(first_token_id)

    # Placeholder for text decoding - would use tokenizer
    first_text = f"<token_{first_token_id}>"
    generated_texts.append(first_text)

    if callback:
        callback(first_token_id, first_text)

    # Check stop condition
    if stop_token_ids and first_token_id in stop_token_ids:
        return generated_tokens, generated_texts

    current_pos = seq_len

    # Generation loop
    for gen_step in range(1, max_gen_len):
        # Decode next token
        next_token_ttnn = ttnn.from_torch(
            next_token.unsqueeze(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=model.mesh_device,
        )

        logits, kv_cache = decode_forward(
            model, next_token_ttnn, current_pos, kv_cache, encoder_hidden_states=encoder_hidden_states
        )

        # Sample next token
        logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer).float()
        next_token_logits = logits_torch[:, -1, :]
        next_token = sample_with_temperature(
            next_token_logits,
            temperature=sampling_params.get_temperature(),
            top_p=sampling_params.get_top_p(),
            top_k=sampling_params.get_top_k() if sampling_params.get_top_k() > 0 else None,
        )

        token_id = next_token.item()
        generated_tokens.append(token_id)

        # Placeholder for text decoding
        text = f"<token_{token_id}>"
        generated_texts.append(text)

        if callback:
            callback(token_id, text)

        # Check stop condition
        if stop_token_ids and token_id in stop_token_ids:
            break

        current_pos += 1

    return generated_tokens, generated_texts
