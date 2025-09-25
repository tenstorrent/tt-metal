# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache

from models.demos.deepseek_v3.reference.modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
)


def pad_input_for_decode_mode(torch_input, position_ids):
    """Pad the input tensor for decode mode to the largest position id in the batch."""
    bsz, q_len, dim = torch_input.shape
    assert q_len == 1, "Decode mode should have sequence length of 1"

    max_position_idx = position_ids.max().item()
    new_torch_input = torch.zeros(bsz, max_position_idx + 1, dim)
    for b in range(bsz):
        pos = position_ids[b].item()
        new_torch_input[b, pos] = torch_input[b, 0]
    return new_torch_input


def pad_input_ids_for_decode_mode(torch_input, position_ids):
    """Pad the input tensor of ids for decode mode to the largest position id in the batch."""
    bsz, q_len = torch_input.shape
    assert q_len == 1, "Decode mode should have sequence length of 1"

    max_position_idx = position_ids.max().item()
    new_torch_input = torch.zeros(bsz, max_position_idx + 1, dtype=torch.long)
    for b in range(bsz):
        pos = position_ids[b].item()
        new_torch_input[b, pos] = torch_input[b, 0]
    return new_torch_input


def create_mask(bsz, q_len, position_ids, mode):
    """Create a mask for the attention mechanism."""
    if mode == "prefill":
        return torch.triu(torch.full((bsz, 1, q_len, q_len), float("-inf")), diagonal=1)
    else:
        mask = torch.full((bsz, 1, q_len, q_len), float("-inf"))

        for b in range(bsz):
            usable_len = position_ids[b].item() + 1
            mask[b, 0, :usable_len, :usable_len] = torch.triu(
                torch.full((usable_len, usable_len), float("-inf")), diagonal=1
            )
    return mask


def reference_forward_mla(
    reference_model: DeepseekV3Attention,
    torch_input: torch.Tensor,
    position_ids: torch.LongTensor,
    mode: str,
) -> torch.Tensor:
    """Run the reference model forward pass.

    This function specifically uses MLA with the "absorption" implementation,
    called with `forward_mla`, which is simply an extension on top of HF's DeepSeek implementation.

    Notes for Decode Mode:
        NOTE: This function currently does not support multi-iteration decoding.

        It is critical to note that since the HF implementation uses DynamicCache, we cannot just
        perform decode on users with arbitrry position_ids. Instead, if we want to simulate this situation,
        we need to pad the input to the largest position id in the batch, and then use it for the forward pass.

        This has the same effect as doing prefill. As such, on the output side, we must also
        slice the output to only return the last token for each user.

        The reference cache is also sliced in a similar manner to only return the last token's cache.

    Notes on position_ids_expanded:
        The HF implementation uses position_ids_expanded, which is a tensor of shape [bsz, q_len].


    Args:
        reference_model (DeepseekV3Attention): The reference model to run.
        torch_input (torch.Tensor): The input tensor to the model.
        position_ids (torch.LongTensor): The position ids for the input.
        mode (str): The mode of operation, either "decode" or "prefill".
    Returns:
        torch.Tensor: The output tensor from the model.
    """

    reference_model = reference_model.to(torch.float32)

    bsz, q_len, dim = torch_input.shape
    torch_input = torch_input.to(dtype=torch.float32)

    assert bsz == position_ids.shape[0], "Batch size of input and position_ids must match"

    # Generate the cache
    cache = DynamicCache()

    # Padding for decode mode
    if mode == "decode":
        torch_input = pad_input_for_decode_mode(torch_input, position_ids)
        q_len = torch_input.shape[1]  # Update q_len to the new sequence length

    # Create the mask
    mask = create_mask(bsz, q_len, position_ids, mode)

    position_ids_expanded = torch.arange(0, q_len, dtype=torch.long).unsqueeze(0).repeat(bsz, 1)

    out, _, past_key_value = reference_model.forward(
        hidden_states=torch_input,
        attention_mask=mask,
        position_ids=position_ids_expanded,
        past_key_value=cache,
    )
    cache = past_key_value.key_cache[reference_model.layer_idx].squeeze(1)

    if mode == "decode":
        # Get last token
        batch_indices = torch.arange(bsz)
        out = out[batch_indices, position_ids, :].unsqueeze(1)  # [bsz, 1, hidden_size]
        cache = cache[batch_indices, position_ids, :].unsqueeze(1)  # [bsz, 1, head_dim + rope_head_dim]

    return out, cache


def reference_forward_decoder(
    reference_model: DeepseekV3DecoderLayer,
    torch_input: torch.Tensor,
    position_ids: torch.LongTensor,
    mode: str,
) -> torch.Tensor:
    """Run the reference model forward pass.

    This function specifically uses MLA with the "absorption" implementation,
    called with `forward_mla`, which is simply an extension on top of HF's DeepSeek implementation.

    Notes for Decode Mode:
        NOTE: This function currently does not support multi-iteration decoding.

        It is critical to note that since the HF implementation uses DynamicCache, we cannot just
        perform decode on users with arbitrry position_ids. Instead, if we want to simulate this situation,
        we need to pad the input to the largest position id in the batch, and then use it for the forward pass.

        This has the same effect as doing prefill. As such, on the output side, we must also
        slice the output to only return the last token for each user.

        The reference cache is also sliced in a similar manner to only return the last token's cache.

    Notes on position_ids_expanded:
        The HF implementation uses position_ids_expanded, which is a tensor of shape [bsz, q_len].


    Args:
        reference_model (DeepseekV3DecoderLayer): The reference model to run.
        torch_input (torch.Tensor): The input tensor to the model.
        position_ids (torch.LongTensor): The position ids for the input.
        mode (str): The mode of operation, either "decode" or "prefill".
    Returns:
        torch.Tensor: The output tensor from the model.
    """

    reference_model = reference_model.to(torch.float32)

    bsz, q_len, dim = torch_input.shape

    assert bsz == position_ids.shape[0], "Batch size of input and position_ids must match"

    # Generate the cache
    cache = DynamicCache()

    # Pre-populate cache with dummy entries for layers before the target layer
    # This is necessary when testing a single layer in isolation
    layer_idx = reference_model.self_attn.layer_idx
    if layer_idx > 0 and len(cache.key_cache) < layer_idx:
        # For DeepSeek v3 MLA, cache entries have shape [bsz, 1, 0, kv_lora_rank + qk_rope_head_dim]
        # where the sequence length is 0 initially
        cache_head_dim = reference_model.self_attn.kv_lora_rank + reference_model.self_attn.qk_rope_head_dim
        dummy_key_value = torch.zeros(bsz, 1, 0, cache_head_dim, dtype=torch.float32)

        # Only add dummy entries for missing layers
        for i in range(len(cache.key_cache), layer_idx):
            cache.key_cache.append(dummy_key_value.clone())
            cache.value_cache.append(dummy_key_value.clone())

    # Padding for decode mode
    if mode == "decode":
        torch_input = pad_input_for_decode_mode(torch_input, position_ids)
        q_len = torch_input.shape[1]  # Update q_len to the new sequence length

    # Create the mask
    mask = create_mask(bsz, q_len, position_ids, mode)

    position_ids_expanded = torch.arange(0, q_len, dtype=torch.long).unsqueeze(0).repeat(bsz, 1)

    out, past_key_value = reference_model.forward(
        hidden_states=torch_input,
        attention_mask=mask,
        position_ids=position_ids_expanded,
        past_key_value=cache,
        use_cache=True,
    )
    cache = past_key_value.key_cache[reference_model.self_attn.layer_idx].squeeze(1)

    if mode == "decode":
        # Get last token
        batch_indices = torch.arange(bsz)
        out = out[batch_indices, position_ids, :].unsqueeze(1)  # [bsz, 1, hidden_size]
        cache = cache[batch_indices, position_ids, :].unsqueeze(1)  # [bsz, 1, head_dim + rope_head_dim]

    return out, cache


def reference_forward_model(
    reference_model: DeepseekV3Model | DeepseekV3ForCausalLM,
    torch_input_ids: torch.Tensor,
    position_ids: torch.LongTensor,
    mode: str,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Run the reference model forward pass.

    This function specifically uses MLA with the "absorption" implementation,
    called with `forward_mla`, which is simply an extension on top of HF's DeepSeek implementation.

    Notes for Decode Mode:
        NOTE: This function currently does not support multi-iteration decoding.

        It is critical to note that since the HF implementation uses DynamicCache, we cannot just
        perform decode on users with arbitrry position_ids. Instead, if we want to simulate this situation,
        we need to pad the input to the largest position id in the batch, and then use it for the forward pass.

        This has the same effect as doing prefill. As such, on the output side, we must also
        slice the output to only return the last token for each user.

        The reference cache is also sliced in a similar manner to only return the last token's cache.

    Notes on position_ids_expanded:
        The HF implementation uses position_ids_expanded, which is a tensor of shape [bsz, q_len].


    Args:
        reference_model (DeepseekV3Model | DeepseekV3CausalLM): The reference model to run.
        torch_input_ids (torch.Tensor): The input tensor to the model.
        position_ids (torch.LongTensor): The position ids for the input.
        mode (str): The mode of operation, either "decode" or "prefill".
    Returns:
        torch.Tensor: The output tensor from the model.
    """

    reference_model = reference_model.to(dtype)

    bsz, seq_len = torch_input_ids.shape

    assert bsz == position_ids.shape[0], "Batch size of input and position_ids must match"

    # Generate the cache

    # Padding for decode mode
    if mode == "decode":
        torch_input_ids = pad_input_ids_for_decode_mode(torch_input_ids, position_ids)
        seq_len = torch_input_ids.shape[1]  # Update seq_len to the new sequence length

    # Create the mask
    mask = create_mask(bsz, seq_len, position_ids, mode)

    position_ids_expanded = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).repeat(bsz, 1)

    out = reference_model.forward(
        input_ids=torch_input_ids,
        attention_mask=mask,
        position_ids=position_ids_expanded,
        use_cache=True,
    )

    if isinstance(reference_model, DeepseekV3Model):
        out = out.last_hidden_state
    elif isinstance(reference_model, DeepseekV3ForCausalLM):
        out = out.logits
    else:
        raise ValueError("Unsupported model type")

    if mode == "decode":
        # Get last token
        batch_indices = torch.arange(bsz)
        out = out[batch_indices, position_ids, :].unsqueeze(1)  # [bsz, 1, hidden_size]

    return out
