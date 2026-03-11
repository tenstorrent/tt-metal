# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN functional implementation of TadaForCausalLM.

TadaForCausalLM extends Llama 3.2 1B with:
1. acoustic_proj: Linear(512, 2048) - projects acoustic features into LLM space
2. time_start_embed / time_end_embed: Embedding(256, 2048) - duration tokens
3. acoustic_mask_emb: Embedding(2, 2048) - mask for acoustic presence
4. prediction_head: VibeVoice diffusion head for acoustic generation

Per autoregressive step:
    inputs_embeds = embed_tokens(text_id) + acoustic_proj(acoustic)
                  + acoustic_mask_emb(mask) + time_start_embed(t_start) + time_end_embed(t_end)
    hidden = LlamaModel(inputs_embeds, past_key_values)
    text_logits = lm_head(hidden)
    speech = VibeVoice ODE(hidden, noise) -> acoustic features + duration

The Llama backbone can leverage tt_transformers (existing Llama implementation).
"""

import torch

import ttnn

TADA_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG


def tada_embed_inputs(
    input_ids,
    acoustic_features,
    acoustic_masks,
    time_len_before,
    time_len_after,
    *,
    parameters,
    device,
    input_mesh_mapper,
):
    """
    Create combined input embeddings for one autoregressive step.

    inputs_embeds = embed_tokens(input_ids) + acoustic_proj(acoustic_features)
                  + acoustic_mask_emb(acoustic_masks)
                  + time_start_embed(time_len_before) + time_end_embed(time_len_after)

    Args:
        input_ids: (B,) text token IDs on CPU (torch.long)
        acoustic_features: (B, 512) acoustic features on CPU
        acoustic_masks: (B,) binary mask on CPU (0=no acoustic, 1=has acoustic)
        time_len_before: (B,) duration before token on CPU (0-255)
        time_len_after: (B,) duration after token on CPU (0-255)
        parameters: contains embed_tokens, acoustic_proj, acoustic_mask_emb,
                    time_start_embed, time_end_embed weights
        device: TT device
        input_mesh_mapper: mesh mapper
    Returns:
        (B, 1, 2048) combined embeddings on device
    """
    # Token embedding
    tt_input_ids = ttnn.from_torch(
        input_ids.unsqueeze(1),  # (B, 1)
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=input_mesh_mapper,
    )
    token_emb = ttnn.embedding(
        tt_input_ids,
        parameters.model.embed_tokens.weight,
        layout=ttnn.TILE_LAYOUT,
        memory_config=TADA_MEMORY_CONFIG,
    )

    # Acoustic projection
    acoustic_tt = ttnn.from_torch(
        acoustic_features.unsqueeze(1),  # (B, 1, 512)
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=input_mesh_mapper,
    )
    try:
        acoustic_proj_bias = parameters.acoustic_proj.bias
    except (AttributeError, KeyError):
        acoustic_proj_bias = None
    acoustic_emb = ttnn.linear(
        acoustic_tt,
        parameters.acoustic_proj.weight,
        bias=acoustic_proj_bias,
        memory_config=TADA_MEMORY_CONFIG,
    )
    ttnn.deallocate(acoustic_tt)

    # Acoustic mask embedding
    tt_masks = ttnn.from_torch(
        acoustic_masks.unsqueeze(1),  # (B, 1)
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=input_mesh_mapper,
    )
    mask_emb = ttnn.embedding(
        tt_masks,
        parameters.acoustic_mask_emb.weight,
        layout=ttnn.TILE_LAYOUT,
        memory_config=TADA_MEMORY_CONFIG,
    )

    # Time embeddings
    tt_time_before = ttnn.from_torch(
        time_len_before.unsqueeze(1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=input_mesh_mapper,
    )
    time_start_emb = ttnn.embedding(
        tt_time_before,
        parameters.time_start_embed.weight,
        layout=ttnn.TILE_LAYOUT,
        memory_config=TADA_MEMORY_CONFIG,
    )

    tt_time_after = ttnn.from_torch(
        time_len_after.unsqueeze(1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=input_mesh_mapper,
    )
    time_end_emb = ttnn.embedding(
        tt_time_after,
        parameters.time_end_embed.weight,
        layout=ttnn.TILE_LAYOUT,
        memory_config=TADA_MEMORY_CONFIG,
    )

    # Sum all embeddings
    inputs_embeds = ttnn.add(token_emb, acoustic_emb, memory_config=TADA_MEMORY_CONFIG)
    ttnn.deallocate(token_emb)
    ttnn.deallocate(acoustic_emb)
    inputs_embeds = ttnn.add(inputs_embeds, mask_emb, memory_config=TADA_MEMORY_CONFIG)
    ttnn.deallocate(mask_emb)
    inputs_embeds = ttnn.add(inputs_embeds, time_start_emb, memory_config=TADA_MEMORY_CONFIG)
    ttnn.deallocate(time_start_emb)
    inputs_embeds = ttnn.add(inputs_embeds, time_end_emb, memory_config=TADA_MEMORY_CONFIG)
    ttnn.deallocate(time_end_emb)

    return inputs_embeds


def tada_lm_head(hidden_states, *, parameters):
    """
    Apply the language model head to get text token logits.

    Args:
        hidden_states: (B, 1, 2048) from Llama backbone on device
        parameters: contains lm_head weight
    Returns:
        (B, 1, vocab_size) logits on device
    """
    logits = ttnn.linear(
        hidden_states,
        parameters.lm_head.weight,
        memory_config=TADA_MEMORY_CONFIG,
    )
    return logits


def convert_to_ttnn(model, name):
    """
    Determine which modules to convert to TTNN tensors.
    Keep embedding weights as-is for the embedding op.
    """
    if name in ["model.embed_tokens", "acoustic_mask_emb", "time_start_embed", "time_end_embed"]:
        return False
    return True


def create_custom_mesh_preprocessor(weights_mesh_mapper):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, weights_mesh_mapper)

    return custom_mesh_preprocessor


def custom_preprocessor(torch_model, name, weights_mesh_mapper):
    """Custom preprocessor for TadaForCausalLM."""
    parameters = {}

    # Handle embedding layers - need ROW_MAJOR for ttnn.embedding
    if isinstance(torch_model, torch.nn.Embedding):
        if any(
            emb_name in name for emb_name in ["embed_tokens", "acoustic_mask_emb", "time_start_embed", "time_end_embed"]
        ):
            parameters["weight"] = ttnn.from_torch(
                torch_model.weight.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=weights_mesh_mapper,
            )

    # Handle VibeVoice RMSNorm
    try:
        from models.demos.audio.tada.reference.tada_reference import RMSNorm as TadaRMSNorm

        if isinstance(torch_model, TadaRMSNorm) and torch_model.elementwise_affine and torch_model.weight is not None:
            parameters["weight"] = ttnn.from_torch(
                torch_model.weight.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=weights_mesh_mapper,
            )
    except ImportError:
        pass

    return parameters
