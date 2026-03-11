# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN functional implementation of the VibeVoice diffusion head from TADA.

VibeVoice is a flow-matching diffusion head that predicts acoustic features
and duration tokens from LLM hidden states.

Architecture:
    noisy_images (B, 528) -> noisy_images_proj -> x (B, 2048)
    timesteps (B,) -> sinusoidal -> TimestepEmbedder MLP -> t (B, 2048)
    condition (B, 2048) -> cond_proj -> cond (B, 2048)
    c = cond + t

    6x HeadLayer:
        adaLN_modulation(c) -> shift, scale, gate
        x = x + gate * FFN(modulate(RMSNorm(x), shift, scale))

    FinalLayer:
        adaLN_modulation(c) -> shift, scale
        x = linear(modulate(RMSNorm(x), shift, scale))

    Output: velocity (B, 528)
"""

import math

import torch

import ttnn

VIBEVOICE_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings on host (CPU).
    This is a small computation that runs once per diffusion step.

    Args:
        t: (B,) tensor of timestep values
        dim: embedding dimension (256)
    Returns:
        (B, dim) tensor of sinusoidal embeddings
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(t.dtype)


def vibevoice_timestep_embedder(t_freq, *, parameters):
    """
    TimestepEmbedder MLP: Linear -> SiLU -> Linear

    Args:
        t_freq: (B, 256) sinusoidal timestep embedding (on device)
        parameters: contains mlp.0 (Linear) and mlp.2 (Linear)
    Returns:
        (B, hidden_size) timestep embedding
    """
    # mlp.0: Linear(256, hidden_size)
    x = ttnn.linear(
        t_freq,
        parameters.mlp[0].weight,
        memory_config=VIBEVOICE_MEMORY_CONFIG,
    )
    x = ttnn.silu(x, memory_config=VIBEVOICE_MEMORY_CONFIG)
    # mlp.2: Linear(hidden_size, hidden_size)
    x = ttnn.linear(
        x,
        parameters.mlp[2].weight,
        memory_config=VIBEVOICE_MEMORY_CONFIG,
    )
    return x


def vibevoice_feedforward(x, *, parameters):
    """
    SwiGLU FeedForward: gate = silu(gate_proj(x)) * up_proj(x); out = down_proj(gate)

    Args:
        x: (B, 1, embed_dim) input
        parameters: contains gate_proj, up_proj, down_proj weights
    Returns:
        (B, 1, embed_dim) output
    """
    gate = ttnn.linear(
        x,
        parameters.gate_proj.weight,
        memory_config=VIBEVOICE_MEMORY_CONFIG,
    )
    up = ttnn.linear(
        x,
        parameters.up_proj.weight,
        memory_config=VIBEVOICE_MEMORY_CONFIG,
    )
    # SwiGLU: silu(gate) * up
    gate = ttnn.silu(gate, memory_config=VIBEVOICE_MEMORY_CONFIG)
    hidden = ttnn.mul(gate, up, memory_config=VIBEVOICE_MEMORY_CONFIG)
    ttnn.deallocate(gate)
    ttnn.deallocate(up)
    out = ttnn.linear(
        hidden,
        parameters.down_proj.weight,
        memory_config=VIBEVOICE_MEMORY_CONFIG,
    )
    ttnn.deallocate(hidden)
    return out


def vibevoice_head_layer(x, c, *, parameters):
    """
    Single HeadLayer with adaptive LayerNorm modulation.

    Args:
        x: (B, 1, embed_dim) input features
        c: (B, 1, cond_dim) conditioning (cond + timestep)
        parameters: contains norm, adaLN_modulation, ffn
    Returns:
        (B, 1, embed_dim) output
    """
    # adaLN_modulation: SiLU -> Linear -> chunk(3)
    mod = ttnn.silu(c, memory_config=VIBEVOICE_MEMORY_CONFIG)
    mod = ttnn.linear(
        mod,
        parameters.adaLN_modulation[1].weight,
        memory_config=VIBEVOICE_MEMORY_CONFIG,
    )
    # chunk into shift, scale, gate (each embed_dim)
    embed_dim = x.shape[-1]
    shift_ffn = mod[:, :, :embed_dim]
    scale_ffn = mod[:, :, embed_dim : 2 * embed_dim]
    gate_ffn = mod[:, :, 2 * embed_dim :]
    ttnn.deallocate(mod)

    # RMSNorm
    x_normed = ttnn.rms_norm(x, weight=parameters.norm.weight, epsilon=1e-5)

    # Modulate: x_normed * (1 + scale) + shift
    ones = ttnn.ones_like(scale_ffn, memory_config=VIBEVOICE_MEMORY_CONFIG)
    scale_plus_one = ttnn.add(ones, scale_ffn, memory_config=VIBEVOICE_MEMORY_CONFIG)
    ttnn.deallocate(ones)
    x_mod = ttnn.mul(x_normed, scale_plus_one, memory_config=VIBEVOICE_MEMORY_CONFIG)
    ttnn.deallocate(x_normed)
    ttnn.deallocate(scale_plus_one)
    x_mod = ttnn.add(x_mod, shift_ffn, memory_config=VIBEVOICE_MEMORY_CONFIG)

    # FFN
    ffn_out = vibevoice_feedforward(x_mod, parameters=parameters.ffn)
    ttnn.deallocate(x_mod)

    # Gated residual: x = x + gate * ffn_out
    gated = ttnn.mul(gate_ffn, ffn_out, memory_config=VIBEVOICE_MEMORY_CONFIG)
    ttnn.deallocate(ffn_out)
    x = ttnn.add(x, gated, memory_config=VIBEVOICE_MEMORY_CONFIG)
    ttnn.deallocate(gated)

    return x


def vibevoice_final_layer(x, c, *, parameters):
    """
    FinalLayer: RMSNorm (no affine) -> modulate -> Linear

    Args:
        x: (B, 1, hidden_size) input
        c: (B, 1, cond_size) conditioning
        parameters: contains norm_final, adaLN_modulation, linear
    Returns:
        (B, 1, output_size) velocity prediction
    """
    # adaLN_modulation: SiLU -> Linear -> chunk(2)
    mod = ttnn.silu(c, memory_config=VIBEVOICE_MEMORY_CONFIG)
    mod = ttnn.linear(
        mod,
        parameters.adaLN_modulation[1].weight,
        memory_config=VIBEVOICE_MEMORY_CONFIG,
    )
    hidden_size = x.shape[-1]
    shift = mod[:, :, :hidden_size]
    scale = mod[:, :, hidden_size:]
    ttnn.deallocate(mod)

    # RMSNorm without affine weights (elementwise_affine=False)
    x_normed = ttnn.rms_norm(x, epsilon=1e-5)

    # Modulate: x_normed * (1 + scale) + shift
    ones = ttnn.ones_like(scale, memory_config=VIBEVOICE_MEMORY_CONFIG)
    scale_plus_one = ttnn.add(ones, scale, memory_config=VIBEVOICE_MEMORY_CONFIG)
    ttnn.deallocate(ones)
    x_mod = ttnn.mul(x_normed, scale_plus_one, memory_config=VIBEVOICE_MEMORY_CONFIG)
    ttnn.deallocate(x_normed)
    ttnn.deallocate(scale_plus_one)
    x_mod = ttnn.add(x_mod, shift, memory_config=VIBEVOICE_MEMORY_CONFIG)

    # Final linear projection
    out = ttnn.linear(
        x_mod,
        parameters.linear.weight,
        memory_config=VIBEVOICE_MEMORY_CONFIG,
    )
    ttnn.deallocate(x_mod)
    return out


def vibevoice_diffusion_head(noisy_images, timesteps_torch, condition, *, parameters, frequency_embedding_size=256):
    """
    Full VibeVoice diffusion head forward pass.

    Args:
        noisy_images: (B, 1, latent_size) noisy speech latent on device
        timesteps_torch: (B,) timestep tensor on HOST (CPU) - used for sinusoidal embedding
        condition: (B, 1, hidden_size) LLM hidden state on device
        parameters: preprocessed model parameters
        frequency_embedding_size: size of sinusoidal embedding (default 256)
    Returns:
        (B, 1, latent_size) velocity prediction on device
    """
    device = noisy_images.device()

    # Project noisy images: (B, 1, latent_size) -> (B, 1, hidden_size)
    x = ttnn.linear(
        noisy_images,
        parameters.noisy_images_proj.weight,
        memory_config=VIBEVOICE_MEMORY_CONFIG,
    )

    # Timestep embedding (computed on host, then transferred to device)
    t_freq = timestep_embedding(timesteps_torch, frequency_embedding_size)
    t_freq = t_freq.unsqueeze(1)  # (B, 1, 256) to match 3D layout
    t_freq = ttnn.from_torch(t_freq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    t = vibevoice_timestep_embedder(t_freq, parameters=parameters.t_embedder)
    ttnn.deallocate(t_freq)

    # Condition projection
    cond = ttnn.linear(
        condition,
        parameters.cond_proj.weight,
        memory_config=VIBEVOICE_MEMORY_CONFIG,
    )

    # c = cond + t
    c = ttnn.add(cond, t, memory_config=VIBEVOICE_MEMORY_CONFIG)
    ttnn.deallocate(cond)
    ttnn.deallocate(t)

    # Process through head layers
    for layer_params in parameters.layers:
        x = vibevoice_head_layer(x, c, parameters=layer_params)

    # Final layer
    x = vibevoice_final_layer(x, c, parameters=parameters.final_layer)
    ttnn.deallocate(c)

    return x


def convert_to_ttnn(model, name):
    """All modules in VibeVoice are standard Linear layers, convert everything."""
    return True


def create_custom_mesh_preprocessor(weights_mesh_mapper):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, weights_mesh_mapper)

    return custom_mesh_preprocessor


def custom_preprocessor(torch_model, name, weights_mesh_mapper):
    """Custom preprocessor for VibeVoice - handles RMSNorm weight conversion."""
    parameters = {}

    # Import the reference RMSNorm class
    try:
        from models.demos.audio.tada.reference.tada_reference import RMSNorm as TadaRMSNorm
    except ImportError:
        TadaRMSNorm = None

    if TadaRMSNorm is not None and isinstance(torch_model, TadaRMSNorm):
        if torch_model.elementwise_affine and torch_model.weight is not None:
            parameters["weight"] = ttnn.from_torch(
                torch_model.weight.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=weights_mesh_mapper,
            )
    return parameters
