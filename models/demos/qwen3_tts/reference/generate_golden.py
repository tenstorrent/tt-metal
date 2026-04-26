# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Generate golden outputs for Qwen3-TTS reference implementation tests.

This script creates deterministic test inputs and outputs for verification.
"""

import os
import sys

import torch

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
)

from models.demos.qwen3_tts.reference.functional import (
    attention,
    compute_rope_frequencies,
    decoder_layer,
    get_default_talker_config,
    rms_norm,
    swiglu_mlp,
)


def get_golden_dir():
    """Get the golden outputs directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    golden_dir = os.path.join(script_dir, "golden")
    os.makedirs(golden_dir, exist_ok=True)
    return golden_dir


def generate_rms_norm_golden():
    """Generate golden output for RMSNorm."""
    torch.manual_seed(0)

    hidden_size = 2048
    batch, seq_len = 2, 10
    eps = 1e-6

    hidden_states = torch.randn(batch, seq_len, hidden_size)
    weight = torch.randn(hidden_size)

    output = rms_norm(hidden_states, weight, eps)

    golden = {
        "input": hidden_states,
        "weight": weight,
        "eps": eps,
        "output": output,
        "config": {
            "hidden_size": hidden_size,
            "batch": batch,
            "seq_len": seq_len,
        },
    }

    golden_path = os.path.join(get_golden_dir(), "rms_norm_golden.pt")
    torch.save(golden, golden_path)
    print(f"Saved RMSNorm golden to {golden_path}")
    return golden


def generate_mlp_golden():
    """Generate golden output for SwiGLU MLP."""
    torch.manual_seed(0)

    hidden_size = 2048
    intermediate_size = 6144
    batch, seq_len = 2, 10

    hidden_states = torch.randn(batch, seq_len, hidden_size)
    gate_proj_weight = torch.randn(intermediate_size, hidden_size)
    up_proj_weight = torch.randn(intermediate_size, hidden_size)
    down_proj_weight = torch.randn(hidden_size, intermediate_size)

    output = swiglu_mlp(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight)

    golden = {
        "input": hidden_states,
        "gate_proj_weight": gate_proj_weight,
        "up_proj_weight": up_proj_weight,
        "down_proj_weight": down_proj_weight,
        "output": output,
        "config": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "batch": batch,
            "seq_len": seq_len,
        },
    }

    golden_path = os.path.join(get_golden_dir(), "mlp_golden.pt")
    torch.save(golden, golden_path)
    print(f"Saved MLP golden to {golden_path}")
    return golden


def generate_attention_golden():
    """Generate golden output for attention."""
    torch.manual_seed(0)

    config = get_default_talker_config()
    batch, seq_len = 2, 10

    hidden_states = torch.randn(batch, seq_len, config.hidden_size)

    # Create weight tensors
    q_proj_weight = torch.randn(config.num_attention_heads * config.head_dim, config.hidden_size)
    k_proj_weight = torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size)
    v_proj_weight = torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size)
    o_proj_weight = torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim)
    q_norm_weight = torch.ones(config.head_dim)
    k_norm_weight = torch.ones(config.head_dim)

    # Use standard RoPE for simpler golden (not MROPE)
    cos, sin = compute_rope_frequencies(config.head_dim, seq_len)

    output = attention(
        hidden_states,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        use_mrope=False,
    )

    golden = {
        "input": hidden_states,
        "q_proj_weight": q_proj_weight,
        "k_proj_weight": k_proj_weight,
        "v_proj_weight": v_proj_weight,
        "o_proj_weight": o_proj_weight,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "cos": cos,
        "sin": sin,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "use_mrope": False,
        "output": output,
        "config": {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "batch": batch,
            "seq_len": seq_len,
        },
    }

    golden_path = os.path.join(get_golden_dir(), "attention_golden.pt")
    torch.save(golden, golden_path)
    print(f"Saved attention golden to {golden_path}")
    return golden


def generate_decoder_layer_golden():
    """Generate golden output for decoder layer."""
    torch.manual_seed(0)

    config = get_default_talker_config()
    batch, seq_len = 2, 10

    hidden_states = torch.randn(batch, seq_len, config.hidden_size)

    # Create layer weights
    layer_weights = {
        "input_layernorm.weight": torch.randn(config.hidden_size),
        "self_attn.q_proj.weight": torch.randn(config.num_attention_heads * config.head_dim, config.hidden_size),
        "self_attn.k_proj.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        "self_attn.v_proj.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        "self_attn.o_proj.weight": torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim),
        "self_attn.q_norm.weight": torch.ones(config.head_dim),
        "self_attn.k_norm.weight": torch.ones(config.head_dim),
        "post_attention_layernorm.weight": torch.randn(config.hidden_size),
        "mlp.gate_proj.weight": torch.randn(config.intermediate_size, config.hidden_size),
        "mlp.up_proj.weight": torch.randn(config.intermediate_size, config.hidden_size),
        "mlp.down_proj.weight": torch.randn(config.hidden_size, config.intermediate_size),
    }

    # Use standard RoPE for simpler golden
    cos, sin = compute_rope_frequencies(config.head_dim, seq_len)

    output = decoder_layer(hidden_states, layer_weights, cos, sin, config, use_mrope=False)

    golden = {
        "input": hidden_states,
        "layer_weights": layer_weights,
        "cos": cos,
        "sin": sin,
        "output": output,
        "config": {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.intermediate_size,
            "rms_norm_eps": config.rms_norm_eps,
            "batch": batch,
            "seq_len": seq_len,
        },
    }

    golden_path = os.path.join(get_golden_dir(), "decoder_layer_golden.pt")
    torch.save(golden, golden_path)
    print(f"Saved decoder layer golden to {golden_path}")
    return golden


def main():
    """Generate all golden outputs."""
    print("Generating golden outputs for Qwen3-TTS reference implementation...")
    print(f"Output directory: {get_golden_dir()}")
    print()

    generate_rms_norm_golden()
    generate_mlp_golden()
    generate_attention_golden()
    generate_decoder_layer_golden()

    print()
    print("All golden outputs generated successfully!")


if __name__ == "__main__":
    main()
