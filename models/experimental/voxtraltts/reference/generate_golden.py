# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic golden outputs for Voxtral TTS functional references."""

from pathlib import Path

import torch

from models.experimental.voxtraltts.reference.functional import (
    acoustic_attention,
    acoustic_transformer_layer,
    compute_rope_frequencies,
    get_default_acoustic_config,
    get_default_text_config,
    rms_norm,
    swiglu_mlp,
    text_attention,
    text_decoder_layer,
)


def get_golden_dir() -> Path:
    golden_dir = Path(__file__).resolve().parent / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    return golden_dir


def generate_rms_norm_golden() -> dict:
    torch.manual_seed(0)
    config = get_default_text_config()
    hidden_states = torch.randn(2, 8, config.hidden_size)
    weight = torch.randn(config.hidden_size)
    output = rms_norm(hidden_states, weight, config.rms_norm_eps)
    golden = {"input": hidden_states, "weight": weight, "eps": config.rms_norm_eps, "output": output}
    torch.save(golden, get_golden_dir() / "rms_norm_golden.pt")
    return golden


def generate_swiglu_mlp_golden() -> dict:
    torch.manual_seed(0)
    config = get_default_text_config()
    hidden_states = torch.randn(2, 8, config.hidden_size)
    w1 = torch.randn(config.intermediate_size, config.hidden_size)
    w2 = torch.randn(config.hidden_size, config.intermediate_size)
    w3 = torch.randn(config.intermediate_size, config.hidden_size)
    output = swiglu_mlp(hidden_states, w1, w2, w3)
    golden = {"input": hidden_states, "w1": w1, "w2": w2, "w3": w3, "output": output}
    torch.save(golden, get_golden_dir() / "swiglu_mlp_golden.pt")
    return golden


def generate_text_attention_golden() -> dict:
    torch.manual_seed(0)
    config = get_default_text_config()
    batch, seq_len = 2, 8
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    weights = {
        "attention.wq.weight": torch.randn(config.num_attention_heads * config.head_dim, config.hidden_size),
        "attention.wk.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        "attention.wv.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        "attention.wo.weight": torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim),
    }
    cos, sin = compute_rope_frequencies(config.head_dim, seq_len, config.rope_theta)
    output = text_attention(hidden_states, weights, cos, sin, config)
    golden = {"input": hidden_states, "weights": weights, "cos": cos, "sin": sin, "output": output}
    torch.save(golden, get_golden_dir() / "text_attention_golden.pt")
    return golden


def generate_text_decoder_layer_golden() -> dict:
    torch.manual_seed(0)
    config = get_default_text_config()
    batch, seq_len = 2, 8
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    weights = {
        "attention_norm.weight": torch.randn(config.hidden_size),
        "attention.wq.weight": torch.randn(config.num_attention_heads * config.head_dim, config.hidden_size),
        "attention.wk.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        "attention.wv.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        "attention.wo.weight": torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim),
        "ffn_norm.weight": torch.randn(config.hidden_size),
        "feed_forward.w1.weight": torch.randn(config.intermediate_size, config.hidden_size),
        "feed_forward.w2.weight": torch.randn(config.hidden_size, config.intermediate_size),
        "feed_forward.w3.weight": torch.randn(config.intermediate_size, config.hidden_size),
    }
    cos, sin = compute_rope_frequencies(config.head_dim, seq_len, config.rope_theta)
    output = text_decoder_layer(hidden_states, weights, cos, sin, config)
    golden = {"input": hidden_states, "weights": weights, "cos": cos, "sin": sin, "output": output}
    torch.save(golden, get_golden_dir() / "text_decoder_layer_golden.pt")
    return golden


def generate_acoustic_attention_golden() -> dict:
    torch.manual_seed(0)
    config = get_default_acoustic_config()
    hidden_states = torch.randn(2, 3, config.hidden_size)
    weights = {
        "attention.wq.weight": torch.randn(config.num_attention_heads * config.head_dim, config.hidden_size),
        "attention.wk.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        "attention.wv.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        "attention.wo.weight": torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim),
    }
    output = acoustic_attention(hidden_states, weights, config)
    golden = {"input": hidden_states, "weights": weights, "output": output}
    torch.save(golden, get_golden_dir() / "acoustic_attention_golden.pt")
    return golden


def generate_acoustic_layer_golden() -> dict:
    torch.manual_seed(0)
    config = get_default_acoustic_config()
    hidden_states = torch.randn(2, 3, config.hidden_size)
    weights = {
        "attention_norm.weight": torch.randn(config.hidden_size),
        "attention.wq.weight": torch.randn(config.num_attention_heads * config.head_dim, config.hidden_size),
        "attention.wk.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        "attention.wv.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        "attention.wo.weight": torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim),
        "ffn_norm.weight": torch.randn(config.hidden_size),
        "feed_forward.w1.weight": torch.randn(config.intermediate_size, config.hidden_size),
        "feed_forward.w2.weight": torch.randn(config.hidden_size, config.intermediate_size),
        "feed_forward.w3.weight": torch.randn(config.intermediate_size, config.hidden_size),
    }
    output = acoustic_transformer_layer(hidden_states, weights, config)
    golden = {"input": hidden_states, "weights": weights, "output": output}
    torch.save(golden, get_golden_dir() / "acoustic_layer_golden.pt")
    return golden


def main() -> None:
    print(f"Generating Voxtral TTS golden outputs in {get_golden_dir()}")
    generate_rms_norm_golden()
    generate_swiglu_mlp_golden()
    generate_text_attention_golden()
    generate_text_decoder_layer_golden()
    generate_acoustic_attention_golden()
    generate_acoustic_layer_golden()
    print("Done")


if __name__ == "__main__":
    main()
