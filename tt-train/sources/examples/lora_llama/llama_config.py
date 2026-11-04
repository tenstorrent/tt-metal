# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the Llama config for both LoRA scripts."""

from __future__ import annotations

from pathlib import Path

from ttml.common.utils import round_up_to_tile
from ttml.models import EmbeddingPlacement, RunnerType, WeightTyingType
from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig

PRETRAINED_HELP = (
    "HuggingFace repo ID (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0) or local directory holding its "
    ".safetensors files, config.json and tokenizer files. Loads the HF tokenizer and the pretrained weights; the "
    "architecture comes from config.json, so -m then only sets max_sequence_length, dropout_prob "
    "and runner_type."
)

_DEFAULT_ARCHITECTURE = dict(
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=6,
    num_key_value_heads=3,
    rope_theta=500000.0,
)


def fetch_pretrained(src: str) -> str:
    """Return a local directory for *src*: the path itself, or a download if *src* is a HuggingFace repo ID."""
    if Path(src).is_dir():
        return src
    from huggingface_hub import snapshot_download

    print(f"Downloading weights from HuggingFace: {src}")
    path = snapshot_download(src, allow_patterns=["*.safetensors", "config.json"])
    print(f"Weights downloaded to: {path}")
    return path


def llama_config(yaml_config: dict | None, pretrained_path: str | None, num_tokens: int, use_tp: bool) -> LlamaConfig:
    """Return the config of the model to fine-tune."""
    tc = (yaml_config or {}).get("transformer_config", {})
    run = dict(
        max_position_embeddings=tc.get("max_sequence_length", 256),
        attention_dropout=tc.get("dropout_prob", 0.0),
        mlp_dropout=tc.get("dropout_prob", 0.0),
        runner_type=RunnerType.from_string(tc["runner_type"]) if "runner_type" in tc else RunnerType.Default,
        use_tp=use_tp,
        embedding_placement=EmbeddingPlacement.VocabParallel if use_tp else EmbeddingPlacement.Replicated,
    )

    if pretrained_path is not None:
        config = LlamaConfig.from_hf(pretrained_path, **run)
        if num_tokens > config.vocab_size:
            raise ValueError(f"the tokenizer has {num_tokens} tokens but the checkpoint embeds {config.vocab_size}")
        return config

    architecture = _DEFAULT_ARCHITECTURE if yaml_config is None else _architecture_from_yaml(tc)
    # Pad the character vocab to a tile.
    return LlamaConfig(vocab_size=round_up_to_tile(num_tokens), **architecture, **run)


def _architecture_from_yaml(tc: dict) -> dict:
    return dict(
        hidden_size=tc.get("embedding_dim", 384),
        intermediate_size=tc.get("intermediate_dim", None),
        num_hidden_layers=tc.get("num_blocks", 6),
        num_attention_heads=tc.get("num_heads", 6),
        num_key_value_heads=tc.get("num_groups", 3),
        rope_theta=tc.get("theta", 10000.0),
        weight_tying=(
            WeightTyingType.from_string(tc["weight_tying"]) if "weight_tying" in tc else WeightTyingType.Disabled
        ),
        rope_scaling=LlamaRopeScalingConfig(**tc.get("rope_scaling", {})),
    )
