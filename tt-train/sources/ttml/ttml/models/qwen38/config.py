# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 text-backbone configuration.

Qwen3.8-27B is a hybrid stack: ``full_attention_interval`` is 4, so the 64
layers run as 16 repeats of ``3 x linear_attention (Gated DeltaNet) + 1 x
full_attention (Gated Attention)`` -- 48 DeltaNet layers and 16 attention
layers.  Every layer is followed by a SwiGLU MLP.

This covers the text backbone only: the vision tower and the MTP head in the
checkpoint are not part of the LoRA training graph and are skipped by the
loader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Qwen38Config:
    """Text-config fields the tt-train Qwen3.8 model needs.

    Defaults are Qwen3.8-27B (``text_config`` of the HF checkpoint).
    """

    # --- shared ---
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False

    # --- hybrid layout ---
    # Layer i is full attention iff (i + 1) % full_attention_interval == 0,
    # which reproduces the checkpoint's layer_types list exactly.
    full_attention_interval: int = 4

    # --- full attention (Gated Attention) layers ---
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    attention_bias: bool = False
    # q_proj is 2x wide; the second half is the output gate.
    attn_output_gate: bool = True
    # Only the first 25% of head_dim is rotated (64 of 256 dims).
    partial_rotary_factor: float = 0.25
    rope_theta: float = 1e7

    # --- linear attention (Gated DeltaNet) layers ---
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # Tokens per chunk in the chunked delta rule. Not a checkpoint field: it is
    # a purely numerical tiling choice, and must be a power of two dividing the
    # sequence length (see delta_rule.wy_inverse).
    delta_chunk_size: int = 64

    # --- parallelism ---
    # Megatron tensor parallelism. The TP width comes from the mesh axis named
    # below, so it is not duplicated here. Note TP is limited to 4 by the 4 KV
    # heads of the attention layers (see models.qwen38.parallel).
    use_tp: bool = False
    tp_axis_name: str = "tp"

    # Recompute the Gated DeltaNet mixer in the backward pass instead of keeping
    # its activations. The mixer is ~80% of a DeltaNet layer's activation
    # memory and DeltaNet is 48 of 64 layers, so this is what makes
    # seq_len=1024 fit; see models.qwen38.checkpoint.
    recompute_deltanet: bool = False

    # --- training ---
    dropout_prob: float = 0.0
    weight_decay: float = 0.0

    # Populated from the checkpoint when available; otherwise derived from
    # full_attention_interval.
    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if self.is_full_attention(i) else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]

    # --- derived sizes -----------------------------------------------------

    def is_full_attention(self, layer_idx: int) -> bool:
        if self.layer_types:
            return self.layer_types[layer_idx] == "full_attention"
        return (layer_idx + 1) % self.full_attention_interval == 0

    @property
    def rotary_dim(self) -> int:
        """Rotated width of each attention head (64 for Qwen3.8)."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def gva_repeats(self) -> int:
        """DeltaNet value heads per key head (3 for Qwen3.8)."""
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError(
                f"linear_num_value_heads ({self.linear_num_value_heads}) must be a multiple of "
                f"linear_num_key_heads ({self.linear_num_key_heads})"
            )
        return self.linear_num_value_heads // self.linear_num_key_heads

    @property
    def key_proj_dim(self) -> int:
        """Width of the Q (and K) slice of the fused DeltaNet in_proj_qkv."""
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_proj_dim(self) -> int:
        """Width of the V slice of in_proj_qkv, and of in_proj_z / out_proj input."""
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def qkv_proj_dim(self) -> int:
        """Total width of the fused DeltaNet in_proj_qkv (10240 for Qwen3.8)."""
        return 2 * self.key_proj_dim + self.value_proj_dim

    @classmethod
    def from_hf_json(cls, path: str | Path) -> "Qwen38Config":
        """Read an HF ``config.json``, taking ``text_config`` if present.

        Only the fields above are consumed; vision and MTP entries are ignored.
        """
        raw = json.loads(Path(path).read_text())
        text = raw.get("text_config", raw)

        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in text.items() if k in known}

        # partial_rotary_factor and rope_theta live under rope_parameters in
        # Qwen3.8 (and at the top level in older Qwen configs).
        rope = text.get("rope_parameters", {})
        if "partial_rotary_factor" in rope:
            kwargs["partial_rotary_factor"] = rope["partial_rotary_factor"]
        if "rope_theta" in rope:
            kwargs["rope_theta"] = rope["rope_theta"]

        return cls(**kwargs)
