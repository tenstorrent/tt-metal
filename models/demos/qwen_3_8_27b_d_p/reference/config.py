# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-27B (``model_type: qwen3_5``) dimension constants — the single source of truth for
everything the checkpoint's ``config.json`` already knows.

Nothing here is a bring-up choice: every field is read from the vendored
``configs/Qwen3.8-27B/config.json`` and asserted against it by
``tests/host/test_config_vs_json.py``. The bring-up *choices* (mesh, chunk, dtypes, PCC bars) live
in :mod:`spec` instead.

Architecture in one paragraph. Qwen3.5 is a **hybrid** decoder: ``layer_types`` alternates three
``linear_attention`` (Gated DeltaNet) layers and one ``full_attention`` (GQA) layer, 64 layers deep,
so 48 GDN + 16 GQA. Every layer carries a dense SiLU-SwiGLU MLP; there are no experts. The full
attention layers are GQA 24q/4kv at ``head_dim`` 256 with per-head QK-norm, an **output gate**
(``attn_output_gate``: ``q_proj`` emits 2x head_dim per head, the second half is a sigmoid gate on
the attention output), and **partial** RoPE over the first 64 of 256 dims. The published checkpoint
is a VL package; this bring-up is the text tower only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "Qwen3.8-27B" / "config.json"

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"


@lru_cache(maxsize=4)
def load_config_json(path: Path | str = CONFIG_PATH) -> dict:
    """The vendored ``config.json``'s ``text_config`` block (the text tower)."""
    with open(path) as f:
        return json.load(f)["text_config"]


@dataclass(frozen=True)
class Qwen35TextConfig:
    """Text-tower constants. Construct with :meth:`from_json`; never hand-edit the values."""

    # --- shared ---
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    hidden_act: str
    tie_word_embeddings: bool
    max_position_embeddings: int
    layer_types: tuple[str, ...]
    full_attention_interval: int

    # --- full attention (GQA) ---
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_bias: bool
    attn_output_gate: bool

    # --- rope (partial, mrope-interleaved) ---
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, int, int]
    mrope_interleaved: bool
    rope_type: str

    # --- linear attention (Gated DeltaNet) ---
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    output_gate_type: str

    # derived, never read from json
    rotary_dim: int = field(init=False)
    num_key_value_groups: int = field(init=False)
    gdn_key_dim: int = field(init=False)
    gdn_value_dim: int = field(init=False)
    gdn_conv_dim: int = field(init=False)
    gdn_num_value_groups: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotary_dim", int(self.head_dim * self.partial_rotary_factor))
        object.__setattr__(self, "num_key_value_groups", self.num_attention_heads // self.num_key_value_heads)
        object.__setattr__(self, "gdn_key_dim", self.linear_key_head_dim * self.linear_num_key_heads)
        object.__setattr__(self, "gdn_value_dim", self.linear_value_head_dim * self.linear_num_value_heads)
        object.__setattr__(self, "gdn_conv_dim", self.gdn_key_dim * 2 + self.gdn_value_dim)
        object.__setattr__(self, "gdn_num_value_groups", self.linear_num_value_heads // self.linear_num_key_heads)
        self._validate()

    def _validate(self) -> None:
        assert len(self.layer_types) == self.num_hidden_layers
        assert set(self.layer_types) <= {LINEAR_ATTENTION, FULL_ATTENTION}
        # The rotary half-width must be exactly what mrope's three sections cover: the HF rotary
        # module builds inv_freq of length rotary_dim/2 and indexes it with mrope_section.
        assert sum(self.mrope_section) * 2 == self.rotary_dim, (
            f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, "
            f"but rotary_dim/2 = {self.rotary_dim // 2}"
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.linear_num_value_heads % self.linear_num_key_heads == 0
        assert self.hidden_act == "silu", f"only SiLU SwiGLU is implemented, got {self.hidden_act}"
        # The layer schedule is a pure period: index % interval == interval-1 is full attention.
        for i, t in enumerate(self.layer_types):
            expected = FULL_ATTENTION if (i + 1) % self.full_attention_interval == 0 else LINEAR_ATTENTION
            assert t == expected, f"layer_types[{i}]={t} breaks the full_attention_interval period"

    @classmethod
    def from_json(cls, path: Path | str = CONFIG_PATH) -> "Qwen35TextConfig":
        c = load_config_json(path)
        rope = c["rope_parameters"]
        return cls(
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            vocab_size=c["vocab_size"],
            rms_norm_eps=c["rms_norm_eps"],
            hidden_act=c["hidden_act"],
            tie_word_embeddings=c["tie_word_embeddings"],
            max_position_embeddings=c["max_position_embeddings"],
            layer_types=tuple(c["layer_types"]),
            full_attention_interval=c["full_attention_interval"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            attention_bias=c["attention_bias"],
            attn_output_gate=c["attn_output_gate"],
            rope_theta=float(rope["rope_theta"]),
            partial_rotary_factor=rope["partial_rotary_factor"],
            mrope_section=tuple(rope["mrope_section"]),
            mrope_interleaved=rope["mrope_interleaved"],
            rope_type=rope["rope_type"],
            linear_num_key_heads=c["linear_num_key_heads"],
            linear_num_value_heads=c["linear_num_value_heads"],
            linear_key_head_dim=c["linear_key_head_dim"],
            linear_value_head_dim=c["linear_value_head_dim"],
            linear_conv_kernel_dim=c["linear_conv_kernel_dim"],
            output_gate_type=c["output_gate_type"],
        )

    # --- layer schedule helpers -------------------------------------------------------------
    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == FULL_ATTENTION

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == FULL_ATTENTION)

    @property
    def linear_attention_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == LINEAR_ATTENTION)

    def kv_slot(self, layer_idx: int) -> int:
        """Index of ``layer_idx`` among the full-attention layers — its row in the KV cache.

        Only the 16 GQA layers own K/V, so the cache packs 16 slots per user rather than 64. The
        GDN layers carry a conv + recurrent state instead (see ``tt/gdn/state.py``).
        """
        assert self.is_full_attention(layer_idx), f"layer {layer_idx} is {self.layer_types[layer_idx]}"
        return self.full_attention_layers.index(layer_idx)

    def reduced(self, num_hidden_layers: int) -> "Qwen35TextConfig":
        """A depth-reduced copy for host-side comparisons (recipe section 4: a diagnostic, never a grade)."""
        assert 0 < num_hidden_layers <= self.num_hidden_layers
        return Qwen35TextConfig(
            **{
                **{k: getattr(self, k) for k in self.__dataclass_fields__ if self.__dataclass_fields__[k].init},
                "num_hidden_layers": num_hidden_layers,
                "layer_types": self.layer_types[:num_hidden_layers],
            }
        )
