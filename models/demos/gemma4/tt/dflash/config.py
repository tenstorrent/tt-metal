# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Config for the Gemma4-31B DFlash *drafter* (z-lab/gemma-4-31B-it-DFlash).

Values are from the checkpoint's own config.json (architecture DFlashDraftModel,
model_type "qwen3" -- the drafter's own 5 decoder layers reuse Qwen3-style GQA
blocks regardless of the target's own architecture; see
models/demos/gemma4/docs/dflash_design.md). Every default here is specific to
THIS checkpoint -- do not reuse for another drafter without re-deriving from its
own config.json (mirrors the same warning on
models/demos/deepseek_v3_d_p/tt/dflash_prefill/dflash_drafter_config.py's
DFlashDrafterConfig for the Kimi-K2.6-DFlash checkpoint).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gemma4DFlashDrafterConfig:
    hidden_size: int = 5376
    head_dim: int = 128
    num_attention_heads: int = 64
    num_key_value_heads: int = 8  # GQA
    num_hidden_layers: int = 5  # draft layers
    intermediate_size: int = 10752
    rms_norm_eps: float = 1e-6
    block_size: int = 16  # speculative block (decode-time)
    mask_token_id: int = 4
    # Residual-stream taps of the 60-layer Gemma4-31B verifier (0-indexed layer
    # OUTPUTS) concatenated (in this order) into the FC context feature.
    target_layer_ids: tuple[int, ...] = (1, 12, 23, 35, 46, 57)
    num_target_layers: int = 60
    # Per-layer attention pattern (sliding-window vs full/global), matching
    # Gemma4's own local/global alternation style -- see dflash_design.md section 1.
    layer_types: tuple[str, ...] = ("sliding_attention",) * 4 + ("full_attention",)
    sliding_window: int = 2048
    rope_theta: float = 1000000.0
    final_logit_softcapping: float = 30.0  # Gemma-specific; absent from Kimi's checkpoint
    vocab_size: int = 262144
    tie_word_embeddings: bool = True  # no dedicated embed_tokens/lm_head; shares the target's

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim  # 8 * 128 = 1024

    @property
    def target_feature_size(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size  # 6 * 5376 = 32256

    @property
    def layer_configs(self) -> tuple[tuple[bool, int | None], ...]:
        """Per-layer (is_causal, sliding_window) for dflash_drafter_forward, derived from
        layer_types/sliding_window -- "sliding_attention" layers are causal with the
        configured window, "full_attention" layers are non-causal/bidirectional with no
        window. Confirmed against the real checkpoint's own module attributes:
        4x (True, 2048) + 1x (False, None)."""
        return tuple(
            (True, self.sliding_window) if lt == "sliding_attention" else (False, None) for lt in self.layer_types
        )

    @classmethod
    def from_pretrained(cls, path: str = "z-lab/gemma-4-31B-it-DFlash") -> "Gemma4DFlashDrafterConfig":
        """Build the device drafter config from the checkpoint's config.json."""
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(path, trust_remote_code=True)
        dfc = dict(getattr(c, "dflash_config", None) or {})
        d = cls()
        # Newer transformers versions nest rope_theta under `rope_parameters` instead of
        # exposing it as a top-level config attribute -- check both, since getattr's
        # default would otherwise silently mask a real mismatch against the checkpoint.
        rope_params = dict(getattr(c, "rope_parameters", None) or {})
        rope_theta = rope_params.get("rope_theta", getattr(c, "rope_theta", d.rope_theta))
        return cls(
            hidden_size=c.hidden_size,
            head_dim=getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=c.num_key_value_heads,
            num_hidden_layers=c.num_hidden_layers,
            intermediate_size=c.intermediate_size,
            rms_norm_eps=c.rms_norm_eps,
            block_size=int(dfc.get("block_size", d.block_size)),
            mask_token_id=int(dfc.get("mask_token_id", d.mask_token_id)),
            target_layer_ids=tuple(dfc.get("target_layer_ids", d.target_layer_ids)),
            num_target_layers=int(getattr(c, "num_target_layers", d.num_target_layers)),
            layer_types=tuple(getattr(c, "layer_types", d.layer_types)),
            sliding_window=int(getattr(c, "sliding_window", d.sliding_window)),
            rope_theta=float(rope_theta),
            final_logit_softcapping=float(getattr(c, "final_logit_softcapping", d.final_logit_softcapping)),
            vocab_size=int(getattr(c, "vocab_size", d.vocab_size)),
            tie_word_embeddings=bool(getattr(c, "tie_word_embeddings", d.tie_word_embeddings)),
        )
