# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Config for the Kimi-K2.6-DFlash *drafter*.

Values are from the HF checkout ``Kimi-K2.6-DFlash/config.json`` (architecture
``DFlashDraftModel``, a Qwen3-style GQA model). Every default here is Kimi-K2.6-DFlash-specific
(dims, target-layer taps, rope) — do NOT reuse for another drafter without re-deriving from its
own ``config.json``. See issue #49586.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DFlashDrafterConfig:
    hidden_size: int = 7168
    head_dim: int = 128
    num_attention_heads: int = 64
    num_key_value_heads: int = 8  # GQA
    num_hidden_layers: int = 6  # draft layers
    # Verifier layers this drafter attaches to. None = the checkpoint config did not declare it: this field
    # exists ONLY to be cross-checked against the loaded verifier, so it must never default to a value that
    # would satisfy that check (61 is Kimi-K2.x's own depth, i.e. exactly the answer being verified). None is
    # falsy, so the build guard (`assert num_target_layers`) rejects a checkpoint that omits the key.
    num_target_layers: int | None = None
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02  # std for random-weight tests (config.json initializer_range)
    block_size: int = 8  # speculative block (decode-time)
    context_len: int = 4096  # spec-decode KV window
    mask_token_id: int = 163838
    # residual-stream taps of the 61-layer verifier (0-indexed layer OUTPUTS) whose hiddens
    # are concatenated (in this order) into the FC context feature.
    target_layer_ids: tuple[int, ...] = (1, 12, 24, 35, 47, 58)
    # deepseek_yarn rope (config.json rope_parameters) — identical params to the Kimi target,
    # but applied to the FULL head_dim (128) in Qwen3 half-split style, not the MLA 64-dim pe.
    rope_theta: float = 50000.0
    rope_factor: float = 64.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_orig_max_pos: int = 4096
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_convention: str = "interleaved"

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim  # 8 * 128 = 1024

    @property
    def target_feature_size(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size  # 6 * 7168 = 43008

    @classmethod
    def from_hf_config(cls, c) -> "DFlashDrafterConfig":
        """Build the device drafter config from an already-loaded HF config object."""
        rs = dict(getattr(c, "rope_scaling", None) or getattr(c, "rope_parameters", None) or {})
        dfc = dict(getattr(c, "dflash_config", None) or {})
        d = cls()  # defaults fill anything the checkpoint config omits
        return cls(
            hidden_size=c.hidden_size,
            head_dim=getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=c.num_key_value_heads,
            num_hidden_layers=c.num_hidden_layers,
            # Top-level key in the drafter's config.json (61 for Kimi-K2.x), NOT under dflash_config. It is
            # the drafter's own declaration of which verifier it attaches to, so the runtime cross-checks it
            # against the verifier's num_hidden_layers before building. Absent -> None (the fail-closed default).
            num_target_layers=(int(v) if (v := getattr(c, "num_target_layers", None)) is not None else None),
            rms_norm_eps=c.rms_norm_eps,
            target_layer_ids=tuple(dfc.get("target_layer_ids", d.target_layer_ids)),
            rope_theta=float(rs.get("rope_theta") or getattr(c, "rope_theta", None) or d.rope_theta),
            rope_factor=float(rs.get("factor", d.rope_factor)),
            rope_beta_fast=float(rs.get("beta_fast", d.rope_beta_fast)),
            rope_beta_slow=float(rs.get("beta_slow", d.rope_beta_slow)),
            rope_orig_max_pos=int(rs.get("original_max_position_embeddings", d.rope_orig_max_pos)),
            rope_mscale=float(rs.get("mscale", d.rope_mscale)),
            rope_mscale_all_dim=float(rs.get("mscale_all_dim", d.rope_mscale_all_dim)),
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "DFlashDrafterConfig":
        """Build the device drafter config from a HF checkpoint dir (``$DFLASH_HF_MODEL``): reads
        ``config.json`` via ``AutoConfig`` and extracts dims + rope params through :meth:`from_hf_config`."""
        from transformers import AutoConfig

        return cls.from_hf_config(AutoConfig.from_pretrained(path, trust_remote_code=True))
