# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Config for the Qwen3.6-27B DFlash drafter, and the checkpoint plumbing both backends share.

Values come from ``z-lab/Qwen3.6-27B-DFlash``'s ``config.json`` (architecture ``DFlashDraftModel``,
a Qwen3-style GQA model). Every default here is that checkpoint's own — do NOT reuse for another
drafter without re-deriving from its ``config.json``.

This lives under ``tt/`` because the device port is the consumer that cares about the derived TP
dimensions; ``reference/dflash/loader.py`` re-exports it so the host path keeps one definition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from transformers import AutoConfig

TARGET_ENV = "HF_MODEL"
DRAFTER_ENV = "DFLASH_HF_MODEL"
DEFAULT_TARGET = "Qwen/Qwen3.6-27B"
DEFAULT_DRAFTER = "z-lab/Qwen3.6-27B-DFlash"


@dataclass(frozen=True)
class DFlashDrafterConfig:
    """The drafter numbers a port needs, read off the checkpoint's ``config.json``.

    Defaults are ``z-lab/Qwen3.6-27B-DFlash``'s own values and exist only so the shape is readable
    here; :meth:`from_hf_config` overwrites every one of them from the actual checkpoint.
    """

    hidden_size: int = 5120
    head_dim: int = 128
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA, 4 query heads per KV head
    num_hidden_layers: int = 5
    intermediate_size: int = 17408
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    # 4 causal sliding layers then 1 bidirectional full-attention layer.
    layer_types: tuple[str, ...] = ("sliding_attention",) * 4 + ("full_attention",)
    sliding_window: int = 2048
    rope_theta: float = 1e7
    # Speculation knobs: a 16-slot block is 1 anchor + 15 drafted tokens.
    block_size: int = 16
    mask_token_id: int = 248070
    # Target residual-stream taps (0-indexed layer OUTPUTS), concatenated in this order into `fc`.
    target_layer_ids: tuple[int, ...] = (1, 16, 31, 46, 61)
    # The drafter's own declaration of which target it attaches to. None when the checkpoint omits
    # it, which must fail the cross-check rather than silently pass — never default this to 64.
    num_target_layers: int | None = None

    @property
    def num_speculative_tokens(self) -> int:
        return self.block_size - 1

    @property
    def target_feature_size(self) -> int:
        """Width of the concatenated tap feature that ``fc`` consumes: 5 * 5120 = 25600."""
        return len(self.target_layer_ids) * self.hidden_size

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "sliding_attention"

    # ---- the one TP-derived dimension the drafter needs ------------------------------------

    def local_hidden(self, tp: int) -> int:
        """Hidden columns per device in the TARGET's fractured residual — i.e. a tap's width.

        This is the only TP-derived dimension the drafter needs: it runs replicated (see
        :mod:`.weights`), but its ``fc`` input arrives fractured because it *is* the target's
        residual stream.
        """
        assert self.hidden_size % tp == 0, f"hidden {self.hidden_size} not divisible by tp {tp}"
        return self.hidden_size // tp

    @classmethod
    def from_hf_config(cls, c) -> "DFlashDrafterConfig":
        d = cls()
        dfc = dict(getattr(c, "dflash_config", None) or {})
        return cls(
            hidden_size=c.hidden_size,
            head_dim=getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=c.num_key_value_heads,
            num_hidden_layers=c.num_hidden_layers,
            intermediate_size=c.intermediate_size,
            vocab_size=c.vocab_size,
            rms_norm_eps=c.rms_norm_eps,
            layer_types=tuple(getattr(c, "layer_types", None) or d.layer_types),
            sliding_window=int(getattr(c, "sliding_window", None) or d.sliding_window),
            rope_theta=float(getattr(c, "rope_theta", None) or d.rope_theta),
            block_size=int(dfc.get("block_size", getattr(c, "block_size", d.block_size))),
            mask_token_id=int(dfc.get("mask_token_id", getattr(c, "mask_token_id", d.mask_token_id))),
            target_layer_ids=tuple(dfc.get("target_layer_ids", d.target_layer_ids)),
            num_target_layers=(int(v) if (v := getattr(c, "num_target_layers", None)) is not None else None),
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "DFlashDrafterConfig":
        return cls.from_hf_config(AutoConfig.from_pretrained(path, trust_remote_code=True))


def _resolve(env: str, default: str, *, offline: bool = False) -> str:
    """Return a local checkpoint dir for ``$env``, downloading the snapshot if it names a hub id."""
    ref = os.environ.setdefault(env, default)
    if os.path.isdir(ref):
        return ref
    from huggingface_hub import snapshot_download

    path = snapshot_download(ref, local_files_only=offline)
    os.environ[env] = path
    return path


def resolve_drafter_path(*, offline: bool = False) -> str:
    return _resolve(DRAFTER_ENV, DEFAULT_DRAFTER, offline=offline)


def resolve_target_path(*, offline: bool = False) -> str:
    return _resolve(TARGET_ENV, DEFAULT_TARGET, offline=offline)


def load_drafter_state_dict(path: str | None = None) -> dict:
    """Load the drafter's 58 tensors from ``$DFLASH_HF_MODEL/model.safetensors``."""
    from safetensors.torch import load_file

    path = path or resolve_drafter_path()
    st = os.path.join(path, "model.safetensors")
    if not os.path.exists(st):
        raise FileNotFoundError(f"drafter weights not found: {st} (set {DRAFTER_ENV})")
    return load_file(st)
