# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Config for the Qwen3.6-27B DFlash *drafter* (``z-lab/Qwen3.6-27B-DFlash``).

Every field is **required** and read from the checkpoint's own ``config.json``. That is
deliberate: the equivalent config in ``models/demos/deepseek_v3_d_p/tt/dflash_prefill/``
carries Kimi-K2.6 dims as dataclass defaults and warns in its docstring not to reuse them,
which only works if a reader notices the warning. Here a missing key is a ``TypeError`` at
construction rather than a silently wrong dim, so there is nothing to notice.

Reading is done straight from ``config.json`` with ``json`` rather than through
``AutoConfig``: the checkpoint's ``auto_map`` names ``dflash.DFlashDraftModel``, but the HF
repo ships no ``dflash.py``, so ``trust_remote_code=True`` has nothing to fetch. Going
direct also keeps this module importable without ``transformers``, which matters because
the repo's ``python_env`` pins 5.12.1 while the upstream ``dflash`` package wants 5.15.0.

Where a value can appear either under ``dflash_config`` or at top level, resolution order
mirrors the reference's ``_draft_value`` exactly (``dflash_config`` wins) so this config and
``reference/dflash/dflash.py`` can never disagree about the same checkpoint.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

SLIDING = "sliding_attention"
FULL = "full_attention"
_LAYER_TYPES = frozenset({SLIDING, FULL})

#: Context lengths the drafter pads to before calling into ttnn. Each distinct context
#: length compiles its own program, and the Muse-Glimmer port measured that cost as the
#: drafter's dominant expense -- 1201.7 ms/call at real varying shapes vs 14.3 ms at one
#: constant shape, and 671 -> 120 ms/call once bucketed, with acceptance unchanged
#: (work_log F6/F7). Buckets are powers of two up to the sliding window; beyond it only the
#: single full-attention layer still sees the whole context.
CONTEXT_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)


def _draft_value(cfg: dict, name: str, default=None):
    """``dflash_config[name]`` if present, else top-level ``cfg[name]``, else ``default``.

    Mirrors ``_draft_value`` in ``reference/dflash/dflash.py``.
    """
    dfc = cfg.get("dflash_config") or {}
    if name in dfc:
        return dfc[name]
    return cfg.get(name, default)


@dataclass(frozen=True)
class DFlashDrafterConfig:
    """Shape/numerics contract for the DFlash drafter, as declared by its checkpoint."""

    hidden_size: int
    intermediate_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    #: Per-layer attention flavour, ``num_hidden_layers`` long. For Qwen3.6-27B-DFlash this
    #: is 4x ``sliding_attention`` then 1x ``full_attention``.
    layer_types: tuple[str, ...]
    sliding_window: int | None
    rope_theta: float
    #: Speculative block width: 1 anchor + ``block_size - 1`` drafted tokens.
    block_size: int
    #: Absorbing state the drafter denoises away. Every non-anchor slot starts here.
    mask_token_id: int
    #: Target residual-stream taps (0-indexed layer *outputs*) concatenated -- in this
    #: order -- into the ``fc`` context feature. Order is load-bearing.
    target_layer_ids: tuple[int, ...]
    #: The drafter's own declaration of which target depth it was trained against. Exists
    #: to be cross-checked against the loaded target via :meth:`assert_matches_target`.
    num_target_layers: int
    #: Top-level ``is_causal`` from the checkpoint, overriding the per-layer default when
    #: present. Absent (``None``) for Qwen3.6-27B-DFlash, so the default applies -- but the
    #: Kimi-era drafter sets it ``False``, which is why the older vendored reference
    #: hard-codes non-causal on every layer. Honouring the override keeps this config
    #: faithful to ``reference/dflash/dflash.py:363-364`` for *any* DFlash checkpoint.
    is_causal_override: bool | None = None
    #: Only used to synthesise random weights in tests.
    initializer_range: float = 0.02
    #: Absent from this checkpoint (present on the Gemma4 drafter). Kept so
    #: :meth:`from_json` never silently drops it if a future checkpoint adds one.
    final_logit_softcapping: float | None = None

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, expected num_hidden_layers={self.num_hidden_layers}"
            )
        if bad := set(self.layer_types) - _LAYER_TYPES:
            raise ValueError(f"unknown layer_types {sorted(bad)}; expected a subset of {sorted(_LAYER_TYPES)}")
        if SLIDING in self.layer_types and not self.sliding_window:
            # Deliberate, documented divergence from the reference. HF would leave such a
            # layer causal-but-unwindowed and run happily; we refuse. "Sliding layers that
            # quietly lost their window" is the exact shape of Muse-Glimmer work_log F3b,
            # where an unwindowed implementation scored 0.99997 against a golden that had
            # itself lost its window while the correct one scored 0.9294 -- a config that
            # cannot express the mistake is worth more than bug-compatibility here. Cannot
            # trigger for Qwen3.6-27B-DFlash, which sets use_sliding_window=true.
            raise ValueError(
                "checkpoint declares sliding_attention layers but no sliding_window (use_sliding_window false?)"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} is not a multiple of "
                f"num_key_value_heads {self.num_key_value_heads}"
            )
        if not self.target_layer_ids:
            raise ValueError("target_layer_ids is empty; the drafter cannot be conditioned")
        # Note: len(target_layer_ids) need NOT equal num_hidden_layers. It does for this
        # checkpoint (5 and 5) and for Kimi (6 and 6), but the Gemma4 drafter has 5 layers
        # and 6 taps, so `fc`'s input width comes from the tap count alone.
        if bad_ids := [t for t in self.target_layer_ids if not 0 <= t < self.num_target_layers]:
            raise ValueError(f"target_layer_ids {bad_ids} outside the target's {self.num_target_layers} layers")
        if self.block_size < 2:
            raise ValueError(f"block_size {self.block_size} leaves no room to draft (need anchor + >=1 slot)")

    # ---- derived shapes ---------------------------------------------------------------

    @property
    def q_dim(self) -> int:
        """Width of ``q_proj``'s output: 32 * 128 = 4096."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Width of ``k_proj``/``v_proj``'s output: 8 * 128 = 1024."""
        return self.num_key_value_heads * self.head_dim

    @property
    def num_key_value_groups(self) -> int:
        """Query heads per KV head (GQA ratio): 4."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def target_feature_size(self) -> int:
        """``fc``'s input width: 5 taps * 5120 = 25600."""
        return len(self.target_layer_ids) * self.hidden_size

    @property
    def num_draft_tokens(self) -> int:
        """Tokens actually proposed per block: ``block_size - 1`` = 15. Slot 0 is the
        anchor (the target's last committed token) and is not itself a proposal."""
        return self.block_size - 1

    # ---- per-layer mask contract ------------------------------------------------------
    #
    # This is the part it is easiest to get wrong, so it lives here rather than being
    # re-derived at each call site. It reproduces Qwen3DFlashAttention.__init__ in
    # reference/dflash/dflash.py:361-366 -- including the consequence that a
    # full_attention layer is NON-causal and unwindowed, i.e. fully bidirectional over
    # context + block. For this checkpoint that single layer is where all of the "block
    # diffusion" bidirectionality lives. Pinned by tests/dflash/test_mask.py.

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == SLIDING

    def is_causal(self, layer_idx: int) -> bool:
        if self.is_causal_override is not None:
            return self.is_causal_override
        return self.is_sliding(layer_idx)

    def window_for(self, layer_idx: int) -> int | None:
        return self.sliding_window if self.is_sliding(layer_idx) else None

    # ---- construction -----------------------------------------------------------------

    def assert_matches_target(self, target_num_hidden_layers: int) -> None:
        """Reject a drafter/target mismatch before any weight is loaded.

        A drafter conditioned on taps from a 64-layer target produces plausible-looking
        garbage against a target of a different depth, so this is checked eagerly.
        """
        if self.num_target_layers != target_num_hidden_layers:
            raise ValueError(
                f"drafter was trained against a {self.num_target_layers}-layer target but the "
                f"loaded target has {target_num_hidden_layers} layers"
            )

    @classmethod
    def from_dict(cls, cfg: dict) -> "DFlashDrafterConfig":
        """Build from a parsed ``config.json``."""
        hidden_size = cfg["hidden_size"]
        num_attention_heads = cfg["num_attention_heads"]
        # `use_sliding_window` gates the window, and the gate must be applied HERE even
        # though `reference/dflash/dflash.py:365` reads a bare `config.sliding_window`.
        # The reference consumes an HF `Qwen3Config`, whose `__post_init__` has already
        # run `self.sliding_window = self.sliding_window if self.use_sliding_window else
        # None` (transformers/models/qwen3/configuration_qwen3.py:87). So the flag is
        # honoured one level up, and skipping it here would make this config disagree
        # with the reference for any checkpoint that sets it False.
        #
        # `max_window_layers` is genuinely inert: Qwen3Config only derives `layer_types`
        # from it when `layer_types` is absent (ibid.:91), and this checkpoint ships an
        # explicit `layer_types`.
        sliding = cfg.get("sliding_window")
        return cls(
            hidden_size=hidden_size,
            intermediate_size=cfg["intermediate_size"],
            head_dim=cfg.get("head_dim", hidden_size // num_attention_heads),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=cfg["num_key_value_heads"],
            num_hidden_layers=cfg["num_hidden_layers"],
            rms_norm_eps=cfg["rms_norm_eps"],
            layer_types=tuple(cfg["layer_types"]),
            sliding_window=int(sliding) if sliding and cfg.get("use_sliding_window", False) else None,
            # transformers 5.x normalises a top-level `rope_theta` into
            # `rope_parameters={"rope_theta": ..., "rope_type": "default"}` and REMOVES the
            # top-level attribute, so a config object round-tripped through Qwen3Config has
            # it only in the nested dict. Raw checkpoint JSON has it at top level. Accept
            # both -- reading only one silently falls back to RoPE base 10000 instead of
            # 1e7, which is the same garbage-RoPE failure mode as work_log F3.
            rope_theta=float((cfg.get("rope_parameters") or {}).get("rope_theta") or cfg["rope_theta"]),
            block_size=int(_draft_value(cfg, "block_size")),
            mask_token_id=int(_draft_value(cfg, "mask_token_id")),
            target_layer_ids=tuple(_draft_value(cfg, "target_layer_ids")),
            num_target_layers=int(cfg["num_target_layers"]),
            is_causal_override=(bool(v) if (v := cfg.get("is_causal")) is not None else None),
            initializer_range=float(cfg.get("initializer_range", 0.02)),
            final_logit_softcapping=(
                float(v) if (v := _draft_value(cfg, "final_logit_softcapping")) is not None else None
            ),
        )

    @classmethod
    def from_hf_config(cls, c) -> "DFlashDrafterConfig":
        """Build from an already-loaded HF config object."""
        return cls.from_dict({k: v for k, v in vars(c).items() if not k.startswith("_")})

    @classmethod
    def from_pretrained(cls, path: str | None = None) -> "DFlashDrafterConfig":
        """Build from a local checkpoint dir or a Hub id.

        Defaults to ``$DFLASH_HF_MODEL``, then to the published checkpoint.
        """
        path = path or os.environ.get("DFLASH_HF_MODEL") or "z-lab/Qwen3.6-27B-DFlash"
        local = os.path.join(path, "config.json")
        if not os.path.isfile(local):
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(
                repo_id=path,
                filename="config.json",
                local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
            )
        with open(local) as f:
            return cls.from_dict(json.load(f))
