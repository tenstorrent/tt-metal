# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA target selection for the Qwen3.8 hybrid stack.

The wiring itself is generic -- :class:`ttml.modules.lora.LoraModel` freezes the
base model and swaps matching linear layers for their LoRA counterparts.  What
is Qwen3.8-specific is *which* layers to target, and that choice is not the
usual one.

Adapting only ``q/k/v/o_proj`` (the standard recipe) would touch just 16 of the
64 layers here, leaving the 48 Gated DeltaNet layers completely unadapted.  The
default target set below therefore covers the DeltaNet's projections as well:

* attention: ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``
* DeltaNet: ``in_proj_qkv``, ``in_proj_z``, ``out_proj``

The DeltaNet's per-head scalar projections (``in_proj_a``, ``in_proj_b``) are
left alone: they are ``hidden -> 48``, so a rank-16 adapter would be larger than
the layer it adapts.  The conv taps, ``A_log``, ``dt_bias`` and the norms are
plain Parameters rather than linear layers, so they stay frozen.

The MLPs are excluded by default even though they hold ~64% of the parameters,
to keep the adapter count moderate; pass ``include_mlp=True`` to add them.
"""

from __future__ import annotations

from ttml.modules.lora import LoraConfig, LoraModel

__all__ = [
    "ATTENTION_TARGETS",
    "DELTANET_TARGETS",
    "MLP_TARGETS",
    "default_targets",
    "build_lora_config",
    "apply_lora",
    "trainable_summary",
]

# Patterns are regex-searched against fully-qualified module names such as
# "layers.3.self_attn.q_proj", so anchoring on the parent module is enough to
# avoid matching the DeltaNet's similarly-named projections.
ATTENTION_TARGETS = [
    r"self_attn\.q_proj$",
    r"self_attn\.k_proj$",
    r"self_attn\.v_proj$",
    r"self_attn\.o_proj$",
]

DELTANET_TARGETS = [
    r"linear_attn\.in_proj_qkv$",
    r"linear_attn\.in_proj_z$",
    r"linear_attn\.out_proj$",
]

MLP_TARGETS = [
    r"mlp\.gate_proj$",
    r"mlp\.up_proj$",
    r"mlp\.down_proj$",
]


def default_targets(*, include_mlp: bool = False) -> list[str]:
    """Target patterns covering both layer types (and optionally the MLPs)."""
    targets = ATTENTION_TARGETS + DELTANET_TARGETS
    if include_mlp:
        targets = targets + MLP_TARGETS
    return targets


def build_lora_config(
    *,
    rank: int = 16,
    alpha: float = 32.0,
    targets: list[str] | None = None,
    include_mlp: bool = False,
    dropout: float = 0.0,
    use_rslora: bool = True,
    verbose: bool = False,
) -> LoraConfig:
    """A :class:`LoraConfig` with Qwen3.8's target set.

    ``use_rslora`` defaults to ``True`` (scaling ``alpha / sqrt(rank)``), which
    keeps the adapter contribution stable if the rank is later changed.
    """
    return LoraConfig(
        rank=rank,
        alpha=alpha,
        target_modules=targets if targets is not None else default_targets(include_mlp=include_mlp),
        lora_dropout=dropout,
        use_rslora=use_rslora,
        verbose=verbose,
    )


def apply_lora(model, config: LoraConfig | None = None, **kwargs) -> LoraModel:
    """Freeze ``model`` and inject LoRA adapters.

    Load the pretrained checkpoint *before* calling this: wrapping renames every
    parameter under the ``LoraModel`` root, which the loader's name mapping does
    not expect.
    """
    return LoraModel(model, config if config is not None else build_lora_config(**kwargs))


def trainable_summary(model) -> dict:
    """Count trainable vs. total parameters.

    Returns ``{"trainable": n, "total": n, "fraction": f, "adapters": n}``.
    """
    import numpy as np

    trainable = total = adapters = 0
    for name, tensor in model.parameters().items():
        count = int(np.prod([int(d) for d in tensor.shape()]))
        total += count
        if tensor.get_requires_grad():
            trainable += count
            if "lora_" in name:
                adapters += count
    return {
        "trainable": trainable,
        "total": total,
        "fraction": trainable / total if total else 0.0,
        "adapters": adapters,
    }
