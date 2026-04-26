# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Kimi K2.5 weight loader — transparently dequantizes INT4 expert weights.

Wraps DSV3's :class:`LazyStateDict` with Kimi K2.5-specific logic:

* **Prefix**: ``language_model.`` is stripped from all exposed keys so callers
  see keys starting with ``model.layers.*``, ``model.norm.*``,
  ``model.embed_tokens.*``, and ``lm_head.*`` — matching the DSV3 convention
  that :func:`RowBatchedModel.convert_weights` expects when it calls
  :func:`~.config_helpers.sub_state_dict` with prefixes such as
  ``"model.embed_tokens."``, ``"model.layers.{i}."``, and ``"lm_head."``.
* **INT4 expert weights**: accessing a ``*.weight_packed`` key whose companion
  ``*.weight_scale`` key exists in the index returns a dequantized BF16 tensor
  instead of the raw I32 buffer.  The ``*.weight_scale`` and ``*.weight_shape``
  keys remain directly accessible for diagnostics.
* **Vision encoder**: keys under ``vision_model.*`` are outside the
  ``language_model.*`` prefix and therefore invisible to this class.
  The text-only inference path never requests them.
* **BF16 pass-through**: all non-expert weights (attention, shared experts,
  dense layer 0, lm_head) are BF16 in the checkpoint and are returned as-is.

Exposed key mapping (checkpoint → KimiLazyStateDict view)::

    language_model.model.embed_tokens.*       →  model.embed_tokens.*
    language_model.model.layers.{L}.*        →  model.layers.{L}.*
    language_model.model.norm.*              →  model.norm.*
    language_model.lm_head.*                →  lm_head.*
    vision_model.*                           →  (invisible)

Weight key naming (real checkpoint, verified 2026-03-12)::

    model.layers.{L}.mlp.experts.{E}.gate_proj.weight_packed   # I32 (out, in//8)
    model.layers.{L}.mlp.experts.{E}.gate_proj.weight_scale    # BF16 (out, in//32)
    model.layers.{L}.mlp.experts.{E}.gate_proj.weight_shape    # I32 [2] = [out, in]
    model.layers.{L}.mlp.experts.{E}.up_proj.weight_packed
    model.layers.{L}.mlp.experts.{E}.up_proj.weight_scale
    model.layers.{L}.mlp.experts.{E}.up_proj.weight_shape
    model.layers.{L}.mlp.experts.{E}.down_proj.weight_packed
    model.layers.{L}.mlp.experts.{E}.down_proj.weight_scale
    model.layers.{L}.mlp.experts.{E}.down_proj.weight_shape

Example usage::

    from models.demos.kimi_k25.utils.weight_loader import KimiLazyStateDict

    state = KimiLazyStateDict("/workspace/extra/Kimi-K2.5")

    # Returns BF16 tensor of shape (2048, 7168) — automatically dequantized
    w = state["model.layers.1.mlp.experts.0.gate_proj.weight_packed"]
    assert w.dtype == torch.bfloat16
    assert w.shape == (2048, 7168)

    # BF16 attention weight — direct pass-through
    q = state["model.layers.0.self_attn.q_proj.weight"]

    # LM head (BF16, at language_model.lm_head.* in checkpoint)
    head = state["lm_head.weight"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.kimi_k25.utils.int4_dequantize import dequantize_int4_weight

# ── Constants ────────────────────────────────────────────────────────────────

#: Top-level prefix in HF checkpoint keys for the text backbone.
#: Stripping this prefix exposes keys as DSV3's convert_weights() expects:
#:   ``model.embed_tokens.*``, ``model.layers.*``, ``model.norm.*``, ``lm_head.*``
KIMI_MODEL_PREFIX: str = "language_model."

#: Suffix that identifies INT4-packed expert weight tensors.
_PACKED_SUFFIX: str = ".weight_packed"

#: Suffix for per-group INT4 scale tensors.
_SCALE_SUFFIX: str = ".weight_scale"

#: Suffix for original ``[out_features, in_features]`` shape tensors.
_SHAPE_SUFFIX: str = ".weight_shape"

#: Appended to the full cache key for dequantized results to avoid collision
#: with the raw I32 tensor cached under the plain packed key.
_DEQUANT_CACHE_TAG: str = "#kimi_bf16"

#: INT4 group size used by Kimi K2.5 routed experts.
_GROUP_SIZE: int = 32


# ── Public helpers ───────────────────────────────────────────────────────────


def is_int4_packed_key(key: str) -> bool:
    """Return *True* if *key* refers to an INT4-packed weight tensor."""
    return key.endswith(_PACKED_SUFFIX)


def packed_key_to_scale_key(packed_key: str) -> str:
    """Derive the ``.weight_scale`` key from a ``.weight_packed`` key."""
    return packed_key[: -len("_packed")] + "_scale"


def packed_key_to_shape_key(packed_key: str) -> str:
    """Derive the ``.weight_shape`` key from a ``.weight_packed`` key."""
    return packed_key[: -len("_packed")] + "_shape"


def dequantize_i32_packed(
    packed_i32: torch.Tensor,
    scales: torch.Tensor,
    weight_shape: Optional[torch.Tensor] = None,
    group_size: int = _GROUP_SIZE,
) -> torch.Tensor:
    """Dequantize a Kimi K2.5 I32-packed INT4 weight to BF16.

    Kimi K2.5 packs **8 INT4 nibbles** into each 32-bit integer, so a weight
    matrix with logical shape ``(out, in)`` is stored as ``(out, in // 8)``
    I32.  By reinterpreting the storage as ``uint8`` we get a ``(out, in // 2)``
    byte tensor where each byte holds two nibbles — exactly the format
    expected by :func:`~.int4_dequantize.dequantize_int4_weight`.

    Args:
        packed_i32:   I32 tensor, shape ``(out_features, in_features // 8)``.
        scales:       BF16 (or float) tensor, shape
                      ``(out_features, in_features // group_size)``.
        weight_shape: Optional I32 tensor of length 2 storing ``[out, in]``.
                      Used only for post-dequantization shape validation.
                      Pass *None* to skip validation.
        group_size:   INT4 group size (default 32).

    Returns:
        BF16 tensor, shape ``(out_features, in_features)``.

    Raises:
        TypeError:  if *packed_i32* is not I32.
        ValueError: if the dequantized shape does not match *weight_shape*.
    """
    if packed_i32.dtype != torch.int32:
        raise TypeError(
            f"Expected torch.int32 packed tensor, got dtype={packed_i32.dtype}. "
            "Ensure the safetensors key ends with '.weight_packed'."
        )

    # Reinterpret the I32 memory as uint8:
    #   (out, in // 8) I32  →  (out, in // 2) uint8
    # Each pair of uint8 bytes corresponds to one int32, preserving nibble order
    # on little-endian hosts (standard x86 / ARM).
    packed_u8: torch.Tensor = packed_i32.view(torch.uint8)

    result = dequantize_int4_weight(
        packed_u8,
        scales,
        group_size=group_size,
        output_dtype=torch.bfloat16,
    )

    if weight_shape is not None:
        expected = tuple(int(v) for v in weight_shape.tolist())
        actual = tuple(result.shape)
        if actual != expected:
            raise ValueError(f"Dequantized shape {actual} does not match " f"weight_shape tensor {expected}.")

    return result


# ── Main class ───────────────────────────────────────────────────────────────


class KimiLazyStateDict(LazyStateDict):
    """Lazy state-dict accessor for Kimi K2.5 HuggingFace checkpoints.

    Inherits from :class:`~models.demos.deepseek_v3.utils.lazy_state_dict.LazyStateDict`
    and adds Kimi-specific behaviour:

    * Strips the ``language_model.`` prefix from all exposed keys, yielding
      a key namespace that matches DSV3's :func:`RowBatchedModel.convert_weights`
      expectations::

          model.embed_tokens.*      ← Embedding2D
          model.layers.{i}.*       ← DecoderBlock2D / MoEDecoderBlock2D
          model.norm.*             ← DistributedRMSNorm
          lm_head.*               ← LMHead1D

    * Transparently dequantizes ``*.weight_packed`` tensors (I32 → BF16) when
      a corresponding ``*.weight_scale`` key exists in the index.
    * BF16 tensors (attention, shared experts, lm_head) are returned unchanged.
    * Dequantized results are cached under a distinct cache key to avoid
      evicting the raw packed tensor from the shared tensor cache.
    * :meth:`view_with_prefix` is overridden to return a :class:`KimiLazyStateDict`
      instance (not a plain :class:`LazyStateDict`), preserving INT4 dequant
      in all sub-views created by :func:`~.config_helpers.sub_state_dict`.

    Args:
        model_path: Directory containing ``model.safetensors.index.json`` and
                    the shard files, e.g.
                    ``Path("/workspace/extra/Kimi-K2.5")``.

    All other keyword arguments are for internal use by :meth:`view_with_prefix`
    and are not intended for direct callers.
    """

    def __init__(
        self,
        model_path: "Path | str",
        *,
        _base_prefix: str = KIMI_MODEL_PREFIX,
        _full_to_file=None,
        _cache=None,
        _num_layers=None,
        _file_handles=None,
    ) -> None:
        super().__init__(
            model_path,
            base_prefix=_base_prefix,
            _full_to_file=_full_to_file,
            _cache=_cache,
            _num_layers=_num_layers,
            _file_handles=_file_handles,
        )

    # ── Prefix view override ──────────────────────────────────────────────

    def view_with_prefix(self, prefix: str, num_layers: Optional[int] = None) -> "KimiLazyStateDict":
        """Return a :class:`KimiLazyStateDict` view narrowed to *prefix*.

        Overrides :meth:`LazyStateDict.view_with_prefix` to return a
        :class:`KimiLazyStateDict` instead of a plain :class:`LazyStateDict`.
        This ensures that INT4 dequantization logic is preserved in all
        sub-views created by :func:`~.config_helpers.sub_state_dict` during
        :func:`RowBatchedModel.convert_weights`.

        The combined ``_base_prefix`` of the returned view is
        ``self._base_prefix + prefix``.

        Args:
            prefix:     Additional key prefix to narrow the view.
            num_layers: Optional layer count filter (inherited from parent).

        Returns:
            A new :class:`KimiLazyStateDict` sharing the same underlying
            file handles and tensor cache.
        """
        combined_prefix = self._base_prefix + prefix
        child_num_layers = self._num_layers if num_layers is None else num_layers
        return KimiLazyStateDict(
            self._model_path,
            _base_prefix=combined_prefix,
            _full_to_file=self._full_to_file,
            _cache=self._cache,
            _num_layers=child_num_layers,
            _file_handles=self._file_handles,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _scale_key_exists(self, relative_packed_key: str) -> bool:
        """Return *True* if the companion scale tensor is in the checkpoint."""
        scale_full = self._base_prefix + packed_key_to_scale_key(relative_packed_key)
        return scale_full in self._full_to_file

    def _load_and_dequantize(self, relative_packed_key: str) -> torch.Tensor:
        """Load raw packed/scale tensors and return a dequantized BF16 weight.

        Dequantized result is cached under ``full_key + _DEQUANT_CACHE_TAG``
        in the shared tensor cache to avoid redundant computation.
        """
        dequant_cache_key = self._base_prefix + relative_packed_key + _DEQUANT_CACHE_TAG
        cached = self._cache.get(dequant_cache_key)
        if cached is not None:
            return cached

        # Load raw tensors via parent (each is cached under its own full key)
        packed_i32: torch.Tensor = super().__getitem__(relative_packed_key)
        scale_relative = packed_key_to_scale_key(relative_packed_key)
        scales: torch.Tensor = super().__getitem__(scale_relative)

        # Optionally load weight_shape for validation
        shape_relative = packed_key_to_shape_key(relative_packed_key)
        weight_shape: Optional[torch.Tensor] = super().__getitem__(shape_relative) if shape_relative in self else None

        result = dequantize_i32_packed(packed_i32, scales, weight_shape=weight_shape)
        self._cache[dequant_cache_key] = result
        return result

    # ── Public interface ──────────────────────────────────────────────────

    def __getitem__(self, key: str) -> torch.Tensor:
        """Return tensor for *key* (relative to ``language_model.``).

        If *key* ends with ``.weight_packed`` **and** a corresponding
        ``.weight_scale`` exists in the checkpoint index, the returned tensor
        is a dequantized BF16 weight rather than the raw I32 buffer.

        Raises:
            KeyError: if *key* does not exist under the current prefix.
        """
        if is_int4_packed_key(key) and self._scale_key_exists(key):
            return self._load_and_dequantize(key)
        return super().__getitem__(key)
