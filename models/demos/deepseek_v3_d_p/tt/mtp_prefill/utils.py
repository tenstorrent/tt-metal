# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers for the GLM-5.2 MTP module: checkpoint loading and the ``eh_proj`` TP layout.

Deliberately free of any ``ttnn`` import so the ``eh_proj`` shard math can be unit-tested on a host
with no device and no built ttnn.
"""

from __future__ import annotations

import json
import os

import torch

MTP_CACHE_PREFIX = "mtp_0"
"""TTNN weight-cache key prefix for the shared MTP weight module. Shared by whoever BUILDS the cache
and whoever checks it: a prefix mismatch does not raise, it just reports the cache incomplete."""

MTP_CACHE_ENV = "TT_GLM52_MTP_TTNN_CACHE"
"""Env override for the MTP cache ROOT (the ``<variant>_<arch>_<N>dev/<sp>x<tp>`` leaf is appended).
The MTP weights are keyed on layer 78, which the trunk cache does not carry, so they live in their
own tree."""

# The four non-decoder-layer tensors an MTP module adds on top of a normal GLM decoder layer.
# `shared_head.norm` is the only one on the OUTPUT side.
MTP_TENSOR_SUFFIXES = {
    "eh_proj.weight": "eh_proj",
    "enorm.weight": "enorm",
    "hnorm.weight": "hnorm",
    "shared_head.norm.weight": "shared_head_norm",
}


def eh_proj_to_tt_layout(eh_proj_weight: torch.Tensor, tp: int) -> torch.Tensor:
    """Transpose ``eh_proj`` to ttnn ``[in, out]`` order and permute its rows for TP sharding.

    ``enorm`` and ``hnorm`` each emit the chip's own contiguous slice of the *global* hidden, so on
    chip ``c`` the concatenated activation covers global input columns::

        {c*H/tp .. (c+1)*H/tp}   union   {H + c*H/tp .. H + (c+1)*H/tp}

    That is NOT the contiguous ``[c*2H/tp, (c+1)*2H/tp)`` block a plain ``dims=(None, -2)``
    mesh-mapper split hands chip ``c``. Sharding the transposed weight naively yields a tensor of
    exactly the right shape holding exactly the wrong rows — no error, just wrong numbers. This
    reorders the rows chip-major up front so the standard mapper lands correctly.

    Args:
        eh_proj_weight: HF layout ``[H, 2H]`` (out, in).
        tp: tensor-parallel width (mesh columns), must divide ``H``.

    Returns:
        ``[2H, H]`` (in, out), rows ordered so block ``c`` is ``[c's enorm rows ; c's hnorm rows]``.
    """
    h_out, w_in = eh_proj_weight.shape
    assert w_in == 2 * h_out, f"eh_proj must be [H, 2H], got {tuple(eh_proj_weight.shape)}"
    assert tp >= 1 and h_out % tp == 0, f"tp={tp} must divide hidden={h_out}"

    w_t = eh_proj_weight.t().contiguous()  # [2H, H] — rows 0..H-1 = enorm half, H..2H-1 = hnorm half
    return w_t.view(2, tp, h_out // tp, h_out).transpose(0, 1).contiguous().reshape(2 * h_out, h_out)


def eh_proj_expected_chip_shard(eh_proj_weight: torch.Tensor, tp: int, chip: int) -> torch.Tensor:
    """The ``[2H/tp, H]`` block chip ``chip`` must hold, derived straight from the HF layout.

    Independent of :func:`eh_proj_to_tt_layout` on purpose — it slices the original ``[H, 2H]``
    weight by the column ranges the two norms actually produce, so comparing the two is a real
    cross-check rather than a restatement of the same expression.
    """
    h_out, w_in = eh_proj_weight.shape
    assert w_in == 2 * h_out, f"eh_proj must be [H, 2H], got {tuple(eh_proj_weight.shape)}"
    assert tp >= 1 and h_out % tp == 0, f"tp={tp} must divide hidden={h_out}"
    block = h_out // tp
    e_cols = eh_proj_weight[:, chip * block : (chip + 1) * block]
    h_cols = eh_proj_weight[:, h_out + chip * block : h_out + (chip + 1) * block]
    return torch.cat([e_cols, h_cols], dim=1).t().contiguous()  # [2H/tp, H]


def _resolve_weight_map(path: str) -> tuple[dict[str, str], bool]:
    """Return ``({tensor_name: shard_file}, is_sharded)`` for a HF checkpoint directory."""
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return json.load(f)["weight_map"], True
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return {}, False
    raise FileNotFoundError(
        f"no safetensors checkpoint at {path} " "(expected model.safetensors.index.json or model.safetensors)"
    )


def mtp_layer_idx_from_config(path: str) -> int:
    """The layer index the MTP weights live on: ``num_hidden_layers`` (78 for GLM-5.2).

    Reads ``config.json`` directly rather than via ``AutoConfig`` — ``glm_moe_dsa`` is not
    AutoConfig-loadable (see ``runners/adapters/glm_5_2.py::load_hf_config``).
    """
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    return int(cfg["num_hidden_layers"])


def load_mtp_state_dict(path: str, *, layer_idx: int | None = None) -> dict[str, torch.Tensor]:
    """Load exactly the four MTP tensors from a GLM-5.2 HF checkpoint directory.

    Reads only the shard(s) that actually hold them (the GLM-5.2 checkout is 141 shards / ~641 GiB),
    on the ``dflash_prefill/utils.py::load_drafter_state_dict`` pattern.

    Args:
        path: checkpoint directory (``$GLM52_HF_MODEL``, default
            ``/mnt/models/deepseek-prefill-cache/GLM-5.2-FP8``).
        layer_idx: the MTP layer. Defaults to ``num_hidden_layers`` from ``config.json``.

    Returns:
        ``{"eh_proj", "enorm", "hnorm", "shared_head_norm"}`` -> torch tensors, HF layout.
    """
    from safetensors import safe_open

    if layer_idx is None:
        layer_idx = mtp_layer_idx_from_config(path)

    prefix = f"model.layers.{layer_idx}."
    wanted = {prefix + suffix: short for suffix, short in MTP_TENSOR_SUFFIXES.items()}

    weight_map, is_sharded = _resolve_weight_map(path)
    if is_sharded:
        by_shard: dict[str, list[str]] = {}
        for key in wanted:
            shard = weight_map.get(key)
            if shard is not None:
                by_shard.setdefault(shard, []).append(key)
    else:
        by_shard = {"model.safetensors": list(wanted)}

    sd: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(path, shard), framework="pt") as f:
            available = set(f.keys())
            for key in keys:
                if key in available:
                    sd[wanted[key]] = f.get_tensor(key)

    missing = sorted(set(MTP_TENSOR_SUFFIXES.values()) - set(sd))
    assert not missing, (
        f"checkpoint {path} has no MTP weights for layer {layer_idx} (missing {missing}). "
        f"Expected e.g. {prefix}eh_proj.weight — check num_hidden_layers and that this is an "
        "MTP-carrying checkout."
    )
    # eh_proj is BF16 in the checkpoint with no weight_scale_inv, unlike every MLA/MoE weight on the
    # same layer. If that ever changes an fp8 dequant path is needed and the raw load below is wrong.
    assert sd["eh_proj"].dtype == torch.bfloat16, (
        f"eh_proj expected bfloat16 (no dequant path), got {sd['eh_proj'].dtype}; "
        f"{prefix}eh_proj.weight_scale_inv present in checkpoint?"
    )
    return sd


def mtp_indexer_types(config, mtp_layer_idx: int | None = None) -> list:
    """``config.indexer_types`` extended so the MTP layer owns an index-cache slot.

    GLM-5.2's map has ``num_hidden_layers`` (78) entries, covering layers 0..77 only, while the MTP
    layer sits at 78 and carries **real indexer weights** (layer 77 carries none). Two consumers read
    past the end and disagree about what they find:

    * :func:`~models.demos.deepseek_v3_d_p.tt.mla.indexer.indexer_layer_is_reused` and
      :func:`~models.demos.deepseek_v3_d_p.tt.mla.indexer.full_indexer_rank` both guard on
      ``layer_idx < len(types)`` and fail *open*, so layer 78 is treated as ``full`` with rank 21 —
      the right answers, but by an out-of-range fallback rather than a declaration.
    * ``TtIndexer``'s compacted cache accounting is **not** right: with a 78-entry map,
      ``num_full_indexer_layers`` is 21 and the layer's own rank is also 21, i.e. slot 21 of 21
      (one past the end). Declaring the layer explicitly makes it slot 21 of 22. The pipeline-stage
      form is worse — ``first_layer_idx=78, layer_num=1`` yields slot 0 of **0** slots — and becomes
      slot 0 of 1.

    Verified: extending the map changes no slot for layers 0..77. See issue #53533.

    Args:
        config: a GLM HF-attribute config carrying ``indexer_types``.
        mtp_layer_idx: the MTP layer. Defaults to ``config.num_hidden_layers`` (78), falling back to
            ``len(config.indexer_types)``. Deriving it from the layer count rather than the current
            map length is what makes this idempotent — defaulting to the length would append a fresh
            entry on every call.

    Returns:
        A NEW list, ``mtp_layer_idx + 1`` entries long, with ``"full"`` at ``mtp_layer_idx``.
        Returns a copy unchanged when the map already covers the layer.
    """
    types = list(getattr(config, "indexer_types", None) or [])
    assert types, "config has no indexer_types (GLM-5.1 and dense variants: every layer is full, nothing to extend)"
    if mtp_layer_idx is None:
        mtp_layer_idx = int(getattr(config, "num_hidden_layers", None) or len(types))
    while len(types) <= mtp_layer_idx:
        # "full": the MTP layer ships its own indexer weights, so it computes its own top-k rather
        # than reusing a trunk layer's. `index_share_for_mtp_iteration` is about sharing ACROSS MTP
        # levels, not with the trunk.
        types.append("full")
    return types


def enable_mtp_indexer_slot(config, mtp_layer_idx: int | None = None) -> int:
    """Extend ``config.indexer_types`` in place via :func:`mtp_indexer_types`. Idempotent.

    Returns the MTP layer index that is now covered. Note this mutates ``config``; callers holding a
    shared/cached config object (``tests/conftest.py::config_only`` is ``lru_cache``d) should copy first.
    """
    types = mtp_indexer_types(config, mtp_layer_idx)
    config.indexer_types = types
    return len(types) - 1
