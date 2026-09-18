# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Safetensors I/O for Python Qwen3.

Two entry points:
  - :func:`load_from_hf` loads HuggingFace ``Qwen3ForCausalLM`` safetensors into
    a :class:`Qwen3` instance, applying RoPE-pair unpermutation to Q/K
    weights/biases, QK-norm unpermutation, and vocab tile-padding.
  - :func:`export_hf_model` re-permutes and crops back to HF layout and writes a
    safetensors directory that ``AutoModelForCausalLM.from_pretrained`` can load.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional

import ml_dtypes
import numpy as np

import ttnn
import ttml

from ttml.common.utils import resolve_padded_load_shape

from .. import WeightTyingType

# ---------------------------------------------------------------------------
# Weight-permutation helpers (module-private)
# ---------------------------------------------------------------------------


def _unpermute_proj_rows(w: np.ndarray, n_heads: int) -> np.ndarray:
    """HF ``[first_half, second_half]`` → ttml interleaved ``[r0, i0, r1, i1, ...]``."""
    if w.ndim == 1:
        total = w.shape[0]
        D = total // n_heads
        half = D // 2
        wv = w.reshape(n_heads, D)
        first_half = wv[:, :half]
        second_half = wv[:, half:]
        return np.stack([first_half, second_half], axis=2).reshape(total)
    if w.ndim == 2:
        rows, cols = w.shape
        D = rows // n_heads
        half = D // 2
        wv = w.reshape(n_heads, D, cols)
        first_half = wv[:, :half, :]
        second_half = wv[:, half:, :]
        return np.stack([first_half, second_half], axis=2).reshape(rows, cols)
    raise ValueError(f"Expected 1D or 2D tensor, got {w.ndim}D")


def _unpermute_norm_weights(w: np.ndarray) -> np.ndarray:
    """QK-Norm: HF ``[x1, x2, ..., y1, y2, ...]`` → ttml ``[x1, y1, x2, y2, ...]``."""
    head_dim = w.shape[0]
    if head_dim % 2 != 0:
        raise ValueError(f"QK-Norm weight dim must be even; got {head_dim}")
    half = head_dim // 2
    return w.reshape(2, half).T.reshape(-1)


def _repermute_proj_rows(w: np.ndarray, n_heads: int) -> np.ndarray:
    if w.ndim == 1:
        total = w.shape[0]
        D = total // n_heads
        wv = w.reshape(n_heads, D)
        reals = wv[:, 0::2]
        imags = wv[:, 1::2]
        return np.concatenate([reals, imags], axis=1).reshape(total)
    if w.ndim == 2:
        rows, cols = w.shape
        D = rows // n_heads
        wv = w.reshape(n_heads, D, cols)
        reals = wv[:, 0::2, :]
        imags = wv[:, 1::2, :]
        return np.concatenate([reals, imags], axis=1).reshape(rows, cols)
    raise ValueError(f"Expected 1D or 2D tensor, got {w.ndim}D")


def _repermute_norm_weights(w: np.ndarray) -> np.ndarray:
    head_dim = w.shape[0]
    if head_dim % 2 != 0:
        raise ValueError(f"QK-Norm weight dim must be even; got {head_dim}")
    half = head_dim // 2
    return w.reshape(half, 2).T.reshape(-1)


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _pad_to_target(arr: np.ndarray, tgt_rows: int, tgt_cols: int) -> np.ndarray:
    src_rows, src_cols = arr.shape
    if src_rows == tgt_rows and src_cols == tgt_cols:
        return arr
    out = np.zeros((tgt_rows, tgt_cols), dtype=arr.dtype)
    cr = min(src_rows, tgt_rows)
    cc = min(src_cols, tgt_cols)
    out[:cr, :cc] = arr[:cr, :cc]
    return out


def _crop_to_hf_2d(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return arr[:rows, :cols]


def _crop_to_hf_1d(arr: np.ndarray, dim: int) -> np.ndarray:
    return arr[:dim]


def _to_bf16_4d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, 1, -1)
    elif arr.ndim == 2:
        arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
    return arr.astype(ml_dtypes.bfloat16)


def _assign(param, arr_4d: np.ndarray) -> None:
    param.assign(ttml.autograd.Tensor.from_numpy(arr_4d, layout=ttnn.Layout.TILE))


def _fit_to_param_shape(arr: np.ndarray, param, hf_name: str, expected_shape: Optional[tuple] = None) -> np.ndarray:
    """Verify ``arr`` matches the model config, then zero-pad it up to the param's shape.

    The shape policy (raise on config divergence / transpose / crop, tile-pad
    only, warn on unverified tile-pad) lives in
    :func:`ttml.common.utils.resolve_padded_load_shape`; this wrapper just applies
    the numpy padding to the returned target shape.
    """
    shape = param.shape()
    if arr.ndim == 2:
        tgt = resolve_padded_load_shape(arr.shape, (shape[-2], shape[-1]), expected_shape, name=hf_name)
        return _pad_to_target(arr, tgt[0], tgt[1])
    if arr.ndim == 1:
        (tgt_dim,) = resolve_padded_load_shape(arr.shape, (shape[-1],), expected_shape, name=hf_name)
        if arr.shape[0] == tgt_dim:
            return arr
        padded = np.zeros((tgt_dim,), dtype=arr.dtype)
        padded[: arr.shape[0]] = arr
        return padded
    return arr


def _fuse_kv(k: np.ndarray, v: np.ndarray, num_kv_heads: int) -> np.ndarray:
    """Build the fused ``kv_proj`` array from HF k_proj (K) and v_proj (V).

    The ttml Qwen3 attention uses a single ``kv_proj`` of width ``2*kv_out`` whose
    rows are ``[all-K ; all-V]`` (single-device; grouped_heads_creation splits the
    last dim at its midpoint into K then V). K carries the RoPE row-permute (like
    q_proj), so it is un-permuted here; V is left as-is. Weights are 2-D
    ``[kv_out, hidden]`` and biases are 1-D ``[kv_out]``.
    """
    k = _unpermute_proj_rows(k, num_kv_heads)
    return np.concatenate([k, v], axis=0)


def _split_kv(fused: np.ndarray, kv_out: int, num_kv_heads: int):
    """Inverse of :func:`_fuse_kv`: fused ``kv_proj`` -> (HF k_proj, HF v_proj).

    Takes the first ``kv_out`` rows as K and the next ``kv_out`` as V (tolerating
    any TILE tail padding by using the config-derived ``kv_out`` rather than the
    padded row count), then re-permutes K back to HF RoPE layout; V is untouched.
    Returns ``(k_hf, v_hf)``.
    """
    k = fused[:kv_out]
    v = fused[kv_out : 2 * kv_out]
    k = _repermute_proj_rows(k, num_kv_heads)
    return k, v


# ---------------------------------------------------------------------------
# HF ↔ ttml name mapping
# ---------------------------------------------------------------------------
#
# HF ``Qwen3ForCausalLM`` exposes parameters under names like
# ``model.embed_tokens.weight``, ``model.layers.{i}.self_attn.q_proj.weight``,
# ``model.norm.weight``, ``lm_head.weight``. The canonical ttml ``Qwen3`` model
# (see ``ttml/models/qwen3/__init__.py``) uses a different sub-tree:
# ``tok_emb/weight``, ``blocks/{i}/...``, ``ln_fc/weight``, ``fc/weight``.
# These helpers translate between the two; both return ``None`` for names that
# do not correspond to a known parameter so callers can skip cleanly.

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def _hf_to_ttml_name(hf_name: str, *, root: str) -> Optional[str]:
    """HF ``Qwen3ForCausalLM`` tensor name → canonical ttml ``Qwen3`` parameter name."""
    if hf_name == "model.embed_tokens.weight":
        return f"{root}/tok_emb/weight"
    if hf_name == "lm_head.weight":
        return f"{root}/fc/weight"
    if hf_name == "model.norm.weight":
        return f"{root}/ln_fc/weight"
    m = _LAYER_RE.match(hf_name)
    if m:
        return f"{root}/blocks/{m.group(1)}/{m.group(2).replace('.', '/')}"
    return None


def _ttml_to_hf_name(ttml_name: str, *, root: str) -> Optional[str]:
    """Inverse of :func:`_hf_to_ttml_name`."""
    prefix = f"{root}/"
    if not ttml_name.startswith(prefix):
        return None
    rest = ttml_name[len(prefix) :]
    if rest == "tok_emb/weight":
        return "model.embed_tokens.weight"
    if rest == "fc/weight":
        return "lm_head.weight"
    if rest == "ln_fc/weight":
        return "model.norm.weight"
    if rest.startswith("blocks/"):
        parts = rest.split("/")
        if len(parts) >= 3:
            i, *tail = parts[1:]
            return f"model.layers.{i}." + ".".join(tail)
    return None


def _proj_transform_heads(hf_name: str, num_attention_heads: int, num_key_value_heads: int) -> int | None:
    """Return the head count to use when un/repermuting, or None if not a Q projection.

    Only q_proj is handled here. k_proj/v_proj are fused into a single ``kv_proj``
    parameter (see :func:`_stage_or_none` / :func:`_fuse_kv`), so they are staged
    and transformed separately rather than through the generic per-tensor path.
    """
    if ".self_attn.q_proj." in hf_name:
        return num_attention_heads
    return None


def _is_norm_proj(hf_name: str) -> bool:
    return hf_name.endswith(".self_attn.q_norm.weight") or hf_name.endswith(".self_attn.k_norm.weight")


# k_proj / v_proj weight & bias, capturing (layer, "k"|"v", "weight"|"bias").
_KV_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(k|v)_proj\.(weight|bias)$")


def _kv_match(hf_name: str):
    """Return (layer_idx, which, kind) for a k_proj/v_proj tensor, else None.

    ``which`` is "k" or "v"; ``kind`` is "weight" or "bias".
    """
    m = _KV_RE.match(hf_name)
    if m is None:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


def _validate_attention_dims(hf_tensors: Dict[str, np.ndarray], config) -> None:
    """Verify ``Qwen3Config`` matches the HF checkpoint's attention dimensions.

    Q/K/V projections are arranged as ``[head_0_rows..., head_1_rows..., ...]``
    with RoPE-pair interleaving applied per head. Loading them under a config
    whose ``num_attention_heads * head_dim`` doesn't match the HF tensor would
    silently misinterpret the layout (and any cropping in
    :func:`load_from_hf` would discard whole heads or split them mid-rotation),
    producing structurally nonsense weights. Catch this up front with a clear,
    actionable error before any data is written.
    """

    def _peek(suffix: str):
        for name, arr in hf_tensors.items():
            if name.endswith(suffix) and arr.ndim == 2:
                return name, arr
        return None, None

    q_name, q_arr = _peek(".self_attn.q_proj.weight")
    if q_arr is not None:
        expected_q = config.num_attention_heads * config.head_dim
        if q_arr.shape[0] != expected_q:
            implied_head_dim = q_arr.shape[0] // config.num_attention_heads
            raise RuntimeError(
                f"Qwen3Config does not match the HF checkpoint: {q_name} has "
                f"{q_arr.shape[0]} output rows, but config implies "
                f"num_attention_heads * head_dim = {config.num_attention_heads} "
                f"* {config.head_dim} = {expected_q}. "
                f"Likely fix: set head_dim={implied_head_dim} (or adjust "
                f"num_attention_heads) so they multiply to {q_arr.shape[0]}."
            )

    k_name, k_arr = _peek(".self_attn.k_proj.weight")
    if k_arr is not None:
        expected_kv = config.num_key_value_heads * config.head_dim
        if k_arr.shape[0] != expected_kv:
            implied_head_dim = k_arr.shape[0] // config.num_key_value_heads
            raise RuntimeError(
                f"Qwen3Config does not match the HF checkpoint: {k_name} has "
                f"{k_arr.shape[0]} output rows, but config implies "
                f"num_key_value_heads * head_dim = {config.num_key_value_heads} "
                f"* {config.head_dim} = {expected_kv}. "
                f"Likely fix: set head_dim={implied_head_dim} (or adjust "
                f"num_key_value_heads) so they multiply to {k_arr.shape[0]}."
            )

    qn_name, qn_arr = _peek(".self_attn.q_norm.weight")
    if qn_arr is None:
        # 1-D q_norm; loop manually since _peek only matches 2-D tensors.
        for name, arr in hf_tensors.items():
            if name.endswith(".self_attn.q_norm.weight"):
                qn_name, qn_arr = name, arr
                break
    if qn_arr is not None and qn_arr.shape[0] != config.head_dim:
        raise RuntimeError(
            f"Qwen3Config does not match the HF checkpoint: {qn_name} has "
            f"shape {tuple(qn_arr.shape)}, but config.head_dim = "
            f"{config.head_dim}. Set head_dim={qn_arr.shape[0]}."
        )


# ---------------------------------------------------------------------------
# Public API: load HF → ttml
# ---------------------------------------------------------------------------


def load_from_safetensors(model, safetensors_path, config) -> None:
    """Load HuggingFace ``Qwen3ForCausalLM`` safetensors into a Python Qwen3 model.

    Args:
        model: A :class:`~ttml.models.qwen3.Qwen3` instance.
        safetensors_path: Directory containing one or more ``.safetensors`` shards.
        config: The :class:`~ttml.models.qwen3.Qwen3Config` used to build the model.
    """
    from safetensors.numpy import load_file

    directory = Path(safetensors_path)
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {directory}")

    hf_tensors: Dict[str, np.ndarray] = {}
    for shard in shards:
        hf_tensors.update(load_file(str(shard)))

    parameters = model.parameters()
    if not parameters:
        raise RuntimeError("model has no parameters; cannot infer root prefix")
    root = next(iter(parameters)).split("/", 1)[0]

    tied = config.weight_tying == WeightTyingType.Enabled
    loaded: set[str] = set()
    unmapped_hf: list[str] = []
    # Config-implied LOGICAL (un-tile-padded) HF shapes, used to verify each
    # checkpoint tensor matches the model config before tile-padding it (catches
    # e.g. a vocab_size that diverges from the config by less than a tile).
    hf_shapes = _build_hf_shapes(config)
    # HF ships separate k_proj/v_proj; the ttml model has a single fused kv_proj.
    # Stage each layer's K and V arrays keyed by (layer, "weight"|"bias") and fuse
    # them into kv_proj after the main loop (both must be present to fuse).
    kv_staged: Dict[tuple, dict] = {}

    _validate_attention_dims(hf_tensors, config)

    for hf_name, hf_arr in hf_tensors.items():
        # ── Fused KV: stage k_proj / v_proj, fuse after the loop ──
        kv = _kv_match(hf_name)
        if kv is not None:
            layer_idx, which, kind = kv
            kv_staged.setdefault((layer_idx, kind), {})[which] = hf_arr.astype(np.float32)
            continue

        ttml_name = _hf_to_ttml_name(hf_name, root=root)
        if ttml_name is None:
            unmapped_hf.append(hf_name)
            continue
        # Tied: tok_emb.weight and fc.weight share a Parameter object; the
        # deduped parameters() may expose only one name. Fall back to whichever
        # actually exists so the underlying tensor gets loaded either way.
        if ttml_name not in parameters and tied:
            alt = None
            if ttml_name == f"{root}/tok_emb/weight":
                alt = f"{root}/fc/weight"
            elif ttml_name == f"{root}/fc/weight":
                alt = f"{root}/tok_emb/weight"
            if alt is not None and alt in parameters:
                ttml_name = alt
        if ttml_name not in parameters:
            unmapped_hf.append(hf_name)
            continue

        arr = hf_arr.astype(np.float32)

        heads = _proj_transform_heads(hf_name, config.num_attention_heads, config.num_key_value_heads)
        if heads is not None:
            arr = _unpermute_proj_rows(arr, heads)
        elif _is_norm_proj(hf_name):
            arr = _unpermute_norm_weights(arr)

        param = parameters[ttml_name]
        # expected_shape uses the ORIGINAL hf_name (before any tie remap) so the
        # config lookup is correct even when embed_tokens/lm_head share a param.
        arr = _fit_to_param_shape(arr, param, hf_name, expected_shape=hf_shapes.get(hf_name))
        _assign(param, _to_bf16_4d(arr))
        loaded.add(ttml_name)

    # ── Fuse staged K/V into kv_proj ──
    for (layer_idx, kind), kv in sorted(kv_staged.items()):
        ttml_name = f"{root}/blocks/{layer_idx}/self_attn/kv_proj/{kind}"
        if ttml_name not in parameters:
            # Model has no fused kv_proj (or a different layout); surface the
            # skipped HF tensors rather than silently dropping K/V.
            unmapped_hf.extend(f"model.layers.{layer_idx}.self_attn.{w}_proj.{kind}" for w in kv)
            continue
        if "k" not in kv or "v" not in kv:
            missing = "v" if "k" in kv else "k"
            print(
                f"  Warning: layer {layer_idx} {kind}: missing {missing}_proj; "
                f"cannot fuse kv_proj, leaving it at init."
            )
            unmapped_hf.extend(f"model.layers.{layer_idx}.self_attn.{w}_proj.{kind}" for w in kv)
            continue
        fused = _fuse_kv(kv["k"], kv["v"], config.num_key_value_heads)
        param = parameters[ttml_name]
        # Fused kv_proj logical shape: K and V stacked -> 2*kv_dim rows (2*kv_dim
        # for a 1-D bias), hidden cols. Verifies k_proj/v_proj match the config.
        kv_dim = config.num_key_value_heads * config.head_dim
        kv_expected = (2 * kv_dim, config.hidden_size) if kind == "weight" else (2 * kv_dim,)
        fused = _fit_to_param_shape(
            fused, param, f"model.layers.{layer_idx}.self_attn.kv_proj.{kind}", expected_shape=kv_expected
        )
        _assign(param, _to_bf16_4d(fused))
        loaded.add(ttml_name)

    unused = sorted(p for p in parameters if p not in loaded)
    if tied:
        # When tying is enabled, ``self.tok_emb.weight`` and ``self.fc.weight`` are
        # the same Parameter object: writing to either name fills both. So if at
        # least one of them was loaded, neither should be reported as unused.
        emb_name = f"{root}/tok_emb/weight"
        fc_name = f"{root}/fc/weight"
        if emb_name in loaded or fc_name in loaded:
            unused = [p for p in unused if p not in (emb_name, fc_name)]
    if unused:
        print(f"Warning: {len(unused)} model parameters were NOT loaded from safetensors:")
        for name in unused:
            print(f"  - {name}")
    if unmapped_hf:
        print(f"Note: {len(unmapped_hf)} HF tensors were ignored (no mapping):")
        for name in sorted(unmapped_hf):
            print(f"  - {name}")


# ---------------------------------------------------------------------------
# Public API: export ttml → HF
# ---------------------------------------------------------------------------


def _build_hf_shapes(config) -> Dict[str, tuple]:
    shapes: Dict[str, tuple] = {}
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim

    shapes["model.embed_tokens.weight"] = (config.vocab_size, config.hidden_size)
    if config.weight_tying == WeightTyingType.Disabled:
        shapes["lm_head.weight"] = (config.vocab_size, config.hidden_size)

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        shapes[f"{p}.self_attn.q_proj.weight"] = (q_dim, config.hidden_size)
        shapes[f"{p}.self_attn.k_proj.weight"] = (kv_dim, config.hidden_size)
        shapes[f"{p}.self_attn.v_proj.weight"] = (kv_dim, config.hidden_size)
        shapes[f"{p}.self_attn.o_proj.weight"] = (config.hidden_size, q_dim)
        if config.attention_bias:
            shapes[f"{p}.self_attn.q_proj.bias"] = (q_dim,)
            shapes[f"{p}.self_attn.k_proj.bias"] = (kv_dim,)
            shapes[f"{p}.self_attn.v_proj.bias"] = (kv_dim,)
            shapes[f"{p}.self_attn.o_proj.bias"] = (config.hidden_size,)
        shapes[f"{p}.self_attn.q_norm.weight"] = (config.head_dim,)
        shapes[f"{p}.self_attn.k_norm.weight"] = (config.head_dim,)
        shapes[f"{p}.input_layernorm.weight"] = (config.hidden_size,)
        shapes[f"{p}.post_attention_layernorm.weight"] = (config.hidden_size,)
        shapes[f"{p}.mlp.gate_proj.weight"] = (config.intermediate_size, config.hidden_size)
        shapes[f"{p}.mlp.up_proj.weight"] = (config.intermediate_size, config.hidden_size)
        shapes[f"{p}.mlp.down_proj.weight"] = (config.hidden_size, config.intermediate_size)

    shapes["model.norm.weight"] = (config.hidden_size,)
    return shapes


_HF_CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


def export_hf_model(model, config, out_dir, hf_source_dir=None) -> str:
    """Export the model in HF-compatible format.

    Applies inverse transforms (re-permute Q/K rows, re-permute QK-Norm weights,
    crop vocab/intermediate back to their HF shape) and writes
    ``model.safetensors`` to ``out_dir``. When ``hf_source_dir`` is provided,
    tokenizer + config files are copied alongside so the directory loads with
    ``AutoModelForCausalLM.from_pretrained``.
    """
    from safetensors.numpy import save_file

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hf_shapes = _build_hf_shapes(config)
    parameters = model.parameters()
    if not parameters:
        raise RuntimeError("model has no parameters; cannot infer root prefix")
    root = next(iter(parameters)).split("/", 1)[0]

    kv_out = config.num_key_value_heads * config.head_dim

    def _emit_hf(hf_name: str, arr: np.ndarray) -> None:
        """Repermute (Q/K) or norm-repermute, crop to HF shape, and record ``arr``."""
        heads = _proj_transform_heads(hf_name, config.num_attention_heads, config.num_key_value_heads)
        if heads is not None:
            arr = _repermute_proj_rows(arr, heads)
        elif _is_norm_proj(hf_name):
            arr = _repermute_norm_weights(arr)
        hf_shape = hf_shapes[hf_name]
        if arr.ndim == 2:
            arr = _crop_to_hf_2d(arr, hf_shape[0], hf_shape[1])
        elif arr.ndim == 1:
            arr = _crop_to_hf_1d(arr, hf_shape[0])
        hf_state[hf_name] = arr.astype(ml_dtypes.bfloat16)

    hf_state: Dict[str, np.ndarray] = {}
    for ttml_name, param in parameters.items():
        hf_name = _ttml_to_hf_name(ttml_name, root=root)
        if hf_name is None:
            continue
        arr = param.to_numpy(ttnn.DataType.FLOAT32).squeeze()

        # ── Fused KV: split kv_proj back into HF k_proj + v_proj ──
        # (``_ttml_to_hf_name`` renders kv_proj as ``...self_attn.kv_proj.{kind}``,
        # which is not a real HF name and not in hf_shapes, so it must be handled
        # here rather than through the generic path below.)
        if ".self_attn.kv_proj." in hf_name:
            kind = "bias" if hf_name.endswith(".bias") else "weight"
            base = hf_name[: -len(f".kv_proj.{kind}")]
            k_name = f"{base}.k_proj.{kind}"
            v_name = f"{base}.v_proj.{kind}"
            if k_name not in hf_shapes or v_name not in hf_shapes:
                continue
            k_arr, v_arr = _split_kv(arr, kv_out, config.num_key_value_heads)
            # _emit_hf re-permutes k_proj (via _proj_transform_heads) and leaves
            # v_proj as-is; both are cropped to their HF shape.
            if k_name not in hf_state:
                _emit_hf(k_name, k_arr)
            if v_name not in hf_state:
                _emit_hf(v_name, v_arr)
            continue

        # Skip params that don't map to an HF name we know how to write.
        # When tying is enabled, ``hf_shapes`` omits ``lm_head.weight`` (HF
        # ships only ``model.embed_tokens.weight``), so ``fc/weight`` is
        # naturally skipped here; the embedding itself is written from
        # ``tok_emb/weight`` which shares the same underlying tensor.
        if hf_name not in hf_shapes or hf_name in hf_state:
            continue
        _emit_hf(hf_name, arr)

    save_file(hf_state, str(out_path / "model.safetensors"))

    # Minimal single-shard index so HF treats this as a valid model dir.
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    with open(out_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    if hf_source_dir is not None and os.path.isdir(hf_source_dir):
        for fname in _HF_CONFIG_FILES:
            src = os.path.join(hf_source_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, out_path / fname)

    return str(out_path)
