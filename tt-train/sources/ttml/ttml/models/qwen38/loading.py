# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load HuggingFace Qwen3.8 safetensors into a tt-train :class:`Qwen38Transformer`.

The checkpoint's parameter names line up almost one-to-one with the module tree,
because the modules were named after the checkpoint. ``LinearLayer`` stores its
weight as ``[1, 1, out, in]``, which is HF's ``[out, in]`` with two leading
singleton dims, so no transpose is needed either.

Four transforms are genuinely required:

1. **Zero-centered RMSNorm.** ``Qwen3_5RMSNorm`` computes
   ``x_normed * (1 + weight)``, so ``1`` is folded into the stored weight and the
   model can use the plain fused ``rmsnorm`` op.  This applies to
   ``input_layernorm``, ``post_attention_layernorm``, the final ``norm``, and the
   attention ``q_norm`` / ``k_norm``.

   It deliberately does **not** apply to the DeltaNet's ``linear_attn.norm``,
   which is ``Qwen3_5RMSNormGated`` and uses the weight as-is.  Getting this
   backwards is silent -- the model still runs, just wrongly -- so the two cases
   are separated explicitly below rather than matched by a name pattern.

2. **Conv taps.** The fused depthwise ``conv1d.weight`` is ``[C, 1, K]``; the
   model holds it as ``K`` separate ``[1, 1, 1, C]`` taps, tap ``j`` being
   ``weight[:, 0, j]``.

3. **Per-head scalars.** ``A_log`` and ``dt_bias`` are ``[H_v]`` in the
   checkpoint and ``[1, 1, 1, H_v]`` in the model, for broadcasting over
   ``[B, 1, T, H_v]``.

4. **Vocab padding.** The embedding table is padded up to a tile multiple
   (Qwen3.8's 248320 is already aligned, so this is usually a no-op).

The vision tower (``visual.*``) and the MTP head (``mtp.*``) are skipped: neither
is part of the text LoRA training graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator

import numpy as np

import ttnn
import ttml

from .parallel import qkv_shard_permutation, tp_shard_dim, tp_size

__all__ = ["load_from_safetensors", "SKIPPED_PREFIXES"]

# Checkpoint subtrees that are not part of the text backbone.
SKIPPED_PREFIXES = ("visual.", "model.visual.", "mtp.")

_HF_ROOT = "model.language_model."

# Norms that HF implements as x_normed * (1 + w); the +1 is folded in at load.
_ZERO_CENTERED_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)


def _iter_shards(directory: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, array) from every shard, one shard resident at a time.

    The 27B checkpoint is 18 shards / ~54 GB, so shards are streamed rather than
    merged into one dict as the smaller Qwen3 loader does.
    """
    from safetensors import safe_open

    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards: Dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            shards.setdefault(shard, []).append(name)
        shard_items = sorted(shards.items())
    else:
        shard_items = [(p.name, None) for p in sorted(directory.glob("*.safetensors"))]

    for shard_name, names in shard_items:
        path = directory / shard_name
        if not path.exists():
            raise FileNotFoundError(f"shard referenced by index is missing: {path}")
        with safe_open(str(path), framework="np") as handle:
            for name in names if names is not None else handle.keys():
                yield name, handle.get_tensor(name)


def _to_4d(arr: np.ndarray) -> np.ndarray:
    """Left-pad the shape with singleton dims to rank 4."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    while arr.ndim < 4:
        arr = arr[np.newaxis, ...]
    return arr


def _assign(
    param,
    arr: np.ndarray,
    name: str,
    tp: int = 1,
    tp_axis: str = "tp",
    dtype=ttnn.DataType.BFLOAT16,
) -> None:
    """Write a numpy array into a ttml Parameter, checking the shape first.

    Under tensor parallelism a sharded parameter reports its *local* shape, so
    the full checkpoint tensor is handed to ``from_numpy`` together with a
    mapper that splits it across the TP axis -- the shape check compares the
    per-chip slice, not the global tensor.
    """
    shard_dim = tp_shard_dim(name) if tp > 1 else None

    expected = tuple(int(d) for d in param.tensor.shape())
    local = list(arr.shape)
    if shard_dim is not None:
        if local[shard_dim] % tp:
            raise ValueError(f"{name}: dim {shard_dim} of {arr.shape} does not divide tp={tp}")
        local[shard_dim] //= tp
    if tuple(local) != expected:
        raise ValueError(
            f"{name}: checkpoint gives {arr.shape}"
            + (f" (local {tuple(local)} at tp={tp})" if shard_dim is not None else "")
            + f", model expects {expected}"
        )

    mapper = None
    if shard_dim is not None:
        mapper = ttml.mesh().axis_mapper(tp_axis, tdim=shard_dim)
    param.assign(
        ttml.autograd.Tensor.from_numpy(
            np.ascontiguousarray(arr, dtype=np.float32),
            layout=ttnn.Layout.TILE,
            # from_numpy's new_type defaults to "keep the numpy dtype", which
            # would store every base weight as FP32. That is both twice the
            # memory ttml computes in and, for the 27B, the difference between
            # fitting and not: 7.7 B parameters per chip is ~31 GB at FP32
            # against ~34 GB of DRAM, so the weights alone fill the device and
            # the first activation fails to allocate.
            new_type=dtype,
            mapper=mapper,
        )
    )


def _pad_vocab(arr: np.ndarray, rows: int) -> np.ndarray:
    """Pad an ``[vocab, hidden]`` table up to ``rows`` with zeros."""
    if arr.shape[0] == rows:
        return arr
    if arr.shape[0] > rows:
        raise ValueError(f"checkpoint vocab {arr.shape[0]} exceeds model's {rows}")
    padded = np.zeros((rows, arr.shape[1]), dtype=np.float32)
    padded[: arr.shape[0]] = arr
    return padded


def _target_name(hf_name: str, root: str, config) -> str | None:
    """Map a checkpoint parameter name onto a ttml parameter name.

    Returns ``None`` for tensors with no counterpart in the text backbone.
    """
    if hf_name.startswith(SKIPPED_PREFIXES):
        return None

    if hf_name == "lm_head.weight":
        return f"{root}/lm_head/weight"
    if hf_name == f"{_HF_ROOT}embed_tokens.weight":
        return f"{root}/embed_tokens/weight"
    if hf_name == f"{_HF_ROOT}norm.weight":
        return f"{root}/norm/weight"

    if not hf_name.startswith(f"{_HF_ROOT}layers."):
        return None

    rest = hf_name[len(f"{_HF_ROOT}layers.") :]
    layer_str, _, tail = rest.partition(".")
    prefix = f"{root}/layers/{int(layer_str)}"

    # Parameters the model stores as a bare Parameter rather than inside a
    # submodule, so the name has no trailing "/weight".
    bare = {
        "linear_attn.A_log": "linear_attn/A_log",
        "linear_attn.dt_bias": "linear_attn/dt_bias",
        "linear_attn.norm.weight": "linear_attn/norm_weight",
    }
    if tail in bare:
        return f"{prefix}/{bare[tail]}"

    # conv1d is expanded into taps by the caller, not mapped to one name.
    if tail == "linear_attn.conv1d.weight":
        return f"{prefix}/linear_attn/conv1d"

    # Everything else: a.b.weight -> a/b/weight.
    return f"{prefix}/{tail.replace('.', '/')}"


def load_from_safetensors(model, checkpoint_dir, config, *, strict: bool = True, dtype=ttnn.DataType.BFLOAT16) -> dict:
    """Load a Qwen3.8 checkpoint into ``model`` in place.

    Args:
        model: a :class:`~ttml.models.qwen38.Qwen38Transformer`.
        checkpoint_dir: directory holding the ``.safetensors`` shards.
        config: the :class:`~ttml.models.qwen38.Qwen38Config` used to build it.
        strict: raise if any model parameter is left unloaded.
        dtype: on-device dtype for the base weights. BF16 by default, matching
            what ttml computes in; FP32 doubles the footprint for no benefit to
            a frozen LoRA base.

    Returns:
        ``{"loaded": n, "skipped": n, "unexpected": [...]}``.
    """
    directory = Path(checkpoint_dir)
    parameters = model.parameters()
    if not parameters:
        raise RuntimeError("model has no parameters")
    root = next(iter(parameters)).split("/", 1)[0]

    kernel = config.linear_conv_kernel_dim
    # Under TP the fused QKV rows (and the conv taps that scale them) are
    # reordered so each chip's contiguous shard is a coherent set of heads.
    tp = tp_size(config)
    qkv_perm = qkv_shard_permutation(config, tp) if tp > 1 else None
    embed_rows = int(parameters[f"{root}/embed_tokens/weight"].tensor.shape()[2])

    loaded: set[str] = set()
    skipped = 0
    unexpected: list[str] = []

    for hf_name, hf_arr in _iter_shards(directory):
        target = _target_name(hf_name, root, config)
        if target is None:
            skipped += 1
            continue

        arr = np.asarray(hf_arr, dtype=np.float32)

        # The fused depthwise conv becomes K separate broadcast taps.
        if target.endswith("/linear_attn/conv1d"):
            channels, _, taps = arr.shape
            if taps != kernel:
                raise ValueError(f"{hf_name}: kernel {taps} != config {kernel}")
            if qkv_perm is not None:
                # Per-channel weights, so they follow the QKV row permutation.
                arr = arr[qkv_perm]
            base = target[: -len("/conv1d")]
            for tap_idx in range(kernel):
                tap_name = f"{base}/conv_tap_{tap_idx}"
                tap = arr[:, 0, tap_idx].reshape(1, 1, 1, channels)
                _assign(parameters[tap_name], tap, tap_name, tp, config.tp_axis_name, dtype)
                loaded.add(tap_name)
            continue

        if target not in parameters:
            unexpected.append(hf_name)
            continue

        # Fold the +1 of zero-centered RMSNorm into the stored weight. The
        # DeltaNet's gated norm is excluded: it uses the weight directly.
        if hf_name.endswith(_ZERO_CENTERED_SUFFIXES) or hf_name == f"{_HF_ROOT}norm.weight":
            arr = arr + 1.0

        if qkv_perm is not None and target.endswith("/linear_attn/in_proj_qkv/weight"):
            arr = arr[qkv_perm]

        if target.endswith(("/embed_tokens/weight", "/lm_head/weight")):
            arr = _pad_vocab(arr, embed_rows)

        _assign(parameters[target], _to_4d(arr), target, tp, config.tp_axis_name, dtype)
        loaded.add(target)

    missing = sorted(set(parameters) - loaded)
    if missing and strict:
        raise RuntimeError(f"{len(missing)} parameters not loaded, e.g. {missing[:8]}")

    return {"loaded": len(loaded), "skipped": skipped, "unexpected": unexpected, "missing": missing}
