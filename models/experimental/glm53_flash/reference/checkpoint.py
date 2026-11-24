# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct safetensors access to the GLM-5.3-Flash checkpoint (no transformers, no ttnn).

The checkpoint is fp8-e4m3 with 128x128 ``weight_scale_inv`` blocks for the fp8 groups; everything
else is bf16/fp32. Keys are the raw checkpoint names (Kimi-K3 KDA naming: ``q_conv1d``,
``f_a_proj``, ``A_log`` directly under ``self_attn``; ``hc_attn_fn`` etc.; per-expert
``mlp.experts.N.{gate,up,down}_proj``). Both the host reference and the TTNN weight loader read
through this module so they see identical dequantized tensors.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import torch

LAYER_PREFIX = "model.language_model.layers."
EMBED_KEY = "model.language_model.embed_tokens.weight"
FINAL_NORM_KEY = "model.language_model.norm.weight"
LM_HEAD_KEY = "lm_head.weight"
FP8_BLOCK = (128, 128)


WEIGHTS_DIR_ENV = "MODEL_WEIGHTS_DIR"  # tt-inference-server's flat weights directory inside the container


def missing_shards(snapshot: Path) -> list[str]:
    """The safetensors files the checkpoint's index names that ``snapshot`` lacks (empty = complete; no index =
    nothing to check)."""
    index = Path(snapshot) / "model.safetensors.index.json"
    if not index.exists():
        return []
    files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    return [f for f in files if not (Path(snapshot) / f).exists()]


def default_snapshot_dir() -> Path:
    """The checkpoint directory: ``MODEL_WEIGHTS_DIR`` (tt-inference-server downloads the weights into a flat
    directory and hands it to the container through that variable; 2026-10-31 the shield's server read the shared
    HF home's config-only snapshot instead and failed on the first weight), else the HF cache's ``refs/main``
    snapshot, else the last snapshot directory. A directory missing shards of its index is refused by name."""
    env = os.environ.get(WEIGHTS_DIR_ENV)
    if env and (Path(env) / "config.json").exists():
        d = Path(env)
    else:
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        repo = hub / "models--zai-org--GLM-5.3-Flash"
        ref = repo / "refs" / "main"
        d = None
        if ref.exists():
            cand = repo / "snapshots" / ref.read_text().strip()
            if (cand / "config.json").exists():
                d = cand
        if d is None:
            snaps = sorted((repo / "snapshots").glob("*"))
            if not snaps:
                raise FileNotFoundError(
                    f"GLM-5.3-Flash snapshot not found: no {WEIGHTS_DIR_ENV} with a config.json and no HF cache "
                    f"snapshot under {repo}"
                )
            d = snaps[-1]
    missing = missing_shards(d)
    if missing:
        raise FileNotFoundError(
            f"GLM-5.3-Flash checkpoint {d} lacks {len(missing)} of its index's shards "
            f"(first: {missing[:3]}): an incomplete download or a config-only snapshot"
        )
    return d


def shard_headers(snapshot: Path) -> dict[str, tuple[str, dict]]:
    """key -> (shard file, safetensors header entry) for every tensor in the checkpoint."""
    out = {}
    for f in sorted(Path(snapshot).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = (f.name, v)
    return out


def load_tensors(snapshot: Path, keys: list[str], headers: dict[str, tuple[str, dict]]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    by_file: dict[str, list[str]] = {}
    for k in keys:
        by_file.setdefault(headers[k][0], []).append(k)
    out = {}
    for fname, ks in by_file.items():
        with safe_open(str(Path(snapshot) / fname), framework="pt", device="cpu") as f:
            for k in ks:
                out[k] = f.get_tensor(k)
    return out


def dequantize_weight_tensor(
    tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape=FP8_BLOCK, *, dtype=torch.bfloat16
) -> torch.Tensor:
    """fp8 block-scaled -> dtype. Same arithmetic as
    models/demos/deepseek_v3/utils/hf_model_utils.dequantize_weight_tensor (inlined so this file
    runs in a plain torch venv without the ttnn tree importable)."""
    assert tensor.ndim == inv_scale.ndim == len(block_shape)
    original_shape = tuple(tensor.shape)
    padded_shape = tuple(inv_scale.shape[i] * int(block_shape[i]) for i in range(tensor.ndim))
    assert all(p >= o for p, o in zip(padded_shape, original_shape)), (tensor.shape, inv_scale.shape)
    original_slices = tuple(slice(0, size) for size in original_shape)
    out = tensor.float()
    if padded_shape != original_shape:
        padded = torch.zeros(padded_shape, dtype=out.dtype)
        padded[original_slices] = out
        out = padded
    interleaved_shape, scale_broadcast_shape = [], []
    for dim, block_dim in enumerate(block_shape):
        interleaved_shape.extend([inv_scale.shape[dim], int(block_dim)])
        scale_broadcast_shape.extend([inv_scale.shape[dim], 1])
    out_view = out.reshape(*interleaved_shape)
    out_view.mul_(inv_scale.float().reshape(*scale_broadcast_shape))
    out = out_view.reshape(*padded_shape)
    return out[original_slices].to(dtype).contiguous()


def load_layer_raw(snapshot: Path, layer_idx: int, headers=None, *, name_filter=None) -> dict[str, torch.Tensor]:
    """Raw checkpoint-named tensors of one layer (prefix stripped), fp8 groups dequantized to bf16.

    ``name_filter(name) -> bool`` restricts which (prefix-stripped) tensors are loaded, e.g. to skip
    the routed experts when only the attention path is needed.
    """
    headers = headers or shard_headers(snapshot)
    prefix = f"{LAYER_PREFIX}{layer_idx}."
    names = [k[len(prefix) :] for k in headers if k.startswith(prefix)]
    if name_filter is not None:
        names = [n for n in names if name_filter(n.replace(".weight_scale_inv", ".weight"))]
    raw = load_tensors(snapshot, [prefix + n for n in names], headers)
    out: dict[str, torch.Tensor] = {}
    for k, t in raw.items():
        name = k[len(prefix) :]
        if name.endswith(".weight_scale_inv"):
            continue
        if t.dtype == torch.float8_e4m3fn:
            inv = raw[k.replace(".weight", ".weight_scale_inv")]
            t = dequantize_weight_tensor(t, inv, FP8_BLOCK, dtype=torch.bfloat16)
        out[name] = t
    return out


def load_embedding(snapshot: Path, headers=None) -> torch.Tensor:
    headers = headers or shard_headers(snapshot)
    return load_tensors(snapshot, [EMBED_KEY], headers)[EMBED_KEY]
