# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Wan2.2-S2V-14B checkpoint loading + naming translation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

try:
    from safetensors.torch import load_file as _load_safetensors_file
except ImportError:  # pragma: no cover - safetensors should be available with diffusers
    _load_safetensors_file = None


# ---------------------------------------------------------------------------
# Snapshot + state-dict loading
# ---------------------------------------------------------------------------


def find_s2v_snapshot(model_id: str = "Wan-AI/Wan2.2-S2V-14B") -> Path:
    """Return the local snapshot for the S2V model, auto-downloading if absent.

    Matches the auto-download behaviour of ``Auto*.from_pretrained`` used for
    the T2V aux checkpoint. Set ``HF_HUB_OFFLINE=1`` to skip the network call
    when the snapshot is already cached.
    """
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_id))


def load_s2v_config(snapshot_dir: Path | str) -> dict[str, Any]:
    """Parse the S2V model's ``config.json`` into a plain dict."""
    cfg_path = Path(snapshot_dir) / "config.json"
    with cfg_path.open() as f:
        return json.load(f)


def load_s2v_state_dict(snapshot_dir: Path | str) -> dict[str, torch.Tensor]:
    """Merge the 4 safetensors shards into a single CPU ``state_dict`` (native naming)."""
    if _load_safetensors_file is None:
        raise ImportError("safetensors is required to load Wan2.2-S2V-14B weights")

    snapshot_dir = Path(snapshot_dir)
    index_path = snapshot_dir / "diffusion_pytorch_model.safetensors.index.json"
    with index_path.open() as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    merged: dict[str, torch.Tensor] = {}
    for shard, _keys in sorted(shard_to_keys.items()):
        shard_path = snapshot_dir / shard
        shard_dict = _load_safetensors_file(str(shard_path))
        # Only keep the keys the index says belong to this shard (the
        # safetensors files may contain extras for sharing/dedup).
        for k in _keys:
            merged[k] = shard_dict[k]

    if len(merged) != len(weight_map):
        missing = set(weight_map) - set(merged)
        raise RuntimeError(
            f"Loaded {len(merged)} keys but expected {len(weight_map)}; "
            f"missing {len(missing)} (first few: {sorted(missing)[:5]})"
        )

    return merged


# Native WanModel_S2V key → Diffusers-layout key. The receiving modules'
# ``_prepare_torch_state`` methods handle all tensor-side reshaping.

_FLAT_RENAMES: dict[str, str] = {
    "head.head.weight": "proj_out.weight",
    "head.head.bias": "proj_out.bias",
    "head.modulation": "scale_shift_table",
    "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
    "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
    "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
    "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
    "time_projection.1.weight": "condition_embedder.time_proj.weight",
    "time_projection.1.bias": "condition_embedder.time_proj.bias",
    "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
    "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
    "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
    "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
    "casual_audio_encoder.weights": "audio_encoder.weights",
    # cond_encoder is an on-device WanPatchEmbed; its `_prepare_torch_state`
    # consumes the raw Conv3d ``weight``/``bias`` and reshapes/permutes them.
    "cond_encoder.weight": "cond_encoder.weight",
    "cond_encoder.bias": "cond_encoder.bias",
    # trainable_cond_mask is an on-device Parameter shape [3, dim].
    "trainable_cond_mask.weight": "trainable_cond_mask",
}


# Inside a block, ref.norm3 ↔ tt.norm2 (the cross-attn pre-norm). Other norms
# are no-affine so they don't appear in the state dict.

_ATTN_SUFFIX_RENAMES: dict[str, str] = {
    "q.weight": "to_q.weight",
    "q.bias": "to_q.bias",
    "k.weight": "to_k.weight",
    "k.bias": "to_k.bias",
    "v.weight": "to_v.weight",
    "v.bias": "to_v.bias",
    "o.weight": "to_out.0.weight",
    "o.bias": "to_out.0.bias",
    "norm_q.weight": "norm_q.weight",
    "norm_k.weight": "norm_k.weight",
}

# blocks.{i}.<piece>.<rest>  e.g.  blocks.7.self_attn.q.weight
_BLOCKS_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
# audio_injector.injector.{i}.<rest>  e.g.  audio_injector.injector.3.q.weight
_AI_INJECTOR_RE = re.compile(r"^audio_injector\.injector\.(\d+)\.(.+)$")
# audio_injector.injector_adain_layers.{i}.<rest>
_AI_ADAIN_RE = re.compile(r"^audio_injector\.injector_adain_layers\.(\d+)\.(.+)$")


def _translate_attn_suffix(piece: str, rest: str) -> str | None:
    target_attn = "attn1" if piece == "self_attn" else "attn2"
    if rest in _ATTN_SUFFIX_RENAMES:
        return f"{target_attn}.{_ATTN_SUFFIX_RENAMES[rest]}"
    return None


def _translate_block_key(idx: int, rest: str) -> str | None:
    if rest == "modulation":
        return f"blocks.{idx}.scale_shift_table"
    if rest in ("norm3.weight", "norm3.bias"):
        return f"blocks.{idx}.norm2.{rest.split('.', 1)[1]}"
    if rest.startswith("ffn.0."):
        return f"blocks.{idx}.ffn.net.0.proj.{rest.split('.', 2)[2]}"
    if rest.startswith("ffn.2."):
        return f"blocks.{idx}.ffn.net.2.{rest.split('.', 2)[2]}"
    if rest.startswith("self_attn.") or rest.startswith("cross_attn."):
        piece, sub = rest.split(".", 1)
        tt_suffix = _translate_attn_suffix(piece, sub)
        return f"blocks.{idx}.{tt_suffix}" if tt_suffix else None
    return None


def _translate_audio_injector_key(idx: int, rest: str) -> str | None:
    if rest in _ATTN_SUFFIX_RENAMES:
        return f"audio_injector.injector.{idx}.{_ATTN_SUFFIX_RENAMES[rest]}"
    return None


def translate_s2v_state_dict(ref_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate a Wan2.2-S2V-14B native state_dict to tt_dit's naming."""
    out: dict[str, torch.Tensor] = {}

    for key, tensor in ref_state_dict.items():
        # 1. Flat renames.
        if key in _FLAT_RENAMES:
            out[_FLAT_RENAMES[key]] = tensor
            continue

        # 3. patch_embedding.{weight,bias} pass through.
        if key in ("patch_embedding.weight", "patch_embedding.bias"):
            out[key] = tensor
            continue

        # 3b. frame_packer.{proj,proj_2x,proj_4x}.{weight,bias} pass through.
        # FramePackMotionerWan's WanPatchEmbed children consume the raw
        # Conv3d weight/bias via their own ``_prepare_torch_state``.
        if key.startswith("frame_packer."):
            out[key] = tensor
            continue

        # 4. casual_audio_encoder.encoder.* → audio_encoder.encoder.* (no
        #    further suffix changes; the CausalConv1d / final_linear /
        #    padding_tokens substates all have matching names in tt_dit).
        if key.startswith("casual_audio_encoder.encoder."):
            out["audio_encoder.encoder." + key[len("casual_audio_encoder.encoder.") :]] = tensor
            continue

        # 5. blocks.{i}.<rest>
        m = _BLOCKS_RE.match(key)
        if m is not None:
            idx, rest = int(m.group(1)), m.group(2)
            translated = _translate_block_key(idx, rest)
            if translated is None:
                msg = f"unrecognized block key suffix: {key!r}"
                raise KeyError(msg)
            out[translated] = tensor
            continue

        # 6. audio_injector.injector.{i}.<rest>
        m = _AI_INJECTOR_RE.match(key)
        if m is not None:
            idx, rest = int(m.group(1)), m.group(2)
            translated = _translate_audio_injector_key(idx, rest)
            if translated is None:
                msg = f"unrecognized audio_injector.injector key suffix: {key!r}"
                raise KeyError(msg)
            out[translated] = tensor
            continue

        # 7. audio_injector.injector_adain_layers.{i}.linear.{weight,bias} —
        #    name matches tt_dit; just copy through.
        m = _AI_ADAIN_RE.match(key)
        if m is not None:
            idx, rest = int(m.group(1)), m.group(2)
            if rest in ("linear.weight", "linear.bias"):
                out[key] = tensor
                continue
            msg = f"unrecognized audio_injector.injector_adain_layers key: {key!r}"
            raise KeyError(msg)

        msg = f"no translation rule for reference key {key!r}"
        raise KeyError(msg)

    return out
