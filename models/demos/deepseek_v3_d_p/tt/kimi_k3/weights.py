# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 checkpoint keys, in the shape the TT modules read.

`utils/transformer_helpers.py::extract_layer_state_dict` cannot be extended to K3 and has to be
replaced for it: it is written against DeepSeek's `layers.{i}.mlp.*` / `layers.{i}.self_attn.*`
layout, assumes every layer is MLA, and emits none of `latent_weights`, `g_proj`, the KDA tensors or
the AttnRes queries. K3 renames the MoE module (`block_sparse_moe`), renames the routed experts
(`w1`/`w3`/`w2` rather than gate/up/down), adds a low-rank latent pair around them, and puts a
different attention in 69 of its 93 layers.

**Two checkpoints, two key roots.** The published MXFP4 checkpoint is a multimodal wrapper and
spells everything `language_model.model.…`; the dequantized export strips the wrapper and uses
`model.…`. Only the dequantized one loads end to end — nothing in this repo dequantizes MXFP4 — but
the quantized one is the right source for anything `quantization_config.ignore` covers (all of
`self_attn`, the shared expert, layer 0's dense MLP, `lm_head`), which is why both must work. The
root is read off the index rather than guessed, and `language_model.model.` is tried first because
`model.` is its suffix and would otherwise match the wrapped keys too.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import load_kda_layer_state_dict, resolve_model_root

# What each TT module reads out of a layer's state dict. Kept here rather than inline so the map
# from checkpoint name to TT name is in one place and can be read against the modules.
LAYER_NORM_KEYS = {
    "attn_norm_weight": "input_layernorm.weight",
    "ffn_norm_weight": "post_attention_layernorm.weight",
}

# `TtFfn` (layer 0's dense MLP) and `TtSharedExpert` both take gate/up/down under these names.
DENSE_FFN_KEYS = {
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}

# ttMLA's seven dense weights, plus `g_proj` — K3 sets `mla_use_output_gate`, and `ttMLA` only looks
# for `g_proj` when it does. `TtPrefillBlock.check_cache_complete` does NOT pass has_output_gate, so
# a K3 cache missing g_proj reports complete and the layer silently loads a placeholder; the K3
# stack has to check it itself.
MLA_KEYS = {
    "q_a_proj.weight": "self_attn.q_a_proj.weight",
    "q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
    "q_b_proj.weight": "self_attn.q_b_proj.weight",
    "kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight": "self_attn.kv_a_layernorm.weight",
    "kv_b_proj.weight": "self_attn.kv_b_proj.weight",
    "o_proj.weight": "self_attn.o_proj.weight",
    "g_proj.weight": "self_attn.g_proj.weight",
}

GATE_KEYS = {
    "weight": "block_sparse_moe.gate.weight",
    "e_score_correction_bias": "block_sparse_moe.gate.e_score_correction_bias",
}

SHARED_EXPERT_KEYS = {
    "gate_proj": "block_sparse_moe.shared_experts.gate_proj.weight",
    "up_proj": "block_sparse_moe.shared_experts.up_proj.weight",
    "down_proj": "block_sparse_moe.shared_experts.down_proj.weight",
}

# LatentMoE's low-rank pair around the routed experts. `down_proj` takes 7168 -> 3584 on the way in
# and `up_proj` 3584 -> 7168 on the way out, so the checkpoint's [3584, 7168] and [7168, 3584] are
# already in torch's (out, in) convention and need no transpose.
LATENT_KEYS = {
    "down_proj": "block_sparse_moe.routed_expert_down_proj.weight",
    "up_proj": "block_sparse_moe.routed_expert_up_proj.weight",
    "norm": "block_sparse_moe.routed_expert_norm.weight",
}

# The checkpoint's routed experts are w1/w3/w2, not gate/up/down. `KimiBlockSparseMLP.forward` is
# `w2(act(w1(h)) * w3(h))`, so w1 is the gate, w3 the up and w2 the down.
ROUTED_EXPERT_KEYS = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


def layer_prefix(checkpoint_dir: Path, layer_idx: int) -> str:
    return f"{resolve_model_root(Path(checkpoint_dir))}layers.{layer_idx}."


def load_tensors(checkpoint_dir: Path, names: dict[str, str]) -> dict[str, torch.Tensor]:
    """Read `{alias: full checkpoint key}`, opening each shard exactly once."""
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "model.safetensors.index.json").open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]

    missing = sorted(key for key in names.values() if key not in weight_map)
    if missing:
        raise ValueError(f"{checkpoint_dir} index is missing {len(missing)} weights, e.g. {missing[:3]}")

    by_shard: dict[str, list[tuple[str, str]]] = {}
    for alias, key in names.items():
        by_shard.setdefault(weight_map[key], []).append((alias, key))

    tensors: dict[str, torch.Tensor] = {}
    for shard, entries in by_shard.items():
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
            for alias, key in entries:
                tensors[alias] = handle.get_tensor(key)
    return tensors


def load_layer_state_dict(
    checkpoint_dir: Path,
    layer_idx: int,
    *,
    model_cfg: type = KimiK3Config,
    kda_checkpoint_dir: Path | None = None,
) -> dict:
    """One Kimi-K3 layer in the shape `TtKimiK3Block` reads.

    `kda_checkpoint_dir` exists because the two checkpoints are not interchangeable per-tensor: the
    routed experts are MXFP4 in the published one and only the dequantized export has them as bf16,
    while everything `quantization_config.ignore` covers is bf16 in both. A caller with both on the
    box can take the attention from whichever is closer to hand.
    """
    checkpoint_dir = Path(checkpoint_dir)
    prefix = layer_prefix(checkpoint_dir, layer_idx)
    is_mla = layer_idx in model_cfg.mla_layer_ids()
    is_moe = layer_idx >= model_cfg.NUM_DENSE_LAYERS

    wanted = {alias: prefix + key for alias, key in LAYER_NORM_KEYS.items()}
    if is_mla:
        wanted.update({f"mla::{alias}": prefix + key for alias, key in MLA_KEYS.items()})
    if is_moe:
        wanted.update({f"gate::{alias}": prefix + key for alias, key in GATE_KEYS.items()})
        wanted.update({f"shared::{alias}": prefix + key for alias, key in SHARED_EXPERT_KEYS.items()})
        wanted.update({f"latent::{alias}": prefix + key for alias, key in LATENT_KEYS.items()})
    else:
        wanted.update({f"ffn::{alias}": prefix + key for alias, key in DENSE_FFN_KEYS.items()})

    flat = load_tensors(checkpoint_dir, wanted)

    def group(tag: str) -> dict[str, torch.Tensor]:
        return {alias.split("::", 1)[1]: tensor for alias, tensor in flat.items() if alias.startswith(f"{tag}::")}

    state_dict: dict = {
        "attn_norm_weight": flat["attn_norm_weight"],
        "ffn_norm_weight": flat["ffn_norm_weight"],
    }
    if is_mla:
        state_dict["mla_weights"] = group("mla")
    else:
        state_dict["kda_weights"] = load_kda_layer_state_dict(
            Path(kda_checkpoint_dir or checkpoint_dir), layer_idx, kimi_k3_kda_config()
        )
    if is_moe:
        state_dict["gate_weights"] = group("gate")
        state_dict["shared_expert_weights"] = group("shared")
        state_dict["latent_weights"] = group("latent")
    else:
        state_dict["ffn_weights"] = group("ffn")
    return state_dict


def load_routed_expert_weights(checkpoint_dir: Path, layer_idx: int, num_experts: int) -> list[dict]:
    """The layer's routed experts, renamed from the checkpoint's `w1`/`w3`/`w2`.

    Held apart from `load_layer_state_dict` because this is the whole cost of a K3 layer: 896
    experts at 33.0 M parameters each is ~59 GB of bf16 for ONE layer, against a few hundred MB for
    everything else in the layer put together. A caller converting layer by layer wants to decide
    when to pay it.
    """
    prefix = layer_prefix(Path(checkpoint_dir), layer_idx)
    wanted = {
        f"{expert}::{alias}": f"{prefix}block_sparse_moe.experts.{expert}.{ckpt}.weight"
        for expert in range(num_experts)
        for alias, ckpt in ROUTED_EXPERT_KEYS.items()
    }
    flat = load_tensors(Path(checkpoint_dir), wanted)
    return [{alias: flat[f"{expert}::{alias}"] for alias in ROUTED_EXPERT_KEYS} for expert in range(num_experts)]


# --- TTNN weight cache -----------------------------------------------------------------------
#
# Building a Kimi-K3 layer from the checkpoint costs ~59 GB of host reads for its 896 routed experts
# plus a bfloat4_b conversion of 28 experts per chip, and the depth ladder pays it again at every
# rung: L12 spends over an hour on weights before the first token moves. Every component in the
# stack already accepts `weight_cache_path` and already writes its tensorbins when given one, so the
# only missing piece is knowing whether a layer's cache is COMPLETE — and therefore whether the
# checkpoint needs opening at all.
#
# The obvious implementation, composing each component's own `check_cache_complete`, is the one to
# avoid. Those checks are per-component and easy to get subtly wrong, and being wrong is not a
# failure but a silently wrong model: a cache missing MLA's `g_proj` reports complete and the layer
# loads a `torch.empty` placeholder. That exact bug was already found once in this bring-up.
#
# So completeness is recorded rather than inferred. A layer's marker is written only after the layer
# has been built end to end from real weights, which is the only moment its cache is known good. The
# marker path carries the checkpoint identity and the mesh geometry, so a different checkpoint, mesh
# or TP axis simply misses and rebuilds instead of loading someone else's tensors.

_CACHE_ENV = "TT_KIMI_K3_PREFILL_TTNN_CACHE"

# Alongside the other prefill caches on the models share, next to `glm52_ttnn_cache` and the GLM and
# DeepSeek caches. `/mnt/models` itself is group-writable only by local-syseng, so this sits in
# `deepseek-prefill-cache/`, which is where the golden traces and the other models' TTNN caches
# already live. This is a DEFAULT rather than a requirement: the point of a cache is that every run
# benefits without anyone remembering an environment variable, and
# `$TT_KIMI_K3_PREFILL_TTNN_CACHE` still relocates it (or disables it, if set empty).
_DEFAULT_CACHE = Path("/mnt/models/deepseek-prefill-cache/kimi-k3-ttnn-cache")


def _checkpoint_identity(checkpoint_dir: Path) -> str:
    """A short stable id for a checkpoint, from its index rather than its 5.5 TB of tensors."""
    index = Path(checkpoint_dir) / "model.safetensors.index.json"
    digest = hashlib.sha256()
    digest.update(Path(checkpoint_dir).resolve().name.encode())
    stat = index.stat()
    digest.update(str(stat.st_size).encode())
    return digest.hexdigest()[:16]


def cache_root(checkpoint_dir: Path, mesh_shape: tuple[int, int], tp_axis: int) -> Path | None:
    """Where this (checkpoint, mesh, tp_axis) keeps its tensorbins, or None if caching is off.

    Defaults to the models share so runs pick it up with no configuration; set
    `$TT_KIMI_K3_PREFILL_TTNN_CACHE` to relocate it, or to the empty string to disable caching. A
    None return is always safe to pass straight through as `weight_cache_path`.
    """
    override = os.getenv(_CACHE_ENV)
    if override == "":
        return None  # explicitly empty disables caching, for a run that must build from weights
    root = Path(override) if override else _DEFAULT_CACHE
    rows, columns = mesh_shape
    path = (
        root
        / f"kimi_k3_bh_{rows * columns}dev"
        / _checkpoint_identity(checkpoint_dir)
        / f"mesh{rows}x{columns}.tpaxis{tp_axis}"
    )
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        # An unwritable share should slow a run down, not fail it.
        logger.warning(f"KDA/K3 weight cache unavailable at {path} ({error}); building from the checkpoint")
        return None
    return path


def _marker(root: Path, layer_idx: int) -> Path:
    return root / f".layer_{layer_idx}.complete"


def layer_is_cached(root: Path | None, layer_idx: int) -> bool:
    return root is not None and _marker(root, layer_idx).is_file()


def mark_layer_cached(root: Path | None, layer_idx: int) -> None:
    """Record that this layer's cache is complete. Call ONLY after building it from real weights."""
    if root is not None:
        _marker(root, layer_idx).touch()


def load_layer_state_dict_cached(
    checkpoint_dir: Path,
    layer_idx: int,
    root: Path | None,
    *,
    model_cfg: type = KimiK3Config,
    num_routed_experts: int | None = None,
) -> dict:
    """This layer's weights, or an empty dict when its cache is already complete.

    An empty dict is the whole mechanism: every component reads its weights with
    `state_dict.get(...)`, and each falls back to its tensorbins when the entry is missing and a
    `weight_cache_path` is set. Nothing in the stack needed changing for this.
    """
    if layer_is_cached(root, layer_idx):
        logger.info(f"layer {layer_idx}: cache hit, not opening the checkpoint")
        return {}
    state_dict = load_layer_state_dict(checkpoint_dir, layer_idx, model_cfg=model_cfg)
    if layer_idx >= model_cfg.NUM_DENSE_LAYERS:
        experts = num_routed_experts if num_routed_experts is not None else model_cfg.NUM_ROUTED_EXPERTS
        logger.info(f"layer {layer_idx}: cache miss, reading {experts} routed experts (~59 GB)")
        state_dict["routed_expert_weights"] = load_routed_expert_weights(checkpoint_dir, layer_idx, experts)
    return state_dict
