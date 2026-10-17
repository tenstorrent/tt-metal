# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein's Qwen3-VL-4B text encoder: config, 32-layer depth, feature taps and weight loading.

The vendored reference (``reference/flux3_klein/text_encoder.py``) runs ``Qwen3VLForConditionalGeneration`` with
``output_hidden_states=True`` and concatenates ``hidden_states[k]`` for ``k in OUTPUT_LAYERS = (4, 8, ..., 32)``.
``hidden_states[0]`` is the embedding output and ``hidden_states[k]`` the RAW output of decoder layer ``k - 1`` (the
forward-hook output, no norm; measured on the real model for every k, ``docs/M5_DESIGN.md`` decision 2). So the port
builds decoder layers ``0..31`` only (``FLUX3_KLEIN_TEXT_ENCODER_LAYERS``) and taps them with
``activation_layers=(3, 7, ..., 31)`` (``FLUX3_KLEIN_ACTIVATION_LAYERS = tuple(k - 1 for k in OUTPUT_LAYERS)``), which
``Qwen3VlTextEncoder`` returns without the final norm. Layers 32-35 and the final norm are never evaluated by the
reference for these taps, so they are never built.

Weights (``text_encoder/model.safetensors.index.json``, 713 keys in 2 shards) are routed EXPLICITLY, nothing is
silently ignored (:func:`flux3_klein_text_state_keys` raises on any key outside these groups):

* kept (354): ``model.language_model.{embed_tokens.weight, norm.weight, layers.{0..31}.*}`` (11 tensors per layer),
  prefix stripped; ``Qwen3VlAttention._prepare_torch_state`` fuses q/k/v so the strict load consumes them into 290
  module parameters (3,618,744,832 bf16 elements, 7.24 GB). ``norm.weight`` is loaded because the module owns the
  final norm and the load is strict; it is never applied (the taps bypass it).
* dropped (44): ``model.language_model.layers.{32..35}.*`` -- layers the taps never reach; the dropped indices must be
  exactly ``range(32, config.num_hidden_layers)`` and the kept ones exactly ``range(32)`` (an extra, renumbered or missing
  layer raises).
* dropped (315): ``model.visual.*`` -- the vision tower; text-only prompts never use it.
* ``lm_head.weight`` is absent from the index (``tie_word_embeddings: true``); it would be reported if it appeared.

Device weights go through ``utils.cache.load_model`` under ``flux3_klein_text_encoder_<blob[:16]>/text_encoder_L32``
with ``blob`` = the HuggingFace blob name of shard 00001 (the base and distilled snapshots link the SAME text_encoder
blobs, so one cache entry per mesh layout serves both).

Modelled on ``loader_minimax_h3.py`` (the precedent for Qwen3-VL as a DiT text conditioner).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...reference.flux3_klein.text_encoder import OUTPUT_LAYERS
from ...utils import cache
from .model_qwen3vl import Qwen3VlTextEncoder

# Decoder layers 0..31 only: the deepest tap, hidden_states[32], is the output of layer 31.
FLUX3_KLEIN_TEXT_ENCODER_LAYERS = 32
# hidden_states[k] == output of 0-based decoder layer k - 1 (hidden_states[0] is the embedding).
FLUX3_KLEIN_ACTIVATION_LAYERS = tuple(k - 1 for k in OUTPUT_LAYERS)
# Not hidden_size // num_attention_heads (2560 / 32 = 80): the checkpoint's heads are 128 wide (q_proj [4096, 2560]).
FLUX3_KLEIN_TEXT_HEAD_DIM = 128
FLUX3_KLEIN_TEXT_SUBFOLDER = "text_encoder"
_CHECKPOINT_PREFIX = "model.language_model."
_VISION_PREFIX = "model.visual."
_CACHE_SUBFOLDER = "text_encoder_L32"
_FIRST_SHARD = "model-00001-of-00002.safetensors"
_INDEX_FILE = "model.safetensors.index.json"
_LAYER_RE = re.compile(rf"^{re.escape(_CHECKPOINT_PREFIX)}layers\.(\d+)\.")


def _text_encoder_dir(snapshot: str | os.PathLike) -> Path:
    return Path(snapshot) / FLUX3_KLEIN_TEXT_SUBFOLDER


def flux3_klein_text_config(snapshot: str | os.PathLike) -> dict:
    """``snapshot/text_encoder/config.json``'s ``text_config`` as plain JSON (no ``AutoConfig``: a dozen integers must
    not tie this to a transformers version that knows the architecture), validated by :func:`check_flux3_klein_text_config`.
    """
    path = _text_encoder_dir(snapshot) / "config.json"
    return check_flux3_klein_text_config(json.loads(path.read_text())["text_config"], where=str(path))


def check_flux3_klein_text_config(config: dict, *, where: str = "text_config") -> dict:
    """Raise ``ValueError`` unless ``config`` has the facts the port relies on: more than 31 decoder layers (the deepest
    tap), ``head_dim`` 128, ``rope_scaling.mrope_section`` ``[24, 20, 20]``, tied embeddings (so ``lm_head.weight`` is
    absent) and a ``rope_theta`` (top level or inside ``rope_scaling``). Returns ``config`` unchanged."""
    rope_scaling = config.get("rope_scaling") or {}
    problems = []
    if config.get("num_hidden_layers", 0) <= FLUX3_KLEIN_ACTIVATION_LAYERS[-1]:
        problems.append(
            f"num_hidden_layers {config.get('num_hidden_layers')} cannot reach tap layer {FLUX3_KLEIN_ACTIVATION_LAYERS[-1]}"
        )
    if config.get("head_dim") != FLUX3_KLEIN_TEXT_HEAD_DIM:
        problems.append(f"head_dim {config.get('head_dim')} != {FLUX3_KLEIN_TEXT_HEAD_DIM}")
    if list(rope_scaling.get("mrope_section", [])) != [24, 20, 20]:
        problems.append(f"rope_scaling.mrope_section {rope_scaling.get('mrope_section')} != [24, 20, 20]")
    if config.get("tie_word_embeddings") is not True:
        problems.append("tie_word_embeddings is not true (lm_head.weight would then be a separate, unrouted key)")
    if "rope_theta" not in config and "rope_theta" not in rope_scaling:
        problems.append("rope_theta missing from text_config and rope_scaling")
    if problems:
        raise ValueError(f"{where} is not the FLUX 3 Klein text encoder: " + "; ".join(problems))
    return config


def flux3_klein_rope_theta(text_config: dict) -> float:
    """``rope_theta`` from the top of ``text_config`` (transformers >= 4.57 layout) or inside ``rope_scaling``."""
    rope_scaling = text_config.get("rope_scaling") or {}
    return float(rope_scaling.get("rope_theta", text_config["rope_theta"]))


def flux3_klein_text_encoder_blob(snapshot: str | os.PathLike) -> str:
    """The HuggingFace blob name of ``text_encoder/model-00001-of-00002.safetensors``: the basename of the symlink's
    OWN target (in a hub cache the file's sha256), or ``name:size`` for a plain file."""
    path = _text_encoder_dir(snapshot) / _FIRST_SHARD
    if path.is_symlink():
        return Path(os.readlink(path)).name
    return f"{path.name}:{path.stat().st_size}"


@dataclass(frozen=True)
class Flux3KleinTextKeys:
    """The checkpoint keys of the text encoder, routed. Every key of the index is in exactly one tuple."""

    kept: tuple[str, ...]
    dropped_layers: tuple[str, ...]
    dropped_vision: tuple[str, ...]


def _weight_map(snapshot: str | os.PathLike) -> dict[str, str]:
    index_path = _text_encoder_dir(snapshot) / _INDEX_FILE
    if not index_path.is_file():
        raise FileNotFoundError(f"no {_INDEX_FILE} under {index_path.parent}")
    return json.loads(index_path.read_text())["weight_map"]


def route_flux3_klein_text_keys(
    keys, *, num_layers: int = FLUX3_KLEIN_TEXT_ENCODER_LAYERS, num_hidden_layers: int | None = None
) -> Flux3KleinTextKeys:
    """Route checkpoint ``keys`` into kept / dropped layers / dropped vision; raise ``ValueError`` naming every key
    that belongs to none of the groups (``lm_head.weight`` included, should it ever appear). With ``num_hidden_layers``
    (the config's depth) the layer indices are cross-checked too: kept must be exactly ``range(num_layers)`` and dropped
    exactly ``range(num_layers, num_hidden_layers)``, so an extra, renumbered or missing layer is reported instead of
    landing silently in the dropped group."""
    kept, dropped_layers, dropped_vision, unknown = [], [], [], []
    for key in keys:
        if key.startswith(_VISION_PREFIX):
            dropped_vision.append(key)
        elif key.startswith(_CHECKPOINT_PREFIX):
            match = _LAYER_RE.match(key)
            if match is not None:
                (kept if int(match.group(1)) < num_layers else dropped_layers).append(key)
            elif key in (f"{_CHECKPOINT_PREFIX}embed_tokens.weight", f"{_CHECKPOINT_PREFIX}norm.weight"):
                kept.append(key)
            else:
                unknown.append(key)
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(
            "FLUX 3 Klein text encoder: checkpoint keys outside the routed groups (language_model layers < "
            f"{num_layers} / embed_tokens / norm kept, layers >= {num_layers} and {_VISION_PREFIX}* dropped): {sorted(unknown)}"
        )
    if num_hidden_layers is not None:
        kept_indices = {int(m.group(1)) for m in map(_LAYER_RE.match, kept) if m is not None}
        dropped_indices = {int(_LAYER_RE.match(key).group(1)) for key in dropped_layers}
        if kept_indices != set(range(num_layers)):
            raise ValueError(
                f"FLUX 3 Klein text encoder: kept layer indices {sorted(kept_indices)} != range({num_layers})"
            )
        if dropped_indices != set(range(num_layers, num_hidden_layers)):
            raise ValueError(
                f"FLUX 3 Klein text encoder: dropped layer indices {sorted(dropped_indices)} != "
                f"range({num_layers}, {num_hidden_layers}) (config num_hidden_layers {num_hidden_layers})"
            )
    return Flux3KleinTextKeys(tuple(sorted(kept)), tuple(sorted(dropped_layers)), tuple(sorted(dropped_vision)))


def flux3_klein_text_state_keys(snapshot: str | os.PathLike) -> Flux3KleinTextKeys:
    """Dry run over ``model.safetensors.index.json`` (no tensor reads): the 354 kept, 44 dropped-layer and 315
    dropped-vision keys of the released checkpoint, layer indices cross-checked against the config's depth (36)."""
    return route_flux3_klein_text_keys(
        _weight_map(snapshot), num_hidden_layers=flux3_klein_text_config(snapshot)["num_hidden_layers"]
    )


def load_flux3_klein_text_state_dict(snapshot: str | os.PathLike) -> dict[str, torch.Tensor]:
    """The kept tensors, prefix stripped, read shard-wise with ``safe_open`` (both shards hold kept tensors:
    ``norm.weight`` and the upper layers live in 00002)."""
    directory = _text_encoder_dir(snapshot)
    weight_map = _weight_map(snapshot)
    keys = route_flux3_klein_text_keys(
        weight_map, num_hidden_layers=flux3_klein_text_config(snapshot)["num_hidden_layers"]
    )

    by_shard: dict[str, list[str]] = {}
    for key in keys.kept:
        by_shard.setdefault(weight_map[key], []).append(key)

    state: dict[str, torch.Tensor] = {}
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(str(directory / shard), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                state[key[len(_CHECKPOINT_PREFIX) :]] = handle.get_tensor(key)
    logger.info(
        f"FLUX 3 Klein text encoder: {len(state)} tensors kept from {len(by_shard)} shards "
        f"({sum(t.numel() for t in state.values()) * 2 / 1e9:.2f} GB bf16); dropped {len(keys.dropped_layers)} "
        f"layer >= {FLUX3_KLEIN_TEXT_ENCODER_LAYERS} keys and {len(keys.dropped_vision)} vision keys"
    )
    return state


def build_flux3_klein_text_encoder(
    snapshot: str | os.PathLike,
    *,
    mesh_device,
    parallel_config: EncoderParallelConfig,
    ccl_manager: CCLManager,
    is_fsdp: bool = True,
    load_weights: bool = True,
    high_fidelity_linears: bool = True,
    num_layers: int = FLUX3_KLEIN_TEXT_ENCODER_LAYERS,
    activation_layers=FLUX3_KLEIN_ACTIVATION_LAYERS,
    kv_gather_capacity: int | None = None,
    use_persistent_ccl_buffers: bool = True,
) -> tuple[Qwen3VlTextEncoder, dict]:
    """Build the 32-layer encoder tapped at layers (3, 7, ..., 31) and (optionally) load its weights.

    ``num_layers`` / ``activation_layers`` exist ONLY for the reduced-depth test rows (depth 2, taps (0, 1)).
    ``high_fidelity_linears`` defaults on: decision 10 (bf16 operands, fp32 accumulation) needs HiFi4; HiFi2 drops
    operand mantissa bits (measured rrmse vs the fp32-exact product 4.75e-3 at HiFi2, 1.08e-3 at HiFi4). ``is_fsdp``
    self-disables on a ``(1, N)`` mesh. Returns ``(encoder, text_config)``.
    """
    config = flux3_klein_text_config(snapshot)
    activation_layers = tuple(activation_layers)
    if any(layer < 0 or layer >= num_layers for layer in activation_layers):
        raise ValueError(f"activation_layers {activation_layers} outside the built layers [0, {num_layers})")

    encoder = Qwen3VlTextEncoder(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        hidden_act=config.get("hidden_act", "silu"),
        num_hidden_layers=num_layers,
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        rms_norm_eps=config["rms_norm_eps"],
        rope_theta=flux3_klein_rope_theta(config),
        mrope_section=config["rope_scaling"]["mrope_section"],
        head_dim=FLUX3_KLEIN_TEXT_HEAD_DIM,
        activation_layers=activation_layers,
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        is_fsdp=is_fsdp,
        high_fidelity_linears=high_fidelity_linears,
        kv_gather_capacity=kv_gather_capacity,
        use_persistent_ccl_buffers=use_persistent_ccl_buffers,
    )

    if load_weights:
        if num_layers != FLUX3_KLEIN_TEXT_ENCODER_LAYERS:
            raise ValueError(
                "the checkpoint loader serves the full 32-layer encoder only; reduced depths take random weights"
            )
        load_flux3_klein_text_encoder_weights(
            encoder, snapshot, parallel_config=parallel_config, mesh_device=mesh_device, is_fsdp=is_fsdp
        )

    return encoder, config


def load_flux3_klein_text_encoder_weights(
    encoder: Qwen3VlTextEncoder,
    snapshot: str | os.PathLike,
    *,
    parallel_config: EncoderParallelConfig,
    mesh_device,
    is_fsdp: bool = True,
) -> None:
    """Upload the 32-layer encoder through ``cache.load_model`` (blob-keyed; reload-safe via ``is_loaded()``)."""
    blob = flux3_klein_text_encoder_blob(snapshot)
    cache.load_model(
        encoder,
        model_name=f"flux3_klein_text_encoder_{blob[:16]}",
        subfolder=_CACHE_SUBFOLDER,
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        is_fsdp=is_fsdp,
        get_torch_state_dict=lambda: load_flux3_klein_text_state_dict(snapshot),
    )
