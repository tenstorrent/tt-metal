# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein text encoder: the Qwen3-VL-4B language model, layers 0..31, with the hidden states after layers
4, 8, ..., 32 concatenated into 20480-wide text tokens."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from .model_qwen3vl import Qwen3VlTextEncoder

NUM_LAYERS = 32
TAP_LAYERS = tuple(k - 1 for k in (4, 8, 12, 16, 20, 24, 28, 32))
SUBFOLDER = "text_encoder"
_CHECKPOINT_PREFIX = "model.language_model."


def flux3_klein_text_config(snapshot: str | os.PathLike) -> dict:
    return json.loads((Path(snapshot) / SUBFOLDER / "config.json").read_text())["text_config"]


def flux3_klein_rope_theta(config: dict) -> float:
    rope_scaling = config.get("rope_scaling") or {}
    return float(rope_scaling.get("rope_theta", config.get("rope_theta")))


def load_flux3_klein_text_state_dict(snapshot: str | os.PathLike) -> dict[str, torch.Tensor]:
    """The `model.language_model.*` sub-tree, layers [0, NUM_LAYERS), prefix stripped."""
    directory = Path(snapshot) / SUBFOLDER
    weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    layer_re = re.compile(rf"^{re.escape(_CHECKPOINT_PREFIX)}layers\.(\d+)\.")
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if not key.startswith(_CHECKPOINT_PREFIX):
            continue  # model.visual.*
        match = layer_re.match(key)
        if match is not None and int(match.group(1)) >= NUM_LAYERS:
            continue
        by_shard.setdefault(shard, []).append(key)

    state: dict[str, torch.Tensor] = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(directory / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                state[key[len(_CHECKPOINT_PREFIX) :]] = handle.get_tensor(key)
    logger.info(f"FLUX 3 Klein text encoder: {len(state)} tensors from {len(by_shard)} shards")
    return state


def build_flux3_klein_text_encoder(
    snapshot: str | os.PathLike,
    *,
    mesh_device,
    parallel_config: EncoderParallelConfig,
    ccl_manager: CCLManager,
    is_fsdp: bool = True,
) -> tuple[Qwen3VlTextEncoder, dict]:
    config = flux3_klein_text_config(snapshot)
    encoder = Qwen3VlTextEncoder(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        hidden_act=config.get("hidden_act", "silu"),
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        rms_norm_eps=config["rms_norm_eps"],
        rope_theta=flux3_klein_rope_theta(config),
        mrope_section=config["rope_scaling"]["mrope_section"],
        head_dim=config["head_dim"],
        activation_layers=TAP_LAYERS,
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        is_fsdp=is_fsdp,
        high_fidelity_linears=True,
    )
    # the first shard's blob name identifies the checkpoint revision
    shard = Path(snapshot) / SUBFOLDER / "model-00001-of-00002.safetensors"
    blob = Path(os.readlink(shard)).name if shard.is_symlink() else f"{shard.name}:{shard.stat().st_size}"
    cache.load_model(
        encoder,
        model_name=f"flux3_klein_text_encoder_{blob[:16]}",
        subfolder=f"text_encoder_L{NUM_LAYERS}",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        is_fsdp=is_fsdp,
        get_torch_state_dict=lambda: load_flux3_klein_text_state_dict(snapshot),
    )
    return encoder, config
