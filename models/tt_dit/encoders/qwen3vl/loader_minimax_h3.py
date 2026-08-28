# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3's Qwen3-VL text conditioner: config, truncated depth, and weight loading.

H3 conditions on ``hidden_states[50]`` of a 64-layer Qwen3-VL, which is the *raw* output of
decoder layer 49 -- not the post-norm final state. So only 50 layers are built and the tap is
taken with ``activation_layers``, which returns hidden states without the final norm. Truncating
the stack to 50 layers and reading its normalized output instead would be a different tensor;
the diffusers reference raises rather than allow that, and
``test_text_encoder_minimax_h3.py::test_minimax_h3_text_conditioner`` asserts the tap differs
from the post-norm state (by O(10^4), not by rounding).

T2VA is text-only, so none of the vision tower (``model.visual.*``, 27 blocks) is built or read,
and mRoPE degenerates: all three position axes carry the same ``arange``, which makes the
checkpoint's ``mrope_interleaved: true`` indistinguishable from the chunked section split
``create_rope_tensors`` implements. That is measured, not assumed --
``test_mrope_is_permutation_invariant_for_text_only``.

FL2VA needs the tower, and **that degeneracy is void the moment an image enters the prompt**: a
vision block carries genuinely different t/h/w positions, so ``mrope_interleaved`` becomes
load-bearing and callers must pass ``position_ids=mrope_position_ids(...), interleaved=True``.
``build_minimax_h3_vision_tower`` and ``load_minimax_h3_vision_state_dict`` below serve that path;
the tower is ~595 M parameters against the conditioner's 32 B, and is replicated rather than
tensor-parallel.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from ...parallel.manager import CCLManager
from ...utils import cache
from .model_qwen3vl import Qwen3VlTextEncoder
from .vision_qwen3vl import Qwen3VlVisionModel

# H3 reads `hidden_states[50]`; `hidden_states[0]` is the embedding output, so index 50 is the
# output of decoder layer 49 and a 50-layer stack tapped at its last layer is exactly that.
MINIMAX_H3_TEXT_ENCODER_LAYER = 50

_CHECKPOINT_PREFIX = "model.language_model."


def minimax_h3_text_config(weights_dir: str | os.PathLike) -> dict:
    """Read `text_encoder/config.json`'s `text_config` as plain JSON.

    Not `AutoConfig.from_pretrained`: that resolves `model_type: qwen3_vl`, which
    ties this to a transformers version that knows the architecture, when all that is needed are
    a dozen integers. The released values are hidden 5120, intermediate 25600, 64 heads, 8 KV
    heads, head_dim 128, rms_eps 1e-6, rope_theta 5e6, mrope_section [24, 20, 20].
    """
    config = json.loads((Path(weights_dir) / "config.json").read_text())["text_config"]
    num_layers = config["num_hidden_layers"]
    if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
        raise ValueError(
            f"MiniMax-H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}], which needs more "
            f"than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but this checkpoint has {num_layers}"
        )
    return config


def load_minimax_h3_text_state_dict(weights_dir: str | os.PathLike, *, num_layers: int) -> dict[str, torch.Tensor]:
    """The `model.language_model.*` sub-tree, layers `[0, num_layers)`, prefix stripped.

    Reads only the shards that hold wanted tensors, so the vision tower and `lm_head` are never
    materialized -- with 50 of 64 layers that is ~50 GB of the checkpoint's 63 GB. `norm.weight`
    *is* kept even though the tap bypasses the final norm: the module owns that parameter and the
    load is strict, so dropping it would fail as a missing key rather than save anything
    meaningful (one 5120-element vector).
    """
    directory = Path(weights_dir)
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no model.safetensors.index.json under {directory}")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    layer_re = re.compile(rf"^{re.escape(_CHECKPOINT_PREFIX)}layers\.(\d+)\.")
    wanted: dict[str, str] = {}
    for key, shard in weight_map.items():
        if not key.startswith(_CHECKPOINT_PREFIX):
            continue  # model.visual.* and lm_head.weight
        match = layer_re.match(key)
        if match is not None and int(match.group(1)) >= num_layers:
            continue  # layers 50..63 are never evaluated
        wanted[key] = shard

    by_shard: dict[str, list[str]] = {}
    for key, shard in wanted.items():
        by_shard.setdefault(shard, []).append(key)

    state: dict[str, torch.Tensor] = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(directory / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                state[key[len(_CHECKPOINT_PREFIX) :]] = handle.get_tensor(key)
    logger.info(
        f"MiniMax-H3 text encoder: {len(state)} tensors from {len(by_shard)} of "
        f"{len(set(weight_map.values()))} shards, {sum(t.numel() for t in state.values()) * 2 / 1e9:.1f} GB bf16"
    )
    return state


def build_minimax_h3_text_encoder(
    weights_dir: str | os.PathLike,
    *,
    mesh_device,
    parallel_config,
    ccl_manager: CCLManager,
    is_fsdp: bool = True,
    num_layers: int = MINIMAX_H3_TEXT_ENCODER_LAYER,
    load_weights: bool = True,
) -> tuple[Qwen3VlTextEncoder, dict]:
    """Build the conditioner at truncated depth, tapped at its last layer, and load its weights.

    `is_fsdp` defaults on: the encoder runs once per prompt, outside the denoise loop, so the
    per-layer weight gather costs nothing that matters, while sharding over the non-TP axis takes
    the resident weights from ~12.5 GB/device at TP=4 to ~1.6 GB/device across a 4x8 mesh. It
    self-disables when the non-TP axis is size 1.

    Returns `(encoder, text_config)`; the config carries `head_dim`, `rope_theta` and
    `mrope_section`, which the caller needs to build the rope tables at the matching width.
    """
    config = minimax_h3_text_config(weights_dir)
    rope_scaling = config.get("rope_scaling") or {}

    encoder = Qwen3VlTextEncoder(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        hidden_act=config.get("hidden_act", "silu"),
        num_hidden_layers=num_layers,
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        rms_norm_eps=config["rms_norm_eps"],
        # transformers >= 4.57 keeps rope_theta at the top of text_config; older layouts put it
        # inside rope_scaling. Accept both, as the Ideogram-4 pipeline does.
        rope_theta=rope_scaling.get("rope_theta", config["rope_theta"]),
        mrope_section=rope_scaling["mrope_section"],
        # Not hidden_size // num_attention_heads for this checkpoint: 5120 / 64 = 80, but the
        # real head_dim is 128 and q_proj is [8192, 5120].
        head_dim=config["head_dim"],
        # The raw output of the last built layer, i.e. hidden_states[50], with no final norm.
        activation_layers=(num_layers - 1,),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        is_fsdp=is_fsdp,
        # HiFi4 decoder linears, unconditionally: measured on the fl2va conditioner at production
        # shape and content, they take the fused-conditioner PCC from 70.89 % to 85.82 % and recover
        # massive-activation rows 102 and 128, at no measurable cost (1184.2 vs 1183.2 ms/forward).
        # Scoped here so other Qwen3-VL users (Ideogram-4) keep the tt_dit-wide default.
        high_fidelity_linears=True,
    )

    if load_weights:
        cache.load_model(
            encoder,
            model_name="minimax-h3",
            subfolder="text_encoder",
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            mesh_device=mesh_device,
            is_fsdp=is_fsdp,
            get_torch_state_dict=lambda: load_minimax_h3_text_state_dict(weights_dir, num_layers=num_layers),
        )

    return encoder, config


_VISION_PREFIX = "model.visual."


def minimax_h3_vision_config(weights_dir: str | os.PathLike) -> dict:
    """Read `text_encoder/config.json`'s `vision_config` as plain JSON.

    Released values: depth 27, hidden 1152, 16 heads, intermediate 4304, patch 16,
    `spatial_merge_size` 2, `temporal_patch_size` 2, `num_position_embeddings` 2304 (= 48^2),
    `out_hidden_size` 5120, `deepstack_visual_indexes` [8, 16, 24].
    """
    return json.loads((Path(weights_dir) / "config.json").read_text())["vision_config"]


def load_minimax_h3_vision_state_dict(weights_dir: str | os.PathLike) -> dict[str, torch.Tensor]:
    """The `model.visual.*` sub-tree, prefix stripped. ~595 M parameters, ~1.2 GB bf16.

    Reads only the shards holding vision tensors, the mirror of
    :func:`load_minimax_h3_text_state_dict`'s treatment of the decoder.
    """
    directory = Path(weights_dir)
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no model.safetensors.index.json under {directory}")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(_VISION_PREFIX):
            by_shard.setdefault(shard, []).append(key)
    if not by_shard:
        raise ValueError(f"no {_VISION_PREFIX}* tensors in {index_path}; this checkpoint has no vision tower")

    state: dict[str, torch.Tensor] = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(directory / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                state[key[len(_VISION_PREFIX) :]] = handle.get_tensor(key)
    logger.info(
        f"MiniMax-H3 vision tower: {len(state)} tensors from {len(by_shard)} of "
        f"{len(set(weight_map.values()))} shards, {sum(t.numel() for t in state.values()) * 2 / 1e9:.2f} GB bf16"
    )
    return state


def build_minimax_h3_vision_tower(
    weights_dir: str | os.PathLike,
    *,
    mesh_device,
    parallel_config=None,
    ccl_manager=None,
    load_weights: bool = True,
) -> tuple[Qwen3VlVisionModel, dict]:
    """Build the released vision tower and load its weights. Returns `(tower, vision_config)`.

    `parallel_config`/`ccl_manager` are forwarded to `Qwen3VlVisionModel`. Passing an
    `EncoderParallelConfig` with both `tensor_parallel` and `sequence_parallel` set (plus a
    `ccl_manager`) turns on the sharded path -- TP head fracturing, ring / windowed-SP attention, and
    the sequence-dim all-reduce. Leaving them `None` keeps the tower **replicated**: at ~1.2 GB bf16
    against the conditioner's ~50 GB it is cheap enough to replicate, and it runs once per request
    outside the denoise loop, so replication is a valid (and historically the default) choice.

    Every config value is read from the checkpoint rather than defaulted, because two of them are
    load-bearing and easy to get wrong silently -- `head_dim` is `1152 // 16 = 72`, which is not tile
    aligned and is padded to 96 internally with the softmax `scale` passed explicitly as `72 ** -0.5`,
    and `num_position_embeddings` is 2304 = 48^2, smaller than any production patch grid, so the
    bilinear interpolation of the position table is the common path rather than an edge case.
    """
    config = minimax_h3_vision_config(weights_dir)
    tower = Qwen3VlVisionModel(
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        depth=config["depth"],
        intermediate_size=config["intermediate_size"],
        in_channels=config.get("in_channels", 3),
        patch_size=config["patch_size"],
        temporal_patch_size=config.get("temporal_patch_size", 2),
        spatial_merge_size=config["spatial_merge_size"],
        num_position_embeddings=config["num_position_embeddings"],
        out_hidden_size=config["out_hidden_size"],
        hidden_act=config.get("hidden_act", "gelu_pytorch_tanh"),
        norm_eps=config.get("rms_norm_eps", 1e-6),
        deepstack_visual_indexes=config["deepstack_visual_indexes"],
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    if load_weights:
        # Strict: `pos_embed.weight` is popped to the host by `_prepare_torch_state` and every other
        # key must map, so an unconsumed one is a real mapping bug.
        tower.load_torch_state_dict(load_minimax_h3_vision_state_dict(weights_dir))
    return tower, config
