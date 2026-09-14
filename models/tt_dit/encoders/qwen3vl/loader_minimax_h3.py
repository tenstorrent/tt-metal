# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3's Qwen3-VL text conditioner: config, truncated depth, and weight loading.

H3 conditions on ``hidden_states[50]`` of a 64-layer Qwen3-VL, which is the *raw* output of
decoder layer 49 -- not the post-norm final state. So only 50 layers are built, without the
final norm. Truncating the stack to 50 layers and reading its normalized
output instead would be a different tensor; the diffusers reference raises rather than allow
that, and ``test_text_encoder_minimax_h3.py::test_minimax_h3_text_conditioner`` asserts the tap
differs from the post-norm state (by O(10^4), not by rounding).

T2VA is text-only, so none of the vision tower (``model.visual.*``, 27 blocks) is built or read.
FL2VA needs the tower: ``build_minimax_h3_vision_tower`` and ``load_minimax_h3_vision_state_dict``
below serve that path; the tower is ~595 M parameters against the conditioner's 32 B, and is
replicated rather than tensor-parallel.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from ...blocks.rope import RopeConfig
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ..transformer import TransformerEncoderConfig
from .model_qwen3vl import Qwen3VlEncoder
from .vision_qwen3vl import Qwen3VlVisionModel

# H3 reads `hidden_states[50]`; `hidden_states[0]` is the embedding output, so index 50 is the
# output of decoder layer 49 and a 50-layer stack read before its final norm is exactly that.
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


def minimax_h3_encoder_config(config: dict, *, num_layers: int) -> TransformerEncoderConfig:
    """The conditioner's architecture from the checkpoint's `text_config`, truncated to `num_layers`.

    Without the language-model head, which the checkpoint loader never reads.
    """
    rope = config.get("rope_scaling") or config.get("rope_parameters") or {}

    return TransformerEncoderConfig(
        vocab_size=config["vocab_size"],
        # Not hidden_size // num_attention_heads for this checkpoint: 5120 / 64 = 80, but the real
        # head_dim is 128 and q_proj is [8192, 5120].
        head_size=config["head_dim"],
        embed_size=config["hidden_size"],
        ff_size=config["intermediate_size"],
        num_layers=num_layers,
        num_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"],
        norm_eps=config["rms_norm_eps"],
        attn_qkv_bias=config.get("attention_bias", False),
        attn_out_bias=False,
        attn_qk_norm=True,
        rope_config=RopeConfig(
            # transformers >= 4.57 keeps rope_theta at the top of text_config; older layouts put it
            # inside rope_scaling. Accept both.
            theta=rope.get("rope_theta", config.get("rope_theta")),
            mrope_section=list(rope["mrope_section"]),
            mrope_interleaved=True,
        ),
        final_norm=False,
        final_linear=False,
    )


def load_minimax_h3_text_state_dict(weights_dir: str | os.PathLike, *, num_layers: int) -> dict[str, torch.Tensor]:
    """The `model.language_model.*` sub-tree, layers `[0, num_layers)`, in the encoder's keys.

    Reads only the shards that hold wanted tensors, so the vision tower, `lm_head` and the final
    `norm` the encoder is built without are never materialized -- with 50 of 64 layers that is
    ~50 GB of the checkpoint's 63 GB.
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
        if key == f"{_CHECKPOINT_PREFIX}norm.weight":
            continue  # the tap is the raw output of the last layer
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
                state[key] = handle.get_tensor(key)
    logger.info(
        f"MiniMax-H3 text encoder: {len(state)} tensors from {len(by_shard)} of "
        f"{len(set(weight_map.values()))} shards, {sum(t.numel() for t in state.values()) * 2 / 1e9:.1f} GB bf16"
    )
    return Qwen3VlEncoder.convert_state(state)


def build_minimax_h3_text_encoder(
    weights_dir: str | os.PathLike,
    *,
    mesh_device,
    parallel_config: EncoderParallelConfig,
    ccl_manager: CCLManager,
    num_layers: int = MINIMAX_H3_TEXT_ENCODER_LAYER,
    load_weights: bool = True,
) -> tuple[Qwen3VlEncoder, dict]:
    """Build the conditioner at truncated depth and load its weights.

    It ends without the final norm, so its output is the raw output of its last layer, i.e.
    `hidden_states[num_layers]` of the full model.

    Returns `(encoder, text_config)`.
    """
    config = minimax_h3_text_config(weights_dir)

    encoder = Qwen3VlEncoder(
        minimax_h3_encoder_config(config, num_layers=num_layers),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    if load_weights:
        state = load_minimax_h3_text_state_dict(weights_dir, num_layers=num_layers)
        # Strict: an unconsumed or missing key here is a real mapping bug, and this is the only
        # place it is cheap to catch.
        encoder.load_torch_state_dict(state)
        del state

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
    load_weights: bool = True,
) -> tuple[Qwen3VlVisionModel, dict]:
    """Build the released vision tower and load its weights. Returns `(tower, vision_config)`.

    No parallel config: the tower is **replicated**, not tensor-parallel. At ~1.2 GB bf16 against the
    conditioner's ~50 GB it is not worth sharding, and it runs once per request outside the denoise
    loop. Every config value is read from the checkpoint rather than defaulted, because two of them are
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
    )
    if load_weights:
        # Strict: `pos_embed.weight` is popped to the host by `_prepare_torch_state` and every other
        # key must map, so an unconsumed one is a real mapping bug.
        tower.load_torch_state_dict(load_minimax_h3_vision_state_dict(weights_dir))
    return tower, config
