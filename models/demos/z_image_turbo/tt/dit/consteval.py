# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Const-eval functions for ZImageTransformer TTNN graph."""

import math

import torch

import ttnn


def _basic(input, device):
    x = ttnn.to_device(input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _permute(input, device):
    x = ttnn.to_device(input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.permute(x, [1, 0], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)


def _reshape_64_2(input, device):
    x = ttnn.to_device(input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(x, [1, 1, 1, 64, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _reshape_1_1_3840(input, device):
    x = ttnn.to_device(input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(x, [1, 1, 3840], memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _reshape_1_3840(input, device):
    x = ttnn.to_device(input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(x, [1, 3840], memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _reshape_1_2560(input, device):
    x = ttnn.to_device(input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(x, [1, 2560], memory_config=ttnn.DRAM_MEMORY_CONFIG)


# Per-block weight transform specs (suffix, transform_fn)
_BLOCK_WEIGHTS = [
    ("attention.norm_k.weight", _reshape_64_2),
    ("attention.norm_q.weight", _reshape_64_2),
    ("attention.to_k.weight", _basic),
    ("attention.to_out.0.weight", _basic),
    ("attention.to_q.weight", _basic),
    ("attention.to_v.weight", _basic),
    ("attention_norm1.weight", _reshape_1_3840),
    ("attention_norm2.weight", _reshape_1_1_3840),
    ("feed_forward.w1.weight", _basic),
    ("feed_forward.w2.weight", _basic),
    ("feed_forward.w3.weight", _basic),
    ("ffn_norm1.weight", _reshape_1_3840),
    ("ffn_norm2.weight", _reshape_1_1_3840),
]

_ADALN_WEIGHTS = [
    ("adaLN_modulation.0.bias", _basic),
    ("adaLN_modulation.0.weight", _permute),
]


def _consteval_block(weights, prefix, specs, device):
    for suffix, fn in specs:
        key = f"{prefix}.{suffix}"
        weights[key] = fn(weights[key], device)


def run_const_evals(weights, device):
    """Apply all consteval transformations to the weights dict in place."""
    mem = ttnn.DRAM_MEMORY_CONFIG
    replicated = ttnn.ReplicateTensorToMesh(device)

    def _sbf16(v):
        t = torch.tensor([[v]], dtype=torch.bfloat16)
        return ttnn.from_torch(
            t,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=device,
            memory_config=mem,
            mesh_mapper=replicated,
        )

    def _ids(vals):
        t = torch.tensor([vals], dtype=torch.int32)
        return ttnn.from_torch(
            t,
            dtype=ttnn.DataType.UINT32,
            layout=ttnn.Layout.ROW_MAJOR,
            device=device,
            memory_config=mem,
            mesh_mapper=replicated,
        )

    # ── Transformer layers (30) ───────────────────────────────────────────────
    for i in range(30):
        _consteval_block(weights, f"layers.{i}", _BLOCK_WEIGHTS + _ADALN_WEIGHTS, device)

    # ── Noise refiner layers (2) ──────────────────────────────────────────────
    for i in range(2):
        _consteval_block(weights, f"noise_refiner.{i}", _BLOCK_WEIGHTS + _ADALN_WEIGHTS, device)

    # ── Context refiner layers (2, no adaLN) ──────────────────────────────────
    for i in range(2):
        _consteval_block(weights, f"context_refiner.{i}", _BLOCK_WEIGHTS, device)

    # ── Timestep embedder ─────────────────────────────────────────────────────
    for i in (0, 2):
        weights[f"t_embedder.mlp.{i}.bias"] = _basic(weights[f"t_embedder.mlp.{i}.bias"], device)
        weights[f"t_embedder.mlp.{i}.weight"] = _permute(weights[f"t_embedder.mlp.{i}.weight"], device)

    # ── Unique weights ────────────────────────────────────────────────────────
    weights["all_final_layer.2-1.adaLN_modulation.1.bias"] = _basic(
        weights["all_final_layer.2-1.adaLN_modulation.1.bias"], device
    )
    weights["all_final_layer.2-1.adaLN_modulation.1.weight"] = _permute(
        weights["all_final_layer.2-1.adaLN_modulation.1.weight"], device
    )
    weights["all_x_embedder.2-1.bias"] = _basic(weights["all_x_embedder.2-1.bias"], device)
    weights["all_x_embedder.2-1.weight"] = _permute(weights["all_x_embedder.2-1.weight"], device)
    weights["cap_embedder.0.weight"] = _reshape_1_2560(weights["cap_embedder.0.weight"], device)
    weights["cap_embedder.1.bias"] = _basic(weights["cap_embedder.1.bias"], device)
    weights["cap_embedder.1.weight"] = _permute(weights["cap_embedder.1.weight"], device)

    # ── Constant generators (computed, not from model weights) ────────────────
    weights["_t_scale"] = _sbf16(1000.0)

    half = 128
    freqs_ts = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half).unsqueeze(0)
    weights["_t_freqs"] = ttnn.from_torch(
        freqs_ts,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=mem,
        mesh_mapper=replicated,
    )

    weights["_one"] = _sbf16(1.0)

    from models.demos.z_image_turbo.tt.dit.model_ttnn import CAP_TOKENS as _CAP

    cap_len = _CAP
    cap_f_ids = list(range(1, cap_len + 1))
    cap_hw_ids = [0] * cap_len
    img_f_start = cap_len + 1
    img_f_ids = [img_f_start] * 1024
    img_h_ids = [h for h in range(32) for _ in range(32)]
    img_w_ids = list(range(32)) * 32
    weights["_cap_f_ids"] = _ids(cap_f_ids)
    weights["_img_f_ids"] = _ids(img_f_ids)
    weights["_img_h_ids"] = _ids(img_h_ids)
    weights["_img_w_ids"] = _ids(img_w_ids)
    weights["_cap_hw_ids"] = _ids(cap_hw_ids)

    return weights
