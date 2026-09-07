# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local decoder-stream probes; these do not reconstruct attention-residual mixing."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.ops import causal_depthwise_conv_reference, kda_gate_reference
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import load_kda_layer_state_dict


def load_trace_rows(path: Path, start: int, count: int) -> torch.Tensor:
    """Read a bounded row interval from a named single-stream safetensor file."""
    if start < 0 or count <= 0:
        raise ValueError("trace row interval must be nonnegative and nonempty")
    with safe_open(path, framework="pt", device="cpu") as shard:
        value = shard.get_slice(path.stem)[start : start + count]
    if value.shape[0] != count:
        raise ValueError(f"{path}: requested {count} rows at {start}, got {value.shape[0]}")
    return value


def load_decoder_stream_probe(
    checkpoint: Path, trace: Path, layer_idx: int, sequence: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor], KDAConfig]:
    """Return FP32 normalized decoder rows and matching layer weights/configuration.

    This matches the issue's diagnostic boundary, not proven exact attention-input
    replay: the trace may include attention-residual mixing that is not reconstructed.
    """
    config = KDAConfig.from_model_config(json.loads((checkpoint / "config.json").read_text()))
    key = f"decoder_output_layer_{layer_idx - 1}" if layer_idx else "decoder_input_layer_0"
    rows = load_trace_rows(trace / "decoder_io" / f"{key}.safetensors", 0, sequence).float()
    if tuple(rows.shape) != (sequence, config.hidden_size):
        raise ValueError(f"unexpected decoder rows: {tuple(rows.shape)}")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    norm_key = f"language_model.model.layers.{layer_idx}.input_layernorm.weight"
    with safe_open(checkpoint / index[norm_key], framework="pt", device="cpu") as shard:
        norm = shard.get_tensor(norm_key).float()
    hidden = rows * torch.rsqrt(rows.square().mean(-1, keepdim=True) + config.norm_eps) * norm
    weights = load_kda_layer_state_dict(checkpoint, layer_idx, config)
    return hidden.unsqueeze(0), weights, config


def decoder_probe_recurrence_inputs(
    hidden: torch.Tensor, weights: dict[str, torch.Tensor], config: KDAConfig
) -> tuple[torch.Tensor, ...]:
    """Return operation inputs using the existing reference projections and gates."""
    hidden = hidden.to(torch.bfloat16).float()
    sequence = hidden.shape[1]
    qkv = tuple(
        causal_depthwise_conv_reference(
            F.linear(hidden, weights[f"{name}_proj.weight"].float()), weights[f"{name}_conv1d.weight"]
        )[0]
        for name in ("q", "k", "v")
    )
    raw_gate = F.linear(F.linear(hidden, weights["f_a_proj.weight"].float()), weights["f_b_proj.weight"].float())
    gate = kda_gate_reference(
        raw_gate.reshape(1, sequence, config.num_heads, config.head_k_dim),
        weights["A_log"],
        weights["dt_bias"],
        config.gate_lower_bound,
    ).reshape_as(qkv[0])
    beta = torch.sigmoid(F.linear(hidden, weights["b_proj.weight"].float()))
    beta = beta.squeeze(0).T.reshape(config.num_heads, sequence // 32, 32, 1).contiguous()
    return *(tensor.to(torch.bfloat16).float() for tensor in (*qkv, gate)), beta
