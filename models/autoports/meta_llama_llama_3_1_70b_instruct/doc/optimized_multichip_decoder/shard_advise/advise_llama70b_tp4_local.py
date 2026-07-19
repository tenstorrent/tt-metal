# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor target for one exact-shape TP4-local Llama 70B dense block.

The advisor does not model mesh CCL operations. The two row-parallel reduction
boundaries are therefore identity edges in this capture: tensor shapes before
and after each edge are the real local-partial/full-hidden shapes, while the
hardware candidate harness owns the reduce-scatter/all-gather or fused CCL
implementation. This keeps the advisor focused on its supported job: the
per-device L1 layout and 1-D matmul seed for the rewritten dense block.
"""

from __future__ import annotations

import os
import sys

import torch

import ttnn

MODEL_DIR = os.environ.get("SHARD_ADVISE_MODEL_DIR", "/home/mvasiljevic/tt-metal")
BATCH_PADDED = 32
HIDDEN = 8192
LOCAL_QKV = 2560
LOCAL_ATTENTION = 2048
LOCAL_INTERMEDIATE = 7168
LOCAL_GATE_UP = 2 * LOCAL_INTERMEDIATE


def _device_tensor(tensor, device, *, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _build(device):
    # Append so the advisor environment's installed ttnn stays authoritative.
    if MODEL_DIR not in sys.path:
        sys.path.append(MODEL_DIR)

    zeros = lambda *shape: torch.zeros(shape, dtype=torch.bfloat16)
    hidden = _device_tensor(zeros(1, 1, BATCH_PADDED, HIDDEN), device)
    input_norm = _device_tensor(torch.ones(HIDDEN, dtype=torch.bfloat16), device)
    post_norm = _device_tensor(torch.ones(HIDDEN, dtype=torch.bfloat16), device)
    weights = (
        _device_tensor(zeros(HIDDEN, LOCAL_QKV), device, dtype=ttnn.bfloat4_b),
        _device_tensor(zeros(LOCAL_ATTENTION, HIDDEN), device, dtype=ttnn.bfloat4_b),
        _device_tensor(zeros(HIDDEN, LOCAL_GATE_UP), device, dtype=ttnn.bfloat4_b),
        _device_tensor(zeros(LOCAL_INTERMEDIATE, HIDDEN), device, dtype=ttnn.bfloat4_b),
    )
    return hidden, input_norm, post_norm, weights


_INPUT_NORM = None
_POST_NORM = None
_QKV_WEIGHT = None
_OUTPUT_WEIGHT = None
_GATE_UP_WEIGHT = None
_DOWN_WEIGHT = None


def decode(hidden):
    residual = hidden
    normed = ttnn.rms_norm(hidden, epsilon=1.0e-5, weight=_INPUT_NORM)
    packed_qkv = ttnn.linear(normed, _QKV_WEIGHT, dtype=ttnn.bfloat16)

    # Exact local attention width after create-heads/SDPA/concat-heads. The
    # slice stands in only for those composite attention ops; it is not a
    # proposed runtime rewrite.
    local_attention = ttnn.slice(
        packed_qkv,
        [0, 0, 0, 0],
        [1, 1, BATCH_PADDED, LOCAL_ATTENTION],
        [1, 1, 1, 1],
    )
    output_partial = ttnn.linear(local_attention, _OUTPUT_WEIGHT, dtype=ttnn.bfloat16)

    # Identity edge for the unsupported row-parallel CCL boundary.
    hidden = ttnn.add(residual, output_partial, dtype=ttnn.bfloat16)
    residual = hidden
    normed = ttnn.rms_norm(hidden, epsilon=1.0e-5, weight=_POST_NORM)
    packed_gate_up = ttnn.linear(normed, _GATE_UP_WEIGHT, dtype=ttnn.bfloat16)
    gate = ttnn.slice(
        packed_gate_up,
        [0, 0, 0, 0],
        [1, 1, BATCH_PADDED, LOCAL_INTERMEDIATE],
        [1, 1, 1, 1],
    )
    up = ttnn.slice(
        packed_gate_up,
        [0, 0, 0, LOCAL_INTERMEDIATE],
        [1, 1, BATCH_PADDED, LOCAL_GATE_UP],
        [1, 1, 1, 1],
    )
    gated = ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat16,
    )
    down_partial = ttnn.linear(gated, _DOWN_WEIGHT, dtype=ttnn.bfloat16)

    # Identity edge for the second unsupported row-parallel CCL boundary.
    return ttnn.add(residual, down_partial, dtype=ttnn.bfloat16)


def make_inputs(device):
    global _INPUT_NORM, _POST_NORM
    global _QKV_WEIGHT, _OUTPUT_WEIGHT, _GATE_UP_WEIGHT, _DOWN_WEIGHT
    hidden, _INPUT_NORM, _POST_NORM, weights = _build(device)
    _QKV_WEIGHT, _OUTPUT_WEIGHT, _GATE_UP_WEIGHT, _DOWN_WEIGHT = weights
    return (hidden,)
