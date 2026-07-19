# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor target for the Qwen2.5-Coder-32B TP4 dense decode subgraph.

The advisor does not model Ring CCL operations.  This target therefore traces
the exact per-rank dense shapes on one Blackhole device, with collective
boundaries represented by inputs/outputs: full-hidden inputs after the two
all-gathers, the local attention input before row-parallel O, and full-hidden
partial outputs before the two reduce-scatters.  It deliberately includes the
post-audit packed gate/up rewrite candidate.  Real TP4 device runs decide
whether any advised layout/program seed is retained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import ttnn

TT_METAL_ROOT = Path(__file__).resolve().parents[6]
if str(TT_METAL_ROOT) not in sys.path:
    # Append so the advisor environment's installed ttnn remains authoritative.
    sys.path.append(str(TT_METAL_ROOT))

BATCH = 32
HIDDEN = 5120
LOCAL_HIDDEN = 1280
LOCAL_QKV_LOGICAL = 1792
LOCAL_QKV_PADDED = 2048
LOCAL_INTERMEDIATE = 6912
LOCAL_INTERMEDIATE_PADDED = 7168

_INPUT_NORM = None
_QKV_WEIGHT = None
_QKV_BIAS = None
_O_WEIGHT = None
_POST_NORM = None
_PACKED_GATE_UP_WEIGHT = None
_DOWN_WEIGHT = None


def _device_tensor(host, device, *, dtype):
    return ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _build(device):
    global _INPUT_NORM, _QKV_WEIGHT, _QKV_BIAS, _O_WEIGHT
    global _POST_NORM, _PACKED_GATE_UP_WEIGHT, _DOWN_WEIGHT

    generator = torch.Generator().manual_seed(20260719)

    def weight(k, n):
        return (torch.randn((k, n), generator=generator) * 0.02).to(torch.bfloat16)

    _INPUT_NORM = _device_tensor(torch.ones(HIDDEN, dtype=torch.bfloat16), device, dtype=ttnn.bfloat16)
    _QKV_WEIGHT = _device_tensor(weight(HIDDEN, LOCAL_QKV_PADDED), device, dtype=ttnn.bfloat8_b)
    _QKV_BIAS = _device_tensor(
        torch.zeros((1, 1, 1, LOCAL_QKV_PADDED), dtype=torch.bfloat16),
        device,
        dtype=ttnn.bfloat16,
    )
    _O_WEIGHT = _device_tensor(weight(LOCAL_HIDDEN, HIDDEN), device, dtype=ttnn.bfloat8_b)
    _POST_NORM = _device_tensor(torch.ones(HIDDEN, dtype=torch.bfloat16), device, dtype=ttnn.bfloat16)
    _PACKED_GATE_UP_WEIGHT = _device_tensor(weight(HIDDEN, 2 * LOCAL_INTERMEDIATE_PADDED), device, dtype=ttnn.bfloat4_b)
    _DOWN_WEIGHT = _device_tensor(weight(LOCAL_INTERMEDIATE_PADDED, HIDDEN), device, dtype=ttnn.bfloat4_b)

    full_hidden = _device_tensor(
        torch.randn((1, 1, BATCH, HIDDEN), generator=generator, dtype=torch.bfloat16),
        device,
        dtype=ttnn.bfloat16,
    )
    local_attention = _device_tensor(
        torch.randn((1, 1, BATCH, LOCAL_HIDDEN), generator=generator, dtype=torch.bfloat16),
        device,
        dtype=ttnn.bfloat16,
    )
    return full_hidden, local_attention


def decode(full_hidden, local_attention):
    """Trace one local TP rank's rewritten dense attention + MLP projections."""

    normed = ttnn.rms_norm(full_hidden, epsilon=1.0e-6, weight=_INPUT_NORM)
    qkv = ttnn.matmul(normed, _QKV_WEIGHT, dtype=ttnn.bfloat16)
    qkv = ttnn.add(qkv, _QKV_BIAS, dtype=ttnn.bfloat16)
    qkv = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, BATCH, LOCAL_QKV_LOGICAL])

    # The attention/cache composite sits between QKV and O in the real graph.
    # Its local concatenated output has width 1280.
    o_partial = ttnn.matmul(local_attention, _O_WEIGHT, dtype=ttnn.bfloat16)

    post_normed = ttnn.rms_norm(full_hidden, epsilon=1.0e-6, weight=_POST_NORM)
    packed = ttnn.matmul(post_normed, _PACKED_GATE_UP_WEIGHT, dtype=ttnn.bfloat16)
    gate = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, BATCH, LOCAL_INTERMEDIATE_PADDED])
    up = ttnn.slice(
        packed,
        [0, 0, 0, LOCAL_INTERMEDIATE_PADDED],
        [1, 1, BATCH, 2 * LOCAL_INTERMEDIATE_PADDED],
    )
    gated = ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat16,
    )
    down_partial = ttnn.matmul(gated, _DOWN_WEIGHT, dtype=ttnn.bfloat16)

    # The advisor tracer requires one return tensor. Keep every dense branch
    # live by combining equal-width views of the three collective boundaries;
    # this synthetic sink is outside the production decoder.
    o_sink = ttnn.slice(o_partial, [0, 0, 0, 0], [1, 1, BATCH, LOCAL_QKV_LOGICAL])
    down_sink = ttnn.slice(down_partial, [0, 0, 0, 0], [1, 1, BATCH, LOCAL_QKV_LOGICAL])
    return ttnn.add(ttnn.add(qkv, o_sink, dtype=ttnn.bfloat16), down_sink, dtype=ttnn.bfloat16)


def make_inputs(device):
    return _build(device)
