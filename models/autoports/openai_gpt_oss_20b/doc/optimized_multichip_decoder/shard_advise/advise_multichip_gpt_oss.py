# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor capture for one GPT-OSS TP4/EP4 decoder device.

The advisor cannot represent ``ttnn.sparse_matmul`` or mesh collectives.  This
shape-only graph therefore captures the dense operations surrounding those
boundaries with the exact per-device widths used by the fixed 1x4 decoder:
local QKV (1280), local attention output (1024), row-parallel O (2880), router,
packed gate/up, and expert down.  The resulting layouts are a seed candidate;
the real mesh path remains the authority for PCC and latency.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import ttnn

_ROOT = Path("/home/mvasiljevic/tt-metal")
_BASE = runpy.run_path(
    str(_ROOT / "models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/shard_advise/advise_gpt_oss.py")
)
_host_tensor = _BASE["_host_tensor"]
_CAPTURE_DEVICE = _BASE["_CAPTURE_DEVICE"]

HIDDEN = 2880
LOCAL_QKV = 1280
LOCAL_ATTN = 1024
EXPERT_INTERMEDIATE = 2880
EXPERTS = 32
PADDED_BATCH = 32

_WEIGHTS = None


def make_inputs(device):
    del device
    global _WEIGHTS

    norm_shape = (1, 1, HIDDEN // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
    _WEIGHTS = {
        "input_norm": _host_tensor(norm_shape, layout=ttnn.ROW_MAJOR_LAYOUT),
        "post_norm": _host_tensor(norm_shape, layout=ttnn.ROW_MAJOR_LAYOUT),
        "qkv": _host_tensor((HIDDEN, LOCAL_QKV)),
        "qkv_bias": _host_tensor((1, 1, LOCAL_QKV)),
        "o": _host_tensor((LOCAL_ATTN, HIDDEN)),
        "o_bias": _host_tensor((1, 1, HIDDEN)),
        "router": _host_tensor((HIDDEN, EXPERTS)),
        "gate_up": _host_tensor((HIDDEN, 2 * EXPERT_INTERMEDIATE)),
        "down": _host_tensor((EXPERT_INTERMEDIATE, HIDDEN)),
    }
    return (
        _host_tensor((1, 1, PADDED_BATCH, HIDDEN)),
        _host_tensor((1, 1, PADDED_BATCH, LOCAL_ATTN)),
        _host_tensor((1, 1, PADDED_BATCH, EXPERT_INTERMEDIATE)),
    )


def decode(hidden, local_attention, expert_hidden):
    """Capture the exact dense shapes from one device of the mesh path."""

    normalized = ttnn.rms_norm(hidden, epsilon=1e-5, weight=_WEIGHTS["input_norm"])
    qkv = ttnn.linear(normalized, _WEIGHTS["qkv"], bias=_WEIGHTS["qkv_bias"])
    projected = ttnn.linear(local_attention, _WEIGHTS["o"], bias=_WEIGHTS["o_bias"])
    residual = ttnn.add(hidden, projected)
    post_norm = ttnn.rms_norm(residual, epsilon=1e-5, weight=_WEIGHTS["post_norm"])
    router = ttnn.linear(post_norm, _WEIGHTS["router"])
    gate_up = ttnn.linear(post_norm, _WEIGHTS["gate_up"])
    down = ttnn.linear(expert_hidden, _WEIGHTS["down"])
    # The current advisor tracer requires one returned tensor.  The other
    # operations still execute and remain in the intercepted graph.
    del qkv, residual, post_norm, router, gate_up
    return down
