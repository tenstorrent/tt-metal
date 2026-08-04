# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device decoder-block sanity checks (Qwen36DecoderLayer).

Full layers run end-to-end without producing NaN/Inf (and, for the DeltaNet
prefill block, a non-constant output). These are sanity checks.

``device`` and ``setup`` come from tests/unit/conftest.py.
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer

from .conftest import DEVICE_PARAMS

pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]

# GDN chunk-seq prefill kernel supports exactly one chunk size (see Qwen36DecoderLayer.forward's
# chunk_size default / its docstring). Using anything else (e.g. 64) raises a 64≠128 matmul mismatch.
GDN_PREFILL_CHUNK = 128


def test_layer0_deltanet_prefill_block(device, setup):
    """Layer 0 (DeltaNet) full decoder block, chunk-seq prefill — runs and is non-constant.

    Uses the only chunk size the GDN chunk-seq kernel supports (128); chunk_size=64 is unsupported.
    """
    args, sd, raw = setup
    device.enable_program_cache()

    B, T = 1, GDN_PREFILL_CHUNK
    x = torch.randn(B, T, args.dim, dtype=torch.bfloat16)

    block = Qwen36DecoderLayer(device, args, sd, layer_num=0)
    block.attention.reset_state(B)

    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(block.forward(x_t, mode="prefill", chunk_size=GDN_PREFILL_CHUNK))

    logger.info(
        f"block0 prefill out: shape={out.shape} range=[{out.min():.4f},{out.max():.4f}] std={out.float().std():.4f}"
    )
    assert out.shape == (B, T, args.dim), f"Wrong shape: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert out.float().std() > 0.01, "Output is near-constant"


def test_layer0_deltanet_decode(device, setup):
    """Layer 0 (DeltaNet) recurrent decode — single token, continuing from a real prefill.

    Mirrors the model's order of operations: prefill first (which populates the GDN recurrent +
    conv state), then decode one token from that state. A cold decode with no prior prefill is not a
    real scenario and compiles the full-L1 conv path (conv state absent) → L1 overflow.
    """
    args, sd, raw = setup
    device.enable_program_cache()

    layer = Qwen36DecoderLayer(device, args, sd, layer_num=0)
    layer.attention.reset_state(1)

    # Prefill to establish GDN recurrent + conv state (the state decode continues from).
    x_pf = torch.randn(1, GDN_PREFILL_CHUNK, args.dim, dtype=torch.bfloat16)
    x_pf_t = ttnn.from_torch(x_pf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    layer.forward(x_pf_t, mode="prefill", chunk_size=GDN_PREFILL_CHUNK)

    # Decode one token from the carried state.
    x_dec = torch.randn(1, 1, args.dim, dtype=torch.bfloat16)
    x_dec_t = ttnn.from_torch(x_dec, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(layer.forward(x_dec_t, mode="decode"))

    assert out.shape == (1, 1, args.dim), f"Wrong shape: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"


def test_layer3_gated_attention_prefill(device, setup):
    """Layer 3 (first full-attention layer) prefill — runs without NaN/Inf."""
    args, sd, raw = setup
    device.enable_program_cache()

    from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

    layer = Qwen36DecoderLayer(device, args, sd, layer_num=3)

    B, T = 1, 128
    x = torch.randn(B, T, args.dim, dtype=torch.bfloat16)
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    rope = Qwen36RoPESetup(device, args)
    pos_ids = torch.arange(T).unsqueeze(0)
    cos, sin = rope.get_rot_mats(pos_ids)

    out = ttnn.to_torch(layer.forward(x_t, cos=cos, sin=sin, mode="prefill"))

    assert out.shape == (B, T, args.dim), f"Wrong shape: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
