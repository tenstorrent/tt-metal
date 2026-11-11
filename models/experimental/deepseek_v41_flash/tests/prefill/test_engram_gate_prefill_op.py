# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for ``ttnn.experimental.deepseek.engram_gate_prefill`` against a torch reference of the Engram gate
math (everything after the ``wkv`` matmul), on DRAM-interleaved TILE operands at prefill sizes.

The prefill counterpart of ``tests/decode/test_engram_gate_op.py``: same math, but the operands are the shapes a
prefill chunk uses (``[1, hc, T, D]`` streams, ``[1, 1, T, (hc+1)*D]`` kv) and T spans several 32-token blocks, so
the block loop, the cohort all-gather and the partial last block are all exercised.

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_engram_gate_prefill_op.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

HC = 4
DIM = 5120  # the checkpoint's hidden size: 160 tiles, which splits over the cohort
EPS = 1e-20
GATE_CLAMP = 1e-6


def _reference(x, kv, gate_weight):
    """``x`` ``[1, hc, T, D]``, ``kv`` ``[1, 1, T, (hc+1)*D]``, ``gate_weight`` ``[hc, D]``; fp32."""
    _, hc, t, d = x.shape
    key = kv[..., : hc * d].reshape(1, t, hc, d).transpose(1, 2)
    value = kv[..., hc * d :].reshape(1, 1, t, d)
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * torch.rsqrt(key.pow(2).mean(-1, keepdim=True) + EPS)
    dot = (x * gate_weight.reshape(1, hc, 1, d) * key).sum(-1, keepdim=True) * rstd
    gate = torch.sigmoid(dot.abs().clamp_min(GATE_CLAMP).sqrt() * dot.sign())
    return x + gate * value


@pytest.mark.parametrize("seq_len", (32, 64, 1024, 2048))  # one block, two, a full chunk, a long chunk
def test_engram_gate_prefill_op(device, seq_len):
    torch.manual_seed(seq_len)
    x = torch.randn(1, HC, seq_len, DIM).bfloat16().float()
    kv = torch.randn(1, 1, seq_len, (HC + 1) * DIM).bfloat16().float()
    q_weight = 1.0 + 0.1 * torch.randn(HC, DIM)
    k_weight = 1.0 + 0.1 * torch.randn(HC, DIM)
    gate_weight = q_weight * k_weight * DIM**-0.5
    expected = _reference(x, kv, gate_weight)

    def to_device(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tt_x = to_device(x)
    tt_kv = to_device(kv)
    # One [32, D] page per copy, every row holding that copy's vector.
    tt_gate = to_device(gate_weight.repeat(ttnn.TILE_SIZE, 1).reshape(1, HC, ttnn.TILE_SIZE, DIM), ttnn.float32)

    # Twice: the second call runs from the program cache, after the exchange semaphore was reset.
    for _ in range(2):
        tt_out = ttnn.experimental.deepseek.engram_gate_prefill(tt_x, tt_kv, tt_gate, EPS, clamp_value=GATE_CLAMP)
    assert tt_out.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    actual = ttnn.to_torch(tt_out).float()

    passing, pcc = comp_pcc(expected, actual, 0.999)
    logger.info(f"T {seq_len}: out PCC {pcc}")
    assert passing, f"T {seq_len}: out PCC {pcc}"
    passing, pcc = comp_pcc(expected - x, actual - x, 0.99)
    logger.info(f"T {seq_len}: out - x PCC {pcc}")
    assert passing, f"T {seq_len}: out - x PCC {pcc}"


@pytest.mark.parametrize("seq_len", (1024,))
def test_engram_gate_prefill_inputs_intact(device, seq_len):
    """The op writes a fresh output; its inputs are left as they were (the layer still needs ``streams``)."""
    torch.manual_seed(7)
    x = torch.randn(1, HC, seq_len, DIM).bfloat16().float()
    kv = torch.randn(1, 1, seq_len, (HC + 1) * DIM).bfloat16().float()
    gate_weight = (1.0 + 0.1 * torch.randn(HC, DIM)) * DIM**-0.5

    def to_device(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tt_x = to_device(x)
    tt_kv = to_device(kv)
    tt_gate = to_device(gate_weight.repeat(ttnn.TILE_SIZE, 1).reshape(1, HC, ttnn.TILE_SIZE, DIM), ttnn.float32)

    ttnn.experimental.deepseek.engram_gate_prefill(tt_x, tt_kv, tt_gate, EPS, clamp_value=GATE_CLAMP)

    for name, host, tt in (("streams", x, tt_x), ("kv", kv, tt_kv)):
        passing, pcc = comp_pcc(host, ttnn.to_torch(tt).float(), 0.999)
        logger.info(f"{name} after the op: PCC {pcc}")
        assert passing, f"{name} was modified by the op: PCC {pcc}"
