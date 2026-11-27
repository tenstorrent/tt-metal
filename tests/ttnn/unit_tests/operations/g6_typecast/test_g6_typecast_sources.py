# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 2 — float32 / int32 / uint16 sources on TILE (R1 `tile_stream`), bit-exact vs torch.

Arms: float32 -> bfloat16 (RNE, exact ties), int32 -> bfloat16 (rounded through float32: the
two-rounding witnesses 33685505, -33685505, 2^24+2^16+1), uint16 -> int32 (zero-extend).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_typecast import g6_typecast

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _fp32_input(shape):
    torch.manual_seed(7)
    n = _numel(shape)
    x = torch.randn(n, dtype=torch.float32) * torch.pow(2.0, torch.randint(-20, 20, (n,)).float())
    # RNE witnesses: exact ties (low 16 bits == 0x8000) with even and odd bf16 lsb, just-above/below ties,
    # specials, the bf16 max-overflow tie. (Subnormals and -0.0 are flushed to +0 by the Tensix datapath —
    # see test_float32_to_bfloat16_ftz; the golden contract's inputs are randn + exact ties.)
    bits = [0x3F808000, 0x3F818000, 0x3F808001, 0x3F807FFF, 0xBF818000, 0x7F7F8000, 0x7F7FFFFF]
    w = torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32)
    w = torch.cat([w, torch.tensor([0.0, float("inf"), float("-inf")])])
    k = min(len(w), n)
    x[:k] = w[:k]
    return x.reshape(shape)


def _int32_input(shape):
    torch.manual_seed(11)
    n = _numel(shape)
    x = torch.randint(-(2**31), 2**31 - 1, (n,), dtype=torch.int64).to(torch.int32)
    w = torch.tensor(
        [33685505, -33685505, 2**24 + 2**16 + 1, -(2**31), 2**31 - 1, 0, 1, -1, 257, 2**24 + 1],
        dtype=torch.int32,
    )
    k = min(len(w), n)
    x[:k] = w[:k]
    return x.reshape(shape)


def _bits16(t):
    return t.to(torch.bfloat16).contiguous().view(torch.int16)


@pytest.mark.parametrize(
    "shape, mem",
    [
        ((1, 512, 128), DRAM),  # F315
        ((1, 1, 196, 196), DRAM),  # F320 (w/h non-aligned)
        ((1, 1, 185, 185), DRAM),  # F323
        ((2, 3, 64, 96), L1),
    ],
)
def test_float32_to_bfloat16(device, shape, mem):
    x = _fp32_input(shape)
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    out = g6_typecast(t, ttnn.bfloat16)
    assert out.dtype == ttnn.bfloat16
    got = ttnn.to_torch(out)
    exp = x.to(torch.bfloat16)
    mism = (_bits16(got) != _bits16(exp)).sum().item()
    assert mism == 0, f"{mism} mismatches"


@pytest.mark.xfail(reason="Tensix FTZ: fp32 subnormals and -0.0 come out as +0 (hardware finding, R2)", strict=False)
def test_float32_to_bfloat16_ftz(device):
    x = torch.zeros(1, 1, 32, 32)
    x.view(-1)[:2] = torch.tensor([0x00018000, 0x80000000], dtype=torch.int64).to(torch.int32).view(torch.float32)
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(g6_typecast(t, ttnn.bfloat16))
    assert torch.equal(_bits16(got), _bits16(x.to(torch.bfloat16)))


@pytest.mark.parametrize("shape, mem", [((1, 1, 32, 128), DRAM), ((1, 3, 70, 50), L1)])
def test_int32_to_bfloat16(device, shape, mem):
    x = _int32_input(shape)
    t = ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    got = ttnn.to_torch(g6_typecast(t, ttnn.bfloat16))
    exp = x.to(torch.bfloat16)  # torch: int32 -> float -> bfloat16 (two roundings)
    bad = (_bits16(got) != _bits16(exp)).flatten()
    assert bad.sum().item() == 0, (
        f"{bad.sum().item()} mismatches; first inputs {x.flatten()[bad][:5].tolist()} "
        f"got {got.flatten()[bad][:5].tolist()} exp {exp.flatten()[bad][:5].tolist()}"
    )


@pytest.mark.parametrize("shape, mem", [((1, 1, 32, 128), DRAM), ((2, 40, 72), L1)])
def test_uint16_to_int32(device, shape, mem):
    torch.manual_seed(3)
    x = torch.randint(0, 2**16, (_numel(shape),), dtype=torch.int32)
    x[:4] = torch.tensor([0, 65535, 32768, 1], dtype=torch.int32)
    x = x.reshape(shape)
    t = ttnn.from_torch(x, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    got = ttnn.to_torch(g6_typecast(t, ttnn.int32)).to(torch.int64)
    assert torch.equal(got, x.to(torch.int64))
