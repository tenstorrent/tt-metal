# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole


@pytest.fixture(autouse=True)
def p100_only(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Llama-3.1-8B P100 configs")


HEAD_DIM = 128


def rope_tables(seq_len):
    pos = torch.arange(seq_len).float()[:, None]
    inv = 1.0 / (500000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    ang = pos * inv[None, :]
    cos = torch.repeat_interleave(torch.cos(ang), 2, dim=-1).reshape(1, 1, seq_len, HEAD_DIM)
    sin = torch.repeat_interleave(torch.sin(ang), 2, dim=-1).reshape(1, 1, seq_len, HEAD_DIM)
    trans = torch.zeros(1, 1, 32, 32)
    trans[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
    trans[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
    return cos, sin, trans


def bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.float().view(torch.int32)


def run_both(device, n_heads, seq_len, scale, output_dtype):
    torch.manual_seed(n_heads * 7 + seq_len + int(scale * 4))
    cos, sin, trans = rope_tables(seq_len)
    dram = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tx = ttnn.from_torch(torch.randn(1, n_heads, seq_len, HEAD_DIM) * scale, **dram)
    tcos, tsin, ttrans = (ttnn.from_torch(t, **dram) for t in (cos, sin, trans))
    ref = ttnn.experimental.rotary_embedding_llama(tx, tcos, tsin, ttrans, is_decode_mode=False)
    if output_dtype != ttnn.bfloat16:
        ref = ttnn.typecast(ref, output_dtype)
    out = ttnn.experimental.llama31_8b.rotary_bf8(tx, tcos, tsin, ttrans, output_dtype=output_dtype)
    assert out.dtype == output_dtype
    assert out.shape == ref.shape
    return ttnn.to_torch(out), ttnn.to_torch(ref)


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole program")
@pytest.mark.parametrize("n_heads", [32, 8])
@pytest.mark.parametrize("scale", [1.0, 16.0, 1 / 64])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_rotary_bf8_matches_rotary_then_typecast(device, n_heads, scale, output_dtype):
    out, ref = run_both(device, n_heads, 1024, scale, output_dtype)
    assert torch.equal(bits(out), bits(ref))


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole program")
@pytest.mark.parametrize("n_heads, seq_len", [(32, 128), (8, 2048)])
def test_rotary_bf8_other_shapes_match(device, n_heads, seq_len):
    out, ref = run_both(device, n_heads, seq_len, 1.0, ttnn.bfloat8_b)
    assert torch.equal(bits(out), bits(ref))


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole program")
def test_rotary_bf8_program_cache(device):
    entries = None
    for scale in (1.0, 2.0, 4.0):
        out, ref = run_both(device, 32, 1024, scale, ttnn.bfloat8_b)
        if entries is None:
            entries = device.num_program_cache_entries()
        assert device.num_program_cache_entries() == entries
        assert torch.equal(bits(out), bits(ref))
