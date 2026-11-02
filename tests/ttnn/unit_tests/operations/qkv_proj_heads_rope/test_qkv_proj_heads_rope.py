# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for qkv_proj_heads_rope (immutable spec).

Phase 0 placement: x, cos/sin and q/k/v in L1, weight/bias in DRAM, tile-aligned S.
Tolerances match eval/golden_tests/qkv_proj_heads_rope/helpers.py TOLERANCES.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.qkv_proj_heads_rope import qkv_proj_heads_rope

HIDDEN = 768
NUM_HEADS = 12
HEAD_DIM = HIDDEN // NUM_HEADS

# dtype -> (min PCC, max relative RMS), per output.
TOLERANCES = {ttnn.bfloat16: (0.9999, 4.0e-3)}


def rope_tables(seq_tab):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    freqs = torch.outer(torch.arange(seq_tab, dtype=torch.float32), inv_freq)
    cos_half = torch.cos(freqs).to(torch.bfloat16)
    sin_half = torch.sin(freqs).to(torch.bfloat16)
    cos = torch.cat((cos_half, cos_half), dim=-1).reshape(1, 1, seq_tab, HEAD_DIM)
    sin = torch.cat((sin_half, sin_half), dim=-1).reshape(1, 1, seq_tab, HEAD_DIM)
    return cos, sin


def make_inputs(batch, seq, seq_tab):
    torch.manual_seed(42)
    x = torch.randn((batch, 1, seq, HIDDEN))
    weight = torch.randn((HIDDEN, 3 * HIDDEN)) / math.sqrt(HIDDEN)
    weight[:, :HIDDEN] /= math.sqrt(HEAD_DIM)  # caller-folded attention scale
    bias = torch.randn((3 * HIDDEN,)) * 0.02
    cos, sin = rope_tables(seq_tab)
    return x.to(torch.bfloat16), weight.to(torch.bfloat16), bias.to(torch.bfloat16), cos, sin


def reference(x, w, b, cos, sin, num_heads=NUM_HEADS):
    batch, _, seq, hidden = x.shape
    head_dim = hidden // num_heads
    p = x.reshape(batch, seq, hidden).double() @ w.double() + b.double().reshape(-1)
    p = p.reshape(batch, seq, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    c, s = cos[0, 0, :seq].double(), sin[0, 0, :seq].double()

    def rot(t):
        return t * c + torch.cat([-t[..., head_dim // 2 :], t[..., : head_dim // 2]], dim=-1) * s

    return rot(p[0]), rot(p[1]), p[2]


def pcc(actual, expected):
    stacked = torch.stack([actual.flatten().double(), expected.flatten().double()])
    return torch.corrcoef(stacked)[0, 1].item()


def relative_rms(actual, expected):
    diff = actual.double() - expected.double()
    return (diff.pow(2).mean().sqrt() / expected.double().pow(2).mean().sqrt()).item()


def to_device(tensor, device, memory_config):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


def device_inputs(device, x, w, b, cos, sin, x_memory_config=ttnn.L1_MEMORY_CONFIG):
    return (
        to_device(x, device, x_memory_config),
        to_device(w, device, ttnn.DRAM_MEMORY_CONFIG),
        to_device(b, device, ttnn.DRAM_MEMORY_CONFIG),
        to_device(cos, device, ttnn.L1_MEMORY_CONFIG),
        to_device(sin, device, ttnn.L1_MEMORY_CONFIG),
    )


def check_outputs(outputs, expected, buffer_type=ttnn.BufferType.L1, dtype=ttnn.bfloat16):
    min_pcc, max_rms = TOLERANCES[dtype]
    assert isinstance(outputs, (tuple, list)) and len(outputs) == 3
    for name, out, ref in zip(("q", "k", "v"), outputs, expected):
        assert list(out.shape) == list(ref.shape), f"{name}: shape {list(out.shape)} != {list(ref.shape)}"
        assert out.dtype == dtype, f"{name}: dtype {out.dtype}"
        assert out.layout == ttnn.TILE_LAYOUT, f"{name}: layout {out.layout}"
        assert out.memory_config().buffer_type == buffer_type, f"{name}: buffer {out.memory_config().buffer_type}"
        actual = ttnn.to_torch(out)
        assert torch.isfinite(actual).all(), f"{name}: non-finite values"
        out_pcc = pcc(actual, ref)
        out_rms = relative_rms(actual, ref)
        assert out_pcc >= min_pcc, f"{name}: pcc {out_pcc:.6f} < {min_pcc}"
        assert out_rms <= max_rms, f"{name}: rel rms {out_rms:.3e} > {max_rms:.1e}"


def explicit_config(math_fidelity=ttnn.MathFidelity.HiFi3):
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# (B, S, S_tab): single tile-row, tables longer than S, multi-batch, odd batch,
# and the production token counts (1024 / 2048 / 4096 tokens).
SHAPES = [
    (1, 32, 32),
    (1, 128, 512),
    (2, 64, 64),
    (3, 96, 96),
    (4, 256, 512),
    (8, 256, 256),
    (32, 128, 128),
    (8, 512, 512),
]


@pytest.mark.parametrize("batch, seq, seq_tab", SHAPES, ids=[f"{b}x{s}_tab{t}" for b, s, t in SHAPES])
def test_qkv_proj_heads_rope(device, batch, seq, seq_tab):
    x, w, b, cos, sin = make_inputs(batch, seq, seq_tab)
    expected = reference(x, w, b, cos, sin)
    outputs = qkv_proj_heads_rope(
        *device_inputs(device, x, w, b, cos, sin),
        num_heads=NUM_HEADS,
        compute_kernel_config=explicit_config(),
    )
    check_outputs(outputs, expected)


def test_default_config_and_inputs_unchanged(device):
    batch, seq, seq_tab = 8, 512, 512
    x, w, b, cos, sin = make_inputs(batch, seq, seq_tab)
    expected = reference(x, w, b, cos, sin)
    tensors = device_inputs(device, x, w, b, cos, sin)
    outputs = qkv_proj_heads_rope(*tensors)
    check_outputs(outputs, expected)
    for host, dev in zip((x, w, cos, sin), (tensors[0], tensors[1], tensors[3], tensors[4])):
        assert torch.equal(ttnn.to_torch(dev).reshape(host.shape), host)


def test_hifi4_explicit_memory_config(device):
    batch, seq, seq_tab = 4, 256, 256
    x, w, b, cos, sin = make_inputs(batch, seq, seq_tab)
    expected = reference(x, w, b, cos, sin)
    outputs = qkv_proj_heads_rope(
        *device_inputs(device, x, w, b, cos, sin),
        compute_kernel_config=explicit_config(ttnn.MathFidelity.HiFi4),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    check_outputs(outputs, expected)


def test_rejects_malformed_weight(device, expect_error):
    x, w, b, cos, sin = make_inputs(1, 32, 32)
    x_d, _, b_d, cos_d, sin_d = device_inputs(device, x, w, b, cos, sin)
    bad_weight = to_device(w[:, :HIDDEN].contiguous(), device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(ValueError, r"(?i)weight"):
        qkv_proj_heads_rope(x_d, bad_weight, b_d, cos_d, sin_d)


def test_rejects_short_tables(device, expect_error):
    x, w, b, cos, sin = make_inputs(1, 64, 32)
    with expect_error(ValueError, r"(?i)(cos|sin|table)"):
        qkv_proj_heads_rope(*device_inputs(device, x, w, b, cos, sin))


# x in DRAM (R3 x multicast): one shape per grid class (h_c = 1 / 2 / 4) plus a single tile-row.
DRAM_X_SHAPES = [(1, 32, 32), (2, 512, 512), (4, 256, 256), (8, 384, 384), (8, 512, 512)]


@pytest.mark.parametrize("batch, seq, seq_tab", DRAM_X_SHAPES, ids=[f"{b}x{s}_tab{t}" for b, s, t in DRAM_X_SHAPES])
def test_x_in_dram(device, batch, seq, seq_tab):
    x, w, b, cos, sin = make_inputs(batch, seq, seq_tab)
    expected = reference(x, w, b, cos, sin)
    outputs = qkv_proj_heads_rope(
        *device_inputs(device, x, w, b, cos, sin, x_memory_config=ttnn.DRAM_MEMORY_CONFIG),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    check_outputs(outputs, expected)


def ceil_tile(n):
    return -(-n // 32) * 32


def to_device_padded(tensor, device, memory_config, fill):
    """TILE tensor whose tile padding (rows beyond the logical extent) holds `fill`."""
    rows = tensor.shape[-2]
    storage = torch.full((*tensor.shape[:-2], ceil_tile(rows), tensor.shape[-1]), fill, dtype=tensor.dtype)
    storage[..., :rows, :] = tensor
    padded = to_device(storage, device, memory_config)
    return ttnn.reshape(padded, ttnn.Shape(list(tensor.shape)), ttnn.Shape(list(storage.shape)))


def padded_rows(out, seq):
    shape = ttnn.Shape(list(out.padded_shape))
    return ttnn.to_torch(ttnn.reshape(out, shape, shape))[..., seq : ceil_tile(seq), :]


# Non-aligned S: single tile-row, odd batch, many images per core, and B=4096 S=1 (multi-block per core).
NON_ALIGNED_SHAPES = [(2, 37, 37), (16, 8, 8), (13, 300, 300), (7, 511, 511), (4096, 1, 1)]


@pytest.mark.parametrize("batch, seq, seq_tab", NON_ALIGNED_SHAPES, ids=[f"{b}x{s}" for b, s, _ in NON_ALIGNED_SHAPES])
def test_non_aligned_padding_zero(device, batch, seq, seq_tab):
    x, w, b, cos, sin = make_inputs(batch, seq, seq_tab)
    expected = reference(x, w, b, cos, sin)
    l1 = ttnn.L1_MEMORY_CONFIG
    # B=4096, S=1 tile-pads x and each output to ~100-200 MB: DRAM only.
    big = ttnn.DRAM_MEMORY_CONFIG if batch * ceil_tile(seq) > 8192 else l1
    inf = float("inf")
    outputs = qkv_proj_heads_rope(
        to_device_padded(x, device, big, inf),
        to_device(w, device, ttnn.DRAM_MEMORY_CONFIG),
        to_device(b, device, ttnn.DRAM_MEMORY_CONFIG),
        to_device_padded(cos, device, l1, inf),
        to_device_padded(sin, device, l1, inf),
        memory_config=big,
    )
    check_outputs(outputs, expected, buffer_type=big.buffer_type)
    for name, out in zip(("q", "k", "v"), outputs):
        pad = padded_rows(out, seq)
        assert (pad == 0).all(), f"{name}: {int((pad != 0).sum())} non-zero padded values"


# DRAM outputs and tables, aligned and non-aligned, across x placements.
PLACEMENT_CASES = [
    (2, 512, 512, ttnn.DRAM_MEMORY_CONFIG),
    (8, 512, 512, ttnn.L1_MEMORY_CONFIG),
    (3, 100, 100, ttnn.DRAM_MEMORY_CONFIG),
]


@pytest.mark.parametrize(
    "batch, seq, seq_tab, x_memory_config",
    PLACEMENT_CASES,
    ids=[f"{b}x{s}_x{'dram' if m.buffer_type == ttnn.BufferType.DRAM else 'l1'}" for b, s, _, m in PLACEMENT_CASES],
)
def test_dram_outputs_and_tables(device, batch, seq, seq_tab, x_memory_config):
    x, w, b, cos, sin = make_inputs(batch, seq, seq_tab)
    expected = reference(x, w, b, cos, sin)
    dram = ttnn.DRAM_MEMORY_CONFIG
    outputs = qkv_proj_heads_rope(
        to_device(x, device, x_memory_config),
        to_device(w, device, dram),
        to_device(b, device, dram),
        to_device(cos, device, dram),
        to_device(sin, device, dram),
        memory_config=dram,
    )
    check_outputs(outputs, expected, buffer_type=ttnn.BufferType.DRAM)
