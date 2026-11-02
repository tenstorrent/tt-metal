# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for linear_gelu (immutable spec).

Phase 0 placement: x and output in L1, weight/bias in DRAM, tile-aligned S.
Cells outside the op's SUPPORTED set (non-aligned S, DRAM x / output) are
skipped when the op refuses them with a SupportRefusal, and checked in full
once a refinement enables them.
Tolerances match eval/golden_tests/linear_gelu/helpers.py TOLERANCES.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.linear_gelu import default_compute_kernel_config, linear_gelu

HIDDEN = 768
FFN = 3072
TILE = 32

# dtype -> (min PCC, max relative RMS), rms(out - ref) / rms(ref).
TOLERANCES = {ttnn.bfloat16: (0.9999, 1.4e-2)}


def make_inputs(batch, seq):
    """bf16 operands with the model's distribution: z ~ N(-2.8, 1.7^2)."""
    torch.manual_seed(42)
    x = torch.randn((batch, 1, seq, HIDDEN))
    weight = torch.randn((HIDDEN, FFN)) * (1.7 / math.sqrt(HIDDEN))
    bias = torch.randn((FFN,)) * 0.1 - 2.8
    return x.to(torch.bfloat16), weight.to(torch.bfloat16), bias.to(torch.bfloat16)


def reference(x, w, b):
    batch, _, seq, hidden = x.shape
    z = x.reshape(batch, seq, hidden).double() @ w.double() + b.double().reshape(-1)
    return torch.nn.functional.gelu(z, approximate="tanh").reshape(batch, 1, seq, -1)


def pcc(actual, expected):
    stacked = torch.stack([actual.flatten().double(), expected.flatten().double()])
    return torch.corrcoef(stacked)[0, 1].item()


def relative_rms(actual, expected):
    diff = actual.double() - expected.double()
    return (diff.pow(2).mean().sqrt() / expected.double().pow(2).mean().sqrt()).item()


def ceil_tile(n):
    return -(-n // TILE) * TILE


def to_device(tensor, device, memory_config):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


def x_with_padding(x, device, memory_config, pad_value):
    """x on device whose tile-padded rows hold pad_value (a metadata-only logical view)."""
    batch, _, seq, hidden = x.shape
    padded = torch.full((batch, 1, ceil_tile(seq), hidden), pad_value, dtype=torch.bfloat16)
    padded[:, :, :seq, :] = x
    storage = to_device(padded, device, memory_config)
    if ceil_tile(seq) == seq:
        return storage
    return ttnn.reshape(storage, ttnn.Shape([batch, 1, seq, hidden]), ttnn.Shape(list(padded.shape)))


def read_padded(tensor):
    padded = ttnn.Shape(list(tensor.padded_shape))
    return ttnn.to_torch(ttnn.reshape(tensor, padded, padded))


def run(device, batch, seq, *, x_memory_config=ttnn.L1_MEMORY_CONFIG, pad_value=0.0, **kwargs):
    x, w, b = make_inputs(batch, seq)
    x_dev = x_with_padding(x, device, x_memory_config, pad_value)
    w_dev = to_device(w, device, ttnn.DRAM_MEMORY_CONFIG)
    b_dev = to_device(b, device, ttnn.DRAM_MEMORY_CONFIG)
    try:
        out = linear_gelu(x_dev, w_dev, b_dev, **kwargs)
    except SupportRefusal as exc:
        pytest.skip(f"outside SUPPORTED: {exc}")
    return x, w, b, x_dev, out


def check(out, x, w, b, expected_memory_config):
    batch, _, seq, _ = x.shape
    assert tuple(out.shape) == (batch, 1, seq, FFN)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.memory_config().buffer_type == expected_memory_config.buffer_type

    actual = ttnn.to_torch(out)
    assert torch.isfinite(actual).all()
    expected = reference(x, w, b)
    min_pcc, max_rms = TOLERANCES[ttnn.bfloat16]
    got_pcc, got_rms = pcc(actual, expected), relative_rms(actual, expected)
    assert got_pcc >= min_pcc, f"pcc {got_pcc:.6f} < {min_pcc}"
    assert got_rms <= max_rms, f"relative rms {got_rms:.4e} > {max_rms}"


# Single tile row, multi tile row, multi-batch, primary production shape and
# shapes whose token rows do not fill the grid.
ALIGNED_SHAPES = [(1, 32), (1, 128), (2, 64), (4, 256), (3, 96), (16, 32), (8, 512)]


@pytest.mark.parametrize("batch, seq", ALIGNED_SHAPES)
def test_linear_gelu(device, batch, seq):
    x, w, b, _, out = run(device, batch, seq, memory_config=ttnn.L1_MEMORY_CONFIG)
    check(out, x, w, b, ttnn.L1_MEMORY_CONFIG)


@pytest.mark.parametrize("batch, seq", [(2, 128)])
def test_default_memory_config_follows_input(device, batch, seq):
    x, w, b, _, out = run(device, batch, seq)
    check(out, x, w, b, ttnn.L1_MEMORY_CONFIG)


@pytest.mark.parametrize("batch, seq", [(2, 256)])
def test_caller_compute_config_honored(device, batch, seq):
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    x, w, b, _, out = run(device, batch, seq, compute_kernel_config=config, memory_config=ttnn.L1_MEMORY_CONFIG)
    check(out, x, w, b, ttnn.L1_MEMORY_CONFIG)


def test_default_compute_kernel_config():
    config = default_compute_kernel_config()
    assert config.math_fidelity == ttnn.MathFidelity.HiFi2
    assert config.fp32_dest_acc_en is True
    assert config.math_approx_mode is False
    assert default_compute_kernel_config() is not config


@pytest.mark.parametrize("bias_shape", [(FFN,), (1, FFN)])
def test_bias_shapes(device, bias_shape):
    x, w, b = make_inputs(2, 64)
    out = linear_gelu(
        to_device(x, device, ttnn.L1_MEMORY_CONFIG),
        to_device(w, device, ttnn.DRAM_MEMORY_CONFIG),
        to_device(b.reshape(bias_shape), device, ttnn.DRAM_MEMORY_CONFIG),
    )
    check(out, x, w, b, ttnn.L1_MEMORY_CONFIG)


def test_inputs_not_modified(device):
    x, w, b, x_dev, out = run(device, 2, 64)
    assert torch.equal(ttnn.to_torch(x_dev), x)
    check(out, x, w, b, ttnn.L1_MEMORY_CONFIG)


# S % 32 != 0: padded output rows must be exactly 0 whatever x's padding holds.
@pytest.mark.parametrize("batch, seq", [(1, 8), (2, 37), (3, 100), (8, 500)])
@pytest.mark.parametrize("pad_value", [0.0, 1.0e4, float("nan")])
def test_non_aligned_padding_is_zero(device, batch, seq, pad_value):
    x, w, b, _, out = run(device, batch, seq, pad_value=pad_value, memory_config=ttnn.L1_MEMORY_CONFIG)
    check(out, x, w, b, ttnn.L1_MEMORY_CONFIG)
    padded = read_padded(out)
    assert padded.shape[-2] == ceil_tile(seq)
    tail = padded[:, :, seq:, :]
    assert torch.isfinite(tail).all()
    assert (tail == 0).all()


@pytest.mark.parametrize(
    "x_memory_config, out_memory_config",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
def test_dram_placement(device, x_memory_config, out_memory_config):
    x, w, b, _, out = run(device, 4, 128, x_memory_config=x_memory_config, memory_config=out_memory_config)
    check(out, x, w, b, out_memory_config)


def test_fp32_dest_acc_off_refused(device, expect_error):
    x, w, b = make_inputs(1, 32)
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    with expect_error(SupportRefusal, ""):
        linear_gelu(
            to_device(x, device, ttnn.L1_MEMORY_CONFIG),
            to_device(w, device, ttnn.DRAM_MEMORY_CONFIG),
            to_device(b, device, ttnn.DRAM_MEMORY_CONFIG),
            compute_kernel_config=config,
        )


@pytest.mark.parametrize(
    "x_shape, w_shape, b_shape",
    [
        ((1, 32, HIDDEN), (HIDDEN, FFN), (FFN,)),  # x rank 3
        ((1, 2, 32, HIDDEN), (HIDDEN, FFN), (FFN,)),  # x dim 1 != 1
        ((1, 1, 32, HIDDEN), (HIDDEN + 32, FFN), (FFN,)),  # K mismatch
        ((1, 1, 32, HIDDEN), (HIDDEN, FFN), (FFN - 32,)),  # bias length
        ((1, 1, 32, HIDDEN), (HIDDEN, FFN), (2, FFN)),  # bias not one row
    ],
)
def test_malformed_raises_value_error(device, expect_error, x_shape, w_shape, b_shape):
    tensors = [
        to_device(torch.randn(shape).to(torch.bfloat16), device, memory_config)
        for shape, memory_config in (
            (x_shape, ttnn.L1_MEMORY_CONFIG),
            (w_shape, ttnn.DRAM_MEMORY_CONFIG),
            (b_shape, ttnn.DRAM_MEMORY_CONFIG),
        )
    ]
    with expect_error(ValueError, ""):
        linear_gelu(*tensors)


def test_sequence_limit_refused(device, expect_error):
    x, w, b = make_inputs(1, 544)
    with expect_error(SupportRefusal, ""):
        linear_gelu(
            to_device(x, device, ttnn.DRAM_MEMORY_CONFIG),
            to_device(w, device, ttnn.DRAM_MEMORY_CONFIG),
            to_device(b, device, ttnn.DRAM_MEMORY_CONFIG),
        )
