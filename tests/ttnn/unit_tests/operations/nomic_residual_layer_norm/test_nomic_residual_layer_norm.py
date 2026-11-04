# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for nomic_residual_layer_norm (immutable spec; see op_design.md).

LayerNorm(x + residual) * gamma + beta over the hidden axis (768), x and the residual in the op's own L1 layout
(input_memory_config), the output in output_memory_config's layout. Tolerances are the golden suite's
(eval/golden_tests/nomic_residual_layer_norm/helpers.py TOLERANCES): PCC >= 0.99999 and RMS error <= 0.005 of the
reference's standard deviation, against a float64 reference.

Shapes pin every per-core row count of the plan (1, 2, 3, 4 tile rows a core), a partial last shard (3x400,
11x352), a padded sequence (3x400, 8x500) and the three focus shapes (2x512, 4x512, 8x512).
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.nomic_residual_layer_norm import nomic_residual_layer_norm
from ttnn.operations.nomic_residual_layer_norm import (
    default_compute_kernel_config,
    input_memory_config,
    output_memory_config,
)

HIDDEN = 768
EPSILON = 1e-5
PCC_THRESHOLD = {ttnn.bfloat16: 0.99999}
RMS_THRESHOLD = {ttnn.bfloat16: 0.005}


def round_to_tf32(tensor):
    """fp32 rounded to nearest even at TF32's 10-bit mantissa (how the model holds gamma)."""
    bits = tensor.float().contiguous().view(torch.int32)
    bits = (bits + 0xFFF + ((bits >> 13) & 1)) & ~0x1FFF
    return bits.view(torch.float32)


def torch_reference(x, gamma, beta, residual=None, epsilon=EPSILON):
    y = x.double()
    if residual is not None:
        y = y + residual.double()
    return torch.nn.functional.layer_norm(
        y, (HIDDEN,), gamma.double().reshape(-1), beta.double().reshape(-1), eps=epsilon
    )


def make_inputs(shape, with_residual, constant_rows=()):
    torch.manual_seed(42)
    x = torch.randn(shape)
    residual = torch.randn(shape) if with_residual else None
    for row, value in constant_rows:
        index = (row // shape[-2], 0, row % shape[-2])
        x[index] = value
        if residual is not None:
            residual[index] = 0.0
    gamma = round_to_tf32(1.0 + 0.25 * torch.randn(HIDDEN))
    beta = (0.1 * torch.randn(HIDDEN)).to(torch.bfloat16)
    x = x.to(torch.bfloat16)
    residual = residual.to(torch.bfloat16) if residual is not None else None
    return x, residual, gamma, beta


def to_device(x, residual, gamma, beta, device):
    layout = input_memory_config(tuple(x.shape), device)

    def activation(tensor):
        return ttnn.from_torch(
            tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout
        )

    def parameter(tensor, dtype):
        return ttnn.from_torch(
            tensor.reshape(1, 1, 1, HIDDEN),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return (
        activation(x),
        activation(residual) if residual is not None else None,
        parameter(gamma, ttnn.float32),
        parameter(beta, ttnn.bfloat16),
    )


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def check(got, expected, dtype=ttnn.bfloat16):
    assert got.shape == expected.shape, f"shape {tuple(got.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(got).all(), "non-finite output"
    measured_pcc = pcc(got, expected)
    rms = float(((got.double() - expected) ** 2).mean().sqrt() / expected.std())
    assert measured_pcc >= PCC_THRESHOLD[dtype], f"PCC {measured_pcc:.7f} < {PCC_THRESHOLD[dtype]}"
    assert rms <= RMS_THRESHOLD[dtype], f"RMS error / std {rms:.5f} > {RMS_THRESHOLD[dtype]}"


SHAPES = [
    (2, 1, 512, 768),  # M = 32, focus shape, 1 tile row a core
    (4, 1, 512, 768),  # M = 64, focus shape, 2 tile rows a core
    (8, 1, 512, 768),  # M = 128, focus shape, 4 tile rows a core
    (5, 1, 512, 768),  # M = 80, 3 tile rows a core, partial last shard per batch
    (3, 1, 400, 768),  # M = 39, padded sequence, 2 tile rows a core, partial last shard
    (11, 1, 352, 768),  # M = 121, 4 tile rows a core, partial last shard, 99 cores (wrapping groups)
    (4, 1, 288, 768),  # M = 36, 108 cores (wrapping groups)
    (8, 1, 500, 768),  # M = 128, padded sequence
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[2]}")
@pytest.mark.parametrize("with_residual", [True, False], ids=["residual", "no_residual"])
def test_nomic_residual_layer_norm(device, shape, with_residual):
    x, residual, gamma, beta = make_inputs(shape, with_residual)
    expected = torch_reference(x, gamma, beta, residual)

    tt_x, tt_residual, tt_gamma, tt_beta = to_device(x, residual, gamma, beta, device)
    assert tt_x.memory_config() == input_memory_config(shape, device)

    output = nomic_residual_layer_norm(
        tt_x,
        tt_gamma,
        tt_beta,
        residual=tt_residual,
        epsilon=EPSILON,
        compute_kernel_config=default_compute_kernel_config(),
    )

    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == output_memory_config(shape, device, tt_x.memory_config())
    assert output.memory_config().buffer_type == ttnn.BufferType.L1
    check(ttnn.to_torch(output).float(), expected)


@pytest.mark.parametrize("shape", [(2, 1, 512, 768), (8, 1, 512, 768)], ids=lambda s: f"{s[0]}x{s[2]}")
def test_constant_rows_give_beta(device, shape):
    rows = shape[0] * shape[2]
    constants = [(0, 0.5), (rows // 3, 10.0), (rows // 2, -37.0), (rows - 1, 100.0)]
    x, residual, gamma, beta = make_inputs(shape, True, constant_rows=constants)
    tt_x, tt_residual, tt_gamma, tt_beta = to_device(x, residual, gamma, beta, device)
    output = nomic_residual_layer_norm(tt_x, tt_gamma, tt_beta, residual=tt_residual, epsilon=EPSILON)
    got = ttnn.to_torch(output).float().reshape(rows, HIDDEN)
    assert torch.isfinite(got).all(), "non-finite output"
    for row, value in constants:
        error = (got[row] - beta.float()).abs().max().item()
        assert error <= 1e-2, f"constant row {row} (value {value}): max |out - beta| = {error:.3e}"


def test_default_compute_kernel_config():
    config = default_compute_kernel_config()
    assert config.math_fidelity == ttnn.MathFidelity.HiFi4
    assert config.fp32_dest_acc_en is True
    assert config.math_approx_mode is False


def test_input_layout_is_l1_sharded_for_every_focus_shape(device):
    for batch in (2, 4, 8):
        shape = (batch, 1, 512, 768)
        config = input_memory_config(shape, device)
        assert config.buffer_type == ttnn.BufferType.L1
        assert config.is_sharded()
        m = batch * math.ceil(512 / 32)
        assert 32 <= m <= 128
