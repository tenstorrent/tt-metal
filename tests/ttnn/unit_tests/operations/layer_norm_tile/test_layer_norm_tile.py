# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for layer_norm (layer_norm_tile design).

Contract: bf16 TILE input (B, 1, S, 768) in L1 interleaved memory, 1024 <= B*S <= 4096,
S <= 512, S tile-aligned or not. Gamma/beta are required and come in two forms:
(768,) TILE in DRAM, or (1, 1, 24, 32) ROW_MAJOR in L1.
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm import layer_norm

PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

WIDTH = 768

# (B, 1, S, 768); comment gives tensor_row_tiles = B * ceil(S / 32).
SHAPES = [
    (32, 1, 32, WIDTH),  # 32 tile-rows, fewer than the grid
    (2, 1, 512, WIDTH),  # 32
    (16, 1, 128, WIDTH),  # 64
    (12, 1, 256, WIDTH),  # 96, ragged split across cores
    (8, 1, 512, WIDTH),  # 128
    (128, 1, 32, WIDTH),  # 128, many batches
    (11, 1, 100, WIDTH),  # 44, S not tile-aligned
    (3, 1, 500, WIDTH),  # 48, S not tile-aligned
    (64, 1, 17, WIDTH),  # 64, S < 32
    (7, 1, 300, WIDTH),  # 70, S not aligned, ragged split
    (124, 1, 33, WIDTH),  # 248, heavy padding, several blocks per core
]

AFFINE_FORMS = ["tile_dram", "rm_l1"]


def reference_layer_norm(x, gamma, beta, epsilon):
    return torch.nn.functional.layer_norm(
        x.to(torch.float32),
        (x.shape[-1],),
        weight=gamma.to(torch.float32),
        bias=beta.to(torch.float32),
        eps=epsilon,
    )


def pcc(expected, actual):
    expected = expected.to(torch.float64).flatten()
    actual = actual.to(torch.float64).flatten()
    if torch.equal(expected, actual):
        return 1.0
    return torch.corrcoef(torch.stack([expected, actual]))[0, 1].item()


def make_affine(torch_vec, form, device):
    if form == "tile_dram":
        return ttnn.from_torch(
            torch_vec,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert form == "rm_l1"
    return ttnn.from_torch(
        torch_vec.reshape(1, 1, WIDTH // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def run_and_check(device, shape, affine_form, *, epsilon=None, compute_kernel_config=None):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn(WIDTH, dtype=torch.float32).to(torch.bfloat16)
    torch_beta = torch.randn(WIDTH, dtype=torch.float32).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_gamma = make_affine(torch_gamma, affine_form, device)
    ttnn_beta = make_affine(torch_beta, affine_form, device)

    kwargs = {}
    if epsilon is not None:
        kwargs["epsilon"] = epsilon
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta, **kwargs)

    assert output.layout == ttnn.TILE_LAYOUT
    assert output.dtype == ttnn.bfloat16
    assert tuple(output.shape) == tuple(shape)
    assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert output.memory_config().buffer_type == ttnn.BufferType.L1

    expected = reference_layer_norm(torch_input, torch_gamma, torch_beta, 1e-5 if epsilon is None else epsilon)
    actual = ttnn.to_torch(output).to(torch.float32)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    measured = pcc(expected, actual)
    assert measured >= PCC_BY_DTYPE[ttnn.bfloat16], f"PCC {measured} below threshold for {shape}, {affine_form}"


@pytest.mark.parametrize("affine_form", AFFINE_FORMS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(str(d) for d in s))
def test_layer_norm_tile(device, shape, affine_form):
    run_and_check(device, shape, affine_form)


CONFIG_SHAPES = [
    (32, 1, 32, WIDTH),
    (8, 1, 512, WIDTH),
    (11, 1, 100, WIDTH),
]


@pytest.mark.parametrize("affine_form", AFFINE_FORMS)
@pytest.mark.parametrize("shape", CONFIG_SHAPES, ids=lambda s: "x".join(str(d) for d in s))
@pytest.mark.parametrize(
    "config_name",
    ["fp32_acc_off", "lofi", "full_config_eps_1e6"],
)
def test_layer_norm_tile_compute_config(device, shape, affine_form, config_name):
    epsilon = None
    if config_name == "fp32_acc_off":
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
        )
    elif config_name == "lofi":
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
        )
    else:
        epsilon = 1e-6
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )
    run_and_check(device, shape, affine_form, epsilon=epsilon, compute_kernel_config=cfg)


def test_layer_norm_tile_rejects_non_bf16_dtype(device, expect_error):
    shape = (32, 1, 32, WIDTH)
    torch.manual_seed(42)
    ttnn_input = ttnn.from_torch(
        torch.randn(shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_gamma = make_affine(torch.randn(WIDTH).to(torch.bfloat16), "tile_dram", device)
    ttnn_beta = make_affine(torch.randn(WIDTH).to(torch.bfloat16), "tile_dram", device)
    with expect_error(ValueError, "dtype"):
        layer_norm(ttnn_input, ttnn_gamma, ttnn_beta)
