# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for layer_norm_tile (immutable spec).

bf16 TILE input (B, 1, S, 768), aligned and non-aligned S, optional residual, gamma/beta as
ROW_MAJOR (1, 1, W/32, 32) or TILE (1, 1, 1, W), every interleaved L1/DRAM placement, selectable
compute config. test_reference_rows replays the stock ttnn.layer_norm reference calls; run it under
`scripts/run_safe_pytest.sh --profile` to report device kernel ns per row.
"""

import contextlib

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_tile import layer_norm

PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

DRAM = ttnn.BufferType.DRAM
L1 = ttnn.BufferType.L1
MEMORY_CONFIG = {DRAM: ttnn.DRAM_MEMORY_CONFIG, L1: ttnn.L1_MEMORY_CONFIG}
W = 768


def reference_layer_norm(x, gamma, beta, residual=None, epsilon=1e-5):
    z = x.to(torch.float32)
    if residual is not None:
        z = z + residual.to(torch.float32)
    return torch.nn.functional.layer_norm(
        z, (z.shape[-1],), weight=gamma.to(torch.float32), bias=beta.to(torch.float32), eps=epsilon
    )


def pcc(expected, actual):
    expected = expected.to(torch.float64).flatten()
    actual = actual.to(torch.float64).flatten()
    if torch.equal(expected, actual):
        return 1.0
    return torch.corrcoef(torch.stack([expected, actual]))[0, 1].item()


def to_device(tensor, device, buffer_type, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=MEMORY_CONFIG[buffer_type]
    )


def make_affine(vector, device, layout, buffer_type):
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        shaped = vector.reshape(1, 1, vector.numel() // 32, 32)
    else:
        shaped = vector.reshape(1, 1, 1, vector.numel())
    return to_device(shaped, device, buffer_type, layout=layout)


def compute_config(fp32_dest_acc_en=True, math_fidelity=ttnn.MathFidelity.HiFi4):
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = math_fidelity
    config.fp32_dest_acc_en = fp32_dest_acc_en
    config.math_approx_mode = False
    return config


@contextlib.contextmanager
def perf_window():
    try:
        from eval import metrics_plugin
    except ImportError:
        yield
        return
    with metrics_plugin.op_window():
        yield


def run_and_check(
    device,
    shape,
    *,
    with_residual=True,
    input_buf=DRAM,
    residual_buf=None,
    output_buf=None,
    affine_layout=ttnn.ROW_MAJOR_LAYOUT,
    affine_buf=L1,
    epsilon=None,
    config=None,
    measure=False,
):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_residual = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16) if with_residual else None
    torch_gamma = torch.randn(shape[-1], dtype=torch.float32).to(torch.bfloat16)
    torch_beta = torch.randn(shape[-1], dtype=torch.float32).to(torch.bfloat16)

    ttnn_input = to_device(torch_input, device, input_buf)
    ttnn_residual = (
        to_device(torch_residual, device, residual_buf if residual_buf is not None else input_buf)
        if with_residual
        else None
    )
    ttnn_gamma = make_affine(torch_gamma, device, affine_layout, affine_buf)
    ttnn_beta = make_affine(torch_beta, device, affine_layout, affine_buf)

    kwargs = {}
    if epsilon is not None:
        kwargs["epsilon"] = epsilon
    if config is not None:
        kwargs["compute_kernel_config"] = config
    if output_buf is not None:
        kwargs["memory_config"] = MEMORY_CONFIG[output_buf]

    window = perf_window() if measure else contextlib.nullcontext()
    with window:
        ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta, ttnn_residual, **kwargs)

    assert tuple(ttnn_output.shape) == tuple(shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.TILE_LAYOUT
    expected_buf = output_buf if output_buf is not None else input_buf
    assert ttnn_output.memory_config().buffer_type == expected_buf
    assert ttnn_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED

    expected = reference_layer_norm(
        torch_input, torch_gamma, torch_beta, torch_residual, epsilon=epsilon if epsilon is not None else 1e-5
    )
    actual = ttnn.to_torch(ttnn_output).to(torch.float32)
    assert torch.isfinite(actual).all()
    measured = pcc(expected, actual)
    assert measured >= PCC_BY_DTYPE[ttnn.bfloat16], f"PCC {measured} for shape {shape}"


SHAPES = [
    pytest.param((32, 1, 32, W), id="32x32_single_tile_row_per_batch"),
    pytest.param((8, 1, 128, W), id="8x128_multi_tile"),
    pytest.param((2, 1, 512, W), id="2x512_max_s"),
    pytest.param((16, 1, 256, W), id="16x256_multi_batch"),
    pytest.param((8, 1, 512, W), id="8x512_headline"),
    pytest.param((3, 1, 500, W), id="3x500_h_non_aligned"),
    pytest.param((64, 1, 17, W), id="64x17_sub_tile_s"),
    pytest.param((5, 1, 257, W), id="5x257_one_past_tile"),
]

RESIDUAL = pytest.mark.parametrize("with_residual", [True, False], ids=["residual", "no_residual"])


@RESIDUAL
@pytest.mark.parametrize("shape", SHAPES)
def test_shapes(device, shape, with_residual):
    run_and_check(device, shape, with_residual=with_residual)


@pytest.mark.parametrize(
    "input_buf,residual_buf,output_buf",
    [
        (L1, DRAM, DRAM),
        (DRAM, DRAM, DRAM),
        (L1, L1, L1),
        (DRAM, L1, L1),
        (L1, L1, DRAM),
        (DRAM, DRAM, L1),
    ],
    ids=["in_l1_res_dram_out_dram", "all_dram", "all_l1", "in_dram_res_l1_out_l1", "in_l1_res_l1_out_dram", "out_l1"],
)
def test_placements(device, input_buf, residual_buf, output_buf):
    run_and_check(device, (8, 1, 512, W), input_buf=input_buf, residual_buf=residual_buf, output_buf=output_buf)


@pytest.mark.parametrize("output_buf", [None, DRAM, L1], ids=["default", "dram", "l1"])
def test_output_memory_config_without_residual(device, output_buf):
    run_and_check(device, (11, 1, 100, W), with_residual=False, input_buf=DRAM, output_buf=output_buf)


@pytest.mark.parametrize("affine_buf", [DRAM, L1], ids=["affine_dram", "affine_l1"])
@pytest.mark.parametrize("affine_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["affine_rm", "affine_tile"])
@pytest.mark.parametrize("shape", [(8, 1, 128, W), (7, 1, 333, W)], ids=["8x128", "7x333"])
def test_affine_formats(device, shape, affine_layout, affine_buf):
    run_and_check(device, shape, affine_layout=affine_layout, affine_buf=affine_buf)


@pytest.mark.parametrize("epsilon", [1e-6, 1e-5, 1e-2])
def test_epsilon(device, epsilon):
    run_and_check(device, (4, 1, 256, W), epsilon=epsilon)


@RESIDUAL
@pytest.mark.parametrize(
    "fp32_dest_acc_en,math_fidelity",
    [
        (True, ttnn.MathFidelity.HiFi4),
        (False, ttnn.MathFidelity.HiFi4),
        (True, ttnn.MathFidelity.LoFi),
        (False, ttnn.MathFidelity.LoFi),
    ],
    ids=["fp32acc_hifi4", "bf16acc_hifi4", "fp32acc_lofi", "bf16acc_lofi"],
)
def test_compute_config(device, fp32_dest_acc_en, math_fidelity, with_residual):
    run_and_check(
        device,
        (3, 1, 500, W),
        with_residual=with_residual,
        config=compute_config(fp32_dest_acc_en, math_fidelity),
    )


def test_default_compute_config_factory():
    from ttnn.operations.layer_norm_tile import default_compute_kernel_config

    first = default_compute_kernel_config()
    second = default_compute_kernel_config()
    assert first is not second
    assert first.math_fidelity == ttnn.MathFidelity.HiFi4
    assert first.fp32_dest_acc_en


def test_prompt_import_path():
    from ttnn.operations.layer_norm import layer_norm as prompt_layer_norm

    assert prompt_layer_norm is layer_norm


def test_rejects_non_bfloat16_input(device, expect_error):
    shape = (2, 1, 512, W)
    x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    gamma = make_affine(torch.randn(W).to(torch.bfloat16), device, ttnn.ROW_MAJOR_LAYOUT, L1)
    beta = make_affine(torch.randn(W).to(torch.bfloat16), device, ttnn.ROW_MAJOR_LAYOUT, L1)
    with expect_error(ValueError, r"(?i)bfloat16|dtype"):
        layer_norm(x, gamma, beta)


def test_rejects_row_major_input(device, expect_error):
    shape = (2, 1, 512, W)
    x = to_device(torch.randn(shape).to(torch.bfloat16), device, DRAM, layout=ttnn.ROW_MAJOR_LAYOUT)
    gamma = make_affine(torch.randn(W).to(torch.bfloat16), device, ttnn.ROW_MAJOR_LAYOUT, L1)
    beta = make_affine(torch.randn(W).to(torch.bfloat16), device, ttnn.ROW_MAJOR_LAYOUT, L1)
    with expect_error(ValueError, r"(?i)tile|layout"):
        layer_norm(x, gamma, beta)


# Stock ttnn.layer_norm device kernel time on Blackhole p100a, default compute config.
# (B, S), input, residual (None = absent), output, affine layout, affine placement, stock us.
REFERENCE_ROWS = [
    pytest.param((8, 512), L1, DRAM, DRAM, ttnn.ROW_MAJOR_LAYOUT, L1, 55.0, id="8x512_in_l1_res_dram_out_dram"),
    pytest.param((8, 512), DRAM, DRAM, DRAM, ttnn.ROW_MAJOR_LAYOUT, L1, 70.0, id="8x512_all_dram"),
    pytest.param((8, 512), DRAM, None, DRAM, ttnn.ROW_MAJOR_LAYOUT, L1, 48.3, id="8x512_dram_no_residual"),
    pytest.param((8, 512), L1, L1, L1, ttnn.ROW_MAJOR_LAYOUT, L1, 43.3, id="8x512_all_l1"),
    pytest.param((8, 512), L1, None, L1, ttnn.ROW_MAJOR_LAYOUT, L1, 34.5, id="8x512_l1_no_residual"),
    pytest.param((8, 384), DRAM, DRAM, DRAM, ttnn.ROW_MAJOR_LAYOUT, L1, 47.0, id="8x384_all_dram"),
    pytest.param((32, 32), DRAM, DRAM, DRAM, ttnn.ROW_MAJOR_LAYOUT, L1, 24.8, id="32x32_all_dram"),
    pytest.param((8, 512), DRAM, DRAM, DRAM, ttnn.TILE_LAYOUT, DRAM, 125.4, id="8x512_all_dram_affine_tile_dram"),
]


@pytest.mark.parametrize("bs,input_buf,residual_buf,output_buf,affine_layout,affine_buf,stock_us", REFERENCE_ROWS)
def test_reference_rows(device, bs, input_buf, residual_buf, output_buf, affine_layout, affine_buf, stock_us):
    batch, seq = bs
    run_and_check(
        device,
        (batch, 1, seq, W),
        with_residual=residual_buf is not None,
        input_buf=input_buf,
        residual_buf=residual_buf,
        output_buf=output_buf,
        affine_layout=affine_layout,
        affine_buf=affine_buf,
        measure=True,
    )
