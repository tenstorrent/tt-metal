# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.layer_norm import layer_norm


SHAPE_VALUES = [
    (1, 1, 32, 32),
    (1, 1, 33, 64),
    (1, 1, 32, 50),
    (2, 3, 65, 97),
]


INPUT_CASE_VALUES = [
    (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    (ttnn.float32, ttnn.TILE_LAYOUT),
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    (ttnn.bfloat16, ttnn.TILE_LAYOUT),
    (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
]


AFFINE_MODE_VALUES = ["no_affine", "gamma_only", "gamma_beta"]


COMPUTE_CONFIG_CASES = [
    pytest.param("implicit_default", lambda: None, False, 1e-5, id="implicit_default"),
    pytest.param("explicit_default", ttnn.ComputeConfigDescriptor, False, 1e-5, id="explicit_default"),
    pytest.param(
        "lofi",
        lambda: ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
        False,
        1e-5,
        id="lofi",
    ),
    pytest.param(
        "fp32_dest_acc",
        lambda: ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
        False,
        1e-5,
        id="fp32_dest_acc",
    ),
    pytest.param(
        "full_config",
        lambda: ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        ),
        True,
        1e-6,
        id="full_config",
    ),
]


PCC_THRESHOLDS = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.999,
    ttnn.bfloat8_b: 0.99,
}


ALLCLOSE_TOLERANCES = {
    ttnn.float32: (0.20, 0.05),
    ttnn.bfloat16: (0.20, 0.05),
    ttnn.bfloat8_b: (0.25, 0.10),
}


DTYPE_LABELS = {
    ttnn.float32: "float32",
    ttnn.bfloat16: "bfloat16",
    ttnn.bfloat8_b: "bfloat8_b",
}


LAYOUT_LABELS = {
    ttnn.ROW_MAJOR_LAYOUT: "row_major",
    ttnn.TILE_LAYOUT: "tile",
}


def _torch_dtype_for(tt_dtype):
    if tt_dtype == ttnn.float32:
        return torch.float32
    return torch.bfloat16


def _make_ttnn_tensor(torch_tensor, *, device, tt_dtype, layout):
    return ttnn.from_torch(
        torch_tensor,
        dtype=tt_dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_affine_tensors(shape, *, device, tt_dtype):
    width = shape[-1]
    torch_dtype = _torch_dtype_for(tt_dtype)
    gamma_torch = torch.randn((1, 1, 1, width), dtype=torch_dtype)
    beta_torch = torch.randn((1, 1, 1, width), dtype=torch_dtype)
    gamma_tt = _make_ttnn_tensor(gamma_torch, device=device, tt_dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    beta_tt = _make_ttnn_tensor(beta_torch, device=device, tt_dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    return gamma_tt, beta_tt


def _torch_reference(input_torch, gamma_torch=None, beta_torch=None, epsilon=1e-5):
    input_fp32 = input_torch.to(torch.float32)
    width = input_fp32.shape[-1]
    weight = gamma_torch.reshape(width).to(torch.float32) if gamma_torch is not None else None
    bias = beta_torch.reshape(width).to(torch.float32) if beta_torch is not None else None
    output = torch.nn.functional.layer_norm(input_fp32, (width,), weight=weight, bias=bias, eps=epsilon)
    return output.to(input_torch.dtype)


def _assert_output_matches_reference(output_tt, expected_torch, shape, tt_dtype, layout):
    assert tuple(output_tt.shape) == shape
    assert output_tt.dtype == tt_dtype
    assert output_tt.layout == layout

    actual_torch = ttnn.to_torch(output_tt).to(torch.float32)
    expected_torch = expected_torch.to(torch.float32)
    assert_with_pcc(expected_torch, actual_torch, PCC_THRESHOLDS[tt_dtype])

    rtol, atol = ALLCLOSE_TOLERANCES[tt_dtype]
    assert torch.allclose(actual_torch, expected_torch, rtol=rtol, atol=atol), (
        f"layer_norm mismatch for dtype={tt_dtype}, layout={layout}, shape={shape}: "
        f"max_abs_diff={(actual_torch - expected_torch).abs().max().item():.6f}"
    )


def _make_positive_cases():
    cases = []
    for shape in SHAPE_VALUES:
        for tt_dtype, layout in INPUT_CASE_VALUES:
            for affine_mode in AFFINE_MODE_VALUES:
                if tt_dtype == ttnn.bfloat8_b and affine_mode != "no_affine":
                    continue
                cases.append(
                    pytest.param(
                        shape,
                        tt_dtype,
                        layout,
                        affine_mode,
                        id=f"{shape[-2]}x{shape[-1]}_{DTYPE_LABELS[tt_dtype]}_{LAYOUT_LABELS[layout]}_{affine_mode}",
                    )
                )
    return cases


@pytest.mark.parametrize("shape,tt_dtype,layout,affine_mode", _make_positive_cases())
def test_layer_norm_matches_torch(device, shape, tt_dtype, layout, affine_mode):
    torch.manual_seed(42)
    input_torch = torch.randn(shape, dtype=_torch_dtype_for(tt_dtype))
    input_tt = _make_ttnn_tensor(input_torch, device=device, tt_dtype=tt_dtype, layout=layout)

    gamma_tt = None
    beta_tt = None
    if affine_mode in {"gamma_only", "gamma_beta"}:
        gamma_tt, beta_tt = _make_affine_tensors(shape, device=device, tt_dtype=tt_dtype)
        if affine_mode == "gamma_only":
            beta_tt = None

    input_ref = ttnn.to_torch(input_tt)
    gamma_ref = ttnn.to_torch(gamma_tt) if gamma_tt is not None else None
    beta_ref = ttnn.to_torch(beta_tt) if beta_tt is not None else None
    expected = _torch_reference(input_ref, gamma_ref, beta_ref)

    if gamma_tt is not None and beta_tt is not None:
        output_tt = layer_norm(input_tt, gamma_tt, beta_tt)
    elif gamma_tt is not None:
        output_tt = layer_norm(input_tt, gamma_tt)
    else:
        output_tt = layer_norm(input_tt)

    _assert_output_matches_reference(output_tt, expected, shape, tt_dtype, layout)


@pytest.mark.parametrize(
    "tt_dtype,layout",
    [
        pytest.param(ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, id="float32_row_major"),
        pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, id="float32_tile"),
        pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="bfloat16_row_major"),
        pytest.param(ttnn.bfloat16, ttnn.TILE_LAYOUT, id="bfloat16_tile"),
    ],
)
@pytest.mark.parametrize("_config_name,config_factory,use_affine,epsilon", COMPUTE_CONFIG_CASES)
def test_layer_norm_compute_kernel_config(device, tt_dtype, layout, _config_name, config_factory, use_affine, epsilon):
    torch.manual_seed(42)
    shape = (2, 3, 65, 97)
    input_torch = torch.randn(shape, dtype=_torch_dtype_for(tt_dtype))
    input_tt = _make_ttnn_tensor(input_torch, device=device, tt_dtype=tt_dtype, layout=layout)

    gamma_tt = None
    beta_tt = None
    if use_affine:
        gamma_tt, beta_tt = _make_affine_tensors(shape, device=device, tt_dtype=tt_dtype)

    input_ref = ttnn.to_torch(input_tt)
    gamma_ref = ttnn.to_torch(gamma_tt) if gamma_tt is not None else None
    beta_ref = ttnn.to_torch(beta_tt) if beta_tt is not None else None
    expected = _torch_reference(input_ref, gamma_ref, beta_ref, epsilon=epsilon)

    compute_kernel_config = config_factory()
    if use_affine:
        output_tt = layer_norm(
            input_tt,
            gamma_tt,
            beta_tt,
            epsilon=epsilon,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_tt = layer_norm(
            input_tt,
            compute_kernel_config=compute_kernel_config,
        )

    _assert_output_matches_reference(output_tt, expected, shape, tt_dtype, layout)


@pytest.mark.parametrize(
    "tt_dtype,layout",
    [
        pytest.param(ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, id="float32_row_major"),
        pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, id="float32_tile"),
        pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="bfloat16_row_major"),
        pytest.param(ttnn.bfloat16, ttnn.TILE_LAYOUT, id="bfloat16_tile"),
    ],
)
def test_layer_norm_none_compute_kernel_config_matches_explicit_default(device, tt_dtype, layout):
    torch.manual_seed(42)
    shape = (1, 2, 65, 97)
    input_tt = _make_ttnn_tensor(
        torch.randn(shape, dtype=_torch_dtype_for(tt_dtype)), device=device, tt_dtype=tt_dtype, layout=layout
    )
    gamma_tt, beta_tt = _make_affine_tensors(shape, device=device, tt_dtype=tt_dtype)

    implicit_output = layer_norm(input_tt, gamma_tt, beta_tt, epsilon=1e-6)
    explicit_output = layer_norm(
        input_tt,
        gamma_tt,
        beta_tt,
        epsilon=1e-6,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(),
    )

    implicit_torch = ttnn.to_torch(implicit_output)
    explicit_torch = ttnn.to_torch(explicit_output)
    assert torch.equal(implicit_torch, explicit_torch)


def test_layer_norm_rejects_rank_lt_2(device):
    torch.manual_seed(42)
    input_tt = _make_ttnn_tensor(
        torch.randn((97,), dtype=torch.bfloat16), device=device, tt_dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    with pytest.raises((ValueError, RuntimeError)):
        layer_norm(input_tt)


def test_layer_norm_rejects_unsupported_input_dtype(device):
    input_tt = _make_ttnn_tensor(
        torch.randint(0, 17, (1, 1, 32, 64), dtype=torch.int32),
        device=device,
        tt_dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    with pytest.raises((ValueError, RuntimeError)):
        layer_norm(input_tt)


@pytest.mark.parametrize("which_affine", ["gamma", "beta"])
def test_layer_norm_rejects_affine_width_mismatch(device, which_affine):
    torch.manual_seed(42)
    input_shape = (1, 1, 32, 64)
    input_tt = _make_ttnn_tensor(
        torch.randn(input_shape, dtype=torch.bfloat16),
        device=device,
        tt_dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    gamma_tt, beta_tt = _make_affine_tensors(input_shape, device=device, tt_dtype=ttnn.bfloat16)

    bad_affine = _make_ttnn_tensor(
        torch.randn((1, 1, 1, 63), dtype=torch.bfloat16),
        device=device,
        tt_dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    if which_affine == "gamma":
        gamma_tt = bad_affine
    else:
        beta_tt = bad_affine

    with pytest.raises((ValueError, RuntimeError)):
        layer_norm(input_tt, gamma_tt, beta_tt)


@pytest.mark.parametrize("which_affine", ["gamma", "beta"])
def test_layer_norm_rejects_non_row_major_affine(device, which_affine):
    torch.manual_seed(42)
    input_shape = (1, 1, 32, 64)
    input_tt = _make_ttnn_tensor(
        torch.randn(input_shape, dtype=torch.bfloat16),
        device=device,
        tt_dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    gamma_tt, beta_tt = _make_affine_tensors(input_shape, device=device, tt_dtype=ttnn.bfloat16)

    bad_affine = _make_ttnn_tensor(
        torch.randn((1, 1, 1, input_shape[-1]), dtype=torch.bfloat16),
        device=device,
        tt_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    if which_affine == "gamma":
        gamma_tt = bad_affine
    else:
        beta_tt = bad_affine

    with pytest.raises((ValueError, RuntimeError)):
        layer_norm(input_tt, gamma_tt, beta_tt)


@pytest.mark.parametrize("which_affine", ["gamma", "beta"])
def test_layer_norm_rejects_affine_dtype_mismatch(device, which_affine):
    torch.manual_seed(42)
    input_shape = (1, 1, 32, 64)
    input_tt = _make_ttnn_tensor(
        torch.randn(input_shape, dtype=torch.float32),
        device=device,
        tt_dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    gamma_tt, beta_tt = _make_affine_tensors(input_shape, device=device, tt_dtype=ttnn.float32)

    bad_affine = _make_ttnn_tensor(
        torch.randn((1, 1, 1, input_shape[-1]), dtype=torch.bfloat16),
        device=device,
        tt_dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    if which_affine == "gamma":
        gamma_tt = bad_affine
    else:
        beta_tt = bad_affine

    with pytest.raises((ValueError, RuntimeError)):
        layer_norm(input_tt, gamma_tt, beta_tt)


def test_layer_norm_rejects_bfloat8_b_affine(device):
    torch.manual_seed(42)
    input_shape = (1, 1, 32, 64)
    input_tt = _make_ttnn_tensor(
        torch.randn(input_shape, dtype=torch.bfloat16),
        device=device,
        tt_dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )

    gamma_tt = _make_ttnn_tensor(
        torch.randn((1, 1, 1, input_shape[-1]), dtype=torch.bfloat16),
        device=device,
        tt_dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    with pytest.raises((ValueError, RuntimeError)):
        layer_norm(input_tt, gamma_tt)
