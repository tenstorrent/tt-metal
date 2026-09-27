# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for rms_norm_ttnn — the immutable spec.

DO NOT MODIFY. The implementer makes this pass; it is not adjusted to fit an
implementation.

The op is the five-stage pipeline

    1. t = x + residual_input_tensor        (optional, BEFORE the statistics)
    2. m = mean(t^2, dim=-1, keepdim=True)
    3. y = t / sqrt(m + epsilon)
    4. y = y * weight                       (optional, per-channel)
    5. y = y + bias                         (optional, per-channel, AFTER 4)

Stage 1 cannot be hoisted out and stage 5 cannot be folded into stage 4, so the
reference below consumes every operand and the parametrizations below exercise
each presence combination.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn

# PCC keyed by dtype — the same thresholds the golden suite uses.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}

EPSILON = 1e-12


# ---------------------------------------------------------------------------
# PyTorch reference — consumes EVERY operand
# ---------------------------------------------------------------------------


def torch_rms_norm(x, *, epsilon=EPSILON, weight=None, bias=None, residual=None):
    """The five stages, in fp32, in order."""
    original_dtype = x.dtype
    t = x.to(torch.float32)
    if residual is not None:
        t = t + residual.to(torch.float32)
    if t.numel() == 0:
        return t.to(original_dtype)
    if t.dim() == 0:
        mean_sq = t * t
    else:
        mean_sq = torch.mean(t * t, dim=-1, keepdim=True)
    y = t / torch.sqrt(mean_sq + epsilon)
    if weight is not None:
        y = y * weight.to(torch.float32).reshape(-1)
    if bias is not None:
        y = y + bias.to(torch.float32).reshape(-1)
    return y.to(original_dtype)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_device(t, device, *, dtype, layout):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _per_channel(width, torch_dtype):
    return torch.randn(width, dtype=torch.float32).to(torch_dtype)


def _run(
    device,
    shape,
    *,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    with_weight=False,
    with_bias=False,
    with_residual=False,
    operand_dtype=None,
    operand_layout=None,
    epsilon=EPSILON,
    compute_kernel_config=None,
    pcc=None,
):
    torch.manual_seed(42)

    torch_dtype = TORCH_DTYPE[dtype]
    operand_dtype = operand_dtype if operand_dtype is not None else dtype
    operand_layout = operand_layout if operand_layout is not None else layout
    operand_torch_dtype = TORCH_DTYPE[operand_dtype]

    width = shape[-1] if len(shape) >= 1 else 1

    torch_x = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    ttnn_x = _to_device(torch_x, device, dtype=dtype, layout=layout)

    kwargs = {"epsilon": epsilon}
    torch_kwargs = {"epsilon": epsilon}

    if with_weight:
        torch_w = _per_channel(width, operand_torch_dtype)
        kwargs["weight"] = _to_device(
            torch_w.reshape(1, 1, 1, width), device, dtype=operand_dtype, layout=operand_layout
        )
        torch_kwargs["weight"] = torch_w

    if with_bias:
        torch_b = _per_channel(width, operand_torch_dtype)
        kwargs["bias"] = _to_device(torch_b.reshape(1, 1, 1, width), device, dtype=operand_dtype, layout=operand_layout)
        torch_kwargs["bias"] = torch_b

    if with_residual:
        torch_r = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
        kwargs["residual_input_tensor"] = _to_device(torch_r, device, dtype=dtype, layout=layout)
        torch_kwargs["residual"] = torch_r

    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    ttnn_out = rms_norm_ttnn(ttnn_x, **kwargs)

    assert ttnn_out.layout == layout, f"output layout {ttnn_out.layout} != input layout {layout}"
    assert list(ttnn_out.shape) == list(shape), f"output shape {list(ttnn_out.shape)} != {list(shape)}"

    expected = torch_rms_norm(torch_x, **torch_kwargs)
    actual = ttnn.to_torch(ttnn_out)

    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), pcc or PCC[dtype])


# ---------------------------------------------------------------------------
# 1. shapes x layouts — the core rectangle
# ---------------------------------------------------------------------------

SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi tile
    (2, 4, 32, 96),  # multi batch, non-square
    (1, 1, 128, 1024),  # wide hidden
    (1, 1, 32, 50),  # W non tile-aligned
    (1, 1, 47, 64),  # H non tile-aligned
    (128, 512),  # rank 2
    (2, 64, 256),  # rank 3
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_rms_norm_ttnn_shapes_and_layouts(device, shape, layout):
    _run(device, shape, layout=layout)


# ---------------------------------------------------------------------------
# 2. the optional-operand presence axis — every combination the spec names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "with_weight, with_bias, with_residual",
    [
        (False, False, False),  # no_gamma
        (True, False, False),  # gamma
        (True, True, False),  # gamma_bias
        (False, True, False),  # bias only, no scale before it
        (False, False, True),  # residual only
        (True, True, True),  # gamma_bias_residual
    ],
    ids=["no_gamma", "gamma", "gamma_bias", "bias", "residual", "gamma_bias_residual"],
)
@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 1, 32, 50), (2, 64, 256)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_rms_norm_ttnn_optional_operands(device, shape, layout, with_weight, with_bias, with_residual):
    _run(
        device,
        shape,
        layout=layout,
        with_weight=with_weight,
        with_bias=with_bias,
        with_residual=with_residual,
    )


# ---------------------------------------------------------------------------
# 3. dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 1, 32, 256)])
def test_rms_norm_ttnn_dtypes(device, dtype, shape):
    _run(device, shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, with_weight=True)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
def test_rms_norm_ttnn_dest_accumulation(device, dtype, fp32_dest_acc_en):
    """Both DEST widths are accepted at every dtype, float32 included."""
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=True,
    )
    _run(
        device,
        (1, 1, 64, 256),
        dtype=dtype,
        with_weight=True,
        with_bias=True,
        compute_kernel_config=config,
        # a 16-bit accumulator over an fp32 input is scored at what 16 bits can deliver
        pcc=PCC[ttnn.bfloat16] if (dtype == ttnn.float32 and not fp32_dest_acc_en) else PCC[dtype],
    )


# ---------------------------------------------------------------------------
# 4. per-channel operands whose dtype / layout differ from the input's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operand_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("operand_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_rms_norm_ttnn_mixed_operand_format(device, operand_dtype, operand_layout):
    """bf16 activations with fp32 weights is the ordinary mixed-precision case."""
    _run(
        device,
        (1, 1, 64, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        with_weight=True,
        with_bias=True,
        operand_dtype=operand_dtype,
        operand_layout=operand_layout,
    )


# ---------------------------------------------------------------------------
# 5. epsilon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [0.0, 1e-12, 1e-5, 1e-2])
def test_rms_norm_ttnn_epsilon(device, epsilon):
    _run(device, (1, 1, 64, 128), epsilon=epsilon, with_weight=True)


def test_rms_norm_ttnn_residual_cancellation(device):
    """residual = -x leaves an all-zero row: epsilon is then the whole denominator.

    Catches both a dropped epsilon (NaN) and statistics taken over x instead of
    x + residual.
    """
    torch.manual_seed(42)
    shape = (1, 1, 32, 64)
    torch_x = torch.randn(shape, dtype=torch.float32)

    ttnn_x = _to_device(torch_x, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    ttnn_r = _to_device(-torch_x, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    out = ttnn.to_torch(rms_norm_ttnn(ttnn_x, epsilon=1e-5, residual_input_tensor=ttnn_r))

    assert torch.isfinite(out).all(), "a cancelled residual must not produce NaN/Inf"
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=1e-6)


def test_rms_norm_ttnn_bias_dominates(device):
    """A bias two orders above the normalized value.

    Catches bias-applied-before-normalize and bias-folded-into-the-scale.
    """
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    torch_x = torch.randn(shape, dtype=torch.float32)
    torch_w = torch.randn(128, dtype=torch.float32)
    torch_b = torch.randn(128, dtype=torch.float32) * 100.0

    ttnn_out = rms_norm_ttnn(
        _to_device(torch_x, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
        epsilon=EPSILON,
        weight=_to_device(torch_w.reshape(1, 1, 1, 128), device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
        bias=_to_device(torch_b.reshape(1, 1, 1, 128), device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
    )

    expected = torch_rms_norm(torch_x, epsilon=EPSILON, weight=torch_w, bias=torch_b)
    assert_with_pcc(expected, ttnn.to_torch(ttnn_out), PCC[ttnn.float32])


# ---------------------------------------------------------------------------
# 6. degenerate ranks and volumes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (64,),  # rank 1
        (50,),  # rank 1, W non-aligned
    ],
)
def test_rms_norm_ttnn_rank_one(device, shape):
    _run(device, shape, layout=ttnn.ROW_MAJOR_LAYOUT)


def test_rms_norm_ttnn_rank_zero(device):
    """A scalar: mean(x^2) over a one-element row is x^2."""
    _run(device, (), layout=ttnn.ROW_MAJOR_LAYOUT)


def test_rms_norm_ttnn_rank_zero_is_zero_not_nan(device):
    """x = 0 => 0 / sqrt(0 + epsilon) = 0. Dropping epsilon turns this into NaN."""
    torch_x = torch.zeros((), dtype=torch.float32)
    ttnn_x = _to_device(torch_x, device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.to_torch(rms_norm_ttnn(ttnn_x, epsilon=1e-12))

    assert torch.isfinite(out).all(), "a zero scalar must come out as zero, never NaN"
    torch.testing.assert_close(out.reshape(()), torch.zeros((), dtype=out.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 1, 1, 32, 64), (2, 1, 1, 47, 50)])
def test_rms_norm_ttnn_rank_five(device, shape):
    _run(device, shape, layout=ttnn.TILE_LAYOUT, with_weight=True)


@pytest.mark.parametrize("shape", [(1, 1, 0, 64), (0, 64), (1, 1, 32, 0)])
def test_rms_norm_ttnn_zero_volume(device, shape):
    """A zero-volume input returns a copy — no normalization work, no error."""
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ttnn_x = _to_device(torch_x, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out = rms_norm_ttnn(ttnn_x)

    assert list(out.shape) == list(shape)
    assert out.layout == ttnn.TILE_LAYOUT
    assert ttnn.to_torch(out).numel() == 0


# ---------------------------------------------------------------------------
# 7. compute-kernel-config surface
# ---------------------------------------------------------------------------


def test_rms_norm_ttnn_default_compute_config_is_exported(device):
    """`None` resolves through one exported factory (see references/precision_convention.md)."""
    from ttnn.operations.rms_norm_ttnn import default_compute_kernel_config

    cfg = default_compute_kernel_config()
    assert cfg.math_fidelity == ttnn.MathFidelity.HiFi4
    assert cfg.math_approx_mode is True
    assert cfg.fp32_dest_acc_en is False

    # a fresh object per call, never a shared mutable constant
    assert default_compute_kernel_config() is not cfg


@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4])
def test_rms_norm_ttnn_math_fidelity_is_not_gated(device, math_fidelity):
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=True,
        math_approx_mode=True,
    )
    _run(device, (1, 1, 64, 128), with_weight=True, compute_kernel_config=config)


def test_rms_norm_ttnn_device_compute_kernel_config_is_accepted(device):
    """The device compute-kernel config is normalized at the door, not refused."""
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    _run(device, (1, 1, 64, 128), with_weight=True, with_bias=True, compute_kernel_config=config)


# ---------------------------------------------------------------------------
# 8. validation
# ---------------------------------------------------------------------------

# Either exception type is accepted; the regex pins only that the message NAMES the
# thing it refused, which is the whole point of refusing with a message.
_REFUSED = (ValueError, RuntimeError)


def test_rms_norm_ttnn_refuses_short_per_channel_operand(device, expect_error):
    x = _to_device(torch.randn(1, 1, 64, 128), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    w = _to_device(torch.randn(1, 1, 1, 64), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(_REFUSED, "(?i)weight"):
        rms_norm_ttnn(x, weight=w)


def test_rms_norm_ttnn_refuses_weight_and_bias_at_different_layouts(device, expect_error):
    x = _to_device(torch.randn(1, 1, 64, 128), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    w = _to_device(torch.randn(1, 1, 1, 128), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = _to_device(torch.randn(1, 1, 1, 128), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(_REFUSED, "(?i)layout"):
        rms_norm_ttnn(x, weight=w, bias=b)


def test_rms_norm_ttnn_refuses_mismatched_residual(device, expect_error):
    x = _to_device(torch.randn(1, 1, 64, 128), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    r = _to_device(torch.randn(1, 1, 128, 128), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(_REFUSED, "(?i)residual"):
        rms_norm_ttnn(x, residual_input_tensor=r)


def test_rms_norm_ttnn_refuses_host_resident_input(expect_error):
    host = ttnn.from_torch(torch.randn(1, 1, 64, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(_REFUSED, "(?i)device|host|storage"):
        rms_norm_ttnn(host)
