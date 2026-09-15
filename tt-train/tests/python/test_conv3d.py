# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.ops.conv.conv3d`` against ``torch.nn.functional.conv3d``.

ttml takes channels-last activations ``[N, D, H, W, C]``; PyTorch uses ``[N, C, D, H, W]``. Weights are in the
PyTorch layout ``[C_out, C_in / groups, kD, kH, kW]`` in both.
"""

import numpy as np
import pytest
import torch

import ttnn
import ttml

pytestmark = pytest.mark.requires_device

BF16 = ttnn.DataType.BFLOAT16
FP32 = ttnn.DataType.FLOAT32

# Float parameters are computed in bf16 (autograd reads them at HALF), so references are bf16-rounded and the bound
# is relative to the tensor's largest magnitude; PCC guards small entries.
REL_OF_MAX = 1e-2
ABS_FLOOR = 2e-2
MIN_PCC = 0.999


@pytest.fixture(autouse=True)
def _reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _round_to(array, dtype):
    """Round host data to the precision the op computes in (bf16 for every float storage dtype)."""
    del dtype
    return torch.from_numpy(array).to(torch.bfloat16).to(torch.float32)


def _assert_close(actual, expected, what, dtype):
    del dtype
    tolerance = ABS_FLOOR + REL_OF_MAX * np.abs(expected).max()
    max_err = np.abs(actual - expected).max()
    assert max_err <= tolerance, f"{what}: max abs err {max_err} > {tolerance}"
    pcc = np.corrcoef(actual.ravel(), expected.ravel())[0, 1]
    assert pcc > MIN_PCC, f"{what}: pcc {pcc} <= {MIN_PCC}"


def _to_ttml(array, layout, dtype, requires_grad):
    tensor = ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(array), layout=layout, new_type=dtype)
    tensor.set_requires_grad(requires_grad)
    return tensor


def _run_case(
    N=1,
    C_in=32,
    C_out=32,
    spatial=(4, 5, 6),
    kernel=(3, 3, 3),
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups=1,
    with_bias=True,
    bias_rank1=False,
    input_requires_grad=True,
    weight_requires_grad=True,
    bias_requires_grad=True,
    layout=ttnn.Layout.ROW_MAJOR,
    dtype=BF16,
):
    torch.manual_seed(0)
    np.random.seed(0)
    D, H, W = spatial

    x_torch = _round_to(np.random.uniform(-1, 1, (N, C_in, D, H, W)).astype(np.float32), dtype)
    w_torch = _round_to(np.random.uniform(-0.5, 0.5, (C_out, C_in // groups, *kernel)).astype(np.float32), dtype)
    b_torch = _round_to(np.random.uniform(-1, 1, (C_out,)).astype(np.float32), dtype) if with_bias else None
    x_torch.requires_grad_(input_requires_grad)
    w_torch.requires_grad_(weight_requires_grad)
    if with_bias:
        b_torch.requires_grad_(bias_requires_grad)

    out_torch = torch.nn.functional.conv3d(
        x_torch, w_torch, b_torch, stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    grad_out_torch = _round_to(np.random.uniform(-1, 1, tuple(out_torch.shape)).astype(np.float32), dtype)
    any_grad = input_requires_grad or weight_requires_grad or (with_bias and bias_requires_grad)
    if any_grad:
        out_torch.backward(grad_out_torch)

    x_ttml = _to_ttml(x_torch.detach().numpy().transpose(0, 2, 3, 4, 1), layout, dtype, input_requires_grad)
    w_ttml = _to_ttml(w_torch.detach().numpy(), layout, dtype, weight_requires_grad)
    b_ttml = None
    if with_bias:
        bias_np = b_torch.detach().numpy()
        # A rank-1 bias cannot be tilized on its own; the op accepts it in ROW_MAJOR.
        b_ttml = (
            _to_ttml(bias_np, ttnn.Layout.ROW_MAJOR, dtype, bias_requires_grad)
            if bias_rank1
            else _to_ttml(bias_np.reshape(1, 1, 1, C_out), layout, dtype, bias_requires_grad)
        )

    out_ttml = ttml.ops.conv.conv3d(
        x_ttml, w_ttml, b_ttml, stride=stride, padding=padding, dilation=dilation, groups=groups
    )

    expected_out = out_torch.detach().numpy().transpose(0, 2, 3, 4, 1)
    assert out_ttml.shape() == list(expected_out.shape)
    _assert_close(out_ttml.to_numpy(FP32), expected_out, "forward", dtype)

    if not any_grad:
        assert not out_ttml.get_requires_grad()
        return

    out_ttml.set_grad_from_tensor(_to_ttml(grad_out_torch.numpy().transpose(0, 2, 3, 4, 1), layout, dtype, False))
    out_ttml.backward(False)

    if input_requires_grad:
        _assert_close(
            x_ttml.get_grad_tensor().to_numpy(FP32), x_torch.grad.numpy().transpose(0, 2, 3, 4, 1), "grad_input", dtype
        )
    else:
        assert not x_ttml.is_grad_initialized(), "frozen input received a gradient"
    if weight_requires_grad:
        _assert_close(w_ttml.get_grad_tensor().to_numpy(FP32), w_torch.grad.numpy(), "grad_weight", dtype)
    else:
        assert not w_ttml.is_grad_initialized(), "frozen weight received a gradient"
    if with_bias:
        if bias_requires_grad:
            _assert_close(
                b_ttml.get_grad_tensor().to_numpy(FP32).reshape(C_out), b_torch.grad.numpy(), "grad_bias", dtype
            )
        else:
            assert not b_ttml.is_grad_initialized(), "frozen bias received a gradient"


# fp32 storage exercises the autocast path (parameters held in float32, computed in bf16), not fp32 compute.
@pytest.mark.parametrize("dtype", [BF16, FP32], ids=["bf16", "fp32_storage"])
def test_conv3d_forward_backward_with_bias(dtype):
    _run_case(padding=(1, 1, 1), dtype=dtype)


def test_conv3d_without_bias():
    _run_case(with_bias=False)


def test_conv3d_one_dimensional_bias():
    _run_case(bias_rank1=True, padding=(1, 1, 1))


def test_conv3d_batch_stride_padding():
    _run_case(N=2, spatial=(5, 6, 7), stride=(2, 2, 2), padding=(1, 1, 1))


def test_conv3d_asymmetric_stride_padding():
    _run_case(spatial=(6, 7, 8), stride=(1, 2, 3), padding=(0, 1, 2))


def test_conv3d_dilation():
    _run_case(spatial=(6, 7, 8), dilation=(2, 2, 2), padding=(2, 2, 2))


def test_conv3d_dilation_with_groups_stride_padding():
    _run_case(
        N=2, spatial=(7, 8, 9), C_in=64, C_out=64, groups=2, dilation=(2, 1, 2), stride=(1, 2, 1), padding=(2, 1, 2)
    )


def test_conv3d_per_axis_dilation():
    _run_case(spatial=(6, 8, 10), dilation=(1, 2, 3), padding=(1, 2, 3))


def test_conv3d_dilation_with_padding_beyond_span():
    # span = 2 * (3 - 1) = 4; padding 5 takes the dX crop path with a dilated kernel
    _run_case(spatial=(5, 6, 7), dilation=(2, 2, 2), padding=(5, 5, 5))


def test_conv3d_groups_with_dilation_and_unaligned_channels():
    _run_case(spatial=(5, 6, 7), C_in=24, C_out=40, groups=4, dilation=(1, 2, 2), padding=(1, 2, 2))


def test_conv3d_padding_larger_than_kernel_span():
    # 3x3x3 kernel has span 2; padding above it takes the dX crop path
    _run_case(padding=(3, 4, 3))


def test_conv3d_padding_larger_than_kernel_span_with_stride():
    _run_case(N=2, stride=(2, 2, 2), padding=(3, 3, 3))


def test_conv3d_output_channels_not_tile_aligned():
    _run_case(C_out=40, padding=(1, 1, 1))


def test_conv3d_input_channels_not_tile_aligned():
    _run_case(C_in=12, padding=(1, 1, 1))


def test_conv3d_rgb_like_input():
    _run_case(N=2, C_in=3, C_out=40, stride=(2, 1, 2), padding=(1, 1, 1))


@pytest.mark.parametrize("layout", [ttnn.Layout.ROW_MAJOR, ttnn.Layout.TILE], ids=["row_major", "tile"])
def test_conv3d_groups_with_unaligned_channels(layout):
    _run_case(N=2, C_in=24, C_out=40, groups=2, padding=(1, 1, 1), layout=layout)


@pytest.mark.parametrize("dtype", [BF16, FP32], ids=["bf16", "fp32_storage"])
def test_conv3d_patch_embed_kernel(dtype):
    # even kernel volume: channel block drops to 16, so padded C_in spans two weight blocks
    _run_case(C_in=16, C_out=40, kernel=(1, 2, 2), stride=(1, 2, 2), dtype=dtype)


def test_conv3d_patch_embed_wide_output():
    # production-like width: 1536 output channels is 96 blocks of 16 in the dX transposed conv
    _run_case(C_in=16, C_out=1536, spatial=(2, 4, 4), kernel=(1, 2, 2), stride=(1, 2, 2))


def test_conv3d_asymmetric_odd_kernel():
    _run_case(kernel=(1, 3, 3), padding=(0, 1, 1))


def test_conv3d_even_cubic_kernel_with_stride():
    _run_case(kernel=(2, 2, 2), stride=(2, 2, 2))


def test_conv3d_larger_channel_counts():
    _run_case(C_in=128, C_out=128, padding=(1, 1, 1))


def test_conv3d_tile_layout_tensors():
    _run_case(padding=(1, 1, 1), layout=ttnn.Layout.TILE)


@pytest.mark.parametrize("groups", [2, 4])
def test_conv3d_groups(groups):
    _run_case(C_in=64, C_out=32, groups=groups, padding=(1, 1, 1))


def test_conv3d_groups_with_stride():
    _run_case(N=2, C_in=64, C_out=64, groups=2, stride=(2, 2, 2), padding=(1, 1, 1))


def test_conv3d_frozen_weight_and_bias():
    _run_case(padding=(1, 1, 1), weight_requires_grad=False, bias_requires_grad=False)


def test_conv3d_frozen_input():
    _run_case(padding=(1, 1, 1), input_requires_grad=False)


def test_conv3d_frozen_weight_trainable_bias():
    _run_case(input_requires_grad=False, weight_requires_grad=False)


def test_conv3d_prepared_weights_match_raw():
    """Caller-prepared kernel-layout weights give the same forward and gradients as on-the-fly preparation."""
    np.random.seed(1)
    N, C_in, C_out, groups, kernel, padding = 1, 64, 40, 2, (3, 3, 3), (1, 1, 1)
    x_np = np.random.uniform(-1, 1, (N, 4, 5, 6, C_in)).astype(np.float32)
    w_np = np.random.uniform(-0.5, 0.5, (C_out, C_in // groups, *kernel)).astype(np.float32)
    dy_np = np.random.uniform(-1, 1, (N, 4, 5, 6, C_out)).astype(np.float32)

    def run(prepared):
        x = _to_ttml(x_np, ttnn.Layout.ROW_MAJOR, BF16, True)
        w = _to_ttml(w_np, ttnn.Layout.ROW_MAJOR, BF16, True)
        kwargs = dict(padding=padding, groups=groups)
        out = ttml.ops.conv.conv3d(x, w, None, prepared, **kwargs) if prepared else ttml.ops.conv.conv3d(x, w, **kwargs)
        out.set_grad_from_tensor(_to_ttml(dy_np, ttnn.Layout.ROW_MAJOR, BF16, False))
        out.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        return (
            out.to_numpy(FP32),
            x.get_grad_tensor().to_numpy(FP32),
            w.get_grad_tensor().to_numpy(FP32),
            w.get_value(),
        )

    raw_out, raw_dx, raw_dw, w_value = run(None)
    prepared = ttml.ops.conv.prepare_conv3d_weight(w_value, groups=groups)

    kvol = kernel[0] * kernel[1] * kernel[2]
    assert prepared.groups == groups and prepared.c_in_block == 32
    assert len(prepared.forward) == groups and len(prepared.transposed) == groups
    for form in (*prepared.forward, *prepared.transposed):
        assert list(form.shape) == [kvol * 32, 32]
        assert form.layout == ttnn.Layout.TILE

    pre_out, pre_dx, pre_dw, _ = run(prepared)
    np.testing.assert_array_equal(pre_out, raw_out)
    np.testing.assert_array_equal(pre_dx, raw_dx)
    np.testing.assert_array_equal(pre_dw, raw_dw)


def test_conv3d_prepared_weight_mismatch_is_rejected(expect_error):
    w = _zeros((40, 32, 3, 3, 3))
    x = _zeros((1, 4, 5, 6, 64))
    prepared_for_one_group = ttml.ops.conv.prepare_conv3d_weight(w.get_value(), groups=1)
    with expect_error(ValueError, "built for shape"):
        ttml.ops.conv.conv3d(x, w, None, prepared_for_one_group, groups=2)


def test_conv3d_prepared_weight_is_a_snapshot():
    """After the parameter's value changes, a prepared weight still computes with the values it was built from."""
    np.random.seed(2)
    x = _to_ttml(np.random.uniform(-1, 1, (1, 4, 5, 6, 32)).astype(np.float32), ttnn.Layout.ROW_MAJOR, BF16, False)
    old_np = np.random.uniform(-0.5, 0.5, (32, 32, 3, 3, 3)).astype(np.float32)
    new_np = np.random.uniform(-0.5, 0.5, (32, 32, 3, 3, 3)).astype(np.float32)
    w = _to_ttml(old_np, ttnn.Layout.ROW_MAJOR, BF16, False)

    prepared = ttml.ops.conv.prepare_conv3d_weight(w.get_value(), groups=1)
    with_old = ttml.ops.conv.conv3d(x, w, padding=(1, 1, 1)).to_numpy(FP32)

    w.assign(_to_ttml(new_np, ttnn.Layout.ROW_MAJOR, BF16, False))
    with_new = ttml.ops.conv.conv3d(x, w, padding=(1, 1, 1)).to_numpy(FP32)
    stale = ttml.ops.conv.conv3d(x, w, None, prepared, padding=(1, 1, 1)).to_numpy(FP32)

    np.testing.assert_array_equal(stale, with_old)
    assert np.abs(stale - with_new).max() > 0.1


def _zeros(shape, dtype=np.float32, layout=ttnn.Layout.ROW_MAJOR, new_type=BF16):
    return ttml.autograd.Tensor.from_numpy(np.zeros(shape, dtype=dtype), layout=layout, new_type=new_type)


def test_conv3d_rejects_unsupported_arguments(expect_error):
    x = _zeros((1, 4, 5, 6, 32))
    w = _zeros((32, 32, 3, 3, 3))

    with expect_error(ValueError, "groups must be non-zero"):
        ttml.ops.conv.conv3d(x, w, groups=0)
    with expect_error(ValueError, "weight in_channels .* groups"):
        # weight has 32 input columns, but C_in / groups = 32 / 2 = 16
        ttml.ops.conv.conv3d(x, w, groups=2)
    with expect_error(ValueError, "input channels .* divisible by groups"):
        ttml.ops.conv.conv3d(_zeros((1, 4, 5, 6, 30)), _zeros((32, 15, 3, 3, 3)), groups=4)
    with expect_error(ValueError, "out_channels .* divisible by groups"):
        ttml.ops.conv.conv3d(_zeros((1, 4, 5, 6, 96)), _zeros((32, 32, 3, 3, 3)), groups=3)
    with expect_error(ValueError, "padding_mode"):
        ttml.ops.conv.conv3d(x, w, padding_mode="replicate")
    with expect_error(ValueError, "stride"):
        ttml.ops.conv.conv3d(x, w, stride=(0, 1, 1))
    with expect_error(ValueError, "dilation"):
        ttml.ops.conv.conv3d(x, w, dilation=(1, 0, 1))
    with expect_error(ValueError, "effective kernel"):
        # dilation 3 makes the effective kernel 7 > D = 4 with no padding
        ttml.ops.conv.conv3d(x, w, dilation=(3, 1, 1))
    with expect_error(ValueError, "rank 5"):
        ttml.ops.conv.conv3d(_zeros((4, 5, 6, 32)), w)
    with expect_error(ValueError, "rank 5"):
        ttml.ops.conv.conv3d(x, _zeros((32, 32, 3, 3)))
    with expect_error(ValueError, "in_channels"):
        ttml.ops.conv.conv3d(x, _zeros((32, 64, 3, 3, 3)))
    # float32 vs bfloat16 never reaches the op (autograd reads floats at bf16); uint32 bypasses autocast
    u32 = dict(dtype=np.uint32, new_type=ttnn.DataType.UINT32)
    with expect_error(ValueError, "dtype"):
        ttml.ops.conv.conv3d(x, _zeros((32, 32, 3, 3, 3), **u32))
    with expect_error(ValueError, "dtype"):
        ttml.ops.conv.conv3d(x, w, _zeros((1, 32), **u32))
    with expect_error(ValueError, "bias"):
        ttml.ops.conv.conv3d(x, w, _zeros((1, 33)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
