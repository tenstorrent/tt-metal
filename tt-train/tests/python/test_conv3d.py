# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.ops.conv.conv3d`` against ``torch.nn.functional.conv3d``.

A few cases against the torch reference; the full case matrix lives in the C++ test. ttml takes channels-last
activations ``[N, D, H, W, C]``; PyTorch uses ``[N, C, D, H, W]``. Weights are in the PyTorch layout
``[C_out, C_in, kD, kH, kW]`` in both.
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


def _round_to_bf16(array):
    """Round host data to the precision the op computes in (bf16 for every float storage dtype)."""
    return torch.from_numpy(array).to(torch.bfloat16).to(torch.float32)


def _assert_close(actual, expected, what):
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
    input_requires_grad=True,
    layout=ttnn.Layout.ROW_MAJOR,
    dtype=BF16,
):
    torch.manual_seed(0)
    np.random.seed(0)
    D, H, W = spatial

    x_torch = _round_to_bf16(np.random.uniform(-1, 1, (N, C_in, D, H, W)).astype(np.float32))
    w_torch = _round_to_bf16(np.random.uniform(-0.5, 0.5, (C_out, C_in, *kernel)).astype(np.float32))
    b_torch = _round_to_bf16(np.random.uniform(-1, 1, (C_out,)).astype(np.float32))
    x_torch.requires_grad_(input_requires_grad)
    w_torch.requires_grad_(True)
    b_torch.requires_grad_(True)

    out_torch = torch.nn.functional.conv3d(x_torch, w_torch, b_torch, stride=stride, padding=padding, dilation=dilation)
    grad_out_torch = _round_to_bf16(np.random.uniform(-1, 1, tuple(out_torch.shape)).astype(np.float32))
    out_torch.backward(grad_out_torch)

    x_ttml = _to_ttml(x_torch.detach().numpy().transpose(0, 2, 3, 4, 1), layout, dtype, input_requires_grad)
    w_ttml = _to_ttml(w_torch.detach().numpy(), layout, dtype, True)
    b_ttml = _to_ttml(b_torch.detach().numpy().reshape(1, 1, 1, C_out), layout, dtype, True)

    out_ttml = ttml.ops.conv.conv3d(x_ttml, w_ttml, b_ttml, stride=stride, padding=padding, dilation=dilation)

    expected_out = out_torch.detach().numpy().transpose(0, 2, 3, 4, 1)
    assert out_ttml.shape() == list(expected_out.shape)
    _assert_close(out_ttml.to_numpy(FP32), expected_out, "forward")

    out_ttml.set_grad_from_tensor(_to_ttml(grad_out_torch.numpy().transpose(0, 2, 3, 4, 1), layout, dtype, False))
    out_ttml.backward(False)

    if input_requires_grad:
        _assert_close(
            x_ttml.get_grad_tensor().to_numpy(FP32), x_torch.grad.numpy().transpose(0, 2, 3, 4, 1), "grad_input"
        )
    else:
        assert not x_ttml.is_grad_initialized(), "frozen input received a gradient"
    _assert_close(w_ttml.get_grad_tensor().to_numpy(FP32), w_torch.grad.numpy(), "grad_weight")
    _assert_close(b_ttml.get_grad_tensor().to_numpy(FP32).reshape(C_out), b_torch.grad.numpy(), "grad_bias")


def test_conv3d_batch_stride_padding():
    _run_case(N=2, spatial=(5, 6, 7), stride=(2, 2, 2), padding=(1, 1, 1))


def test_conv3d_dilation_with_stride_and_padding():
    _run_case(N=2, spatial=(7, 8, 9), dilation=(2, 1, 2), stride=(1, 2, 1), padding=(2, 1, 2))


def test_conv3d_padding_beyond_kernel_span():
    # span = 2 * (3 - 1) = 4; padding 5 takes the dX crop path with a dilated kernel
    _run_case(spatial=(5, 6, 7), dilation=(2, 2, 2), padding=(5, 5, 5))


@pytest.mark.parametrize("layout", [ttnn.Layout.ROW_MAJOR, ttnn.Layout.TILE], ids=["row_major", "tile"])
def test_conv3d_unaligned_channels_with_batch(layout):
    _run_case(N=2, C_in=24, C_out=40, padding=(1, 1, 1), layout=layout)


# fp32 storage exercises the autocast path (parameters held in float32, computed in bf16), not fp32 compute.
@pytest.mark.parametrize("dtype", [BF16, FP32], ids=["bf16", "fp32_storage"])
def test_conv3d_patch_embed_kernel(dtype):
    # even kernel volume: channel block drops to 16, so the 16-channel input runs unpadded
    _run_case(C_in=16, C_out=40, kernel=(1, 2, 2), stride=(1, 2, 2), dtype=dtype)


def test_conv3d_patch_embed_wide_output():
    # Wan2.1 1.3B width: 1536 output channels is 96 blocks of 16 in the dX transposed conv
    _run_case(C_in=16, C_out=1536, spatial=(2, 4, 4), kernel=(1, 2, 2), stride=(1, 2, 2))


WAN_A14B_DIM = 5120


def test_conv3d_patch_embed_wan_a14b_width():
    # the latent never needs a gradient, so forward, dW and db run without the dX transposed conv
    _run_case(
        C_in=16, C_out=WAN_A14B_DIM, spatial=(2, 4, 4), kernel=(1, 2, 2), stride=(1, 2, 2), input_requires_grad=False
    )


def test_conv3d_patch_embed_wan_a14b_width_rejects_input_grad(expect_error):
    # dX would run the transposed conv over 5120 / 16 = 320 channel blocks, more than any grid has cores
    x = _zeros((1, 2, 4, 4, 16))
    x.set_requires_grad(True)
    w = _zeros((WAN_A14B_DIM, 16, 1, 2, 2))
    with expect_error(ValueError, "transposed conv .* channel blocks"):
        ttml.ops.conv.conv3d(x, w, stride=(1, 2, 2))


def _zeros(shape, dtype=np.float32, layout=ttnn.Layout.ROW_MAJOR, new_type=BF16):
    return ttml.autograd.Tensor.from_numpy(np.zeros(shape, dtype=dtype), layout=layout, new_type=new_type)


def test_conv3d_rejects_unsupported_arguments(expect_error):
    x = _zeros((1, 4, 5, 6, 32))
    w = _zeros((32, 32, 3, 3, 3))

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
