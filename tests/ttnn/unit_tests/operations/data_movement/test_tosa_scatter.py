# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_allclose


def select_torch_dtype(ttnn_dtype):
    if ttnn_dtype is ttnn.bfloat16:
        return torch.bfloat16
    if ttnn_dtype is ttnn.float32:
        return torch.float32
    if ttnn_dtype is ttnn.uint8:
        return torch.uint8
    if ttnn_dtype is ttnn.int32:
        return (
            torch.int64
        )  # !!! there is a strict requirement for the index tensor in Torch to be int64, and there is no int64 in ttnn


@pytest.mark.parametrize(
    "N, K, W, C, input_dtype, index_dtype, input_layout",
    [
        (1, 1, 1, 1, ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        (20, 40, 40, 10, ttnn.float32, ttnn.uint32, ttnn.Layout.ROW_MAJOR),
        (20, 10, 10, 10, ttnn.bfloat16, ttnn.uint32, ttnn.Layout.TILE),
        (20, 10, 10, 10, ttnn.float32, ttnn.uint32, ttnn.Layout.ROW_MAJOR),
        (20, 10, 10, 10, ttnn.bfloat16, ttnn.uint32, ttnn.Layout.ROW_MAJOR),
    ],
)
def test_tosa_scatter_normal(N, K, W, C, input_dtype, index_dtype, input_layout, device):
    torch.manual_seed(0)
    input_torch_dtype = select_torch_dtype(input_dtype)

    input_shape = [N, K, C]
    index_shape = [N, W]
    source_shape = [N, W, C]

    torch_input = torch.randn(input_shape, dtype=input_torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=input_layout, device=device)
    torch_index = torch.randint(0, K, index_shape, dtype=torch.int64)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=input_layout, device=device)
    torch_source = torch.randn(source_shape, dtype=input_torch_dtype)
    ttnn_source = ttnn.from_torch(torch_source, dtype=input_dtype, layout=input_layout, device=device)
    dim = 1

    # adapt torch_index (expand [N, W] into [N, W, C])
    torch_index = torch_index.unsqueeze(-1).expand([N, W, C])

    torch_output = torch.scatter(torch_input, dim=dim, index=torch_index, src=torch_source)
    for _ in range(2):
        ttnn_output = ttnn.tosa_scatter(ttnn_input, ttnn_index, ttnn_source)
        assert ttnn_output.shape == ttnn_input.shape
        assert ttnn_output.dtype == ttnn_input.dtype
        assert_allclose(ttnn.to_torch(ttnn_output), torch_output, rtol=1e-3)


# tosa_scatter builds the device op directly instead of going through ttnn::scatter, so it needs
# its own empty-operand guard: without one, a zero extent reached the program factory's
# logical_volume() / input_shape[-1] work split and SIGFPE'd the host process. See issue #56881.
@pytest.mark.parametrize(
    "N, K, W, C",
    [
        (2, 4, 3, 0),  # zero C - every operand is empty
        (2, 4, 0, 5),  # zero W - empty index and source, non-empty input
        (0, 4, 3, 5),  # zero N - every operand is empty
    ],
)
@pytest.mark.parametrize("input_layout", [ttnn.Layout.ROW_MAJOR, ttnn.Layout.TILE])
def test_tosa_scatter_zero_volume(N, K, W, C, input_layout, device):
    torch.manual_seed(0)

    torch_input = torch.randn([N, K, C], dtype=torch.bfloat16)
    torch_index = torch.randint(0, max(K, 1), [N, W], dtype=torch.int64)
    torch_source = torch.randn([N, W, C], dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=input_layout, device=device)
    ttnn_index = ttnn.from_torch(torch_index, dtype=ttnn.uint32, layout=input_layout, device=device)
    ttnn_source = ttnn.from_torch(torch_source, dtype=ttnn.bfloat16, layout=input_layout, device=device)

    torch_output = torch.scatter(
        torch_input, dim=1, index=torch_index.unsqueeze(-1).expand([N, W, C]), src=torch_source
    )

    ttnn_output = ttnn.tosa_scatter(ttnn_input, ttnn_index, ttnn_source)

    assert ttnn_output.shape == ttnn_input.shape, f"shape {ttnn_output.shape} != input {ttnn_input.shape}"
    assert ttnn_output.dtype == ttnn_input.dtype, f"dtype {ttnn_output.dtype} != input {ttnn_input.dtype}"
    result = ttnn.to_torch(ttnn_output)
    if result.numel():
        assert_allclose(result, torch_output, rtol=1e-3)


# Same as test_scatter_zero_volume_rejects_invalid_operands: the empty path runs the device
# operation's validation itself, and nothing else would catch its removal.
@pytest.mark.parametrize(
    "input_dtype, index_dtype, source_dtype, expected_message",
    [
        (ttnn.bfloat16, ttnn.int32, ttnn.float32, "input_dtype differs from src_dtype"),
        (ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, "index_dtype"),
    ],
)
def test_tosa_scatter_zero_volume_rejects_invalid_operands(
    input_dtype, index_dtype, source_dtype, expected_message, device, expect_error
):
    def make(shape, dtype):
        torch_dtype = select_torch_dtype(dtype)
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch_dtype), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

    with expect_error(RuntimeError, expected_message):
        ttnn.tosa_scatter(make([2, 0, 4], input_dtype), make([2, 0], index_dtype), make([2, 0, 4], source_dtype))


@pytest.mark.parametrize("index_dtype", [ttnn.int32, ttnn.uint32])
def test_tosa_scatter_index_above_uint16(index_dtype, device):
    """Indices past 65535 must reach the right rows.

    The index tensor used to be cast to UINT16 on the host, so anything above 65535 wrapped
    modulo 65536 onto a lower row that still passed the kernel's bounds check, and the scatter
    silently wrote to the wrong place.
    """
    torch.manual_seed(0)
    N, K, W, C = 1, 70000, 4, 1

    # 69999 and 66000 wrap to 4463 and 464 under a UINT16 cast; 65535 and 3 are unaffected
    torch_index = torch.tensor([[69999, 66000, 65535, 3]], dtype=torch.int64)
    torch_input = torch.zeros([N, K, C], dtype=torch.bfloat16)
    torch_source = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.bfloat16)

    layout = ttnn.Layout.ROW_MAJOR
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=layout, device=device)
    ttnn_source = ttnn.from_torch(torch_source, dtype=ttnn.bfloat16, layout=layout, device=device)

    torch_output = torch.scatter(
        torch_input, dim=1, index=torch_index.unsqueeze(-1).expand([N, W, C]), src=torch_source
    )
    ttnn_output = ttnn.to_torch(ttnn.tosa_scatter(ttnn_input, ttnn_index, ttnn_source))

    # name the wrapped rows directly, so a regression reports where it landed rather than a norm
    written = (ttnn_output.reshape(-1) != 0).nonzero().flatten().tolist()
    assert written == [3, 65535, 66000, 69999], f"scatter wrote rows {written}"
    assert_allclose(ttnn_output, torch_output, rtol=1e-3)
