# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tt_py_test_utils_common.utility_functions import comp_allclose_and_pcc

TEST_PADDING_VALUE = -42


def get_backward_tensors(output_grad_shape, input_grad_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT
    torch_output_grad = torch.randint(-2, 3, output_grad_shape, dtype=cpu_dtype, requires_grad=True)
    torch_input_grad = torch.randint(-2, 3, input_grad_shape, dtype=cpu_dtype)

    tt_output_grad = ttnn.Tensor(torch_output_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_input_grad = ttnn.Tensor(torch_input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_output_grad, tt_input_grad, torch_output_grad


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [2],
        [8, 64, 64],
        [5, 5, 5, 5, 1, 1, 1],
        [10, 3, 18, 20],
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.float32),
    ],
)
def test_cumprod_normal(dim, shape, dtypes, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        for _ in range(2):
            torch_input_tensor = torch.randn(shape, dtype=dtypes[0])
            torch_result_tensor = torch.cumprod(torch_input_tensor, dim)
            ttnn_input_tensor = ttnn.from_torch(
                torch_input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device
            )
            ttnn_input_tensor = ttnn.fill_implicit_tile_padding(ttnn_input_tensor, TEST_PADDING_VALUE)
            ttnn_result_tensor = ttnn.cumprod(ttnn_input_tensor, dim, dtype=dtypes[1])

            # assert metadata
            assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
            assert torch_input_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_result_tensor.shape

            ttnn_result_torch = ttnn.to_torch(ttnn_result_tensor)
            if dtypes[0] == torch.float32:
                pcc_threshold = 0.999
                rtol = 0.09
                atol = 1.47
                frobenius_threshold = 0.02
            else:
                pcc_threshold = 0.999
                rtol = 0.03
                atol = 1.04
                frobenius_threshold = 0.01
            # test for equivalance
            assert_numeric_metrics(
                torch_result_tensor,
                ttnn_result_torch,
                pcc_threshold=pcc_threshold,
                rtol=rtol,
                atol=atol,
                frobenius_threshold=frobenius_threshold,
            )
        assert device.num_program_cache_entries() >= 1
    else:
        pytest.skip(f"skipping for dim == {dim} and shape == {shape}")


# low pcc issue.https://github.com/tenstorrent/tt-metal/issues/40878
@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [2],
        [8, 64, 64],
        [5, 5, 5, 5, 1, 1, 1],
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.float32),
    ],
)
def test_cumprod_backward(dim, shape, dtypes, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        (torch_dtype, ttnn_dtype) = dtypes

        # Generate integer input on [-2; 2];
        # by generating around 0, this avoids FP-related issues when adding large sums with small inputs
        # which are not handled yet
        torch_input_tensor = torch.randint(-2, 3, size=shape, dtype=torch_dtype, requires_grad=True)

        (tt_output_grad, tt_input_grad, torch_output_grad) = get_backward_tensors(shape, shape, device)

        torch_output = torch.cumprod(torch_input_tensor, dim)
        torch_output.backward(torch_output_grad)

        tt_input_grad_cpu = ttnn.to_torch(
            ttnn.cumprod(tt_output_grad, dim, dtype=ttnn_dtype, reverse_order=True, out=tt_input_grad)
        )

        assert tt_input_grad_cpu.shape == torch_input_tensor.grad.shape

        # test for equivalance
        rtol = atol = 0.1
        assert comp_allclose_and_pcc(torch_input_tensor.grad, tt_input_grad_cpu, pcc=0.999, rtol=rtol, atol=atol)

    else:
        pytest.skip(f"skipping for dim == {dim} and shape == {shape}")


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [1],
        [8, 64, 64],
        [5, 5, 5, 5, 1, 1, 1],
        [1, 1, 20, 16],
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
    ],
)
def test_cumprod_preallocated(dim, shape, dtypes, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        for _ in range(2):
            torch_input_tensor = torch.randn(shape, dtype=dtypes[0])
            torch_preallocated_tensor = torch.zeros_like(torch_input_tensor)
            torch_result_tensor = torch.cumprod(torch_input_tensor, dim, out=torch_preallocated_tensor)
            ttnn_input_tensor = ttnn.from_torch(
                torch_input_tensor, dtype=dtypes[1], layout=ttnn.Layout.TILE, device=device
            )
            ttnn_input_tensor = ttnn.fill_implicit_tile_padding(ttnn_input_tensor, TEST_PADDING_VALUE)
            ttnn_preallocated_tensor = ttnn.zeros_like(ttnn_input_tensor)
            ttnn_result_tensor = ttnn.cumprod(ttnn_input_tensor, dim, out=ttnn_preallocated_tensor)

            # assert metadata
            assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
            assert ttnn_preallocated_tensor.shape == ttnn_result_tensor.shape
            assert ttnn_preallocated_tensor.dtype == ttnn_result_tensor.dtype
            assert torch_input_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_result_tensor.shape

            ttnn_result_torch = ttnn.to_torch(ttnn_result_tensor)
            ttnn_preallocated_torch = ttnn.to_torch(ttnn_preallocated_tensor)
            pcc_threshold = 0.999
            rtol = 0.09
            atol = 1.47
            frobenius_threshold = 0.01
            for expected, actual in (
                (torch_result_tensor, ttnn_result_torch),
                (torch_preallocated_tensor, ttnn_preallocated_torch),
            ):
                # test for equivalance
                assert_numeric_metrics(
                    expected,
                    actual,
                    pcc_threshold=pcc_threshold,
                    rtol=rtol,
                    atol=atol,
                    frobenius_threshold=frobenius_threshold,
                )
    else:
        pytest.skip(f"skipping for dim == {dim} and shape == {shape}")


# The preallocated `out` tensor must live on device: the on-device check in
# validate_output_tensor fires before any other validation, so a host `out`
# would mask the check each row targets. Each row asserts the exact message of
# the check it exercises.
@pytest.mark.parametrize(
    "dim, input_shape, output_shape, torch_dtype, input_dtype, output_dtype, memory_config, layout, error_msg",
    [
        (
            -10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "The requested accumulation axis is -10, while the input tensor has rank 9",
        ),  # input_rank vs dim
        (
            10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "The requested accumulation axis is 10, while the input tensor has rank 9",
        ),  # input_rank vs dim
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "Shape mismatch: input tensor shape",
        ),  # input_shape vs output_shape
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 1],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "Shape mismatch: input tensor shape",
        ),  # input_shape vs output_shape
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.ROW_MAJOR,
            "The provided input tensor has a non-tile layout: ROW_MAJOR",
        ),  # unsupported layout
    ],
)
def test_cumprod_failing_cases(
    dim,
    input_shape,
    output_shape,
    torch_dtype,
    input_dtype,
    output_dtype,
    memory_config,
    layout,
    error_msg,
    device,
    expect_error,
):
    torch.manual_seed(0)
    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=layout, device=device, memory_config=memory_config
    )
    ttnn_preallocated_tensor = ttnn.zeros(output_shape, dtype=output_dtype, layout=ttnn.Layout.TILE, device=device)
    with expect_error(RuntimeError, error_msg):
        ttnn.cumprod(ttnn_input_tensor, memory_config=memory_config, dim=dim, out=ttnn_preallocated_tensor)


def _cumprod_calltrace(input_tensor, **kwargs):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        output_tensor = ttnn.cumprod(input_tensor, **kwargs)
    finally:
        captured_graph = ttnn.graph.end_graph_capture()
    return output_tensor, ttnn.graph.extract_calltrace(captured_graph)


def _skip_without_cumprod_tile(device):
    if device.arch() == ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar has no cumprod_tile, so cumprod keeps the permuted path there")


def _exact_cumprod_input(size, dim, torch_dtype):
    # x[k] = +-2^(e[k] - e[k-1]) with random e in [-20, 20], so every prefix product is +-2^e[k]: exact in
    # bf16 and fp32, and a wrong, missing or extra factor changes it. About a quarter of the rows get one zero.
    e = torch.randint(-20, 21, size)
    e_prev = torch.cat([torch.zeros_like(e.narrow(dim, 0, 1)), e.narrow(dim, 0, e.shape[dim] - 1)], dim)
    sign = torch.randint(0, 2, size) * 2 - 1
    x = sign * torch.exp2((e - e_prev).double())
    x[torch.rand(size) < 0.25 / size[dim]] = 0
    return x.to(torch_dtype)


@pytest.mark.parametrize("torch_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "size, dim",
    [
        ([4096], 0),  # 1D
        ([1, 4096], -1),
        ([4096, 1], -2),
        ([33, 35, 37], -1),
        ([33, 35, 37], -2),
        ([100, 64, 64], -1),  # both core groups
        ([100, 64, 64], -2),
        ([2, 3, 5, 33, 128], -1),
    ],
)
def test_cumprod_tile_axis_runs_in_place(size, dim, torch_dtype, device):
    """Forward bf16/fp32 cumprod along H or W scans the tiles in place (#24824) and is exact."""
    _skip_without_cumprod_tile(device)
    torch.manual_seed(29112024)
    torch_input = _exact_cumprod_input(size, dim, torch_dtype)
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor, calltrace = _cumprod_calltrace(input_tensor, dim=dim)

    assert "PermuteDeviceOperation" not in calltrace, calltrace
    assert calltrace.count("tt::tt_metal::create_device_tensor") == 1, calltrace
    expected = torch.cumprod(torch_input.double(), dim).to(torch_dtype)
    output = ttnn.to_torch(output_tensor)
    assert torch.equal(output, expected)
    if device.arch() == ttnn.device.Arch.BLACKHOLE and torch_dtype == torch.float32:
        assert torch.equal(torch.signbit(output), torch.signbit(expected))


def _with_special_values(torch_input, dim):
    # Puts 0, -0, +-inf, NaN, a huge and a tiny value each into scans of their own, at the first position and on
    # both sides of each tile boundary (31/32 and 63/64), where the running product moves into the next tile.
    moved = torch_input.movedim(dim, -1).contiguous()
    scans = moved.view(-1, moved.shape[-1])
    k = 0
    for value in (0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e30, 1e-30):
        for position in (0, 31, 32, 63, 64):
            scans[k, position] = value
            k += 1
    return moved.movedim(-1, dim).contiguous()


@pytest.mark.parametrize("values", ["near_one", "special"])
@pytest.mark.parametrize("torch_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [-1, -2])
def test_cumprod_tile_axis_matches_dim0_path(dim, torch_dtype, values, device):
    """Same bits as the permuted path's kernel (dim 0 of a 4D tensor needs no permute), on full-precision inputs and
    on special values. NaNs only have to be at the same places."""
    _skip_without_cumprod_tile(device)
    torch.manual_seed(29112024)
    if values == "near_one":
        torch_input = 1.0 + 0.01 * torch.randn([5, 70, 70])
    else:
        torch_input = _with_special_values(torch.empty([5, 70, 70]).uniform_(-3, 3), dim)
    torch_input = torch_input.to(torch_dtype)
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)
    moved = torch_input.movedim(dim, 0)
    dim0_tensor = ttnn.from_torch(moved.unsqueeze(1).contiguous(), device=device, layout=ttnn.Layout.TILE)

    tile_axis = ttnn.to_torch(ttnn.cumprod(input_tensor, dim=dim))
    dim0 = ttnn.to_torch(ttnn.cumprod(dim0_tensor, dim=0)).squeeze(1).movedim(0, dim)

    nan = torch.isnan(dim0)
    assert torch.equal(torch.isnan(tile_axis), nan)
    bits = torch.int16 if torch_dtype == torch.bfloat16 else torch.int32
    assert torch.equal(tile_axis[~nan].view(bits), dim0[~nan].view(bits))


def test_cumsum_cumprod_tile_axis_program_cache(device):
    """Interleaved cumsum/cumprod over bf16/fp32 and H/W: one program per configuration, and cache hits on new input
    buffers give exact results."""
    _skip_without_cumprod_tile(device)
    configs = [
        ("cumsum", torch.bfloat16, -1),
        ("cumprod", torch.bfloat16, -1),
        ("cumsum", torch.bfloat16, -2),
        ("cumprod", torch.bfloat16, -2),
        ("cumprod", torch.float32, -1),
        ("cumprod", torch.float32, -2),
    ]
    size = [2, 64, 96]
    torch.manual_seed(29112024)
    device.clear_program_cache()
    inputs = []  # kept alive, so that every call reads a new input buffer
    for repeat in range(2):
        for op, torch_dtype, dim in configs:
            if op == "cumsum":
                torch_input = torch.randint(-2, 3, size).to(torch_dtype)  # the partial sums stay exact in bf16
            else:
                torch_input = _exact_cumprod_input(size, dim, torch_dtype)
            inputs.append(ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE))
            output = ttnn.to_torch(getattr(ttnn, op)(inputs[-1], dim=dim))
            expected = getattr(torch, op)(torch_input.double(), dim).to(torch_dtype)
            assert torch.equal(output, expected), (repeat, op, torch_dtype, dim)
        assert device.num_program_cache_entries() == len(configs)


@pytest.mark.parametrize(
    "torch_dtype, reverse_order",
    [
        (torch.int32, False),
        (torch.bfloat16, True),
    ],
)
def test_cumprod_tile_axis_fallback_permutes(torch_dtype, reverse_order, device):
    """int32 and reverse cumprod still take the permuted path, and stay correct."""
    torch.manual_seed(29112024)
    size, dim = [33, 35, 20], -1  # products of [-2, 2] over 20 steps are exact in bf16 and int32
    torch_input = torch.randint(-2, 3, size, dtype=torch_dtype)
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)

    output_tensor, calltrace = _cumprod_calltrace(input_tensor, dim=dim, reverse_order=reverse_order)

    assert "PermuteDeviceOperation" in calltrace, calltrace
    expected = torch.cumprod(torch_input.flip(dim) if reverse_order else torch_input, dim)
    expected = expected.flip(dim) if reverse_order else expected
    assert torch.equal(ttnn.to_torch(output_tensor, dtype=torch_dtype), expected)


def test_cumprod_host_input_raises(expect_error):
    """A host tensor reaches the tile-axis check before any device check: it must raise, not crash."""
    input_tensor = ttnn.from_torch(torch.ones(64, 64), dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE)
    with expect_error(RuntimeError, "is_device_tensor"):
        ttnn.cumprod(input_tensor, dim=-1)
