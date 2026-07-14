# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc, is_blackhole, torch_random
from tests.ttnn.utils_for_testing import assert_numeric_metrics

TEST_PADDING_VALUE = -42


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [-1, -2, 0, (-2, -1), None])
@pytest.mark.parametrize("correction", [True, False])
@pytest.mark.parametrize("keepdim", [True, False])
def test_std(device, batch_size, h, w, dim, correction, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.std(torch_input_tensor, dim=dim, keepdim=keepdim, correction=correction)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.std(input_tensor, dim=dim, keepdim=keepdim, correction=correction)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence

    rtol = 0.01
    atol = 0.01
    frobenius = 0.005
    pcc = 0.9999
    if dim == (-2, -1):
        # For 2D reduction, all output values are close to 1, and we're using bfloat16,
        # so a rounding error of even 1 ULP impacts PCC.
        # ATOL/RTOL/Frobenius should catch any significant errors.
        pcc = 0.98

    outputs_all_finite = torch.isfinite(torch_output_tensor).all() and torch.isfinite(output_tensor).all()
    if outputs_all_finite and torch_output_tensor.numel() > 0:
        assert_numeric_metrics(
            torch_output_tensor,
            output_tensor,
            pcc_threshold=pcc,
            rtol=rtol,
            atol=atol,
            frobenius_threshold=frobenius,
        )
    else:
        passing, output_pcc = comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=pcc, rtol=rtol, atol=atol)
        assert passing, f"{output_pcc}, torch: {torch_output_tensor}, ttnn: {output_tensor}"


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
@pytest.mark.parametrize("dim", [None, [], -1, -2, (-2, -1)])
@pytest.mark.parametrize("keepdim", [True])
@pytest.mark.parametrize("correction", [True, False])
def test_var(device, batch_size, h, w, dim, keepdim, correction):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.var(torch_input_tensor, dim=dim, keepdim=keepdim, correction=correction)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.var(input_tensor, dim=dim, keepdim=keepdim, correction=correction)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert len(torch_output_tensor.shape) == len(output_tensor.shape)
    assert torch_output_tensor.shape == output_tensor.shape

    # test for equivalence
    rtol = 0.01
    atol = 0.01
    pcc = 0.99999
    frobenius = 0.007

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=pcc,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius,
    )


# Regression test for fp32 Welford variance precision under large mean offsets.
# Uses a bit-exact integer input where the true variance is known analytically:
# variance of N consecutive integers is (N^2 - 1) / 12 (population); with N=32 and Bessel's
# correction, sample variance = 32 * (32^2 - 1) / (12 * 31) = 88.0 exactly. Variance is
# translation-invariant *and* sign-invariant, so neither adding a large offset to every
# element nor flipping its sign should change the answer.the scalar is applied after the
# reduction as var(s*x) = s^2 * var(x).
# test covers all three reduction kernels (H, W, HW) and both code paths
@pytest.mark.parametrize("scalar", [1.0, 0, -1.0])
@pytest.mark.parametrize("offset", [0.0, 1e6])
@pytest.mark.parametrize("dim", [-1, -2, (-2, -1)])
def test_var_fp32_translation_invariance(device, dim, offset, scalar):
    correction = True
    # The input is read at full fp32 and reduced unscaled; scalar is now applied after the (unscaled, precise) Welford reduction
    # as var(s*x) = s^2 * var(x), so this case is accurate regardless of offset.
    N = 32
    seq = torch.arange(N, dtype=torch.float32) + offset
    # Lay out the input so the reduction axis is the integer sequence.
    if dim == -1:
        # Each row of the tile is the sequence; reducing along W gives var of [offset..offset+31].
        torch_input = seq.unsqueeze(0).expand(N, N).contiguous()
    else:
        # dim=-2 or dim=(-2,-1): each column is the sequence so H-reduce
        # (or HW-reduce of a rank-deficient tile) sees the sequence.
        torch_input = seq.unsqueeze(-1).expand(N, N).contiguous()
    torch_input = torch_input.unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 32)

    # Reference computed in fp64 so it isn't itself contaminated by any fp32 precision loss.
    torch_ref = torch.var((torch_input * scalar).to(torch.float64), dim=dim, keepdim=True, correction=correction)

    tt_in = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.var(tt_in, dim=dim, scalar=scalar, keepdim=True, correction=correction)
    actual = ttnn.to_torch(ttnn.from_device(tt_out))

    # Tight tolerances: the unscaled fp32 reduction is essentially exact, so we only allow
    # small accumulation noise from the SFPU Welford recurrence.
    assert_numeric_metrics(
        torch_ref,
        actual,
        rtol=1e-5,
        atol=1e-4,
        frobenius_threshold=1e-5,
        pcc_threshold=0.9999,
        check_ulp=False,
    )


# Regression test for fp32 Welford variance precision when do_scale=true (non-unity scalar)
# AND the reduction dimension crosses a tile boundary (Wt>1).
# Unlike test_var_fp32_translation_invariance above, which tests whether inputs preserve
# FP32 precision by using a large offset, this test checks that the FPU MUL result
# is preserved in FP32, which requires UnpackToDestFp32 on cb_scaled.
#
# Variance of N consecutive integers 0..N-1 is (N^2 - 1) / 12 (population); with Bessel's
# correction (sample variance) it is N * (N^2 - 1) / (12 * (N - 1)) = N * (N + 1) / 12. The
# torch.var below in fp64 computes the ground truth for any N; the formula is noted only to
# make the smallest case (N=33, sample variance 93.5; scaled by 2.0 -> 374.0) easy to verify
# by inspection.
#
# Parametrized across two N values to cover two Wt regimes of the wt-inner loop in
# welford_reduce_w with do_scale=true:
#   - N=33  -> Wt = ceil(33/32)  = 2   (smallest multi-tile case; original regression target)
#   - N=129 -> Wt = ceil(129/32) = 5   (deeper inner loop, exercises the per-iter UNPACK
#                                       hw_configure flip between cb_in's Default mode and
#                                       cb_scaled's UnpackToDestFp32 mode across many
#                                       iterations rather than just one boundary crossing)
@pytest.mark.parametrize("scalar", [2.0, -2.0, 0.5, 4.0])
@pytest.mark.parametrize("N", [33, 129], ids=["Wt2", "Wt5"])
def test_var_fp32_doscale_wt_gt_1(device, scalar, N):
    correction = True
    seq = torch.arange(N, dtype=torch.float32)
    # Each row of the input tile is the sequence; reducing along W (dim=-1) gives per-row var.
    # Shape (1, 1, 32, N): one H tile (Ht=1), Wt = ceil(N/32) W tiles.
    torch_input = seq.unsqueeze(0).expand(32, N).contiguous().unsqueeze(0).unsqueeze(0)

    # Reference in fp64 so it isn't contaminated by fp32 precision loss in torch.
    torch_ref = torch.var((torch_input * scalar).to(torch.float64), dim=-1, keepdim=True, correction=correction)

    tt_in = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.var(tt_in, dim=-1, keepdim=True, correction=correction, scalar=scalar)
    actual = ttnn.to_torch(ttnn.from_device(tt_out))

    # Tolerances tighter than the BF16 floor: the FPU mul + cb_scaled UnpackToDest path
    # should preserve enough precision to land well within 0.5 ULP-relative of the exact
    # reference at this magnitude.
    assert_numeric_metrics(
        torch_ref,
        actual,
        rtol=1e-3,
        atol=1e-2,
        frobenius_threshold=1e-3,
        pcc_threshold=0.9999,
        check_ulp=False,
    )


# Test a 1D, 2D, 3D, and 4D tensor
@pytest.mark.parametrize("input_shape", [(2,), (3, 10), (6, 3, 60), (1, 11, 67, 77)])
@pytest.mark.parametrize("dim", [None, 0, 1, 2, 3])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("force_implicit_pad", [False, True])
# prod supports only bfloat16, per ttnn/cpp/ttnn/operations/reduction/prod/prod_nanobind.hpp
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_prod(device, input_shape, dim, keepdim, force_implicit_pad, dtype):
    torch.manual_seed(0)

    rank = len(input_shape)
    # rank = 0 is special case for scalar tensor and it usually supports dim=0 or dim=-1.
    if dim is not None and ((dim < -rank) or (dim > rank - 1)) and not (rank == 0 and dim in [0, -1]):
        pytest.skip("Dimension not applicable for input shape")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    # tensor.size, which is called by torch.prod, doesn't accept dim=None,
    # so we need to handle it separately.
    # See https://github.com/pytorch/pytorch/issues/127882
    if dim is None:
        torch_output_tensor = torch.prod(torch_input_tensor)
        if keepdim:
            # torch.prod does not support keepdim=True for dim=None,
            # so we need to reshape to match the input tensor.
            new_shape = [1] * torch_input_tensor.dim()
            torch_output_tensor = torch_output_tensor.reshape(new_shape)
    else:
        torch_output_tensor = torch.prod(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=dtype,
    )

    # Padding forced to 0 would annihilate a product if used; prod must still match torch after host-side pad reset
    if force_implicit_pad:
        input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, 0.0)

    output_tensor = ttnn.prod(input_tensor, dim=dim, keepdim=keepdim, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.bfloat16)
    assert len(output_tensor.shape) == len(torch_output_tensor.shape)
    assert output_tensor.shape == torch_output_tensor.shape
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.01,
        atol=0.25,
        frobenius_threshold=0.071,
    )


@pytest.mark.parametrize("dim_1", [1])
@pytest.mark.parametrize("dim_2", [2])
@pytest.mark.parametrize("dim_3", [3])
@pytest.mark.parametrize("dim_4", [4])
@pytest.mark.parametrize("dim_5", [4])
@pytest.mark.parametrize("dim_6", [6])
@pytest.mark.parametrize("dim_7", [7])
@pytest.mark.parametrize("dim_8", [8, 32, 63])
@pytest.mark.parametrize("dim", [[3, 7], [6, 7]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum_8d_tensor_dims(device, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim_8, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim_8), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.01,
        atol=0.03,
        frobenius_threshold=0.003,
    )


@pytest.mark.parametrize("dim_1", [1])
@pytest.mark.parametrize("dim_2", [2])
@pytest.mark.parametrize("dim_3", [3])
@pytest.mark.parametrize("dim_4", [4])
@pytest.mark.parametrize("dim_5", [5])
@pytest.mark.parametrize("dim_6", [6])
@pytest.mark.parametrize("dim_7", [7])
@pytest.mark.parametrize("dim", [[2, 5]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum_7d_tensor_dims(device, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.01,
        atol=0.03,
        frobenius_threshold=0.003,
    )


@pytest.mark.parametrize("dim_1", [1])
@pytest.mark.parametrize("dim_2", [2])
@pytest.mark.parametrize("dim_3", [3])
@pytest.mark.parametrize("dim_4", [4])
@pytest.mark.parametrize("dim_5", [5])
@pytest.mark.parametrize("dim_6", [6])
@pytest.mark.parametrize("dim", [[1, 4], -1, None])
@pytest.mark.parametrize("keepdim", [True])
def test_sum_6d_tensor_dims(device, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((dim_1, dim_2, dim_3, dim_4, dim_5, dim_6), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.01,
        atol=0.01,
        frobenius_threshold=0.007,
    )


@pytest.mark.parametrize("dim_1", [33])
@pytest.mark.parametrize("dim_2", [5])
@pytest.mark.parametrize("dim_3", [7])
@pytest.mark.parametrize("dim_4", [2])
@pytest.mark.parametrize("dim_5", [59])
@pytest.mark.parametrize("dim", [[1, 4], -1, None])
@pytest.mark.parametrize("keepdim", [True])
def test_sum_5d_tensor_dims(device, dim_1, dim_2, dim_3, dim_4, dim_5, dim, keepdim):
    torch.manual_seed(2)

    torch_input_tensor = torch.randn((dim_1, dim_2, dim_3, dim_4, dim_5), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.01,
        atol=0.2,
        frobenius_threshold=0.015,
    )


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("c", [32])
@pytest.mark.parametrize("h", [37])
@pytest.mark.parametrize("w", [63])
@pytest.mark.parametrize("dim", [None, [], 0, 2, [0, 1], [1, 3], [0, 1, 2], [1, 2, 3], [0, 1, 2, 3]])
@pytest.mark.parametrize("keepdim", [True])
def test_sum_4d_tensor_dims(device, batch_size, c, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.7,
        frobenius_threshold=0.004,
    )


@pytest.mark.parametrize("dim1", [1])
# This test picks the maximum dim2 that will pick the singlecore implementation.
# TopK multicore uses 8 cores in blackhole, so we need to add support for bitonic sort with 8 cores
# and non power of 2 dims as compared to wormhole. Issue #23465.
@pytest.mark.parametrize(
    "dim2",
    [8192 - 64, pytest.param(50257, marks=pytest.mark.xfail(condition=is_blackhole(), reason="Issue #23465"))],
)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("k", [50, 3200])
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_2d_topk(device, dim1, dim2, dim, k, largest, dtype):
    torch.manual_seed(2005)
    shape = [dim1, dim2]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype) * 0.9

    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, TEST_PADDING_VALUE)
    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(ttnn_input, k, dim=dim, largest=largest, sorted=True)

    desired_shape = [dim1, dim2]
    desired_shape[dim] = k

    assert list(ttnn_topk_values.shape) == desired_shape
    assert list(ttnn_topk_indices.shape) == desired_shape

    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices)

    # Add 2^16 to negative values
    ttnn_torch_indices = ttnn_torch_indices.to(dtype=torch.int32)
    ttnn_torch_indices = torch.where(ttnn_torch_indices < 0, ttnn_torch_indices + 65536, ttnn_torch_indices)

    if dtype == ttnn.bfloat8_b:
        pcc_values = 0.99
    else:
        pcc_values = 1.0

    # Convert to int64 only for torch.gather which requires signed indices
    ttnn_torch_gather_from_indices = torch.gather(
        input, dim, ttnn_torch_indices.to(torch.int64)  # Convert to signed only for PyTorch API compatibility
    )

    cosine = torch.nn.CosineSimilarity(dim=dim)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))
    assert (
        ttnn_torch_cosine > 0.99
    ), f"Cosine similarity between topk values and gather from indices is {ttnn_torch_cosine} which is less than 0.99"
    if dtype == ttnn.bfloat16:
        pcc_threshold = 0.9999
        rtol = 1e-6
        atol = 1e-6
        frobenius_threshold = 1e-9
    else:
        pcc_threshold = 0.996
        rtol = 0.044
        atol = 0.016
        frobenius_threshold = 0.007
    # test for equivalence
    assert_numeric_metrics(
        pyt_topk_values,
        ttnn_torch_values,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("dim1", [1])
@pytest.mark.parametrize("dim2", [128256, 151936])
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("k", [50])
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_large_2d_topk(device, dim1, dim2, dim, k, largest, dtype):
    torch.manual_seed(2005)
    shape = [dim1, dim2]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype) * 0.9

    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, TEST_PADDING_VALUE)
    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(ttnn_input, k, dim=dim, largest=largest, sorted=True)

    desired_shape = [dim1, dim2]
    desired_shape[dim] = k

    assert list(ttnn_topk_values.shape) == desired_shape
    assert list(ttnn_topk_indices.shape) == desired_shape

    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices)

    # Add 2^16 to negative values
    ttnn_torch_indices = ttnn_torch_indices.to(dtype=torch.int32)
    ttnn_torch_indices = torch.where(ttnn_torch_indices < 0, ttnn_torch_indices + 65536, ttnn_torch_indices)

    if dtype == ttnn.bfloat8_b:
        pcc_values = 0.99
    else:
        pcc_values = 1.0

    # Convert to int64 only for torch.gather which requires signed indices
    ttnn_torch_gather_from_indices = torch.gather(
        input, dim, ttnn_torch_indices.to(torch.int64)  # Convert to signed only for PyTorch API compatibility
    )

    cosine = torch.nn.CosineSimilarity(dim=dim)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))
    assert (
        ttnn_torch_cosine > 0.99
    ), f"Cosine similarity between topk values and gather from indices is {ttnn_torch_cosine} which is less than 0.99"
    # test for equivalence
    assert_numeric_metrics(
        pyt_topk_values,
        ttnn_torch_values,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )


@pytest.mark.parametrize("dim1", [1])
@pytest.mark.parametrize("dim2", [1])
@pytest.mark.parametrize("dim3", [8])
@pytest.mark.parametrize("dim4", [256])
@pytest.mark.parametrize("dim5", [64])
@pytest.mark.parametrize("dim", [3, 4])
@pytest.mark.parametrize("k", [17, 32, 64])
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_5d_topk(device, dim1, dim2, dim3, dim4, dim5, dim, k, largest, dtype):
    torch.manual_seed(2005)
    shape = [dim1, dim2, dim3, dim4, dim5]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, TEST_PADDING_VALUE)
    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(ttnn_input, k, dim=dim, largest=largest, sorted=True)

    desired_shape = [dim1, dim2, dim3, dim4, dim5]
    desired_shape[dim] = k

    assert list(ttnn_topk_values.shape) == desired_shape
    assert list(ttnn_topk_indices.shape) == desired_shape

    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices).to(torch.int64)

    if dtype == ttnn.bfloat8_b:
        pcc_values = 0.99
    else:
        pcc_values = 1.0

    # pcc is not a good measure for the raw indices
    # if index 49 and index 8 are tied, the order of the indices can be different
    # but the values associated with the indices should be the same
    # if index 7 and 8 are tied, but swapped, the pcc will be better than if index 49 and 8 are tied but swapped
    # rounding may also cause more ties than expected
    # the bigger we get, the tighter the distribution of the top K elements, so the pcc will be worse as stability/rounding will cause more ties
    # use cosine similarity on the gathered indices as this will show the top elements are all about the same
    ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn_torch_indices.to(torch.int64))
    cosine = torch.nn.CosineSimilarity(dim=dim)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))

    assert (
        ttnn_torch_cosine > 0.99
    ), f"Cosine similarity between topk values and gather from indices is {ttnn_torch_cosine} which is less than 0.99"
    if dtype == ttnn.bfloat16:
        pcc_threshold = 0.9999
        rtol = 1e-6
        atol = 1e-6
        frobenius_threshold = 1e-9
    else:
        pcc_threshold = 0.999
        rtol = 2.933
        atol = 0.026
        frobenius_threshold = 0.010
    # test for equivalence
    assert_numeric_metrics(
        pyt_topk_values,
        ttnn_torch_values,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


# returns larger padded tensor instead of desired shape
@pytest.mark.parametrize("dim1", [1])
@pytest.mark.parametrize("dim2", [1])
@pytest.mark.parametrize("dim3", [8])
@pytest.mark.parametrize("dim4", [1])
@pytest.mark.parametrize("dim5", [128])
@pytest.mark.parametrize("dim6", [64])
# @pytest.mark.parametrize("dim", [0, 1, 2, 3, 4, 5]) transpose cannot handle N-D tensor for all dims
@pytest.mark.parametrize("dim", [4, 5])
@pytest.mark.parametrize("k", [50, 64])
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_6d_topk(device, dim1, dim2, dim3, dim4, dim5, dim6, dim, k, largest, dtype):
    torch.manual_seed(2005)
    shape = [dim1, dim2, dim3, dim4, dim5, dim6]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, TEST_PADDING_VALUE)
    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(ttnn_input, k, dim=dim, largest=largest, sorted=True)

    desired_shape = [dim1, dim2, dim3, dim4, dim5, dim6]
    desired_shape[dim] = k

    assert list(ttnn_topk_values.shape) == desired_shape
    assert list(ttnn_topk_indices.shape) == desired_shape

    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices).to(torch.int64)

    # pcc is not a good measure for the raw indices
    # if index 49 and index 8 are tied, the order of the indices can be different
    # but the values associated with the indices should be the same
    # if index 7 and 8 are tied, but swapped, the pcc will be better than if index 49 and 8 are tied but swapped
    # rounding may also cause more ties than expected
    # the bigger we get, the tighter the distribution of the top K elements, so the pcc will be worse as stability/rounding will cause more ties
    # use cosine similarity on the gathered indices as this will show the top elements are all about the same
    ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn_torch_indices.to(torch.int64))
    cosine = torch.nn.CosineSimilarity(dim=dim)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))

    assert (
        ttnn_torch_cosine > 0.99
    ), f"Cosine similarity between topk values and gather from indices is {ttnn_torch_cosine} which is less than 0.99"
    if dtype == ttnn.bfloat16:
        pcc_threshold = 0.9999
        rtol = 1e-6
        atol = 1e-6
        frobenius_threshold = 1e-9
    else:
        pcc_threshold = 0.999
        rtol = 2.997
        atol = 0.026
        frobenius_threshold = 0.011
    # test for equivalence
    assert_numeric_metrics(
        pyt_topk_values,
        ttnn_torch_values,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [31])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [[0, 2], [0, 1, 2], None])
@pytest.mark.parametrize("keepdim", [True])
def test_sum_3d_tensor_dims(device, c, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.01,
        atol=0.01,
        frobenius_threshold=0.005,
    )


@pytest.mark.parametrize("h", [41])
@pytest.mark.parametrize("w", [31])
@pytest.mark.parametrize("dim", [0, 1, [0, 1], None])
@pytest.mark.parametrize("keepdim", [True])
def test_sum_2d_tensor_dims(device, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.01,
        frobenius_threshold=0.005,
    )


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("c", [5])
@pytest.mark.parametrize("h", [37])
@pytest.mark.parametrize("w", [63])
@pytest.mark.parametrize("dim", [None, [], 0, 2, [0, 1], [1, 3], [0, 1, 2], [1, 2, 3], [0, 1, 2, 3]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_4d_tensor_dims(device, batch_size, c, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.01,
        atol=0.01,
        frobenius_threshold=0.007,
    )


@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [31])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [[0, 2], [0, 1, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_3d_tensor_dims(device, c, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.01,
        atol=0.001,
        frobenius_threshold=0.007,
    )


@pytest.mark.parametrize("h", [41])
@pytest.mark.parametrize("w", [31])
@pytest.mark.parametrize("dim", [0, 1, [0, 1]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_2d_tensor_dims(device, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalence
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.002,
        frobenius_threshold=0.006,
    )


def run_maxpool(device, input_shape, kernel_size, stride, padding, dilation):
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    batch_size, in_c, in_h, in_w = input_shape
    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, in_c))
    input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.max_pool2d(
        input_tensor,
        batch_size,
        in_h,
        in_w,
        in_c,
        kernel_size,
        stride,
        padding,
        dilation,
    )

    torch_output_tensor = torch.nn.functional.max_pool2d(torch_input, kernel_size, stride, padding)

    output_tensor = ttnn.to_torch(output_tensor)
    _, out_c, out_h, out_w = torch_output_tensor.shape
    output_tensor = torch.reshape(output_tensor, (batch_size, out_h, out_w, out_c))
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    assert_numeric_metrics(
        output_tensor,
        torch_output_tensor,
        pcc_threshold=0.9999,
        rtol=1e-6,
        atol=1e-6,
        frobenius_threshold=1e-9,
    )


def run_reduce_sum_h(device, batch_size, h, w, dim):
    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    output_tensor = ttnn.mean(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        output_tensor,
        torch_output_tensor,
        pcc_threshold=0.9999,
        rtol=0.001,
        atol=0.008,
        frobenius_threshold=0.003,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4096}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 192, 56, 56),  # Multi core face height not default
    ],
)
@pytest.mark.parametrize(
    "kernel_size",
    [
        (2, 2),  # Small kernel
        (5, 5),  # Large kernel
    ],
)
def test_run_reduce_sum_h_after_max_pool(device, input_shape, kernel_size):
    torch.manual_seed(0)
    run_maxpool(device, input_shape, kernel_size, kernel_size, (0, 0), (1, 1))
    run_reduce_sum_h(device, 1, 32, 32, -2)


@pytest.mark.parametrize(
    argnames="tensor_shape, keepdim, dim, op",
    argvalues=[
        ([], True, None, "mean"),
        ([], True, None, "std"),
        ([32], False, -1, "sum"),
        ([32, 0], True, 0, "max"),
        ([0, 0, 0], True, 2, "min"),
        ([0, 32, 0], False, -2, "std"),
        ([32, 32, 32, 0], False, 3, "var"),
    ],
)
def test_torch_compatibility(device, tensor_shape, keepdim, dim, op):
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes, keepdim, and dim values.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    torch.manual_seed(42)

    rank = len(tensor_shape)

    torch_tensor = torch.randn(*tensor_shape) if rank > 0 else torch.randn(())
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_tensor = ttnn.fill_implicit_tile_padding(ttnn_tensor, TEST_PADDING_VALUE)

    torch_op, ttnn_op = getattr(torch, op), getattr(ttnn, op)

    # Run on both and flag exceptions
    torch_errored = False
    try:
        torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim)
    except IndexError:
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim)
    except RuntimeError:
        ttnn_errored = True

    assert torch_errored == ttnn_errored, f"torch: {torch_errored}, ttnn: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    # torch's min/max double as argmin/argmax, so we need to extract the values only
    torch_result = (
        torch_result.values
        if isinstance(torch_result, (torch.return_types.min, torch.return_types.max))
        else torch_result
    )

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    outputs_all_finite = torch.isfinite(torch_result).all() and torch.isfinite(ttnn_result).all()
    if outputs_all_finite and torch_result.numel() > 0:
        # test for equivalence
        assert_numeric_metrics(
            torch_result,
            ttnn_result,
            pcc_threshold=0.999,
            rtol=0.004,
            atol=0.01,
            frobenius_threshold=0.004,
        )
    else:
        atol = rtol = 0.1
        assert torch.allclose(
            torch_result, ttnn_result, atol=atol, rtol=rtol, equal_nan=True
        ), f"torch: {torch_result}, ttnn: {ttnn_result}"
