# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from math import isnan
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp, assert_equal


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_mac_all_tensors(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor2 = torch.rand((h, w), dtype=torch.bfloat16)

    golden_fn = ttnn.get_golden_function(ttnn.mac)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn.mac(input_tensor, input_tensor1, input_tensor2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("scalar1", [5.5])
@pytest.mark.parametrize("scalar2", [-13.2])
def test_mac_tensor_with_2_scalaras(device, h, w, scalar1, scalar2):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = scalar1
    torch_input_tensor2 = scalar2

    golden_fn = ttnn.get_golden_function(ttnn.mac)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.mac(input_tensor, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


def assert_where_with_pcc(torch_input_tensor, torch_input1, torch_input2, device, pcc=0.9999):
    def from_torch_if_tensor(x):
        if not isinstance(x, torch.Tensor):
            return x

        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor, input1, input2 = (
        from_torch_if_tensor(arg)
        for arg in ((torch_input_tensor > 0).to(torch_input_tensor.dtype), torch_input1, torch_input2)
    )
    golden_fn = ttnn.get_golden_function(ttnn.where)
    torch_output_tensor = golden_fn(torch_input_tensor > 0, torch_input1, torch_input2)
    output_tensor = ttnn.where(input_tensor, input1, input2)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "hc, ht, hf, wc, wt, wf",
    [
        [64, 64, 64, 128, 128, 1],
        [64, 64, 64, 128, 1, 128],
        [64, 64, 64, 1, 128, 128],
        [64, 64, 1, 128, 128, 128],
        [64, 1, 64, 128, 128, 128],
        [1, 64, 64, 128, 128, 128],
        [64, 64, 1, 128, 128, 1],
        [64, 1, 64, 128, 1, 128],
        [64, 1, 64, 128, 128, 1],
        [64, 64, 1, 128, 1, 128],
        [1, 1, 64, 128, 128, 1],
        [64, 1, 64, 1, 1, 128],
    ],
)
def test_where_bcast(device, dtype, hc, ht, hf, wc, wt, wf):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hc, wc), dtype=dtype).uniform_(-100, 100)
    torch_input_tensor1 = torch.rand((ht, wt), dtype=dtype).uniform_(-100, 100)
    torch_input_tensor2 = torch.rand((hf, wf), dtype=dtype).uniform_(-100, 100)

    assert_where_with_pcc(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, device)


def run_ternary_test_value(device, h, w, value, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16).uniform_(-100, 100)
    torch_input_tensor1 = torch.rand((h, w), dtype=torch.bfloat16).uniform_(-100, 100)
    torch_input_tensor2 = torch.rand((h, w), dtype=torch.bfloat16).uniform_(-100, 100)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn_function(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("value", [15.5])
def test_addcdiv(device, h, w, value):
    run_ternary_test_value(device, h, w, value, ttnn.addcdiv)


@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("value", [15.5, 0, 1.0, -5.0])
@pytest.mark.parametrize(
    "hc, ht, hf, wc, wt, wf",
    [
        [64, 64, 64, 128, 128, 128],  # no bcast
        # Row / Col bcast cases
        [64, 64, 64, 128, 128, 1],
        [64, 64, 64, 128, 1, 128],
        [64, 64, 64, 1, 128, 128],
        [64, 64, 1, 128, 128, 128],
        [64, 1, 64, 128, 128, 128],
        [1, 64, 64, 128, 128, 128],
        [64, 64, 1, 128, 128, 1],
        [64, 1, 64, 128, 1, 128],
        [1, 64, 64, 1, 128, 128],
        [64, 1, 1, 128, 1, 1],  # scalar bcast case
    ],
)
def test_addcmul_with_bcast(device, tor_dtype, ttnn_dtype, hc, ht, hf, wc, wt, wf, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hc, wc), dtype=tor_dtype).uniform_(-100, 500)
    torch_input_tensor1 = torch.rand((ht, wt), dtype=tor_dtype).uniform_(-200, 200)
    torch_input_tensor2 = torch.rand((hf, wf), dtype=tor_dtype).uniform_(-300, 400)

    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn.addcmul(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.0,
        -0.5,
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape",
    [
        ((1, 2, 1088, 1024), (1, 2, 1, 1024), (1, 2, 1088, 1024)),  # Composite
        ((1, 2, 1088, 1024), (1, 2, 1, 1), (1, 2, 1088, 1024)),  # Composite
        ((4, 2, 1088, 1024), (1, 2, 1088, 1024), (1, 1, 1088, 1024)),  # HLK
    ],
)
def test_addcmul_with_bcast_bf8b(device, torch_dtype, ttnn_dtype, a_shape, b_shape, c_shape, value):
    """
    Test addcmul: Block format datatype inputs with subtile broadcast use composite Addcmul implementation.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(a_shape, dtype=torch_dtype)
    torch_input_tensor1 = torch.randn(b_shape, dtype=torch_dtype)
    torch_input_tensor2 = torch.randn(c_shape, dtype=torch_dtype)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("value", [1.0, 0.5])
@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)),
        ((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1)),
        ((1, 1, 1, 1), (1, 1, 1024, 1024), (1, 1, 1, 1)),
    ],
)
def test_addcmul(device, torch_dtype, ttnn_dtype, value, in_data1_shape, in_data2_shape, in_data3_shape):
    in_data1 = torch.full(in_data1_shape, 0.0031, dtype=torch_dtype)
    in_data2 = torch.full(in_data2_shape, 508.0, dtype=torch_dtype)
    in_data3 = torch.full(in_data3_shape, 748.0, dtype=torch_dtype)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    assert_with_ulp(output_tensor, golden_tensor)


def test_addcmul_with_int32_inputs(device):
    in_data1 = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.int32)
    in_data2 = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.int32)
    in_data3 = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.int32)
    value = 1
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    assert_equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        # (torch.float32, ttnn.float32),
    ],
)
# @pytest.mark.parametrize("value", [1.0, 0.5, 7.7, 22.4, 107.6])
# @pytest.mark.parametrize("value", [108.6])
@pytest.mark.parametrize("value", [1.0])
@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        # ((1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)),
        # ((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1)),
        # ((1, 1, 1, 1), (1, 1, 1024, 1024), (1, 1, 1, 1)),
    ],
)
def test_addcdiv(device, torch_dtype, ttnn_dtype, value, in_data1_shape, in_data2_shape, in_data3_shape):
    torch.manual_seed(0)
    # in_data1 = torch.full(in_data1_shape, 90, dtype=torch_dtype)
    # in_data2 = torch.full(in_data2_shape, -18.75, dtype=torch_dtype)
    # in_data3 = torch.full(in_data3_shape, 22.625, dtype=torch_dtype)
    in_data1 = torch.full(in_data1_shape, -32.0, dtype=torch_dtype)
    in_data2 = torch.full(in_data2_shape, -25.0, dtype=torch_dtype)
    in_data3 = torch.full(in_data3_shape, -0.78125, dtype=torch_dtype)
    # in_data1 = torch.full(in_data1_shape, -25.0, dtype=torch_dtype)
    # in_data2 = torch.full(in_data2_shape, 97.0, dtype=torch_dtype)
    # in_data3 = torch.full(in_data3_shape, 3.90625, dtype=torch_dtype)
    # in_data1 = torch.empty(in_data1_shape, dtype=torch_dtype).uniform_(-100, 100)
    # in_data2 = torch.empty(in_data2_shape, dtype=torch_dtype).uniform_(-100, 100)
    # in_data3 = torch.empty(in_data3_shape, dtype=torch_dtype).uniform_(-100, 100)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcdiv(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.addcdiv)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    # Normalize: if input3 is zero and golden is nan and ttnn output is inf, change ttnn output to nan
    zero_mask = in_data3 == 0
    golden_nan_mask = torch.isnan(golden_tensor)
    output_inf_mask = torch.isinf(output_tensor)
    normalize_mask = zero_mask & golden_nan_mask & output_inf_mask
    if normalize_mask.any():
        output_tensor = torch.where(
            normalize_mask,
            torch.tensor(float("nan"), dtype=output_tensor.dtype, device=output_tensor.device),
            output_tensor,
        )

    ttnn.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=200, threshold=10000, precision=5, sci_mode=False, edgeitems=17)

    # # Print inputs
    # print("in_data1", in_data1)
    # print("in_data2", in_data2)
    # print("in_data3", in_data3)
    # print("value", value)
    print("golden_tensor", golden_tensor)
    print("output_tensor", output_tensor)
    print(f"  Actual main (output):   {output_tensor[0,0,0,0].item()}", flush=True)

    # Helper function to map output index to input index considering broadcasting
    def map_index_to_input(idx_tuple, input_shape):
        """Map an output index to the corresponding input index based on broadcasting rules.
        If a dimension in input has size 1, use index 0 for that dimension."""
        mapped_idx = []
        for dim_idx, dim_size in enumerate(input_shape):
            if dim_size == 1:
                mapped_idx.append(0)
            else:
                mapped_idx.append(idx_tuple[dim_idx])
        return tuple(mapped_idx)

    # Check for ULP failures and print failing values
    try:
        assert_with_ulp(output_tensor, golden_tensor, allow_nonfinite=True)
    except AssertionError as e:
        # Compute ULP errors for each element
        from models.common.utility_functions import ulp, comp_ulp

        ulp_passed, ulp_message = comp_ulp(golden_tensor, output_tensor, ulp_threshold=10, allow_nonfinite=True)

        # Match comp_ulp's behavior: normalize non-finite values to 0 before computing ULP
        # This is important because non-finite values can interfere with ULP calculation
        mask_finite = ~torch.isfinite(golden_tensor)
        golden_for_ulp = golden_tensor.clone()
        output_for_ulp = output_tensor.clone()
        golden_for_ulp[mask_finite] = 0
        output_for_ulp[mask_finite] = 0

        # Compute ULP value for each element in golden tensor
        # Convert to same dtype as output for proper ULP calculation (matching comp_ulp)
        golden_for_ulp = golden_for_ulp.type(output_for_ulp.dtype)
        ulp_value = ulp(golden_for_ulp)

        # If dtypes differ, convert to higher precision (matching comp_ulp)
        if golden_tensor.dtype != output_tensor.dtype:
            output_for_ulp = output_for_ulp.type(golden_tensor.dtype)
            ulp_value = ulp_value.type(golden_tensor.dtype)

        # Compute ULP error for each element: |calculated - golden| / ULP(golden)
        diff = torch.abs(output_for_ulp - golden_for_ulp)
        ulp_error = diff / ulp_value

        # Find all indices with ULP error > 3 (including inf values, but we'll handle them separately)
        high_ulp_mask = (ulp_error > 3.0) & torch.isfinite(ulp_error)
        # Also check for inf ULP errors (which indicate very large differences when golden is 0)
        inf_ulp_mask = torch.isinf(ulp_error)

        # Get max ULP error (matching comp_ulp behavior - includes inf)
        max_ulp_all = torch.max(ulp_error).item() if ulp_error.numel() > 0 else float("nan")

        if high_ulp_mask.any() or inf_ulp_mask.any():
            high_ulp_indices = torch.nonzero(high_ulp_mask, as_tuple=False)
            high_ulp_errors = ulp_error[high_ulp_mask]

            # Sort by ULP error (worst first)
            sorted_indices = torch.argsort(high_ulp_errors, descending=True)
            sorted_high_ulp_indices = high_ulp_indices[sorted_indices]

            # Also find all failing indices for summary (using torch.isclose)
            failing_mask = ~torch.isclose(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8, equal_nan=True)
            failing_indices = (
                torch.nonzero(failing_mask, as_tuple=False)
                if failing_mask.any()
                else torch.empty((0, len(output_tensor.shape)), dtype=torch.long)
            )

            if len(sorted_high_ulp_indices) > 0:
                sorted_indices = torch.argsort(high_ulp_errors, descending=True)
                sorted_high_ulp_indices = high_ulp_indices[sorted_indices]

                print(f"\n=== ULP FAILURE: Found {len(failing_indices)} failing values ===")
                print(f"Found {len(sorted_high_ulp_indices)} cases with finite ULP error > 3.0")
                if inf_ulp_mask.any():
                    inf_count = torch.sum(inf_ulp_mask).item()
                    print(f"Found {inf_count} cases with inf ULP error")
                print(f"Assertion error message: {str(e)}")
                # Compute max/mean - use max_ulp_all which includes inf
                print(
                    f"Max ULP error (all): {max_ulp_all:.6f}"
                    if not torch.isinf(torch.tensor(max_ulp_all))
                    else f"Max ULP error (all): inf"
                )
                # Also show max finite ULP
                finite_ulp_errors = ulp_error[torch.isfinite(ulp_error)]
                if finite_ulp_errors.numel() > 0:
                    max_ulp_finite = torch.max(finite_ulp_errors).item()
                    print(f"Max ULP error (finite): {max_ulp_finite:.6f}")
                else:
                    print(f"Max ULP error (finite): N/A (all are inf)")
                if len(sorted_high_ulp_indices) > 0:
                    mean_high_ulp_error = torch.mean(high_ulp_errors).item()
                    print(f"Mean ULP error (for ULP > 3): {mean_high_ulp_error:.6f}")
                else:
                    print(f"Mean ULP error: N/A")

                # Print first 20 high ULP failures (ULP > 3) with detailed info (sorted by worst ULP error first)
                import sys

                for i, idx in enumerate(sorted_high_ulp_indices[:20]):
                    try:
                        idx_tuple = tuple(idx.tolist())
                        ulp_err = ulp_error[idx_tuple].item()
                        # Use scientific notation for very large ULP errors
                        if abs(ulp_err) > 1e6 or (abs(ulp_err) < 1e-6 and ulp_err != 0):
                            ulp_err_str = f"{ulp_err:.6e}"
                        else:
                            ulp_err_str = f"{ulp_err:.6f}"
                        print(f"\nFailure #{i+1} at index {idx_tuple} (ULP error: {ulp_err_str}):", flush=True)
                        # Map indices to input shapes considering broadcasting
                        idx1 = map_index_to_input(idx_tuple, in_data1.shape)
                        idx2 = map_index_to_input(idx_tuple, in_data2.shape)
                        idx3 = map_index_to_input(idx_tuple, in_data3.shape)
                        print(f"  Input1 (in_data1) at {idx1}: {in_data1[idx1].item()}", flush=True)
                        print(f"  Input2 (in_data2) at {idx2}: {in_data2[idx2].item()}", flush=True)
                        print(f"  Input3 (in_data3) at {idx3}: {in_data3[idx3].item()}", flush=True)
                        print(f"  Value: {value}", flush=True)
                        print(f"  Expected (golden): {golden_tensor[idx_tuple].item()}", flush=True)
                        print(f"  Actual (output):   {output_tensor[idx_tuple].item()}", flush=True)
                        print(f"  Absolute diff:     {diff[idx_tuple].item()}", flush=True)
                        print(f"  ULP value:         {ulp_value[idx_tuple].item()}", flush=True)
                        print(f"  ULP error:         {ulp_err_str}", flush=True)
                        golden_val = golden_tensor[idx_tuple].item()
                        if golden_val != 0:
                            rel_diff = diff[idx_tuple].item() / abs(golden_val)
                            print(f"  Relative diff:     {rel_diff}", flush=True)
                    except Exception as ex:
                        print(
                            f"\nFailure #{i+1} at index {tuple(idx.tolist())} - Error printing details: {ex}",
                            flush=True,
                        )
                        import traceback

                        traceback.print_exc()
                if len(sorted_high_ulp_indices) > 20:
                    print(f"\n  ... and {len(sorted_high_ulp_indices) - 20} more failures with ULP > 3")
        else:
            # No cases with finite ULP > 3, but assertion still failed
            # This can happen if:
            # 1. All ULP errors are inf (when golden is 0 and output is not 0)
            # 2. Max ULP is between threshold and 3.0
            failing_mask = ~torch.isclose(output_tensor, golden_tensor, rtol=1e-5, atol=1e-8, equal_nan=True)
            failing_indices = (
                torch.nonzero(failing_mask, as_tuple=False)
                if failing_mask.any()
                else torch.empty((0, len(output_tensor.shape)), dtype=torch.long)
            )

            # Get max ULP error (may be inf)
            max_ulp_all = torch.max(ulp_error).item() if ulp_error.numel() > 0 else float("nan")
            finite_ulp_errors = ulp_error[torch.isfinite(ulp_error)]
            max_ulp_finite = torch.max(finite_ulp_errors).item() if finite_ulp_errors.numel() > 0 else float("nan")
            inf_ulp_count = torch.sum(torch.isinf(ulp_error)).item()

            print(f"\n=== ULP FAILURE: Found {len(failing_indices)} failing values ===")
            if inf_ulp_count > 0:
                print(f"No cases with finite ULP error > 3.0, but found {inf_ulp_count} cases with inf ULP error")
            else:
                print(f"No cases with ULP error > 3.0 (max finite ULP: {max_ulp_finite:.6f})")
            print(
                f"Max ULP error (all): {max_ulp_all:.6f}"
                if not torch.isinf(torch.tensor(max_ulp_all))
                else f"Max ULP error (all): inf"
            )
            print(f"Assertion error message: {str(e)}")
        raise
