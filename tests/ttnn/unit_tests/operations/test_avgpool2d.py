# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_avg_pool(device, input_shape, kernel_size, stride, padding, dilation):
    # Test setup for both.
    batch_size, in_c, in_h, in_w = input_shape
    ## Decimal
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    ## Integer
    # torch_input = torch.zeros(input_shape, dtype=torch.bfloat16)
    # for n in range(input_shape[0]):
    #     for c in range(input_shape[1]):
    #         for h in range(input_shape[2]):
    #             for w in range(input_shape[3]):
    #                 torch_input[n, c, h, w] = h * in_w + w

    # Test setup for Actual.
    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, in_c))
    input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Get Expected output.
    expected_output = torch.nn.functional.avg_pool2d(
        torch_input, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None
    )

    # Get Actual output
    output_tensor = ttnn.avg_pool2d(input_tensor, batch_size, in_h, in_w, in_c, kernel_size, stride, padding, dilation)

    # Test teardown for Actual.
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = torch.reshape(output_tensor, expected_output.shape)

    # Assertion
    # assert_with_pcc(expected_output, output_tensor, 0.99)
    # print(input_tensor.shape)
    # torch.set_printoptions(profile="full")
    print(f"Expected_output.shape: {expected_output.shape}")
    print(f"Output_tensor.shape: {output_tensor.shape}")
    # print(torch_input)
    # print(expected_output)
    # print(f"output_tensor[0][0]: {output_tensor[0][0]}")

    ## Find and print mismatched indices and values.
    # mismatches = (expected_output != output_tensor).nonzero(as_tuple=True)
    # for idx in zip(*mismatches):
    #     print(f"Index: {idx}, | Expected: {expected_output[idx]}, | Actual: {output_tensor[idx]}, | Diff: {expected_output[idx] - output_tensor[idx]}, | Error: {100*abs(expected_output[idx] - output_tensor[idx])/expected_output[idx]} %")
    ## Find and print indices and values based on rtol=0.01.
    # mismatch_mask = ~torch.isclose(expected_output, output_tensor, rtol=0.01) # Find mismatches based on rtol.
    # mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=True) # Get indices where mismatch occurs.
    # mismatched_expected = expected_output[mismatch_indices] # Get mismatched values
    # mismatched_output = output_tensor[mismatch_indices] # Get mismatched values
    # for idx, (exp_val, out_val) in enumerate(zip(mismatched_expected, mismatched_output)):
    #     print(f"Mismatch {idx+1}: Index {tuple(mismatch_indices[i][idx].item() for i in range(len(mismatch_indices)))}, "
    #       f"Expected = {exp_val.item()}, Actual = {out_val.item()}, Diff = {exp_val.item() - out_val.item()}, Error: {100*abs(exp_val.item() - out_val.item())/exp_val.item()} %")

    assert torch.allclose(expected_output, output_tensor, rtol=0.01)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, dilation",
    [
        ## Case 1. Normal compute & Normal reader, Batch = 1.
        (
            (1, 32, 16, 16),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
        ),  # Decimal Passed. Integer Passed. DONE!
        ((1, 32, 112, 112), (2, 2), (2, 2), (0, 0), (1, 1)),  # Decimal Passed. Integer Passed. DONE!
        ## Case 2. Normal computer & Normal reader, Batch > 1.
        # ((2, 32, 16, 16), (2, 2), (2, 2), (0, 0), (1, 1)),        # Decimal FAILED. Integer Passed.
        # ((2, 32, 112, 112), (2, 2), (1, 1), (0, 0), (1, 1)),      # Decimal FAILED. Integer Passed.
        ## Case 3. Normal compute & Wide (C>256) reader kernel, Batch = 1.
        ((1, 512, 16, 16), (2, 2), (2, 2), (0, 0), (1, 1)),  # Decimal FAILED. Integer FAILED.
        ((1, 512, 112, 112), (2, 2), (2, 2), (0, 0), (1, 1)),  # Decimal FAILED. Integer FAILED.
        ## Case 4. Normal compute & Wide (C>256) reader kernel, Batch > 1.
        # ((2, 512, 16, 16), (2, 2), (2, 2), (0, 0), (1, 1)),         # Decimal FAILED. Integer FAILED.
        # ((2, 512, 112, 112), (2, 2), (2, 2), (0, 0), (1, 1)),       # Decimal FAILED. Integer FAILED.
        ## Case 5. Large compute & Large reader kernel, Batch = 1.  # TODO(jongbinlimTT): This case fails. Need to remove double division.
        # ((1, 32, 16, 16), (5, 5), (2, 2), (0, 0), (1, 1)),        # Decimal FAILED. Integer FAILED.
        # ((1, 32, 112, 112), (5, 5), (2, 2), (0, 0), (1, 1)),      # Decimal FAILED. Integer FAILED.
        ## Case 6. Large compute & Large reader kernel, Batch > 1.
        # ((2, 32, 16, 16), (5, 5), (2, 2), (0, 0), (1, 1)),        # Decimal FAILED. Integer FAILED.
        # ((2, 32, 112, 12), (5, 5), (2, 2), (0, 0), (1, 1)),       # Decimal FAILED. Integer FAILED.
        ## Case 7. Large compute & Large + wide -> Large reader kernel, Batch = 1.
        # ((1, 512, 32, 32), (5, 5), (2, 2), (0, 0), (1, 1)),       # Decimal FAILED. Integer FAILED.
        # ((1, 512, 32, 32), (5, 5), (2, 2), (0, 0), (1, 1)),       # Decimal FAILED. Integer FAILED.
        ## Case 8. Large compute & Large + wide -> Large reader kernel, Batch > 1.
        # ((2, 512, 112, 112), (5, 5), (2, 2), (0, 0), (1, 1)),     # Decimal FAILED. Integer FAILED.
        # ((2, 512, 112, 112), (5, 5), (2, 2), (0, 0), (1, 1)),     # Decimal FAILED. Integer FAILED.
        # ((4, 256, 40, 40), (2, 2), (2, 2), (0, 0), (1, 1)),
        # ((1, 128, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
        # ((1, 256, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1)),
        # ((1, 192, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
        # ((1, 160, 7, 7), (2, 2), (2, 2), (0, 0), (1, 1)),
        # ((1, 256, 56, 56), (13, 13), (1, 1), (0, 0), (1, 1)),
        # pytest.param((1, 512, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#14459)")),
        # pytest.param((1, 384, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 1056, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 640, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 896, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 24, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 40, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#15731)")),
        # pytest.param((1, 80, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#15731)")),
        # pytest.param((1, 112, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#15731)")),
        # pytest.param((1, 384, 35, 35), (3, 3), (1, 1), (1, 1), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 1024, 17, 17), (3, 3), (1, 1), (1, 1), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#14459)")),
        # pytest.param((1, 1536, 8, 8), (3, 3), (1, 1), (1, 1), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#14459)")),
    ],
)
def test_run_avg_pool(device, input_shape, kernel_size, stride, padding, dilation):
    run_avg_pool(device, input_shape, kernel_size, stride, padding, dilation)
