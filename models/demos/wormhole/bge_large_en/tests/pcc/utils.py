# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from models.common.utility_functions import comp_pcc


def construct_pcc_assert_message(message, expected_pytorch_result, actual_pytorch_result):
    messages = []
    messages.append(message)
    messages = [str(m) for m in messages]
    return "\n".join(messages)


def assert_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    """
    Assert that two PyTorch tensors are similar within a specified Pearson Correlation Coefficient (PCC) threshold.

    This function compares two tensors using PCC, which measures the linear correlation between them.
    It's particularly useful for floating-point comparisons where exact equality is not expected due to
    numerical precision differences.

    Args:
        expected_pytorch_result (torch.Tensor): The expected reference tensor
        actual_pytorch_result (torch.Tensor): The actual tensor to compare against the reference
        pcc (float, optional): The minimum PCC threshold for the comparison to pass. Defaults to 0.9999.
                              Values closer to 1.0 indicate stronger correlation.

    Returns:
        tuple: A tuple containing:
            - pcc_passed (bool): True if the PCC check passed, False otherwise
            - pcc_message (str): A message describing the PCC comparison result

    Raises:
        AssertionError: If the tensor shapes don't match or if the PCC is below the specified threshold
    """
    assert list(expected_pytorch_result.shape) == list(
        actual_pytorch_result.shape
    ), f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}"
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    assert pcc_passed, construct_pcc_assert_message(pcc_message, expected_pytorch_result, actual_pytorch_result)
    return pcc_passed, pcc_message
