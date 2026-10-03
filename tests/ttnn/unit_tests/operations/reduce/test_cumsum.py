import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

@pytest.mark.parametrize("scan_length", [4, 8, 33])
@pytest.mark.parametrize("dim", [0, -2])
@pytest.mark.parametrize("disable_compensated_sum", [False, True])
def test_cumsum_inf_overflow(scan_length, dim, disable_compensated_sum):
    # Test +inf input
    input_tensor = torch.full((scan_length,), float('inf'))
    input_tensor[0] = 1.0
    input_tensor[1] = 2.0
    input_tensor[2] = 3.0
    input_tensor[3] = float('inf')
    
    # Compute expected output using PyTorch
    expected_output = torch.cumsum(input_tensor, dim=dim)
    
    # Compute actual output using TTNN
    ttnn_input = ttnn.from_torch(input_tensor)
    ttnn_output = ttnn.cumsum(ttnn_input, dim=dim, disable_compensated_sum=disable_compensated_sum)
    actual_output = ttnn.to_torch(ttnn_output)
    
    # Verify outputs match
    assert_with_pcc(actual_output, expected_output)

    # Test -inf input
    input_tensor = torch.full((scan_length,), float('-inf'))
    input_tensor[0] = -1.0
    input_tensor[1] = -2.0
    input_tensor[2] = -3.0
    input_tensor[3] = float('-inf')
    
    # Compute expected output using PyTorch
    expected_output = torch.cumsum(input_tensor, dim=dim)
    
    # Compute actual output using TTNN
    ttnn_input = ttnn.from_torch(input_tensor)
    ttnn_output = ttnn.cumsum(ttnn_input, dim=dim, disable_compensated_sum=disable_compensated_sum)
    actual_output = ttnn.to_torch(ttnn_output)
    
    # Verify outputs match
    assert_with_pcc(actual_output, expected_output)

    # Test finite-input overflow
    input_tensor = torch.full((scan_length,), 1e38)
    input_tensor[0] = 1.0
    input_tensor[1] = 2.0
    input_tensor[2] = 3.0
    input_tensor[3] = 1e38
    
    # Compute expected output using PyTorch
    expected_output = torch.cumsum(input_tensor, dim=dim)
    
    # Compute actual output using TTNN
    ttnn_input = ttnn.from_torch(input_tensor)
    ttnn_output = ttnn.cumsum(ttnn_input, dim=dim, disable_compensated_sum=disable_compensated_sum)
    actual_output = ttnn.to_torch(ttnn_output)
    
    # Verify outputs match
    assert_with_pcc(actual_output, expected_output)

# ... rest of the test file ...
