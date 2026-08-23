# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
def test_erfinv_accurate(device, shape):
    torch.manual_seed(0)
    # Test values in the tails where error is highest
    in_data = torch.cat((torch.linspace(-0.999, -0.9, 512), torch.linspace(0.9, 0.999, 512)))
    in_data = in_data.reshape(shape)
    
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    
    # We want to use compute_kernel_config to enable fp32 dest acc
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    
    output_tensor = ttnn.erfinv(input_tensor, compute_kernel_config=compute_kernel_config)
    output_data = ttnn.to_torch(output_tensor)
    
    golden = torch.erfinv(in_data)
    
    # Accurate mode should have very high PCC, much better than 0.98. Winitzki without NR achieves ~0.998.
    assert_with_pcc(golden, output_data, 0.9999)

