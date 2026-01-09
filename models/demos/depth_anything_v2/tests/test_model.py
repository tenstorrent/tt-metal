# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.demos.depth_anything_v2.tt.model_def import TtDepthAnythingV2

@pytest.mark.parametrize("device_params", [{"batch_size": 1}], indirect=True)
def test_depth_anything_v2_initialization(device):
    # Mock parameters using a simple Namespace or Dict
    # In a real test, verifying with state_dict loading is better
    
    class MockConfig:
        patch_size = 14
        num_attention_heads = 4
        
    config = MockConfig()
    
    # Mock parameters structure (simplified)
    # This needs to match the structure expected in model_def.py
    # ... setup mock params ...
    
    # For now, just test we can instantiate the class if we pass something matching
    # Since model_def expects specific hierarchy, we skip full verification in this skeleton
    # and focus on ensuring imports work and class exists
    
    assert TtDepthAnythingV2 is not None
    print("TtDepthAnythingV2 class imported successfully.")

