# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.depth_anything_v2.tt.model_def import TtDepthAnythingV2

@pytest.mark.parametrize("device_params", [{"batch_size": 1}], indirect=True)
def test_depth_anything_v2_class_import(device):
    # Ensure the device fixture is initialized for this test
    assert device is not None
    
    class MockConfig:
        patch_size = 14
        num_attention_heads = 16
        hidden_size = 1024
        
    config = MockConfig()
    
    # Mock parameters structure (simplified)
    class MockParams:
        def __init__(self):
            self.backbone = type('obj', (object,), {'encoder': type('obj', (object,), {'layer': []})})
            self.neck = type('obj', (object,), {'reassemble': [{} for _ in range(4)], 'fusion': {}})
            self.head = {}
            
    parameters = MockParams()
    
    assert TtDepthAnythingV2 is not None
    print("TtDepthAnythingV2 class imported successfully.")

