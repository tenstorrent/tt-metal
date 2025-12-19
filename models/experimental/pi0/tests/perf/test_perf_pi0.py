# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance tests for PI0 model.

Benchmarks TTNN implementation performance.
"""

import pytest
import torch
import time

from models.experimental.pi0.common.configs import PI0ModelConfig


class TestPI0Performance:
    """Performance tests for PI0 model."""
    
    @pytest.mark.parametrize("batch_size", [1])
    def test_full_model_perf(self, device, batch_size):
        """
        Test full PI0 model performance.
        
        This test requires a real checkpoint and device.
        """
        # Placeholder for performance testing
        pytest.skip(
            "Performance test requires real checkpoint and device setup. "
            "Use ttnn_pi0_reference/test_full_model_inference_pcc.py for benchmarking."
        )
    
    @pytest.mark.parametrize("batch_size", [1])
    def test_vision_tower_perf(self, device, batch_size):
        """Test vision tower performance."""
        pytest.skip("Requires real checkpoint")
    
    @pytest.mark.parametrize("batch_size", [1])
    def test_suffix_embedding_perf(self, device, batch_size):
        """Test suffix embedding performance."""
        pytest.skip("Requires real checkpoint")
