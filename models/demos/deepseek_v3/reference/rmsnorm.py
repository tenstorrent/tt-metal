# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeepSeek RMSNorm module."""

import torch
from torch import nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import ttnn


class DeepseekV3RMSNorm(nn.Module):
    """
    DeepseekV3RMSNorm is equivalent to T5LayerNorm

    Pseudo-code:
    ```
    def forward(hidden_states):
        # Input shape: [..., hidden_size=7168]
        # Output shape: [..., hidden_size=7168] (same as input)

        # 1. Convert to float32 for stability
        x = hidden_states.to(float32)

        # 2. Compute variance along last dimension
        variance = mean(x^2, dim=-1, keepdim=True)  # [..., 1]

        # 3. Apply RMS normalization
        x = x * rsqrt(variance + eps)  # [..., hidden_size]

        # 4. Apply learned weight and convert back to input dtype
        output = weight * x.to(input_dtype)  # [..., hidden_size]

        return output
    ```

    Shape Examples:
    - Prefill: [batch=1, seq=100, hidden=7168] -> [1, 100, 7168]
    - Decode:  [batch=1, seq=1, hidden=7168] -> [1, 1, 7168]
    """

    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(DeepseekV3RMSNorm)


class DeepseekV3RMSNormTTNN(nn.Module):
    """
    TTNN implementation of DeepseekV3 RMSNorm

    Pseudo-code:
    ```
    def forward(hidden_states):
        # Input shape: [..., hidden_size=7168]
        # Output shape: [..., hidden_size=7168] (same as input)

        # Compute RMS norm with fused weight multiplication
        output = ttnn.rms_norm(hidden_states,
                              epsilon=eps,
                              weight=weight,
                              memory_config=memory_config,
                              compute_kernel_config=compute_kernel_config)

        return output
    ```

    Shape Examples:
    - Prefill: [1, 1, SEQ_LEN, 7168] -> [1, 1, SEQ_LEN, 7168]
    - Decode:  [1, 1, BATCH_SIZE, 7168] -> [1, 1, BATCH_SIZE, 7168]

    Notes:
    - TTNN's rms_norm handles float32 computation internally
    - Weight multiplication is fused into the operation
    - For distributed mode, use rms_norm_pre_all_gather + rms_norm_post_all_gather
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        # Weight will be loaded as ttnn tensor on device
        self.weight = None  # Will be set during weight loading

    def forward(self, hidden_states, memory_config=None, compute_kernel_config=None):
        """
        Args:
            hidden_states: TTNN tensor of shape [1, 1, SEQ_LEN/BATCH_SIZE, 7168]
            memory_config: Optional memory configuration
            compute_kernel_config: Optional compute configuration

        Returns:
            TTNN tensor of same shape as input
        """
        output = ttnn.rms_norm(
            hidden_states,
            epsilon=self.variance_epsilon,
            weight=self.weight,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )

        return output

    def forward_distributed(
        self, hidden_states, stats_memory_config=None, memory_config=None, compute_kernel_config=None
    ):
        """
        Distributed version for tensor parallel models

        First computes local statistics, then all-gathers and applies normalization
        """
        # Pre all-gather: compute local statistics
        stats = ttnn.rms_norm_pre_all_gather(
            hidden_states,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=stats_memory_config,
        )

        # All-gather statistics across devices
        stats_gathered = ttnn.all_gather(
            stats, dim=3, num_links=1, memory_config=memory_config  # Gather along hidden dimension
        )

        # Post all-gather: apply normalization with gathered stats
        output = ttnn.rms_norm_post_all_gather(
            hidden_states,
            stats=stats_gathered,
            epsilon=self.variance_epsilon,
            weight=self.weight,
            compute_kernel_config=compute_kernel_config,
            memory_config=memory_config,
        )

        return output
