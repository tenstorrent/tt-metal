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
""" PyTorch DeepSeek MLP module."""

from torch import nn
from transformers.activations import ACT2FN

import ttnn


class DeepseekV3MLP(nn.Module):
    """
    Standard MLP layer with gated activation (SwiGLU variant).

    Pseudo-code:
    ```
    def forward(x):
        # Input shape: [batch_size, seq_len, hidden_size]
        # Output shape: [batch_size, seq_len, hidden_size]

        # Gate projection
        gate = gate_proj(x)  # [batch, seq, intermediate_size]

        # Up projection
        up = up_proj(x)  # [batch, seq, intermediate_size]

        # Apply activation function (SiLU) and element-wise multiply
        activated = act_fn(gate) * up  # [batch, seq, intermediate_size]

        # Down projection back to hidden size
        output = down_proj(activated)  # [batch, seq, hidden_size]

        return output
    ```

    Shape Examples (dense MLP with intermediate_size=18432):
    - Prefill: [1, 100, 7168] -> [1, 100, 18432] -> [1, 100, 7168]
    - Decode:  [1, 1, 7168] -> [1, 1, 18432] -> [1, 1, 7168]

    Shape Examples (expert MLP with intermediate_size=2048):
    - Prefill: [1, 100, 7168] -> [1, 100, 2048] -> [1, 100, 7168]
    - Decode:  [1, 1, 7168] -> [1, 1, 2048] -> [1, 1, 7168]

    Notes:
    - Uses SwiGLU activation: act_fn(gate) * up
    - No biases in any linear layers
    - intermediate_size can be either 18432 (dense) or 2048 (expert)
    """

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DeepseekV3MLPTTNN(nn.Module):
    """
    TTNN implementation of Standard MLP layer with gated activation (SwiGLU variant).

    Pseudo-code:
    ```
    def forward(x):
        # Input shape: [1, 1, batch_size*seq_len, hidden_size]
        # Output shape: [1, 1, batch_size*seq_len, hidden_size]

        # Gate and up projections (can be computed in parallel)
        gate = ttnn.linear(x, gate_weight, memory_config=mem_config)  # [1, 1, batch*seq, intermediate_size]
        up = ttnn.linear(x, up_weight, memory_config=mem_config)      # [1, 1, batch*seq, intermediate_size]

        # Apply SiLU activation to gate and multiply with up
        # Using fused multiply with activation on first input
        activated = ttnn.mul(gate, up, input_tensor_a_activation=ttnn.UnaryOpType.SILU)

        # Down projection back to hidden size
        output = ttnn.linear(activated, down_weight, memory_config=mem_config)  # [1, 1, batch*seq, hidden_size]

        return output
    ```

    Shape Examples (dense MLP with intermediate_size=18432):
    - Prefill: [1, 1, SEQ_LEN, 7168] -> [1, 1, SEQ_LEN, 18432] -> [1, 1, SEQ_LEN, 7168]
    - Decode:  [1, 1, BATCH_SIZE, 7168] -> [1, 1, BATCH_SIZE, 18432] -> [1, 1, BATCH_SIZE, 7168]

    Shape Examples (expert MLP with intermediate_size=2048):
    - Prefill: [1, 1, SEQ_LEN, 7168] -> [1, 1, SEQ_LEN, 2048] -> [1, 1, SEQ_LEN, 7168]
    - Decode:  [1, 1, BATCH_SIZE, 7168] -> [1, 1, BATCH_SIZE, 2048] -> [1, 1, BATCH_SIZE, 7168]

    Notes:
    - Uses SwiGLU activation: silu(gate) * up
    - No biases in any linear layers
    - intermediate_size can be either 18432 (dense) or 2048 (expert)
    - Fused activation with multiply for efficiency
    """

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        # Weights will be loaded as ttnn tensors on device
        self.gate_weight = None  # Will be set during weight loading
        self.up_weight = None  # Will be set during weight loading
        self.down_weight = None  # Will be set during weight loading

    def forward(self, x, memory_config=None, compute_kernel_config=None):
        """
        Args:
            x: TTNN tensor of shape [1, 1, BATCH_SIZE*SEQ_LEN, hidden_size]
            memory_config: Optional memory configuration
                          (L1_WIDTH_SHARDED for decode, DRAM for large prefill)
            compute_kernel_config: Optional compute configuration

        Returns:
            TTNN tensor of shape [1, 1, BATCH_SIZE*SEQ_LEN, hidden_size]
        """
        # Gate and up projections can be computed in parallel
        gate = ttnn.linear(
            x,
            self.gate_weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        up = ttnn.linear(
            x,
            self.up_weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        # Fused SiLU activation on gate and multiply with up
        activated = ttnn.mul(
            gate, up, input_tensor_a_activation=ttnn.UnaryOpType.SILU, memory_config=memory_config, dtype=ttnn.bfloat16
        )

        # Down projection
        output = ttnn.linear(
            activated,
            self.down_weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        return output

    def forward_fused_qkv_style(self, x, memory_config=None, compute_kernel_config=None):
        """
        Alternative implementation that fuses gate and up projections
        Similar to how QKV projections are often fused
        """
        # Fused gate and up projection
        # Weight shape: [hidden_size, intermediate_size * 2]
        gate_up = ttnn.linear(
            x,
            self.fused_gate_up_weight,  # Would need to be created by concatenating weights
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        # Split into gate and up
        # Output shape: [1, 1, batch*seq, intermediate_size * 2]
        split_size = self.intermediate_size
        gate = gate_up[..., :split_size]
        up = gate_up[..., split_size:]

        # Apply activation and multiply
        activated = ttnn.mul(
            gate, up, input_tensor_a_activation=ttnn.UnaryOpType.SILU, memory_config=memory_config, dtype=ttnn.bfloat16
        )

        # Down projection
        output = ttnn.linear(
            activated,
            self.down_weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        return output
