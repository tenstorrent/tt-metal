# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""mLSTM Cell implementation.

The mLSTM cell is the core computational unit of the mLSTM layer.
It computes input and forget gates based on Q, K, V inputs and
applies the parallel mLSTM computation with stabilization.

Reference:
    - Paper: "xLSTM: Extended Long Short-Term Memory" (https://arxiv.org/abs/2405.04517)
    - JAX reference: https://github.com/NX-AI/mlstm_kernels
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import ttnn

import ttml
from ttml.modules import AbstractModuleBase, Parameter, LinearLayer
from .components import MultiHeadLayerNorm
from .mlstm import mlstm_parallel


@dataclass
class mLSTMCellConfig:
    """Configuration for mLSTMCell."""

    context_length: int
    embedding_dim: int
    num_heads: int


class mLSTMCell(AbstractModuleBase):
    """mLSTM Cell with input/forget gates and layer normalization.

    The cell takes Q, K, V tensors, computes input and forget gate
    pre-activations, applies the parallel mLSTM computation, and
    normalizes the output.

    Args:
        config: mLSTMCellConfig with context_length, embedding_dim, num_heads
    """

    def __init__(self, config: mLSTMCellConfig) -> None:
        super().__init__()
        self.config = config

        # Gate linear layers
        # Input: concat(q, k, v) of shape (B, S, 3*embedding_dim)
        # Output: (B, S, num_heads)
        gate_input_dim = 3 * config.embedding_dim

        self.igate = LinearLayer(gate_input_dim, config.num_heads, bias=True)
        self.fgate = LinearLayer(gate_input_dim, config.num_heads, bias=True)

        # Output normalization (per head)
        self.outnorm = MultiHeadLayerNorm(
            ndim=config.embedding_dim,
            num_heads=config.num_heads,
            weight=True,
            bias=False,
        )

        # Initialize gate biases
        self._init_gate_biases()

    def _init_gate_biases(self) -> None:
        """Initialize gate biases according to xLSTM paper.

        Forget gate: linspace from 3.0 to 6.0
        Input gate: normal(0, 0.1)
        """
        num_heads = self.config.num_heads

        # Forget gate: linspace init from 3.0 to 6.0
        fgate_bias = np.linspace(3.0, 6.0, num_heads).astype(np.float32)
        # Update the bias parameter
        # Note: LinearLayer parameters are accessible via the layer
        # For now, we'll rely on the default initialization

        # Input gate: normal init
        igate_bias = np.random.normal(0.0, 0.1, num_heads).astype(np.float32)

    def forward(
        self,
        q: ttml.autograd.Tensor,
        k: ttml.autograd.Tensor,
        v: ttml.autograd.Tensor,
    ) -> ttml.autograd.Tensor:
        """Forward pass of mLSTM cell.

        Args:
            q: Query tensor of shape (B, S, H) where H = embedding_dim
            k: Key tensor of shape (B, S, H)
            v: Value tensor of shape (B, S, H)

        Returns:
            Output tensor of shape (B, S, H)
        """
        B = q.get_value().shape[0]
        S = q.get_value().shape[1]
        H = self.config.embedding_dim
        NH = self.config.num_heads
        DH = H // NH

        # Concatenate q, k, v for gate input
        q_val = q.get_value()
        k_val = k.get_value()
        v_val = v.get_value()
        if_gate_input = ttnn.concat([q_val, k_val, v_val], dim=-1)
        if_gate_input_tensor = ttml.autograd.create_tensor(
            if_gate_input, requires_grad=True
        )

        # Reshape q, k, v to (B, S, NH, DH) then transpose to (B, NH, S, DH)
        q_heads = ttnn.reshape(q_val, (B, S, NH, DH))
        q_heads = ttnn.transpose(q_heads, 1, 2)  # (B, NH, S, DH)

        k_heads = ttnn.reshape(k_val, (B, S, NH, DH))
        k_heads = ttnn.transpose(k_heads, 1, 2)  # (B, NH, S, DH)

        v_heads = ttnn.reshape(v_val, (B, S, NH, DH))
        v_heads = ttnn.transpose(v_heads, 1, 2)  # (B, NH, S, DH)

        # Compute gate pre-activations
        igate_preact = self.igate(if_gate_input_tensor)  # (B, S, NH)
        fgate_preact = self.fgate(if_gate_input_tensor)  # (B, S, NH)

        # Transpose gate activations to (B, NH, S)
        igate_val = ttnn.transpose(igate_preact.get_value(), 1, 2)  # (B, NH, S)
        fgate_val = ttnn.transpose(fgate_preact.get_value(), 1, 2)  # (B, NH, S)

        # Create autograd tensors for mLSTM
        q_tensor = ttml.autograd.create_tensor(q_heads, requires_grad=True)
        k_tensor = ttml.autograd.create_tensor(k_heads, requires_grad=True)
        v_tensor = ttml.autograd.create_tensor(v_heads, requires_grad=True)
        i_tensor = ttml.autograd.create_tensor(igate_val, requires_grad=True)
        f_tensor = ttml.autograd.create_tensor(fgate_val, requires_grad=True)

        # Apply parallel mLSTM
        h_state = mlstm_parallel(q_tensor, k_tensor, v_tensor, i_tensor, f_tensor)

        # Apply output normalization
        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)

        # Reshape back to (B, S, H)
        h_out = ttnn.transpose(h_state_norm.get_value(), 1, 2)  # (B, S, NH, DH)
        h_out = ttnn.reshape(h_out, (B, S, H))  # (B, S, H)

        return ttml.autograd.create_tensor(h_out, requires_grad=True)
