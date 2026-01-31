# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Custom GRU implementation for TTNN.

TTNN doesn't have a native GRU op, so we implement it using
linear layers and activations.
"""

from typing import Optional, Any, Tuple

import torch
import torch.nn as nn

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


def ttnn_gru_cell(
    x: Any,
    h: Any,
    weight_ih: Any,
    weight_hh: Any,
    bias_ih: Optional[Any] = None,
    bias_hh: Optional[Any] = None,
) -> Any:
    """
    Single GRU cell computation.

    GRU equations:
        r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)  # reset gate
        z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)  # update gate
        n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))  # new gate
        h' = (1 - z) * n + z * h  # output hidden state

    Args:
        x: Input tensor [B, input_size]
        h: Hidden state [B, hidden_size]
        weight_ih: Input-hidden weights [3*hidden_size, input_size]
        weight_hh: Hidden-hidden weights [3*hidden_size, hidden_size]
        bias_ih: Input-hidden bias [3*hidden_size]
        bias_hh: Hidden-hidden bias [3*hidden_size]

    Returns:
        New hidden state [B, hidden_size]
    """
    # Check if inputs are PyTorch tensors - use PyTorch path
    is_torch = isinstance(x, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch:
        # PyTorch fallback for development or when given torch tensors
        hidden_size = h.shape[-1]

        # Compute all gates at once
        gi = torch.nn.functional.linear(x, weight_ih, bias_ih)
        gh = torch.nn.functional.linear(h, weight_hh, bias_hh)

        # Split into reset, update, new components
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        # Gates
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)

        # Output
        h_new = (1 - update_gate) * new_gate + update_gate * h
        return h_new

    # TTNN implementation
    hidden_size = h.shape[-1]

    # Compute input projections: [B, 3*hidden]
    # Use L1 memory for faster access
    gi = ttnn.linear(x, weight_ih, bias=bias_ih, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Compute hidden projections: [B, 3*hidden]
    gh = ttnn.linear(h, weight_hh, bias=bias_hh, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Split into r, z, n components
    # Note: ttnn.split may need specific implementation
    # For now, use slicing
    i_r = gi[:, :hidden_size]
    i_z = gi[:, hidden_size:2*hidden_size]
    i_n = gi[:, 2*hidden_size:]

    h_r = gh[:, :hidden_size]
    h_z = gh[:, hidden_size:2*hidden_size]
    h_n = gh[:, 2*hidden_size:]

    # Reset gate: sigmoid(i_r + h_r)
    reset_gate = ttnn.sigmoid(ttnn.add(i_r, h_r))

    # Update gate: sigmoid(i_z + h_z)
    update_gate = ttnn.sigmoid(ttnn.add(i_z, h_z))

    # New gate: tanh(i_n + reset * h_n)
    reset_h_n = ttnn.multiply(reset_gate, h_n)
    new_gate = ttnn.tanh(ttnn.add(i_n, reset_h_n))

    # Output: (1 - z) * n + z * h
    ones = ttnn.ones_like(update_gate)
    one_minus_z = ttnn.subtract(ones, update_gate)

    term1 = ttnn.multiply(one_minus_z, new_gate)
    term2 = ttnn.multiply(update_gate, h)
    h_new = ttnn.add(term1, term2)

    return h_new


def ttnn_gru(
    x: Any,
    h0: Optional[Any],
    weight_ih: Any,
    weight_hh: Any,
    bias_ih: Optional[Any] = None,
    bias_hh: Optional[Any] = None,
    batch_first: bool = True,
) -> Tuple[Any, Any]:
    """
    Full GRU layer processing a sequence.

    Args:
        x: Input sequence [B, T, input_size] if batch_first else [T, B, input_size]
        h0: Initial hidden state [B, hidden_size] or None (zeros)
        weight_ih: Input-hidden weights [3*hidden_size, input_size]
        weight_hh: Hidden-hidden weights [3*hidden_size, hidden_size]
        bias_ih: Input-hidden bias
        bias_hh: Hidden-hidden bias
        batch_first: If True, input is [B, T, input_size]

    Returns:
        Tuple of:
            - output: All hidden states [B, T, hidden_size] or [T, B, hidden_size]
            - h_n: Final hidden state [1, B, hidden_size]
    """
    # Check if input is PyTorch tensor
    is_torch = isinstance(x, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch:
        # Helper to convert TTNN tensors to PyTorch (float32 for CPU compatibility)
        def to_torch(t, dtype=torch.float32):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.to(dtype)
            if TTNN_AVAILABLE:
                return ttnn.to_torch(t).to(dtype)
            return t

        # Convert input to float32 if needed
        x = x.to(torch.float32)

        # PyTorch fallback
        weight_ih_pt = to_torch(weight_ih)
        weight_hh_pt = to_torch(weight_hh)
        bias_ih_pt = to_torch(bias_ih)
        bias_hh_pt = to_torch(bias_hh)

        gru = nn.GRU(
            input_size=weight_ih_pt.shape[1],
            hidden_size=weight_ih_pt.shape[0] // 3,
            batch_first=batch_first,
        )
        gru.weight_ih_l0.data = weight_ih_pt
        gru.weight_hh_l0.data = weight_hh_pt
        if bias_ih_pt is not None:
            gru.bias_ih_l0.data = bias_ih_pt
        if bias_hh_pt is not None:
            gru.bias_hh_l0.data = bias_hh_pt

        return gru(x, h0.unsqueeze(0) if h0 is not None else None)

    # TTNN implementation

    # Get dimensions
    if batch_first:
        batch_size, seq_len, input_size = x.shape
    else:
        seq_len, batch_size, input_size = x.shape
        x = ttnn.permute(x, (1, 0, 2))  # [T, B, D] -> [B, T, D]

    hidden_size = weight_ih.shape[0] // 3

    # Initialize hidden state if not provided
    if h0 is None:
        if TTNN_AVAILABLE:
            h = ttnn.zeros((batch_size, hidden_size), dtype=x.dtype, device=x.device())
        else:
            h = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)
    else:
        h = h0

    # Process sequence
    outputs = []
    for t in range(seq_len):
        # Get input at timestep t
        x_t = x[:, t, :]  # [B, input_size]

        # Run GRU cell
        h = ttnn_gru_cell(x_t, h, weight_ih, weight_hh, bias_ih, bias_hh)

        outputs.append(h)

    # Stack outputs
    if TTNN_AVAILABLE:
        # Stack along time dimension
        output = ttnn.stack(outputs, dim=1)  # [B, T, hidden_size]
    else:
        output = torch.stack(outputs, dim=1)

    if not batch_first:
        output = ttnn.permute(output, (1, 0, 2)) if TTNN_AVAILABLE else output.permute(1, 0, 2)

    # h_n has shape [1, B, hidden_size] for compatibility
    if TTNN_AVAILABLE:
        h_n = ttnn.unsqueeze(h, 0)
    else:
        h_n = h.unsqueeze(0)

    return output, h_n


class GRULayer:
    """
    GRU layer wrapper with pre-loaded weights.

    Provides interface compatible with PyTorch nn.GRU.
    """

    def __init__(
        self,
        weight_ih: Any,
        weight_hh: Any,
        bias_ih: Optional[Any] = None,
        bias_hh: Optional[Any] = None,
        batch_first: bool = True,
    ):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.batch_first = batch_first

        # Infer dimensions
        self.hidden_size = weight_ih.shape[0] // 3
        self.input_size = weight_ih.shape[1]

    def __call__(
        self,
        x: Any,
        h0: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        return ttnn_gru(
            x, h0,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
            batch_first=self.batch_first,
        )

    def flatten_parameters(self):
        """No-op for compatibility with PyTorch GRU."""
        pass
