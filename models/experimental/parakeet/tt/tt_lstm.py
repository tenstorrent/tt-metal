import ttnn
import torch
from typing import Optional, Tuple, List, Dict


class TtLSTMCell:
    """
    TTNN implementation of LSTM cell that matches PyTorch's torch.nn.LSTM behavior.

    This class implements LSTM computation equivalent to PyTorch's _VF.lstm
    for a single timestep. It supports multiple layers and handles:
    - Gate computation (input, forget, cell, output)
    - Proper weight and bias handling for all layers
    - Tensor reshaping for TTNN compatibility
    - Memory configuration management
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        device,
        num_layers: int = 1,
        dtype=ttnn.float32,
        memory_config=None,
        weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Initialize TTNN LSTM Cell with multiple layers.

        Args:
            input_size: Size of input features (for first layer)
            hidden_size: Size of hidden state
            device: TTNN device
            num_layers: Number of LSTM layers
            dtype: TTNN data type
            memory_config: Memory configuration for tensors
            weights: Optional list of weight dictionaries for each layer.
                     Each dict should contain: weight_ih, weight_hh, bias_ih, bias_hh
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        # Store weights for each layer
        self.layer_weights = []

        if weights is not None:
            self._load_weights(weights)
        else:
            # Initialize with random weights (for testing)
            self._init_weights()

    def _init_weights(self):
        """Initialize weights randomly for all layers."""
        weights = []
        for layer_idx in range(self.num_layers):
            # First layer takes input_size, subsequent layers take hidden_size
            layer_input_size = self.input_size if layer_idx == 0 else self.hidden_size

            weight_ih = torch.randn(4 * self.hidden_size, layer_input_size) * 0.1
            weight_hh = torch.randn(4 * self.hidden_size, self.hidden_size) * 0.1
            bias_ih = torch.zeros(4 * self.hidden_size)
            bias_hh = torch.zeros(4 * self.hidden_size)

            weights.append(
                {
                    "weight_ih": weight_ih,
                    "weight_hh": weight_hh,
                    "bias_ih": bias_ih,
                    "bias_hh": bias_hh,
                }
            )

        self._load_weights(weights)

    def _load_weights(self, weights: List[Dict[str, torch.Tensor]]):
        """
        Load weights from PyTorch format and convert to TTNN format for all layers.

        Args:
            weights: List of weight dictionaries, one per layer.
                    Each dict should contain: weight_ih, weight_hh, bias_ih, bias_hh
        """
        if len(weights) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} weight dictionaries, got {len(weights)}")

        self.layer_weights = []

        for layer_idx, layer_weights in enumerate(weights):
            weight_ih = layer_weights["weight_ih"]
            weight_hh = layer_weights["weight_hh"]
            bias_ih = layer_weights.get("bias_ih")
            bias_hh = layer_weights.get("bias_hh")

            # Transpose weights for TTNN: [out_features, in_features] -> [in_features, out_features]
            weight_ih_tt = weight_ih.transpose(0, 1).contiguous()
            weight_hh_tt = weight_hh.transpose(0, 1).contiguous()

            # Convert to TTNN tensors
            weight_ih_ttnn = ttnn.from_torch(
                weight_ih_tt,
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
            )

            weight_hh_ttnn = ttnn.from_torch(
                weight_hh_tt,
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
            )

            # Combine biases: bias = bias_ih + bias_hh (PyTorch LSTM convention)
            if bias_ih is not None and bias_hh is not None:
                bias = (bias_ih + bias_hh).contiguous()
            elif bias_ih is not None:
                bias = bias_ih.contiguous()
            elif bias_hh is not None:
                bias = bias_hh.contiguous()
            else:
                bias = torch.zeros(4 * self.hidden_size)

            # Convert bias to TTNN (reshape to [1, 1, 1, 4*hidden_size] for broadcasting)
            bias_reshaped = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            bias_ttnn = ttnn.from_torch(
                bias_reshaped,
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
            )

            self.layer_weights.append(
                {
                    "weight_ih": weight_ih_ttnn,
                    "weight_hh": weight_hh_ttnn,
                    "bias": bias_ttnn,
                }
            )

    def _ensure_4d(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Ensure tensor is 4D for TTNN operations."""
        logical_shape = tensor.shape

        if len(logical_shape) == 3:
            # [1, B, H] -> [1, 1, B, H]
            return ttnn.unsqueeze(tensor, dim=1)
        elif len(logical_shape) == 2:
            # [B, H] -> [1, 1, B, H]
            tensor = ttnn.unsqueeze(tensor, dim=0)  # [1, B, H]
            tensor = ttnn.unsqueeze(tensor, dim=1)  # [1, 1, B, H]
            return tensor
        return tensor

    def _ensure_3d(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Ensure tensor is 3D [1, B, H] for output."""
        logical_shape = tensor.shape
        if len(logical_shape) == 4 and logical_shape[1] == 1:
            # [1, 1, B, H] -> [1, B, H]
            return ttnn.reshape(tensor, [logical_shape[0], logical_shape[2], logical_shape[3]])
        return tensor

    def _lstm_single_layer(
        self,
        x: ttnn.Tensor,
        h: ttnn.Tensor,
        c: ttnn.Tensor,
        layer_idx: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Single LSTM layer forward pass.

        Args:
            x: Input tensor [1, 1, B, input_size] or [1, 1, B, hidden_size]
            h: Hidden state [1, 1, B, hidden_size]
            c: Cell state [1, 1, B, hidden_size]
            layer_idx: Layer index

        Returns:
            h_new, c_new: New hidden and cell states [1, 1, B, hidden_size]
        """
        weights = self.layer_weights[layer_idx]

        # Compute gates: gates = W_ih·x + W_hh·h + bias

        gates_x = ttnn.linear(x, weights["weight_ih"], bias=None, memory_config=self.memory_config)
        gates_h = ttnn.linear(h, weights["weight_hh"], bias=None, memory_config=self.memory_config)

        # Add gates and bias
        gates = ttnn.add(gates_x, gates_h, memory_config=self.memory_config)
        gates = ttnn.add(gates, weights["bias"], memory_config=self.memory_config)

        # Split gates into 4 parts: [input, forget, cell, output]
        # Gate ordering matches PyTorch: [i, f, g, o]
        chunk_size = self.hidden_size
        gates_shape = gates.shape

        # Slice gates: gates shape is [1, 1, B, 4*hidden_size]
        i = ttnn.slice(gates, [0, 0, 0, 0], [gates_shape[0], gates_shape[1], gates_shape[2], chunk_size])
        f = ttnn.slice(gates, [0, 0, 0, chunk_size], [gates_shape[0], gates_shape[1], gates_shape[2], 2 * chunk_size])
        g = ttnn.slice(
            gates, [0, 0, 0, 2 * chunk_size], [gates_shape[0], gates_shape[1], gates_shape[2], 3 * chunk_size]
        )
        o = ttnn.slice(
            gates, [0, 0, 0, 3 * chunk_size], [gates_shape[0], gates_shape[1], gates_shape[2], 4 * chunk_size]
        )

        # Apply activations
        i = ttnn.sigmoid(i)
        f = ttnn.sigmoid(f)
        g = ttnn.tanh(g)
        o = ttnn.sigmoid(o)

        # Compute new cell state: c_new = f * c + i * g
        c_new = ttnn.add(
            ttnn.multiply(f, c, memory_config=self.memory_config),
            ttnn.multiply(i, g, memory_config=self.memory_config),
            memory_config=self.memory_config,
        )

        # Compute new hidden state: h_new = o * tanh(c_new)
        h_new = ttnn.multiply(
            o,
            ttnn.tanh(c_new, memory_config=self.memory_config),
            memory_config=self.memory_config,
        )

        return h_new, c_new

    def forward(
        self,
        x: ttnn.Tensor,
        h: Optional[ttnn.Tensor] = None,
        c: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Tuple[List[ttnn.Tensor], List[ttnn.Tensor]]]:
        """
        Forward pass of multi-layer LSTM cell.

        Args:
            x: Input tensor [B, input_size] or [1, B, input_size] or [1, 1, B, input_size]
            h: Hidden states [num_layers, B, hidden_size] or list of [1, B, hidden_size] tensors
            c: Cell states [num_layers, B, hidden_size] or list of [1, B, hidden_size] tensors

        Returns:
            h_new: New hidden state [1, B, hidden_size] (from last layer)
            (h_all, c_all): Tuple of lists containing hidden and cell states for all layers
        """
        # Get batch size
        x_shape = x.shape
        batch_size = x_shape[-2] if len(x_shape) >= 2 else x_shape[0]

        # Initialize states if not provided
        if h is None:
            h_tensor = ttnn.zeros(
                [self.num_layers, batch_size, self.hidden_size],
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
            )
        else:
            h_tensor = h

        if c is None:
            c_tensor = ttnn.zeros(
                [self.num_layers, batch_size, self.hidden_size],
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
            )
        else:
            c_tensor = c

        # Ensure input is 4D
        x_4d = self._ensure_4d(x)

        h_new_list = []
        c_new_list = []
        for layer_idx in range(self.num_layers):
            # Get states for this layer
            h_layer = self._ensure_4d(h_tensor[layer_idx])
            c_layer = self._ensure_4d(c_tensor[layer_idx])

            # Forward pass through layer
            # import pdb; pdb.set_trace()
            h_new, c_new = self._lstm_single_layer(x_4d, h_layer, c_layer, layer_idx)

            # Store new states
            h_new_list.append(self._ensure_3d(h_new))
            c_new_list.append(self._ensure_3d(c_new))

            # Output of this layer becomes input to next layer
            x_4d = h_new

        # Return output from last layer and all states
        h_new_tensor = ttnn.concat(h_new_list, dim=0)
        c_new_tensor = ttnn.concat(c_new_list, dim=0)
        return h_new_list[-1], (h_new_tensor, c_new_tensor)

    def __call__(
        self,
        x: ttnn.Tensor,
        h: Optional[ttnn.Tensor] = None,
        c: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Tuple[List[ttnn.Tensor], List[ttnn.Tensor]]]:
        """Alias for forward method."""
        return self.forward(x, h, c)


# Convenience function for backward compatibility (single layer)
def ttnn_lstm_cell(
    x: ttnn.Tensor,
    h: ttnn.Tensor,
    c: ttnn.Tensor,
    weight_ih: ttnn.Tensor,
    weight_hh: ttnn.Tensor,
    bias: ttnn.Tensor,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Functional version of LSTM cell (for backward compatibility - single layer only).

    Args:
        x: Input tensor
        h: Hidden state
        c: Cell state
        weight_ih: Input-to-hidden weight (TTNN format: [input_size, 4*hidden_size])
        weight_hh: Hidden-to-hidden weight (TTNN format: [hidden_size, 4*hidden_size])
        bias: Combined bias (TTNN format)

    Returns:
        h_new, c_new: New hidden and cell states
    """
    gates_x = ttnn.linear(x, weight_ih, bias=None)
    gates_h = ttnn.linear(h, weight_hh, bias=None)

    gates = ttnn.add(gates_x, gates_h)
    gates = ttnn.add(gates, bias)

    # Split gates: [i, f, g, o]
    hidden_size = gates.shape[-1] // 4
    gates_shape = gates.shape
    i = ttnn.slice(gates, [0, 0, 0, 0], [gates_shape[0], gates_shape[1], gates_shape[2], hidden_size])
    f = ttnn.slice(gates, [0, 0, 0, hidden_size], [gates_shape[0], gates_shape[1], gates_shape[2], 2 * hidden_size])
    g = ttnn.slice(gates, [0, 0, 0, 2 * hidden_size], [gates_shape[0], gates_shape[1], gates_shape[2], 3 * hidden_size])
    o = ttnn.slice(gates, [0, 0, 0, 3 * hidden_size], [gates_shape[0], gates_shape[1], gates_shape[2], 4 * hidden_size])

    i = ttnn.sigmoid(i)
    f = ttnn.sigmoid(f)
    g = ttnn.tanh(g)
    o = ttnn.sigmoid(o)

    c_new = ttnn.add(ttnn.multiply(f, c), ttnn.multiply(i, g))

    h_new = ttnn.multiply(o, ttnn.tanh(c_new))

    return h_new, c_new
