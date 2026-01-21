# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Reference Encoder for speaker embedding extraction.

Extracts speaker characteristics from mel spectrograms using
Conv2d layers followed by GRU for temporal modeling.
"""

from typing import Optional, Any, List

import torch
import torch.nn.functional as F

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from models.demos.openvoice.tt.modules.gru import GRULayer


class TTNNReferenceEncoder:
    """
    Reference Encoder for extracting speaker embeddings from mel spectrograms.

    Architecture:
        Input: [N, Ty/r, n_mels*r] mel spectrogram
        -> 6 Conv2d layers (32->32->64->64->128->128) with ReLU
        -> Reshape for GRU
        -> GRU (bidirectional=False, hidden=128)
        -> Linear projection to gin_channels

    Output: [N, gin_channels] speaker embedding
    """

    # Standard filter configuration from OpenVoice
    REF_ENC_FILTERS = [32, 32, 64, 64, 128, 128]

    def __init__(
        self,
        spec_channels: int,
        gin_channels: int,
        conv_weights: List[Any],
        conv_biases: List[Any],
        gru_weight_ih: Any,
        gru_weight_hh: Any,
        gru_bias_ih: Optional[Any],
        gru_bias_hh: Optional[Any],
        proj_weight: Any,
        proj_bias: Any,
        layernorm_weight: Optional[Any] = None,
        layernorm_bias: Optional[Any] = None,
        device: Optional[Any] = None,
    ):
        """
        Initialize Reference Encoder.

        Args:
            spec_channels: Number of mel channels (n_mels)
            gin_channels: Output speaker embedding dimension
            conv_weights: List of 6 Conv2d weight tensors
            conv_biases: List of 6 Conv2d bias tensors
            gru_weight_ih: GRU input-hidden weight
            gru_weight_hh: GRU hidden-hidden weight
            gru_bias_ih: GRU input-hidden bias
            gru_bias_hh: GRU hidden-hidden bias
            proj_weight: Final projection weight
            proj_bias: Final projection bias
            layernorm_weight: Optional LayerNorm weight
            layernorm_bias: Optional LayerNorm bias
            device: TTNN device
        """
        self.spec_channels = spec_channels
        self.gin_channels = gin_channels
        self.device = device

        # Conv2d layers
        self.conv_weights = conv_weights
        self.conv_biases = conv_biases

        # LayerNorm (optional)
        self.layernorm_weight = layernorm_weight
        self.layernorm_bias = layernorm_bias
        self.use_layernorm = layernorm_weight is not None

        # GRU
        self.gru = GRULayer(
            weight_ih=gru_weight_ih,
            weight_hh=gru_weight_hh,
            bias_ih=gru_bias_ih,
            bias_hh=gru_bias_hh,
            batch_first=True,
        )

        # Final projection
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias

        # Calculate output channels after 6 conv layers with stride 2
        self.out_channels = self._calculate_channels(spec_channels, 3, 2, 1, 6)

    def _calculate_channels(self, L: int, kernel_size: int, stride: int, pad: int, n_convs: int) -> int:
        """Calculate spatial dimension after n convolutions."""
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

    def __call__(self, inputs: Any, mask: Optional[Any] = None) -> Any:
        """
        Extract speaker embedding from mel spectrogram.

        Args:
            inputs: Mel spectrogram [N, Ty, n_mels]
            mask: Optional mask (unused)

        Returns:
            Speaker embedding [N, gin_channels]
        """
        # Check if input is PyTorch tensor
        is_torch = isinstance(inputs, torch.Tensor)

        # Note: Reference encoder uses PyTorch path due to L1 memory constraints
        # on TT hardware for large mel spectrograms. This is acceptable since
        # speaker embedding extraction is not in the real-time inference path.
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(inputs)

        # Convert TTNN tensor to PyTorch for reference encoder
        # (Falls back to CPU due to memory constraints on device)
        inputs_torch = ttnn.to_torch(inputs)
        return self._forward_pytorch(inputs_torch)

    def _forward_pytorch(self, inputs):
        """PyTorch fallback implementation."""
        # Helper to convert TTNN tensors to PyTorch
        def to_torch(t, dtype=torch.float32):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.to(dtype)
            if TTNN_AVAILABLE:
                return ttnn.to_torch(t).to(dtype)
            return t

        # Ensure input is float32
        inputs = inputs.to(torch.float32)
        N = inputs.size(0)

        # Reshape: [N, T, n_mels] -> [N, 1, T, n_mels]
        out = inputs.view(N, 1, -1, self.spec_channels)

        # LayerNorm on last dimension
        if self.use_layernorm:
            ln_w = to_torch(self.layernorm_weight)
            ln_b = to_torch(self.layernorm_bias)
            out = F.layer_norm(out, (self.spec_channels,), ln_w, ln_b)

        # 6 Conv2d layers with ReLU
        for i, (conv_w, conv_b) in enumerate(zip(self.conv_weights, self.conv_biases)):
            w = to_torch(conv_w)
            b = to_torch(conv_b)
            out = F.conv2d(out, w, b, stride=2, padding=1)
            out = F.relu(out)

        # Reshape for GRU: [N, 128, T', n_mels'] -> [N, T', 128*n_mels']
        out = out.transpose(1, 2)  # [N, T', 128, n_mels']
        T = out.size(1)
        out = out.contiguous().view(N, T, -1)

        # GRU - only need final hidden state
        _, h = self.gru(out, h0=None)

        # Project to speaker embedding
        # h is [1, N, hidden_size], squeeze to [N, hidden_size]
        h = h.squeeze(0)
        proj_w = to_torch(self.proj_weight)
        proj_b = to_torch(self.proj_bias)
        out = F.linear(h, proj_w, proj_b)

        return out

    def _forward_ttnn(self, inputs):
        """TTNN implementation."""
        batch_size = inputs.shape[0]

        # Reshape: [N, T, n_mels] -> [N, 1, T, n_mels]
        # TTNN uses NHWC, so we go to [N, T, n_mels, 1] then permute
        out = ttnn.reshape(inputs, (batch_size, -1, self.spec_channels, 1))
        out = ttnn.permute(out, (0, 3, 1, 2))  # -> [N, 1, T, n_mels]

        # LayerNorm (if enabled)
        if self.use_layernorm:
            # TTNN layer_norm requires TILE layout
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
            if self.device is not None and not ttnn.is_tensor_storage_on_device(out):
                out = ttnn.to_device(out, self.device)

            # Prepare layernorm weights
            ln_weight = self.layernorm_weight
            ln_bias = self.layernorm_bias
            if ln_weight is not None:
                ln_weight = ttnn.to_layout(ln_weight, ttnn.TILE_LAYOUT)
                if self.device is not None and not ttnn.is_tensor_storage_on_device(ln_weight):
                    ln_weight = ttnn.to_device(ln_weight, self.device)
            if ln_bias is not None:
                ln_bias = ttnn.to_layout(ln_bias, ttnn.TILE_LAYOUT)
                if self.device is not None and not ttnn.is_tensor_storage_on_device(ln_bias):
                    ln_bias = ttnn.to_device(ln_bias, self.device)

            # Apply layernorm on last dimension
            out = ttnn.layer_norm(
                out,
                weight=ln_weight,
                bias=ln_bias,
                epsilon=1e-5,
            )

        # 6 Conv2d layers
        filters = [1] + self.REF_ENC_FILTERS
        for i in range(6):
            in_channels = filters[i]
            out_channels = filters[i + 1]

            # Configure conv2d with WIDTH_SHARDED for channel parallelism
            conv_config = ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat16,
                config_tensors_in_dram=True,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            )

            # Get current spatial dimensions
            _, _, height, width = out.shape

            conv_kwargs = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "batch_size": batch_size,
                "input_height": height,
                "input_width": width,
                "kernel_size": (3, 3),
                "stride": (2, 2),
                "padding": (1, 1),
                "dilation": (1, 1),
                "groups": 1,
            }

            # Prepare conv weights
            weight = self.conv_weights[i]
            if self.device is not None and not ttnn.is_tensor_storage_on_device(weight):
                # Compute config
                compute_config = ttnn.init_device_compute_kernel_config(
                    self.device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=True,
                    fp32_dest_acc_en=False,
                )
                weight = ttnn.prepare_conv_weights(
                    weight_tensor=weight,
                    weights_format="OIHW",
                    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    input_layout=out.get_layout(),
                    input_dtype=out.dtype,
                    has_bias=False,
                    device=self.device,
                    conv_config=conv_config,
                    **conv_kwargs,
                )
                if not ttnn.is_tensor_storage_on_device(weight):
                    weight = ttnn.to_device(weight, self.device)

            out = ttnn.conv2d(
                input_tensor=out,
                weight_tensor=weight,
                bias_tensor=None,
                device=self.device,
                conv_config=conv_config,
                **conv_kwargs,
            )

            # Add bias manually if present
            bias = self.conv_biases[i]
            if bias is not None:
                # Reshape bias for broadcasting: [C] -> [1, C, 1, 1]
                if bias.shape.rank == 1:
                    bias = ttnn.reshape(bias, (1, out_channels, 1, 1))
                bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
                if self.device is not None and not ttnn.is_tensor_storage_on_device(bias):
                    bias = ttnn.to_device(bias, self.device)
                out = ttnn.add(out, bias)

            # ReLU activation
            out = ttnn.relu(out)

        # Reshape for GRU
        # out is [N, C, T', mel'] in NCHW-ish format
        # Need [N, T', C*mel'] for GRU
        out = ttnn.permute(out, (0, 2, 1, 3))  # [N, T', C, mel']
        T = out.shape[1]
        out = ttnn.reshape(out, (batch_size, T, -1))  # [N, T', C*mel']

        # GRU
        _, h = self.gru(out, h0=None)

        # Project to speaker embedding
        # h is [1, N, hidden_size]
        h = ttnn.squeeze(h, 0)  # [N, hidden_size]
        out = ttnn.linear(h, self.proj_weight, bias=self.proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        return out

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        prefix: str,
        spec_channels: int,
        gin_channels: int,
        device: Optional[Any] = None,
    ) -> "TTNNReferenceEncoder":
        """
        Create ReferenceEncoder from state dict.

        Args:
            state_dict: Model state dict (with weight norm already fused)
            prefix: Key prefix (e.g., "ref_enc")
            spec_channels: Number of mel channels
            gin_channels: Speaker embedding dimension
            device: TTNN device

        Returns:
            Initialized TTNNReferenceEncoder
        """
        # Load conv weights
        conv_weights = []
        conv_biases = []
        for i in range(6):
            w = state_dict.get(f"{prefix}.convs.{i}.weight")
            b = state_dict.get(f"{prefix}.convs.{i}.bias")
            conv_weights.append(w)
            conv_biases.append(b)

        # Load GRU weights
        gru_weight_ih = state_dict.get(f"{prefix}.gru.weight_ih_l0")
        gru_weight_hh = state_dict.get(f"{prefix}.gru.weight_hh_l0")
        gru_bias_ih = state_dict.get(f"{prefix}.gru.bias_ih_l0")
        gru_bias_hh = state_dict.get(f"{prefix}.gru.bias_hh_l0")

        # Load projection
        proj_weight = state_dict.get(f"{prefix}.proj.weight")
        proj_bias = state_dict.get(f"{prefix}.proj.bias")

        # Load layernorm (optional)
        layernorm_weight = state_dict.get(f"{prefix}.layernorm.weight")
        layernorm_bias = state_dict.get(f"{prefix}.layernorm.bias")

        return cls(
            spec_channels=spec_channels,
            gin_channels=gin_channels,
            conv_weights=conv_weights,
            conv_biases=conv_biases,
            gru_weight_ih=gru_weight_ih,
            gru_weight_hh=gru_weight_hh,
            gru_bias_ih=gru_bias_ih,
            gru_bias_hh=gru_bias_hh,
            proj_weight=proj_weight,
            proj_bias=proj_bias,
            layernorm_weight=layernorm_weight,
            layernorm_bias=layernorm_bias,
            device=device,
        )
