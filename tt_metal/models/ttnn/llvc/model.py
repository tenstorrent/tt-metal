#!/usr/bin/env python3
"""
Core model definition for Low-Latency Low-Resource Voice Conversion (LLVC) using TTNN.

This module implements a lightweight voice conversion model optimized for
Tenstorrent hardware via the TTNN tensor library. The architecture consists of
an encoder stack, a bottleneck projection, and a decoder stack, all built from
feed‑forward blocks. The model operates on mel‑spectrogram frames and is
designed for low latency on resource‑constrained devices.

All tensor operations are performed using TTNN primitives, with tensors stored
in TILE_LAYOUT on a specified device. The configuration is validated at
construction time to ensure correctness.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import ttnn


logger = logging.getLogger(__name__)


class LLVCConfig:
    """Configuration dataclass for the LLVC model.

    Attributes:
        input_channels: Number of input mel bands (default 80).
        hidden_channels: Number of channels in hidden layers.
        latent_dim: Dimensionality of the bottleneck embedding.
        output_channels: Number of output mel bands (should match input).
        num_layers: Number of feedforward blocks in encoder and decoder.
        dropout: Dropout probability (applied after activations).
        activation: Activation function name ('silu', 'gelu', 'relu').
    """

    VALID_ACTIVATIONS: Tuple[str, ...] = ("silu", "gelu", "relu")

    def __init__(
        self,
        input_channels: int = 80,
        hidden_channels: int = 512,
        latent_dim: int = 128,
        output_channels: int = 80,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "silu",
    ) -> None:
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if output_channels <= 0:
            raise ValueError(f"output_channels must be positive, got {output_channels}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0], got {dropout}")
        if activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {self.VALID_ACTIVATIONS}, got {activation}"
            )

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation

    def __repr__(self) -> str:
        return (
            f"LLVCConfig(input_channels={self.input_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"latent_dim={self.latent_dim}, "
            f"output_channels={self.output_channels}, "
            f"num_layers={self.num_layers}, "
            f"dropout={self.dropout}, "
            f"activation='{self.activation}')"
        )


class FeedForwardBlock:
    """A single feedforward block: Linear → LayerNorm → Activation → Dropout.

    Weights, biases, and layer‑norm parameters are stored as TTNN tensors in
    TILE_LAYOUT. The block operates on tensors with the same layout and device.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        activation: Activation function name ('silu', 'gelu', 'relu').
        dropout: Dropout probability (0.0 means no dropout).
        dtype: TTNN data type for parameters.
        device: Optional TTNN device to place tensors on.
            If None, parameters are not moved to a device (use in host memory).

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If parameter initialization fails.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "silu",
        dropout: float = 0.0,
        dtype: ttnn.DataType = ttnn.bfloat16,
        device: Optional[ttnn.Device] = None,
    ) -> None:
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if activation not in LLVCConfig.VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {LLVCConfig.VALID_ACTIVATIONS}, got {activation}"
            )
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0], got {dropout}")

        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.dropout_p = dropout
        self.dtype = dtype
        self.device = device

        # Parameter tensors (initialised with random or constant values)
        try:
            self.weight: ttnn.Tensor = ttnn.randn(
                [out_features, in_features],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.bias: ttnn.Tensor = ttnn.randn(
                [out_features],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.ln_weight: ttnn.Tensor = ttnn.ones(
                [out_features],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.ln_bias: ttnn.Tensor = ttnn.zeros(
                [out_features],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        except Exception as exc:
            logger.error("Failed to initialise FeedForwardBlock parameters: %s", exc)
            raise RuntimeError("Parameter initialisation failed") from exc

        logger.debug(
            "FeedForwardBlock created: in=%d out=%d activation=%s dropout=%.2f",
            in_features,
            out_features,
            activation,
            dropout,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply the block: Linear → LayerNorm → Activation → Dropout.

        Args:
            x: Input TTNN tensor of shape ``(batch, seq_len, in_features)``
                in ``TILE_LAYOUT`` on the same device as the block parameters.

        Returns:
            Output TTNN tensor of shape ``(batch, seq_len, out_features)``.

        Raises:
            ValueError: If the input tensor does not have the expected
                number of features.
            RuntimeError: If any TTNN operation fails.
        """
        # Validate input shape
        if not hasattr(x, 'shape') or len(x.shape) != 3:
            raise ValueError(
                f"Expected input tensor with 3 dimensions (batch, seq, features), "
                f"got shape {x.shape}"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input last dimension {x.shape[-1]} does not match expected "
                f"in_features {self.in_features}"
            )

        try:
            # Linear transformation
            x = ttnn.linear(x, self.weight, bias=self.bias)

            # Layer normalisation
            x = ttnn.layer_norm(x, weight=self.ln_weight, bias=self.ln_bias)

            # Activation
            if self.activation_name == "silu":
                x = ttnn.silu(x)
            elif self.activation_name == "gelu":
                x = ttnn.gelu(x)
            elif self.activation_name == "relu":
                x = ttnn.relu(x)
            else:
                # Should never happen due to validation in __init__
                raise ValueError(f"Unsupported activation: {self.activation_name}")

            # Dropout (if enabled)
            if self.dropout_p > 0.0:
                x = ttnn.dropout(x, p=self.dropout_p)

        except Exception as exc:
            logger.error("FeedForwardBlock forward failed: %s", exc)
            raise RuntimeError("Forward pass failed") from exc

        return x

    def to_device(self, device: ttnn.Device) -> None:
        """Move all parameters to a new device.

        The parameters are re‑initialised on the target device. If any
        transfer fails, the block is left in an undefined state.

        Args:
            device: Target TTNN device.

        Raises:
            RuntimeError: If parameter transfer fails.
        """
        try:
            self.weight = ttnn.to_device(self.weight, device)
            self.bias = ttnn.to_device(self.bias, device)
            self.ln_weight = ttnn.to_device(self.ln_weight, device)
            self.ln_bias = ttnn.to_device(self.ln_bias, device)
            self.device = device
        except Exception as exc:
            logger.error("Failed to move FeedForwardBlock to device: %s", exc)
            raise RuntimeError("Device transfer failed") from exc

    def state_dict(self) -> Dict[str, ttnn.Tensor]:
        """Return a dictionary of all parameters (as references)."""
        return {
            "weight": self.weight,
            "bias": self.bias,
            "ln_weight": self.ln_weight,
            "ln_bias": self.ln_bias,
        }

    def load_state_dict(self, state_dict: Dict[str, ttnn.Tensor]) -> None:
        """Load parameters from a state dictionary.

        The tensors must be compatible with the existing shape and dtype.
        They are directly assigned (no copy guarantee).

        Args:
            state_dict: Dictionary containing parameter tensors.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If tensor shapes do not match.
        """
        required_keys = {"weight", "bias", "ln_weight", "ln_bias"}
        if not required_keys.issubset(state_dict.keys()):
            missing = required_keys - state_dict.keys()
            raise KeyError(f"Missing keys in state_dict: {missing}")

        # Validate shapes
        for key, expected_shape in [
            ("weight", (self.out_features, self.in_features)),
            ("bias", (self.out_features,)),
            ("ln_weight", (self.out_features,)),
            ("ln_bias", (self.out_features,)),
        ]:
            actual_shape = state_dict[key].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': expected {expected_shape}, "
                    f"got {actual_shape}"
                )

        self.weight = state_dict["weight"]
        self.bias = state_dict["bias"]
        self.ln_weight = state_dict["ln_weight"]
        self.ln_bias = state_dict["ln_bias"]

        logger.debug(
            "FeedForwardBlock state loaded (in=%d, out=%d)",
            self.in_features,
            self.out_features,
        )


class LLVC:
    """Low-Latency Low-Resource Voice Conversion model.

    The model consists of an encoder (stack of num_layers feed‑forward blocks),
    a bottleneck linear projection, and a decoder (stack of num_layers
    feed‑forward blocks). All parameters are stored as TTNN tensors.

    Args:
        config: Model configuration.
        dtype: TTNN data type for all parameters.
        device: Optional TTNN device to place parameters on.

    Raises:
        ValueError: If the configuration is invalid.
        RuntimeError: If parameter initialisation fails.
    """

    def __init__(
        self,
        config: LLVCConfig,
        dtype: ttnn.DataType = ttnn.bfloat16,
        device: Optional[ttnn.Device] = None,
    ) -> None:
        if not isinstance(config, LLVCConfig):
            raise TypeError("config must be an LLVCConfig instance")
        if dtype is None:
            dtype = ttnn.bfloat16

        self.config = config
        self.dtype = dtype
        self.device = device

        try:
            # ===== Encoder =====
            self.encoder: List[FeedForwardBlock] = []
            # First layer: input -> hidden
            self.encoder.append(
                FeedForwardBlock(
                    config.input_channels,
                    config.hidden_channels,
                    activation=config.activation,
                    dropout=config.dropout,
                    dtype=dtype,
                    device=device,
                )
            )
            # Intermediate layers: hidden -> hidden
            for _ in range(config.num_layers - 1):
                self.encoder.append(
                    FeedForwardBlock(
                        config.hidden_channels,
                        config.hidden_channels,
                        activation=config.activation,
                        dropout=config.dropout,
                        dtype=dtype,
                        device=device,
                    )
                )

            # ===== Bottleneck: hidden -> latent -> hidden =====
            self.bottleneck_down: ttnn.Tensor = ttnn.randn(
                [config.hidden_channels, config.latent_dim],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.bottleneck_bias_down: ttnn.Tensor = ttnn.randn(
                [config.latent_dim],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.bottleneck_up: ttnn.Tensor = ttnn.randn(
                [config.latent_dim, config.hidden_channels],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.bottleneck_bias_up: ttnn.Tensor = ttnn.randn(
                [config.hidden_channels],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

            # ===== Decoder =====
            self.decoder: List[FeedForwardBlock] = []
            # First layer: hidden -> hidden
            for _ in range(config.num_layers - 1):
                self.decoder.append(
                    FeedForwardBlock(
                        config.hidden_channels,
                        config.hidden_channels,
                        activation=config.activation,
                        dropout=config.dropout,
                        dtype=dtype,
                        device=device,
                    )
                )
            # Last layer: hidden -> output
            self.decoder.append(
                FeedForwardBlock(
                    config.hidden_channels,
                    config.output_channels,
                    activation=config.activation,
                    dropout=config.dropout,
                    dtype=dtype,
                    device=device,
                )
            )
        except Exception as exc:
            logger.error("Failed to initialise LLVC model: %s", exc)
            raise RuntimeError("Model initialisation failed") from exc

        logger.info(
            "LLVC model created: config=%s, dtype=%s, device=%s",
            config,
            dtype,
            device,
        )

    def encode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Encode input mel spectrogram into latent representation.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_channels)``.

        Returns:
            Latent tensor of shape ``(batch, seq_len, latent_dim)``.

        Raises:
            ValueError: If input shape does not match configuration.
            RuntimeError: If any computation fails.
        """
        if not hasattr(x, 'shape') or len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq, features), got shape {x.shape}"
            )
        if x.shape[-1] != self.config.input_channels:
            raise ValueError(
                f"Input last dimension {x.shape[-1]} does not match "
                f"config.input_channels {self.config.input_channels}"
            )

        try:
            # Encoder
            h = x
            for block in self.encoder:
                h = block.forward(h)

            # Bottleneck (project down)
            h = ttnn.linear(h, self.bottleneck_down, bias=self.bottleneck_bias_down)
            h = ttnn.silu(h)  # Activation in bottleneck (common choice)
        except Exception as exc:
            logger.error("Encode failed: %s", exc)
            raise RuntimeError("Encode failed") from exc

        return h

    def decode(self, z: ttnn.Tensor) -> ttnn.Tensor:
        """Decode latent representation into mel spectrogram.

        Args:
            z: Latent tensor of shape ``(batch, seq_len, latent_dim)``.

        Returns:
            Output mel tensor of shape ``(batch, seq_len, output_channels)``.

        Raises:
            ValueError: If input shape does not match configuration.
            RuntimeError: If any computation fails.
        """
        if not hasattr(z, 'shape') or len(z.shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq, features), got shape {z.shape}"
            )
        if z.shape[-1] != self.config.latent_dim:
            raise ValueError(
                f"Input last dimension {z.shape[-1]} does not match "
                f"config.latent_dim {self.config.latent_dim}"
            )

        try:
            # Bottleneck (project up)
            h = ttnn.linear(z, self.bottleneck_up, bias=self.bottleneck_bias_up)
            h = ttnn.silu(h)

            # Decoder
            for block in self.decoder:
                h = block.forward(h)
        except Exception as exc:
            logger.error("Decode failed: %s", exc)
            raise RuntimeError("Decode failed") from exc

        return h

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Full forward pass: encode → decode.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_channels)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, output_channels)``.
        """
        return self.decode(self.encode(x))

    def to_device(self, device: ttnn.Device) -> None:
        """Move all parameters to the specified device.

        Args:
            device: Target TTNN device.

        Raises:
            RuntimeError: If any transfer fails.
        """
        try:
            # Encoder blocks
            for block in self.encoder:
                block.to_device(device)
            # Bottleneck weights
            self.bottleneck_down = ttnn.to_device(self.bottleneck_down, device)
            self.bottleneck_bias_down = ttnn.to_device(self.bottleneck_bias_down, device)
            self.bottleneck_up = ttnn.to_device(self.bottleneck_up, device)
            self.bottleneck_bias_up = ttnn.to_device(self.bottleneck_bias_up, device)
            # Decoder blocks
            for block in self.decoder:
                block.to_device(device)

            self.device = device
        except Exception as exc:
            logger.error("Failed to move LLVC to device: %s", exc)
            raise RuntimeError("Device transfer failed") from exc

    def state_dict(self) -> Dict[str, Any]:
        """Return a dictionary of all model parameters (as TTNN tensors).

        Returns:
            A flat dictionary with unique keys for each parameter.
        """
        sd: Dict[str, Any] = {}
        for i, block in enumerate(self.encoder):
            prefix = f"encoder.{i}."
            for k, v in block.state_dict().items():
                sd[prefix + k] = v
        sd["bottleneck_down"] = self.bottleneck_down
        sd["bottleneck_bias_down"] = self.bottleneck_bias_down
        sd["bottleneck_up"] = self.bottleneck_up
        sd["bottleneck_bias_up"] = self.bottleneck_bias_up
        for i, block in enumerate(self.decoder):
            prefix = f"decoder.{i}."
            for k, v in block.state_dict().items():
                sd[prefix + k] = v
        return sd

    def load_state_dict(self, state_dict: Dict[str, ttnn.Tensor]) -> None:
        """Load parameters from a state dictionary.

        The tensors are loaded by delegating to the individual submodules.
        The expected keys must match those produced by :meth:`state_dict`.

        Args:
            state_dict: Dictionary containing parameter tensors.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If tensor shapes do not match.
        """
        # Helper to extract parameters for a block given a prefix
        def _load_block(
            block: FeedForwardBlock,
            prefix: str,
            sd: Dict[str, ttnn.Tensor],
        ) -> None:
            block_sd = {
                k.replace(prefix, "", 1): v
                for k, v in sd.items()
                if k.startswith(prefix)
            }
            if block_sd:
                block.load_state_dict(block_sd)

        # Load encoder
        for i, block in enumerate(self.encoder):
            _load_block(block, f"encoder.{i}.", state_dict)

        # Load bottleneck
        self.bottleneck_down = state_dict["bottleneck_down"]
        self.bottleneck_bias_down = state_dict["bottleneck_bias_down"]
        self.bottleneck_up = state_dict["bottleneck_up"]
        self.bottleneck_bias_up = state_dict["bottleneck_bias_up"]

        # Load decoder
        for i, block in enumerate(self.decoder):
            _load_block(block, f"decoder.{i}.", state_dict)

        logger.info("LLVC model state loaded successfully")

    def save(self, path: Union[str, Path]) -> None:
        """Save model parameters to a file in TTNN dump format.

        The state dictionary is converted to a map of PyTorch tensors (via
        ``ttnn.to_torch``) and then pickled. This ensures portability across
        devices and host systems.

        Args:
            path: File path to save the checkpoint.

        Raises:
            RuntimeError: If serialisation fails.
        """
        path = Path(path)
        try:
            state = self.state_dict()
            # Convert TTNN tensors to PyTorch (or host tensors) for serialisation
            torch_state = {
                k: ttnn.to_torch(v) for k, v in state.items()
            }
            data = {
                "config": self.config,
                "dtype": self.dtype,
                "state_dict": torch_state,
            }
            with open(path, "wb") as f:
                torch.save(data, f)
            logger.info("Model saved to %s", path)
        except Exception as exc:
            logger.error("Failed to save model to %s: %s", path, exc)
            raise RuntimeError(f"Model save to {path} failed") from exc

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[ttnn.Device] = None) -> "LLVC":
        """Load a model from a checkpoint file.

        Args:
            path: File path of the saved checkpoint.
            device: Optional device to place parameters on.

        Returns:
            An initialised :class:`LLVC` instance with loaded parameters.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If deserialisation or model construction fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            with open(path, "rb") as f:
                data = torch.load(f, map_location="cpu")
            config: LLVCConfig = data["config"]
            dtype: ttnn.DataType = data["dtype"]
            torch_state_dict: Dict[str, torch.Tensor] = data["state_dict"]

            # Construct model on CPU first (device=None)
            model = cls(config=config, dtype=dtype, device=None)

            # Convert torch tensors back to TTNN tensors on the desired device
            ttnn_state_dict: Dict[str, ttnn.Tensor] = {}
            for k, v in torch_state_dict.items():
                ttnn_tensor = ttnn.from_torch(
                    v,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                ttnn_state_dict[k] = ttnn_tensor

            model.load_state_dict(ttnn_state_dict)

            # If device is specified, ensure model is on that device
            if device is not None:
                model.to_device(device)

            logger.info("Model loaded from %s", path)
            return model
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", path, exc)
            raise RuntimeError(f"Model load from {path} failed") from exc

    def __repr__(self) -> str:
        return (
            f"LLVC(config={self.config}, "
            f"dtype={self.dtype}, "
            f"device={self.device})"
        )