"""TTNN implementation of extra layers for SSD."""

import ttnn
from typing import List, Dict

from models.experimental.SSD512.tt.utils import TtConv2d


class TtExtraLayers:
    """TTNN implementation of extra convolutional layers added to VGG for SSD."""

    def __init__(self, config: Dict, state_dict=None, base_address="", device=None):
        """Initialize extra layers.

        Args:
            config: Model configuration
            state_dict: Optional state dict for weight initialization
            base_address: Base prefix for state dict keys
            device: Device to place ops on
        """
        self.device = device
        self.layers = []

        # Build extra layers
        self._build_layers(config)

        if state_dict is not None:
            self._load_state_dict(state_dict, base_address)

    def _build_layers(self, config: Dict):
        """Construct extra convolutional layers."""
        # SSD512 extra layer config
        extra_cfg = [256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256]  # conv8  # conv9  # conv10  # conv11

        in_channels = 1024  # Output channels from VGG conv7

        # Create layers according to config
        for i, v in enumerate(extra_cfg):
            if v == "S":
                # Strided conv for downsampling
                self.layers.append(
                    TtConv2d(
                        in_channels=in_channels,
                        out_channels=extra_cfg[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        device=self.device,
                    )
                )
            else:
                if i == len(extra_cfg) - 1:
                    # Final layer has kernel_size=4
                    self.layers.append(
                        TtConv2d(in_channels=in_channels, out_channels=v, kernel_size=4, padding=1, device=self.device)
                    )
                else:
                    # Regular 1x1 or 3x3 conv
                    kernel_size = 1 if i % 2 == 0 else 3
                    padding = 0 if kernel_size == 1 else 1
                    self.layers.append(
                        TtConv2d(
                            in_channels=in_channels,
                            out_channels=v,
                            kernel_size=kernel_size,
                            padding=padding,
                            device=self.device,
                        )
                    )
                in_channels = v

    def _load_state_dict(self, state_dict: Dict, base_address: str):
        """Load weights from state dict.

        Args:
            state_dict: State dict containing weights
            base_address: Base prefix for state dict keys
        """
        # TODO: Implement weight loading

    def __call__(self, x: ttnn.Tensor) -> List[ttnn.Tensor]:
        """Forward pass through extra layers.

        Args:
            x: Input tensor from VGG conv7

        Returns:
            List of output feature maps from intermediate layers
        """
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = ttnn.relu(x)
            # Keep feature maps from every other layer
            if i % 2 == 1:
                features.append(x)

        return features
