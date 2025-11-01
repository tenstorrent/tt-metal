"""TTNN implementation of multibox head layers for SSD."""

import ttnn
from typing import List, Dict, Tuple

from models.experimental.SSD512.tt.utils import TtConv2d


class TtMultiBoxHead:
    """TTNN implementation of SSD multibox head containing localization and classification layers."""

    def __init__(self, config: Dict, num_classes: int, state_dict=None, base_address="", device=None):
        """Initialize multibox head layers.

        Args:
            config: Model configuration
            num_classes: Number of object classes including background
            state_dict: Optional state dict for weight initialization
            base_address: Base prefix for state dict keys
            device: Device to place ops on
        """
        self.num_classes = num_classes
        self.device = device

        # Number of boxes per feature map cell
        # SSD512: [4, 6, 6, 6, 4, 4]
        self.anchor_box_specs = [4, 6, 6, 6, 4, 4]

        # Build loc and conf layers
        self.loc_layers = []
        self.conf_layers = []
        self._build_layers(config)

        if state_dict is not None:
            self._load_state_dict(state_dict, base_address)

    def _build_layers(self, config: Dict):
        """Construct localization and classification conv layers."""
        # Source layers channels (VGG conv4_3, conv7 and extra layer outputs)
        vgg_source_channels = [512, 1024]  # conv4_3, conv7
        extra_channels = [512, 256, 256, 256, 256]
        in_channels = vgg_source_channels + extra_channels

        # Create conv layers for each feature map
        for i, (channels, n_boxes) in enumerate(zip(in_channels, self.anchor_box_specs)):
            # Localization layers (predict 4 coords per box)
            self.loc_layers.append(
                TtConv2d(in_channels=channels, out_channels=n_boxes * 4, kernel_size=3, padding=1, device=self.device)
            )

            # Classification layers (predict class scores per box)
            self.conf_layers.append(
                TtConv2d(
                    in_channels=channels,
                    out_channels=n_boxes * self.num_classes,
                    kernel_size=3,
                    padding=1,
                    device=self.device,
                )
            )

    def _load_state_dict(self, state_dict: Dict, base_address: str):
        """Load weights from state dict.

        Args:
            state_dict: State dict containing weights
            base_address: Base prefix for state dict keys
        """
        # TODO: Implement weight loading

    def __call__(self, features: List[ttnn.Tensor]) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Forward pass through multibox head.

        Args:
            features: List of feature maps from backbone and extra layers

        Returns:
            tuple (loc_preds, conf_preds) where:
                loc_preds: Predicted box coordinates [batch_size, num_priors * 4]
                conf_preds: Predicted class scores [batch_size, num_priors * num_classes]
        """
        loc_preds = []
        conf_preds = []

        # Apply loc and conf layers to each feature map
        for i, (x, loc, conf) in enumerate(zip(features, self.loc_layers, self.conf_layers)):
            # Get predictions
            loc_out = loc(x)
            conf_out = conf(x)

            # Reshape predictions
            # From: [batch, channels, height, width]
            # To: [batch, height, width, channels]
            loc_out = ttnn.permute(loc_out, [0, 2, 3, 1])
            conf_out = ttnn.permute(conf_out, [0, 2, 3, 1])

            # Make continuous in memory and flatten
            batch_size = loc_out.padded_shape[0]
            loc_out = ttnn.reshape(loc_out, [batch_size, -1])
            conf_out = ttnn.reshape(conf_out, [batch_size, -1])

            loc_preds.append(loc_out)
            conf_preds.append(conf_out)

        # Concatenate predictions from all feature maps
        loc_preds = ttnn.cat(loc_preds, dim=1)
        conf_preds = ttnn.cat(conf_preds, dim=1)

        # Reshape loc preds to [batch, num_priors, 4]
        # and conf preds to [batch, num_priors, num_classes]
        loc_preds = ttnn.reshape(loc_preds, [batch_size, -1, 4])
        conf_preds = ttnn.reshape(conf_preds, [batch_size, -1, self.num_classes])

        return loc_preds, conf_preds
