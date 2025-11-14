# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of EfficientDet D0 backbone.

This module provides the TTNN-accelerated EfficientDet D0 model for object detection.
"""

import ttnn
from typing import Tuple, List

from models.experimental.efficientdetd0.tt.efficientnetb0 import TtEfficientNet
from models.experimental.efficientdetd0.tt.bifpn import TtBiFPN
from models.experimental.efficientdetd0.tt.regressor import TtRegressor
from models.experimental.efficientdetd0.tt.classifier import TtClassifier


# EfficientDet D0 configuration constants
FPN_NUM_FILTERS = 64
FPN_CELL_REPEATS = 3
BOX_CLASS_REPEATS = 3
PYRAMID_LEVELS = 5

# Default anchor configuration
DEFAULT_ASPECT_RATIOS = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
DEFAULT_SCALES = [2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

# BiFPN configuration constants for D0
BIFPN_EPSILON = 1e-4
USE_ATTENTION = True  # D0 uses attention
USE_P8 = False  # D0 does not use P8


def _get_param(obj, key: str, default=None):
    """
    Helper function to safely get parameters from object or dict.

    Args:
        obj: Object with attributes or dictionary
        key: Key/attribute name to retrieve
        default: Default value if key not found

    Returns:
        Parameter value or default
    """
    if hasattr(obj, key):
        return getattr(obj, key)
    elif isinstance(obj, dict):
        return obj.get(key, default)
    return default


class TtEfficientDetBackbone:
    """
    TTNN implementation of EfficientDet D0 backbone for object detection.

    This class implements the EfficientDet D0 architecture optimized for Tenstorrent hardware,
    consisting of:
    - EfficientNet-B0 backbone for feature extraction
    - BiFPN (Bidirectional Feature Pyramid Network) for multi-scale feature fusion
    - Detection head with regressor and classifier branches

    Args:
        device: TTNN device instance
        parameters: Model parameters containing weights for all components
        conv_params: Configuration parameters for all operations
        num_classes: Number of object classes (default: 80)
        **kwargs: Additional arguments (ratios, scales, etc.)
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters,
        conv_params,
        num_classes: int = 80,
        compound_coef: int = 0,
        **kwargs,
    ):
        self.device = device
        self.num_classes = num_classes

        # Anchor configuration
        self.aspect_ratios = kwargs.get("ratios", DEFAULT_ASPECT_RATIOS)
        self.num_scales = len(kwargs.get("scales", DEFAULT_SCALES))

        # Initialize backbone network
        backbone_params = parameters.backbone_net.model
        backbone_conv_params = conv_params.backbone_net.model

        self.backbone_net = TtEfficientNet(
            device=device,
            parameters=backbone_params,
            conv_params=backbone_conv_params,
            batch=1,
        )

        # Initialize BiFPN layers
        self.bifpn_layers = self._create_bifpn_layers(
            device=device,
            parameters=parameters,
            conv_params=conv_params,
        )

        # Initialize detection head components
        self.regressor = self._create_regressor(
            device=device,
            parameters=parameters,
            conv_params=conv_params,
        )

        self.classifier = self._create_classifier(
            device=device,
            parameters=parameters,
            conv_params=conv_params,
        )

    def _create_bifpn_layers(
        self,
        device: ttnn.Device,
        parameters,
        conv_params,
    ) -> List[TtBiFPN]:
        """
        Create BiFPN layers for EfficientDet D0.

        Args:
            device: TTNN device
            parameters: Model parameters
            conv_params: Convolution parameters

        Returns:
            List of BiFPN layer instances
        """
        bifpn_layers = []
        bifpn_params = _get_param(parameters, "bifpn", {})
        bifpn_conv_params = _get_param(conv_params, "bifpn", {})

        for i in range(FPN_CELL_REPEATS):
            is_first_time = i == 0

            bifpn = TtBiFPN(
                device=device,
                parameters=bifpn_params[i] if isinstance(bifpn_params, (list, tuple)) else bifpn_params.get(i, {}),
                conv_params=bifpn_conv_params[i]
                if isinstance(bifpn_conv_params, (list, tuple))
                else bifpn_conv_params.get(i, {}),
                num_channels=FPN_NUM_FILTERS,
                first_time=is_first_time,
                epsilon=BIFPN_EPSILON,
                attention=USE_ATTENTION,
                use_p8=USE_P8,
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                deallocate_activation=True,
            )
            bifpn_layers.append(bifpn)

        return bifpn_layers

    def _create_regressor(
        self,
        device: ttnn.Device,
        parameters,
        conv_params,
    ) -> TtRegressor:
        """
        Create regressor component for bounding box prediction.

        Args:
            device: TTNN device
            parameters: Model parameters
            conv_params: Convolution parameters

        Returns:
            Regressor instance
        """
        regressor_params = _get_param(parameters, "regressor", {})
        regressor_conv_params = _get_param(conv_params, "regressor", {})

        return TtRegressor(
            device=device,
            parameters=regressor_params,
            conv_params=regressor_conv_params,
            num_layers=BOX_CLASS_REPEATS,
            pyramid_levels=PYRAMID_LEVELS,
        )

    def _create_classifier(
        self,
        device: ttnn.Device,
        parameters,
        conv_params,
    ) -> TtClassifier:
        """
        Create classifier component for class prediction.

        Args:
            device: TTNN device
            parameters: Model parameters
            conv_params: Convolution parameters

        Returns:
            Classifier instance
        """
        classifier_params = _get_param(parameters, "classifier", {})
        classifier_conv_params = _get_param(conv_params, "classifier", {})

        return TtClassifier(
            device=device,
            parameters=classifier_params,
            conv_params=classifier_conv_params,
            num_classes=self.num_classes,
            num_layers=BOX_CLASS_REPEATS,
            pyramid_levels=PYRAMID_LEVELS,
        )

    def __call__(self, inputs: ttnn.Tensor) -> Tuple[Tuple[ttnn.Tensor, ...], ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass of EfficientDet D0.

        Args:
            inputs: Input tensor in TTNN format (NHWC layout), shape [batch, height, width, channels]

        Returns:
            Tuple containing:
                - features: Tuple of BiFPN feature maps (P3, P4, P5, P6, P7)
                - regression: Regression predictions for bounding box coordinates
                - classification: Classification predictions for object classes
        """
        # Extract multi-scale features from backbone
        p3, p4, p5 = self.backbone_net(inputs)

        # Process through BiFPN layers for feature fusion
        features = (p3, p4, p5)
        for bifpn in self.bifpn_layers:
            features = bifpn(features)

        # Move features to DRAM and prepare for detection head
        features = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in features]

        # Clone features for regressor and classifier (they process independently)
        regressor_features = [ttnn.clone(t) for t in features]
        classifier_features = [ttnn.clone(t) for t in regressor_features]

        # Generate predictions
        classification = self.classifier(classifier_features)
        regression = self.regressor(regressor_features)

        # Cleanup intermediate feature tensors
        for reg_feat, cls_feat in zip(regressor_features, classifier_features):
            ttnn.deallocate(reg_feat)
            ttnn.deallocate(cls_feat)

        return tuple(features), regression, classification
