# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Tuple

from models.experimental.efficientdetd0.tt.efficientnetb0 import TtnnEfficientNet
from models.experimental.efficientdetd0.tt.bifpn import TtBiFPN
from models.experimental.efficientdetd0.tt.regressor import Regressor as TtRegressor
from models.experimental.efficientdetd0.tt.classifier import Classifier as TtClassifier


def dict_to_namespace(d):
    """
    Recursively convert a dictionary to an object that supports dot notation access.
    Handles both string and integer keys.
    """
    if isinstance(d, dict):
        # Create a SimpleNamespace-like object that supports both attribute and integer key access
        class DictNamespace:
            def __init__(self, data):
                self._dict = data
                # Set string keys as attributes
                for k, v in data.items():
                    if isinstance(k, str) and k.isidentifier():
                        setattr(self, k, dict_to_namespace(v) if isinstance(v, dict) else v)

            def __getitem__(self, key):
                """Support dictionary-style access with integer keys"""
                if key in self._dict:
                    value = self._dict[key]
                    return dict_to_namespace(value) if isinstance(value, dict) else value
                raise KeyError(f"Key '{key}' not found")

            def __getattr__(self, name):
                """Support attribute access for string keys"""
                if name in self._dict:
                    value = self._dict[name]
                    return dict_to_namespace(value) if isinstance(value, dict) else value
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return DictNamespace(d)
    elif isinstance(d, list):
        # Convert lists of dictionaries
        return [dict_to_namespace(item) if isinstance(item, dict) else item for item in d]
    else:
        return d


class TtEfficientDetBackbone:
    """
    TTNN implementation of EfficientDet backbone.
    Based on the reference implementation in models.experimental.efficientdetd0.reference.efficientdet
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
        """
        Args:
            device: TTNN device
            parameters: Model parameters containing weights for all components
            conv_params: Configuration parameters for all operations
            num_classes: Number of classes for classification
            compound_coef: Compound coefficient for EfficientDet variant (0-8)
            **kwargs: Additional arguments (ratios, scales, etc.)
        """
        self.device = device
        self.compound_coef = compound_coef
        self.num_classes = num_classes

        # Configuration parameters (matching reference)
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 4.0]
        self.aspect_ratios = kwargs.get("ratios", [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get("scales", [2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))

        backbone_params = parameters.backbone_net.model
        backbone_conv_params = conv_params.backbone_net.model

        self.backbone_net = TtnnEfficientNet(
            device=device,
            parameters=backbone_params,
            conv_params=backbone_conv_params,
            batch=1,
        )

        # Initialize BiFPN layers
        # Create multiple BiFPN layers based on fpn_cell_repeats
        self.bifpn_layers = []
        for i in range(self.fpn_cell_repeats[compound_coef]):
            is_first_time = i == 0
            bifpn = TtBiFPN(
                device=device,
                parameters=parameters.bifpn[i] if hasattr(parameters, "bifpn") else parameters.get("bifpn", {})[i],
                conv_params=conv_params.bifpn[i] if hasattr(conv_params, "bifpn") else conv_params.get("bifpn", {})[i],
                num_channels=self.fpn_num_filters[compound_coef],
                first_time=is_first_time,
                epsilon=1e-4,
                attention=True if compound_coef < 6 else False,
                use_p8=compound_coef > 7,
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                deallocate_activation=True,
            )
            self.bifpn_layers.append(bifpn)

        # Initialize Regressor
        self.regressor = TtRegressor(
            device=device,
            parameters=parameters.regressor if hasattr(parameters, "regressor") else parameters.get("regressor", {}),
            conv_params=conv_params.regressor
            if hasattr(conv_params, "regressor")
            else conv_params.get("regressor", {}),
            num_layers=self.box_class_repeats[compound_coef],
            pyramid_levels=self.pyramid_levels[compound_coef],
        )

        # Initialize Classifier
        self.classifier = TtClassifier(
            device=device,
            parameters=parameters.classifier if hasattr(parameters, "classifier") else parameters.get("classifier", {}),
            conv_params=conv_params.classifier
            if hasattr(conv_params, "classifier")
            else conv_params.get("classifier", {}),
            num_classes=num_classes,
            num_layers=self.box_class_repeats[compound_coef],
            pyramid_levels=self.pyramid_levels[compound_coef],
        )

    def __call__(self, inputs: ttnn.Tensor) -> Tuple[Tuple, ttnn.Tensor, ttnn.Tensor, torch.Tensor]:
        """
        Forward pass of EfficientDet.

        Args:
            inputs: Input tensor in TTNN format (NHWC layout)

        Returns:
            features: Tuple of BiFPN feature maps (P3, P4, P5, P6, P7)
            regression: Regression predictions (bounding box coordinates)
            classification: Classification predictions (class probabilities)
            anchors: Anchor boxes (CPU tensor)
        """
        # Extract backbone features
        p3, p4, p5 = self.backbone_net(inputs)

        # Convert to tuple for BiFPN
        features = (p3, p4, p5)

        # Process through BiFPN layers
        for bifpn in self.bifpn_layers:
            features = bifpn(features)

        features = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in features]
        regressor_features = [ttnn.clone(t) for t in features]
        classifier_features = [ttnn.clone(t) for t in regressor_features]

        # Generate classification predictions
        classification = self.classifier(classifier_features)
        # Generate regression predictions
        regression = self.regressor(regressor_features)

        # Deallocate feature tensors
        for t1, t2 in zip(regressor_features, classifier_features):
            ttnn.deallocate(t1)
            ttnn.deallocate(t2)

        return features, regression, classification
