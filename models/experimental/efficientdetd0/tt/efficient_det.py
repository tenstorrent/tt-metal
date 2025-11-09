# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Tuple

from models.experimental.efficientdetd0.tt.efficient_netb0 import Efficientnetb0 as TtEfficientnetb0
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

        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales
        backbone_params = parameters.backbone_net.model
        backbone_conv_params = conv_params.backbone_net.model

        self.backbone_net = TtEfficientnetb0(
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
                deallocate_activation=False,
            )
            self.bifpn_layers.append(bifpn)
        # Convert conv_params.regressor structure to match parameters.regressor structure
        regressor_conv_params = (
            conv_params.regressor if hasattr(conv_params, "regressor") else conv_params.get("regressor", {})
        )

        # Check if conversion is needed (if structure is already converted, it will have 'conv_list' key at top level)
        if regressor_conv_params and "conv_list" not in regressor_conv_params:
            converted_conv_params = {"conv_list": {}, "header": {}}

            # Convert structure: extract conv_list and header from each pyramid level
            # Input: {0: {'conv_list': {0: {...}, 1: {...}, 2: {...}}, 'header': {...}}, ...}
            # Output: {'conv_list': {0: {0: {...}, 1: {...}, 2: {...}}, 1: {0: {...}, 1: {...}, 2: {...}}, ...}, 'header': {0: {...}, 1: {...}, ...}}
            for pyramid_level in sorted(regressor_conv_params.keys()):
                pyramid_level_int = int(pyramid_level) if isinstance(pyramid_level, (str, int)) else pyramid_level

                if "conv_list" in regressor_conv_params[pyramid_level]:
                    converted_conv_params["conv_list"][pyramid_level_int] = regressor_conv_params[pyramid_level][
                        "conv_list"
                    ]

                if "header" in regressor_conv_params[pyramid_level]:
                    converted_conv_params["header"][pyramid_level_int] = regressor_conv_params[pyramid_level]["header"]

            regressor_conv_params = converted_conv_params

        # Convert dictionary to namespace object to support dot notation access
        regressor_conv_params = dict_to_namespace(regressor_conv_params)

        # Initialize Regressor
        self.regressor = TtRegressor(
            device=device,
            parameters=parameters.regressor if hasattr(parameters, "regressor") else parameters.get("regressor", {}),
            conv_params=regressor_conv_params,
            num_layers=self.box_class_repeats[compound_coef],
            pyramid_levels=self.pyramid_levels[compound_coef],
        )

        # Convert conv_params.classifier structure to match parameters.classifier structure
        # Same conversion as regressor: From {pyramid_level: {'conv_list': {layer_idx: {...}}, 'header': {...}}}
        # To: {'conv_list': {pyramid_level: {layer_idx: {...}}}, 'header': {pyramid_level: {...}}}
        classifier_conv_params = (
            conv_params.classifier if hasattr(conv_params, "classifier") else conv_params.get("classifier", {})
        )

        # Check if conversion is needed (if structure is already converted, it will have 'conv_list' key at top level)
        if classifier_conv_params and "conv_list" not in classifier_conv_params:
            converted_conv_params = {"conv_list": {}, "header": {}}

            # Convert structure: extract conv_list and header from each pyramid level
            # Input: {0: {'conv_list': {0: {...}, 1: {...}, 2: {...}}, 'header': {...}}, ...}
            # Output: {'conv_list': {0: {0: {...}, 1: {...}, 2: {...}}, 1: {0: {...}, 1: {...}, 2: {...}}, ...}, 'header': {0: {...}, 1: {...}, ...}}
            for pyramid_level in sorted(classifier_conv_params.keys()):
                pyramid_level_int = int(pyramid_level) if isinstance(pyramid_level, (str, int)) else pyramid_level

                if "conv_list" in classifier_conv_params[pyramid_level]:
                    converted_conv_params["conv_list"][pyramid_level_int] = classifier_conv_params[pyramid_level][
                        "conv_list"
                    ]

                if "header" in classifier_conv_params[pyramid_level]:
                    converted_conv_params["header"][pyramid_level_int] = classifier_conv_params[pyramid_level]["header"]

            classifier_conv_params = converted_conv_params

        # Convert dictionary to namespace object to support dot notation access
        classifier_conv_params = dict_to_namespace(classifier_conv_params)

        # Initialize Classifier
        self.classifier = TtClassifier(
            device=device,
            parameters=parameters.classifier if hasattr(parameters, "classifier") else parameters.get("classifier", {}),
            conv_params=classifier_conv_params,
            num_classes=num_classes,
            num_layers=self.box_class_repeats[compound_coef],
            pyramid_levels=self.pyramid_levels[compound_coef],
        )

        # # Initialize Anchors (CPU-based, no TTNN implementation needed)
        # self.anchors = Anchors(
        #     anchor_scale=self.anchor_scale[compound_coef],
        #     pyramid_levels=(torch.arange(self.pyramid_levels[compound_coef]) + 3).tolist(),
        #     **kwargs,
        # )

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
        features_end = features

        # Reshape features from [1, 1, H*W, C] to [1, C, H, W] for regressor
        spatial_dims = [
            (64, 64),  # P3
            (32, 32),  # P4
            (16, 16),  # P5
            (8, 8),  # P6
            (4, 4),  # P7
        ]

        reshaped_features = []
        for i, feat in enumerate(features):
            h, w = spatial_dims[i]
            c = feat.shape[-1]

            # Convert to interleaved memory if sharded
            if feat.is_sharded():
                feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

            # Reshape from [1, 1, H*W, C] to [1, H, W, C]
            feat = ttnn.reshape(feat, (1, h, w, c))

            # Permute from NHWC to NCHW
            feat = ttnn.permute(feat, (0, 3, 1, 2))

            reshaped_features.append(feat)

        features = tuple(reshaped_features)

        # Generate regression predictions
        regression = self.regressor(features)

        # Generate classification predictions
        classification = self.classifier(features)
        return features_end, regression, classification

        # Generate anchors (CPU-based)
        # Convert input to torch for anchor generation
        if isinstance(inputs, ttnn.Tensor):
            # Get input shape for anchor generation
            # Note: Anchors are generated on CPU, so we need the input shape
            # This is a simplified approach - in practice, you might want to pass image shape separately
            input_torch = ttnn.to_torch(inputs)
            # Convert from NHWC to NCHW if needed
            if len(input_torch.shape) == 4:
                # Assuming NHWC format
                input_torch = input_torch.permute(0, 3, 1, 2)
        else:
            input_torch = inputs

        anchors = self.anchors(input_torch, input_torch.dtype)

        return features, regression, classification, anchors
