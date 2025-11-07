# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Tuple

from models.experimental.efficientdetd0.tt.bifpn import TtBiFPN
from models.experimental.efficientdetd0.tt.classifier import Classifier as TtClassifier
from models.experimental.efficientdetd0.tt.regressor import Regressor as TtRegressor
from models.experimental.efficientdetd0.reference.utils import Anchors
from models.experimental.efficientnetb0.tt.efficientnetb0 import Efficientnetb0


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
        # Extract backbone parameters
        if isinstance(parameters, dict):
            backbone_params = parameters.get("backbone_net", parameters)
        elif hasattr(parameters, "backbone_net"):
            backbone_params = parameters.backbone_net
        else:
            backbone_params = parameters

        if isinstance(conv_params, dict):
            backbone_conv_params = conv_params.get("backbone_net", conv_params)
        elif hasattr(conv_params, "backbone_net"):
            backbone_conv_params = conv_params.backbone_net
        else:
            backbone_conv_params = conv_params

        #
        # if isinstance(parameters, dict):
        #     if "backbone_net" in parameters:
        #         backbone_params = parameters["backbone_net"]
        #         # The EfficientNet wrapper has a "model" attribute
        #         if isinstance(backbone_params, dict) and "model" in backbone_params:
        #             # Get the actual Efficientnetb0 model parameters
        #             backbone_params = backbone_params["model"]
        #     else:
        #         # Parameters are for backbone directly
        #         backbone_params = parameters
        # elif hasattr(parameters, "backbone_net"):
        #     backbone_params = parameters.backbone_net
        # elif hasattr(parameters, "get"):
        #     # Try dict-like access
        #     backbone_params = parameters.get("backbone_net", parameters)
        # else:
        #     # Assume parameters are for backbone directly
        #     backbone_params = parameters

        # # # Handle conv_params - similar logic
        # # if hasattr(conv_params, "backbone_net"):
        # #     backbone_conv_params = conv_params.backbone_net
        # # elif isinstance(conv_params, dict):
        # #     if "backbone_net" in conv_params:
        # #         backbone_conv_params = conv_params["backbone_net"]
        # #         if isinstance(backbone_conv_params, dict) and "model" in backbone_conv_params:
        # #             backbone_conv_params = backbone_conv_params["model"]
        # #     else:
        # #         backbone_conv_params = conv_params
        # # else:
        # #     # Assume conv_params are for backbone directly
        # #     backbone_conv_params = conv_params

        # if isinstance(conv_params, dict):
        #     if "backbone_net" in conv_params:
        #         backbone_conv_params = conv_params["backbone_net"]
        #         # The EfficientNet wrapper has a "model" attribute
        #         if isinstance(backbone_conv_params, dict) and "model" in backbone_conv_params:
        #             # Get the actual Efficientnetb0 model parameters
        #             backbone_conv_params = backbone_conv_params["model"]
        #     else:
        #         # Parameters are for backbone directly
        #         backbone_conv_params = conv_params
        # elif hasattr(conv_params, "backbone_net"):
        #     backbone_params = conv_params.backbone_net
        # elif hasattr(conv_params, "get"):
        #     # Try dict-like access
        #     backbone_params = conv_params.get("backbone_net", conv_params)
        # else:
        #     # Assume parameters are for backbone directly
        #     backbone_params = conv_params

        # print(".........................................................")
        # print(f"backbone_params: {backbone_params=}")
        # print(".........................................................")
        # print(f"backbone_conv_params: {backbone_conv_params=}")
        # print(".........................................................")
        self.backbone_net = Efficientnetb0(
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

        # Initialize Anchors (CPU-based, no TTNN implementation needed)
        self.anchors = Anchors(
            anchor_scale=self.anchor_scale[compound_coef],
            pyramid_levels=(torch.arange(self.pyramid_levels[compound_coef]) + 3).tolist(),
            **kwargs,
        )

    def _extract_backbone_features(self, x: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """ """
        backbone = self.backbone_net

        # Input preprocessing (same as EfficientNet)
        N, C, H, W = x.shape
        min_channels = 16
        if C < min_channels:
            channel_padding_needed = min_channels - C
            nchw = ttnn.pad(x, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        else:
            nchw = x
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))
        if C < min_channels:
            ttnn.deallocate(nchw)
        ttnn.deallocate(x)
        nhwc = ttnn.reallocate(nhwc)
        x = ttnn.reshape(nhwc, [1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[-1]])

        # Stem
        x = backbone._conv_stem(x)
        x = ttnn.swish(x)

        # Block 0
        x = backbone._blocks0(x)

        # Blocks 1-2 (with residual connection)
        x_1 = backbone._blocks1(x)
        x = backbone._blocks2(x_1)
        x = ttnn.add(x, x_1)
        ttnn.deallocate(x_1)

        # Block 3 - Extract P3 (40 channels, stride 2)
        x_3 = backbone._blocks3(x)
        p3 = x_3  # P3: 40 channels from block 3 - keep reference

        # Block 4 (with residual connection)
        x = backbone._blocks4(x_3)
        x = ttnn.add(x, x_3)
        # Note: x_3 is still referenced as p3, so don't deallocate

        # Block 5
        x_5 = backbone._blocks5(x)

        # Blocks 6-7 (with residual connections)
        x = backbone._blocks6(x_5)
        x_7_in = ttnn.add(x, x_5)
        ttnn.deallocate(x_5)
        x = backbone._blocks7(x_7_in)
        x = ttnn.add(x, x_7_in)
        ttnn.deallocate(x_7_in)

        # Block 8 - Extract P4 (112 channels)
        x_8 = backbone._blocks8(x)
        p4 = x_8  # P4: 112 channels from block 8 - keep reference

        # Blocks 9-10 (with residual connections)
        x = backbone._blocks9(x_8)
        x_10_in = ttnn.add(x, x_8)
        # Note: x_8 is still referenced as p4, so don't deallocate
        x = backbone._blocks10(x_10_in)
        x = ttnn.add(x, x_10_in)
        ttnn.deallocate(x_10_in)

        # Block 11
        x_11 = backbone._blocks11(x)

        # Blocks 12-14 (with residual connections)
        x = backbone._blocks12(x_11)
        x_13_in = ttnn.add(x, x_11)
        ttnn.deallocate(x_11)
        x = backbone._blocks13(x_13_in)
        x_14_in = ttnn.add(x, x_13_in)
        ttnn.deallocate(x_13_in)
        ttnn.deallocate(x)
        x = backbone._blocks14(x_14_in)
        x = ttnn.add(x, x_14_in)
        ttnn.deallocate(x_14_in)

        # Block 15 - Extract P5 (320 channels, final)
        x = backbone._blocks15(x)
        p5 = x  # P5: 320 channels from block 15

        return p3, p4, p5

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
        p3, p4, p5 = self._extract_backbone_features(inputs)

        # Convert to tuple for BiFPN
        features = (p3, p4, p5)

        # Process through BiFPN layers
        for bifpn in self.bifpn_layers:
            features = bifpn(features)

        # Generate regression predictions
        regression = self.regressor(features)

        # Generate classification predictions
        classification = self.classifier(features)

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
