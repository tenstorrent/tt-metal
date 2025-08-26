import torch
from typing import Dict, List, Union, Optional, Tuple
from torch import nn
import ttnn

# We will reuse the TtASPP and helper functions you provided
from .tt_aspp import TtASPP, get_ttnn_norm, get_ttnn_activation

# We will also reuse the TtConv2d wrapper
from .tt_conv2dWrapper import TtConv2d, TtConv2dParameters
from .tt_pytorch_semSeg import ShapeSpec


class TtDeepLabV3PlusHead(nn.Module):
    """
    TTNN implementation of the DeepLabV3+ segmentation head.
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        device: ttnn.Device,
        *,
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        norm: str,
        train_size: Optional[Tuple],
        # --- All weights must be provided externally ---
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        shared_fuse_conv_0_weight: torch.Tensor,
        shared_fuse_conv_1_weight: torch.Tensor,
        res3_project_conv_weight: torch.Tensor,
        res2_project_conv_weight: torch.Tensor,
        predictor_weight: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        # Sort features by stride to ensure correct processing order
        sorted_input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in sorted_input_shape]
        in_channels = [v.channels for k, v in sorted_input_shape]
        in_strides = [v.stride for k, v in sorted_input_shape]
        aspp_channels = decoder_channels[-1]

        self.common_stride = common_stride
        self.decoder_only = num_classes is None
        self.device = device
        self.activation = get_ttnn_activation("relu")

        # A helper function to create TtConv2d modules, inspired by your TtASPP
        def _create_tt_conv2d(
            weight: torch.Tensor, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int, use_bias: bool
        ):
            param_dict = {"weight": weight}
            if use_bias:
                # We assume the bias will be fused or handled separately in TTNN.
                # Here, we can create a dummy bias if the wrapper requires it.
                param_dict["bias"] = torch.zeros(1, 1, 1, out_ch)

            parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)
            return TtConv2d(parameters, stride=(stride, stride), padding=(padding, padding))

        self.decoder = {}
        use_bias = norm == ""

        for idx, in_channel in enumerate(in_channels):
            decoder_stage = {}
            feature_name = self.in_features[idx]

            if idx == len(self.in_features) - 1:  # ASPP stage
                if train_size is not None:
                    train_h, train_w = train_size
                    encoder_stride = in_strides[-1]
                    pool_h, pool_w = train_h // encoder_stride, train_w // encoder_stride
                    pool_kernel_size = (pool_h, pool_w)
                else:
                    pool_kernel_size = None

                project_conv = TtASPP(
                    in_channels=in_channel,
                    out_channels=aspp_channels,
                    dilations=aspp_dilations,
                    device=self.device,
                    norm=norm,
                    activation="relu",
                    dropout=aspp_dropout,
                    pool_kernel_size=pool_kernel_size,
                    shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
                    shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
                    shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
                )
                decoder_stage["project_conv"] = project_conv
                decoder_stage["fuse_conv_0"] = None  # Placeholder
            else:  # Low-level feature stages
                proj_out_ch = project_channels[idx]
                project_conv = _create_tt_conv2d(
                    weight=res2_project_conv_weight if feature_name == "res2" else res3_project_conv_weight,
                    in_ch=in_channel,
                    out_ch=proj_out_ch,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    use_bias=use_bias,
                )

                fuse_in_ch = proj_out_ch + decoder_channels[idx + 1]
                fuse_out_ch = decoder_channels[idx]
                fuse_conv_0 = _create_tt_conv2d(
                    weight=shared_fuse_conv_0_weight,
                    in_ch=fuse_in_ch,
                    out_ch=fuse_out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_bias=use_bias,
                )
                fuse_conv_1 = _create_tt_conv2d(
                    weight=shared_fuse_conv_1_weight,
                    in_ch=fuse_out_ch,
                    out_ch=fuse_out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_bias=use_bias,
                )

                decoder_stage["project_conv"] = project_conv
                decoder_stage["project_norm"] = get_ttnn_norm(norm, proj_out_ch, device)
                decoder_stage["fuse_conv_0"] = fuse_conv_0
                decoder_stage["fuse_norm_0"] = get_ttnn_norm(norm, fuse_out_ch, device)
                decoder_stage["fuse_conv_1"] = fuse_conv_1
                decoder_stage["fuse_norm_1"] = get_ttnn_norm(norm, fuse_out_ch, device)

            self.decoder[feature_name] = decoder_stage

        # if not self.decoder_only:
        #     self.predictor = _create_tt_conv2d(
        #         weight=predictor_weight,
        #         in_ch=decoder_channels[0], out_ch=num_classes, kernel_size=1,
        #         stride=1, padding=0, use_bias=True
        #     )

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Union[ttnn.Tensor, Tuple[ttnn.Tensor, Dict]]:
        y = self.layers(features)
        if self.decoder_only:
            return y

        # In inference mode, upsample to the final output size
        # Assuming y is NHWC layout
        y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        y = ttnn.upsample(y, scale_factor=self.common_stride)
        y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
        return y, {}

    # def layers(self, features: Dict[str, ttnn.Tensor]) -> ttnn.Tensor:
    #     y = None
    #     # Reverse feature maps for top-down processing
    #     for f in self.in_features[::-1]:
    #         x = features[f]
    #         stage = self.decoder[f]
    #         proj_x = stage["project_conv"](x)

    #         if stage["fuse_conv_0"] is None: # This is the ASPP module
    #             y = proj_x
    #         else:
    #             # Apply norm and activation for the projected features
    #             proj_x = stage["project_norm"](proj_x)
    #             proj_x = self.activation(proj_x)

    #             # Upsample y from the previous, coarser stage
    #             # Assuming NHWC layout, shapes are (N, H, W, C)
    #             # target H, W are from proj_x
    #             scale_h = proj_x.shape[1] // y.shape[1]
    #             scale_w = proj_x.shape[2] // y.shape[2]
    #             y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
    #             y = ttnn.upsample(y, scale_factor=(scale_h, scale_w), mode="bilinear")
    #             y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)

    #             # Fuse by concatenation
    #             y = ttnn.concat([proj_x, y], dim=3) # Concat on channel dim (NHWC)

    #             # Apply the two fuse convolutions with norm and activation
    #             y = stage["fuse_conv_0"](y)
    #             y = stage["fuse_norm_0"](y)
    #             y = self.activation(y)
    #             y = stage["fuse_conv_1"](y)
    #             y = stage["fuse_norm_1"](y)
    #             y = self.activation(y)

    #     if not self.decoder_only:
    #         y = self.predictor(y)
    #     return y
    # This is the main layers method but I am going to write one for debugging
    def layers(self, features: Dict[str, ttnn.Tensor], debug_stage: Optional[str] = None) -> ttnn.Tensor:
        """
        Executes the decoder pipeline, mirroring the PyTorch version's logic.
        Accepts a debug_stage to return intermediate tensors for comparison.
        """
        y = None
        # Reverse feature maps for top-down processing (e.g., ['res5', 'res3', 'res2'])
        feature_keys = self.in_features[::-1]

        # --- Stage 1: ASPP (on the coarsest feature map, e.g., 'res5') ---
        # This corresponds to the first key in our reversed list.
        aspp_feature_key = feature_keys[0]
        x = features[aspp_feature_key]
        stage = self.decoder[aspp_feature_key]
        # The 'project_conv' for this stage is the entire TtASPP module.
        y = stage["project_conv"](x)
        if debug_stage == "aspp_out":
            return y

        # --- Subsequent Fusion Stages (e.g., 'res3', then 'res2') ---
        # We loop through the remaining, higher-resolution features.

        for i, f_key in enumerate(feature_keys[1:]):  # Start from the second feature
            x = features[f_key]
            stage = self.decoder[f_key]

            # 1. Project the low-level features with a 1x1 Conv, followed by Norm and Activation.
            proj_x = stage["project_conv"](x)
            proj_x = stage["project_norm"](proj_x)
            proj_x = self.activation(proj_x)
            if debug_stage == "proj_x_out":
                return proj_x

            # 2. Upsample the output 'y' from the previous, coarser stage.
            # We calculate the scale factor needed to match proj_x's spatial dimensions.
            # TTNN tensors are NHWC, so H is shape[1] and W is shape[2].
            scale_h = proj_x.shape[1] // y.shape[1]
            scale_w = proj_x.shape[2] // y.shape[2]

            # ttnn.upsample requires ROW_MAJOR_LAYOUT.
            y_upsampled = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
            y_upsampled = ttnn.upsample(y_upsampled, scale_factor=(scale_h, scale_w), mode="bilinear")
            y_upsampled = ttnn.to_memory_config(y_upsampled, ttnn.DRAM_MEMORY_CONFIG)
            y_upsampled = ttnn.to_layout(y_upsampled, ttnn.TILE_LAYOUT)

            print("THIS IS Y_UPSAMPLED")
            print(y_upsampled)

            print("THIS IS PROJ_X")
            print(proj_x)

            if debug_stage == "upsample_out":
                return y_upsampled

            # 3. Fuse the features by concatenating along the channel dimension (dim=3 for NHWC).
            y = ttnn.concat([proj_x, y_upsampled], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if debug_stage == "concat_out":
                return y

            # 4. Apply the two 3x3 fuse convolutions, each followed by Norm and Activation.
            y = stage["fuse_conv_0"](y)
            y = stage["fuse_norm_0"](y)
            y = self.activation(y)

            y = stage["fuse_conv_1"](y)
            y = stage["fuse_norm_1"](y)
            y = self.activation(y)

            # 5. Check for debug hooks after each full fusion stage.
            # i=0 is the first fusion stage (e.g., 'res3')
            if i == 0 and debug_stage == "fuse_1_out":
                return y
            # i=1 is the second fusion stage (e.g., 'res2'), which is the final decoder output.
            if i == 1 and debug_stage == "decoder_out":
                return y

        # This part of the logic from the PyTorch 'layers' method is for the standalone mode.
        # It's called after the loop finishes.
        if not self.decoder_only:
            y = self.predictor(y)

        return y


class TtPanopticDeepLabSemSegHead(TtDeepLabV3PlusHead):
    """
    TTNN implementation of the Panoptic-DeepLab semantic segmentation head.
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        device: ttnn.Device,
        *,
        # --- Parameters for this class ---
        head_channels: int,
        num_classes: int,
        norm: str,
        # --- Parameters that will be passed down to the base class ---
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        train_size: Optional[Tuple],
        # --- ALL weights for base class & this class ---
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        shared_fuse_conv_0_weight: torch.Tensor,
        shared_fuse_conv_1_weight: torch.Tensor,
        res3_project_conv_weight: torch.Tensor,
        res2_project_conv_weight: torch.Tensor,
        panoptic_head_0_weight: torch.Tensor,
        panoptic_head_1_weight: torch.Tensor,
        panoptic_predictor_weight: torch.Tensor,
    ):
        # Call the base class __init__ in decoder_only mode
        super().__init__(
            input_shape=input_shape,
            device=device,
            norm=norm,
            num_classes=None,
            predictor_weight=None,
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=decoder_channels,
            common_stride=common_stride,
            train_size=train_size,
            shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
            shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
            shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
            shared_fuse_conv_0_weight=shared_fuse_conv_0_weight,
            shared_fuse_conv_1_weight=shared_fuse_conv_1_weight,
            res3_project_conv_weight=res3_project_conv_weight,
            res2_project_conv_weight=res2_project_conv_weight,
        )
        assert self.decoder_only

        # Create the specific head and predictor layers for this class
        use_bias = norm == ""
        decoder_out_ch = decoder_channels[0]

        # Helper to create TTNN Conv2d
        def _create_tt_conv2d(
            weight: torch.Tensor, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int, use_bias: bool
        ):
            param_dict = {"weight": weight}
            if use_bias:
                param_dict["bias"] = torch.zeros(1, 1, 1, out_ch)
            parameters = TtConv2dParameters.from_torch(param_dict, device=self.device)
            return TtConv2d(parameters, stride=(stride, stride), padding=(padding, padding))

        self.head_0 = _create_tt_conv2d(panoptic_head_0_weight, decoder_out_ch, decoder_out_ch, 3, 1, 1, use_bias)
        self.head_norm_0 = get_ttnn_norm(norm, decoder_out_ch, device)

        self.head_1 = _create_tt_conv2d(panoptic_head_1_weight, decoder_out_ch, head_channels, 3, 1, 1, use_bias)
        self.head_norm_1 = get_ttnn_norm(norm, head_channels, device)

        self.predictor = _create_tt_conv2d(panoptic_predictor_weight, head_channels, num_classes, 1, 1, 0, True)

    def forward(self, features: Dict[str, ttnn.Tensor]) -> Tuple[ttnn.Tensor, Dict]:
        """
        The forward pass for the Panoptic head in inference mode.
        """
        # 1. Get the final logits from the full pipeline (base decoder + this head + this predictor)
        y = self.layers(features)

        # 2. Perform the final upsampling to match the target output size
        # This mirrors the behavior of the PyTorch version's forward pass
        y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        y = ttnn.upsample(y, scale_factor=self.common_stride)
        y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)

        # 3. Return the results in the (tensor, dict) format expected by the test
        return y, {}

    # def layers(self, features: Dict[str, ttnn.Tensor]) -> ttnn.Tensor:
    #     # 1. Get the refined feature map from the base class decoder
    #     y = super().layers(features)

    #     # 2. Apply the specific head layers of this class
    #     y = self.head_0(y)
    #     y = self.head_norm_0(y)
    #     y = self.activation(y)

    #     y = self.head_1(y)
    #     y = self.head_norm_1(y)
    #     y = self.activation(y)

    #     # 3. Apply the final predictor
    #     y = self.predictor(y)
    #     return y
    # This is the main layers method but I am going to write one for debugging
    def layers(self, features, debug_stage=None):
        # Pass debug_stage down to the base class
        y = super().layers(features, debug_stage=debug_stage)

        # If the debug stage was handled by the parent, y will be the intermediate
        # result, so we should return it immediately.
        if debug_stage in ["aspp_out", "fuse_1_out", "decoder_out", "proj_x_out", "upsample_out", "concat_out"]:
            return y

        # --- Panoptic Head stages ---
        y = self.head_0(y)
        y = self.head_norm_0(y)
        y = self.activation(y)
        y = self.head_1(y)
        y = self.head_norm_1(y)
        y = self.activation(y)
        if debug_stage == "panoptic_head_out":
            return y

        y = self.predictor(y)
        if debug_stage == "predictor_out":
            return y

        return y
