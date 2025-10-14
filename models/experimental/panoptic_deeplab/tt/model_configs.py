# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import replace
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    AutoShardedStrategyConfiguration,
    HeightSliceStrategyConfiguration,
    WidthSliceStrategyConfiguration,
    ChannelSliceStrategyConfiguration,
)


class ModelOptimisations:
    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat8_b,
        conv_w_dtype=ttnn.bfloat8_b,
    ):
        # Store default data types for convolutions
        self.conv_output_dtype = conv_act_dtype
        self.conv_w_dtype = conv_w_dtype
        self.conv_ws_dtype = ttnn.bfloat8_b

        # Default override configuration - these will be applied on top of base Conv2dConfiguration
        self.default_overrides = {
            "weights_dtype": conv_w_dtype,
            "output_dtype": conv_act_dtype,
            "activation_dtype": conv_act_dtype,
            "sharding_strategy": AutoShardedStrategyConfiguration(),
            "slice_strategy": None,
            "math_fidelity": ttnn.MathFidelity.LoFi,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "deallocate_activation": False,
            "enable_act_double_buffer": False,
            "enable_weights_double_buffer": True,
            "reallocate_halo_output": True,
        }

        # Layer-specific overrides: map from conv_path to override dict
        self.layer_overrides = {}

    def apply_conv_overrides(self, base_config: Conv2dConfiguration, conv_path: str = None) -> Conv2dConfiguration:
        """
        Apply configuration overrides to a base Conv2dConfiguration.

        This method takes a base Conv2dConfiguration extracted from preprocessing and applies:
        1. Default overrides (common to all layers)
        2. Layer-specific overrides (if conv_path is provided and has custom overrides)

        Args:
            base_config: Base Conv2dConfiguration from preprocessing
            conv_path: String path identifying the convolution layer (e.g., "stem.conv1", "res2.0.conv1")

        Returns:
            Conv2dConfiguration: Updated configuration with overrides applied
        """
        # Start with default overrides
        overrides = self.default_overrides.copy()

        # Apply layer-specific overrides if available
        if conv_path is not None and conv_path in self.layer_overrides:
            overrides.update(self.layer_overrides[conv_path])

        # Use dataclass replace to create new configuration with overrides
        return replace(base_config, **overrides)

    def register_layer_override(self, conv_path: str, **overrides):
        """
        Register layer-specific overrides for a given convolution path.

        Args:
            conv_path: String path identifying the convolution layer
            **overrides: Keyword arguments for Conv2dConfiguration fields to override

        Example:
            config.register_layer_override(
                "stem.conv1",
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=64),
                slice_strategy=WidthSliceStrategyConfiguration(num_slices=4)
            )
        """
        if conv_path not in self.layer_overrides:
            self.layer_overrides[conv_path] = {}
        self.layer_overrides[conv_path].update(overrides)

    def get_conv_output_dtype(self):
        """Get the default output dtype for convolutions."""
        return self.conv_output_dtype

    def setup_default_layer_overrides(self):
        """
        Setup commonly used layer-specific overrides for Panoptic-DeepLab model.

        This method pre-configures slicing strategies and sharding strategies for
        specific layers that benefit from custom configurations.
        """
        # STEM CONVOLUTIONS: Use width slicing for all stem layers
        for conv_name in ["stem.conv1", "stem.conv2", "stem.conv3"]:
            self.register_layer_override(conv_name, slice_strategy=WidthSliceStrategyConfiguration(num_slices=4))

        # RESNET BOTTLENECKS: Use width slicing for res3 shortcuts
        for i in range(4):  # res3 typically has 4 blocks
            self.register_layer_override(
                f"res3.{i}.shortcut", slice_strategy=WidthSliceStrategyConfiguration(num_slices=2)
            )

    def setup_aspp_layer_overrides(self):
        """
        Setup layer-specific overrides for ASPP (Atrous Spatial Pyramid Pooling) layers.

        ASPP branches use different slicing strategies:
        - Branch 0: 1x1 conv (no slicing)
        - Branches 1-3: Dilated convs (channel slicing with increasing num_slices)
        - Branch 4: Global pooling (no slicing)
        """
        # Branch 0: 1x1 conv branch (no slicing)
        self.register_layer_override("aspp.convs.0", slice_strategy=None)

        # Branches 1-3: Dilated conv branches (channel slicing)
        channel_slices = [2, 4, 8]
        for i, num_slices in enumerate(channel_slices, start=1):
            self.register_layer_override(
                f"aspp.convs.{i}", slice_strategy=ChannelSliceStrategyConfiguration(num_slices=num_slices)
            )

        # Branch 4: Global pooling branch (no slicing)
        self.register_layer_override("aspp.convs.4", slice_strategy=None)

        # Project layer
        self.register_layer_override("aspp.project", slice_strategy=None)

    def setup_decoder_layer_overrides(self, iteration_index=0):
        """
        Setup layer-specific overrides for decoder convolutions.

        Args:
            iteration_index: Index of the decoder iteration (0 for first, 1+ for subsequent)
        """
        # Decoder projection layers (no slicing typically)
        for stage in ["res5", "res4", "res3"]:
            self.register_layer_override(f"decoder.{stage}.project_conv", slice_strategy=None)

        # Decoder fuse convolutions
        for stage in ["res4", "res3"]:
            # fuse_conv.0
            fuse_conv_0_path = f"decoder.{stage}.fuse_conv.0"
            if iteration_index == 0 and stage == "res3":
                # res3 stage uses channel slicing
                self.register_layer_override(
                    fuse_conv_0_path, slice_strategy=ChannelSliceStrategyConfiguration(num_slices=5)
                )
            else:
                # Other stages use height slicing
                self.register_layer_override(
                    fuse_conv_0_path, slice_strategy=HeightSliceStrategyConfiguration(num_slices=4)
                )

            # fuse_conv.1: Always use height slicing
            self.register_layer_override(
                f"decoder.{stage}.fuse_conv.1", slice_strategy=HeightSliceStrategyConfiguration(num_slices=2)
            )

    def setup_head_layer_overrides(self):
        """
        Setup layer-specific overrides for head convolutions.

        All head convolutions typically use height slicing with num_slices=2.
        """
        # Semantic segmentation head
        for i in [0, 1]:
            self.register_layer_override(
                f"semantic_head.head.{i}", slice_strategy=HeightSliceStrategyConfiguration(num_slices=2)
            )
        self.register_layer_override(
            "semantic_head.predictor", slice_strategy=HeightSliceStrategyConfiguration(num_slices=2)
        )

        # Instance embedding head - center
        for i in [0, 1]:
            self.register_layer_override(
                f"instance_head.center_head.{i}", slice_strategy=HeightSliceStrategyConfiguration(num_slices=2)
            )
        self.register_layer_override(
            "instance_head.center_predictor", slice_strategy=HeightSliceStrategyConfiguration(num_slices=2)
        )

        # Instance embedding head - offset
        for i in [0, 1]:
            self.register_layer_override(
                f"instance_head.offset_head.{i}", slice_strategy=HeightSliceStrategyConfiguration(num_slices=2)
            )
        self.register_layer_override(
            "instance_head.offset_predictor", slice_strategy=HeightSliceStrategyConfiguration(num_slices=2)
        )

    def setup_all_layer_overrides(self):
        """
        Setup all commonly used layer-specific overrides for the complete Panoptic-DeepLab model.

        This is a convenience method that calls all the individual setup methods.
        """
        self.setup_default_layer_overrides()
        self.setup_aspp_layer_overrides()
        self.setup_decoder_layer_overrides()
        self.setup_head_layer_overrides()
