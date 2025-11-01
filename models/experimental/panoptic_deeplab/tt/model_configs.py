# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import replace
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
    HeightSliceStrategyConfiguration,
    WidthSliceStrategyConfiguration,
    ChannelSliceStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    HeightShardedStrategyConfiguration,
)


class ModelOptimisations:
    """
    Configuration manager for Panoptic-DeepLab model optimizations.

    This class manages default and layer-specific overrides for Conv2dConfiguration,
    allowing fine-grained control over sharding strategies, slicing, data types,
    and other convolution parameters on a per-layer basis.
    """

    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat8_b,
        conv_w_dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize model optimization configurations.

        Args:
            conv_act_dtype: Default data type for convolution activations
            conv_w_dtype: Default data type for convolution weights
        """
        self.conv_output_dtype = conv_act_dtype
        self.conv_w_dtype = conv_w_dtype
        self.conv_ws_dtype = ttnn.bfloat8_b

        # Default overrides applied to all layers
        self.default_overrides = {
            "weights_dtype": conv_w_dtype,
            "output_dtype": conv_act_dtype,
            "activation_dtype": conv_act_dtype,
            "sharding_strategy": AutoShardedStrategyConfiguration(),
            "slice_strategy": L1FullSliceStrategyConfiguration(),
            "math_fidelity": ttnn.MathFidelity.LoFi,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "deallocate_activation": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": False,
            "reallocate_halo_output": True,
            "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "config_tensors_in_dram": True,
        }

        # Default overrides applied to all MaxPool2d layers
        self.default_maxpool_overrides = {}

        self.layer_overrides = {}

    # =========================================================================
    # CORE API METHODS
    # =========================================================================

    def apply_conv_overrides(self, base_config: Conv2dConfiguration, conv_path: str = None) -> Conv2dConfiguration:
        """
        Apply configuration overrides to a base Conv2dConfiguration.

        Args:
            base_config: Base Conv2dConfiguration from preprocessing
            conv_path: String path identifying the layer (e.g., "stem.conv1")

        Returns:
            Updated Conv2dConfiguration with overrides applied
        """
        overrides = self.default_overrides.copy()

        if conv_path is not None and conv_path in self.layer_overrides:
            overrides.update(self.layer_overrides[conv_path])

        # Debug logging for stem layers
        if conv_path and conv_path.startswith("stem."):
            import loguru

            slice_strat = overrides.get("slice_strategy", None)
            slice_type = type(slice_strat).__name__ if slice_strat else "None"
            loguru.logger.debug(f"[CONFIG_DEBUG] {conv_path}: slice_strategy={slice_type}")
            if conv_path in self.layer_overrides:
                loguru.logger.debug(f"[CONFIG_DEBUG] {conv_path}: layer_overrides={self.layer_overrides[conv_path]}")

        return replace(base_config, **overrides)

    def apply_maxpool_overrides(
        self, base_config: MaxPool2dConfiguration, maxpool_path: str = None
    ) -> MaxPool2dConfiguration:
        """
        Apply configuration overrides to a base MaxPool2dConfiguration.

        Args:
            base_config: Base MaxPool2dConfiguration from preprocessing
            maxpool_path: String path identifying the layer (e.g., "stem.maxpool")

        Returns:
            Updated MaxPool2dConfiguration with overrides applied
        """
        overrides = self.default_maxpool_overrides.copy()

        if maxpool_path is not None and maxpool_path in self.layer_overrides:
            overrides.update(self.layer_overrides[maxpool_path])

        return replace(base_config, **overrides)

    def register_layer_override(self, layer_path: str, **overrides):
        """
        Register layer-specific overrides for a layer (convolution or maxpool).

        Args:
            layer_path: String path identifying the layer (e.g., "stem.conv1", "stem.maxpool")
            **overrides: Keyword arguments for layer configuration fields

        Example:
            config.register_layer_override(
                "stem.conv1",
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=64),
                slice_strategy=WidthSliceStrategyConfiguration(num_slices=4)
            )
            config.register_layer_override(
                "stem.maxpool",
                slice_strategy=ChannelSliceStrategyConfiguration(num_slices=2),
                dtype=ttnn.bfloat8_b
            )
        """
        if layer_path not in self.layer_overrides:
            self.layer_overrides[layer_path] = {}
        self.layer_overrides[layer_path].update(overrides)

    def get_conv_output_dtype(self):
        """Get the default output dtype for convolutions."""
        return self.conv_output_dtype

    # =========================================================================
    # HELPER METHODS FOR BULK REGISTRATION
    # =========================================================================

    def _register_multiple_layers(self, layer_paths: list, **overrides):
        """Register the same overrides for multiple layers."""
        for path in layer_paths:
            self.register_layer_override(path, **overrides)

    def _register_stage_blocks(self, stage: str, num_blocks: int, conv_name: str, **overrides):
        """Register overrides for a specific conv across all blocks in a stage."""
        for i in range(num_blocks):
            self.register_layer_override(f"{stage}.{i}.{conv_name}", **overrides)

    # =========================================================================
    # SETUP METHODS
    # =========================================================================

    def setup_all_layer_overrides(self):
        """
        Setup all layer-specific overrides for the complete Panoptic-DeepLab model.
        Convenience method that calls all individual setup methods.
        """
        self.setup_resnet_backbone()
        self.setup_aspp()
        self.setup_decoder()
        self.setup_heads()

    # -------------------------------------------------------------------------
    # RESNET BACKBONE CONFIGURATION
    # -------------------------------------------------------------------------

    def setup_resnet_backbone(self):
        """Setup ResNet50 backbone configurations (stem + res2-5 stages)."""
        self._setup_stem()
        self._setup_res2_stage()
        self._setup_res3_stage()
        self._setup_res4_stage()
        self._setup_res5_stage()
        self._setup_resnet_activation_fusion()

    def _setup_stem(self):
        """
        Stem: 3 convolutions that downsample 512x1024 -> 128x256
        - conv1: 3->64, stride=2, needs height sharding with large act_block_h
        - conv2, conv3: Use width slicing
        """
        self.register_layer_override(
            "stem.conv1",
            slice_strategy=L1FullSliceStrategyConfiguration(),
            sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=1312),  # 41 tiles
        )
        self._register_multiple_layers(
            ["stem.conv2", "stem.conv3"],
            slice_strategy=WidthSliceStrategyConfiguration(num_slices=0),
        )
        self.register_layer_override(
            "stem.maxpool",
            slice_strategy=ChannelSliceStrategyConfiguration(num_slices=2),
            dtype=ttnn.bfloat8_b,
            output_layout=ttnn.TILE_LAYOUT,
        )

    def _setup_res2_stage(self):
        """
        Res2 (128x256): 3 bottleneck blocks, 256 output channels
        - conv2: 3x3 convs need height sharding
        - conv3: 1x1 convs need width slicing to reduce buffer size
        """
        # Conv2: 3x3 spatial convolutions
        self._register_stage_blocks(
            "res2",
            3,
            "conv2",
            sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=128),  # 4 tiles
        )

    def _setup_res3_stage(self):
        """
        Res3 (128x256->64x128): 4 bottleneck blocks, 512 output channels
        - First block has stride=2 downsample
        - conv2: 3x3 convs with height sharding
        - shortcut: First block uses width slicing and block sharding
        """
        # Conv2: First block downsamples with stride=2
        self.register_layer_override(
            "res3.0.conv2",
            sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=256),  # 8 tiles
        )

        # Conv2: Remaining blocks at 64x128 resolution
        self._register_stage_blocks(
            "res3",
            3,
            "conv2",
            sharding_strategy=HeightShardedStrategyConfiguration(),
            start_idx=1,
        )

        # Shortcut: Downsample layer in first block
        self.register_layer_override(
            "res3.0.shortcut",
            slice_strategy=WidthSliceStrategyConfiguration(num_slices=0),
            sharding_strategy=HeightShardedStrategyConfiguration(),
        )

    def _setup_res4_stage(self):
        """
        Res4 (64x128->32x64): 6 bottleneck blocks, 1024 output channels
        - First block has stride=2 downsample
        - conv2: 3x3 convs with height sharding
        - shortcut: First block uses block sharding
        """
        # Conv2: First block downsamples with stride=2
        self.register_layer_override(
            "res4.0.conv2",
            sharding_strategy=HeightShardedStrategyConfiguration(),
        )

        # Conv2: Remaining blocks at 32x64 resolution
        for i in range(1, 6):
            self.register_layer_override(
                f"res4.{i}.conv2",
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=64),  # 2 tiles
            )

        # Shortcut: Downsample layer in first block
        self.register_layer_override(
            "res4.0.shortcut",
            sharding_strategy=BlockShardedStrategyConfiguration(),
            enable_weights_double_buffer=True,
        )

    def _setup_res5_stage(self):
        """
        Res5 (32x64): 3 bottleneck blocks with dilation=2, 2048 output channels
        - All conv2: 3x3 dilated convs with block sharding and width slicing
        """
        for i in range(3):
            self.register_layer_override(
                f"res5.{i}.conv2",
                sharding_strategy=BlockShardedStrategyConfiguration(),
                slice_strategy=WidthSliceStrategyConfiguration(num_slices=2),
                enable_weights_double_buffer=True,
            )

    def _setup_resnet_activation_fusion(self):
        """
        Disable ReLU fusion for layers where ReLU is applied after residual add.

        In ResNet bottlenecks:
        - conv3 (final 1x1 conv): Output goes to residual add, then ReLU
        - shortcut: Output goes to residual add, then ReLU
        """
        # Disable activation fusion for all conv3 and shortcut layers
        stage_configs = [("res2", 3), ("res3", 4), ("res4", 6), ("res5", 3)]

        for stage, num_blocks in stage_configs:
            for i in range(num_blocks):
                # Disable ReLU fusion for conv3 (output goes to residual add)
                self.register_layer_override(f"{stage}.{i}.conv3", activation=None)

                # First block in each stage has shortcut (downsample)
                if i == 0:
                    self.register_layer_override(
                        f"{stage}.{i}.shortcut",
                        activation=None,
                        deallocate_activation=False,
                    )

    def _register_stage_blocks(self, stage: str, num_blocks: int, conv_name: str, start_idx: int = 0, **overrides):
        """Helper to register overrides for blocks starting from start_idx."""
        for i in range(start_idx, num_blocks):
            self.register_layer_override(f"{stage}.{i}.{conv_name}", **overrides)

    # -------------------------------------------------------------------------
    # ASPP (ATROUS SPATIAL PYRAMID POOLING) CONFIGURATION
    # -------------------------------------------------------------------------

    def setup_aspp(self):
        """
        Setup ASPP configurations for all 5 branches.

        ASPP has 5 parallel branches:
        - Branch 0: 1x1 conv (no dilation)
        - Branch 1-3: 3x3 dilated convs (dilation=6, 12, 18)
        - Branch 4: Global average pooling + 1x1 conv
        - Project: 1x1 conv to combine branches
        """
        # Branches without slicing (1x1 convs and pooling)
        no_slice_branches = ["aspp.convs.0", "aspp.convs.4", "aspp.project"]
        self._register_multiple_layers(
            no_slice_branches,
            deallocate_activation=False,
        )

        # Dilated conv branches with channel slicing
        dilated_configs = [
            ("aspp.convs.1", 2, 128),  # dilation=6,  2 slices, act_block_h=128
            ("aspp.convs.2", 4, 128),  # dilation=12, 4 slices, act_block_h=128
            ("aspp.convs.3", 4, 64),  # dilation=18, 4 slices, act_block_h=64
        ]

        for path, num_slices, act_block_h in dilated_configs:
            self.register_layer_override(
                path,
                slice_strategy=ChannelSliceStrategyConfiguration(num_slices=num_slices),
                sharding_strategy=BlockShardedStrategyConfiguration(act_block_h_override=act_block_h),
                deallocate_activation=False,
                activation=None,
                enable_weights_double_buffer=True,
            )

    # -------------------------------------------------------------------------
    # DECODER CONFIGURATION
    # -------------------------------------------------------------------------

    def setup_decoder(self, iteration_index: int = 0):
        """
        Setup decoder configurations for projection and fusion layers.

        Decoder processes features from res5->res4->res3->res2 stages,
        progressively upsampling and fusing features.

        Args:
            iteration_index: Decoder iteration (0 for first pass)
        """
        # Projection layers: No slicing, preserve backbone outputs for head sharing
        projection_layers = [f"decoder.{stage}.project_conv" for stage in ["res5", "res4", "res3", "res2"]]
        self._register_multiple_layers(
            projection_layers,
            deallocate_activation=False,
        )

        # Fusion layers: Two convs per stage (except res5 which only has projection)
        for stage in ["res4", "res3", "res2"]:
            self._setup_decoder_fuse_conv_0(stage, iteration_index)
            self._setup_decoder_fuse_conv_1(stage)

    def _setup_decoder_fuse_conv_0(self, stage: str, iteration_index: int):
        """
        Setup fuse_conv.0 (first fusion conv) for a decoder stage.

        Different stages use different slicing strategies:
        - res2, res3: Channel slicing (higher resolution needs more slicing)
        - res4: Height slicing
        """
        path = f"decoder.{stage}.fuse_conv.0"

        if iteration_index == 0 and stage in ["res3", "res2"]:
            num_slices = 2 if stage == "res3" else 4
            act_block_h = 32 if stage == "res3" else 64

            self.register_layer_override(
                path,
                slice_strategy=ChannelSliceStrategyConfiguration(num_slices=num_slices),
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=act_block_h),
                deallocate_activation=False,
                activation=None,
            )
        else:
            self.register_layer_override(
                path,
                slice_strategy=HeightSliceStrategyConfiguration(num_slices=2),
                sharding_strategy=HeightShardedStrategyConfiguration(),
                deallocate_activation=False,
            )

    def _setup_decoder_fuse_conv_1(self, stage: str):
        """Setup fuse_conv.1 (second fusion conv) with width slicing."""
        self.register_layer_override(
            f"decoder.{stage}.fuse_conv.1",
            slice_strategy=WidthSliceStrategyConfiguration(num_slices=0),
            sharding_strategy=HeightShardedStrategyConfiguration(),
            deallocate_activation=False,
        )

    # -------------------------------------------------------------------------
    # HEAD CONFIGURATION
    # -------------------------------------------------------------------------

    def setup_heads(self):
        """Setup all head configurations (semantic, center, offset)."""
        self._setup_semantic_head()
        self._setup_instance_center_head()
        self._setup_instance_offset_head()

    def _setup_semantic_head(self):
        """
        Semantic segmentation head: 2 intermediate convs + predictor.
        - head.0, head.1: Width slicing with height sharding
        - predictor: No activation (raw logits)
        """
        head_layers = ["semantic_head.head.0", "semantic_head.head.1"]
        self._register_multiple_layers(
            head_layers,
            slice_strategy=WidthSliceStrategyConfiguration(num_slices=0),
            sharding_strategy=HeightShardedStrategyConfiguration(),
            deallocate_activation=False,
        )

        self.register_layer_override(
            "semantic_head.predictor",
            activation=None,  # Raw logits, no ReLU
            deallocate_activation=False,
        )

    def _setup_instance_center_head(self):
        """
        Instance center head: 2 intermediate convs + predictor.
        Same configuration as semantic head.
        """
        head_layers = ["instance_head.center_head.0", "instance_head.center_head.1"]
        self._register_multiple_layers(
            head_layers,
            slice_strategy=WidthSliceStrategyConfiguration(num_slices=0),
            sharding_strategy=HeightShardedStrategyConfiguration(),
            deallocate_activation=False,
        )

        self.register_layer_override(
            "instance_head.center_predictor",
            activation=None,  # Raw logits, no ReLU
            deallocate_activation=False,
        )

    def _setup_instance_offset_head(self):
        """
        Instance offset head: 2 intermediate convs + predictor.
        Same configuration as semantic head.
        """
        head_layers = ["instance_head.offset_head.0", "instance_head.offset_head.1"]
        self._register_multiple_layers(
            head_layers,
            slice_strategy=WidthSliceStrategyConfiguration(num_slices=0),
            sharding_strategy=HeightShardedStrategyConfiguration(),
            deallocate_activation=False,
        )

        self.register_layer_override(
            "instance_head.offset_predictor",
            activation=None,  # Raw logits, no ReLU
            deallocate_activation=False,
        )
