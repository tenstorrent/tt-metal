# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.transfuser.tt.utils import TTConv2D


class TtTransfuserBackbone:
    def __init__(
        self,
        parameters,
        stride,
        # model_config,
        # layer_optimisations=neck_optimisations,
    ) -> None:
        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
            "WEIGHTS_DTYPE": ttnn.bfloat8_b,
            "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
        }
        # import pdb; pdb.set_trace()
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=parameters.image_encoder.features.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=2),
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat16,
            # is_reshape=True,
        )

        # **layer_optimisations.conv1,

    def __call__(
        self,
        x,
        device,
    ):
        x = ttnn.permute(x, (0, 2, 3, 1))  # ch last
        # conv1 is stride 2 conv 3x3
        out, shape = self.conv1(device, x, x.shape)
        # Reshape to spatial dimensions: 80 * 352 = 28160
        out = ttnn.reshape(out, (1, 80, 352, 32))
        out = ttnn.permute(out, (0, 3, 1, 2))
        return out


# class TtTransfuserBackbone(LightweightModule):
#     """
#     TT-NN implementation of Multi-scale Fusion Transformer for image + LiDAR feature fusion
#     """

#     def __init__(self, config, mesh_device, state_dict, dtype=ttnn.bfloat16):
#         super().__init__()
#         self.config = config
#         self.mesh_device = mesh_device
#         self.dtype = dtype
#         self.state_dict = state_dict

#         # Adaptive pooling configurations
#         self.img_pool_config = ttnn.MemoryConfig(
#             memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
#             buffer_type=ttnn.BufferType.L1
#         )

#         # Image and LiDAR encoders
#         self.image_encoder = TtImageCNN(config, mesh_device, state_dict, dtype)
#         self.lidar_encoder = TtLidarEncoder(config, mesh_device, state_dict, dtype)

#         # Multi-scale transformers - following existing transformer patterns
#         self.transformer1 = TTGptBlock(config, mesh_device, state_dict, layer_idx=1, dtype=dtype)
#         self.transformer2 = TTGptBlock(config, mesh_device, state_dict, layer_idx=2, dtype=dtype)
#         self.transformer3 = TTGptBlock(config, mesh_device, state_dict, layer_idx=3, dtype=dtype)
#         self.transformer4 = TTGptBlock(config, mesh_device, state_dict, layer_idx=4, dtype=dtype)

#         # Channel adjustment convolutions
#         self._load_channel_conv_weights()

#         # FPN components
#         self._load_fpn_weights()

#     def _load_channel_conv_weights(self):
#         """Load channel adjustment convolution weights"""
#         if "change_channel_conv_image.weight" in self.state_dict:
#             self.change_channel_conv_image_weight = ttnn.from_torch(
#                 self.state_dict["change_channel_conv_image.weight"],
#                 dtype=self.dtype,
#                 device=self.mesh_device,
#                 layout=ttnn.TILE_LAYOUT
#             )
#             self.change_channel_conv_lidar_weight = ttnn.from_torch(
#                 self.state_dict["change_channel_conv_lidar.weight"],
#                 dtype=self.dtype,
#                 device=self.mesh_device,
#                 layout=ttnn.TILE_LAYOUT
#             )
#         else:
#             self.change_channel_conv_image_weight = None
#             self.change_channel_conv_lidar_weight = None

#     def _load_fpn_weights(self):
#         """Load FPN weights following existing conv patterns"""
#         # FPN convolution weights
#         self.c5_conv_weight = ttnn.from_torch(
#             self.state_dict["c5_conv.weight"],
#             dtype=self.dtype,
#             device=self.mesh_device,
#             layout=ttnn.TILE_LAYOUT
#         )
#         self.up_conv5_weight = ttnn.from_torch(
#             self.state_dict["up_conv5.weight"],
#             dtype=self.dtype,
#             device=self.mesh_device,
#             layout=ttnn.TILE_LAYOUT
#         )
#         self.up_conv4_weight = ttnn.from_torch(
#             self.state_dict["up_conv4.weight"],
#             dtype=self.dtype,
#             device=self.mesh_device,
#             layout=ttnn.TILE_LAYOUT
#         )
#         self.up_conv3_weight = ttnn.from_torch(
#             self.state_dict["up_conv3.weight"],
#             dtype=self.dtype,
#             device=self.mesh_device,
#             layout=ttnn.TILE_LAYOUT
#         )

#     def top_down(self, x):
#         """FPN-style top-down processing"""
#         # Following conv2d patterns from existing implementations
#         conv_config = ttnn.Conv2dConfig(
#             dtype=self.dtype,
#             weights_dtype=self.dtype,
#             math_fidelity=ttnn.MathFidelity.HiFi4,
#             activation="relu",
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#         )

#         p5 = ttnn.conv2d(x, self.c5_conv_weight, **conv_config)
#         p5 = ttnn.relu(p5)

#         # Upsample and process
#         p5_up = ttnn.upsample(p5, scale_factor=self.config.bev_upsample_factor, mode="bilinear")
#         p4 = ttnn.conv2d(p5_up, self.up_conv5_weight, **conv_config)
#         p4 = ttnn.relu(p4)

#         p4_up = ttnn.upsample(p4, scale_factor=self.config.bev_upsample_factor, mode="bilinear")
#         p3 = ttnn.conv2d(p4_up, self.up_conv4_weight, **conv_config)
#         p3 = ttnn.relu(p3)

#         p3_up = ttnn.upsample(p3, scale_factor=self.config.bev_upsample_factor, mode="bilinear")
#         p2 = ttnn.conv2d(p3_up, self.up_conv3_weight, **conv_config)
#         p2 = ttnn.relu(p2)

#         return p2, p3, p4, p5

#     def forward(self, image, lidar, velocity):
#         """
#         Forward pass implementing multi-scale fusion
#         """
#         # Normalize image if needed
#         if self.image_encoder.normalize:
#             image_tensor = self._normalize_imagenet(image)
#         else:
#             image_tensor = image

#         lidar_tensor = lidar

#         # Initial CNN processing - following existing conv patterns
#         image_features, lidar_features = self._initial_cnn_processing(image_tensor, lidar_tensor)

#         # Multi-scale fusion at 4 different levels
#         for layer_idx in range(1, 5):
#             image_features, lidar_features = self._process_layer(
#                 image_features, lidar_features, velocity, layer_idx
#             )

#         # Channel adjustment
#         if self.change_channel_conv_image_weight is not None:
#             image_features = ttnn.conv2d(
#                 image_features,
#                 self.change_channel_conv_image_weight,
#                 kernel_size=(1, 1),
#                 memory_config=ttnn.L1_MEMORY_CONFIG
#             )
#             lidar_features = ttnn.conv2d(
#                 lidar_features,
#                 self.change_channel_conv_lidar_weight,
#                 kernel_size=(1, 1),
#                 memory_config=ttnn.L1_MEMORY_CONFIG
#             )

#         # Store for auxiliary outputs
#         x4 = lidar_features
#         image_features_grid = image_features

#         # Global pooling and fusion
#         image_features_pooled = ttnn.global_avg_pool2d(image_features)
#         image_features_flat = ttnn.reshape(image_features_pooled, (image_features_pooled.shape[0], -1))

#         lidar_features_pooled = ttnn.global_avg_pool2d(lidar_features)
#         lidar_features_flat = ttnn.reshape(lidar_features_pooled, (lidar_features_pooled.shape[0], -1))

#         fused_features = ttnn.add(image_features_flat, lidar_features_flat)

#         # FPN processing
#         features = self.top_down(x4)

#         return features, image_features_grid, fused_features

#     def _normalize_imagenet(self, x):
#         """ImageNet normalization using TT-NN operations"""
#         # Convert normalization to TT-NN operations
#         x_div = ttnn.multiply(x, 1.0/255.0)

#         # Channel-wise normalization constants
#         mean_vals = [0.485, 0.456, 0.406]
#         std_vals = [0.229, 0.224, 0.225]

#         # Apply normalization per channel
#         channels = []
#         for i in range(3):
#             channel = ttnn.slice(x_div, (None, i, None, None), (None, i+1, None, None))
#             channel = ttnn.subtract(channel, mean_vals[i])
#             channel = ttnn.multiply(channel, 1.0/std_vals[i])
#             channels.append(channel)

#         return ttnn.concat(channels, dim=1)

#     def _initial_cnn_processing(self, image_tensor, lidar_tensor):
#         """Initial CNN processing for both modalities"""
#         # Process through initial CNN layers
#         image_features = self.image_encoder.forward_initial(image_tensor)
#         lidar_features = self.lidar_encoder.forward_initial(lidar_tensor)

#         return image_features, lidar_features

#     def _process_layer(self, image_features, lidar_features, velocity, layer_idx):
#         """Process a single layer with transformer fusion"""
#         # Get the appropriate transformer
#         transformer = getattr(self, f'transformer{layer_idx}')

#         # Adaptive pooling to anchor sizes
#         image_embd = ttnn.adaptive_avg_pool2d(
#             image_features,
#             output_size=(self.config.img_vert_anchors, self.config.img_horz_anchors),
#             memory_config=self.img_pool_config
#         )
#         lidar_embd = ttnn.adaptive_avg_pool2d(
#             lidar_features,
#             output_size=(self.config.lidar_vert_anchors, self.config.lidar_horz_anchors),
#             memory_config=self.img_pool_config
#         )

#         # Transformer fusion
#         image_fused, lidar_fused = transformer(image_embd, lidar_embd, velocity)

#         # Interpolate back to original size
#         image_fused_resized = ttnn.upsample(
#             image_fused,
#             size=(image_features.shape[2], image_features.shape[3]),
#             mode="bilinear"
#         )
#         lidar_fused_resized = ttnn.upsample(
#             lidar_fused,
#             size=(lidar_features.shape[2], lidar_features.shape[3]),
#             mode="bilinear"
#         )

#         # Residual connection
#         image_features = ttnn.add(image_features, image_fused_resized)
#         lidar_features = ttnn.add(lidar_features, lidar_fused_resized)

#         # Process through next CNN layer
#         image_features = self.image_encoder.forward_layer(image_features, layer_idx)
#         lidar_features = self.lidar_encoder.forward_layer(lidar_features, layer_idx)

#         return image_features, lidar_features
