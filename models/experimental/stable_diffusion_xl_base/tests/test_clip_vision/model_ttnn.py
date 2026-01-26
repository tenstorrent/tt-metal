# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CLIP + Resampler model forward pass."""

from consteval import *

import ttnn
from models.common.lightweightmodule import LightweightModule


class CLIPVisionEncoderAndResamplerTTNN(LightweightModule):
    """CLIP Vision Encoder + IP-Adapter Resampler model."""

    def __init__(self, weights, device):
        """
        Initialize the model with pre-loaded weights.

        Args:
            weights: List of weight tensors (loaded via load_inputs_from_pytorch)
            device: TTNN device
        """
        self.weights = weights
        self.device = device

        # Run const-eval functions once at init and store the results dict
        self._ce = run_const_evals(self.weights, {}, device)

        # self.LAYER_NORM_EPSILON = 9.9999997473787516e-06
        self.LAYER_NORM_EPSILON = 1e-5

    def forward(self, pixel_values):
        """
        Run the CLIP + Resampler model.

        Args:
            pixel_values: Input image tensor [batch, channels, height, width]

        Returns:
            List containing the output tensor
        """

        # Move input to device
        assert pixel_values.device() is None, "pixel_values must be on host"
        pixel_values = ttnn.to_device(
            pixel_values,
            self.device,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )

        ttnn_to_layout_287 = ttnn.to_layout(
            pixel_values,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(pixel_values, False)
        ttnn_permute_3 = ttnn.permute(
            ttnn_to_layout_287,
            [0, 2, 3, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_to_layout_287, False)
        ttnn_reshape_192 = ttnn.reshape(
            ttnn_permute_3,
            [1, 1, 50176, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_3, False)
        ttnn_conv2d_0 = ttnn.conv2d(
            input_tensor=ttnn_reshape_192,
            weight_tensor=self._ce["ce_46_0"],
            device=self.device,
            in_channels=3,
            out_channels=1280,
            batch_size=1,
            input_height=224,
            input_width=224,
            kernel_size=[14, 14],
            stride=[14, 14],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            bias_tensor=None,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_192, False)
        ttnn_reshape_193 = ttnn.reshape(
            ttnn_conv2d_0,
            [1, 16, 16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_conv2d_0, False)
        ttnn_permute_4 = ttnn.permute(
            ttnn_reshape_193,
            [0, 3, 1, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_193, False)
        ttnn_reshape_194 = ttnn.reshape(
            ttnn_permute_4,
            [1, 1280, 256],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_4, False)
        list_394 = [self._ce["ce_116_0"], ttnn_reshape_194]
        ttnn_concat_62 = ttnn.concat(
            list_394,
            2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_194, False)
        ttnn_add_0 = ttnn.add(
            ttnn_concat_62,
            self._ce["ce_53_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_concat_62, False)
        ttnn_permute_5 = ttnn.permute(
            ttnn_add_0,
            [0, 2, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_add_0, False)
        ttnn_layer_norm_1 = ttnn.layer_norm(
            ttnn_permute_5,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[386],
            bias=self.weights[385],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn.deallocate(ttnn_permute_5, False)
        ttnn_layer_norm_2 = ttnn.layer_norm(
            ttnn_layer_norm_1,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[384],
            bias=self.weights[383],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_195 = ttnn.reshape(
            ttnn_layer_norm_2,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_2, False)
        ttnn_matmul_1 = ttnn.matmul(
            ttnn_reshape_195,
            self._ce["ce_123_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_195, False)
        ttnn_add_1 = ttnn.add(
            ttnn_matmul_1,
            self._ce["ce_138_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_1, False)
        ttnn_slice_0 = ttnn.slice(
            ttnn_add_1,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_1 = ttnn.slice(
            ttnn_add_1,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_2 = ttnn.slice(
            ttnn_add_1,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_1, False)
        ttnn_reshape_196 = ttnn.reshape(
            ttnn_slice_0,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_0, False)
        ttnn_reshape_197 = ttnn.reshape(
            ttnn_slice_1,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_1, False)
        ttnn_reshape_198 = ttnn.reshape(
            ttnn_slice_2,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_2, False)
        ttnn_permute_6 = ttnn.permute(
            ttnn_reshape_196,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_196, False)
        ttnn_permute_7 = ttnn.permute(
            ttnn_reshape_197,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_197, False)
        ttnn_permute_8 = ttnn.permute(
            ttnn_reshape_198,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_198, False)
        ttnn_typecast_2 = ttnn.typecast(
            ttnn_permute_6,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_6, False)
        ttnn_multiply_1 = ttnn.multiply(
            ttnn_typecast_2,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_2, False)
        ttnn_typecast_3 = ttnn.typecast(
            ttnn_permute_7,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_7, False)
        ttnn_permute_9 = ttnn.permute(
            ttnn_typecast_3,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_3, False)
        ttnn_multiply_2 = ttnn.multiply(
            ttnn_permute_9,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_9, False)
        ttnn_matmul_2 = ttnn.matmul(
            ttnn_multiply_1,
            ttnn_multiply_2,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_2, False)
        ttnn.deallocate(ttnn_multiply_1, False)
        ttnn_eq_0 = ttnn.eq(
            ttnn_matmul_2,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_0 = ttnn.logical_not(
            ttnn_eq_0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_0, False)
        ttnn_sum_0 = ttnn.sum(
            ttnn_logical_not_0,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_0, False)
        ttnn_ne_0 = ttnn.ne(
            ttnn_sum_0,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_0, False)
        ttnn_logical_not_1 = ttnn.logical_not(
            ttnn_ne_0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_0, False)
        ttnn_reshape_199 = ttnn.reshape(
            ttnn_logical_not_1,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_1, False)
        ttnn_softmax_0 = ttnn.softmax(
            ttnn_matmul_2,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_2, False)
        ttnn_repeat_188 = ttnn.repeat(ttnn_reshape_199, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_199, False)
        ttnn_typecast_4 = ttnn.typecast(
            ttnn_repeat_188,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_188, False)
        ttnn_where_0 = ttnn.where(
            ttnn_typecast_4,
            self._ce["cez_4_0"],
            ttnn_softmax_0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_4, False)
        ttnn.deallocate(ttnn_softmax_0, False)
        ttnn_typecast_5 = ttnn.typecast(
            ttnn_permute_8,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_8, False)
        ttnn_matmul_3 = ttnn.matmul(
            ttnn_where_0,
            ttnn_typecast_5,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_5, False)
        ttnn.deallocate(ttnn_where_0, False)
        ttnn_typecast_6 = ttnn.typecast(
            ttnn_matmul_3,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_3, False)
        ttnn_permute_10 = ttnn.permute(
            ttnn_typecast_6,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_6, False)
        ttnn_reshape_200 = ttnn.reshape(
            ttnn_permute_10,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_10, False)
        ttnn_matmul_4 = ttnn.matmul(
            ttnn_reshape_200,
            self.weights[380],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_200, False)
        ttnn_add_2 = ttnn.add(
            ttnn_matmul_4,
            self._ce["ce_76_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_4, False)
        ttnn_add_3 = ttnn.add(
            ttnn_layer_norm_1,
            ttnn_add_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_2, False)
        ttnn.deallocate(ttnn_layer_norm_1, False)
        ttnn_layer_norm_3 = ttnn.layer_norm(
            ttnn_add_3,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[378],
            bias=self.weights[377],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_201 = ttnn.reshape(
            ttnn_layer_norm_3,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_3, False)
        ttnn_matmul_5 = ttnn.matmul(
            ttnn_reshape_201,
            self.weights[376],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_201, False)
        ttnn_add_4 = ttnn.add(
            ttnn_matmul_5,
            self._ce["ce_63_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_5, False)
        ttnn_gelu_0 = ttnn.gelu(
            ttnn_add_4,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_4, False)
        ttnn_reshape_202 = ttnn.reshape(
            ttnn_gelu_0,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_0, False)
        ttnn_matmul_6 = ttnn.matmul(
            ttnn_reshape_202,
            self.weights[374],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_202, False)
        ttnn_add_5 = ttnn.add(
            ttnn_matmul_6,
            self._ce["ce_157_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_6, False)
        ttnn_add_6 = ttnn.add(
            ttnn_add_3,
            ttnn_add_5,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_5, False)
        ttnn.deallocate(ttnn_add_3, False)
        ttnn_layer_norm_4 = ttnn.layer_norm(
            ttnn_add_6,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[372],
            bias=self.weights[371],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_203 = ttnn.reshape(
            ttnn_layer_norm_4,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_4, False)
        ttnn_matmul_7 = ttnn.matmul(
            ttnn_reshape_203,
            self._ce["ce_88_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_203, False)
        ttnn_add_7 = ttnn.add(
            ttnn_matmul_7,
            self._ce["ce_107_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_7, False)
        ttnn_slice_3 = ttnn.slice(
            ttnn_add_7,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_4 = ttnn.slice(
            ttnn_add_7,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_5 = ttnn.slice(
            ttnn_add_7,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_7, False)
        ttnn_reshape_204 = ttnn.reshape(
            ttnn_slice_3,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_3, False)
        ttnn_reshape_205 = ttnn.reshape(
            ttnn_slice_4,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_4, False)
        ttnn_reshape_206 = ttnn.reshape(
            ttnn_slice_5,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_5, False)
        ttnn_permute_11 = ttnn.permute(
            ttnn_reshape_204,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_204, False)
        ttnn_permute_12 = ttnn.permute(
            ttnn_reshape_205,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_205, False)
        ttnn_permute_13 = ttnn.permute(
            ttnn_reshape_206,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_206, False)
        ttnn_typecast_7 = ttnn.typecast(
            ttnn_permute_11,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_11, False)
        ttnn_multiply_3 = ttnn.multiply(
            ttnn_typecast_7,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_7, False)
        ttnn_typecast_8 = ttnn.typecast(
            ttnn_permute_12,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_12, False)
        ttnn_permute_14 = ttnn.permute(
            ttnn_typecast_8,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_8, False)
        ttnn_multiply_4 = ttnn.multiply(
            ttnn_permute_14,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_14, False)
        ttnn_matmul_8 = ttnn.matmul(
            ttnn_multiply_3,
            ttnn_multiply_4,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_4, False)
        ttnn.deallocate(ttnn_multiply_3, False)
        ttnn_eq_1 = ttnn.eq(
            ttnn_matmul_8,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_2 = ttnn.logical_not(
            ttnn_eq_1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_1, False)
        ttnn_sum_1 = ttnn.sum(
            ttnn_logical_not_2,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_2, False)
        ttnn_ne_1 = ttnn.ne(
            ttnn_sum_1,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_1, False)
        ttnn_logical_not_3 = ttnn.logical_not(
            ttnn_ne_1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_1, False)
        ttnn_reshape_207 = ttnn.reshape(
            ttnn_logical_not_3,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_3, False)
        ttnn_softmax_1 = ttnn.softmax(
            ttnn_matmul_8,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_8, False)
        ttnn_repeat_189 = ttnn.repeat(ttnn_reshape_207, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_207, False)
        ttnn_typecast_9 = ttnn.typecast(
            ttnn_repeat_189,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_189, False)
        ttnn_where_1 = ttnn.where(
            ttnn_typecast_9,
            self._ce["cez_4_0"],
            ttnn_softmax_1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_9, False)
        ttnn.deallocate(ttnn_softmax_1, False)
        ttnn_typecast_10 = ttnn.typecast(
            ttnn_permute_13,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_13, False)
        ttnn_matmul_9 = ttnn.matmul(
            ttnn_where_1,
            ttnn_typecast_10,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_10, False)
        ttnn.deallocate(ttnn_where_1, False)
        ttnn_typecast_11 = ttnn.typecast(
            ttnn_matmul_9,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_9, False)
        ttnn_permute_15 = ttnn.permute(
            ttnn_typecast_11,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_11, False)
        ttnn_reshape_208 = ttnn.reshape(
            ttnn_permute_15,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_15, False)
        ttnn_matmul_10 = ttnn.matmul(
            ttnn_reshape_208,
            self.weights[368],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_208, False)
        ttnn_add_8 = ttnn.add(
            ttnn_matmul_10,
            self._ce["ce_143_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_10, False)
        ttnn_add_9 = ttnn.add(
            ttnn_add_6,
            ttnn_add_8,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_8, False)
        ttnn.deallocate(ttnn_add_6, False)
        ttnn_layer_norm_5 = ttnn.layer_norm(
            ttnn_add_9,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[366],
            bias=self.weights[365],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_209 = ttnn.reshape(
            ttnn_layer_norm_5,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_5, False)
        ttnn_matmul_11 = ttnn.matmul(
            ttnn_reshape_209,
            self.weights[364],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_209, False)
        ttnn_add_10 = ttnn.add(
            ttnn_matmul_11,
            self._ce["ce_148_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_11, False)
        ttnn_gelu_1 = ttnn.gelu(
            ttnn_add_10,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_10, False)
        ttnn_reshape_210 = ttnn.reshape(
            ttnn_gelu_1,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_1, False)
        ttnn_matmul_12 = ttnn.matmul(
            ttnn_reshape_210,
            self.weights[362],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_210, False)
        ttnn_add_11 = ttnn.add(
            ttnn_matmul_12,
            self._ce["ce_25_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_12, False)
        ttnn_add_12 = ttnn.add(
            ttnn_add_9,
            ttnn_add_11,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_11, False)
        ttnn.deallocate(ttnn_add_9, False)
        ttnn_layer_norm_6 = ttnn.layer_norm(
            ttnn_add_12,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[360],
            bias=self.weights[359],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_211 = ttnn.reshape(
            ttnn_layer_norm_6,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_6, False)
        ttnn_matmul_13 = ttnn.matmul(
            ttnn_reshape_211,
            self._ce["ce_84_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_211, False)
        ttnn_add_13 = ttnn.add(
            ttnn_matmul_13,
            self._ce["ce_124_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_13, False)
        ttnn_slice_6 = ttnn.slice(
            ttnn_add_13,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_7 = ttnn.slice(
            ttnn_add_13,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_8 = ttnn.slice(
            ttnn_add_13,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_13, False)
        ttnn_reshape_212 = ttnn.reshape(
            ttnn_slice_6,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_6, False)
        ttnn_reshape_213 = ttnn.reshape(
            ttnn_slice_7,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_7, False)
        ttnn_reshape_214 = ttnn.reshape(
            ttnn_slice_8,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_8, False)
        ttnn_permute_16 = ttnn.permute(
            ttnn_reshape_212,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_212, False)
        ttnn_permute_17 = ttnn.permute(
            ttnn_reshape_213,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_213, False)
        ttnn_permute_18 = ttnn.permute(
            ttnn_reshape_214,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_214, False)
        ttnn_typecast_12 = ttnn.typecast(
            ttnn_permute_16,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_16, False)
        ttnn_multiply_5 = ttnn.multiply(
            ttnn_typecast_12,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_12, False)
        ttnn_typecast_13 = ttnn.typecast(
            ttnn_permute_17,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_17, False)
        ttnn_permute_19 = ttnn.permute(
            ttnn_typecast_13,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_13, False)
        ttnn_multiply_6 = ttnn.multiply(
            ttnn_permute_19,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_19, False)
        ttnn_matmul_14 = ttnn.matmul(
            ttnn_multiply_5,
            ttnn_multiply_6,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_6, False)
        ttnn.deallocate(ttnn_multiply_5, False)
        ttnn_eq_2 = ttnn.eq(
            ttnn_matmul_14,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_4 = ttnn.logical_not(
            ttnn_eq_2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_2, False)
        ttnn_sum_2 = ttnn.sum(
            ttnn_logical_not_4,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_4, False)
        ttnn_ne_2 = ttnn.ne(
            ttnn_sum_2,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_2, False)
        ttnn_logical_not_5 = ttnn.logical_not(
            ttnn_ne_2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_2, False)
        ttnn_reshape_215 = ttnn.reshape(
            ttnn_logical_not_5,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_5, False)
        ttnn_softmax_2 = ttnn.softmax(
            ttnn_matmul_14,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_14, False)
        ttnn_repeat_190 = ttnn.repeat(ttnn_reshape_215, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_215, False)
        ttnn_typecast_14 = ttnn.typecast(
            ttnn_repeat_190,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_190, False)
        ttnn_where_2 = ttnn.where(
            ttnn_typecast_14,
            self._ce["cez_4_0"],
            ttnn_softmax_2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_14, False)
        ttnn.deallocate(ttnn_softmax_2, False)
        ttnn_typecast_15 = ttnn.typecast(
            ttnn_permute_18,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_18, False)
        ttnn_matmul_15 = ttnn.matmul(
            ttnn_where_2,
            ttnn_typecast_15,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_15, False)
        ttnn.deallocate(ttnn_where_2, False)
        ttnn_typecast_16 = ttnn.typecast(
            ttnn_matmul_15,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_15, False)
        ttnn_permute_20 = ttnn.permute(
            ttnn_typecast_16,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_16, False)
        ttnn_reshape_216 = ttnn.reshape(
            ttnn_permute_20,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_20, False)
        ttnn_matmul_16 = ttnn.matmul(
            ttnn_reshape_216,
            self.weights[356],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_216, False)
        ttnn_add_14 = ttnn.add(
            ttnn_matmul_16,
            self._ce["ce_147_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_16, False)
        ttnn_add_15 = ttnn.add(
            ttnn_add_12,
            ttnn_add_14,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_14, False)
        ttnn.deallocate(ttnn_add_12, False)
        ttnn_layer_norm_7 = ttnn.layer_norm(
            ttnn_add_15,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[354],
            bias=self.weights[353],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_217 = ttnn.reshape(
            ttnn_layer_norm_7,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_7, False)
        ttnn_matmul_17 = ttnn.matmul(
            ttnn_reshape_217,
            self.weights[352],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_217, False)
        ttnn_add_16 = ttnn.add(
            ttnn_matmul_17,
            self._ce["ce_118_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_17, False)
        ttnn_gelu_2 = ttnn.gelu(
            ttnn_add_16,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_16, False)
        ttnn_reshape_218 = ttnn.reshape(
            ttnn_gelu_2,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_2, False)
        ttnn_matmul_18 = ttnn.matmul(
            ttnn_reshape_218,
            self.weights[350],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_218, False)
        ttnn_add_17 = ttnn.add(
            ttnn_matmul_18,
            self._ce["ce_97_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_18, False)
        ttnn_add_18 = ttnn.add(
            ttnn_add_15,
            ttnn_add_17,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_17, False)
        ttnn.deallocate(ttnn_add_15, False)
        ttnn_layer_norm_8 = ttnn.layer_norm(
            ttnn_add_18,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[348],
            bias=self.weights[347],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_219 = ttnn.reshape(
            ttnn_layer_norm_8,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_8, False)
        ttnn_matmul_19 = ttnn.matmul(
            ttnn_reshape_219,
            self._ce["ce_110_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_219, False)
        ttnn_add_19 = ttnn.add(
            ttnn_matmul_19,
            self._ce["ce_111_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_19, False)
        ttnn_slice_9 = ttnn.slice(
            ttnn_add_19,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_10 = ttnn.slice(
            ttnn_add_19,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_11 = ttnn.slice(
            ttnn_add_19,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_19, False)
        ttnn_reshape_220 = ttnn.reshape(
            ttnn_slice_9,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_9, False)
        ttnn_reshape_221 = ttnn.reshape(
            ttnn_slice_10,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_10, False)
        ttnn_reshape_222 = ttnn.reshape(
            ttnn_slice_11,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_11, False)
        ttnn_permute_21 = ttnn.permute(
            ttnn_reshape_220,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_220, False)
        ttnn_permute_22 = ttnn.permute(
            ttnn_reshape_221,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_221, False)
        ttnn_permute_23 = ttnn.permute(
            ttnn_reshape_222,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_222, False)
        ttnn_typecast_17 = ttnn.typecast(
            ttnn_permute_21,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_21, False)
        ttnn_multiply_7 = ttnn.multiply(
            ttnn_typecast_17,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_17, False)
        ttnn_typecast_18 = ttnn.typecast(
            ttnn_permute_22,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_22, False)
        ttnn_permute_24 = ttnn.permute(
            ttnn_typecast_18,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_18, False)
        ttnn_multiply_8 = ttnn.multiply(
            ttnn_permute_24,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_24, False)
        ttnn_matmul_20 = ttnn.matmul(
            ttnn_multiply_7,
            ttnn_multiply_8,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_8, False)
        ttnn.deallocate(ttnn_multiply_7, False)
        ttnn_eq_3 = ttnn.eq(
            ttnn_matmul_20,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_6 = ttnn.logical_not(
            ttnn_eq_3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_3, False)
        ttnn_sum_3 = ttnn.sum(
            ttnn_logical_not_6,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_6, False)
        ttnn_ne_3 = ttnn.ne(
            ttnn_sum_3,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_3, False)
        ttnn_logical_not_7 = ttnn.logical_not(
            ttnn_ne_3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_3, False)
        ttnn_reshape_223 = ttnn.reshape(
            ttnn_logical_not_7,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_7, False)
        ttnn_softmax_3 = ttnn.softmax(
            ttnn_matmul_20,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_20, False)
        ttnn_repeat_191 = ttnn.repeat(ttnn_reshape_223, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_223, False)
        ttnn_typecast_19 = ttnn.typecast(
            ttnn_repeat_191,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_191, False)
        ttnn_where_3 = ttnn.where(
            ttnn_typecast_19,
            self._ce["cez_4_0"],
            ttnn_softmax_3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_19, False)
        ttnn.deallocate(ttnn_softmax_3, False)
        ttnn_typecast_20 = ttnn.typecast(
            ttnn_permute_23,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_23, False)
        ttnn_matmul_21 = ttnn.matmul(
            ttnn_where_3,
            ttnn_typecast_20,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_20, False)
        ttnn.deallocate(ttnn_where_3, False)
        ttnn_typecast_21 = ttnn.typecast(
            ttnn_matmul_21,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_21, False)
        ttnn_permute_25 = ttnn.permute(
            ttnn_typecast_21,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_21, False)
        ttnn_reshape_224 = ttnn.reshape(
            ttnn_permute_25,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_25, False)
        ttnn_matmul_22 = ttnn.matmul(
            ttnn_reshape_224,
            self.weights[344],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_224, False)
        ttnn_add_20 = ttnn.add(
            ttnn_matmul_22,
            self._ce["ce_11_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_22, False)
        ttnn_add_21 = ttnn.add(
            ttnn_add_18,
            ttnn_add_20,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_20, False)
        ttnn.deallocate(ttnn_add_18, False)
        ttnn_layer_norm_9 = ttnn.layer_norm(
            ttnn_add_21,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[342],
            bias=self.weights[341],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_225 = ttnn.reshape(
            ttnn_layer_norm_9,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_9, False)
        ttnn_matmul_23 = ttnn.matmul(
            ttnn_reshape_225,
            self.weights[340],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_225, False)
        ttnn_add_22 = ttnn.add(
            ttnn_matmul_23,
            self._ce["ce_45_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_23, False)
        ttnn_gelu_3 = ttnn.gelu(
            ttnn_add_22,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_22, False)
        ttnn_reshape_226 = ttnn.reshape(
            ttnn_gelu_3,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_3, False)
        ttnn_matmul_24 = ttnn.matmul(
            ttnn_reshape_226,
            self.weights[338],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_226, False)
        ttnn_add_23 = ttnn.add(
            ttnn_matmul_24,
            self._ce["ce_74_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_24, False)
        ttnn_add_24 = ttnn.add(
            ttnn_add_21,
            ttnn_add_23,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_23, False)
        ttnn.deallocate(ttnn_add_21, False)
        ttnn_layer_norm_10 = ttnn.layer_norm(
            ttnn_add_24,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[336],
            bias=self.weights[335],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_227 = ttnn.reshape(
            ttnn_layer_norm_10,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_10, False)
        ttnn_matmul_25 = ttnn.matmul(
            ttnn_reshape_227,
            self._ce["ce_6_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_227, False)
        ttnn_add_25 = ttnn.add(
            ttnn_matmul_25,
            self._ce["ce_75_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_25, False)
        ttnn_slice_12 = ttnn.slice(
            ttnn_add_25,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_13 = ttnn.slice(
            ttnn_add_25,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_14 = ttnn.slice(
            ttnn_add_25,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_25, False)
        ttnn_reshape_228 = ttnn.reshape(
            ttnn_slice_12,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_12, False)
        ttnn_reshape_229 = ttnn.reshape(
            ttnn_slice_13,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_13, False)
        ttnn_reshape_230 = ttnn.reshape(
            ttnn_slice_14,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_14, False)
        ttnn_permute_26 = ttnn.permute(
            ttnn_reshape_228,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_228, False)
        ttnn_permute_27 = ttnn.permute(
            ttnn_reshape_229,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_229, False)
        ttnn_permute_28 = ttnn.permute(
            ttnn_reshape_230,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_230, False)
        ttnn_typecast_22 = ttnn.typecast(
            ttnn_permute_26,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_26, False)
        ttnn_multiply_9 = ttnn.multiply(
            ttnn_typecast_22,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_22, False)
        ttnn_typecast_23 = ttnn.typecast(
            ttnn_permute_27,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_27, False)
        ttnn_permute_29 = ttnn.permute(
            ttnn_typecast_23,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_23, False)
        ttnn_multiply_10 = ttnn.multiply(
            ttnn_permute_29,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_29, False)
        ttnn_matmul_26 = ttnn.matmul(
            ttnn_multiply_9,
            ttnn_multiply_10,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_10, False)
        ttnn.deallocate(ttnn_multiply_9, False)
        ttnn_eq_4 = ttnn.eq(
            ttnn_matmul_26,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_8 = ttnn.logical_not(
            ttnn_eq_4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_4, False)
        ttnn_sum_4 = ttnn.sum(
            ttnn_logical_not_8,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_8, False)
        ttnn_ne_4 = ttnn.ne(
            ttnn_sum_4,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_4, False)
        ttnn_logical_not_9 = ttnn.logical_not(
            ttnn_ne_4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_4, False)
        ttnn_reshape_231 = ttnn.reshape(
            ttnn_logical_not_9,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_9, False)
        ttnn_softmax_4 = ttnn.softmax(
            ttnn_matmul_26,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_26, False)
        ttnn_repeat_192 = ttnn.repeat(ttnn_reshape_231, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_231, False)
        ttnn_typecast_24 = ttnn.typecast(
            ttnn_repeat_192,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_192, False)
        ttnn_where_4 = ttnn.where(
            ttnn_typecast_24,
            self._ce["cez_4_0"],
            ttnn_softmax_4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_24, False)
        ttnn.deallocate(ttnn_softmax_4, False)
        ttnn_typecast_25 = ttnn.typecast(
            ttnn_permute_28,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_28, False)
        ttnn_matmul_27 = ttnn.matmul(
            ttnn_where_4,
            ttnn_typecast_25,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_25, False)
        ttnn.deallocate(ttnn_where_4, False)
        ttnn_typecast_26 = ttnn.typecast(
            ttnn_matmul_27,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_27, False)
        ttnn_permute_30 = ttnn.permute(
            ttnn_typecast_26,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_26, False)
        ttnn_reshape_232 = ttnn.reshape(
            ttnn_permute_30,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_30, False)
        ttnn_matmul_28 = ttnn.matmul(
            ttnn_reshape_232,
            self.weights[332],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_232, False)
        ttnn_add_26 = ttnn.add(
            ttnn_matmul_28,
            self._ce["ce_89_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_28, False)
        ttnn_add_27 = ttnn.add(
            ttnn_add_24,
            ttnn_add_26,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_26, False)
        ttnn.deallocate(ttnn_add_24, False)
        ttnn_layer_norm_11 = ttnn.layer_norm(
            ttnn_add_27,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[330],
            bias=self.weights[329],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_233 = ttnn.reshape(
            ttnn_layer_norm_11,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_11, False)
        ttnn_matmul_29 = ttnn.matmul(
            ttnn_reshape_233,
            self.weights[328],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_233, False)
        ttnn_add_28 = ttnn.add(
            ttnn_matmul_29,
            self._ce["ce_91_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_29, False)
        ttnn_gelu_4 = ttnn.gelu(
            ttnn_add_28,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_28, False)
        ttnn_reshape_234 = ttnn.reshape(
            ttnn_gelu_4,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_4, False)
        ttnn_matmul_30 = ttnn.matmul(
            ttnn_reshape_234,
            self.weights[326],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_234, False)
        ttnn_add_29 = ttnn.add(
            ttnn_matmul_30,
            self._ce["ce_99_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_30, False)
        ttnn_add_30 = ttnn.add(
            ttnn_add_27,
            ttnn_add_29,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_29, False)
        ttnn.deallocate(ttnn_add_27, False)
        ttnn_layer_norm_12 = ttnn.layer_norm(
            ttnn_add_30,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[324],
            bias=self.weights[323],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_235 = ttnn.reshape(
            ttnn_layer_norm_12,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_12, False)
        ttnn_matmul_31 = ttnn.matmul(
            ttnn_reshape_235,
            self._ce["ce_94_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_235, False)
        ttnn_add_31 = ttnn.add(
            ttnn_matmul_31,
            self._ce["ce_71_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_31, False)
        ttnn_slice_15 = ttnn.slice(
            ttnn_add_31,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_16 = ttnn.slice(
            ttnn_add_31,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_17 = ttnn.slice(
            ttnn_add_31,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_31, False)
        ttnn_reshape_236 = ttnn.reshape(
            ttnn_slice_15,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_15, False)
        ttnn_reshape_237 = ttnn.reshape(
            ttnn_slice_16,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_16, False)
        ttnn_reshape_238 = ttnn.reshape(
            ttnn_slice_17,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_17, False)
        ttnn_permute_31 = ttnn.permute(
            ttnn_reshape_236,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_236, False)
        ttnn_permute_32 = ttnn.permute(
            ttnn_reshape_237,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_237, False)
        ttnn_permute_33 = ttnn.permute(
            ttnn_reshape_238,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_238, False)
        ttnn_typecast_27 = ttnn.typecast(
            ttnn_permute_31,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_31, False)
        ttnn_multiply_11 = ttnn.multiply(
            ttnn_typecast_27,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_27, False)
        ttnn_typecast_28 = ttnn.typecast(
            ttnn_permute_32,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_32, False)
        ttnn_permute_34 = ttnn.permute(
            ttnn_typecast_28,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_28, False)
        ttnn_multiply_12 = ttnn.multiply(
            ttnn_permute_34,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_34, False)
        ttnn_matmul_32 = ttnn.matmul(
            ttnn_multiply_11,
            ttnn_multiply_12,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_12, False)
        ttnn.deallocate(ttnn_multiply_11, False)
        ttnn_eq_5 = ttnn.eq(
            ttnn_matmul_32,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_10 = ttnn.logical_not(
            ttnn_eq_5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_5, False)
        ttnn_sum_5 = ttnn.sum(
            ttnn_logical_not_10,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_10, False)
        ttnn_ne_5 = ttnn.ne(
            ttnn_sum_5,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_5, False)
        ttnn_logical_not_11 = ttnn.logical_not(
            ttnn_ne_5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_5, False)
        ttnn_reshape_239 = ttnn.reshape(
            ttnn_logical_not_11,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_11, False)
        ttnn_softmax_5 = ttnn.softmax(
            ttnn_matmul_32,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_32, False)
        ttnn_repeat_193 = ttnn.repeat(ttnn_reshape_239, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_239, False)
        ttnn_typecast_29 = ttnn.typecast(
            ttnn_repeat_193,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_193, False)
        ttnn_where_5 = ttnn.where(
            ttnn_typecast_29,
            self._ce["cez_4_0"],
            ttnn_softmax_5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_29, False)
        ttnn.deallocate(ttnn_softmax_5, False)
        ttnn_typecast_30 = ttnn.typecast(
            ttnn_permute_33,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_33, False)
        ttnn_matmul_33 = ttnn.matmul(
            ttnn_where_5,
            ttnn_typecast_30,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_30, False)
        ttnn.deallocate(ttnn_where_5, False)
        ttnn_typecast_31 = ttnn.typecast(
            ttnn_matmul_33,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_33, False)
        ttnn_permute_35 = ttnn.permute(
            ttnn_typecast_31,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_31, False)
        ttnn_reshape_240 = ttnn.reshape(
            ttnn_permute_35,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_35, False)
        ttnn_matmul_34 = ttnn.matmul(
            ttnn_reshape_240,
            self.weights[320],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_240, False)
        ttnn_add_32 = ttnn.add(
            ttnn_matmul_34,
            self._ce["ce_70_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_34, False)
        ttnn_add_33 = ttnn.add(
            ttnn_add_30,
            ttnn_add_32,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_32, False)
        ttnn.deallocate(ttnn_add_30, False)
        ttnn_layer_norm_13 = ttnn.layer_norm(
            ttnn_add_33,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[318],
            bias=self.weights[317],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_241 = ttnn.reshape(
            ttnn_layer_norm_13,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_13, False)
        ttnn_matmul_35 = ttnn.matmul(
            ttnn_reshape_241,
            self.weights[316],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_241, False)
        ttnn_add_34 = ttnn.add(
            ttnn_matmul_35,
            self._ce["ce_58_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_35, False)
        ttnn_gelu_5 = ttnn.gelu(
            ttnn_add_34,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_34, False)
        ttnn_reshape_242 = ttnn.reshape(
            ttnn_gelu_5,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_5, False)
        ttnn_matmul_36 = ttnn.matmul(
            ttnn_reshape_242,
            self.weights[314],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_242, False)
        ttnn_add_35 = ttnn.add(
            ttnn_matmul_36,
            self._ce["ce_149_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_36, False)
        ttnn_add_36 = ttnn.add(
            ttnn_add_33,
            ttnn_add_35,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_35, False)
        ttnn.deallocate(ttnn_add_33, False)
        ttnn_layer_norm_14 = ttnn.layer_norm(
            ttnn_add_36,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[312],
            bias=self.weights[311],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_243 = ttnn.reshape(
            ttnn_layer_norm_14,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_14, False)
        ttnn_matmul_37 = ttnn.matmul(
            ttnn_reshape_243,
            self._ce["ce_120_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_243, False)
        ttnn_add_37 = ttnn.add(
            ttnn_matmul_37,
            self._ce["ce_153_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_37, False)
        ttnn_slice_18 = ttnn.slice(
            ttnn_add_37,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_19 = ttnn.slice(
            ttnn_add_37,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_20 = ttnn.slice(
            ttnn_add_37,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_37, False)
        ttnn_reshape_244 = ttnn.reshape(
            ttnn_slice_18,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_18, False)
        ttnn_reshape_245 = ttnn.reshape(
            ttnn_slice_19,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_19, False)
        ttnn_reshape_246 = ttnn.reshape(
            ttnn_slice_20,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_20, False)
        ttnn_permute_36 = ttnn.permute(
            ttnn_reshape_244,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_244, False)
        ttnn_permute_37 = ttnn.permute(
            ttnn_reshape_245,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_245, False)
        ttnn_permute_38 = ttnn.permute(
            ttnn_reshape_246,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_246, False)
        ttnn_typecast_32 = ttnn.typecast(
            ttnn_permute_36,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_36, False)
        ttnn_multiply_13 = ttnn.multiply(
            ttnn_typecast_32,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_32, False)
        ttnn_typecast_33 = ttnn.typecast(
            ttnn_permute_37,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_37, False)
        ttnn_permute_39 = ttnn.permute(
            ttnn_typecast_33,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_33, False)
        ttnn_multiply_14 = ttnn.multiply(
            ttnn_permute_39,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_39, False)
        ttnn_matmul_38 = ttnn.matmul(
            ttnn_multiply_13,
            ttnn_multiply_14,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_14, False)
        ttnn.deallocate(ttnn_multiply_13, False)
        ttnn_eq_6 = ttnn.eq(
            ttnn_matmul_38,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_12 = ttnn.logical_not(
            ttnn_eq_6,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_6, False)
        ttnn_sum_6 = ttnn.sum(
            ttnn_logical_not_12,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_12, False)
        ttnn_ne_6 = ttnn.ne(
            ttnn_sum_6,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_6, False)
        ttnn_logical_not_13 = ttnn.logical_not(
            ttnn_ne_6,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_6, False)
        ttnn_reshape_247 = ttnn.reshape(
            ttnn_logical_not_13,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_13, False)
        ttnn_softmax_6 = ttnn.softmax(
            ttnn_matmul_38,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_38, False)
        ttnn_repeat_194 = ttnn.repeat(ttnn_reshape_247, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_247, False)
        ttnn_typecast_34 = ttnn.typecast(
            ttnn_repeat_194,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_194, False)
        ttnn_where_6 = ttnn.where(
            ttnn_typecast_34,
            self._ce["cez_4_0"],
            ttnn_softmax_6,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_34, False)
        ttnn.deallocate(ttnn_softmax_6, False)
        ttnn_typecast_35 = ttnn.typecast(
            ttnn_permute_38,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_38, False)
        ttnn_matmul_39 = ttnn.matmul(
            ttnn_where_6,
            ttnn_typecast_35,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_35, False)
        ttnn.deallocate(ttnn_where_6, False)
        ttnn_typecast_36 = ttnn.typecast(
            ttnn_matmul_39,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_39, False)
        ttnn_permute_40 = ttnn.permute(
            ttnn_typecast_36,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_36, False)
        ttnn_reshape_248 = ttnn.reshape(
            ttnn_permute_40,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_40, False)
        ttnn_matmul_40 = ttnn.matmul(
            ttnn_reshape_248,
            self.weights[308],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_248, False)
        ttnn_add_38 = ttnn.add(
            ttnn_matmul_40,
            self._ce["ce_112_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_40, False)
        ttnn_add_39 = ttnn.add(
            ttnn_add_36,
            ttnn_add_38,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_38, False)
        ttnn.deallocate(ttnn_add_36, False)
        ttnn_layer_norm_15 = ttnn.layer_norm(
            ttnn_add_39,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[306],
            bias=self.weights[305],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_249 = ttnn.reshape(
            ttnn_layer_norm_15,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_15, False)
        ttnn_matmul_41 = ttnn.matmul(
            ttnn_reshape_249,
            self.weights[304],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_249, False)
        ttnn_add_40 = ttnn.add(
            ttnn_matmul_41,
            self._ce["ce_87_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_41, False)
        ttnn_gelu_6 = ttnn.gelu(
            ttnn_add_40,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_40, False)
        ttnn_reshape_250 = ttnn.reshape(
            ttnn_gelu_6,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_6, False)
        ttnn_matmul_42 = ttnn.matmul(
            ttnn_reshape_250,
            self.weights[302],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_250, False)
        ttnn_add_41 = ttnn.add(
            ttnn_matmul_42,
            self._ce["ce_160_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_42, False)
        ttnn_add_42 = ttnn.add(
            ttnn_add_39,
            ttnn_add_41,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_41, False)
        ttnn.deallocate(ttnn_add_39, False)
        ttnn_layer_norm_16 = ttnn.layer_norm(
            ttnn_add_42,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[300],
            bias=self.weights[299],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_251 = ttnn.reshape(
            ttnn_layer_norm_16,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_16, False)
        ttnn_matmul_43 = ttnn.matmul(
            ttnn_reshape_251,
            self._ce["ce_18_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_251, False)
        ttnn_add_43 = ttnn.add(
            ttnn_matmul_43,
            self._ce["ce_96_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_43, False)
        ttnn_slice_21 = ttnn.slice(
            ttnn_add_43,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_22 = ttnn.slice(
            ttnn_add_43,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_23 = ttnn.slice(
            ttnn_add_43,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_43, False)
        ttnn_reshape_252 = ttnn.reshape(
            ttnn_slice_21,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_21, False)
        ttnn_reshape_253 = ttnn.reshape(
            ttnn_slice_22,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_22, False)
        ttnn_reshape_254 = ttnn.reshape(
            ttnn_slice_23,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_23, False)
        ttnn_permute_41 = ttnn.permute(
            ttnn_reshape_252,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_252, False)
        ttnn_permute_42 = ttnn.permute(
            ttnn_reshape_253,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_253, False)
        ttnn_permute_43 = ttnn.permute(
            ttnn_reshape_254,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_254, False)
        ttnn_typecast_37 = ttnn.typecast(
            ttnn_permute_41,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_41, False)
        ttnn_multiply_15 = ttnn.multiply(
            ttnn_typecast_37,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_37, False)
        ttnn_typecast_38 = ttnn.typecast(
            ttnn_permute_42,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_42, False)
        ttnn_permute_44 = ttnn.permute(
            ttnn_typecast_38,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_38, False)
        ttnn_multiply_16 = ttnn.multiply(
            ttnn_permute_44,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_44, False)
        ttnn_matmul_44 = ttnn.matmul(
            ttnn_multiply_15,
            ttnn_multiply_16,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_16, False)
        ttnn.deallocate(ttnn_multiply_15, False)
        ttnn_eq_7 = ttnn.eq(
            ttnn_matmul_44,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_14 = ttnn.logical_not(
            ttnn_eq_7,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_7, False)
        ttnn_sum_7 = ttnn.sum(
            ttnn_logical_not_14,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_14, False)
        ttnn_ne_7 = ttnn.ne(
            ttnn_sum_7,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_7, False)
        ttnn_logical_not_15 = ttnn.logical_not(
            ttnn_ne_7,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_7, False)
        ttnn_reshape_255 = ttnn.reshape(
            ttnn_logical_not_15,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_15, False)
        ttnn_softmax_7 = ttnn.softmax(
            ttnn_matmul_44,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_44, False)
        ttnn_repeat_195 = ttnn.repeat(ttnn_reshape_255, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_255, False)
        ttnn_typecast_39 = ttnn.typecast(
            ttnn_repeat_195,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_195, False)
        ttnn_where_7 = ttnn.where(
            ttnn_typecast_39,
            self._ce["cez_4_0"],
            ttnn_softmax_7,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_39, False)
        ttnn.deallocate(ttnn_softmax_7, False)
        ttnn_typecast_40 = ttnn.typecast(
            ttnn_permute_43,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_43, False)
        ttnn_matmul_45 = ttnn.matmul(
            ttnn_where_7,
            ttnn_typecast_40,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_40, False)
        ttnn.deallocate(ttnn_where_7, False)
        ttnn_typecast_41 = ttnn.typecast(
            ttnn_matmul_45,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_45, False)
        ttnn_permute_45 = ttnn.permute(
            ttnn_typecast_41,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_41, False)
        ttnn_reshape_256 = ttnn.reshape(
            ttnn_permute_45,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_45, False)
        ttnn_matmul_46 = ttnn.matmul(
            ttnn_reshape_256,
            self.weights[296],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_256, False)
        ttnn_add_44 = ttnn.add(
            ttnn_matmul_46,
            self._ce["ce_62_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_46, False)
        ttnn_add_45 = ttnn.add(
            ttnn_add_42,
            ttnn_add_44,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_44, False)
        ttnn.deallocate(ttnn_add_42, False)
        ttnn_layer_norm_17 = ttnn.layer_norm(
            ttnn_add_45,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[294],
            bias=self.weights[293],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_257 = ttnn.reshape(
            ttnn_layer_norm_17,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_17, False)
        ttnn_matmul_47 = ttnn.matmul(
            ttnn_reshape_257,
            self.weights[292],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_257, False)
        ttnn_add_46 = ttnn.add(
            ttnn_matmul_47,
            self._ce["ce_61_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_47, False)
        ttnn_gelu_7 = ttnn.gelu(
            ttnn_add_46,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_46, False)
        ttnn_reshape_258 = ttnn.reshape(
            ttnn_gelu_7,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_7, False)
        ttnn_matmul_48 = ttnn.matmul(
            ttnn_reshape_258,
            self.weights[290],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_258, False)
        ttnn_add_47 = ttnn.add(
            ttnn_matmul_48,
            self._ce["ce_0_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_48, False)
        ttnn_add_48 = ttnn.add(
            ttnn_add_45,
            ttnn_add_47,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_47, False)
        ttnn.deallocate(ttnn_add_45, False)
        ttnn_layer_norm_18 = ttnn.layer_norm(
            ttnn_add_48,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[288],
            bias=self.weights[287],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_259 = ttnn.reshape(
            ttnn_layer_norm_18,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_18, False)
        ttnn_matmul_49 = ttnn.matmul(
            ttnn_reshape_259,
            self._ce["ce_24_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_259, False)
        ttnn_add_49 = ttnn.add(
            ttnn_matmul_49,
            self._ce["ce_13_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_49, False)
        ttnn_slice_24 = ttnn.slice(
            ttnn_add_49,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_25 = ttnn.slice(
            ttnn_add_49,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_26 = ttnn.slice(
            ttnn_add_49,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_49, False)
        ttnn_reshape_260 = ttnn.reshape(
            ttnn_slice_24,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_24, False)
        ttnn_reshape_261 = ttnn.reshape(
            ttnn_slice_25,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_25, False)
        ttnn_reshape_262 = ttnn.reshape(
            ttnn_slice_26,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_26, False)
        ttnn_permute_46 = ttnn.permute(
            ttnn_reshape_260,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_260, False)
        ttnn_permute_47 = ttnn.permute(
            ttnn_reshape_261,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_261, False)
        ttnn_permute_48 = ttnn.permute(
            ttnn_reshape_262,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_262, False)
        ttnn_typecast_42 = ttnn.typecast(
            ttnn_permute_46,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_46, False)
        ttnn_multiply_17 = ttnn.multiply(
            ttnn_typecast_42,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_42, False)
        ttnn_typecast_43 = ttnn.typecast(
            ttnn_permute_47,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_47, False)
        ttnn_permute_49 = ttnn.permute(
            ttnn_typecast_43,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_43, False)
        ttnn_multiply_18 = ttnn.multiply(
            ttnn_permute_49,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_49, False)
        ttnn_matmul_50 = ttnn.matmul(
            ttnn_multiply_17,
            ttnn_multiply_18,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_18, False)
        ttnn.deallocate(ttnn_multiply_17, False)
        ttnn_eq_8 = ttnn.eq(
            ttnn_matmul_50,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_16 = ttnn.logical_not(
            ttnn_eq_8,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_8, False)
        ttnn_sum_8 = ttnn.sum(
            ttnn_logical_not_16,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_16, False)
        ttnn_ne_8 = ttnn.ne(
            ttnn_sum_8,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_8, False)
        ttnn_logical_not_17 = ttnn.logical_not(
            ttnn_ne_8,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_8, False)
        ttnn_reshape_263 = ttnn.reshape(
            ttnn_logical_not_17,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_17, False)
        ttnn_softmax_8 = ttnn.softmax(
            ttnn_matmul_50,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_50, False)
        ttnn_repeat_196 = ttnn.repeat(ttnn_reshape_263, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_263, False)
        ttnn_typecast_44 = ttnn.typecast(
            ttnn_repeat_196,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_196, False)
        ttnn_where_8 = ttnn.where(
            ttnn_typecast_44,
            self._ce["cez_4_0"],
            ttnn_softmax_8,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_44, False)
        ttnn.deallocate(ttnn_softmax_8, False)
        ttnn_typecast_45 = ttnn.typecast(
            ttnn_permute_48,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_48, False)
        ttnn_matmul_51 = ttnn.matmul(
            ttnn_where_8,
            ttnn_typecast_45,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_45, False)
        ttnn.deallocate(ttnn_where_8, False)
        ttnn_typecast_46 = ttnn.typecast(
            ttnn_matmul_51,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_51, False)
        ttnn_permute_50 = ttnn.permute(
            ttnn_typecast_46,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_46, False)
        ttnn_reshape_264 = ttnn.reshape(
            ttnn_permute_50,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_50, False)
        ttnn_matmul_52 = ttnn.matmul(
            ttnn_reshape_264,
            self.weights[284],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_264, False)
        ttnn_add_50 = ttnn.add(
            ttnn_matmul_52,
            self._ce["ce_77_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_52, False)
        ttnn_add_51 = ttnn.add(
            ttnn_add_48,
            ttnn_add_50,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_50, False)
        ttnn.deallocate(ttnn_add_48, False)
        ttnn_layer_norm_19 = ttnn.layer_norm(
            ttnn_add_51,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[282],
            bias=self.weights[281],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_265 = ttnn.reshape(
            ttnn_layer_norm_19,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_19, False)
        ttnn_matmul_53 = ttnn.matmul(
            ttnn_reshape_265,
            self.weights[280],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_265, False)
        ttnn_add_52 = ttnn.add(
            ttnn_matmul_53,
            self._ce["ce_98_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_53, False)
        ttnn_gelu_8 = ttnn.gelu(
            ttnn_add_52,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_52, False)
        ttnn_reshape_266 = ttnn.reshape(
            ttnn_gelu_8,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_8, False)
        ttnn_matmul_54 = ttnn.matmul(
            ttnn_reshape_266,
            self.weights[278],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_266, False)
        ttnn_add_53 = ttnn.add(
            ttnn_matmul_54,
            self._ce["ce_86_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_54, False)
        ttnn_add_54 = ttnn.add(
            ttnn_add_51,
            ttnn_add_53,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_53, False)
        ttnn.deallocate(ttnn_add_51, False)
        ttnn_layer_norm_20 = ttnn.layer_norm(
            ttnn_add_54,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[276],
            bias=self.weights[275],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_267 = ttnn.reshape(
            ttnn_layer_norm_20,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_20, False)
        ttnn_matmul_55 = ttnn.matmul(
            ttnn_reshape_267,
            self._ce["ce_7_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_267, False)
        ttnn_add_55 = ttnn.add(
            ttnn_matmul_55,
            self._ce["ce_144_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_55, False)
        ttnn_slice_27 = ttnn.slice(
            ttnn_add_55,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_28 = ttnn.slice(
            ttnn_add_55,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_29 = ttnn.slice(
            ttnn_add_55,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_55, False)
        ttnn_reshape_268 = ttnn.reshape(
            ttnn_slice_27,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_27, False)
        ttnn_reshape_269 = ttnn.reshape(
            ttnn_slice_28,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_28, False)
        ttnn_reshape_270 = ttnn.reshape(
            ttnn_slice_29,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_29, False)
        ttnn_permute_51 = ttnn.permute(
            ttnn_reshape_268,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_268, False)
        ttnn_permute_52 = ttnn.permute(
            ttnn_reshape_269,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_269, False)
        ttnn_permute_53 = ttnn.permute(
            ttnn_reshape_270,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_270, False)
        ttnn_typecast_47 = ttnn.typecast(
            ttnn_permute_51,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_51, False)
        ttnn_multiply_19 = ttnn.multiply(
            ttnn_typecast_47,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_47, False)
        ttnn_typecast_48 = ttnn.typecast(
            ttnn_permute_52,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_52, False)
        ttnn_permute_54 = ttnn.permute(
            ttnn_typecast_48,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_48, False)
        ttnn_multiply_20 = ttnn.multiply(
            ttnn_permute_54,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_54, False)
        ttnn_matmul_56 = ttnn.matmul(
            ttnn_multiply_19,
            ttnn_multiply_20,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_20, False)
        ttnn.deallocate(ttnn_multiply_19, False)
        ttnn_eq_9 = ttnn.eq(
            ttnn_matmul_56,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_18 = ttnn.logical_not(
            ttnn_eq_9,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_9, False)
        ttnn_sum_9 = ttnn.sum(
            ttnn_logical_not_18,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_18, False)
        ttnn_ne_9 = ttnn.ne(
            ttnn_sum_9,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_9, False)
        ttnn_logical_not_19 = ttnn.logical_not(
            ttnn_ne_9,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_9, False)
        ttnn_reshape_271 = ttnn.reshape(
            ttnn_logical_not_19,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_19, False)
        ttnn_softmax_9 = ttnn.softmax(
            ttnn_matmul_56,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_56, False)
        ttnn_repeat_197 = ttnn.repeat(ttnn_reshape_271, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_271, False)
        ttnn_typecast_49 = ttnn.typecast(
            ttnn_repeat_197,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_197, False)
        ttnn_where_9 = ttnn.where(
            ttnn_typecast_49,
            self._ce["cez_4_0"],
            ttnn_softmax_9,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_49, False)
        ttnn.deallocate(ttnn_softmax_9, False)
        ttnn_typecast_50 = ttnn.typecast(
            ttnn_permute_53,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_53, False)
        ttnn_matmul_57 = ttnn.matmul(
            ttnn_where_9,
            ttnn_typecast_50,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_50, False)
        ttnn.deallocate(ttnn_where_9, False)
        ttnn_typecast_51 = ttnn.typecast(
            ttnn_matmul_57,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_57, False)
        ttnn_permute_55 = ttnn.permute(
            ttnn_typecast_51,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_51, False)
        ttnn_reshape_272 = ttnn.reshape(
            ttnn_permute_55,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_55, False)
        ttnn_matmul_58 = ttnn.matmul(
            ttnn_reshape_272,
            self.weights[272],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_272, False)
        ttnn_add_56 = ttnn.add(
            ttnn_matmul_58,
            self._ce["ce_82_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_58, False)
        ttnn_add_57 = ttnn.add(
            ttnn_add_54,
            ttnn_add_56,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_56, False)
        ttnn.deallocate(ttnn_add_54, False)
        ttnn_layer_norm_21 = ttnn.layer_norm(
            ttnn_add_57,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[270],
            bias=self.weights[269],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_273 = ttnn.reshape(
            ttnn_layer_norm_21,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_21, False)
        ttnn_matmul_59 = ttnn.matmul(
            ttnn_reshape_273,
            self.weights[268],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_273, False)
        ttnn_add_58 = ttnn.add(
            ttnn_matmul_59,
            self._ce["ce_23_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_59, False)
        ttnn_gelu_9 = ttnn.gelu(
            ttnn_add_58,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_58, False)
        ttnn_reshape_274 = ttnn.reshape(
            ttnn_gelu_9,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_9, False)
        ttnn_matmul_60 = ttnn.matmul(
            ttnn_reshape_274,
            self.weights[266],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_274, False)
        ttnn_add_59 = ttnn.add(
            ttnn_matmul_60,
            self._ce["ce_152_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_60, False)
        ttnn_add_60 = ttnn.add(
            ttnn_add_57,
            ttnn_add_59,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_59, False)
        ttnn.deallocate(ttnn_add_57, False)
        ttnn_layer_norm_22 = ttnn.layer_norm(
            ttnn_add_60,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[264],
            bias=self.weights[263],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_275 = ttnn.reshape(
            ttnn_layer_norm_22,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_22, False)
        ttnn_matmul_61 = ttnn.matmul(
            ttnn_reshape_275,
            self._ce["ce_101_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_275, False)
        ttnn_add_61 = ttnn.add(
            ttnn_matmul_61,
            self._ce["ce_65_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_61, False)
        ttnn_slice_30 = ttnn.slice(
            ttnn_add_61,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_31 = ttnn.slice(
            ttnn_add_61,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_32 = ttnn.slice(
            ttnn_add_61,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_61, False)
        ttnn_reshape_276 = ttnn.reshape(
            ttnn_slice_30,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_30, False)
        ttnn_reshape_277 = ttnn.reshape(
            ttnn_slice_31,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_31, False)
        ttnn_reshape_278 = ttnn.reshape(
            ttnn_slice_32,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_32, False)
        ttnn_permute_56 = ttnn.permute(
            ttnn_reshape_276,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_276, False)
        ttnn_permute_57 = ttnn.permute(
            ttnn_reshape_277,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_277, False)
        ttnn_permute_58 = ttnn.permute(
            ttnn_reshape_278,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_278, False)
        ttnn_typecast_52 = ttnn.typecast(
            ttnn_permute_56,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_56, False)
        ttnn_multiply_21 = ttnn.multiply(
            ttnn_typecast_52,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_52, False)
        ttnn_typecast_53 = ttnn.typecast(
            ttnn_permute_57,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_57, False)
        ttnn_permute_59 = ttnn.permute(
            ttnn_typecast_53,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_53, False)
        ttnn_multiply_22 = ttnn.multiply(
            ttnn_permute_59,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_59, False)
        ttnn_matmul_62 = ttnn.matmul(
            ttnn_multiply_21,
            ttnn_multiply_22,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_22, False)
        ttnn.deallocate(ttnn_multiply_21, False)
        ttnn_eq_10 = ttnn.eq(
            ttnn_matmul_62,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_20 = ttnn.logical_not(
            ttnn_eq_10,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_10, False)
        ttnn_sum_10 = ttnn.sum(
            ttnn_logical_not_20,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_20, False)
        ttnn_ne_10 = ttnn.ne(
            ttnn_sum_10,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_10, False)
        ttnn_logical_not_21 = ttnn.logical_not(
            ttnn_ne_10,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_10, False)
        ttnn_reshape_279 = ttnn.reshape(
            ttnn_logical_not_21,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_21, False)
        ttnn_softmax_10 = ttnn.softmax(
            ttnn_matmul_62,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_62, False)
        ttnn_repeat_198 = ttnn.repeat(ttnn_reshape_279, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_279, False)
        ttnn_typecast_54 = ttnn.typecast(
            ttnn_repeat_198,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_198, False)
        ttnn_where_10 = ttnn.where(
            ttnn_typecast_54,
            self._ce["cez_4_0"],
            ttnn_softmax_10,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_54, False)
        ttnn.deallocate(ttnn_softmax_10, False)
        ttnn_typecast_55 = ttnn.typecast(
            ttnn_permute_58,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_58, False)
        ttnn_matmul_63 = ttnn.matmul(
            ttnn_where_10,
            ttnn_typecast_55,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_55, False)
        ttnn.deallocate(ttnn_where_10, False)
        ttnn_typecast_56 = ttnn.typecast(
            ttnn_matmul_63,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_63, False)
        ttnn_permute_60 = ttnn.permute(
            ttnn_typecast_56,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_56, False)
        ttnn_reshape_280 = ttnn.reshape(
            ttnn_permute_60,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_60, False)
        ttnn_matmul_64 = ttnn.matmul(
            ttnn_reshape_280,
            self.weights[260],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_280, False)
        ttnn_add_62 = ttnn.add(
            ttnn_matmul_64,
            self._ce["ce_129_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_64, False)
        ttnn_add_63 = ttnn.add(
            ttnn_add_60,
            ttnn_add_62,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_62, False)
        ttnn.deallocate(ttnn_add_60, False)
        ttnn_layer_norm_23 = ttnn.layer_norm(
            ttnn_add_63,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[258],
            bias=self.weights[257],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_281 = ttnn.reshape(
            ttnn_layer_norm_23,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_23, False)
        ttnn_matmul_65 = ttnn.matmul(
            ttnn_reshape_281,
            self.weights[256],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_281, False)
        ttnn_add_64 = ttnn.add(
            ttnn_matmul_65,
            self._ce["ce_81_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_65, False)
        ttnn_gelu_10 = ttnn.gelu(
            ttnn_add_64,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_64, False)
        ttnn_reshape_282 = ttnn.reshape(
            ttnn_gelu_10,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_10, False)
        ttnn_matmul_66 = ttnn.matmul(
            ttnn_reshape_282,
            self.weights[254],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_282, False)
        ttnn_add_65 = ttnn.add(
            ttnn_matmul_66,
            self._ce["ce_20_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_66, False)
        ttnn_add_66 = ttnn.add(
            ttnn_add_63,
            ttnn_add_65,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_65, False)
        ttnn.deallocate(ttnn_add_63, False)
        ttnn_layer_norm_24 = ttnn.layer_norm(
            ttnn_add_66,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[252],
            bias=self.weights[251],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_283 = ttnn.reshape(
            ttnn_layer_norm_24,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_24, False)
        ttnn_matmul_67 = ttnn.matmul(
            ttnn_reshape_283,
            self._ce["ce_156_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_283, False)
        ttnn_add_67 = ttnn.add(
            ttnn_matmul_67,
            self._ce["ce_9_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_67, False)
        ttnn_slice_33 = ttnn.slice(
            ttnn_add_67,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_34 = ttnn.slice(
            ttnn_add_67,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_35 = ttnn.slice(
            ttnn_add_67,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_67, False)
        ttnn_reshape_284 = ttnn.reshape(
            ttnn_slice_33,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_33, False)
        ttnn_reshape_285 = ttnn.reshape(
            ttnn_slice_34,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_34, False)
        ttnn_reshape_286 = ttnn.reshape(
            ttnn_slice_35,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_35, False)
        ttnn_permute_61 = ttnn.permute(
            ttnn_reshape_284,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_284, False)
        ttnn_permute_62 = ttnn.permute(
            ttnn_reshape_285,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_285, False)
        ttnn_permute_63 = ttnn.permute(
            ttnn_reshape_286,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_286, False)
        ttnn_typecast_57 = ttnn.typecast(
            ttnn_permute_61,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_61, False)
        ttnn_multiply_23 = ttnn.multiply(
            ttnn_typecast_57,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_57, False)
        ttnn_typecast_58 = ttnn.typecast(
            ttnn_permute_62,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_62, False)
        ttnn_permute_64 = ttnn.permute(
            ttnn_typecast_58,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_58, False)
        ttnn_multiply_24 = ttnn.multiply(
            ttnn_permute_64,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_64, False)
        ttnn_matmul_68 = ttnn.matmul(
            ttnn_multiply_23,
            ttnn_multiply_24,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_24, False)
        ttnn.deallocate(ttnn_multiply_23, False)
        ttnn_eq_11 = ttnn.eq(
            ttnn_matmul_68,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_22 = ttnn.logical_not(
            ttnn_eq_11,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_11, False)
        ttnn_sum_11 = ttnn.sum(
            ttnn_logical_not_22,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_22, False)
        ttnn_ne_11 = ttnn.ne(
            ttnn_sum_11,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_11, False)
        ttnn_logical_not_23 = ttnn.logical_not(
            ttnn_ne_11,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_11, False)
        ttnn_reshape_287 = ttnn.reshape(
            ttnn_logical_not_23,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_23, False)
        ttnn_softmax_11 = ttnn.softmax(
            ttnn_matmul_68,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_68, False)
        ttnn_repeat_199 = ttnn.repeat(ttnn_reshape_287, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_287, False)
        ttnn_typecast_59 = ttnn.typecast(
            ttnn_repeat_199,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_199, False)
        ttnn_where_11 = ttnn.where(
            ttnn_typecast_59,
            self._ce["cez_4_0"],
            ttnn_softmax_11,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_59, False)
        ttnn.deallocate(ttnn_softmax_11, False)
        ttnn_typecast_60 = ttnn.typecast(
            ttnn_permute_63,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_63, False)
        ttnn_matmul_69 = ttnn.matmul(
            ttnn_where_11,
            ttnn_typecast_60,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_60, False)
        ttnn.deallocate(ttnn_where_11, False)
        ttnn_typecast_61 = ttnn.typecast(
            ttnn_matmul_69,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_69, False)
        ttnn_permute_65 = ttnn.permute(
            ttnn_typecast_61,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_61, False)
        ttnn_reshape_288 = ttnn.reshape(
            ttnn_permute_65,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_65, False)
        ttnn_matmul_70 = ttnn.matmul(
            ttnn_reshape_288,
            self.weights[248],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_288, False)
        ttnn_add_68 = ttnn.add(
            ttnn_matmul_70,
            self._ce["ce_151_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_70, False)
        ttnn_add_69 = ttnn.add(
            ttnn_add_66,
            ttnn_add_68,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_68, False)
        ttnn.deallocate(ttnn_add_66, False)
        ttnn_layer_norm_25 = ttnn.layer_norm(
            ttnn_add_69,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[246],
            bias=self.weights[245],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_289 = ttnn.reshape(
            ttnn_layer_norm_25,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_25, False)
        ttnn_matmul_71 = ttnn.matmul(
            ttnn_reshape_289,
            self.weights[244],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_289, False)
        ttnn_add_70 = ttnn.add(
            ttnn_matmul_71,
            self._ce["ce_141_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_71, False)
        ttnn_gelu_11 = ttnn.gelu(
            ttnn_add_70,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_70, False)
        ttnn_reshape_290 = ttnn.reshape(
            ttnn_gelu_11,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_11, False)
        ttnn_matmul_72 = ttnn.matmul(
            ttnn_reshape_290,
            self.weights[242],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_290, False)
        ttnn_add_71 = ttnn.add(
            ttnn_matmul_72,
            self._ce["ce_132_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_72, False)
        ttnn_add_72 = ttnn.add(
            ttnn_add_69,
            ttnn_add_71,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_71, False)
        ttnn.deallocate(ttnn_add_69, False)
        ttnn_layer_norm_26 = ttnn.layer_norm(
            ttnn_add_72,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[240],
            bias=self.weights[239],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_291 = ttnn.reshape(
            ttnn_layer_norm_26,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_26, False)
        ttnn_matmul_73 = ttnn.matmul(
            ttnn_reshape_291,
            self._ce["ce_159_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_291, False)
        ttnn_add_73 = ttnn.add(
            ttnn_matmul_73,
            self._ce["ce_67_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_73, False)
        ttnn_slice_36 = ttnn.slice(
            ttnn_add_73,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_37 = ttnn.slice(
            ttnn_add_73,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_38 = ttnn.slice(
            ttnn_add_73,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_73, False)
        ttnn_reshape_292 = ttnn.reshape(
            ttnn_slice_36,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_36, False)
        ttnn_reshape_293 = ttnn.reshape(
            ttnn_slice_37,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_37, False)
        ttnn_reshape_294 = ttnn.reshape(
            ttnn_slice_38,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_38, False)
        ttnn_permute_66 = ttnn.permute(
            ttnn_reshape_292,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_292, False)
        ttnn_permute_67 = ttnn.permute(
            ttnn_reshape_293,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_293, False)
        ttnn_permute_68 = ttnn.permute(
            ttnn_reshape_294,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_294, False)
        ttnn_typecast_62 = ttnn.typecast(
            ttnn_permute_66,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_66, False)
        ttnn_multiply_25 = ttnn.multiply(
            ttnn_typecast_62,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_62, False)
        ttnn_typecast_63 = ttnn.typecast(
            ttnn_permute_67,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_67, False)
        ttnn_permute_69 = ttnn.permute(
            ttnn_typecast_63,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_63, False)
        ttnn_multiply_26 = ttnn.multiply(
            ttnn_permute_69,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_69, False)
        ttnn_matmul_74 = ttnn.matmul(
            ttnn_multiply_25,
            ttnn_multiply_26,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_26, False)
        ttnn.deallocate(ttnn_multiply_25, False)
        ttnn_eq_12 = ttnn.eq(
            ttnn_matmul_74,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_24 = ttnn.logical_not(
            ttnn_eq_12,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_12, False)
        ttnn_sum_12 = ttnn.sum(
            ttnn_logical_not_24,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_24, False)
        ttnn_ne_12 = ttnn.ne(
            ttnn_sum_12,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_12, False)
        ttnn_logical_not_25 = ttnn.logical_not(
            ttnn_ne_12,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_12, False)
        ttnn_reshape_295 = ttnn.reshape(
            ttnn_logical_not_25,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_25, False)
        ttnn_softmax_12 = ttnn.softmax(
            ttnn_matmul_74,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_74, False)
        ttnn_repeat_200 = ttnn.repeat(ttnn_reshape_295, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_295, False)
        ttnn_typecast_64 = ttnn.typecast(
            ttnn_repeat_200,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_200, False)
        ttnn_where_12 = ttnn.where(
            ttnn_typecast_64,
            self._ce["cez_4_0"],
            ttnn_softmax_12,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_64, False)
        ttnn.deallocate(ttnn_softmax_12, False)
        ttnn_typecast_65 = ttnn.typecast(
            ttnn_permute_68,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_68, False)
        ttnn_matmul_75 = ttnn.matmul(
            ttnn_where_12,
            ttnn_typecast_65,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_65, False)
        ttnn.deallocate(ttnn_where_12, False)
        ttnn_typecast_66 = ttnn.typecast(
            ttnn_matmul_75,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_75, False)
        ttnn_permute_70 = ttnn.permute(
            ttnn_typecast_66,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_66, False)
        ttnn_reshape_296 = ttnn.reshape(
            ttnn_permute_70,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_70, False)
        ttnn_matmul_76 = ttnn.matmul(
            ttnn_reshape_296,
            self.weights[236],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_296, False)
        ttnn_add_74 = ttnn.add(
            ttnn_matmul_76,
            self._ce["ce_51_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_76, False)
        ttnn_add_75 = ttnn.add(
            ttnn_add_72,
            ttnn_add_74,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_74, False)
        ttnn.deallocate(ttnn_add_72, False)
        ttnn_layer_norm_27 = ttnn.layer_norm(
            ttnn_add_75,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[234],
            bias=self.weights[233],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_297 = ttnn.reshape(
            ttnn_layer_norm_27,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_27, False)
        ttnn_matmul_77 = ttnn.matmul(
            ttnn_reshape_297,
            self.weights[232],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_297, False)
        ttnn_add_76 = ttnn.add(
            ttnn_matmul_77,
            self._ce["ce_1_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_77, False)
        ttnn_gelu_12 = ttnn.gelu(
            ttnn_add_76,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_76, False)
        ttnn_reshape_298 = ttnn.reshape(
            ttnn_gelu_12,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_12, False)
        ttnn_matmul_78 = ttnn.matmul(
            ttnn_reshape_298,
            self.weights[230],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_298, False)
        ttnn_add_77 = ttnn.add(
            ttnn_matmul_78,
            self._ce["ce_105_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_78, False)
        ttnn_add_78 = ttnn.add(
            ttnn_add_75,
            ttnn_add_77,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_77, False)
        ttnn.deallocate(ttnn_add_75, False)
        ttnn_layer_norm_28 = ttnn.layer_norm(
            ttnn_add_78,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[228],
            bias=self.weights[227],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_299 = ttnn.reshape(
            ttnn_layer_norm_28,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_28, False)
        ttnn_matmul_79 = ttnn.matmul(
            ttnn_reshape_299,
            self._ce["ce_95_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_299, False)
        ttnn_add_79 = ttnn.add(
            ttnn_matmul_79,
            self._ce["ce_128_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_79, False)
        ttnn_slice_39 = ttnn.slice(
            ttnn_add_79,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_40 = ttnn.slice(
            ttnn_add_79,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_41 = ttnn.slice(
            ttnn_add_79,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_79, False)
        ttnn_reshape_300 = ttnn.reshape(
            ttnn_slice_39,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_39, False)
        ttnn_reshape_301 = ttnn.reshape(
            ttnn_slice_40,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_40, False)
        ttnn_reshape_302 = ttnn.reshape(
            ttnn_slice_41,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_41, False)
        ttnn_permute_71 = ttnn.permute(
            ttnn_reshape_300,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_300, False)
        ttnn_permute_72 = ttnn.permute(
            ttnn_reshape_301,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_301, False)
        ttnn_permute_73 = ttnn.permute(
            ttnn_reshape_302,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_302, False)
        ttnn_typecast_67 = ttnn.typecast(
            ttnn_permute_71,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_71, False)
        ttnn_multiply_27 = ttnn.multiply(
            ttnn_typecast_67,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_67, False)
        ttnn_typecast_68 = ttnn.typecast(
            ttnn_permute_72,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_72, False)
        ttnn_permute_74 = ttnn.permute(
            ttnn_typecast_68,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_68, False)
        ttnn_multiply_28 = ttnn.multiply(
            ttnn_permute_74,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_74, False)
        ttnn_matmul_80 = ttnn.matmul(
            ttnn_multiply_27,
            ttnn_multiply_28,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_28, False)
        ttnn.deallocate(ttnn_multiply_27, False)
        ttnn_eq_13 = ttnn.eq(
            ttnn_matmul_80,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_26 = ttnn.logical_not(
            ttnn_eq_13,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_13, False)
        ttnn_sum_13 = ttnn.sum(
            ttnn_logical_not_26,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_26, False)
        ttnn_ne_13 = ttnn.ne(
            ttnn_sum_13,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_13, False)
        ttnn_logical_not_27 = ttnn.logical_not(
            ttnn_ne_13,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_13, False)
        ttnn_reshape_303 = ttnn.reshape(
            ttnn_logical_not_27,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_27, False)
        ttnn_softmax_13 = ttnn.softmax(
            ttnn_matmul_80,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_80, False)
        ttnn_repeat_201 = ttnn.repeat(ttnn_reshape_303, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_303, False)
        ttnn_typecast_69 = ttnn.typecast(
            ttnn_repeat_201,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_201, False)
        ttnn_where_13 = ttnn.where(
            ttnn_typecast_69,
            self._ce["cez_4_0"],
            ttnn_softmax_13,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_69, False)
        ttnn.deallocate(ttnn_softmax_13, False)
        ttnn_typecast_70 = ttnn.typecast(
            ttnn_permute_73,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_73, False)
        ttnn_matmul_81 = ttnn.matmul(
            ttnn_where_13,
            ttnn_typecast_70,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_70, False)
        ttnn.deallocate(ttnn_where_13, False)
        ttnn_typecast_71 = ttnn.typecast(
            ttnn_matmul_81,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_81, False)
        ttnn_permute_75 = ttnn.permute(
            ttnn_typecast_71,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_71, False)
        ttnn_reshape_304 = ttnn.reshape(
            ttnn_permute_75,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_75, False)
        ttnn_matmul_82 = ttnn.matmul(
            ttnn_reshape_304,
            self.weights[224],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_304, False)
        ttnn_add_80 = ttnn.add(
            ttnn_matmul_82,
            self._ce["ce_109_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_82, False)
        ttnn_add_81 = ttnn.add(
            ttnn_add_78,
            ttnn_add_80,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_80, False)
        ttnn.deallocate(ttnn_add_78, False)
        ttnn_layer_norm_29 = ttnn.layer_norm(
            ttnn_add_81,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[222],
            bias=self.weights[221],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_305 = ttnn.reshape(
            ttnn_layer_norm_29,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_29, False)
        ttnn_matmul_83 = ttnn.matmul(
            ttnn_reshape_305,
            self.weights[220],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_305, False)
        ttnn_add_82 = ttnn.add(
            ttnn_matmul_83,
            self._ce["ce_66_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_83, False)
        ttnn_gelu_13 = ttnn.gelu(
            ttnn_add_82,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_82, False)
        ttnn_reshape_306 = ttnn.reshape(
            ttnn_gelu_13,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_13, False)
        ttnn_matmul_84 = ttnn.matmul(
            ttnn_reshape_306,
            self.weights[218],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_306, False)
        ttnn_add_83 = ttnn.add(
            ttnn_matmul_84,
            self._ce["ce_27_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_84, False)
        ttnn_add_84 = ttnn.add(
            ttnn_add_81,
            ttnn_add_83,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_83, False)
        ttnn.deallocate(ttnn_add_81, False)
        ttnn_layer_norm_30 = ttnn.layer_norm(
            ttnn_add_84,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[216],
            bias=self.weights[215],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_307 = ttnn.reshape(
            ttnn_layer_norm_30,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_30, False)
        ttnn_matmul_85 = ttnn.matmul(
            ttnn_reshape_307,
            self._ce["ce_142_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_307, False)
        ttnn_add_85 = ttnn.add(
            ttnn_matmul_85,
            self._ce["ce_12_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_85, False)
        ttnn_slice_42 = ttnn.slice(
            ttnn_add_85,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_43 = ttnn.slice(
            ttnn_add_85,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_44 = ttnn.slice(
            ttnn_add_85,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_85, False)
        ttnn_reshape_308 = ttnn.reshape(
            ttnn_slice_42,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_42, False)
        ttnn_reshape_309 = ttnn.reshape(
            ttnn_slice_43,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_43, False)
        ttnn_reshape_310 = ttnn.reshape(
            ttnn_slice_44,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_44, False)
        ttnn_permute_76 = ttnn.permute(
            ttnn_reshape_308,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_308, False)
        ttnn_permute_77 = ttnn.permute(
            ttnn_reshape_309,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_309, False)
        ttnn_permute_78 = ttnn.permute(
            ttnn_reshape_310,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_310, False)
        ttnn_typecast_72 = ttnn.typecast(
            ttnn_permute_76,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_76, False)
        ttnn_multiply_29 = ttnn.multiply(
            ttnn_typecast_72,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_72, False)
        ttnn_typecast_73 = ttnn.typecast(
            ttnn_permute_77,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_77, False)
        ttnn_permute_79 = ttnn.permute(
            ttnn_typecast_73,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_73, False)
        ttnn_multiply_30 = ttnn.multiply(
            ttnn_permute_79,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_79, False)
        ttnn_matmul_86 = ttnn.matmul(
            ttnn_multiply_29,
            ttnn_multiply_30,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_30, False)
        ttnn.deallocate(ttnn_multiply_29, False)
        ttnn_eq_14 = ttnn.eq(
            ttnn_matmul_86,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_28 = ttnn.logical_not(
            ttnn_eq_14,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_14, False)
        ttnn_sum_14 = ttnn.sum(
            ttnn_logical_not_28,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_28, False)
        ttnn_ne_14 = ttnn.ne(
            ttnn_sum_14,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_14, False)
        ttnn_logical_not_29 = ttnn.logical_not(
            ttnn_ne_14,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_14, False)
        ttnn_reshape_311 = ttnn.reshape(
            ttnn_logical_not_29,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_29, False)
        ttnn_softmax_14 = ttnn.softmax(
            ttnn_matmul_86,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_86, False)
        ttnn_repeat_202 = ttnn.repeat(ttnn_reshape_311, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_311, False)
        ttnn_typecast_74 = ttnn.typecast(
            ttnn_repeat_202,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_202, False)
        ttnn_where_14 = ttnn.where(
            ttnn_typecast_74,
            self._ce["cez_4_0"],
            ttnn_softmax_14,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_74, False)
        ttnn.deallocate(ttnn_softmax_14, False)
        ttnn_typecast_75 = ttnn.typecast(
            ttnn_permute_78,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_78, False)
        ttnn_matmul_87 = ttnn.matmul(
            ttnn_where_14,
            ttnn_typecast_75,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_75, False)
        ttnn.deallocate(ttnn_where_14, False)
        ttnn_typecast_76 = ttnn.typecast(
            ttnn_matmul_87,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_87, False)
        ttnn_permute_80 = ttnn.permute(
            ttnn_typecast_76,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_76, False)
        ttnn_reshape_312 = ttnn.reshape(
            ttnn_permute_80,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_80, False)
        ttnn_matmul_88 = ttnn.matmul(
            ttnn_reshape_312,
            self.weights[212],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_312, False)
        ttnn_add_86 = ttnn.add(
            ttnn_matmul_88,
            self._ce["ce_19_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_88, False)
        ttnn_add_87 = ttnn.add(
            ttnn_add_84,
            ttnn_add_86,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_86, False)
        ttnn.deallocate(ttnn_add_84, False)
        ttnn_layer_norm_31 = ttnn.layer_norm(
            ttnn_add_87,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[210],
            bias=self.weights[209],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_313 = ttnn.reshape(
            ttnn_layer_norm_31,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_31, False)
        ttnn_matmul_89 = ttnn.matmul(
            ttnn_reshape_313,
            self.weights[208],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_313, False)
        ttnn_add_88 = ttnn.add(
            ttnn_matmul_89,
            self._ce["ce_37_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_89, False)
        ttnn_gelu_14 = ttnn.gelu(
            ttnn_add_88,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_88, False)
        ttnn_reshape_314 = ttnn.reshape(
            ttnn_gelu_14,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_14, False)
        ttnn_matmul_90 = ttnn.matmul(
            ttnn_reshape_314,
            self.weights[206],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_314, False)
        ttnn_add_89 = ttnn.add(
            ttnn_matmul_90,
            self._ce["ce_38_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_90, False)
        ttnn_add_90 = ttnn.add(
            ttnn_add_87,
            ttnn_add_89,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_89, False)
        ttnn.deallocate(ttnn_add_87, False)
        ttnn_layer_norm_32 = ttnn.layer_norm(
            ttnn_add_90,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[204],
            bias=self.weights[203],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_315 = ttnn.reshape(
            ttnn_layer_norm_32,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_32, False)
        ttnn_matmul_91 = ttnn.matmul(
            ttnn_reshape_315,
            self._ce["ce_10_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_315, False)
        ttnn_add_91 = ttnn.add(
            ttnn_matmul_91,
            self._ce["ce_40_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_91, False)
        ttnn_slice_45 = ttnn.slice(
            ttnn_add_91,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_46 = ttnn.slice(
            ttnn_add_91,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_47 = ttnn.slice(
            ttnn_add_91,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_91, False)
        ttnn_reshape_316 = ttnn.reshape(
            ttnn_slice_45,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_45, False)
        ttnn_reshape_317 = ttnn.reshape(
            ttnn_slice_46,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_46, False)
        ttnn_reshape_318 = ttnn.reshape(
            ttnn_slice_47,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_47, False)
        ttnn_permute_81 = ttnn.permute(
            ttnn_reshape_316,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_316, False)
        ttnn_permute_82 = ttnn.permute(
            ttnn_reshape_317,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_317, False)
        ttnn_permute_83 = ttnn.permute(
            ttnn_reshape_318,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_318, False)
        ttnn_typecast_77 = ttnn.typecast(
            ttnn_permute_81,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_81, False)
        ttnn_multiply_31 = ttnn.multiply(
            ttnn_typecast_77,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_77, False)
        ttnn_typecast_78 = ttnn.typecast(
            ttnn_permute_82,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_82, False)
        ttnn_permute_84 = ttnn.permute(
            ttnn_typecast_78,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_78, False)
        ttnn_multiply_32 = ttnn.multiply(
            ttnn_permute_84,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_84, False)
        ttnn_matmul_92 = ttnn.matmul(
            ttnn_multiply_31,
            ttnn_multiply_32,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_32, False)
        ttnn.deallocate(ttnn_multiply_31, False)
        ttnn_eq_15 = ttnn.eq(
            ttnn_matmul_92,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_30 = ttnn.logical_not(
            ttnn_eq_15,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_15, False)
        ttnn_sum_15 = ttnn.sum(
            ttnn_logical_not_30,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_30, False)
        ttnn_ne_15 = ttnn.ne(
            ttnn_sum_15,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_15, False)
        ttnn_logical_not_31 = ttnn.logical_not(
            ttnn_ne_15,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_15, False)
        ttnn_reshape_319 = ttnn.reshape(
            ttnn_logical_not_31,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_31, False)
        ttnn_softmax_15 = ttnn.softmax(
            ttnn_matmul_92,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_92, False)
        ttnn_repeat_203 = ttnn.repeat(ttnn_reshape_319, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_319, False)
        ttnn_typecast_79 = ttnn.typecast(
            ttnn_repeat_203,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_203, False)
        ttnn_where_15 = ttnn.where(
            ttnn_typecast_79,
            self._ce["cez_4_0"],
            ttnn_softmax_15,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_79, False)
        ttnn.deallocate(ttnn_softmax_15, False)
        ttnn_typecast_80 = ttnn.typecast(
            ttnn_permute_83,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_83, False)
        ttnn_matmul_93 = ttnn.matmul(
            ttnn_where_15,
            ttnn_typecast_80,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_80, False)
        ttnn.deallocate(ttnn_where_15, False)
        ttnn_typecast_81 = ttnn.typecast(
            ttnn_matmul_93,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_93, False)
        ttnn_permute_85 = ttnn.permute(
            ttnn_typecast_81,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_81, False)
        ttnn_reshape_320 = ttnn.reshape(
            ttnn_permute_85,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_85, False)
        ttnn_matmul_94 = ttnn.matmul(
            ttnn_reshape_320,
            self.weights[200],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_320, False)
        ttnn_add_92 = ttnn.add(
            ttnn_matmul_94,
            self._ce["ce_130_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_94, False)
        ttnn_add_93 = ttnn.add(
            ttnn_add_90,
            ttnn_add_92,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_92, False)
        ttnn.deallocate(ttnn_add_90, False)
        ttnn_layer_norm_33 = ttnn.layer_norm(
            ttnn_add_93,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[198],
            bias=self.weights[197],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_321 = ttnn.reshape(
            ttnn_layer_norm_33,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_33, False)
        ttnn_matmul_95 = ttnn.matmul(
            ttnn_reshape_321,
            self.weights[196],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_321, False)
        ttnn_add_94 = ttnn.add(
            ttnn_matmul_95,
            self._ce["ce_125_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_95, False)
        ttnn_gelu_15 = ttnn.gelu(
            ttnn_add_94,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_94, False)
        ttnn_reshape_322 = ttnn.reshape(
            ttnn_gelu_15,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_15, False)
        ttnn_matmul_96 = ttnn.matmul(
            ttnn_reshape_322,
            self.weights[194],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_322, False)
        ttnn_add_95 = ttnn.add(
            ttnn_matmul_96,
            self._ce["ce_92_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_96, False)
        ttnn_add_96 = ttnn.add(
            ttnn_add_93,
            ttnn_add_95,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_95, False)
        ttnn.deallocate(ttnn_add_93, False)
        ttnn_layer_norm_34 = ttnn.layer_norm(
            ttnn_add_96,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[192],
            bias=self.weights[191],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_323 = ttnn.reshape(
            ttnn_layer_norm_34,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_34, False)
        ttnn_matmul_97 = ttnn.matmul(
            ttnn_reshape_323,
            self._ce["ce_106_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_323, False)
        ttnn_add_97 = ttnn.add(
            ttnn_matmul_97,
            self._ce["ce_21_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_97, False)
        ttnn_slice_48 = ttnn.slice(
            ttnn_add_97,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_49 = ttnn.slice(
            ttnn_add_97,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_50 = ttnn.slice(
            ttnn_add_97,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_97, False)
        ttnn_reshape_324 = ttnn.reshape(
            ttnn_slice_48,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_48, False)
        ttnn_reshape_325 = ttnn.reshape(
            ttnn_slice_49,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_49, False)
        ttnn_reshape_326 = ttnn.reshape(
            ttnn_slice_50,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_50, False)
        ttnn_permute_86 = ttnn.permute(
            ttnn_reshape_324,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_324, False)
        ttnn_permute_87 = ttnn.permute(
            ttnn_reshape_325,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_325, False)
        ttnn_permute_88 = ttnn.permute(
            ttnn_reshape_326,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_326, False)
        ttnn_typecast_82 = ttnn.typecast(
            ttnn_permute_86,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_86, False)
        ttnn_multiply_33 = ttnn.multiply(
            ttnn_typecast_82,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_82, False)
        ttnn_typecast_83 = ttnn.typecast(
            ttnn_permute_87,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_87, False)
        ttnn_permute_89 = ttnn.permute(
            ttnn_typecast_83,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_83, False)
        ttnn_multiply_34 = ttnn.multiply(
            ttnn_permute_89,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_89, False)
        ttnn_matmul_98 = ttnn.matmul(
            ttnn_multiply_33,
            ttnn_multiply_34,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_34, False)
        ttnn.deallocate(ttnn_multiply_33, False)
        ttnn_eq_16 = ttnn.eq(
            ttnn_matmul_98,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_32 = ttnn.logical_not(
            ttnn_eq_16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_16, False)
        ttnn_sum_16 = ttnn.sum(
            ttnn_logical_not_32,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_32, False)
        ttnn_ne_16 = ttnn.ne(
            ttnn_sum_16,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_16, False)
        ttnn_logical_not_33 = ttnn.logical_not(
            ttnn_ne_16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_16, False)
        ttnn_reshape_327 = ttnn.reshape(
            ttnn_logical_not_33,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_33, False)
        ttnn_softmax_16 = ttnn.softmax(
            ttnn_matmul_98,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_98, False)
        ttnn_repeat_204 = ttnn.repeat(ttnn_reshape_327, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_327, False)
        ttnn_typecast_84 = ttnn.typecast(
            ttnn_repeat_204,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_204, False)
        ttnn_where_16 = ttnn.where(
            ttnn_typecast_84,
            self._ce["cez_4_0"],
            ttnn_softmax_16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_84, False)
        ttnn.deallocate(ttnn_softmax_16, False)
        ttnn_typecast_85 = ttnn.typecast(
            ttnn_permute_88,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_88, False)
        ttnn_matmul_99 = ttnn.matmul(
            ttnn_where_16,
            ttnn_typecast_85,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_85, False)
        ttnn.deallocate(ttnn_where_16, False)
        ttnn_typecast_86 = ttnn.typecast(
            ttnn_matmul_99,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_99, False)
        ttnn_permute_90 = ttnn.permute(
            ttnn_typecast_86,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_86, False)
        ttnn_reshape_328 = ttnn.reshape(
            ttnn_permute_90,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_90, False)
        ttnn_matmul_100 = ttnn.matmul(
            ttnn_reshape_328,
            self.weights[188],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_328, False)
        ttnn_add_98 = ttnn.add(
            ttnn_matmul_100,
            self._ce["ce_146_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_100, False)
        ttnn_add_99 = ttnn.add(
            ttnn_add_96,
            ttnn_add_98,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_98, False)
        ttnn.deallocate(ttnn_add_96, False)
        ttnn_layer_norm_35 = ttnn.layer_norm(
            ttnn_add_99,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[186],
            bias=self.weights[185],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_329 = ttnn.reshape(
            ttnn_layer_norm_35,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_35, False)
        ttnn_matmul_101 = ttnn.matmul(
            ttnn_reshape_329,
            self.weights[184],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_329, False)
        ttnn_add_100 = ttnn.add(
            ttnn_matmul_101,
            self._ce["ce_52_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_101, False)
        ttnn_gelu_16 = ttnn.gelu(
            ttnn_add_100,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_100, False)
        ttnn_reshape_330 = ttnn.reshape(
            ttnn_gelu_16,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_16, False)
        ttnn_matmul_102 = ttnn.matmul(
            ttnn_reshape_330,
            self.weights[182],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_330, False)
        ttnn_add_101 = ttnn.add(
            ttnn_matmul_102,
            self._ce["ce_134_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_102, False)
        ttnn_add_102 = ttnn.add(
            ttnn_add_99,
            ttnn_add_101,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_101, False)
        ttnn.deallocate(ttnn_add_99, False)
        ttnn_layer_norm_36 = ttnn.layer_norm(
            ttnn_add_102,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[180],
            bias=self.weights[179],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_331 = ttnn.reshape(
            ttnn_layer_norm_36,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_36, False)
        ttnn_matmul_103 = ttnn.matmul(
            ttnn_reshape_331,
            self._ce["ce_100_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_331, False)
        ttnn_add_103 = ttnn.add(
            ttnn_matmul_103,
            self._ce["ce_122_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_103, False)
        ttnn_slice_51 = ttnn.slice(
            ttnn_add_103,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_52 = ttnn.slice(
            ttnn_add_103,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_53 = ttnn.slice(
            ttnn_add_103,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_103, False)
        ttnn_reshape_332 = ttnn.reshape(
            ttnn_slice_51,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_51, False)
        ttnn_reshape_333 = ttnn.reshape(
            ttnn_slice_52,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_52, False)
        ttnn_reshape_334 = ttnn.reshape(
            ttnn_slice_53,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_53, False)
        ttnn_permute_91 = ttnn.permute(
            ttnn_reshape_332,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_332, False)
        ttnn_permute_92 = ttnn.permute(
            ttnn_reshape_333,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_333, False)
        ttnn_permute_93 = ttnn.permute(
            ttnn_reshape_334,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_334, False)
        ttnn_typecast_87 = ttnn.typecast(
            ttnn_permute_91,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_91, False)
        ttnn_multiply_35 = ttnn.multiply(
            ttnn_typecast_87,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_87, False)
        ttnn_typecast_88 = ttnn.typecast(
            ttnn_permute_92,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_92, False)
        ttnn_permute_94 = ttnn.permute(
            ttnn_typecast_88,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_88, False)
        ttnn_multiply_36 = ttnn.multiply(
            ttnn_permute_94,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_94, False)
        ttnn_matmul_104 = ttnn.matmul(
            ttnn_multiply_35,
            ttnn_multiply_36,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_36, False)
        ttnn.deallocate(ttnn_multiply_35, False)
        ttnn_eq_17 = ttnn.eq(
            ttnn_matmul_104,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_34 = ttnn.logical_not(
            ttnn_eq_17,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_17, False)
        ttnn_sum_17 = ttnn.sum(
            ttnn_logical_not_34,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_34, False)
        ttnn_ne_17 = ttnn.ne(
            ttnn_sum_17,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_17, False)
        ttnn_logical_not_35 = ttnn.logical_not(
            ttnn_ne_17,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_17, False)
        ttnn_reshape_335 = ttnn.reshape(
            ttnn_logical_not_35,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_35, False)
        ttnn_softmax_17 = ttnn.softmax(
            ttnn_matmul_104,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_104, False)
        ttnn_repeat_205 = ttnn.repeat(ttnn_reshape_335, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_335, False)
        ttnn_typecast_89 = ttnn.typecast(
            ttnn_repeat_205,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_205, False)
        ttnn_where_17 = ttnn.where(
            ttnn_typecast_89,
            self._ce["cez_4_0"],
            ttnn_softmax_17,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_89, False)
        ttnn.deallocate(ttnn_softmax_17, False)
        ttnn_typecast_90 = ttnn.typecast(
            ttnn_permute_93,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_93, False)
        ttnn_matmul_105 = ttnn.matmul(
            ttnn_where_17,
            ttnn_typecast_90,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_90, False)
        ttnn.deallocate(ttnn_where_17, False)
        ttnn_typecast_91 = ttnn.typecast(
            ttnn_matmul_105,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_105, False)
        ttnn_permute_95 = ttnn.permute(
            ttnn_typecast_91,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_91, False)
        ttnn_reshape_336 = ttnn.reshape(
            ttnn_permute_95,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_95, False)
        ttnn_matmul_106 = ttnn.matmul(
            ttnn_reshape_336,
            self.weights[176],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_336, False)
        ttnn_add_104 = ttnn.add(
            ttnn_matmul_106,
            self._ce["ce_104_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_106, False)
        ttnn_add_105 = ttnn.add(
            ttnn_add_102,
            ttnn_add_104,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_104, False)
        ttnn.deallocate(ttnn_add_102, False)
        ttnn_layer_norm_37 = ttnn.layer_norm(
            ttnn_add_105,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[174],
            bias=self.weights[173],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_337 = ttnn.reshape(
            ttnn_layer_norm_37,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_37, False)
        ttnn_matmul_107 = ttnn.matmul(
            ttnn_reshape_337,
            self.weights[172],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_337, False)
        ttnn_add_106 = ttnn.add(
            ttnn_matmul_107,
            self._ce["ce_44_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_107, False)
        ttnn_gelu_17 = ttnn.gelu(
            ttnn_add_106,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_106, False)
        ttnn_reshape_338 = ttnn.reshape(
            ttnn_gelu_17,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_17, False)
        ttnn_matmul_108 = ttnn.matmul(
            ttnn_reshape_338,
            self.weights[170],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_338, False)
        ttnn_add_107 = ttnn.add(
            ttnn_matmul_108,
            self._ce["ce_115_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_108, False)
        ttnn_add_108 = ttnn.add(
            ttnn_add_105,
            ttnn_add_107,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_107, False)
        ttnn.deallocate(ttnn_add_105, False)
        ttnn_layer_norm_38 = ttnn.layer_norm(
            ttnn_add_108,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[168],
            bias=self.weights[167],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_339 = ttnn.reshape(
            ttnn_layer_norm_38,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_38, False)
        ttnn_matmul_109 = ttnn.matmul(
            ttnn_reshape_339,
            self._ce["ce_72_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_339, False)
        ttnn_add_109 = ttnn.add(
            ttnn_matmul_109,
            self._ce["ce_17_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_109, False)
        ttnn_slice_54 = ttnn.slice(
            ttnn_add_109,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_55 = ttnn.slice(
            ttnn_add_109,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_56 = ttnn.slice(
            ttnn_add_109,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_109, False)
        ttnn_reshape_340 = ttnn.reshape(
            ttnn_slice_54,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_54, False)
        ttnn_reshape_341 = ttnn.reshape(
            ttnn_slice_55,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_55, False)
        ttnn_reshape_342 = ttnn.reshape(
            ttnn_slice_56,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_56, False)
        ttnn_permute_96 = ttnn.permute(
            ttnn_reshape_340,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_340, False)
        ttnn_permute_97 = ttnn.permute(
            ttnn_reshape_341,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_341, False)
        ttnn_permute_98 = ttnn.permute(
            ttnn_reshape_342,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_342, False)
        ttnn_typecast_92 = ttnn.typecast(
            ttnn_permute_96,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_96, False)
        ttnn_multiply_37 = ttnn.multiply(
            ttnn_typecast_92,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_92, False)
        ttnn_typecast_93 = ttnn.typecast(
            ttnn_permute_97,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_97, False)
        ttnn_permute_99 = ttnn.permute(
            ttnn_typecast_93,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_93, False)
        ttnn_multiply_38 = ttnn.multiply(
            ttnn_permute_99,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_99, False)
        ttnn_matmul_110 = ttnn.matmul(
            ttnn_multiply_37,
            ttnn_multiply_38,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_38, False)
        ttnn.deallocate(ttnn_multiply_37, False)
        ttnn_eq_18 = ttnn.eq(
            ttnn_matmul_110,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_36 = ttnn.logical_not(
            ttnn_eq_18,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_18, False)
        ttnn_sum_18 = ttnn.sum(
            ttnn_logical_not_36,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_36, False)
        ttnn_ne_18 = ttnn.ne(
            ttnn_sum_18,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_18, False)
        ttnn_logical_not_37 = ttnn.logical_not(
            ttnn_ne_18,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_18, False)
        ttnn_reshape_343 = ttnn.reshape(
            ttnn_logical_not_37,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_37, False)
        ttnn_softmax_18 = ttnn.softmax(
            ttnn_matmul_110,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_110, False)
        ttnn_repeat_206 = ttnn.repeat(ttnn_reshape_343, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_343, False)
        ttnn_typecast_94 = ttnn.typecast(
            ttnn_repeat_206,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_206, False)
        ttnn_where_18 = ttnn.where(
            ttnn_typecast_94,
            self._ce["cez_4_0"],
            ttnn_softmax_18,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_94, False)
        ttnn.deallocate(ttnn_softmax_18, False)
        ttnn_typecast_95 = ttnn.typecast(
            ttnn_permute_98,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_98, False)
        ttnn_matmul_111 = ttnn.matmul(
            ttnn_where_18,
            ttnn_typecast_95,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_95, False)
        ttnn.deallocate(ttnn_where_18, False)
        ttnn_typecast_96 = ttnn.typecast(
            ttnn_matmul_111,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_111, False)
        ttnn_permute_100 = ttnn.permute(
            ttnn_typecast_96,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_96, False)
        ttnn_reshape_344 = ttnn.reshape(
            ttnn_permute_100,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_100, False)
        ttnn_matmul_112 = ttnn.matmul(
            ttnn_reshape_344,
            self.weights[164],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_344, False)
        ttnn_add_110 = ttnn.add(
            ttnn_matmul_112,
            self._ce["ce_31_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_112, False)
        ttnn_add_111 = ttnn.add(
            ttnn_add_108,
            ttnn_add_110,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_110, False)
        ttnn.deallocate(ttnn_add_108, False)
        ttnn_layer_norm_39 = ttnn.layer_norm(
            ttnn_add_111,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[162],
            bias=self.weights[161],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_345 = ttnn.reshape(
            ttnn_layer_norm_39,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_39, False)
        ttnn_matmul_113 = ttnn.matmul(
            ttnn_reshape_345,
            self.weights[160],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_345, False)
        ttnn_add_112 = ttnn.add(
            ttnn_matmul_113,
            self._ce["ce_83_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_113, False)
        ttnn_gelu_18 = ttnn.gelu(
            ttnn_add_112,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_112, False)
        ttnn_reshape_346 = ttnn.reshape(
            ttnn_gelu_18,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_18, False)
        ttnn_matmul_114 = ttnn.matmul(
            ttnn_reshape_346,
            self.weights[158],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_346, False)
        ttnn_add_113 = ttnn.add(
            ttnn_matmul_114,
            self._ce["ce_90_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_114, False)
        ttnn_add_114 = ttnn.add(
            ttnn_add_111,
            ttnn_add_113,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_113, False)
        ttnn.deallocate(ttnn_add_111, False)
        ttnn_layer_norm_40 = ttnn.layer_norm(
            ttnn_add_114,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[156],
            bias=self.weights[155],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_347 = ttnn.reshape(
            ttnn_layer_norm_40,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_40, False)
        ttnn_matmul_115 = ttnn.matmul(
            ttnn_reshape_347,
            self._ce["ce_47_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_347, False)
        ttnn_add_115 = ttnn.add(
            ttnn_matmul_115,
            self._ce["ce_54_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_115, False)
        ttnn_slice_57 = ttnn.slice(
            ttnn_add_115,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_58 = ttnn.slice(
            ttnn_add_115,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_59 = ttnn.slice(
            ttnn_add_115,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_115, False)
        ttnn_reshape_348 = ttnn.reshape(
            ttnn_slice_57,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_57, False)
        ttnn_reshape_349 = ttnn.reshape(
            ttnn_slice_58,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_58, False)
        ttnn_reshape_350 = ttnn.reshape(
            ttnn_slice_59,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_59, False)
        ttnn_permute_101 = ttnn.permute(
            ttnn_reshape_348,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_348, False)
        ttnn_permute_102 = ttnn.permute(
            ttnn_reshape_349,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_349, False)
        ttnn_permute_103 = ttnn.permute(
            ttnn_reshape_350,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_350, False)
        ttnn_typecast_97 = ttnn.typecast(
            ttnn_permute_101,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_101, False)
        ttnn_multiply_39 = ttnn.multiply(
            ttnn_typecast_97,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_97, False)
        ttnn_typecast_98 = ttnn.typecast(
            ttnn_permute_102,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_102, False)
        ttnn_permute_104 = ttnn.permute(
            ttnn_typecast_98,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_98, False)
        ttnn_multiply_40 = ttnn.multiply(
            ttnn_permute_104,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_104, False)
        ttnn_matmul_116 = ttnn.matmul(
            ttnn_multiply_39,
            ttnn_multiply_40,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_40, False)
        ttnn.deallocate(ttnn_multiply_39, False)
        ttnn_eq_19 = ttnn.eq(
            ttnn_matmul_116,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_38 = ttnn.logical_not(
            ttnn_eq_19,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_19, False)
        ttnn_sum_19 = ttnn.sum(
            ttnn_logical_not_38,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_38, False)
        ttnn_ne_19 = ttnn.ne(
            ttnn_sum_19,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_19, False)
        ttnn_logical_not_39 = ttnn.logical_not(
            ttnn_ne_19,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_19, False)
        ttnn_reshape_351 = ttnn.reshape(
            ttnn_logical_not_39,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_39, False)
        ttnn_softmax_19 = ttnn.softmax(
            ttnn_matmul_116,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_116, False)
        ttnn_repeat_207 = ttnn.repeat(ttnn_reshape_351, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_351, False)
        ttnn_typecast_99 = ttnn.typecast(
            ttnn_repeat_207,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_207, False)
        ttnn_where_19 = ttnn.where(
            ttnn_typecast_99,
            self._ce["cez_4_0"],
            ttnn_softmax_19,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_99, False)
        ttnn.deallocate(ttnn_softmax_19, False)
        ttnn_typecast_100 = ttnn.typecast(
            ttnn_permute_103,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_103, False)
        ttnn_matmul_117 = ttnn.matmul(
            ttnn_where_19,
            ttnn_typecast_100,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_100, False)
        ttnn.deallocate(ttnn_where_19, False)
        ttnn_typecast_101 = ttnn.typecast(
            ttnn_matmul_117,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_117, False)
        ttnn_permute_105 = ttnn.permute(
            ttnn_typecast_101,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_101, False)
        ttnn_reshape_352 = ttnn.reshape(
            ttnn_permute_105,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_105, False)
        ttnn_matmul_118 = ttnn.matmul(
            ttnn_reshape_352,
            self.weights[152],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_352, False)
        ttnn_add_116 = ttnn.add(
            ttnn_matmul_118,
            self._ce["ce_34_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_118, False)
        ttnn_add_117 = ttnn.add(
            ttnn_add_114,
            ttnn_add_116,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_116, False)
        ttnn.deallocate(ttnn_add_114, False)
        ttnn_layer_norm_41 = ttnn.layer_norm(
            ttnn_add_117,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[150],
            bias=self.weights[149],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_353 = ttnn.reshape(
            ttnn_layer_norm_41,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_41, False)
        ttnn_matmul_119 = ttnn.matmul(
            ttnn_reshape_353,
            self.weights[148],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_353, False)
        ttnn_add_118 = ttnn.add(
            ttnn_matmul_119,
            self._ce["ce_158_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_119, False)
        ttnn_gelu_19 = ttnn.gelu(
            ttnn_add_118,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_118, False)
        ttnn_reshape_354 = ttnn.reshape(
            ttnn_gelu_19,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_19, False)
        ttnn_matmul_120 = ttnn.matmul(
            ttnn_reshape_354,
            self.weights[146],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_354, False)
        ttnn_add_119 = ttnn.add(
            ttnn_matmul_120,
            self._ce["ce_113_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_120, False)
        ttnn_add_120 = ttnn.add(
            ttnn_add_117,
            ttnn_add_119,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_119, False)
        ttnn.deallocate(ttnn_add_117, False)
        ttnn_layer_norm_42 = ttnn.layer_norm(
            ttnn_add_120,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[144],
            bias=self.weights[143],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_355 = ttnn.reshape(
            ttnn_layer_norm_42,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_42, False)
        ttnn_matmul_121 = ttnn.matmul(
            ttnn_reshape_355,
            self._ce["ce_102_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_355, False)
        ttnn_add_121 = ttnn.add(
            ttnn_matmul_121,
            self._ce["ce_154_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_121, False)
        ttnn_slice_60 = ttnn.slice(
            ttnn_add_121,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_61 = ttnn.slice(
            ttnn_add_121,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_62 = ttnn.slice(
            ttnn_add_121,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_121, False)
        ttnn_reshape_356 = ttnn.reshape(
            ttnn_slice_60,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_60, False)
        ttnn_reshape_357 = ttnn.reshape(
            ttnn_slice_61,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_61, False)
        ttnn_reshape_358 = ttnn.reshape(
            ttnn_slice_62,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_62, False)
        ttnn_permute_106 = ttnn.permute(
            ttnn_reshape_356,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_356, False)
        ttnn_permute_107 = ttnn.permute(
            ttnn_reshape_357,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_357, False)
        ttnn_permute_108 = ttnn.permute(
            ttnn_reshape_358,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_358, False)
        ttnn_typecast_102 = ttnn.typecast(
            ttnn_permute_106,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_106, False)
        ttnn_multiply_41 = ttnn.multiply(
            ttnn_typecast_102,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_102, False)
        ttnn_typecast_103 = ttnn.typecast(
            ttnn_permute_107,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_107, False)
        ttnn_permute_109 = ttnn.permute(
            ttnn_typecast_103,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_103, False)
        ttnn_multiply_42 = ttnn.multiply(
            ttnn_permute_109,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_109, False)
        ttnn_matmul_122 = ttnn.matmul(
            ttnn_multiply_41,
            ttnn_multiply_42,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_42, False)
        ttnn.deallocate(ttnn_multiply_41, False)
        ttnn_eq_20 = ttnn.eq(
            ttnn_matmul_122,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_40 = ttnn.logical_not(
            ttnn_eq_20,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_20, False)
        ttnn_sum_20 = ttnn.sum(
            ttnn_logical_not_40,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_40, False)
        ttnn_ne_20 = ttnn.ne(
            ttnn_sum_20,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_20, False)
        ttnn_logical_not_41 = ttnn.logical_not(
            ttnn_ne_20,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_20, False)
        ttnn_reshape_359 = ttnn.reshape(
            ttnn_logical_not_41,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_41, False)
        ttnn_softmax_20 = ttnn.softmax(
            ttnn_matmul_122,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_122, False)
        ttnn_repeat_208 = ttnn.repeat(ttnn_reshape_359, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_359, False)
        ttnn_typecast_104 = ttnn.typecast(
            ttnn_repeat_208,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_208, False)
        ttnn_where_20 = ttnn.where(
            ttnn_typecast_104,
            self._ce["cez_4_0"],
            ttnn_softmax_20,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_104, False)
        ttnn.deallocate(ttnn_softmax_20, False)
        ttnn_typecast_105 = ttnn.typecast(
            ttnn_permute_108,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_108, False)
        ttnn_matmul_123 = ttnn.matmul(
            ttnn_where_20,
            ttnn_typecast_105,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_105, False)
        ttnn.deallocate(ttnn_where_20, False)
        ttnn_typecast_106 = ttnn.typecast(
            ttnn_matmul_123,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_123, False)
        ttnn_permute_110 = ttnn.permute(
            ttnn_typecast_106,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_106, False)
        ttnn_reshape_360 = ttnn.reshape(
            ttnn_permute_110,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_110, False)
        ttnn_matmul_124 = ttnn.matmul(
            ttnn_reshape_360,
            self.weights[140],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_360, False)
        ttnn_add_122 = ttnn.add(
            ttnn_matmul_124,
            self._ce["ce_85_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_124, False)
        ttnn_add_123 = ttnn.add(
            ttnn_add_120,
            ttnn_add_122,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_122, False)
        ttnn.deallocate(ttnn_add_120, False)
        ttnn_layer_norm_43 = ttnn.layer_norm(
            ttnn_add_123,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[138],
            bias=self.weights[137],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_361 = ttnn.reshape(
            ttnn_layer_norm_43,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_43, False)
        ttnn_matmul_125 = ttnn.matmul(
            ttnn_reshape_361,
            self.weights[136],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_361, False)
        ttnn_add_124 = ttnn.add(
            ttnn_matmul_125,
            self._ce["ce_114_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_125, False)
        ttnn_gelu_20 = ttnn.gelu(
            ttnn_add_124,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_124, False)
        ttnn_reshape_362 = ttnn.reshape(
            ttnn_gelu_20,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_20, False)
        ttnn_matmul_126 = ttnn.matmul(
            ttnn_reshape_362,
            self.weights[134],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_362, False)
        ttnn_add_125 = ttnn.add(
            ttnn_matmul_126,
            self._ce["ce_36_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_126, False)
        ttnn_add_126 = ttnn.add(
            ttnn_add_123,
            ttnn_add_125,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_125, False)
        ttnn.deallocate(ttnn_add_123, False)
        ttnn_layer_norm_44 = ttnn.layer_norm(
            ttnn_add_126,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[132],
            bias=self.weights[131],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_363 = ttnn.reshape(
            ttnn_layer_norm_44,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_44, False)
        ttnn_matmul_127 = ttnn.matmul(
            ttnn_reshape_363,
            self._ce["ce_55_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_363, False)
        ttnn_add_127 = ttnn.add(
            ttnn_matmul_127,
            self._ce["ce_127_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_127, False)
        ttnn_slice_63 = ttnn.slice(
            ttnn_add_127,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_64 = ttnn.slice(
            ttnn_add_127,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_65 = ttnn.slice(
            ttnn_add_127,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_127, False)
        ttnn_reshape_364 = ttnn.reshape(
            ttnn_slice_63,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_63, False)
        ttnn_reshape_365 = ttnn.reshape(
            ttnn_slice_64,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_64, False)
        ttnn_reshape_366 = ttnn.reshape(
            ttnn_slice_65,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_65, False)
        ttnn_permute_111 = ttnn.permute(
            ttnn_reshape_364,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_364, False)
        ttnn_permute_112 = ttnn.permute(
            ttnn_reshape_365,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_365, False)
        ttnn_permute_113 = ttnn.permute(
            ttnn_reshape_366,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_366, False)
        ttnn_typecast_107 = ttnn.typecast(
            ttnn_permute_111,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_111, False)
        ttnn_multiply_43 = ttnn.multiply(
            ttnn_typecast_107,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_107, False)
        ttnn_typecast_108 = ttnn.typecast(
            ttnn_permute_112,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_112, False)
        ttnn_permute_114 = ttnn.permute(
            ttnn_typecast_108,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_108, False)
        ttnn_multiply_44 = ttnn.multiply(
            ttnn_permute_114,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_114, False)
        ttnn_matmul_128 = ttnn.matmul(
            ttnn_multiply_43,
            ttnn_multiply_44,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_44, False)
        ttnn.deallocate(ttnn_multiply_43, False)
        ttnn_eq_21 = ttnn.eq(
            ttnn_matmul_128,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_42 = ttnn.logical_not(
            ttnn_eq_21,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_21, False)
        ttnn_sum_21 = ttnn.sum(
            ttnn_logical_not_42,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_42, False)
        ttnn_ne_21 = ttnn.ne(
            ttnn_sum_21,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_21, False)
        ttnn_logical_not_43 = ttnn.logical_not(
            ttnn_ne_21,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_21, False)
        ttnn_reshape_367 = ttnn.reshape(
            ttnn_logical_not_43,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_43, False)
        ttnn_softmax_21 = ttnn.softmax(
            ttnn_matmul_128,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_128, False)
        ttnn_repeat_209 = ttnn.repeat(ttnn_reshape_367, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_367, False)
        ttnn_typecast_109 = ttnn.typecast(
            ttnn_repeat_209,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_209, False)
        ttnn_where_21 = ttnn.where(
            ttnn_typecast_109,
            self._ce["cez_4_0"],
            ttnn_softmax_21,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_109, False)
        ttnn.deallocate(ttnn_softmax_21, False)
        ttnn_typecast_110 = ttnn.typecast(
            ttnn_permute_113,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_113, False)
        ttnn_matmul_129 = ttnn.matmul(
            ttnn_where_21,
            ttnn_typecast_110,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_110, False)
        ttnn.deallocate(ttnn_where_21, False)
        ttnn_typecast_111 = ttnn.typecast(
            ttnn_matmul_129,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_129, False)
        ttnn_permute_115 = ttnn.permute(
            ttnn_typecast_111,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_111, False)
        ttnn_reshape_368 = ttnn.reshape(
            ttnn_permute_115,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_115, False)
        ttnn_matmul_130 = ttnn.matmul(
            ttnn_reshape_368,
            self.weights[128],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_368, False)
        ttnn_add_128 = ttnn.add(
            ttnn_matmul_130,
            self._ce["ce_68_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_130, False)
        ttnn_add_129 = ttnn.add(
            ttnn_add_126,
            ttnn_add_128,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_128, False)
        ttnn.deallocate(ttnn_add_126, False)
        ttnn_layer_norm_45 = ttnn.layer_norm(
            ttnn_add_129,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[126],
            bias=self.weights[125],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_369 = ttnn.reshape(
            ttnn_layer_norm_45,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_45, False)
        ttnn_matmul_131 = ttnn.matmul(
            ttnn_reshape_369,
            self.weights[124],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_369, False)
        ttnn_add_130 = ttnn.add(
            ttnn_matmul_131,
            self._ce["ce_56_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_131, False)
        ttnn_gelu_21 = ttnn.gelu(
            ttnn_add_130,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_130, False)
        ttnn_reshape_370 = ttnn.reshape(
            ttnn_gelu_21,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_21, False)
        ttnn_matmul_132 = ttnn.matmul(
            ttnn_reshape_370,
            self.weights[122],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_370, False)
        ttnn_add_131 = ttnn.add(
            ttnn_matmul_132,
            self._ce["ce_133_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_132, False)
        ttnn_add_132 = ttnn.add(
            ttnn_add_129,
            ttnn_add_131,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_131, False)
        ttnn.deallocate(ttnn_add_129, False)
        ttnn_layer_norm_46 = ttnn.layer_norm(
            ttnn_add_132,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[120],
            bias=self.weights[119],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_371 = ttnn.reshape(
            ttnn_layer_norm_46,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_46, False)
        ttnn_matmul_133 = ttnn.matmul(
            ttnn_reshape_371,
            self._ce["ce_5_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_371, False)
        ttnn_add_133 = ttnn.add(
            ttnn_matmul_133,
            self._ce["ce_32_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_133, False)
        ttnn_slice_66 = ttnn.slice(
            ttnn_add_133,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_67 = ttnn.slice(
            ttnn_add_133,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_68 = ttnn.slice(
            ttnn_add_133,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_133, False)
        ttnn_reshape_372 = ttnn.reshape(
            ttnn_slice_66,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_66, False)
        ttnn_reshape_373 = ttnn.reshape(
            ttnn_slice_67,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_67, False)
        ttnn_reshape_374 = ttnn.reshape(
            ttnn_slice_68,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_68, False)
        ttnn_permute_116 = ttnn.permute(
            ttnn_reshape_372,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_372, False)
        ttnn_permute_117 = ttnn.permute(
            ttnn_reshape_373,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_373, False)
        ttnn_permute_118 = ttnn.permute(
            ttnn_reshape_374,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_374, False)
        ttnn_typecast_112 = ttnn.typecast(
            ttnn_permute_116,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_116, False)
        ttnn_multiply_45 = ttnn.multiply(
            ttnn_typecast_112,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_112, False)
        ttnn_typecast_113 = ttnn.typecast(
            ttnn_permute_117,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_117, False)
        ttnn_permute_119 = ttnn.permute(
            ttnn_typecast_113,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_113, False)
        ttnn_multiply_46 = ttnn.multiply(
            ttnn_permute_119,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_119, False)
        ttnn_matmul_134 = ttnn.matmul(
            ttnn_multiply_45,
            ttnn_multiply_46,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_46, False)
        ttnn.deallocate(ttnn_multiply_45, False)
        ttnn_eq_22 = ttnn.eq(
            ttnn_matmul_134,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_44 = ttnn.logical_not(
            ttnn_eq_22,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_22, False)
        ttnn_sum_22 = ttnn.sum(
            ttnn_logical_not_44,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_44, False)
        ttnn_ne_22 = ttnn.ne(
            ttnn_sum_22,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_22, False)
        ttnn_logical_not_45 = ttnn.logical_not(
            ttnn_ne_22,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_22, False)
        ttnn_reshape_375 = ttnn.reshape(
            ttnn_logical_not_45,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_45, False)
        ttnn_softmax_22 = ttnn.softmax(
            ttnn_matmul_134,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_134, False)
        ttnn_repeat_210 = ttnn.repeat(ttnn_reshape_375, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_375, False)
        ttnn_typecast_114 = ttnn.typecast(
            ttnn_repeat_210,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_210, False)
        ttnn_where_22 = ttnn.where(
            ttnn_typecast_114,
            self._ce["cez_4_0"],
            ttnn_softmax_22,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_114, False)
        ttnn.deallocate(ttnn_softmax_22, False)
        ttnn_typecast_115 = ttnn.typecast(
            ttnn_permute_118,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_118, False)
        ttnn_matmul_135 = ttnn.matmul(
            ttnn_where_22,
            ttnn_typecast_115,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_115, False)
        ttnn.deallocate(ttnn_where_22, False)
        ttnn_typecast_116 = ttnn.typecast(
            ttnn_matmul_135,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_135, False)
        ttnn_permute_120 = ttnn.permute(
            ttnn_typecast_116,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_116, False)
        ttnn_reshape_376 = ttnn.reshape(
            ttnn_permute_120,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_120, False)
        ttnn_matmul_136 = ttnn.matmul(
            ttnn_reshape_376,
            self.weights[116],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_376, False)
        ttnn_add_134 = ttnn.add(
            ttnn_matmul_136,
            self._ce["ce_135_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_136, False)
        ttnn_add_135 = ttnn.add(
            ttnn_add_132,
            ttnn_add_134,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_134, False)
        ttnn.deallocate(ttnn_add_132, False)
        ttnn_layer_norm_47 = ttnn.layer_norm(
            ttnn_add_135,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[114],
            bias=self.weights[113],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_377 = ttnn.reshape(
            ttnn_layer_norm_47,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_47, False)
        ttnn_matmul_137 = ttnn.matmul(
            ttnn_reshape_377,
            self.weights[112],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_377, False)
        ttnn_add_136 = ttnn.add(
            ttnn_matmul_137,
            self._ce["ce_150_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_137, False)
        ttnn_gelu_22 = ttnn.gelu(
            ttnn_add_136,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_136, False)
        ttnn_reshape_378 = ttnn.reshape(
            ttnn_gelu_22,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_22, False)
        ttnn_matmul_138 = ttnn.matmul(
            ttnn_reshape_378,
            self.weights[110],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_378, False)
        ttnn_add_137 = ttnn.add(
            ttnn_matmul_138,
            self._ce["ce_155_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_138, False)
        ttnn_add_138 = ttnn.add(
            ttnn_add_135,
            ttnn_add_137,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_137, False)
        ttnn.deallocate(ttnn_add_135, False)
        ttnn_layer_norm_48 = ttnn.layer_norm(
            ttnn_add_138,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[108],
            bias=self.weights[107],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_379 = ttnn.reshape(
            ttnn_layer_norm_48,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_48, False)
        ttnn_matmul_139 = ttnn.matmul(
            ttnn_reshape_379,
            self._ce["ce_16_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_379, False)
        ttnn_add_139 = ttnn.add(
            ttnn_matmul_139,
            self._ce["ce_103_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_139, False)
        ttnn_slice_69 = ttnn.slice(
            ttnn_add_139,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_70 = ttnn.slice(
            ttnn_add_139,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_71 = ttnn.slice(
            ttnn_add_139,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_139, False)
        ttnn_reshape_380 = ttnn.reshape(
            ttnn_slice_69,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_69, False)
        ttnn_reshape_381 = ttnn.reshape(
            ttnn_slice_70,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_70, False)
        ttnn_reshape_382 = ttnn.reshape(
            ttnn_slice_71,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_71, False)
        ttnn_permute_121 = ttnn.permute(
            ttnn_reshape_380,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_380, False)
        ttnn_permute_122 = ttnn.permute(
            ttnn_reshape_381,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_381, False)
        ttnn_permute_123 = ttnn.permute(
            ttnn_reshape_382,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_382, False)
        ttnn_typecast_117 = ttnn.typecast(
            ttnn_permute_121,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_121, False)
        ttnn_multiply_47 = ttnn.multiply(
            ttnn_typecast_117,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_117, False)
        ttnn_typecast_118 = ttnn.typecast(
            ttnn_permute_122,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_122, False)
        ttnn_permute_124 = ttnn.permute(
            ttnn_typecast_118,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_118, False)
        ttnn_multiply_48 = ttnn.multiply(
            ttnn_permute_124,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_124, False)
        ttnn_matmul_140 = ttnn.matmul(
            ttnn_multiply_47,
            ttnn_multiply_48,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_48, False)
        ttnn.deallocate(ttnn_multiply_47, False)
        ttnn_eq_23 = ttnn.eq(
            ttnn_matmul_140,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_46 = ttnn.logical_not(
            ttnn_eq_23,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_23, False)
        ttnn_sum_23 = ttnn.sum(
            ttnn_logical_not_46,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_46, False)
        ttnn_ne_23 = ttnn.ne(
            ttnn_sum_23,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_23, False)
        ttnn_logical_not_47 = ttnn.logical_not(
            ttnn_ne_23,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_23, False)
        ttnn_reshape_383 = ttnn.reshape(
            ttnn_logical_not_47,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_47, False)
        ttnn_softmax_23 = ttnn.softmax(
            ttnn_matmul_140,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_140, False)
        ttnn_repeat_211 = ttnn.repeat(ttnn_reshape_383, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_383, False)
        ttnn_typecast_119 = ttnn.typecast(
            ttnn_repeat_211,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_211, False)
        ttnn_where_23 = ttnn.where(
            ttnn_typecast_119,
            self._ce["cez_4_0"],
            ttnn_softmax_23,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_119, False)
        ttnn.deallocate(ttnn_softmax_23, False)
        ttnn_typecast_120 = ttnn.typecast(
            ttnn_permute_123,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_123, False)
        ttnn_matmul_141 = ttnn.matmul(
            ttnn_where_23,
            ttnn_typecast_120,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_120, False)
        ttnn.deallocate(ttnn_where_23, False)
        ttnn_typecast_121 = ttnn.typecast(
            ttnn_matmul_141,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_141, False)
        ttnn_permute_125 = ttnn.permute(
            ttnn_typecast_121,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_121, False)
        ttnn_reshape_384 = ttnn.reshape(
            ttnn_permute_125,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_125, False)
        ttnn_matmul_142 = ttnn.matmul(
            ttnn_reshape_384,
            self.weights[104],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_384, False)
        ttnn_add_140 = ttnn.add(
            ttnn_matmul_142,
            self._ce["ce_22_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_142, False)
        ttnn_add_141 = ttnn.add(
            ttnn_add_138,
            ttnn_add_140,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_140, False)
        ttnn.deallocate(ttnn_add_138, False)
        ttnn_layer_norm_49 = ttnn.layer_norm(
            ttnn_add_141,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[102],
            bias=self.weights[101],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_385 = ttnn.reshape(
            ttnn_layer_norm_49,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_49, False)
        ttnn_matmul_143 = ttnn.matmul(
            ttnn_reshape_385,
            self.weights[100],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_385, False)
        ttnn_add_142 = ttnn.add(
            ttnn_matmul_143,
            self._ce["ce_60_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_143, False)
        ttnn_gelu_23 = ttnn.gelu(
            ttnn_add_142,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_142, False)
        ttnn_reshape_386 = ttnn.reshape(
            ttnn_gelu_23,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_23, False)
        ttnn_matmul_144 = ttnn.matmul(
            ttnn_reshape_386,
            self.weights[98],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_386, False)
        ttnn_add_143 = ttnn.add(
            ttnn_matmul_144,
            self._ce["ce_140_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_144, False)
        ttnn_add_144 = ttnn.add(
            ttnn_add_141,
            ttnn_add_143,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_143, False)
        ttnn.deallocate(ttnn_add_141, False)
        ttnn_layer_norm_50 = ttnn.layer_norm(
            ttnn_add_144,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[96],
            bias=self.weights[95],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_387 = ttnn.reshape(
            ttnn_layer_norm_50,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_50, False)
        ttnn_matmul_145 = ttnn.matmul(
            ttnn_reshape_387,
            self._ce["ce_42_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_387, False)
        ttnn_add_145 = ttnn.add(
            ttnn_matmul_145,
            self._ce["ce_26_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_145, False)
        ttnn_slice_72 = ttnn.slice(
            ttnn_add_145,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_73 = ttnn.slice(
            ttnn_add_145,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_74 = ttnn.slice(
            ttnn_add_145,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_145, False)
        ttnn_reshape_388 = ttnn.reshape(
            ttnn_slice_72,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_72, False)
        ttnn_reshape_389 = ttnn.reshape(
            ttnn_slice_73,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_73, False)
        ttnn_reshape_390 = ttnn.reshape(
            ttnn_slice_74,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_74, False)
        ttnn_permute_126 = ttnn.permute(
            ttnn_reshape_388,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_388, False)
        ttnn_permute_127 = ttnn.permute(
            ttnn_reshape_389,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_389, False)
        ttnn_permute_128 = ttnn.permute(
            ttnn_reshape_390,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_390, False)
        ttnn_typecast_122 = ttnn.typecast(
            ttnn_permute_126,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_126, False)
        ttnn_multiply_49 = ttnn.multiply(
            ttnn_typecast_122,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_122, False)
        ttnn_typecast_123 = ttnn.typecast(
            ttnn_permute_127,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_127, False)
        ttnn_permute_129 = ttnn.permute(
            ttnn_typecast_123,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_123, False)
        ttnn_multiply_50 = ttnn.multiply(
            ttnn_permute_129,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_129, False)
        ttnn_matmul_146 = ttnn.matmul(
            ttnn_multiply_49,
            ttnn_multiply_50,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_50, False)
        ttnn.deallocate(ttnn_multiply_49, False)
        ttnn_eq_24 = ttnn.eq(
            ttnn_matmul_146,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_48 = ttnn.logical_not(
            ttnn_eq_24,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_24, False)
        ttnn_sum_24 = ttnn.sum(
            ttnn_logical_not_48,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_48, False)
        ttnn_ne_24 = ttnn.ne(
            ttnn_sum_24,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_24, False)
        ttnn_logical_not_49 = ttnn.logical_not(
            ttnn_ne_24,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_24, False)
        ttnn_reshape_391 = ttnn.reshape(
            ttnn_logical_not_49,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_49, False)
        ttnn_softmax_24 = ttnn.softmax(
            ttnn_matmul_146,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_146, False)
        ttnn_repeat_212 = ttnn.repeat(ttnn_reshape_391, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_391, False)
        ttnn_typecast_124 = ttnn.typecast(
            ttnn_repeat_212,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_212, False)
        ttnn_where_24 = ttnn.where(
            ttnn_typecast_124,
            self._ce["cez_4_0"],
            ttnn_softmax_24,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_124, False)
        ttnn.deallocate(ttnn_softmax_24, False)
        ttnn_typecast_125 = ttnn.typecast(
            ttnn_permute_128,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_128, False)
        ttnn_matmul_147 = ttnn.matmul(
            ttnn_where_24,
            ttnn_typecast_125,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_125, False)
        ttnn.deallocate(ttnn_where_24, False)
        ttnn_typecast_126 = ttnn.typecast(
            ttnn_matmul_147,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_147, False)
        ttnn_permute_130 = ttnn.permute(
            ttnn_typecast_126,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_126, False)
        ttnn_reshape_392 = ttnn.reshape(
            ttnn_permute_130,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_130, False)
        ttnn_matmul_148 = ttnn.matmul(
            ttnn_reshape_392,
            self.weights[92],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_392, False)
        ttnn_add_146 = ttnn.add(
            ttnn_matmul_148,
            self._ce["ce_121_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_148, False)
        ttnn_add_147 = ttnn.add(
            ttnn_add_144,
            ttnn_add_146,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_146, False)
        ttnn.deallocate(ttnn_add_144, False)
        ttnn_layer_norm_51 = ttnn.layer_norm(
            ttnn_add_147,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[90],
            bias=self.weights[89],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_393 = ttnn.reshape(
            ttnn_layer_norm_51,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_51, False)
        ttnn_matmul_149 = ttnn.matmul(
            ttnn_reshape_393,
            self.weights[88],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_393, False)
        ttnn_add_148 = ttnn.add(
            ttnn_matmul_149,
            self._ce["ce_108_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_149, False)
        ttnn_gelu_24 = ttnn.gelu(
            ttnn_add_148,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_148, False)
        ttnn_reshape_394 = ttnn.reshape(
            ttnn_gelu_24,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_24, False)
        ttnn_matmul_150 = ttnn.matmul(
            ttnn_reshape_394,
            self.weights[86],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_394, False)
        ttnn_add_149 = ttnn.add(
            ttnn_matmul_150,
            self._ce["ce_80_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_150, False)
        ttnn_add_150 = ttnn.add(
            ttnn_add_147,
            ttnn_add_149,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_149, False)
        ttnn.deallocate(ttnn_add_147, False)
        ttnn_layer_norm_52 = ttnn.layer_norm(
            ttnn_add_150,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[84],
            bias=self.weights[83],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_395 = ttnn.reshape(
            ttnn_layer_norm_52,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_52, False)
        ttnn_matmul_151 = ttnn.matmul(
            ttnn_reshape_395,
            self._ce["ce_119_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_395, False)
        ttnn_add_151 = ttnn.add(
            ttnn_matmul_151,
            self._ce["ce_30_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_151, False)
        ttnn_slice_75 = ttnn.slice(
            ttnn_add_151,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_76 = ttnn.slice(
            ttnn_add_151,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_77 = ttnn.slice(
            ttnn_add_151,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_151, False)
        ttnn_reshape_396 = ttnn.reshape(
            ttnn_slice_75,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_75, False)
        ttnn_reshape_397 = ttnn.reshape(
            ttnn_slice_76,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_76, False)
        ttnn_reshape_398 = ttnn.reshape(
            ttnn_slice_77,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_77, False)
        ttnn_permute_131 = ttnn.permute(
            ttnn_reshape_396,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_396, False)
        ttnn_permute_132 = ttnn.permute(
            ttnn_reshape_397,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_397, False)
        ttnn_permute_133 = ttnn.permute(
            ttnn_reshape_398,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_398, False)
        ttnn_typecast_127 = ttnn.typecast(
            ttnn_permute_131,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_131, False)
        ttnn_multiply_51 = ttnn.multiply(
            ttnn_typecast_127,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_127, False)
        ttnn_typecast_128 = ttnn.typecast(
            ttnn_permute_132,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_132, False)
        ttnn_permute_134 = ttnn.permute(
            ttnn_typecast_128,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_128, False)
        ttnn_multiply_52 = ttnn.multiply(
            ttnn_permute_134,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_134, False)
        ttnn_matmul_152 = ttnn.matmul(
            ttnn_multiply_51,
            ttnn_multiply_52,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_52, False)
        ttnn.deallocate(ttnn_multiply_51, False)
        ttnn_eq_25 = ttnn.eq(
            ttnn_matmul_152,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_50 = ttnn.logical_not(
            ttnn_eq_25,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_25, False)
        ttnn_sum_25 = ttnn.sum(
            ttnn_logical_not_50,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_50, False)
        ttnn_ne_25 = ttnn.ne(
            ttnn_sum_25,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_25, False)
        ttnn_logical_not_51 = ttnn.logical_not(
            ttnn_ne_25,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_25, False)
        ttnn_reshape_399 = ttnn.reshape(
            ttnn_logical_not_51,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_51, False)
        ttnn_softmax_25 = ttnn.softmax(
            ttnn_matmul_152,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_152, False)
        ttnn_repeat_213 = ttnn.repeat(ttnn_reshape_399, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_399, False)
        ttnn_typecast_129 = ttnn.typecast(
            ttnn_repeat_213,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_213, False)
        ttnn_where_25 = ttnn.where(
            ttnn_typecast_129,
            self._ce["cez_4_0"],
            ttnn_softmax_25,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_129, False)
        ttnn.deallocate(ttnn_softmax_25, False)
        ttnn_typecast_130 = ttnn.typecast(
            ttnn_permute_133,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_133, False)
        ttnn_matmul_153 = ttnn.matmul(
            ttnn_where_25,
            ttnn_typecast_130,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_130, False)
        ttnn.deallocate(ttnn_where_25, False)
        ttnn_typecast_131 = ttnn.typecast(
            ttnn_matmul_153,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_153, False)
        ttnn_permute_135 = ttnn.permute(
            ttnn_typecast_131,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_131, False)
        ttnn_reshape_400 = ttnn.reshape(
            ttnn_permute_135,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_135, False)
        ttnn_matmul_154 = ttnn.matmul(
            ttnn_reshape_400,
            self.weights[80],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_400, False)
        ttnn_add_152 = ttnn.add(
            ttnn_matmul_154,
            self._ce["ce_35_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_154, False)
        ttnn_add_153 = ttnn.add(
            ttnn_add_150,
            ttnn_add_152,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_152, False)
        ttnn.deallocate(ttnn_add_150, False)
        ttnn_layer_norm_53 = ttnn.layer_norm(
            ttnn_add_153,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[78],
            bias=self.weights[77],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_401 = ttnn.reshape(
            ttnn_layer_norm_53,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_53, False)
        ttnn_matmul_155 = ttnn.matmul(
            ttnn_reshape_401,
            self.weights[76],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_401, False)
        ttnn_add_154 = ttnn.add(
            ttnn_matmul_155,
            self._ce["ce_79_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_155, False)
        ttnn_gelu_25 = ttnn.gelu(
            ttnn_add_154,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_154, False)
        ttnn_reshape_402 = ttnn.reshape(
            ttnn_gelu_25,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_25, False)
        ttnn_matmul_156 = ttnn.matmul(
            ttnn_reshape_402,
            self.weights[74],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_402, False)
        ttnn_add_155 = ttnn.add(
            ttnn_matmul_156,
            self._ce["ce_117_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_156, False)
        ttnn_add_156 = ttnn.add(
            ttnn_add_153,
            ttnn_add_155,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_155, False)
        ttnn.deallocate(ttnn_add_153, False)
        ttnn_layer_norm_54 = ttnn.layer_norm(
            ttnn_add_156,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[72],
            bias=self.weights[71],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_403 = ttnn.reshape(
            ttnn_layer_norm_54,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_54, False)
        ttnn_matmul_157 = ttnn.matmul(
            ttnn_reshape_403,
            self._ce["ce_93_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_403, False)
        ttnn_add_157 = ttnn.add(
            ttnn_matmul_157,
            self._ce["ce_28_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_157, False)
        ttnn_slice_78 = ttnn.slice(
            ttnn_add_157,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_79 = ttnn.slice(
            ttnn_add_157,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_80 = ttnn.slice(
            ttnn_add_157,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_157, False)
        ttnn_reshape_404 = ttnn.reshape(
            ttnn_slice_78,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_78, False)
        ttnn_reshape_405 = ttnn.reshape(
            ttnn_slice_79,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_79, False)
        ttnn_reshape_406 = ttnn.reshape(
            ttnn_slice_80,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_80, False)
        ttnn_permute_136 = ttnn.permute(
            ttnn_reshape_404,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_404, False)
        ttnn_permute_137 = ttnn.permute(
            ttnn_reshape_405,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_405, False)
        ttnn_permute_138 = ttnn.permute(
            ttnn_reshape_406,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_406, False)
        ttnn_typecast_132 = ttnn.typecast(
            ttnn_permute_136,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_136, False)
        ttnn_multiply_53 = ttnn.multiply(
            ttnn_typecast_132,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_132, False)
        ttnn_typecast_133 = ttnn.typecast(
            ttnn_permute_137,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_137, False)
        ttnn_permute_139 = ttnn.permute(
            ttnn_typecast_133,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_133, False)
        ttnn_multiply_54 = ttnn.multiply(
            ttnn_permute_139,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_139, False)
        ttnn_matmul_158 = ttnn.matmul(
            ttnn_multiply_53,
            ttnn_multiply_54,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_54, False)
        ttnn.deallocate(ttnn_multiply_53, False)
        ttnn_eq_26 = ttnn.eq(
            ttnn_matmul_158,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_52 = ttnn.logical_not(
            ttnn_eq_26,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_26, False)
        ttnn_sum_26 = ttnn.sum(
            ttnn_logical_not_52,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_52, False)
        ttnn_ne_26 = ttnn.ne(
            ttnn_sum_26,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_26, False)
        ttnn_logical_not_53 = ttnn.logical_not(
            ttnn_ne_26,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_26, False)
        ttnn_reshape_407 = ttnn.reshape(
            ttnn_logical_not_53,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_53, False)
        ttnn_softmax_26 = ttnn.softmax(
            ttnn_matmul_158,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_158, False)
        ttnn_repeat_214 = ttnn.repeat(ttnn_reshape_407, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_407, False)
        ttnn_typecast_134 = ttnn.typecast(
            ttnn_repeat_214,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_214, False)
        ttnn_where_26 = ttnn.where(
            ttnn_typecast_134,
            self._ce["cez_4_0"],
            ttnn_softmax_26,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_134, False)
        ttnn.deallocate(ttnn_softmax_26, False)
        ttnn_typecast_135 = ttnn.typecast(
            ttnn_permute_138,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_138, False)
        ttnn_matmul_159 = ttnn.matmul(
            ttnn_where_26,
            ttnn_typecast_135,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_135, False)
        ttnn.deallocate(ttnn_where_26, False)
        ttnn_typecast_136 = ttnn.typecast(
            ttnn_matmul_159,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_159, False)
        ttnn_permute_140 = ttnn.permute(
            ttnn_typecast_136,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_136, False)
        ttnn_reshape_408 = ttnn.reshape(
            ttnn_permute_140,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_140, False)
        ttnn_matmul_160 = ttnn.matmul(
            ttnn_reshape_408,
            self.weights[68],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_408, False)
        ttnn_add_158 = ttnn.add(
            ttnn_matmul_160,
            self._ce["ce_8_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_160, False)
        ttnn_add_159 = ttnn.add(
            ttnn_add_156,
            ttnn_add_158,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_158, False)
        ttnn.deallocate(ttnn_add_156, False)
        ttnn_layer_norm_55 = ttnn.layer_norm(
            ttnn_add_159,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[66],
            bias=self.weights[65],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_409 = ttnn.reshape(
            ttnn_layer_norm_55,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_55, False)
        ttnn_matmul_161 = ttnn.matmul(
            ttnn_reshape_409,
            self.weights[64],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_409, False)
        ttnn_add_160 = ttnn.add(
            ttnn_matmul_161,
            self._ce["ce_43_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_161, False)
        ttnn_gelu_26 = ttnn.gelu(
            ttnn_add_160,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_160, False)
        ttnn_reshape_410 = ttnn.reshape(
            ttnn_gelu_26,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_26, False)
        ttnn_matmul_162 = ttnn.matmul(
            ttnn_reshape_410,
            self.weights[62],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_410, False)
        ttnn_add_161 = ttnn.add(
            ttnn_matmul_162,
            self._ce["ce_2_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_162, False)
        ttnn_add_162 = ttnn.add(
            ttnn_add_159,
            ttnn_add_161,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_161, False)
        ttnn.deallocate(ttnn_add_159, False)
        ttnn_layer_norm_56 = ttnn.layer_norm(
            ttnn_add_162,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[60],
            bias=self.weights[59],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_411 = ttnn.reshape(
            ttnn_layer_norm_56,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_56, False)
        ttnn_matmul_163 = ttnn.matmul(
            ttnn_reshape_411,
            self._ce["ce_33_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_411, False)
        ttnn_add_163 = ttnn.add(
            ttnn_matmul_163,
            self._ce["ce_73_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_163, False)
        ttnn_slice_81 = ttnn.slice(
            ttnn_add_163,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_82 = ttnn.slice(
            ttnn_add_163,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_83 = ttnn.slice(
            ttnn_add_163,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_163, False)
        ttnn_reshape_412 = ttnn.reshape(
            ttnn_slice_81,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_81, False)
        ttnn_reshape_413 = ttnn.reshape(
            ttnn_slice_82,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_82, False)
        ttnn_reshape_414 = ttnn.reshape(
            ttnn_slice_83,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_83, False)
        ttnn_permute_141 = ttnn.permute(
            ttnn_reshape_412,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_412, False)
        ttnn_permute_142 = ttnn.permute(
            ttnn_reshape_413,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_413, False)
        ttnn_permute_143 = ttnn.permute(
            ttnn_reshape_414,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_414, False)
        ttnn_typecast_137 = ttnn.typecast(
            ttnn_permute_141,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_141, False)
        ttnn_multiply_55 = ttnn.multiply(
            ttnn_typecast_137,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_137, False)
        ttnn_typecast_138 = ttnn.typecast(
            ttnn_permute_142,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_142, False)
        ttnn_permute_144 = ttnn.permute(
            ttnn_typecast_138,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_138, False)
        ttnn_multiply_56 = ttnn.multiply(
            ttnn_permute_144,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_144, False)
        ttnn_matmul_164 = ttnn.matmul(
            ttnn_multiply_55,
            ttnn_multiply_56,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_56, False)
        ttnn.deallocate(ttnn_multiply_55, False)
        ttnn_eq_27 = ttnn.eq(
            ttnn_matmul_164,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_54 = ttnn.logical_not(
            ttnn_eq_27,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_27, False)
        ttnn_sum_27 = ttnn.sum(
            ttnn_logical_not_54,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_54, False)
        ttnn_ne_27 = ttnn.ne(
            ttnn_sum_27,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_27, False)
        ttnn_logical_not_55 = ttnn.logical_not(
            ttnn_ne_27,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_27, False)
        ttnn_reshape_415 = ttnn.reshape(
            ttnn_logical_not_55,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_55, False)
        ttnn_softmax_27 = ttnn.softmax(
            ttnn_matmul_164,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_164, False)
        ttnn_repeat_215 = ttnn.repeat(ttnn_reshape_415, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_415, False)
        ttnn_typecast_139 = ttnn.typecast(
            ttnn_repeat_215,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_215, False)
        ttnn_where_27 = ttnn.where(
            ttnn_typecast_139,
            self._ce["cez_4_0"],
            ttnn_softmax_27,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_139, False)
        ttnn.deallocate(ttnn_softmax_27, False)
        ttnn_typecast_140 = ttnn.typecast(
            ttnn_permute_143,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_143, False)
        ttnn_matmul_165 = ttnn.matmul(
            ttnn_where_27,
            ttnn_typecast_140,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_140, False)
        ttnn.deallocate(ttnn_where_27, False)
        ttnn_typecast_141 = ttnn.typecast(
            ttnn_matmul_165,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_165, False)
        ttnn_permute_145 = ttnn.permute(
            ttnn_typecast_141,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_141, False)
        ttnn_reshape_416 = ttnn.reshape(
            ttnn_permute_145,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_145, False)
        ttnn_matmul_166 = ttnn.matmul(
            ttnn_reshape_416,
            self.weights[56],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_416, False)
        ttnn_add_164 = ttnn.add(
            ttnn_matmul_166,
            self._ce["ce_29_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_166, False)
        ttnn_add_165 = ttnn.add(
            ttnn_add_162,
            ttnn_add_164,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_164, False)
        ttnn.deallocate(ttnn_add_162, False)
        ttnn_layer_norm_57 = ttnn.layer_norm(
            ttnn_add_165,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[54],
            bias=self.weights[53],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_417 = ttnn.reshape(
            ttnn_layer_norm_57,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_57, False)
        ttnn_matmul_167 = ttnn.matmul(
            ttnn_reshape_417,
            self.weights[52],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_417, False)
        ttnn_add_166 = ttnn.add(
            ttnn_matmul_167,
            self._ce["ce_49_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_167, False)
        ttnn_gelu_27 = ttnn.gelu(
            ttnn_add_166,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_166, False)
        ttnn_reshape_418 = ttnn.reshape(
            ttnn_gelu_27,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_27, False)
        ttnn_matmul_168 = ttnn.matmul(
            ttnn_reshape_418,
            self.weights[50],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_418, False)
        ttnn_add_167 = ttnn.add(
            ttnn_matmul_168,
            self._ce["ce_145_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_168, False)
        ttnn_add_168 = ttnn.add(
            ttnn_add_165,
            ttnn_add_167,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_167, False)
        ttnn.deallocate(ttnn_add_165, False)
        ttnn_layer_norm_58 = ttnn.layer_norm(
            ttnn_add_168,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[48],
            bias=self.weights[47],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_419 = ttnn.reshape(
            ttnn_layer_norm_58,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_58, False)
        ttnn_matmul_169 = ttnn.matmul(
            ttnn_reshape_419,
            self._ce["ce_15_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_419, False)
        ttnn_add_169 = ttnn.add(
            ttnn_matmul_169,
            self._ce["ce_41_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_169, False)
        ttnn_slice_84 = ttnn.slice(
            ttnn_add_169,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_85 = ttnn.slice(
            ttnn_add_169,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_86 = ttnn.slice(
            ttnn_add_169,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_169, False)
        ttnn_reshape_420 = ttnn.reshape(
            ttnn_slice_84,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_84, False)
        ttnn_reshape_421 = ttnn.reshape(
            ttnn_slice_85,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_85, False)
        ttnn_reshape_422 = ttnn.reshape(
            ttnn_slice_86,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_86, False)
        ttnn_permute_146 = ttnn.permute(
            ttnn_reshape_420,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_420, False)
        ttnn_permute_147 = ttnn.permute(
            ttnn_reshape_421,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_421, False)
        ttnn_permute_148 = ttnn.permute(
            ttnn_reshape_422,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_422, False)
        ttnn_typecast_142 = ttnn.typecast(
            ttnn_permute_146,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_146, False)
        ttnn_multiply_57 = ttnn.multiply(
            ttnn_typecast_142,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_142, False)
        ttnn_typecast_143 = ttnn.typecast(
            ttnn_permute_147,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_147, False)
        ttnn_permute_149 = ttnn.permute(
            ttnn_typecast_143,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_143, False)
        ttnn_multiply_58 = ttnn.multiply(
            ttnn_permute_149,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_149, False)
        ttnn_matmul_170 = ttnn.matmul(
            ttnn_multiply_57,
            ttnn_multiply_58,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_58, False)
        ttnn.deallocate(ttnn_multiply_57, False)
        ttnn_eq_28 = ttnn.eq(
            ttnn_matmul_170,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_56 = ttnn.logical_not(
            ttnn_eq_28,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_28, False)
        ttnn_sum_28 = ttnn.sum(
            ttnn_logical_not_56,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_56, False)
        ttnn_ne_28 = ttnn.ne(
            ttnn_sum_28,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_28, False)
        ttnn_logical_not_57 = ttnn.logical_not(
            ttnn_ne_28,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_28, False)
        ttnn_reshape_423 = ttnn.reshape(
            ttnn_logical_not_57,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_57, False)
        ttnn_softmax_28 = ttnn.softmax(
            ttnn_matmul_170,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_170, False)
        ttnn_repeat_216 = ttnn.repeat(ttnn_reshape_423, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_423, False)
        ttnn_typecast_144 = ttnn.typecast(
            ttnn_repeat_216,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_216, False)
        ttnn_where_28 = ttnn.where(
            ttnn_typecast_144,
            self._ce["cez_4_0"],
            ttnn_softmax_28,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_144, False)
        ttnn.deallocate(ttnn_softmax_28, False)
        ttnn_typecast_145 = ttnn.typecast(
            ttnn_permute_148,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_148, False)
        ttnn_matmul_171 = ttnn.matmul(
            ttnn_where_28,
            ttnn_typecast_145,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_145, False)
        ttnn.deallocate(ttnn_where_28, False)
        ttnn_typecast_146 = ttnn.typecast(
            ttnn_matmul_171,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_171, False)
        ttnn_permute_150 = ttnn.permute(
            ttnn_typecast_146,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_146, False)
        ttnn_reshape_424 = ttnn.reshape(
            ttnn_permute_150,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_150, False)
        ttnn_matmul_172 = ttnn.matmul(
            ttnn_reshape_424,
            self.weights[44],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_424, False)
        ttnn_add_170 = ttnn.add(
            ttnn_matmul_172,
            self._ce["ce_57_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_172, False)
        ttnn_add_171 = ttnn.add(
            ttnn_add_168,
            ttnn_add_170,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_170, False)
        ttnn.deallocate(ttnn_add_168, False)
        ttnn_layer_norm_59 = ttnn.layer_norm(
            ttnn_add_171,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[42],
            bias=self.weights[41],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_425 = ttnn.reshape(
            ttnn_layer_norm_59,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_59, False)
        ttnn_matmul_173 = ttnn.matmul(
            ttnn_reshape_425,
            self.weights[40],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_425, False)
        ttnn_add_172 = ttnn.add(
            ttnn_matmul_173,
            self._ce["ce_136_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_173, False)
        ttnn_gelu_28 = ttnn.gelu(
            ttnn_add_172,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_172, False)
        ttnn_reshape_426 = ttnn.reshape(
            ttnn_gelu_28,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_28, False)
        ttnn_matmul_174 = ttnn.matmul(
            ttnn_reshape_426,
            self.weights[38],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_426, False)
        ttnn_add_173 = ttnn.add(
            ttnn_matmul_174,
            self._ce["ce_59_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_174, False)
        ttnn_add_174 = ttnn.add(
            ttnn_add_171,
            ttnn_add_173,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_173, False)
        ttnn.deallocate(ttnn_add_171, False)
        ttnn_layer_norm_60 = ttnn.layer_norm(
            ttnn_add_174,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[36],
            bias=self.weights[35],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_427 = ttnn.reshape(
            ttnn_layer_norm_60,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_60, False)
        ttnn_matmul_175 = ttnn.matmul(
            ttnn_reshape_427,
            self._ce["ce_69_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_427, False)
        ttnn_add_175 = ttnn.add(
            ttnn_matmul_175,
            self._ce["ce_14_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_175, False)
        ttnn_slice_87 = ttnn.slice(
            ttnn_add_175,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_88 = ttnn.slice(
            ttnn_add_175,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_89 = ttnn.slice(
            ttnn_add_175,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_175, False)
        ttnn_reshape_428 = ttnn.reshape(
            ttnn_slice_87,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_87, False)
        ttnn_reshape_429 = ttnn.reshape(
            ttnn_slice_88,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_88, False)
        ttnn_reshape_430 = ttnn.reshape(
            ttnn_slice_89,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_89, False)
        ttnn_permute_151 = ttnn.permute(
            ttnn_reshape_428,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_428, False)
        ttnn_permute_152 = ttnn.permute(
            ttnn_reshape_429,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_429, False)
        ttnn_permute_153 = ttnn.permute(
            ttnn_reshape_430,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_430, False)
        ttnn_typecast_147 = ttnn.typecast(
            ttnn_permute_151,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_151, False)
        ttnn_multiply_59 = ttnn.multiply(
            ttnn_typecast_147,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_147, False)
        ttnn_typecast_148 = ttnn.typecast(
            ttnn_permute_152,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_152, False)
        ttnn_permute_154 = ttnn.permute(
            ttnn_typecast_148,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_148, False)
        ttnn_multiply_60 = ttnn.multiply(
            ttnn_permute_154,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_154, False)
        ttnn_matmul_176 = ttnn.matmul(
            ttnn_multiply_59,
            ttnn_multiply_60,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_60, False)
        ttnn.deallocate(ttnn_multiply_59, False)
        ttnn_eq_29 = ttnn.eq(
            ttnn_matmul_176,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_58 = ttnn.logical_not(
            ttnn_eq_29,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_29, False)
        ttnn_sum_29 = ttnn.sum(
            ttnn_logical_not_58,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_58, False)
        ttnn_ne_29 = ttnn.ne(
            ttnn_sum_29,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_29, False)
        ttnn_logical_not_59 = ttnn.logical_not(
            ttnn_ne_29,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_29, False)
        ttnn_reshape_431 = ttnn.reshape(
            ttnn_logical_not_59,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_59, False)
        ttnn_softmax_29 = ttnn.softmax(
            ttnn_matmul_176,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_176, False)
        ttnn_repeat_217 = ttnn.repeat(ttnn_reshape_431, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_431, False)
        ttnn_typecast_149 = ttnn.typecast(
            ttnn_repeat_217,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_217, False)
        ttnn_where_29 = ttnn.where(
            ttnn_typecast_149,
            self._ce["cez_4_0"],
            ttnn_softmax_29,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_149, False)
        ttnn.deallocate(ttnn_softmax_29, False)
        ttnn_typecast_150 = ttnn.typecast(
            ttnn_permute_153,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_153, False)
        ttnn_matmul_177 = ttnn.matmul(
            ttnn_where_29,
            ttnn_typecast_150,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_150, False)
        ttnn.deallocate(ttnn_where_29, False)
        ttnn_typecast_151 = ttnn.typecast(
            ttnn_matmul_177,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_177, False)
        ttnn_permute_155 = ttnn.permute(
            ttnn_typecast_151,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_151, False)
        ttnn_reshape_432 = ttnn.reshape(
            ttnn_permute_155,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_155, False)
        ttnn_matmul_178 = ttnn.matmul(
            ttnn_reshape_432,
            self.weights[32],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_432, False)
        ttnn_add_176 = ttnn.add(
            ttnn_matmul_178,
            self._ce["ce_131_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_178, False)
        ttnn_add_177 = ttnn.add(
            ttnn_add_174,
            ttnn_add_176,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_176, False)
        ttnn.deallocate(ttnn_add_174, False)
        ttnn_layer_norm_61 = ttnn.layer_norm(
            ttnn_add_177,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[30],
            bias=self.weights[29],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_433 = ttnn.reshape(
            ttnn_layer_norm_61,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_61, False)
        ttnn_matmul_179 = ttnn.matmul(
            ttnn_reshape_433,
            self.weights[28],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_433, False)
        ttnn_add_178 = ttnn.add(
            ttnn_matmul_179,
            self._ce["ce_64_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_179, False)
        ttnn_gelu_29 = ttnn.gelu(
            ttnn_add_178,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_178, False)
        ttnn_reshape_434 = ttnn.reshape(
            ttnn_gelu_29,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_29, False)
        ttnn_matmul_180 = ttnn.matmul(
            ttnn_reshape_434,
            self.weights[26],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_434, False)
        ttnn_add_179 = ttnn.add(
            ttnn_matmul_180,
            self._ce["ce_139_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_180, False)
        ttnn_add_180 = ttnn.add(
            ttnn_add_177,
            ttnn_add_179,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_179, False)
        ttnn.deallocate(ttnn_add_177, False)
        ttnn_layer_norm_62 = ttnn.layer_norm(
            ttnn_add_180,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[24],
            bias=self.weights[23],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_435 = ttnn.reshape(
            ttnn_layer_norm_62,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_62, False)
        ttnn_matmul_181 = ttnn.matmul(
            ttnn_reshape_435,
            self._ce["ce_3_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_435, False)
        ttnn_add_181 = ttnn.add(
            ttnn_matmul_181,
            self._ce["ce_50_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_181, False)
        ttnn_slice_90 = ttnn.slice(
            ttnn_add_181,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_91 = ttnn.slice(
            ttnn_add_181,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_slice_92 = ttnn.slice(
            ttnn_add_181,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_181, False)
        ttnn_reshape_436 = ttnn.reshape(
            ttnn_slice_90,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_90, False)
        ttnn_reshape_437 = ttnn.reshape(
            ttnn_slice_91,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_91, False)
        ttnn_reshape_438 = ttnn.reshape(
            ttnn_slice_92,
            [1, 257, 16, 80],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_slice_92, False)
        ttnn_permute_156 = ttnn.permute(
            ttnn_reshape_436,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_436, False)
        ttnn_permute_157 = ttnn.permute(
            ttnn_reshape_437,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_437, False)
        ttnn_permute_158 = ttnn.permute(
            ttnn_reshape_438,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_438, False)
        ttnn_typecast_152 = ttnn.typecast(
            ttnn_permute_156,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_156, False)
        ttnn_multiply_61 = ttnn.multiply(
            ttnn_typecast_152,
            self._ce["cez_8_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_152, False)
        ttnn_typecast_153 = ttnn.typecast(
            ttnn_permute_157,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_157, False)
        ttnn_permute_159 = ttnn.permute(
            ttnn_typecast_153,
            [0, 1, 3, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_153, False)
        ttnn_multiply_62 = ttnn.multiply(
            ttnn_permute_159,
            self._ce["cez_2_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_159, False)
        ttnn_matmul_182 = ttnn.matmul(
            ttnn_multiply_61,
            ttnn_multiply_62,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_62, False)
        ttnn.deallocate(ttnn_multiply_61, False)
        ttnn_eq_30 = ttnn.eq(
            ttnn_matmul_182,
            self._ce["cez_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_60 = ttnn.logical_not(
            ttnn_eq_30,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_30, False)
        ttnn_sum_30 = ttnn.sum(
            ttnn_logical_not_60,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_60, False)
        ttnn_ne_30 = ttnn.ne(
            ttnn_sum_30,
            self._ce["cez_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_30, False)
        ttnn_logical_not_61 = ttnn.logical_not(
            ttnn_ne_30,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_30, False)
        ttnn_reshape_439 = ttnn.reshape(
            ttnn_logical_not_61,
            [1, 16, 257, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_61, False)
        ttnn_softmax_30 = ttnn.softmax(
            ttnn_matmul_182,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_182, False)
        ttnn_repeat_218 = ttnn.repeat(ttnn_reshape_439, ttnn.Shape([1, 1, 1, 257]))
        ttnn.deallocate(ttnn_reshape_439, False)
        ttnn_typecast_154 = ttnn.typecast(
            ttnn_repeat_218,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_218, False)
        ttnn_where_30 = ttnn.where(
            ttnn_typecast_154,
            self._ce["cez_4_0"],
            ttnn_softmax_30,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_154, False)
        ttnn.deallocate(ttnn_softmax_30, False)
        ttnn_typecast_155 = ttnn.typecast(
            ttnn_permute_158,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_158, False)
        ttnn_matmul_183 = ttnn.matmul(
            ttnn_where_30,
            ttnn_typecast_155,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_typecast_155, False)
        ttnn.deallocate(ttnn_where_30, False)
        ttnn_typecast_156 = ttnn.typecast(
            ttnn_matmul_183,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_183, False)
        ttnn_permute_160 = ttnn.permute(
            ttnn_typecast_156,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_156, False)
        ttnn_reshape_440 = ttnn.reshape(
            ttnn_permute_160,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_160, False)
        ttnn_matmul_184 = ttnn.matmul(
            ttnn_reshape_440,
            self.weights[20],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_440, False)
        ttnn_add_182 = ttnn.add(
            ttnn_matmul_184,
            self._ce["ce_78_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_184, False)
        ttnn_add_183 = ttnn.add(
            ttnn_add_180,
            ttnn_add_182,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_182, False)
        ttnn.deallocate(ttnn_add_180, False)
        ttnn_layer_norm_63 = ttnn.layer_norm(
            ttnn_add_183,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[18],
            bias=self.weights[17],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_441 = ttnn.reshape(
            ttnn_layer_norm_63,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_63, False)
        ttnn_matmul_185 = ttnn.matmul(
            ttnn_reshape_441,
            self.weights[16],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_441, False)
        ttnn_add_184 = ttnn.add(
            ttnn_matmul_185,
            self._ce["ce_39_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_185, False)
        ttnn_gelu_30 = ttnn.gelu(
            ttnn_add_184,
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_184, False)
        ttnn_reshape_442 = ttnn.reshape(
            ttnn_gelu_30,
            [257, 5120],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_gelu_30, False)
        ttnn_matmul_186 = ttnn.matmul(
            ttnn_reshape_442,
            self.weights[14],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_442, False)
        ttnn_add_185 = ttnn.add(
            ttnn_matmul_186,
            self._ce["ce_4_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_186, False)
        ttnn_add_186 = ttnn.add(
            ttnn_add_183,
            ttnn_add_185,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_185, False)
        ttnn.deallocate(ttnn_add_183, False)
        ttnn_reshape_443 = ttnn.reshape(
            ttnn_add_186,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_186, False)
        ttnn_matmul_187 = ttnn.matmul(
            ttnn_reshape_443,
            self.weights[12],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_443, False)
        ttnn_add_187 = ttnn.add(
            ttnn_matmul_187,
            self._ce["ce_126_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_187, False)
        ttnn_layer_norm_64 = ttnn.layer_norm(
            ttnn_add_187,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[10],
            bias=self.weights[9],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_444 = ttnn.reshape(
            ttnn_layer_norm_64,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_64, False)
        list_395 = [ttnn_reshape_444, self._ce["ce_137_2"]]
        ttnn_concat_63 = ttnn.concat(
            list_395,
            0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_444, False)
        ttnn_matmul_188 = ttnn.matmul(
            ttnn_concat_63,
            self.weights[516],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_157 = ttnn.typecast(
            ttnn_matmul_188,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_188, False)
        ttnn_reshape_445 = ttnn.reshape(
            ttnn_typecast_157,
            [1, 273, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_157, False)
        ttnn_permute_161 = ttnn.permute(
            ttnn_reshape_445,
            [0, 2, 3, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_445, False)
        ttnn_multiply_63 = ttnn.multiply(
            ttnn_permute_161,
            self._ce["cez_9_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_161, False)
        ttnn_matmul_189 = ttnn.matmul(
            self._ce["ce_137_1"],
            ttnn_multiply_63,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_63, False)
        ttnn_eq_31 = ttnn.eq(
            ttnn_matmul_189,
            self._ce["cez_3_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_62 = ttnn.logical_not(
            ttnn_eq_31,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_31, False)
        ttnn_sum_31 = ttnn.sum(
            ttnn_logical_not_62,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_62, False)
        ttnn_ne_31 = ttnn.ne(
            ttnn_sum_31,
            self._ce["cez_0_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_31, False)
        ttnn_logical_not_63 = ttnn.logical_not(
            ttnn_ne_31,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_31, False)
        ttnn_reshape_446 = ttnn.reshape(
            ttnn_logical_not_63,
            [1, 20, 16, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_63, False)
        ttnn_softmax_31 = ttnn.softmax(
            ttnn_matmul_189,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_189, False)
        ttnn_repeat_219 = ttnn.repeat(ttnn_reshape_446, ttnn.Shape([1, 1, 1, 273]))
        ttnn.deallocate(ttnn_reshape_446, False)
        ttnn_typecast_158 = ttnn.typecast(
            ttnn_repeat_219,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_219, False)
        ttnn_where_31 = ttnn.where(
            ttnn_typecast_158,
            self._ce["cez_6_0"],
            ttnn_softmax_31,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_158, False)
        ttnn.deallocate(ttnn_softmax_31, False)
        ttnn_matmul_190 = ttnn.matmul(
            ttnn_concat_63,
            self.weights[6],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_concat_63, False)
        ttnn_typecast_159 = ttnn.typecast(
            ttnn_matmul_190,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_190, False)
        ttnn_reshape_447 = ttnn.reshape(
            ttnn_typecast_159,
            [1, 273, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_159, False)
        ttnn_permute_162 = ttnn.permute(
            ttnn_reshape_447,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_447, False)
        ttnn_matmul_191 = ttnn.matmul(
            ttnn_where_31,
            ttnn_permute_162,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_permute_162, False)
        ttnn.deallocate(ttnn_where_31, False)
        ttnn_typecast_160 = ttnn.typecast(
            ttnn_matmul_191,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_191, False)
        ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
            ttnn_typecast_160,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_160, False)
        ttnn_reshape_448 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_0,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_0, False)
        ttnn_matmul_192 = ttnn.matmul(
            ttnn_reshape_448,
            self.weights[5],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_448, False)
        ttnn_reshape_449 = ttnn.reshape(
            ttnn_matmul_192,
            [1, 16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_192, False)
        ttnn_divide_0 = ttnn.divide(
            ttnn_reshape_449,
            self._ce["cez_1_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_449, False)
        ttnn_add_188 = ttnn.add(
            ttnn_divide_0,
            self.weights[4],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_divide_0, False)
        ttnn_layer_norm_65 = ttnn.layer_norm(
            ttnn_add_188,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[521],
            bias=self.weights[520],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_450 = ttnn.reshape(
            ttnn_layer_norm_65,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_65, False)
        ttnn_matmul_193 = ttnn.matmul(
            ttnn_reshape_450,
            self.weights[519],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        ttnn.deallocate(ttnn_reshape_450, False)
        ttnn_matmul_194 = ttnn.matmul(
            ttnn_matmul_193,
            self.weights[518],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_matmul_193, False)
        ttnn_reshape_451 = ttnn.reshape(
            ttnn_matmul_194,
            [1, 16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_194, False)
        ttnn_add_189 = ttnn.add(
            ttnn_reshape_451,
            ttnn_add_188,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_451, False)
        ttnn.deallocate(ttnn_add_188, False)
        ttnn_layer_norm_66 = ttnn.layer_norm(
            ttnn_add_189,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[525],
            bias=self.weights[524],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_452 = ttnn.reshape(
            ttnn_layer_norm_66,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_matmul_195 = ttnn.matmul(
            ttnn_reshape_452,
            self.weights[529],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_452, False)
        ttnn_typecast_161 = ttnn.typecast(
            ttnn_matmul_195,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_195, False)
        ttnn_reshape_453 = ttnn.reshape(
            ttnn_typecast_161,
            [1, 16, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_161, False)
        ttnn_permute_163 = ttnn.permute(
            ttnn_reshape_453,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_453, False)
        ttnn_multiply_64 = ttnn.multiply(
            ttnn_permute_163,
            self._ce["ce_137_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_163, False)
        ttnn_layer_norm_67 = ttnn.layer_norm(
            ttnn_add_187,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[527],
            bias=self.weights[526],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_454 = ttnn.reshape(
            ttnn_layer_norm_67,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_67, False)
        ttnn_reshape_455 = ttnn.reshape(
            ttnn_layer_norm_66,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_66, False)
        list_396 = [ttnn_reshape_454, ttnn_reshape_455]
        ttnn_concat_64 = ttnn.concat(
            list_396,
            0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_455, False)
        ttnn.deallocate(ttnn_reshape_454, False)
        ttnn_matmul_196 = ttnn.matmul(
            ttnn_concat_64,
            self.weights[528],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_162 = ttnn.typecast(
            ttnn_matmul_196,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_196, False)
        ttnn_reshape_456 = ttnn.reshape(
            ttnn_typecast_162,
            [1, 273, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_162, False)
        ttnn_permute_164 = ttnn.permute(
            ttnn_reshape_456,
            [0, 2, 3, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_456, False)
        ttnn_multiply_65 = ttnn.multiply(
            ttnn_permute_164,
            self._ce["cez_9_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_164, False)
        ttnn_matmul_197 = ttnn.matmul(
            ttnn_multiply_64,
            ttnn_multiply_65,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_65, False)
        ttnn.deallocate(ttnn_multiply_64, False)
        ttnn_eq_32 = ttnn.eq(
            ttnn_matmul_197,
            self._ce["cez_3_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_64 = ttnn.logical_not(
            ttnn_eq_32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_32, False)
        ttnn_sum_32 = ttnn.sum(
            ttnn_logical_not_64,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_64, False)
        ttnn_ne_32 = ttnn.ne(
            ttnn_sum_32,
            self._ce["cez_0_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_32, False)
        ttnn_logical_not_65 = ttnn.logical_not(
            ttnn_ne_32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_32, False)
        ttnn_reshape_457 = ttnn.reshape(
            ttnn_logical_not_65,
            [1, 20, 16, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_65, False)
        ttnn_softmax_32 = ttnn.softmax(
            ttnn_matmul_197,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_197, False)
        ttnn_repeat_220 = ttnn.repeat(ttnn_reshape_457, ttnn.Shape([1, 1, 1, 273]))
        ttnn.deallocate(ttnn_reshape_457, False)
        ttnn_typecast_163 = ttnn.typecast(
            ttnn_repeat_220,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_220, False)
        ttnn_where_32 = ttnn.where(
            ttnn_typecast_163,
            self._ce["cez_6_0"],
            ttnn_softmax_32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_163, False)
        ttnn.deallocate(ttnn_softmax_32, False)
        ttnn_matmul_198 = ttnn.matmul(
            ttnn_concat_64,
            self.weights[523],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_concat_64, False)
        ttnn_typecast_164 = ttnn.typecast(
            ttnn_matmul_198,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_198, False)
        ttnn_reshape_458 = ttnn.reshape(
            ttnn_typecast_164,
            [1, 273, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_164, False)
        ttnn_permute_165 = ttnn.permute(
            ttnn_reshape_458,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_458, False)
        ttnn_matmul_199 = ttnn.matmul(
            ttnn_where_32,
            ttnn_permute_165,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_permute_165, False)
        ttnn.deallocate(ttnn_where_32, False)
        ttnn_typecast_165 = ttnn.typecast(
            ttnn_matmul_199,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_199, False)
        ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
            ttnn_typecast_165,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_165, False)
        ttnn_reshape_459 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_1,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_1, False)
        ttnn_matmul_200 = ttnn.matmul(
            ttnn_reshape_459,
            self.weights[522],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_459, False)
        ttnn_reshape_460 = ttnn.reshape(
            ttnn_matmul_200,
            [1, 16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_200, False)
        ttnn_divide_1 = ttnn.divide(
            ttnn_reshape_460,
            self._ce["cez_1_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_460, False)
        ttnn_add_190 = ttnn.add(
            ttnn_divide_1,
            ttnn_add_189,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_divide_1, False)
        ttnn.deallocate(ttnn_add_189, False)
        ttnn_layer_norm_68 = ttnn.layer_norm(
            ttnn_add_190,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[533],
            bias=self.weights[532],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_461 = ttnn.reshape(
            ttnn_layer_norm_68,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_68, False)
        ttnn_matmul_201 = ttnn.matmul(
            ttnn_reshape_461,
            self.weights[531],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        ttnn.deallocate(ttnn_reshape_461, False)
        ttnn_matmul_202 = ttnn.matmul(
            ttnn_matmul_201,
            self.weights[530],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_matmul_201, False)
        ttnn_reshape_462 = ttnn.reshape(
            ttnn_matmul_202,
            [1, 16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_202, False)
        ttnn_add_191 = ttnn.add(
            ttnn_reshape_462,
            ttnn_add_190,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_462, False)
        ttnn.deallocate(ttnn_add_190, False)
        ttnn_layer_norm_69 = ttnn.layer_norm(
            ttnn_add_191,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[537],
            bias=self.weights[536],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_463 = ttnn.reshape(
            ttnn_layer_norm_69,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_matmul_203 = ttnn.matmul(
            ttnn_reshape_463,
            self.weights[541],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_463, False)
        ttnn_typecast_166 = ttnn.typecast(
            ttnn_matmul_203,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_203, False)
        ttnn_reshape_464 = ttnn.reshape(
            ttnn_typecast_166,
            [1, 16, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_166, False)
        ttnn_permute_166 = ttnn.permute(
            ttnn_reshape_464,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_464, False)
        ttnn_multiply_66 = ttnn.multiply(
            ttnn_permute_166,
            self._ce["ce_137_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_166, False)
        ttnn_layer_norm_70 = ttnn.layer_norm(
            ttnn_add_187,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[539],
            bias=self.weights[538],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_465 = ttnn.reshape(
            ttnn_layer_norm_70,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_70, False)
        ttnn_reshape_466 = ttnn.reshape(
            ttnn_layer_norm_69,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_69, False)
        list_397 = [ttnn_reshape_465, ttnn_reshape_466]
        ttnn_concat_65 = ttnn.concat(
            list_397,
            0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_466, False)
        ttnn.deallocate(ttnn_reshape_465, False)
        ttnn_matmul_204 = ttnn.matmul(
            ttnn_concat_65,
            self.weights[540],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_167 = ttnn.typecast(
            ttnn_matmul_204,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_204, False)
        ttnn_reshape_467 = ttnn.reshape(
            ttnn_typecast_167,
            [1, 273, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_167, False)
        ttnn_permute_167 = ttnn.permute(
            ttnn_reshape_467,
            [0, 2, 3, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_467, False)
        ttnn_multiply_67 = ttnn.multiply(
            ttnn_permute_167,
            self._ce["cez_9_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_167, False)
        ttnn_matmul_205 = ttnn.matmul(
            ttnn_multiply_66,
            ttnn_multiply_67,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_67, False)
        ttnn.deallocate(ttnn_multiply_66, False)
        ttnn_eq_33 = ttnn.eq(
            ttnn_matmul_205,
            self._ce["cez_3_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_66 = ttnn.logical_not(
            ttnn_eq_33,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_33, False)
        ttnn_sum_33 = ttnn.sum(
            ttnn_logical_not_66,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_66, False)
        ttnn_ne_33 = ttnn.ne(
            ttnn_sum_33,
            self._ce["cez_0_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_33, False)
        ttnn_logical_not_67 = ttnn.logical_not(
            ttnn_ne_33,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_33, False)
        ttnn_reshape_468 = ttnn.reshape(
            ttnn_logical_not_67,
            [1, 20, 16, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_67, False)
        ttnn_softmax_33 = ttnn.softmax(
            ttnn_matmul_205,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_205, False)
        ttnn_repeat_221 = ttnn.repeat(ttnn_reshape_468, ttnn.Shape([1, 1, 1, 273]))
        ttnn.deallocate(ttnn_reshape_468, False)
        ttnn_typecast_168 = ttnn.typecast(
            ttnn_repeat_221,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_221, False)
        ttnn_where_33 = ttnn.where(
            ttnn_typecast_168,
            self._ce["cez_6_0"],
            ttnn_softmax_33,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_168, False)
        ttnn.deallocate(ttnn_softmax_33, False)
        ttnn_matmul_206 = ttnn.matmul(
            ttnn_concat_65,
            self.weights[535],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_concat_65, False)
        ttnn_typecast_169 = ttnn.typecast(
            ttnn_matmul_206,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_206, False)
        ttnn_reshape_469 = ttnn.reshape(
            ttnn_typecast_169,
            [1, 273, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_169, False)
        ttnn_permute_168 = ttnn.permute(
            ttnn_reshape_469,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_469, False)
        ttnn_matmul_207 = ttnn.matmul(
            ttnn_where_33,
            ttnn_permute_168,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_permute_168, False)
        ttnn.deallocate(ttnn_where_33, False)
        ttnn_typecast_170 = ttnn.typecast(
            ttnn_matmul_207,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_207, False)
        ttnn_transformer_concatenate_heads_2 = ttnn.transformer.concatenate_heads(
            ttnn_typecast_170,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_170, False)
        ttnn_reshape_470 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_2,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_2, False)
        ttnn_matmul_208 = ttnn.matmul(
            ttnn_reshape_470,
            self.weights[534],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_470, False)
        ttnn_reshape_471 = ttnn.reshape(
            ttnn_matmul_208,
            [1, 16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_208, False)
        ttnn_divide_2 = ttnn.divide(
            ttnn_reshape_471,
            self._ce["cez_1_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_471, False)
        ttnn_add_192 = ttnn.add(
            ttnn_divide_2,
            ttnn_add_191,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_divide_2, False)
        ttnn.deallocate(ttnn_add_191, False)
        ttnn_layer_norm_71 = ttnn.layer_norm(
            ttnn_add_192,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[545],
            bias=self.weights[544],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_472 = ttnn.reshape(
            ttnn_layer_norm_71,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_71, False)
        ttnn_matmul_209 = ttnn.matmul(
            ttnn_reshape_472,
            self.weights[543],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        ttnn.deallocate(ttnn_reshape_472, False)
        ttnn_matmul_210 = ttnn.matmul(
            ttnn_matmul_209,
            self.weights[542],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_matmul_209, False)
        ttnn_reshape_473 = ttnn.reshape(
            ttnn_matmul_210,
            [1, 16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_210, False)
        ttnn_add_193 = ttnn.add(
            ttnn_reshape_473,
            ttnn_add_192,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_473, False)
        ttnn.deallocate(ttnn_add_192, False)
        ttnn_layer_norm_72 = ttnn.layer_norm(
            ttnn_add_193,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[549],
            bias=self.weights[548],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_474 = ttnn.reshape(
            ttnn_layer_norm_72,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_matmul_211 = ttnn.matmul(
            ttnn_reshape_474,
            self.weights[553],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_474, False)
        ttnn_typecast_171 = ttnn.typecast(
            ttnn_matmul_211,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_211, False)
        ttnn_reshape_475 = ttnn.reshape(
            ttnn_typecast_171,
            [1, 16, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_171, False)
        ttnn_permute_169 = ttnn.permute(
            ttnn_reshape_475,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_475, False)
        ttnn_multiply_68 = ttnn.multiply(
            ttnn_permute_169,
            self._ce["ce_137_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_169, False)
        ttnn_layer_norm_73 = ttnn.layer_norm(
            ttnn_add_187,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[551],
            bias=self.weights[550],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn.deallocate(ttnn_add_187, False)
        ttnn_reshape_476 = ttnn.reshape(
            ttnn_layer_norm_73,
            [257, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_73, False)
        ttnn_reshape_477 = ttnn.reshape(
            ttnn_layer_norm_72,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_72, False)
        list_398 = [ttnn_reshape_476, ttnn_reshape_477]
        ttnn_concat_66 = ttnn.concat(
            list_398,
            0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_477, False)
        ttnn.deallocate(ttnn_reshape_476, False)
        ttnn_matmul_212 = ttnn.matmul(
            ttnn_concat_66,
            self.weights[552],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_172 = ttnn.typecast(
            ttnn_matmul_212,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_212, False)
        ttnn_reshape_478 = ttnn.reshape(
            ttnn_typecast_172,
            [1, 273, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_172, False)
        ttnn_permute_170 = ttnn.permute(
            ttnn_reshape_478,
            [0, 2, 3, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_478, False)
        ttnn_multiply_69 = ttnn.multiply(
            ttnn_permute_170,
            self._ce["cez_9_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permute_170, False)
        ttnn_matmul_213 = ttnn.matmul(
            ttnn_multiply_68,
            ttnn_multiply_69,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_multiply_69, False)
        ttnn.deallocate(ttnn_multiply_68, False)
        ttnn_eq_34 = ttnn.eq(
            ttnn_matmul_213,
            self._ce["cez_3_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logical_not_68 = ttnn.logical_not(
            ttnn_eq_34,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_eq_34, False)
        ttnn_sum_34 = ttnn.sum(
            ttnn_logical_not_68,
            [3],
            False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_68, False)
        ttnn_ne_34 = ttnn.ne(
            ttnn_sum_34,
            self._ce["cez_0_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_sum_34, False)
        ttnn_logical_not_69 = ttnn.logical_not(
            ttnn_ne_34,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_ne_34, False)
        ttnn_reshape_479 = ttnn.reshape(
            ttnn_logical_not_69,
            [1, 20, 16, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_logical_not_69, False)
        ttnn_softmax_34 = ttnn.softmax(
            ttnn_matmul_213,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_213, False)
        ttnn_repeat_222 = ttnn.repeat(ttnn_reshape_479, ttnn.Shape([1, 1, 1, 273]))
        ttnn.deallocate(ttnn_reshape_479, False)
        ttnn_typecast_173 = ttnn.typecast(
            ttnn_repeat_222,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_repeat_222, False)
        ttnn_where_34 = ttnn.where(
            ttnn_typecast_173,
            self._ce["cez_6_0"],
            ttnn_softmax_34,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_173, False)
        ttnn.deallocate(ttnn_softmax_34, False)
        ttnn_matmul_214 = ttnn.matmul(
            ttnn_concat_66,
            self.weights[547],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_concat_66, False)
        ttnn_typecast_174 = ttnn.typecast(
            ttnn_matmul_214,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_214, False)
        ttnn_reshape_480 = ttnn.reshape(
            ttnn_typecast_174,
            [1, 273, 20, 64],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_174, False)
        ttnn_permute_171 = ttnn.permute(
            ttnn_reshape_480,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_480, False)
        ttnn_matmul_215 = ttnn.matmul(
            ttnn_where_34,
            ttnn_permute_171,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_permute_171, False)
        ttnn.deallocate(ttnn_where_34, False)
        ttnn_typecast_175 = ttnn.typecast(
            ttnn_matmul_215,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_215, False)
        ttnn_transformer_concatenate_heads_3 = ttnn.transformer.concatenate_heads(
            ttnn_typecast_175,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_175, False)
        ttnn_reshape_481 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_3,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_3, False)
        ttnn_matmul_216 = ttnn.matmul(
            ttnn_reshape_481,
            self.weights[546],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_reshape_481, False)
        ttnn_reshape_482 = ttnn.reshape(
            ttnn_matmul_216,
            [1, 16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_216, False)
        ttnn_divide_3 = ttnn.divide(
            ttnn_reshape_482,
            self._ce["cez_1_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_482, False)
        ttnn_add_194 = ttnn.add(
            ttnn_divide_3,
            ttnn_add_193,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_divide_3, False)
        ttnn.deallocate(ttnn_add_193, False)
        ttnn_layer_norm_74 = ttnn.layer_norm(
            ttnn_add_194,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[557],
            bias=self.weights[556],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn_reshape_483 = ttnn.reshape(
            ttnn_layer_norm_74,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_layer_norm_74, False)
        ttnn_matmul_217 = ttnn.matmul(
            ttnn_reshape_483,
            self.weights[555],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        ttnn.deallocate(ttnn_reshape_483, False)
        ttnn_reshape_484 = ttnn.reshape(
            ttnn_add_194,
            [16, 1280],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_194, False)
        ttnn_matmul_218 = ttnn.matmul(
            ttnn_matmul_217,
            self.weights[554],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_matmul_217, False)
        ttnn_add_195 = ttnn.add(
            ttnn_matmul_218,
            ttnn_reshape_484,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_218, False)
        ttnn.deallocate(ttnn_reshape_484, False)
        ttnn_matmul_219 = ttnn.matmul(
            ttnn_add_195,
            self.weights[3],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn.deallocate(ttnn_add_195, False)
        ttnn_add_196 = ttnn.add(
            ttnn_matmul_219,
            self._ce["ce_48_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_219, False)
        ttnn_layer_norm_75 = ttnn.layer_norm(
            ttnn_add_196,
            epsilon=self.LAYER_NORM_EPSILON,
            weight=self.weights[1],
            bias=self.weights[0],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn.deallocate(ttnn_add_196, False)
        list_399 = [ttnn_layer_norm_75]
        return list_399
