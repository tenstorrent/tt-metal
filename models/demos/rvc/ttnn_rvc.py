# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: MIT

import torch
import ttnn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import pickle


class TtnnPosteriorEncoder:
    def __init__(self, device, state_dict, layer_name, model_config, dtype=ttnn.bfloat16):
        self.device = device
        self.state_dict = state_dict
        self.layer_name = layer_name
        self.model_config = model_config
        self.dtype = dtype

    def _conv1d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        weight = ttnn.to_device(weight, self.device)
        if bias is not None:
            bias = ttnn.to_device(bias, self.device)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [1, 1, x.shape[0], x.shape[1]])  # [B, C, H, W]
        weight = ttnn.reshape(weight, [weight.shape[0], weight.shape[1], 1, 1])
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            batch_size=1,
            input_height=x.shape[2],
            input_width=x.shape[3],
            device=self.device,
            use_1d_systolic_array=True,
            weight_dtype=self.dtype,
            output_dtype=self.dtype,
            reader_patterns_cache=self.model_config["reader_patterns_cache"],
            deallocate_activation=True,
            conv_op_cache=self.model_config["conv_cache"],
        )[0]
        x = ttnn.reshape(x, [x.shape[2], x.shape[3]])
        if bias is not None:
            x = ttnn.add(x, bias)
            ttnn.deallocate(bias)
        ttnn.deallocate(weight)
        return x

    def __call__(self, x, g):
        x = ttnn.to_device(x, self.device)
        g = ttnn.to_device(g, self.device)

        # Initial conv
        conv1_weight = self.state_dict[f"{self.layer_name}.pre_net.conv_0.weight"]
        conv1_bias = self.state_dict.get(f"{self.layer_name}.pre_net.conv_0.bias", None)
        x = self._conv1d(x, conv1_weight, conv1_bias, padding=1)
        x = ttnn.tanh(x)

        # Residual blocks
        for i in range(2):
            x_skip = x
            x = ttnn.tanh(x)
            x = self._conv1d(x, self.state_dict[f"{self.layer_name}.resblocks.{i}.conv1.weight"], padding=1)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            g_conv = self._conv1d(g, self.state_dict[f"{self.layer_name}.resblocks.{i}.cond_layer_norm.weight"], bias=None)
            g_conv = ttnn.reshape(g_conv, [1, -1, 1])
            g_conv = ttnn.to_layout(g_conv, ttnn.TILE_LAYOUT)
            x = ttnn.add(x, g_conv)
            ttnn.deallocate(g_conv)
            x = ttnn.tanh(x)
            x = self._conv1d(x, self.state_dict[f"{self.layer_name}.resblocks.{i}.conv2.weight"], padding=1)
            x = ttnn.add(x, x_skip)

        # Final conv
        x = ttnn.tanh(x)
        x = self._conv1d(x, self.state_dict[f"{self.layer_name}.post_net.conv_0.weight"], padding=1)
        ttnn.deallocate(x)
        ttnn.deallocate(g)
        return x


class TtnnFeatureRetrieval:
    def __init__(self, index_file: str, device, feature_bank_shape: Tuple[int, int], dtype=ttnn.bfloat16):
        self.index_file = Path(index_file)
        self.device = device
        self.feature_bank_shape = feature_bank_shape
        self.dtype = dtype
        self.feature_bank = None
        self._load_index_features()

    def _load_index_features(self):
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file {self.index_file} not found.")
        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
            features = data["feature_bank"]
        except Exception as e:
            raise RuntimeError(f"Failed to load index file: {e}")
        self.feature_bank = ttnn.from_torch(
            torch.tensor(features, dtype=torch.bfloat16),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def __call__(self, query: ttnn.Tensor, search_ratio: float = 0.75) -> ttnn.Tensor:
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        query = ttnn.to_device(query, self.device)
        query_norm = ttnn.sqrt(ttnn.sum(ttnn.square(query), dim=-1, keepdim=True))
        bank_norm = ttnn.sqrt(ttnn.sum(ttnn.square(self.feature_bank), dim=-1, keepdim=True))
        similarity = ttnn.matmul(query, ttnn.transpose(self.feature_bank, -2, -1))
        similarity = ttnn.div(similarity, ttnn.add(ttnn.multiply(query_norm, ttnn.transpose(bank_norm, -2, -1)), 1e-8))
        top_k = max(1, int(self.feature_bank_shape[0] * search_ratio))
        values, indices = ttnn.topk(similarity, top_k, dim=-1)
        weights = ttnn.softmax(values, dim=-1)
        retrieved = ttnn.matmul(weights, ttnn.gather(self.feature_bank, -2, indices))
        ttnn.deallocate(query)
        ttnn.deallocate(similarity)
        ttnn.deallocate(values)
        ttnn.deallocate(indices)
        ttnn.deallocate(weights)
        return retrieved


class TtnnHiFiGANVocoder:
    def __init__(self, device, state_dict, layer_name, model_config, dtype=ttnn.bfloat16):
        self.device = device
        self.state_dict = state_dict
        self.layer_name = layer_name
        self.model_config = model_config
        self.dtype = dtype

    def _conv1d_transpose(self, x, weight, bias, stride, padding, output_padding=0):
        weight = ttnn.to_device(weight, self.device)
        bias = ttnn.to_device(bias, self.device)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [1, 1, x.shape[0], x.shape[1]])
        weight = ttnn.permute(weight, (0, 1, 3, 2))  # swap W and H for transpose
        weight = ttnn.reshape(weight, [weight.shape[0], weight.shape[1], 1, weight.shape[3]])
        out_h = (x.shape[2] - 1) * stride + 1 + output_padding
        out_w = (x.shape[3] - 1) * stride + weight.shape[3] - 2 * padding
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
            kernel_size=(1, weight.shape[3]),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(1, 1),
            groups=1,
            batch_size=1,
            input_height=x.shape[2],
            input_width=x.shape[3],
            device=self.device,
            use_1d_systolic_array=True,
            weight_dtype=self.dtype,
            output_dtype=self.dtype,
            reader_patterns_cache=self.model_config["reader_patterns_cache"],
            deallocate_activation=True,
            conv_op_cache=self.model_config["conv_cache"],
            transpose_mcast=True,
            padded_input_width=out_w,
        )[0]
        x = ttnn.reshape(x, [x.shape[2], x.shape[3]])
        x = ttnn.add(x, bias)
        ttnn.deallocate(weight)
        ttnn.deallocate(bias)
        return x

    def _conv1d(self, x, weight, bias, padding=0):
        weight = ttnn.to_device(weight, self.device)
        bias = ttnn.to_device(bias, self.device)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [1, 1, x.shape[0], x.shape[1]])
        weight = ttnn.reshape(weight, [weight.shape[0], weight.shape[1], 1, 1])
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(padding, padding),
            dilation=(1, 1),
            groups=1,
            batch_size=1,
            input_height=x.shape[2],
            input_width=x.shape[3],
            device=self.device,
            use_1d_systolic_array=True,
            weight_dtype=self.dtype,
            output_dtype=self.dtype,
            reader_patterns_cache=self.model_config["reader_patterns_cache"],
            deallocate_activation=True,
            conv_op_cache=self.model_config["conv_cache"],
        )[0]
        x = ttnn.reshape(x, [x.shape[2], x.shape[3]])
        x = ttnn.add(x, bias)
        ttnn.deallocate(weight)
        ttnn.deallocate(bias)
        return x

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_device(x, self.device)
        x = ttnn.transpose(x, -2, -1)

        # Initial layer
        x = self._conv1d(x, self.state_dict[f"{self.layer_name}.conv_pre.weight"], self.state_dict[f"{self.layer_name}.conv_pre.bias"])

        # Transposed conv blocks
        for i in range(5):
            x = ttnn.tanh(x)
            weight = self.state_dict[f"{self.layer_name}.upsampler.{i}.weight"]
            bias = self.state_dict[f"{self.layer_name}.upsampler.{i}.bias"]
            stride = self.model_config["upsample_rates"][i]
            pad = self.model_config["upsample_kernel_sizes"][i] // 2
            x = self._conv1d_transpose(x, weight, bias, stride=stride, padding=pad)

            # Residual blocks
            res_x = x
            for j in range(3):
                x_res = ttnn.tanh(x)
                weight_h = self.state_dict[f"{self.layer_name}.resblocks.{i * 3 + j}.conv_r1.weight"]
                bias_h = self.state_dict[f"{self.layer_name}.resblocks.{i * 3 + j}.conv_r1.bias"]
                x_res = self._conv1d(x_res, weight_h, bias_h, padding=3)
                x_res = ttnn.tanh(x_res)
                weight_l = self.state_dict[f"{self.layer_name}.resblocks.{i * 3 + j}.conv_r2.weight"]
                bias_l = self.state_dict[f"{self.layer_name}.resblocks.{i * 3 + j}.conv_r2.bias"]
                x_res = self._conv1d(x_res, weight_l, bias_l, padding=9)
                res_x = ttnn.add(res_x, x_res)
                ttnn.deallocate(x_res)
            x = res_x

        # Final layer
        x = ttnn.tanh(x)
        x = self._conv1d(x, self.state_dict[f"{self.layer_name}.conv_post.weight"], self.state_dict[f"{self.layer_name}.conv_post.bias"])
        x = ttnn.tanh(x)

        # Squeeze and return
        x = ttnn.squeeze(x, 0)
        ttnn.deallocate(x)
        return x


class TtnnRVC:
    def __init__(self, device, state_dict, index_file, model_config, dtype=ttnn.bfloat16):
        self.device = device
        self.state_dict = state_dict
        self.model_config = model_config
        self.dtype = dtype

        self.encoder = TtnnPosteriorEncoder(device, state_dict, "encoder", model_config, dtype)
        self.feature_retrieval = TtnnFeatureRetrieval(index_file, device, model_config["feature_bank_shape"], dtype)
        self.vocoder = TtnnHiFiGANVocoder(device, state_dict, "vocoder", model_config, dtype)

    def __call__(self, audio: torch.Tensor, pitch: Optional[torch.Tensor] = None, search_ratio: float = 0.75) -> torch.Tensor:
        # Move input to device
        audio_tt = ttnn.from_torch(audio, device=self.device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        g = ttnn.from_torch(torch.zeros(1, 256), device=self.device, dtype=self.dtype)

        # Posterior encoder
        z = self.encoder(audio_tt, g)

        # Feature retrieval
        c = self.feature_retrieval(z, search_ratio=search_ratio)

        # Vocoder
        audio_out = self.vocoder(c)

        # Move result back to host
        audio_out_torch = ttnn.to_torch(audio_out)

        # Deallocate all intermediate tensors
        ttnn.deallocate(audio_tt)
        ttnn.deallocate(g)
        ttnn.deallocate(z)
        ttnn.deallocate(c)

        return audio_out_torch