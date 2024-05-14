# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
import ttnn
from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0, comp_allclose_and_pcc


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True
    ttnn_module_args["activation"] = "relu"


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, UNet_enc_1_1):
            ttnn_module_args["encoder1_c1"] = ttnn_module_args.encoder1_1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.encoder1_1[0], model.encoder1_1[1])
            update_ttnn_module_args(ttnn_module_args["encoder1_c1"])
            ttnn_module_args["encoder1_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder1_c1"]["conv_blocking_and_parallelization_config_override"] = {
                "act_block_h": 16 * 32
            }
            ttnn_module_args["encoder1_c1"]["use_shallow_conv_variant"] = True
            parameters["encoder1_c1"], encoder1_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["encoder1_c1"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_enc_1_2):
            ttnn_module_args["encoder1_c2"] = ttnn_module_args.encoder1_2["0"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.encoder1_2[0], model.encoder1_2[1])
            update_ttnn_module_args(ttnn_module_args["encoder1_c2"])
            ttnn_module_args["encoder1_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder1_c2"]["conv_blocking_and_parallelization_config_override"] = {
                "act_block_h": 16 * 32
            }
            ttnn_module_args["encoder1_c2"]["use_shallow_conv_variant"] = True
            parameters["encoder1_c2"], encoder1_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["encoder1_c2"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_enc_2_1):
            ttnn_module_args["encoder2_c1"] = ttnn_module_args.encoder2_1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.encoder2_1[0], model.encoder2_1[1])
            update_ttnn_module_args(ttnn_module_args["encoder2_c1"])
            ttnn_module_args["encoder2_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder2_c1"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder2_c1"]["use_shallow_conv_variant"] = True
            parameters["encoder2_c1"], encoder2_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["encoder2_c1"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_enc_2_2):
            ttnn_module_args["encoder2_c2"] = ttnn_module_args.encoder2_2["0"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.encoder2_2[0], model.encoder2_2[1])
            update_ttnn_module_args(ttnn_module_args["encoder2_c2"])
            ttnn_module_args["encoder2_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder2_c2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder2_c2"]["use_shallow_conv_variant"] = True
            # print("encoder2_c2", ttnn_module_args["encoder2_c2"])
            parameters["encoder2_c2"], encoder2_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["encoder2_c2"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_enc_3_1):
            ttnn_module_args["encoder3_c1"] = ttnn_module_args.encoder3_1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.encoder3_1[0], model.encoder3_1[1])

            update_ttnn_module_args(ttnn_module_args["encoder3_c1"])
            ttnn_module_args["encoder3_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder3_c1"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder3_c1"]["use_shallow_conv_variant"] = False

            parameters["encoder3_c1"], encoder3_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["encoder3_c1"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_enc_3_2):
            ttnn_module_args["encoder3_c2"] = ttnn_module_args.encoder3_2["0"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.encoder3_2[0], model.encoder3_2[1])
            update_ttnn_module_args(ttnn_module_args["encoder3_c2"])
            ttnn_module_args["encoder3_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder3_c2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder3_c2"]["use_shallow_conv_variant"] = False
            parameters["encoder3_c2"], encoder3_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["encoder3_c2"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_enc_4_1):
            ttnn_module_args["encoder4_c1"] = ttnn_module_args.encoder4_1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.encoder4_1[0], model.encoder4_1[1])
            update_ttnn_module_args(ttnn_module_args["encoder4_c1"])
            ttnn_module_args["encoder4_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder4_c1"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder4_c1"]["use_shallow_conv_variant"] = False
            # print("encoder4_c1", ttnn_module_args["encoder4_c1"])
            parameters["encoder4_c1"], encoder4_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["encoder4_c1"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_enc_4_2):
            ttnn_module_args["encoder4_c2"] = ttnn_module_args.encoder4_2["0"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.encoder4_2[0], model.encoder4_2[1])
            update_ttnn_module_args(ttnn_module_args["encoder4_c2"])
            ttnn_module_args["encoder4_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder4_c2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder4_c2"]["use_shallow_conv_variant"] = True
            parameters["encoder4_c2"], encoder4_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["encoder4_c2"], return_parallel_config=True
            )
            return parameters
        if isinstance(model, UNet_bottle_1_1):
            ttnn_module_args["bottleneck_c1"] = ttnn_module_args.bottleneck1_1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.bottleneck1_1[0], model.bottleneck1_1[1])
            update_ttnn_module_args(ttnn_module_args["bottleneck_c1"])
            ttnn_module_args["bottleneck_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["bottleneck_c1"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["bottleneck_c1"]["use_shallow_conv_variant"] = False
            ttnn_module_args["bottleneck_c1"]["use_1d_systolic_array"] = True
            parameters["bottleneck_c1"], bottleneck_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["bottleneck_c1"], return_parallel_config=True
            )
            return parameters
        if isinstance(model, UNet_dec_4_1):
            ttnn_module_args["decoder4_c1"] = ttnn_module_args.decoder4_1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.decoder4_1[0], model.decoder4_1[1])
            update_ttnn_module_args(ttnn_module_args["decoder4_c1"])
            ttnn_module_args["decoder4_c1"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder4_c1"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args["decoder4_c1"]["use_shallow_conv_variant"] = False
            parameters["decoder4_c1"], decoder4_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["decoder4_c1"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_dec_4_2):
            ttnn_module_args["decoder4_c2"] = ttnn_module_args.decoder4_2["0"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.decoder4_2[0], model.decoder4_2[1])
            update_ttnn_module_args(ttnn_module_args["decoder4_c2"])
            ttnn_module_args["decoder4_c2"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder4_c2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["decoder4_c2"]["use_shallow_conv_variant"] = False
            parameters["decoder4_c2"], decoder4_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["decoder4_c2"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_dec_3_1):
            ttnn_module_args["decoder3_c1"] = ttnn_module_args.decoder3_1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.decoder3_1[0], model.decoder3_1[1])
            update_ttnn_module_args(ttnn_module_args["decoder3_c1"])
            ttnn_module_args["decoder3_c1"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder3_c1"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args["decoder3_c1"]["use_shallow_conv_variant"] = False
            parameters["decoder3_c1"], decoder3_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["decoder3_c1"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_dec_3_2):
            ttnn_module_args["decoder3_c2"] = ttnn_module_args.decoder3_2["0"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.decoder3_2[0], model.decoder3_2[1])
            update_ttnn_module_args(ttnn_module_args["decoder3_c2"])
            ttnn_module_args["decoder3_c2"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder3_c2"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args["decoder3_c2"]["use_shallow_conv_variant"] = False
            parameters["decoder3_c2"], decoder3_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["decoder3_c2"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_dec_2_1):
            ttnn_module_args["decoder2_c1"] = ttnn_module_args.decoder2_1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.decoder2_1[0], model.decoder2_1[1])
            update_ttnn_module_args(ttnn_module_args["decoder2_c1"])
            ttnn_module_args["decoder2_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["decoder2_c1"]["conv_blocking_and_parallelization_config_override"] = {
                "act_block_h": 4 * 32
            }
            ttnn_module_args["decoder2_c1"]["use_shallow_conv_variant"] = False
            parameters["decoder2_c1"], decoder2_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["decoder2_c1"], return_parallel_config=True
            )
            return parameters
        if isinstance(model, UNet_dec_2_2):
            ttnn_module_args["decoder2_c2"] = ttnn_module_args.decoder2_2["0"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.decoder2_2[0], model.decoder2_2[1])
            update_ttnn_module_args(ttnn_module_args["decoder2_c2"])
            ttnn_module_args["decoder2_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["decoder2_c2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["decoder2_c2"]["use_shallow_conv_variant"] = True
            parameters["decoder2_c2"], decoder2_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["decoder2_c2"], return_parallel_config=True
            )
            return parameters
        if isinstance(model, UNet_dec_1_2):
            ttnn_module_args["decoder1_c2"] = ttnn_module_args.decoder1_2["0"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.decoder1_2[0], model.decoder1_2[1])
            update_ttnn_module_args(ttnn_module_args["decoder1_c2"])
            ttnn_module_args["decoder1_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["decoder1_c2"]["conv_blocking_and_parallelization_config_override"] = {
                "act_block_h": 16 * 32
            }
            ttnn_module_args["decoder1_c2"]["use_shallow_conv_variant"] = True
            parameters["decoder1_c2"], decoder1_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["decoder1_c2"], return_parallel_config=True
            )
            return parameters

    return custom_preprocessor


class UNet_enc_1_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_enc_1_1, self).__init__()

        features = init_features
        self.encoder1_1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.encoder1_1(x)
        return output


class TtUnet_enc_1_1:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.enc1_1 = parameters.encoder1_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.enc1_1.conv.input_sharded_memory_config)
        output_tensor_enc1_1 = self.enc1_1(input_tensor)

        return output_tensor_enc1_1


class UNet_enc_1_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_enc_1_2, self).__init__()

        features = init_features
        self.encoder1_2 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.encoder1_2(x)
        return output


class TtUnet_enc_1_2:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.enc1_2 = parameters.encoder1_c2

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.enc1_2.conv.input_sharded_memory_config)
        output_tensor_enc1_2 = self.enc1_2(input_tensor)

        return output_tensor_enc1_2


class UNet_enc_2_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_enc_2_1, self).__init__()

        features = init_features
        self.encoder2_1 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.encoder2_1(x)
        return output


class TtUnet_enc_2_1:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.enc2_1 = parameters.encoder2_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.enc2_1.conv.input_sharded_memory_config)
        output_tensor_enc2_1 = self.enc2_1(input_tensor)

        return output_tensor_enc2_1


class UNet_enc_2_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_enc_2_2, self).__init__()

        features = init_features
        self.encoder2_2 = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.encoder2_2(x)
        return output


class TtUnet_enc_2_2:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.enc2_2 = parameters.encoder2_c2

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.enc2_2.conv.input_sharded_memory_config)
        output_tensor_enc2_2 = self.enc2_2(input_tensor)

        return output_tensor_enc2_2


class UNet_enc_3_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_enc_3_1, self).__init__()

        features = init_features
        self.encoder3_1 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.encoder3_1(x)
        return output


class TtUnet_enc_3_1:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.enc3_1 = parameters.encoder3_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.enc3_1.conv.input_sharded_memory_config)
        output_tensor_enc3_1 = self.enc3_1(input_tensor)

        return output_tensor_enc3_1


class UNet_enc_3_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_enc_3_2, self).__init__()

        features = init_features
        self.encoder3_2 = nn.Sequential(
            nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.encoder3_2(x)
        return output


class TtUnet_enc_3_2:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.enc3_2 = parameters.encoder3_c2

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.enc3_2.conv.input_sharded_memory_config)
        output_tensor_enc3_2 = self.enc3_2(input_tensor)

        return output_tensor_enc3_2


class UNet_enc_4_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_enc_4_1, self).__init__()

        features = init_features
        self.encoder4_1 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.encoder4_1(x)
        return output


class TtUnet_enc_4_1:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.enc4_1 = parameters.encoder4_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.enc4_1.conv.input_sharded_memory_config)
        output_tensor_enc4_1 = self.enc4_1(input_tensor)

        return output_tensor_enc4_1


class UNet_enc_4_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_enc_4_2, self).__init__()

        features = init_features
        self.encoder4_2 = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.encoder4_2(x)
        return output


class TtUnet_enc_4_2:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.enc4_2 = parameters.encoder4_c2

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.enc4_2.conv.input_sharded_memory_config)
        output_tensor_enc4_2 = self.enc4_2(input_tensor)

        return output_tensor_enc4_2


class UNet_bottle_1_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_bottle_1_1, self).__init__()

        features = init_features
        self.bottleneck1_1 = nn.Sequential(
            nn.Conv2d(features * 8, features * 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 16),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.bottleneck1_1(x)
        return output


class TtUnet_bottle_1_1:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.bottleneck1_1 = parameters.bottleneck_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.bottleneck1_1.conv.input_sharded_memory_config)
        output_tensor_bottle1_1 = self.bottleneck1_1(input_tensor)

        return output_tensor_bottle1_1


class UNet_dec_4_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_dec_4_1, self).__init__()

        features = init_features
        self.decoder4_1 = nn.Sequential(
            nn.Conv2d(features * 16, features * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.decoder4_1(x)
        return output


class TtUnet_dec_4_1:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.dec4_1 = parameters.decoder4_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.dec4_1.conv.input_sharded_memory_config)
        output_tensor_dec4_1 = self.dec4_1(input_tensor)

        return output_tensor_dec4_1


class UNet_dec_4_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_dec_4_2, self).__init__()

        features = init_features
        self.decoder4_2 = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.decoder4_2(x)
        return output


class TtUnet_dec_4_2:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.dec4_2 = parameters.decoder4_c2

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.dec4_2.conv.input_sharded_memory_config)
        output_tensor_dec4_2 = self.dec4_2(input_tensor)

        return output_tensor_dec4_2


class UNet_dec_3_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_dec_3_1, self).__init__()

        features = init_features
        self.decoder3_1 = nn.Sequential(
            nn.Conv2d(features * 8, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.decoder3_1(x)
        return output


class TtUnet_dec_3_1:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.dec3_1 = parameters.decoder3_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.dec3_1.conv.input_sharded_memory_config)
        output_tensor_dec3_1 = self.dec3_1(input_tensor)

        return output_tensor_dec3_1


class UNet_dec_3_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_dec_3_2, self).__init__()

        features = init_features
        self.decoder3_2 = nn.Sequential(
            nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.decoder3_2(x)
        return output


class TtUnet_dec_3_2:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.dec3_2 = parameters.decoder3_c2

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.dec3_2.conv.input_sharded_memory_config)
        output_tensor_dec3_2 = self.dec3_2(input_tensor)

        return output_tensor_dec3_2


class UNet_dec_2_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_dec_2_1, self).__init__()

        features = init_features
        self.decoder2_1 = nn.Sequential(
            nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.decoder2_1(x)
        return output


class TtUnet_dec_2_1:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.dec2_1 = parameters.decoder2_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.dec2_1.conv.input_sharded_memory_config)
        output_tensor_dec2_1 = self.dec2_1(input_tensor)

        return output_tensor_dec2_1


class UNet_dec_2_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_dec_2_2, self).__init__()

        features = init_features
        self.decoder2_2 = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.decoder2_2(x)
        return output


class TtUnet_dec_2_2:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.dec2_2 = parameters.decoder2_c2

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.dec2_2.conv.input_sharded_memory_config)
        output_tensor_dec2_2 = self.dec2_2(input_tensor)

        return output_tensor_dec2_2


class UNet_dec_1_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_dec_1_2, self).__init__()

        features = init_features
        self.decoder1_2 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.decoder1_2(x)
        return output


class TtUnet_dec_1_2:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.dec1_2 = parameters.decoder1_c2

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.dec1_2.conv.input_sharded_memory_config)
        output_tensor_dec1_2 = self.dec1_2(input_tensor)

        return output_tensor_dec1_2


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_enc_1_1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("encoder1."))}

    torch_model = UNet_enc_1_1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 3, 480, 640)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_enc_1_1(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 480, 640, 32)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp encoder1_1", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9991010675318949
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_enc_2_1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("encoder2."))}

    torch_model = UNet_enc_2_1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 32, 240, 320)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_enc_2_1(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 240, 320, 64)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp encoder2_1", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9992943214210384
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_enc_3_1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("encoder3."))}

    torch_model = UNet_enc_3_1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 64, 120, 160)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_enc_3_1(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 120, 160, 128)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp encoder3_1", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9992638200717393
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_enc_4_1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("encoder4."))}

    torch_model = UNet_enc_4_1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 128, 60, 80)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_enc_4_1(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 60, 80, 256)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp encoder4_1", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9992864261866354
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_bottleneck_1_1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("bottleneck."))}

    torch_model = UNet_bottle_1_1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 256, 30, 40)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_bottle_1_1(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 30, 40, 512)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print(
        "comp bottleneck_1", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99)
    )  # 0.2932217371744389
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_dec_4_1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("decoder4."))}

    torch_model = UNet_dec_4_1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 512, 60, 80)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_dec_4_1(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 60, 80, 256)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp decoder4_1", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.3347515936821286
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_dec_3_1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("decoder3."))}

    torch_model = UNet_dec_3_1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 256, 120, 160)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_dec_3_1(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 120, 160, 128)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp decoder3_1", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.44777801628917024
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_dec_2_1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("decoder2."))}

    torch_model = UNet_dec_2_1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 128, 240, 320)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_dec_2_1(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 240, 320, 64)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp decoder2_1", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9992873884242356
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_enc_1_2(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "encoder1.enc1conv2.weight",
                "encoder1.enc1norm2.weight",
                "encoder1.enc1norm2.bias",
                "encoder1.enc1norm2.running_mean",
                "encoder1.enc1norm2.running_var",
                "encoder1.enc1norm2.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_enc_1_2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 32, 480, 640)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_enc_1_2(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 480, 640, 32)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp encoder1_2", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9993259423032697
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_enc_2_2(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "encoder2.enc2conv2.weight",
                "encoder2.enc2norm2.weight",
                "encoder2.enc2norm2.bias",
                "encoder2.enc2norm2.running_mean",
                "encoder2.enc2norm2.running_var",
                "encoder2.enc2norm2.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_enc_2_2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 64, 240, 320)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_enc_2_2(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 240, 320, 64)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp encoder2_2", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9993404273234131
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_enc_3_2(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "encoder3.enc3conv2.weight",
                "encoder3.enc3norm2.weight",
                "encoder3.enc3norm2.bias",
                "encoder3.enc3norm2.running_mean",
                "encoder3.enc3norm2.running_var",
                "encoder3.enc3norm2.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_enc_3_2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 128, 120, 160)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_enc_3_2(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 120, 160, 128)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp encoder3_2", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9993155395170065
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_enc_4_2(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "encoder4.enc4conv2.weight",
                "encoder4.enc4norm2.weight",
                "encoder4.enc4norm2.bias",
                "encoder4.enc4norm2.running_mean",
                "encoder4.enc4norm2.running_var",
                "encoder4.enc4norm2.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_enc_4_2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 256, 60, 80)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_enc_4_2(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 60, 80, 256)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp encoder4_2", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9993818484150027
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_dec_4_2(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "decoder4.dec4conv2.weight",
                "decoder4.dec4norm2.weight",
                "decoder4.dec4norm2.bias",
                "decoder4.dec4norm2.running_mean",
                "decoder4.dec4norm2.running_var",
                "decoder4.dec4norm2.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_dec_4_2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 256, 60, 80)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_dec_4_2(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 60, 80, 256)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp decoder4_2", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.998905814760247
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_dec_3_2(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "decoder3.dec3conv2.weight",
                "decoder3.dec3norm2.weight",
                "decoder3.dec3norm2.bias",
                "decoder3.dec3norm2.running_mean",
                "decoder3.dec3norm2.running_var",
                "decoder3.dec3norm2.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_dec_3_2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 128, 120, 160)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_dec_3_2(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 120, 160, 128)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp decoder3_2", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.36713229144766146
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_dec_2_2(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "decoder2.dec2conv2.weight",
                "decoder2.dec2norm2.weight",
                "decoder2.dec2norm2.bias",
                "decoder2.dec2norm2.running_mean",
                "decoder2.dec2norm2.running_var",
                "decoder2.dec2norm2.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_dec_2_2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 64, 240, 320)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_dec_2_2(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 240, 320, 64)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp decoder2_2", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9991970892466855
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_dec_1_2(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "decoder1.dec1conv2.weight",
                "decoder1.dec1norm2.weight",
                "decoder1.dec1norm2.bias",
                "decoder1.dec1norm2.running_mean",
                "decoder1.dec1norm2.running_var",
                "decoder1.dec1norm2.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_dec_1_2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 32, 480, 640)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_dec_1_2(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #

    output_tensor = ttnn.to_torch(output_tensor)

    output_tensor = output_tensor.reshape(1, 480, 640, 32)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp decoder1_2", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9991580031750404
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
