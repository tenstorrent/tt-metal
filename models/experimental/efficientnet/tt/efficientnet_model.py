# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import copy
from typing import List, Sequence, Union, Tuple, Optional, Any
from tt_lib.fallback_ops import fallback_ops
import torchvision
from functools import partial
from loguru import logger

from models.utility_functions import (
    torch2tt_tensor,
)
from models.experimental.efficientnet.tt.efficientnet_conv import TtEfficientnetConv2dNormActivation
from models.experimental.efficientnet.reference.efficientnet_lite import build_efficientnet_lite
from models.experimental.efficientnet.tt.efficientnet_mbconv import (
    _MBConvConfig,
    MBConvConfig,
)
from models.experimental.efficientnet.tt.efficientnet_fused_mbconv import (
    FusedMBConvConfig,
)


class TtEfficientNet(torch.nn.Module):
    def __init__(
        self,
        state_dict,
        device,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        norm_layer_eps: float = 1e-05,
        norm_layer_momentum: float = 0.1,
        last_channel: Optional[int] = None,
        is_lite=False,
    ):
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()

        self.device = device

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        layers: List[torch.nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels

        layers.append(
            TtEfficientnetConv2dNormActivation(
                state_dict=state_dict,
                conv_base_address=f"stem.0" if is_lite else f"features.{len(layers)}.0",
                bn_base_address=f"stem.1" if is_lite else f"features.{len(layers)}.1",
                device=device,
                in_channels=3,
                out_channels=firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer_eps=norm_layer_eps,
                norm_layer_momentum=norm_layer_momentum,
                activation_layer=True,
                is_lite=is_lite,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0

        for cnf in inverted_residual_setting:
            stage: List[torch.nn.Module] = []

            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(
                    block_cnf.block(
                        state_dict=state_dict,
                        base_address=f"blocks.{len(layers)-1}.{len(stage)}"
                        if is_lite
                        else f"features.{len(layers)}.{len(stage)}",
                        device=device,
                        cnf=block_cnf,
                        stochastic_depth_prob=sd_prob,
                        norm_layer_eps=norm_layer_eps,
                        norm_layer_momentum=norm_layer_momentum,
                        is_lite=is_lite,
                    )
                )

                stage_block_id += 1

            layers.append(torch.nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels

        if is_lite:
            lastconv_output_channels = 1280
        else:
            lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels

        layers.append(
            TtEfficientnetConv2dNormActivation(
                state_dict=state_dict,
                conv_base_address=f"head.0" if is_lite else f"features.{len(layers)}.0",
                bn_base_address=f"head.1" if is_lite else f"features.{len(layers)}.1",
                device=device,
                in_channels=lastconv_input_channels,
                out_channels=lastconv_output_channels,
                kernel_size=1,
                norm_layer_eps=norm_layer_eps,
                norm_layer_momentum=norm_layer_momentum,
                activation_layer=True,
                is_lite=is_lite,
            )
        )

        self.features = torch.nn.Sequential(*layers)
        self.avgpool = fallback_ops.AdaptiveAvgPool2d(1)

        self.classifier_weight = torch2tt_tensor(
            state_dict["fc.weight" if is_lite else "classifier.1.weight"],
            device,
            tt_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        bias_key = "fc.bias" if is_lite else "classifier.1.bias"

        if bias_key in state_dict:
            self.classifier_bias = torch2tt_tensor(
                state_dict[bias_key],
                device,
                tt_layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        else:
            self.classifier_bias = None

        self.classifier_weight = ttnn.transpose(self.classifier_weight, -2, -1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        last_shape = x.shape.with_tile_padding()[-1] * x.shape.with_tile_padding()[-2] * x.shape.with_tile_padding()[-3]
        # ttnn.reshape_on_device won't work here since input tensor is of shape [1, n, 1, 1]
        x = fallback_ops.reshape(x, x.shape.with_tile_padding()[0], 1, 1, last_shape)

        x = ttnn.matmul(x, self.classifier_weight)

        if self.classifier_bias is not None:
            x = ttnn.add(x, self.classifier_bias)

        return x


def _efficientnet(
    state_dict,
    device,
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    norm_layer_eps: float = 1e-05,
    norm_layer_momentum: float = 0.1,
    is_lite=False,
) -> TtEfficientNet:
    model = TtEfficientNet(
        state_dict=state_dict,
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        norm_layer_eps=norm_layer_eps,
        norm_layer_momentum=norm_layer_momentum,
        last_channel=last_channel,
        is_lite=is_lite,
    )

    return model


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]

    if arch.startswith("efficientnet_b") or arch.startswith("efficientnet_lite"):
        is_lite = arch.startswith("efficientnet_lite")
        bneck_conf = partial(
            MBConvConfig,
            width_mult=kwargs.pop("width_mult"),
            depth_mult=kwargs.pop("depth_mult"),
        )
        inverted_residual_setting = [
            bneck_conf(
                1,
                3,
                1,
                32,
                16,
                1,
                False if is_lite else True,
                False if is_lite else True,
            ),
            bneck_conf(6, 3, 2, 16, 24, 2, True, True),
            bneck_conf(6, 5, 2, 24, 40, 2, True, True),
            bneck_conf(6, 3, 2, 40, 80, 3, True, True),
            bneck_conf(6, 5, 1, 80, 112, 3, True, True),
            bneck_conf(6, 5, 2, 112, 192, 4, True, True),
            bneck_conf(6, 3, 1, 192, 320, 1, False if is_lite else True, True),
        ]

        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def efficientnet_b0(device) -> TtEfficientNet:
    """EfficientNet B0 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    reference_model = torchvision.models.efficientnet_b0(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.2,
        last_channel=last_channel,
    )


def efficientnet_b1(device) -> TtEfficientNet:
    """EfficientNet B1 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    reference_model = torchvision.models.efficientnet_b1(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.2,
        last_channel=last_channel,
    )


def efficientnet_b2(device) -> TtEfficientNet:
    """EfficientNet B2 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    reference_model = torchvision.models.efficientnet_b2(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.3,
        last_channel=last_channel,
    )


def efficientnet_b3(device) -> TtEfficientNet:
    """EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    reference_model = torchvision.models.efficientnet_b3(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.3,
        last_channel=last_channel,
    )


def efficientnet_b4(device) -> TtEfficientNet:
    """EfficientNet B4 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    reference_model = torchvision.models.efficientnet_b4(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.4,
        last_channel=last_channel,
    )


def efficientnet_b5(device) -> TtEfficientNet:
    """EfficientNet B5 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    reference_model = torchvision.models.efficientnet_b5(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.4,
        last_channel=last_channel,
        norm_layer_eps=0.001,
        norm_layer_momentum=0.01,
    )


def efficientnet_b6(device) -> TtEfficientNet:
    """EfficientNet B6 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    reference_model = torchvision.models.efficientnet_b6(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.5,
        last_channel=last_channel,
        norm_layer_eps=0.001,
        norm_layer_momentum=0.01,
    )


def efficientnet_b7(device) -> TtEfficientNet:
    """EfficientNet B7 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    reference_model = torchvision.models.efficientnet_b7(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.5,
        last_channel=last_channel,
        norm_layer_eps=0.001,
        norm_layer_momentum=0.01,
    )


def efficientnet_v2_s(device) -> TtEfficientNet:
    """
    Constructs an EfficientNetV2-S architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    reference_model = torchvision.models.efficientnet_v2_s(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.2,
        last_channel=last_channel,
        norm_layer_eps=1e-03,
    )


def efficientnet_v2_m(device) -> TtEfficientNet:
    """
    Constructs an EfficientNetV2-M architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    reference_model = torchvision.models.efficientnet_v2_m(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.3,
        last_channel=last_channel,
        norm_layer_eps=1e-03,
    )


def efficientnet_v2_l(device) -> TtEfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    reference_model = torchvision.models.efficientnet_v2_l(pretrained=True)
    reference_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.4,
        last_channel=last_channel,
        norm_layer_eps=1e-03,
    )


import requests


def save_file_from_url(url, path):
    response = requests.get(url)
    open(path, "wb").write(response.content)


def reference_efficientnet_lite0(pretrained: bool = True) -> torch.nn.Module:
    reference_model = build_efficientnet_lite("efficientnet_lite0", 1000)

    save_path = "./efficientnet_lite0.pth"
    save_file_from_url(
        "https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite0.pth",
        save_path,
    )
    logger.info(f"Weights for efficientnet_lite0 saved to '{save_path}'")

    reference_model.load_pretrain(save_path)
    reference_model.eval()

    return reference_model


def reference_efficientnet_lite1(pretrained: bool = True) -> torch.nn.Module:
    reference_model = build_efficientnet_lite("efficientnet_lite1", 1000)

    save_path = "./efficientnet_lite1.pth"
    save_file_from_url(
        "https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite1.pth",
        save_path,
    )
    logger.info(f"Weights for efficientnet_lite1 saved to '{save_path}'")

    reference_model.load_pretrain(save_path)
    reference_model.eval()

    return reference_model


def reference_efficientnet_lite2(pretrained: bool = True) -> torch.nn.Module:
    reference_model = build_efficientnet_lite("efficientnet_lite2", 1000)

    save_path = "./efficientnet_lite2.pth"
    save_file_from_url(
        "https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite2.pth",
        save_path,
    )
    logger.info(f"Weights for efficientnet_lite2 saved to '{save_path}'")

    reference_model.load_pretrain(save_path)
    reference_model.eval()

    return reference_model


def reference_efficientnet_lite3(pretrained: bool = True) -> torch.nn.Module:
    reference_model = build_efficientnet_lite("efficientnet_lite3", 1000)

    save_path = "./efficientnet_lite3.pth"
    save_file_from_url(
        "https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite3.pth",
        save_path,
    )
    logger.info(f"Weights for efficientnet_lite3 saved to '{save_path}'")

    reference_model.load_pretrain(save_path)
    reference_model.eval()

    return reference_model


def reference_efficientnet_lite4(pretrained: bool = True) -> torch.nn.Module:
    reference_model = build_efficientnet_lite("efficientnet_lite4", 1000)

    save_path = "./efficientnet_lite4.pth"
    save_file_from_url(
        "https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite4.pth",
        save_path,
    )
    logger.info(f"Weights for efficientnet_lite4 saved to '{save_path}'")

    reference_model.load_pretrain(save_path)
    reference_model.eval()

    return reference_model


# efficientnet_lite_params = {
#     # width_coefficient, depth_coefficient, image_size, dropout_rate
#     "efficientnet_lite0": [1.0, 1.0, 224, 0.2],
def efficientnet_lite0(device) -> TtEfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    reference_model = reference_efficientnet_lite0()
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_lite0", width_mult=1.0, depth_mult=1.0)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.2,
        last_channel=last_channel,
        norm_layer_eps=1e-3,
        norm_layer_momentum=0.01,
        is_lite=True,
    )


# efficientnet_lite_params = {
#     # width_coefficient, depth_coefficient, image_size, dropout_rate
#     "efficientnet_lite1": [1.0, 1.1, 240, 0.2],
def efficientnet_lite1(device) -> TtEfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    reference_model = reference_efficientnet_lite1()
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_lite1", width_mult=1.0, depth_mult=1.1)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.2,
        last_channel=last_channel,
        norm_layer_eps=1e-3,
        norm_layer_momentum=0.01,
        is_lite=True,
    )


# efficientnet_lite_params = {
#     # width_coefficient, depth_coefficient, image_size, dropout_rate
#     "efficientnet_lite2": [1.1, 1.2, 260, 0.3],
def efficientnet_lite2(device) -> TtEfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    reference_model = reference_efficientnet_lite2()
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_lite2", width_mult=1.1, depth_mult=1.2)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.3,
        last_channel=last_channel,
        norm_layer_eps=1e-3,
        norm_layer_momentum=0.01,
        is_lite=True,
    )


# efficientnet_lite_params = {
#     # width_coefficient, depth_coefficient, image_size, dropout_rate
#     "efficientnet_lite3": [1.2, 1.4, 280, 0.3],
def efficientnet_lite3(device) -> TtEfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    reference_model = reference_efficientnet_lite3()
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_lite3", width_mult=1.2, depth_mult=1.4)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.3,
        last_channel=last_channel,
        norm_layer_eps=1e-3,
        norm_layer_momentum=0.01,
        is_lite=True,
    )


# efficientnet_lite_params = {
#     # width_coefficient, depth_coefficient, image_size, dropout_rate
#     "efficientnet_lite4": [1.4, 1.8, 300, 0.3],
def efficientnet_lite4(device) -> TtEfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    reference_model = reference_efficientnet_lite4()
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_lite4", width_mult=1.4, depth_mult=1.8)

    return _efficientnet(
        state_dict=reference_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.3,
        last_channel=last_channel,
        norm_layer_eps=1e-3,
        norm_layer_momentum=0.01,
        is_lite=True,
    )
