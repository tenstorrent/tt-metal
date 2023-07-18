import torch
import copy
import tt_lib
from typing import List, Sequence, Union, Tuple, Optional, Any
from tt_lib.fallback_ops import fallback_ops
import torchvision
from functools import partial
from loguru import logger

from tests.python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from tests.python_api_testing.models.EfficientNet.tt.efficientnet_conv import (
    TtEfficientnetConv2dNormActivation,
)
from tests.python_api_testing.models.EfficientNet.tt.efficientnet_mbconv import (
    TtEfficientnetMbConv,
    _MBConvConfig,
    MBConvConfig,
)
from tests.python_api_testing.models.EfficientNet.tt.efficientnet_fused_mbconv import (
    TtEfficientnetFusedMBConv,
    FusedMBConvConfig,
)


def flatten_via_reshape(x, start_dim):
    shape = x.shape()


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
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
            )

        layers: List[torch.nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels

        layers.append(
            TtEfficientnetConv2dNormActivation(
                state_dict=state_dict,
                base_address=f"features.{len(layers)}",
                device=device,
                in_channels=3,
                out_channels=firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer_eps=norm_layer_eps,
                norm_layer_momentum=norm_layer_momentum,
                activation_layer=True,
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
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )

                stage.append(
                    block_cnf.block(
                        state_dict=state_dict,
                        base_address=f"features.{len(layers)}.{len(stage)}",
                        device=device,
                        cnf=block_cnf,
                        stochastic_depth_prob=sd_prob,
                        norm_layer_eps=norm_layer_eps,
                        norm_layer_momentum=norm_layer_momentum,
                    )
                )

                stage_block_id += 1

            layers.append(torch.nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )

        layers.append(
            TtEfficientnetConv2dNormActivation(
                state_dict=state_dict,
                base_address=f"features.{len(layers)}",
                device=device,
                in_channels=lastconv_input_channels,
                out_channels=lastconv_output_channels,
                kernel_size=1,
                norm_layer_eps=norm_layer_eps,
                norm_layer_momentum=norm_layer_momentum,
                activation_layer=True,
            )
        )

        self.features = torch.nn.Sequential(*layers)
        self.avgpool = fallback_ops.AdaptiveAvgPool2d(1)

        self.classifier_weight = torch2tt_tensor(
            state_dict[f"classifier.1.weight"],
            device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
        )

        if "classifier.1.bias" in state_dict:
            self.classifier_bias = torch2tt_tensor(
                state_dict[f"classifier.1.bias"],
                device,
                tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            )
        else:
            self.classifier_bias = None

        self.classifier_weight = tt_lib.tensor.transpose(self.classifier_weight)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        last_shape = x.shape()[-1] * x.shape()[-2] * x.shape()[-3]
        # tt_lib.tensor.reshape won't work here since input tensor is of shape [1, n, 1, 1]
        x = tt_lib.fallback_ops.reshape(x, x.shape()[0], 1, 1, last_shape)

        x = tt_lib.tensor.matmul(x, self.classifier_weight)

        if self.classifier_bias is not None:
            x = tt_lib.tensor.add(x, self.classifier_bias)

        return x


def _efficientnet(
    state_dict,
    device,
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    norm_layer_eps: float = 1e-05,
    norm_layer_momentum: float = 0.1,
) -> TtEfficientNet:
    model = TtEfficientNet(
        state_dict=state_dict,
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        norm_layer_eps=norm_layer_eps,
        norm_layer_momentum=norm_layer_momentum,
        last_channel=last_channel,
    )

    return model


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]

    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(
            MBConvConfig,
            width_mult=kwargs.pop("width_mult"),
            depth_mult=kwargs.pop("depth_mult"),
        )
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
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

    refence_model = torchvision.models.efficientnet_b0(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b0", width_mult=1.0, depth_mult=1.0
    )

    return _efficientnet(
        state_dict=refence_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.2,
        last_channel=last_channel,
    )


def efficientnet_b1(device) -> TtEfficientNet:
    """EfficientNet B1 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    refence_model = torchvision.models.efficientnet_b1(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b1", width_mult=1.0, depth_mult=1.1
    )

    return _efficientnet(
        state_dict=refence_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.2,
        last_channel=last_channel,
    )


def efficientnet_b2(device) -> TtEfficientNet:
    """EfficientNet B2 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    refence_model = torchvision.models.efficientnet_b2(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b2", width_mult=1.1, depth_mult=1.2
    )

    return _efficientnet(
        state_dict=refence_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.3,
        last_channel=last_channel,
    )


def efficientnet_b3(device) -> TtEfficientNet:
    """EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    refence_model = torchvision.models.efficientnet_b3(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b3", width_mult=1.2, depth_mult=1.4
    )

    return _efficientnet(
        state_dict=refence_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.3,
        last_channel=last_channel,
    )


def efficientnet_b4(device) -> TtEfficientNet:
    """EfficientNet B4 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    refence_model = torchvision.models.efficientnet_b4(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b4", width_mult=1.4, depth_mult=1.8
    )

    return _efficientnet(
        state_dict=refence_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.4,
        last_channel=last_channel,
    )


def efficientnet_b5(device) -> TtEfficientNet:
    """EfficientNet B5 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    refence_model = torchvision.models.efficientnet_b5(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b5", width_mult=1.6, depth_mult=2.2
    )

    return _efficientnet(
        state_dict=refence_model.state_dict(),
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

    refence_model = torchvision.models.efficientnet_b6(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b6", width_mult=1.8, depth_mult=2.6
    )

    return _efficientnet(
        state_dict=refence_model.state_dict(),
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

    refence_model = torchvision.models.efficientnet_b7(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b7", width_mult=2.0, depth_mult=3.1
    )

    return _efficientnet(
        state_dict=refence_model.state_dict(),
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

    refence_model = torchvision.models.efficientnet_v2_s(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")

    return _efficientnet(
        state_dict=refence_model.state_dict(),
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

    refence_model = torchvision.models.efficientnet_v2_m(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")

    return _efficientnet(
        state_dict=refence_model.state_dict(),
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

    refence_model = torchvision.models.efficientnet_v2_l(pretrained=True)
    refence_model.eval()

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")

    return _efficientnet(
        state_dict=refence_model.state_dict(),
        device=device,
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.4,
        last_channel=last_channel,
        norm_layer_eps=1e-03,
    )
