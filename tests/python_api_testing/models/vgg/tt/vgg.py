from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
from torchvision import models

import tt_lib
from tt_lib.fallback_ops import fallback_ops
from typing import List, Union, Dict, cast
from vgg_utils import get_shape
from vgg_helper_funcs import tt_linear
from tt_models.utility_functions import (
    is_conv_supported_on_device,
    run_conv_on_device_wrapper,
)

from tt_models.utility_functions import torch_to_tt_tensor_rm

num_classes = 1000


class TtVGG(nn.Module):
    def __init__(
        self,
        features: List,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
        device=None,
        state_dict=None,
        base_address="",
    ) -> None:
        super().__init__()
        assert init_weights == False, "we are loading weights, not initializing them"
        self.device = device
        self.state_dict = state_dict
        self.base_address = base_address

        self.features = features
        self.avgpool = fallback_ops.AdaptiveAvgPool2d((7, 7))

        linear1_weight = state_dict[f"classifier.0.weight"]
        linear1_weight = tt_lib.tensor.Tensor(
            linear1_weight.reshape(-1).tolist(),
            get_shape(linear1_weight.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        linear1_bias = state_dict[f"classifier.0.bias"]
        linear1_bias = tt_lib.tensor.Tensor(
            linear1_bias.reshape(-1).tolist(),
            get_shape(linear1_bias.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        linear2_weight = state_dict[f"classifier.3.weight"]
        linear2_weight = tt_lib.tensor.Tensor(
            linear2_weight.reshape(-1).tolist(),
            get_shape(linear2_weight.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        linear2_bias = state_dict[f"classifier.3.bias"]
        linear2_bias = tt_lib.tensor.Tensor(
            linear2_bias.reshape(-1).tolist(),
            get_shape(linear2_bias.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        linear3_weight = state_dict[f"classifier.6.weight"]
        linear3_weight = tt_lib.tensor.Tensor(
            linear3_weight.reshape(-1).tolist(),
            get_shape(linear3_weight.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        linear3_bias = state_dict[f"classifier.6.bias"]
        linear3_bias = tt_lib.tensor.Tensor(
            linear3_bias.reshape(-1).tolist(),
            get_shape(linear3_bias.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        linear1 = tt_linear(linear1_weight, linear1_bias, self.device)

        linear2 = tt_linear(linear2_weight, linear2_bias, self.device)

        linear3 = tt_linear(linear3_weight, linear3_bias, self.device)

        self.classifier = [
            linear1,
            tt_lib.tensor.relu,
            linear2,
            tt_lib.tensor.relu,
            linear3,
        ]

    def forward(self, tt_x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        batch_size = tt_x.shape()[0]
        assert batch_size == 1

        for layer in self.features:
            if layer is tt_lib.tensor.relu:
                tt_x = layer(tt_x)
            else:
                tt_x = layer(tt_x)

        batch, c, w, h = tt_x.shape()

        tt_x = self.avgpool(tt_x)
        tt_x = fallback_ops.reshape(tt_x, batch, 1, 1, c * w * h)
        for layer in self.classifier:
            tt_x = layer(tt_x)

        return tt_x


def make_layers(
    cfg: List[Union[str, int]],
    batch_norm: bool = False,
    state_dict=None,
    base_address="features",
    device=None,
    disable_conv_on_tt_device=True,
) -> nn.Sequential:
    layers: List = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers += [fallback_ops.MaxPool2d(kernel_size=2, stride=2)]

        else:
            v = cast(int, v)
            if not batch_norm:
                ind = len(layers)
                conv2d_params = [v, in_channels, 3, 3, 1, 1, 1, 1, 1, 1]
                if not disable_conv_on_tt_device and is_conv_supported_on_device(
                    conv2d_params
                ):
                    assert device is not None
                    conv2d_weight = state_dict[f"{base_address}.{ind}.weight"]
                    conv2d_bias = state_dict[f"{base_address}.{ind}.bias"].tolist()
                    conv2d = run_conv_on_device_wrapper(
                        conv2d_weight.reshape(-1).tolist(),
                        conv2d_params,
                        device,
                        conv2d_bias,
                    )
                else:
                    weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.{ind}.weight"], device=device, put_on_device=False)
                    bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.{ind}.bias"], device=device, put_on_device=False)
                    conv2d = fallback_ops.Conv2d(
                                weights=weight,
                                biases=bias,
                                in_channels=in_channels,
                                out_channels=v,
                                kernel_size=3,
                                padding=1
                                )

                layers += [conv2d, tt_lib.tensor.relu]
            else:
                assert False, "we do not support batchnorm"
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v

    return layers


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(features, init_weights, device, state_dict, base_address=""):
    return TtVGG(
                features,
                init_weights=init_weights,
                device=device,
                state_dict=state_dict,
                base_address=base_address
                )

def vgg16(device, disable_conv_on_tt_device=True) -> TtVGG:

    torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_vgg.eval()
    state_dict = torch_vgg.state_dict()
    model = _vgg(
        make_layers(
            cfgs["D"],
            batch_norm=False,
            state_dict=state_dict,
            device=device,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        ),
        init_weights=False,
        device=device,
        state_dict=state_dict,
    )
    return model

def vgg11(device, disable_conv_on_tt_device=True) -> TtVGG:

    torch_vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    torch_vgg.eval()
    state_dict = torch_vgg.state_dict()
    model = _vgg(
        make_layers(
            cfgs["A"],
            batch_norm=False,
            state_dict=state_dict,
            device=device,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        ),
        init_weights=False,
        device=device,
        state_dict=state_dict,
    )
    return model
