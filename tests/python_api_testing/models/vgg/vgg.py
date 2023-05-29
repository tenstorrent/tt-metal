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
import torchvision
from torchvision import models
import torchvision.transforms as transforms

from libs import tt_lib
from typing import List, Union, Optional, Dict, cast
from utils import tt_linear, get_shape, is_torch_tensor
from python_api_testing.models.conv_on_device_utils import (
    is_conv_supported_on_device,
    run_conv_on_device_wrapper,
)


num_classes = 1000


class TtVGG(nn.Module):
    def __init__(
        self,
        features: List,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
        device=None,
        host=None,
        state_dict=None,
        base_address="",
    ) -> None:
        super().__init__()
        assert init_weights == False, "we are loading weights, not initializing them"
        self.device = device
        self.host = host
        self.state_dict = state_dict
        self.base_address = base_address

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert batch_size == 1
        for layer in self.features:
            if layer is tt_lib.tensor.relu:
                if is_torch_tensor(x):
                    tt_x = tt_lib.tensor.Tensor(
                        x.reshape(-1).tolist(),
                        x.size(),
                        tt_lib.tensor.DataType.BFLOAT16,
                        tt_lib.tensor.Layout.ROW_MAJOR,
                    )
                x = layer(tt_x)
            else:
                if not is_torch_tensor(x):
                    x = x.to(self.host)
                    x = torch.Tensor(x.data()).reshape(x.shape())
                x = layer(x)

        if not is_torch_tensor(x):
            x = x.to(self.host)
            x = torch.Tensor(x.data()).reshape(x.shape())

        x = self.avgpool(x)
        x = torch.flatten(x, 1).unsqueeze(1).unsqueeze(1)

        x = tt_lib.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.size(),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        for layer in self.classifier:
            x = layer(x)

        x = x.to(self.host)
        x = torch.Tensor(x.data()).reshape(x.shape())
        return x


def make_layers(
    cfg: List[Union[str, int]],
    batch_norm: bool = False,
    state_dict=None,
    base_address="features",
    device=None,
    host=None,
    disable_conv_on_tt_device=True,
) -> nn.Sequential:
    layers: List = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

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
                        host,
                        conv2d_bias,
                    )
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    conv2d.weight = nn.Parameter(
                        state_dict[f"{base_address}.{ind}.weight"]
                    )
                    conv2d.bias = nn.Parameter(state_dict[f"{base_address}.{ind}.bias"])
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


def vgg16(
    device, host, state_dict, base_address="", disable_conv_on_tt_device=True
) -> TtVGG:
    model = TtVGG(
        make_layers(
            cfgs["D"],
            batch_norm=False,
            state_dict=state_dict,
            device=device,
            host=host,
            disable_conv_on_tt_device=disable_conv_on_tt_device,
        ),
        init_weights=False,
        device=device,
        host=host,
        state_dict=state_dict,
    )
    return model
