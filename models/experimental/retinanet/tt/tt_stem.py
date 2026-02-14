# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttnn
from dataclasses import dataclass
from models.experimental.retinanet.tt.utils import MaxPoolConfiguration
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d
import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)


@dataclass
class NeckOptimizer:
    conv1: dict


neck_optimisations = NeckOptimizer(
    conv1={
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    }
)


class resnet50Stem:
    def __init__(
        self,
        parameters,
        device,
        model_config,
        model_args,
        layer_optimisations=neck_optimisations,
    ) -> None:
        self.conv_config_1 = Conv2dConfiguration.from_model_args(
            model_args["conv1"],
            weights=parameters.conv1["weight"],
            bias=parameters.conv1["bias"],
            **layer_optimisations.conv1,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv1 = TtConv2d(self.conv_config_1, device)

        self.maxpool_config = MaxPoolConfiguration.from_model_args(model_args["maxpool"])
        self.maxpool = TtMaxPool2d(self.maxpool_config, device)

    def __call__(
        self,
        x,
        device,
    ):
        # conv1 is stride 2 conv 3x3
        out, shape = self.conv1(x, return_output_dim=True)
        out = self.maxpool(out)
        out = ttnn.reshape(out, (1, shape[-2] // 2, shape[-1] // 2, 64))

        return out
