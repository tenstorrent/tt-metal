# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttnn
from dataclasses import dataclass
from models.experimental.retinanet.tt.utils import TTUpsample
from collections import OrderedDict

from models.tt_cnn.tt.builder import TtConv2d
from models.tt_cnn.tt.builder import Conv2dConfiguration


@dataclass
class FpnOptimizer:
    conv1: dict
    conv2: dict
    conv3: dict
    conv4: dict
    conv5: dict
    conv6: dict
    conv7: dict
    conv8: dict


fpn_optimisations = FpnOptimizer(
    conv1={
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
    },
    conv2={
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
    },
    conv3={
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
    },
    conv4={
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
    },
    conv5={
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
    },
    conv6={
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
    },
    conv7={
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
    },
    conv8={
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
    },
)


class resnet50Fpn:
    def __init__(
        self,
        device,
        parameters,
        model_config,
        model_args,
        layer_optimisations=fpn_optimisations,
    ) -> None:
        self.conv_config_1 = Conv2dConfiguration.from_model_args(
            model_args["inner_blocks"][0],
            weights=parameters["inner_blocks"].get("0", {}).get("0", None)["weight"],
            bias=parameters["inner_blocks"].get("0", {}).get("0", None)["bias"],
            **layer_optimisations.conv1,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv1 = TtConv2d(self.conv_config_1, device)

        self.conv_config_2 = Conv2dConfiguration.from_model_args(
            model_args["inner_blocks"][1],
            weights=parameters["inner_blocks"].get("1", {}).get("0", None)["weight"],
            bias=parameters["inner_blocks"].get("1", {}).get("0", None)["bias"],
            **layer_optimisations.conv2,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv2 = TtConv2d(self.conv_config_2, device)

        self.conv_config_3 = Conv2dConfiguration.from_model_args(
            model_args["inner_blocks"][2],
            weights=parameters["inner_blocks"].get("2", {}).get("0", None)["weight"],
            bias=parameters["inner_blocks"].get("2", {}).get("0", None)["bias"],
            **layer_optimisations.conv3,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv3 = TtConv2d(self.conv_config_3, device)

        self.conv_config_4 = Conv2dConfiguration.from_model_args(
            model_args["layer_blocks"][0],
            weights=parameters["layer_blocks"].get("0", {}).get("0", None)["weight"],
            bias=parameters["layer_blocks"].get("0", {}).get("0", None)["bias"],
            **layer_optimisations.conv4,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv4 = TtConv2d(self.conv_config_4, device)

        self.conv_config_5 = Conv2dConfiguration.from_model_args(
            model_args["layer_blocks"][1],
            weights=parameters["layer_blocks"].get("1", {}).get("0", None)["weight"],
            bias=parameters["layer_blocks"].get("1", {}).get("0", None)["bias"],
            **layer_optimisations.conv5,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv5 = TtConv2d(self.conv_config_5, device)

        self.conv_config_6 = Conv2dConfiguration.from_model_args(
            model_args["layer_blocks"][2],
            weights=parameters["layer_blocks"].get("2", {}).get("0", None)["weight"],
            bias=parameters["layer_blocks"].get("2", {}).get("0", None)["bias"],
            **layer_optimisations.conv6,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv6 = TtConv2d(self.conv_config_6, device)

        self.conv_config_7 = Conv2dConfiguration.from_model_args(
            model_args["extra_blocks"]["p6"],
            weights=getattr(parameters.extra_blocks, "p6", None)["weight"],
            bias=getattr(parameters.extra_blocks, "p6", None)["bias"],
            **layer_optimisations.conv7,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv7 = TtConv2d(self.conv_config_7, device)

        self.conv_config_8 = Conv2dConfiguration.from_model_args(
            model_args["extra_blocks"]["p7"],
            weights=getattr(parameters.extra_blocks, "p7", None)["weight"],
            bias=getattr(parameters.extra_blocks, "p7", None)["bias"],
            **layer_optimisations.conv8,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv8 = TtConv2d(self.conv_config_8, device)

        self.upsample1 = TTUpsample(
            scale_factor=(2),
            mode="nearest",
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

        self.upsample2 = TTUpsample(
            scale_factor=(2),
            mode="nearest",
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

    def __call__(
        self,
        x,
        device,
    ):
        C3, C4, C5 = x.values()
        C5_clone = ttnn.clone(C5)

        L3, [_out_height, _out_width] = self.conv1(C3, return_output_dim=True)
        L4, [_out_height, _out_width] = self.conv2(C4, return_output_dim=True)
        L5, [_out_height, _out_width] = self.conv3(C5, return_output_dim=True)

        P5 = L5
        P5_shape = (1, _out_height, _out_width, 256)
        P5_interpolated = self.upsample1(device, P5, P5_shape, reshape_output=True, sent_to_dram=False)

        P4 = ttnn.add(L4, P5_interpolated)
        P4_shape = (1, 32, 32, 256)
        P4_interpolated = self.upsample1(device, P4, P4_shape, reshape_output=True, sent_to_dram=False)

        P3 = ttnn.add(L3, P4_interpolated)

        P3, [_out_height, _out_width] = self.conv4(P3, return_output_dim=True)
        P4, [_out_height, _out_width] = self.conv5(P4, return_output_dim=True)
        P5, [_out_height, _out_width] = self.conv6(P5, return_output_dim=True)

        P6, [_out_height, _out_width] = self.conv7(C5_clone, return_output_dim=True)
        P6_relu = ttnn.relu(P6)
        P7, [_out_height, _out_width] = self.conv8(P6_relu, return_output_dim=True)

        out = OrderedDict([("0", P3), ("1", P4), ("2", P5), ("p6", P6), ("p7", P7)])
        return out
