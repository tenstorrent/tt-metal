# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false

import ttnn
from dataclasses import dataclass
from models.experimental.retinanet.TTNN.utils import TTConv2D, TTUpsample
from collections import OrderedDict


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
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv2={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv3={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv4={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv5={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv6={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv7={
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
    conv8={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "dtype": ttnn.bfloat16,
    },
)


class resnet50Fpn:
    def __init__(
        self,
        parameters,
        model_config,
        layer_optimisations=fpn_optimisations,
    ) -> None:
        self.conv1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["inner_blocks"].get("0", {}).get("0", None),
            kernel_fidelity=model_config,
            activation=None,
            **layer_optimisations.conv1,
        )

        self.conv2 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["inner_blocks"].get("1", {}).get("0", None),
            kernel_fidelity=model_config,
            activation=None,
            **layer_optimisations.conv2,
        )

        self.conv3 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters["inner_blocks"].get("2", {}).get("0", None),
            kernel_fidelity=model_config,
            activation=None,
            **layer_optimisations.conv3,
        )

        self.conv4 = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=1,
            parameters=parameters["layer_blocks"].get("0", {}).get("0", None),
            kernel_fidelity=model_config,
            activation=None,
            **layer_optimisations.conv4,
        )

        self.conv5 = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=1,
            parameters=parameters["layer_blocks"].get("1", {}).get("0", None),
            kernel_fidelity=model_config,
            activation=None,
            **layer_optimisations.conv5,
        )

        self.conv6 = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=1,
            parameters=parameters["layer_blocks"].get("2", {}).get("0", None),
            kernel_fidelity=model_config,
            activation=None,
            **layer_optimisations.conv6,
        )

        self.conv7 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=getattr(parameters.extra_blocks, "p6", None),
            kernel_fidelity=model_config,
            activation=None,
            **layer_optimisations.conv7,
        )

        self.conv8 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            parameters=getattr(parameters.extra_blocks, "p7", None),
            kernel_fidelity=model_config,
            activation=None,
            **layer_optimisations.conv8,
        )

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

        L3, _ = self.conv1(device, C3, C3.shape)
        L4, _ = self.conv2(device, C4, C4.shape)
        L5, _ = self.conv3(device, C5, C5.shape)

        P5 = L5
        P5_interpolated = self.upsample1(device, P5, P5.shape, reshape_output=False, sent_to_dram=False)

        P4 = ttnn.add(L4, P5_interpolated)
        P4_interpolated = self.upsample1(device, P4, P4.shape, reshape_output=False, sent_to_dram=False)

        P3 = ttnn.add(L3, P4_interpolated)

        P3, _ = self.conv4(device, P3, P3.shape)
        P4, _ = self.conv5(device, P4, P4.shape)
        P5, _ = self.conv6(device, P5, P5.shape)

        P6, _ = self.conv7(device, C5_clone, C5_clone.shape)
        P6_relu = ttnn.relu(P6)
        P7, _ = self.conv8(device, P6_relu, P6_relu.shape)

        out = OrderedDict([("0", P3), ("1", P4), ("2", P5), ("p6", P6), ("p7", P7)])
        return out
