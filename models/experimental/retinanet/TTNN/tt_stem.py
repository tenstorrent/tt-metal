# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false

import ttnn
from dataclasses import dataclass
from models.experimental.retinanet.TTNN.utils import TTConv2D


@dataclass
class NeckOptimizer:
    conv1: dict


neck_optimisations = NeckOptimizer(
    conv1={
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
        # "reshard_if_not_optimal": True,
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        # "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=2),
        "dtype": ttnn.bfloat16,
    }
)


class resnet50Stem:
    def __init__(
        self,
        parameters,
        stride,
        model_config,
        layer_optimisations=neck_optimisations,
    ) -> None:
        self.conv1 = TTConv2D(
            kernel_size=7,
            stride=2,
            padding=3,
            parameters=parameters.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            **layer_optimisations.conv1,
        )

    def __call__(
        self,
        x,
        device,
    ):
        # conv1 is stride 2 conv 3x3
        out, shape = self.conv1(device, x, x.shape)

        out = ttnn.max_pool2d(
            input_tensor=out,
            batch_size=shape[-4],
            input_h=shape[-3],
            input_w=shape[-2],
            channels=shape[-1],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            in_place_halo=True,
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ceil_mode=False,
        )
        out = ttnn.reshape(out, (shape[-4], shape[-3] // 2, shape[-2] // 2, shape[-1]))

        return out
