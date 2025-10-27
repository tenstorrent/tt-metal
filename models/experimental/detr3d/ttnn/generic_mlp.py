# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.ttnn.utils import TtnnConv1D


class TtnnGenericMLP(LightweightModule):
    def __init__(
        self,
        parameters,
        device,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        output_use_activation=False,
        use_conv=True,
        deallocate_activation=False,
    ):
        super().__init__()
        assert use_conv, f"Currently only supports Conv1d"
        self.deallocate_activation = deallocate_activation

        layer_args = list()
        layer_params = list()
        for idx in range(len(parameters.layers)):
            if parameters.layers[idx]:
                layer_args.append(parameters.conv_args.layers[str(idx)])
                layer_params.append(parameters.layers[idx])

        self.layers = list()
        for conv_args, parameters in zip(layer_args[:-1], layer_params[:-1]):
            self.layers.append(
                TtnnConv1D(
                    conv_args,
                    parameters,
                    device,
                    activation=activation,
                    return_dims=True,
                    deallocate_activation=True,
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                )
            )
        self.layers.append(
            TtnnConv1D(
                layer_args[-1],
                layer_params[-1],
                device,
                activation=activation if output_use_activation else None,
                return_dims=True,
                deallocate_activation=True,
                math_fidelity=ttnn.MathFidelity.HiFi2,
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            )
        )

    def forward(self, input):
        shape = input.shape
        out = ttnn.clone(input)
        if self.deallocate_activation:
            ttnn.deallocate(input)
        for layer in self.layers:
            out, shape = layer(out, shape)
        out = ttnn.reshape(out, shape)
        return out
