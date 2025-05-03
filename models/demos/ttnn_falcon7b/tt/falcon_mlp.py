# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.utility_functions import is_wormhole_b0


class TtFalconMLP:
    def __init__(self, model_config, parameters):
        super().__init__()
        self.model_config = model_config
        self.dense_h_to_4h_weights = parameters.dense_h_to_4h.weight
        self.dense_4h_to_h_weights = parameters.dense_4h_to_h.weight
        if is_wormhole_b0():
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
            )
            self.core_grid = ttnn.CoreGrid(y=7, x=8)
        else:
            self.compute_kernel_config = None
            self.core_grid = ttnn.CoreGrid(y=9, x=12)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ff1_linear: ttnn.Tensor = ttnn.linear(
            x,
            self.dense_h_to_4h_weights,
            memory_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
            activation="gelu",
            compute_kernel_config=self.compute_kernel_config,
            core_grid=self.core_grid,
        )
        ff2_linear: ttnn.Tensor = ttnn.linear(
            ff1_linear,
            self.dense_4h_to_h_weights,
            memory_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.compute_kernel_config,
            core_grid=self.core_grid,
        )
        ttnn.deallocate(ff1_linear)

        return ff2_linear
