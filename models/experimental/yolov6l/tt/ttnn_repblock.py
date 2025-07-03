# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.yolov6l.tt.ttnn_bottlerep import TtBottleRep


class TtRepBlock:
    def __init__(self, device, parameters, model_params, n=1):
        self.parameters = parameters
        self.model_params = model_params
        self.conv1 = TtBottleRep(device, parameters.conv1, model_params.conv1)
        n = n // 2
        self.n_blocks = n - 1
        for i in range(self.n_blocks):
            setattr(
                self,
                f"bottle_rep{i}",
                TtBottleRep(device, parameters.block[i], model_params.block[i]),
            )

    def __call__(self, inpur_tensor):
        output = self.conv1(inpur_tensor)
        for i in range(self.n_blocks):
            block = getattr(self, f"bottle_rep{i}")
            output = block(output)
        return output
