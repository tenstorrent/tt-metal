# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov6l.tt.ttnn_bottlerep import TtBottleRep


class TtRepBlock:
    def __init__(
        self,
        device,
        parameters,
        model_params,
        n=1,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        shard_layout_rep_block_first_two=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ):
        self.parameters = parameters
        self.model_params = model_params
        self.conv1 = TtBottleRep(
            device, parameters.conv1, model_params.conv1, shard_layout=shard_layout_rep_block_first_two
        )
        n = n // 2
        self.n_blocks = n - 1
        for i in range(self.n_blocks):
            if i > 1:
                shard_layout = shard_layout
            else:
                shard_layout = shard_layout_rep_block_first_two
            setattr(
                self,
                f"bottle_rep{i}",
                TtBottleRep(device, parameters.block[i], model_params.block[i], shard_layout=shard_layout),
            )

    def __call__(self, inpur_tensor):
        output, out_h, out_w = self.conv1(inpur_tensor)
        for i in range(self.n_blocks):
            block = getattr(self, f"bottle_rep{i}")
            output, out_h, out_w = block(output)
        return output, out_h, out_w
