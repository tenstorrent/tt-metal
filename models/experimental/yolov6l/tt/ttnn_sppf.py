# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov6l.tt.common import Yolov6l_Conv2D


class TtSppf:
    def __init__(self, device, parameters, model_params):
        self.parameters = parameters
        self.model_params = model_params
        self.cv1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv1.block.conv,
            conv_pth=parameters.cv1.block.conv,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            activation="silu",
            deallocate_activation=True,
        )
        self.cv2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cv2.block.conv,
            conv_pth=parameters.cv2.block.conv,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            activation="silu",
            reshape=True,
        )

    def __call__(self, x):
        conv1 = self.cv1(x)
        y = [conv1]
        for i in range(3):
            output = ttnn.max_pool2d(
                input_tensor=(y[-1] if y[-1].is_sharded() else y[-1]),
                batch_size=1,
                input_h=20,
                input_w=20,
                channels=y[-1].shape[-1],
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[2, 2],
                dilation=[1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                applied_shard_scheme=None if y[-1].is_sharded() else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            )
            y.append(output)

        for i in range(len(y)):
            y[i] = ttnn.sharded_to_interleaved(y[i])
            y[i] = ttnn.to_layout(y[i], ttnn.ROW_MAJOR_LAYOUT)
        concat_output = ttnn.concat(y, dim=-1)

        for i in range(len(y)):
            ttnn.deallocate(y[i])

        conv2 = self.cv2(concat_output)
        return conv2
