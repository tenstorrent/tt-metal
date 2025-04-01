# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.experimental.functional_yolov6x.tt.common import Yolov6x_Conv2D, sharded_concat, deallocate_tensors


class Ttnn_Sppf:
    def __init__(self, device, parameter, model_params):
        self.parameter = parameter
        self.model_params = model_params
        self.cv1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.cv1.conv,
            conv_pth=parameter.cv1.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
        )
        self.cv2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.cv2.conv,
            conv_pth=parameter.cv2.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

    def __call__(self, device, x):
        x = self.cv1(x)
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x1 = x
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        m1 = ttnn.max_pool2d(
            x,
            batch_size=x.shape[0],
            input_h=int(math.sqrt(x.shape[2])),
            input_w=int(math.sqrt(x.shape[2])),
            channels=320,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m2 = ttnn.max_pool2d(
            m1,
            batch_size=m1.shape[0],
            input_h=int(math.sqrt(m1.shape[2])),
            input_w=int(math.sqrt(m1.shape[2])),
            channels=320,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m3 = ttnn.max_pool2d(
            m2,
            batch_size=m2.shape[0],
            input_h=int(math.sqrt(m2.shape[2])),
            input_w=int(math.sqrt(m2.shape[2])),
            channels=320,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        use_sharded_concat = True
        if use_sharded_concat:
            y = sharded_concat([x1, m1, m2, m3])
        else:
            y = ttnn.concat([x1, m1, m2, m3], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv2(y)

        deallocate_tensors(x1, m1, m2, m3)
        return x
