# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# PResNet-50 backbone forward pass.

import ttnn
import torch
from tt.resnet_blocks import conv_block, residual_block

_STAGE_BLOCKS = [3, 4, 6, 3]

def _stem(x, params, device):
    h, w = 640, 640
    configs = [(3, 2, 1), (3, 1, 1), (3, 1, 1)]
    for i, (k, s, p) in enumerate(configs):
        x, (h, w) = conv_block(
            x, params.stem[i], device,
            kernel_size=(k, k), stride=(s, s), padding=(p, p),
            input_height=h, input_width=w, activation="relu",
        )
    x = ttnn.max_pool2d(
        x, batch_size=x.shape[0], input_h=h, input_w=w, channels=x.shape[-1],
        kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  
    )
    return x, h // 2, w // 2  # 160x160

def _stage(x, block_params, device, stride_first, h, w):
    for i, bp in enumerate(block_params):
        stride = stride_first if i == 0 else 1
        x, h, w = residual_block(x, bp, device, stride=stride, input_height=h, input_width=w)
    return x, h, w

def presnet50(x, params, device):
    x, h, w = _stem(x, params, device)
    x, h, w = _stage(x, params.stages[0], device, 1, h, w)   
    x, h, w = _stage(x, params.stages[1], device, 2, h, w)   
    
    s3 = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=x.dtype)
    x, h, w = _stage(x, params.stages[2], device, 2, h, w)   
    s4 = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=x.dtype)
    x, h, w = _stage(x, params.stages[3], device, 2, h, w)   
    s5 = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=x.dtype)
    return s3, s4, s5