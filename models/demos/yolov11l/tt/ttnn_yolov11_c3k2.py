# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov11l.tt.common import TtnnConv, deallocate_tensors, sharded_concat
from models.demos.yolov11l.tt.ttnn_yolov11_bottleneck import TtnnBottleneck
from models.demos.yolov11l.tt.ttnn_yolov11_c3k import TtnnC3K


class TtnnC3k2:
    def __init__(self, device, parameter, conv_pt, is_bk_enabled=False, reshard=False):
        self.is_bk_enabled = is_bk_enabled
        self.parameter = parameter
        n_inner = len(conv_pt.m)

        if is_bk_enabled:
            self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1, reshard=reshard, deallocate_activation=True)
            self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2, reshard=True)
            self.inner = [TtnnBottleneck(device, parameter[i], conv_pt.m[i]) for i in range(n_inner)]
        else:
            self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1, reshard=reshard, deallocate_activation=True)
            self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2, reshard=True)
            self.inner = [TtnnC3K(device, parameter[i], conv_pt.m[i]) for i in range(n_inner)]

    def __call__(self, device, x, use_shard_concat=True, tile_shape=32):
        x = self.cv1(device, x)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        y1 = x[:, :, :, : x.shape[-1] // 2]
        y2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        branches = [y1, y2]
        chain = y2
        for mod in self.inner:
            chain = mod(device, chain)
            branches.append(chain)

        for i, t in enumerate(branches):
            if t.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                branches[i] = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
            else:
                branches[i] = t

        if use_shard_concat:
            to_interleaved = True if (branches[0].shape[3] < tile_shape) else False
            x = sharded_concat(branches, to_interleaved=to_interleaved)
        else:
            interleaved = [ttnn.sharded_to_interleaved(t, ttnn.L1_MEMORY_CONFIG) for t in branches]
            x = ttnn.concat(tuple(interleaved), 3, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = self.cv2(device, x)

        deallocate_tensors(*branches)
        return x
