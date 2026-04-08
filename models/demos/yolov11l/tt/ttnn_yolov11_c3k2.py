# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov11l.tt.common import TtnnConv, deallocate_tensors, sharded_concat
from models.demos.yolov11l.tt.ttnn_yolov11_bottleneck import TtnnBottleneck
from models.demos.yolov11l.tt.ttnn_yolov11_c3k import TtnnC3K


class TtnnC3k2:
    def __init__(self, device, parameter, conv_pt, is_bk_enabled=False, reshard=False):
        self.is_bk_enabled = is_bk_enabled
        self.reshard = reshard
        self.parameter = parameter
        n_inner = len(conv_pt.m)

        if is_bk_enabled:
            self.cv1 = TtnnConv(
                device,
                parameter.cv1,
                conv_pt.cv1,
                reshard=False,
                deallocate_activation=True,
                slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=8),
            )
            # cv2: DRAM activations + auto slice (like conv2) — L1 sharded concat + HEIGHT_SHARDED cv2 peaked L1 (tilize/matmul OOM).
            self.cv2 = TtnnConv(
                device,
                parameter.cv2,
                conv_pt.cv2,
                reshard=True,
                deallocate_activation=True,
                shard_layout=None,
            )
            self.inner = [TtnnBottleneck(device, parameter[i], conv_pt.m[i]) for i in range(n_inner)]
        else:
            cv1_slice_count = 16 if reshard else 8
            # Conv2d here requires sharded layout; WIDTH_SHARDED is typically less L1-heavy
            # than forcing HEIGHT_SHARDED for these high-resolution PAN/FPN stages.
            cv1_shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED if reshard else None
            self.cv1 = TtnnConv(
                device,
                parameter.cv1,
                conv_pt.cv1,
                reshard=reshard,
                deallocate_activation=True,
                shard_layout=cv1_shard_layout,
                slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=cv1_slice_count),
            )
            self.cv2 = TtnnConv(
                device,
                parameter.cv2,
                conv_pt.cv2,
                reshard=True,
                deallocate_activation=True,
                shard_layout=None,
            )
            self.inner = [TtnnC3K(device, parameter[i], conv_pt.m[i]) for i in range(n_inner)]

    def __call__(self, device, x, use_shard_concat=True):
        if self.reshard:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
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
            x = sharded_concat(branches, to_interleaved=False)
        else:
            interleaved = [ttnn.sharded_to_interleaved(t, ttnn.L1_MEMORY_CONFIG) for t in branches]
            x = ttnn.concat(tuple(interleaved), 3, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Spill concat to DRAM before freeing branches: deallocating y1/y2/chain first can free storage still aliased by
        # the sharded concat output and corrupt the tensor, leading to conv2d reshape "different volumes" later.
        if use_shard_concat:
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            else:
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        else:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        deallocate_tensors(*branches)
        x = self.cv2(device, x)
        return x
