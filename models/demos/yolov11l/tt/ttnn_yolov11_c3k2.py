# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov11l.tt.common import TtnnConv, deallocate_tensors, sharded_concat
from models.demos.yolov11l.tt.ttnn_yolov11_bottleneck import TtnnBottleneck
from models.demos.yolov11l.tt.ttnn_yolov11_c3k import TtnnC3K


class TtnnC3k2:
    def __init__(
        self,
        device,
        parameter,
        conv_pt,
        is_bk_enabled=False,
        reshard=False,
        use_block_sharded=False,
        cv1_config_override=None,
        cv2_l1_fallback_threshold_bytes=512 * 1024,
    ):
        self.is_bk_enabled = is_bk_enabled
        self.reshard = reshard
        self.parameter = parameter
        self.cv2_l1_fallback_threshold_bytes = cv2_l1_fallback_threshold_bytes
        n_inner = len(conv_pt.m)

        if is_bk_enabled:
            self.cv1 = TtnnConv(
                device,
                parameter.cv1,
                conv_pt.cv1,
                reshard=False,
                deallocate_activation=True,
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
            if reshard:
                cv1_shard_layout = (
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED
                    if use_block_sharded
                    else ttnn.TensorMemoryLayout.WIDTH_SHARDED
                )
            else:
                cv1_shard_layout = None
            self.cv1 = TtnnConv(
                device,
                parameter.cv1,
                conv_pt.cv1,
                reshard=reshard,
                deallocate_activation=True,
                shard_layout=cv1_shard_layout,
                # config_override=cv1_config_override,
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
            x = sharded_concat(branches, to_interleaved=False, prefer_l1_concat=True)
        else:
            interleaved = [ttnn.sharded_to_interleaved(t, ttnn.L1_MEMORY_CONFIG) for t in branches]
            x = ttnn.concat(tuple(interleaved), 3, memory_config=ttnn.L1_MEMORY_CONFIG)

        tensor_bytes = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3] * 2
        keep_cv2_input_in_l1 = tensor_bytes <= self.cv2_l1_fallback_threshold_bytes

        # Break potential aliasing between concat output and branch buffers before branch deallocation.
        # For small tensors keep cv2 input in L1; large tensors still fall back to DRAM to avoid L1 pressure.
        if keep_cv2_input_in_l1:
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            else:
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.clone(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            if use_shard_concat and x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            else:
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        deallocate_tensors(*branches)
        x = self.cv2(device, x)
        return x
