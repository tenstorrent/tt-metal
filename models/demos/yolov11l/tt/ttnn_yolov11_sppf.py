# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.demos.yolov11l.tt.common import TtnnConv, deallocate_tensors, sharded_concat


def _max_pool2d_args_from_tensor(x, nominal_input_h, nominal_input_w):
    """
    max_pool2d expects batch_size * input_h * input_w == shape[2] (flattened spatial).
    Inferred conv_args use a mesh batch (e.g. 8) while each device tensor may only
    hold one spatial slab (e.g. sf == 400 == 1 * 20 * 20). Passing the inferred
    batch_size into max_pool2d then breaks halo / UntilizeWithHalo (segfault).
    """
    sf = x.shape[2]
    channels = x.shape[-1]
    ph = int(nominal_input_h)
    pw = int(nominal_input_w)
    single_sf = ph * pw
    if single_sf > 0 and sf % single_sf == 0:
        return x, sf // single_sf, ph, pw, channels
    s = int(math.sqrt(sf))
    if s * s != sf:
        raise ValueError(
            f"SPPF max_pool2d: shape[2]={sf} incompatible with nominal {ph}×{pw} and not a square spatial count"
        )
    return x, 1, s, s, channels


class TtnnSPPF:
    def __init__(self, device, parameter, conv_pt):
        self.parameter = parameter
        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2, reshard=True)

    def __call__(self, device, x, use_sharded_concat=True):
        x = self.cv1(device, x)
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        ph0, pw0 = self.parameter.cv2.conv.input_height, self.parameter.cv2.conv.input_width
        x, pb, ph, pw, ch = _max_pool2d_args_from_tensor(x, ph0, pw0)
        x1 = x
        m1 = ttnn.max_pool2d(
            x,
            batch_size=pb,
            input_h=ph,
            input_w=pw,
            channels=ch,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m1, pb2, ph2, pw2, ch2 = _max_pool2d_args_from_tensor(m1, ph, pw)
        m2 = ttnn.max_pool2d(
            m1,
            batch_size=pb2,
            input_h=ph2,
            input_w=pw2,
            channels=ch2,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m2, pb3, ph3, pw3, ch3 = _max_pool2d_args_from_tensor(m2, ph, pw)
        m3 = ttnn.max_pool2d(
            m2,
            batch_size=pb3,
            input_h=ph3,
            input_w=pw3,
            channels=ch3,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        if use_sharded_concat:
            y = sharded_concat([x1, m1, m2, m3], to_interleaved=False)
        else:
            y = ttnn.concat([x1, m1, m2, m3], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        # cv2 (reshard=True) uses conv2d_L1, which rejects activations whose storage is still INTERLEAVED at the C++
        # tensor level. Python memory_config can disagree with that; normalize by round-tripping sharded → interleaved
        # → height-sharded when needed, else interleaved → sharded.
        num_cores = 64
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        shard_height = (y.shape[2] + num_cores - 1) // num_cores
        y_shard = ttnn.create_sharded_memory_config(
            (shard_height, y.shape[-1]),
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        if y.is_sharded():
            y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
        y = ttnn.interleaved_to_sharded(y, y_shard)
        x = self.cv2(device, y)
        deallocate_tensors(x1, m1, m2, m3)
        return x
