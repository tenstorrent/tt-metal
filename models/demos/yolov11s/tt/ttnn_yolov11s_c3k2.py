# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from ttnn.model_preprocessing import ParameterDict

import ttnn
from models.demos.yolov11s.tt.common import TtnnConv, deallocate_tensors, sharded_concat
from models.demos.yolov11s.tt.ttnn_yolov11s_bottleneck import TtnnBottleneck
from models.demos.yolov11s.tt.ttnn_yolov11s_c3k import TtnnC3K


def _align_spatial_for_concat(t, spatial_hw):
    """Trim padded spatial only; layout for concat is handled in common.sharded_concat."""
    if int(t.shape[2]) > int(spatial_hw):
        t = t[:, :, :spatial_hw, :]
    return t


def _cv1_ab_conv_pts(cv1_weights, out_channels):
    """Preprocessing may expose cv1.conv only (unsplit); C3k2 still needs a/b half-conv weights."""
    if "a" in cv1_weights and "b" in cv1_weights:
        return cv1_weights.a, cv1_weights.b
    half = out_channels // 2
    c = cv1_weights.conv
    w = c.weight
    # o, i_, kh, kw = (int(w.shape[0]), int(w.shape[1]), int(w.shape[2]), int(w.shape[3]))
    # wa = ttnn.slice(w, (0, 0, 0, 0), (half, i_, kh, kw))
    # wb = ttnn.slice(w, (half, 0, 0, 0), (o, i_, kh, kw))
    wa, wb = ttnn.split(w, half, dim=0)
    if "bias" in c:
        b = c.bias
        # b0, b1, b2 = int(b.shape[0]), int(b.shape[1]), int(b.shape[2])
        # bc = int(b.shape[3])
        # ba = ttnn.slice(b, (0, 0, 0, 0), (b0, b1, b2, half))
        # bb = ttnn.slice(b, (0, 0, 0, half), (b0, b1, b2, bc))
        ba, bb = ttnn.split(b, half, dim=0)
        a_inner = ParameterDict({"weight": wa, "bias": ba})
        b_inner = ParameterDict({"weight": wb, "bias": bb})
    else:
        a_inner = ParameterDict({"weight": wa})
        b_inner = ParameterDict({"weight": wb})
    return ParameterDict({"conv": a_inner}), ParameterDict({"conv": b_inner})


class TtnnC3k2:
    def __init__(self, device, parameter, conv_pt, is_bk_enabled=False, reshard=False):
        self.is_bk_enabled = is_bk_enabled
        self.parameter = parameter
        cv1_a_pt, cv1_b_pt = _cv1_ab_conv_pts(conv_pt.cv1, parameter.cv1.conv.out_channels)
        self.cv1_a = TtnnConv(
            device,
            parameter.cv1,
            cv1_a_pt,
            reshard=reshard,
            split_weights=True,
        )
        self.cv1_b = TtnnConv(
            device,
            parameter.cv1,
            cv1_b_pt,
            reshard=reshard,
            split_weights=True,
        )
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2, reshard=True)
        if is_bk_enabled:
            self.k = TtnnBottleneck(device, parameter[0], conv_pt.m[0])
        else:
            self.c3k = TtnnC3K(device, parameter[0], conv_pt.m[0])

    def __call__(self, device, x, use_shard_concat=True, tile_shape=32):
        cfg = self.cv1_a.conv.conv
        spatial_hw = int(cfg.input_height) * int(cfg.input_width)
        cv1_a = self.cv1_a(device, x, output_rm_needed=True)
        # Match cv1_a: conv can return flat dim > H*W; without output_rm, cv1_b keeps padding and bk path PCC collapses.
        cv1_b = self.cv1_b(device, x, output_rm_needed=True, to_interleaved=x.shape[2] > spatial_hw)
        mid = self.k if self.is_bk_enabled else self.c3k
        y3 = mid(device, cv1_b)

        cv1_a = _align_spatial_for_concat(cv1_a, spatial_hw)
        cv1_b = _align_spatial_for_concat(cv1_b, spatial_hw)
        y3 = _align_spatial_for_concat(y3, spatial_hw)
        # Height-sharded 3-way concat is fine for C3K-only C3k2; with Bottleneck + large spatial (e.g. 160²)
        # it still yields ~0.07 PCC vs torch — use L1 interleaved concat for that path only.
        if use_shard_concat and not self.is_bk_enabled:
            x = sharded_concat([cv1_a, cv1_b, y3], dim=3, to_interleaved=False)
        else:
            cv1_a = ttnn.sharded_to_interleaved(cv1_a, ttnn.L1_MEMORY_CONFIG) if cv1_a.is_sharded() else cv1_a
            cv1_b = ttnn.sharded_to_interleaved(cv1_b, ttnn.L1_MEMORY_CONFIG) if cv1_b.is_sharded() else cv1_b
            y3 = ttnn.sharded_to_interleaved(y3, ttnn.L1_MEMORY_CONFIG) if y3.is_sharded() else y3
            x = ttnn.concat((cv1_a, cv1_b, y3), 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv2(device, x)
        deallocate_tensors(cv1_a, cv1_b, y3)
        return x
