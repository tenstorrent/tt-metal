# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device-only DyHead — runs the full DyHead forward pass on TTNN with NO host roundtrips.

Built on top of `tt_deform_conv.TtDeformConv2dV2` for the modulated deformable convs.
This is the missing piece that unblocks trace + 2CQ for the ATSS-Swin-L-DyHead model.

Layout: NHWC throughout (FPN output is NHWC; ATSS Head consumes NHWC).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import ttnn

from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores

from models.experimental.atss_swin_l_dyhead.tt.tt_deform_conv import TtDeformConv2dV2


# ---------------------------------------------------------------------------
# NHWC variants of the scale-aware and task-aware attention modules
# ---------------------------------------------------------------------------


def _hifi2_compute_config(device):
    """Shared compute config: HiFi3 + fp32 accumulator + L1 acc + no math approx.

    NOTE: On Wormhole, HiFi4 + fp32_dest_acc has a hardware-level accuracy bug; HiFi3 is
    actually MORE accurate. HiFi3 is the highest practical math_fidelity on this hardware.
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        math_approx_mode=False,
    )


class TtScaleAttnNHWC:
    """Scale-aware attention on NHWC input.

    global_avg_pool over (H, W) → linear(C→1) → ReLU → HSigmoid.
    Returns a (B, 1, 1, 1) ttnn tensor for broadcast multiplication against (B, H, W, C).
    """

    def __init__(self, device, conv_weight: torch.Tensor, conv_bias: torch.Tensor):
        self.device = device
        self._compute_config = _hifi2_compute_config(device)
        C = conv_weight.shape[1]
        # weight: (1, C, 1, 1) -> (C, 1) for matmul; reshape to column vector
        w = conv_weight.reshape(1, C).T.contiguous()  # (C, 1)
        b = conv_bias.reshape(1, 1)
        self.weight = ttnn.from_torch(
            w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.bias = ttnn.from_torch(
            b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def __call__(self, feat_nhwc):
        B, H, W, C = feat_nhwc.shape
        pooled = ttnn.mean(feat_nhwc, dim=(-3, -2), keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        pooled = ttnn.reshape(pooled, (B, C))

        out = ttnn.matmul(
            pooled, self.weight, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=self._compute_config
        )
        out = ttnn.add(out, self.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.relu(out, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.hardsigmoid(out, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.reshape(out, (B, 1, 1, 1))
        return out


class TtDyReLUNHWC:
    """Task-aware attention (DyReLU) on NHWC input.

    Computes per-channel piecewise-linear activation parameters via squeeze-and-excite,
    then applies them with broadcast multiplications across spatial dims.
    """

    def __init__(self, device, conv1_w, conv1_b, conv2_w, conv2_b, channels=256):
        self.device = device
        self.channels = channels
        self._compute_config = _hifi2_compute_config(device)

        ratio_ch = conv1_w.shape[0]
        exp_ch = conv2_w.shape[0]

        w1 = conv1_w.reshape(ratio_ch, channels).T.contiguous()  # (C, C/r)
        b1 = conv1_b.reshape(1, ratio_ch)
        w2 = conv2_w.reshape(exp_ch, ratio_ch).T.contiguous()  # (C/r, 4*C)
        b2 = conv2_b.reshape(1, exp_ch)

        self.weight1 = ttnn.from_torch(
            w1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.bias1 = ttnn.from_torch(
            b1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.weight2 = ttnn.from_torch(
            w2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.bias2 = ttnn.from_torch(
            b2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def __call__(self, feat_nhwc):
        B, H, W, C = feat_nhwc.shape
        feat_bytes = B * H * W * C * 2  # bf16
        full_mem = ttnn.L1_MEMORY_CONFIG if feat_bytes <= 2 * 1024 * 1024 else ttnn.DRAM_MEMORY_CONFIG

        pooled = ttnn.mean(feat_nhwc, dim=(-3, -2), keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        pooled = ttnn.reshape(pooled, (B, C))

        h = ttnn.matmul(
            pooled, self.weight1, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=self._compute_config
        )
        h = ttnn.add(h, self.bias1, memory_config=ttnn.L1_MEMORY_CONFIG)
        h = ttnn.relu(h, memory_config=ttnn.L1_MEMORY_CONFIG)

        coeffs = ttnn.matmul(
            h, self.weight2, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=self._compute_config
        )
        coeffs = ttnn.add(coeffs, self.bias2, memory_config=ttnn.L1_MEMORY_CONFIG)
        coeffs = ttnn.hardsigmoid(coeffs, memory_config=ttnn.L1_MEMORY_CONFIG)
        coeffs = ttnn.add(coeffs, -0.5, memory_config=ttnn.L1_MEMORY_CONFIG)

        a1, b1, a2, b2 = ttnn.split(coeffs, C, dim=1)
        ttnn.deallocate(coeffs)

        a1 = ttnn.reshape(a1, (B, 1, 1, C))
        b1 = ttnn.reshape(b1, (B, 1, 1, C))
        a2 = ttnn.reshape(a2, (B, 1, 1, C))
        b2 = ttnn.reshape(b2, (B, 1, 1, C))

        a1 = ttnn.add(
            ttnn.multiply(a1, 2.0, memory_config=ttnn.L1_MEMORY_CONFIG), 1.0, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        a2 = ttnn.multiply(a2, 2.0, memory_config=ttnn.L1_MEMORY_CONFIG)

        branch1 = ttnn.multiply(feat_nhwc, a1, memory_config=full_mem)
        branch1 = ttnn.add(branch1, b1, memory_config=full_mem)
        branch2 = ttnn.multiply(feat_nhwc, a2, memory_config=full_mem)
        branch2 = ttnn.add(branch2, b2, memory_config=full_mem)

        return ttnn.maximum(branch1, branch2)


# ---------------------------------------------------------------------------
# Helper: offset+mask 1×27 conv (the spatial_conv_offset of each DyHead block)
# ---------------------------------------------------------------------------


def _run_conv2d_nhwc(
    device, x_nhwc, weight_tt, bias_tt, in_ch, out_ch, kH, kW, padding, stride, H_in, W_in, compute_config
):
    """Generic NHWC conv2d helper using ttnn.conv2d. Returns (out_nhwc, weight, bias, H_out, W_out)."""
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
    )
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = min(max_cores, H_in * W_in)
    if num_cores > 0:
        conv_config.core_grid = get_shard_grid_from_num_cores(num_cores, device)
        conv_config.override_sharding_config = True

    [output, [H_out, W_out], [weight_tt, bias_tt]] = ttnn.conv2d(
        input_tensor=x_nhwc,
        weight_tensor=weight_tt,
        bias_tensor=bias_tt,
        in_channels=in_ch,
        out_channels=out_ch,
        device=device,
        kernel_size=(kH, kW),
        stride=(stride, stride),
        padding=(padding, padding),
        batch_size=1,
        input_height=H_in,
        input_width=W_in,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )
    output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.reshape(output, (1, H_out, W_out, out_ch))
    return output, weight_tt, bias_tt, H_out, W_out


# ---------------------------------------------------------------------------
# Helper: GroupNorm on NHWC tensor with TTNN backend
# ---------------------------------------------------------------------------


class TtGroupNorm:
    """Fused GroupNorm using ttnn.group_norm with Welford's algorithm in fp32.

    Replaces the previous 14-op custom GN with a single fused kernel call.
    Welford's algorithm computes mean+variance in one numerically stable pass,
    giving PCC ≥ 0.9999 vs PyTorch reference at all FPN level sizes.

    Pre-computes grid/weight/bias/mask parameters for each expected spatial
    shape so that __call__ is a pure device op with no host-side computation.
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,
        bias: torch.Tensor,
        num_groups: int,
        level_shapes: List[Tuple[int, int]] = (),
    ):
        self.device = device
        self.num_groups = num_groups
        self.C = weight.shape[0]
        assert self.C % num_groups == 0, f"C={self.C} not divisible by num_groups={num_groups}"

        self._compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

        self._params: dict = {}
        unique_shapes = set(level_shapes)
        for H, W in unique_shapes:
            nhw = H * W
            grid = ttnn.determine_expected_group_norm_dram_grid_size(
                device=device, num_channels=self.C, num_groups=num_groups, input_nhw=nhw
            )
            [w_tt, b_tt], mask_tt = ttnn.dram_group_norm_params_from_torch(
                [weight, bias], self.C, num_groups, device, core_grid=grid, dtype=ttnn.bfloat16
            )
            self._params[(H, W)] = (grid, w_tt, b_tt, mask_tt)

    def __call__(self, x_nhwc):
        N, H, W, C = x_nhwc.shape
        grid, w_tt, b_tt, mask_tt = self._params[(H, W)]

        x_flat = ttnn.reshape(x_nhwc, (N, 1, H * W, C))
        if x_flat.dtype != ttnn.bfloat16:
            x_flat = ttnn.typecast(x_flat, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if x_flat.layout != ttnn.TILE_LAYOUT:
            x_flat = ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if x_flat.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x_dram = ttnn.to_memory_config(x_flat, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x_flat)
            x_flat = x_dram

        out = ttnn.group_norm(
            x_flat,
            num_groups=self.num_groups,
            weight=w_tt,
            bias=b_tt,
            input_mask=mask_tt,
            core_grid=grid,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_config,
            inplace=False,
            use_welford=True,
            num_out_blocks=-1,
        )
        return ttnn.reshape(out, (N, H, W, C))


# ---------------------------------------------------------------------------
# Helper: bilinear resize via grid_sample (downsample 2x for cross-level)
# ---------------------------------------------------------------------------


# Cache of pre-built grid_sample grids keyed by (device_id, src_H, src_W, tgt_H, tgt_W).
# This is critical for trace capture: ttnn.from_torch on the hot path would issue a
# host→device write, which is forbidden inside ttnn.begin_trace_capture.
_RESIZE_GRID_CACHE: dict = {}


def _get_resize_grid(device, src_H, src_W, target_H, target_W):
    """Return a cached (1, target_H, target_W, 2) grid for the given resize.

    Reference DyHead uses F.interpolate(mode='bilinear', align_corners=True) for cross-level
    offset/mask/feature resizing. We replicate that sampling position with ttnn.grid_sample,
    which itself always uses align_corners=False normalization regardless of the flag passed.

    align_corners=True target-to-source mapping:
        src_y = ty * (H_in - 1) / (H_out - 1)        for H_out > 1
        src_y = 0                                     for H_out == 1
    Convert to ttnn (align_corners=False) grid:
        ny = (2 * src_y + 1) / H_in - 1
    """
    key = (id(device), src_H, src_W, target_H, target_W)
    g = _RESIZE_GRID_CACHE.get(key)
    if g is not None:
        return g
    ty = torch.arange(target_H, dtype=torch.float32)
    tx = torch.arange(target_W, dtype=torch.float32)
    # align_corners=True source-pixel positions
    if target_H > 1:
        src_y = ty * (src_H - 1.0) / (target_H - 1.0)
    else:
        src_y = torch.zeros(1, dtype=torch.float32)
    if target_W > 1:
        src_x = tx * (src_W - 1.0) / (target_W - 1.0)
    else:
        src_x = torch.zeros(1, dtype=torch.float32)
    # Convert to ttnn grid (align_corners=False) coordinates
    ny = (2 * src_y + 1) / src_H - 1
    nx = (2 * src_x + 1) / src_W - 1
    grid = torch.zeros(1, target_H, target_W, 2, dtype=torch.float32)
    grid[:, :, :, 0] = nx.view(1, 1, target_W)
    grid[:, :, :, 1] = ny.view(1, target_H, 1)
    grid_tt = ttnn.from_torch(
        grid.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _RESIZE_GRID_CACHE[key] = grid_tt
    return grid_tt


def _bilinear_resize_via_grid_sample(device, x_nhwc, target_H, target_W, channels):
    """Bilinear resize a NHWC tensor from (1, H, W, C) to (1, target_H, target_W, C) using grid_sample.

    Uses a pre-cached grid (built at first call per shape pair) — safe for trace capture.
    ttnn.grid_sample requires C % 32 == 0; pads channels if needed.
    """
    N, H, W, C = x_nhwc.shape
    if H == target_H and W == target_W:
        return x_nhwc

    TILE_WIDTH = 32
    padded_C = ((C + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
    needs_pad = padded_C != C
    if needs_pad:
        if x_nhwc.layout != ttnn.ROW_MAJOR_LAYOUT:
            x_nhwc = ttnn.to_layout(x_nhwc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_nhwc = ttnn.pad(x_nhwc, padding=[(0, 0), (0, 0), (0, 0), (0, padded_C - C)], value=0.0)

    grid_tt = _get_resize_grid(device, H, W, target_H, target_W)
    if x_nhwc.layout != ttnn.ROW_MAJOR_LAYOUT:
        x_nhwc = ttnn.to_layout(x_nhwc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.grid_sample(x_nhwc, grid_tt, mode="bilinear", padding_mode="zeros", align_corners=False)
    if needs_pad:
        out = ttnn.slice(out, [0, 0, 0, 0], [1, target_H, target_W, channels], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out


# ---------------------------------------------------------------------------
# Full on-device DyHead block (one block, all 5 levels)
# ---------------------------------------------------------------------------


class TtDyHeadBlockDevice:
    """One DyHead block, fully on TTNN device.

    Pre-uploads all weights and constructs TtDeformConv2dV2 instances for each
    (level, conv_role) combination so that base grids are cached per shape.
    """

    def __init__(self, device, pt_block: nn.Module, level_shapes: List[Tuple[int, int]]):
        """
        Args:
            device: TTNN device
            pt_block: PyTorch DyHeadBlock with loaded weights
            level_shapes: list of (H, W) per FPN level (e.g., [(80,80), (40,40), (20,20), (10,10), (5,5)])
        """
        self.device = device
        self.level_shapes = level_shapes
        self.num_levels = len(level_shapes)
        in_channels = pt_block.spatial_conv_offset.in_channels
        out_channels = pt_block.spatial_conv_mid.conv.out_channels
        assert in_channels == out_channels == 256, f"only 256ch supported, got in={in_channels} out={out_channels}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH = self.kW = 3
        self.K = self.kH * self.kW
        self.offset_dim = 2 * self.K  # 18
        self.mask_dim = self.K  # 9

        # spatial_conv_offset weight/bias (256 → 27, 3x3 padding=1)
        # Permute output channels from (dy,dx) to (dx,dy) order so the offset
        # matches the base_grid's (x,y) layout — avoids a per-call swap.
        scoff = pt_block.spatial_conv_offset
        oc_perm = torch.zeros(self.offset_dim + self.mask_dim, dtype=torch.long)
        for k in range(self.K):
            oc_perm[2 * k] = 2 * k + 1  # dx first
            oc_perm[2 * k + 1] = 2 * k  # dy second
        for k in range(self.K):
            oc_perm[self.offset_dim + k] = self.offset_dim + k
        self.so_weight = self._prep_conv_weight(scoff.weight.data[oc_perm], device)
        self.so_bias = self._prep_conv_bias(scoff.bias.data[oc_perm], device)
        self.so_out = scoff.weight.shape[0]  # 27

        # spatial_conv_{mid, low, high} — one TtDeformConv2dV2 per (level, role)
        # mid: input level i, output level i, stride 1
        # low: input level i-1, output level i, stride 2
        # high: input level i+1, output level i+1 (no upsample baked in), stride 1
        # Each DCN has its own (256, 256, 3, 3) weight (no bias).
        self.dcn_mid: List[TtDeformConv2dV2] = []
        self.dcn_low: List[TtDeformConv2dV2] = []
        self.dcn_high: List[TtDeformConv2dV2] = []
        for level in range(self.num_levels):
            H_curr, W_curr = level_shapes[level]
            # mid: in=curr, out=curr
            self.dcn_mid.append(
                TtDeformConv2dV2(
                    device=device,
                    weight=pt_block.spatial_conv_mid.conv.weight.data,
                    bias=None,
                    C_in=in_channels,
                    C_out=out_channels,
                    kH=3,
                    kW=3,
                    H_in=H_curr,
                    W_in=W_curr,
                    H_out=H_curr,
                    W_out=W_curr,
                    stride=(1, 1),
                    padding=(1, 1),
                    dilation=(1, 1),
                )
            )
            # low: in=prev level (larger), out=curr level, stride 2
            if level > 0:
                H_prev, W_prev = level_shapes[level - 1]
                self.dcn_low.append(
                    TtDeformConv2dV2(
                        device=device,
                        weight=pt_block.spatial_conv_low.conv.weight.data,
                        bias=None,
                        C_in=in_channels,
                        C_out=out_channels,
                        kH=3,
                        kW=3,
                        H_in=H_prev,
                        W_in=W_prev,
                        H_out=H_curr,
                        W_out=W_curr,
                        stride=(2, 2),
                        padding=(1, 1),
                        dilation=(1, 1),
                    )
                )
            else:
                self.dcn_low.append(None)
            # high: in=next level (smaller), out=next level dims, stride 1
            if level < self.num_levels - 1:
                H_next, W_next = level_shapes[level + 1]
                self.dcn_high.append(
                    TtDeformConv2dV2(
                        device=device,
                        weight=pt_block.spatial_conv_high.conv.weight.data,
                        bias=None,
                        C_in=in_channels,
                        C_out=out_channels,
                        kH=3,
                        kW=3,
                        H_in=H_next,
                        W_in=W_next,
                        H_out=H_next,
                        W_out=W_next,
                        stride=(1, 1),
                        padding=(1, 1),
                        dilation=(1, 1),
                    )
                )
            else:
                self.dcn_high.append(None)

        # GroupNorms — fused ttnn.group_norm with Welford fp32 accumulation.
        # Pre-compute per-level params: mid runs at all levels, low/high at subsets.
        all_shapes = list(level_shapes)
        low_shapes = [level_shapes[i] for i in range(1, self.num_levels)]
        high_shapes = [level_shapes[i + 1] for i in range(self.num_levels - 1)]
        self.gn_mid = TtGroupNorm(
            device,
            pt_block.spatial_conv_mid.norm.weight.data,
            pt_block.spatial_conv_mid.norm.bias.data,
            num_groups=16,
            level_shapes=all_shapes,
        )
        self.gn_low = TtGroupNorm(
            device,
            pt_block.spatial_conv_low.norm.weight.data,
            pt_block.spatial_conv_low.norm.bias.data,
            num_groups=16,
            level_shapes=low_shapes,
        )
        self.gn_high = TtGroupNorm(
            device,
            pt_block.spatial_conv_high.norm.weight.data,
            pt_block.spatial_conv_high.norm.bias.data,
            num_groups=16,
            level_shapes=high_shapes,
        )

        # scale_attn (one Conv 256→1) and task_attn (DyReLU)
        scale_conv = pt_block.scale_attn_module[1]
        self.scale_attn = TtScaleAttnNHWC(device, scale_conv.weight.data, scale_conv.bias.data)

        ta = pt_block.task_attn_module
        self.task_attn = TtDyReLUNHWC(
            device,
            ta.conv1[0].weight.data,
            ta.conv1[0].bias.data,
            ta.conv2[0].weight.data,
            ta.conv2[0].bias.data,
            channels=out_channels,
        )

        # HiFi2 + fp32 accumulator for the spatial_conv_offset conv. This conv runs once per
        # block per level and feeds directly into the DCN sampling positions — small precision
        # errors here ripple through 9 bilinear samples per output pixel and compound through
        # 6 blocks. Worth the kernel-time cost.
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

    @staticmethod
    def _prep_conv_weight(weight_torch: torch.Tensor, device) -> ttnn.Tensor:
        """Prepare conv weight for ttnn.conv2d (no special pre-permute needed for non-fused conv)."""
        # ttnn.conv2d accepts (out_C, in_C, kH, kW) torch-style
        return ttnn.from_torch(
            weight_torch.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _prep_conv_bias(bias_torch: torch.Tensor, device) -> ttnn.Tensor:
        return ttnn.from_torch(
            bias_torch.reshape(1, 1, 1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _spatial_offset_mask(self, x_nhwc, H, W):
        """Run spatial_conv_offset and split into (offset, mask) with sigmoid on mask."""
        om_mem = ttnn.DRAM_MEMORY_CONFIG
        out, self.so_weight, self.so_bias, _, _ = _run_conv2d_nhwc(
            self.device,
            x_nhwc,
            self.so_weight,
            self.so_bias,
            in_ch=self.in_channels,
            out_ch=self.so_out,
            kH=3,
            kW=3,
            padding=1,
            stride=1,
            H_in=H,
            W_in=W,
            compute_config=self.compute_config,
        )  # (1, H, W, 27)
        offset = ttnn.slice(out, [0, 0, 0, 0], [1, H, W, self.offset_dim], memory_config=om_mem)
        mask_logit = ttnn.slice(
            out,
            [0, 0, 0, self.offset_dim],
            [1, H, W, self.offset_dim + self.mask_dim],
            memory_config=om_mem,
        )
        ttnn.deallocate(out)
        mask = ttnn.sigmoid(mask_logit, memory_config=om_mem)
        ttnn.deallocate(mask_logit)
        return offset, mask

    def __call__(self, x_list_nhwc: List["ttnn.Tensor"]) -> List["ttnn.Tensor"]:
        """Forward one block over all FPN levels.

        Args:
            x_list_nhwc: list of NHWC ttnn tensors, one per FPN level
        Returns:
            list of NHWC ttnn tensors, one per FPN level
        """
        offset_per_level = []
        mask_per_level = []
        for level in range(self.num_levels):
            H, W = self.level_shapes[level]
            offset, mask = self._spatial_offset_mask(x_list_nhwc[level], H, W)
            offset_per_level.append(offset)
            mask_per_level.append(mask)

        outs: List["ttnn.Tensor"] = []
        for level in range(self.num_levels):
            H_curr, W_curr = self.level_shapes[level]
            offset_curr = offset_per_level[level]
            mask_curr = mask_per_level[level]
            # L1 for feature-sized tensors at small levels (L2=40×40=820KB, L3=205KB, L4=51KB bf16)
            feat_bytes = H_curr * W_curr * self.out_channels * 2
            lv_mem = ttnn.L1_MEMORY_CONFIG if feat_bytes <= 2 * 1024 * 1024 else ttnn.DRAM_MEMORY_CONFIG

            # mid branch
            mid = self.dcn_mid[level](x_list_nhwc[level], offset_curr, mask_curr)
            mid = self.gn_mid(mid)
            scale_w_mid = self.scale_attn(mid)
            sum_feat = ttnn.multiply(mid, scale_w_mid, memory_config=lv_mem)
            ttnn.deallocate(scale_w_mid)
            summed_levels = 1

            # low branch (from previous level)
            if level > 0:
                H_prev, W_prev = self.level_shapes[level - 1]
                low = self.dcn_low[level](x_list_nhwc[level - 1], offset_curr, mask_curr)
                low = self.gn_low(low)
                scale_w_low = self.scale_attn(low)
                weighted_low = ttnn.multiply(low, scale_w_low, memory_config=lv_mem)
                ttnn.deallocate(scale_w_low)
                ttnn.deallocate(low)
                sum_feat = ttnn.add(sum_feat, weighted_low, memory_config=lv_mem)
                ttnn.deallocate(weighted_low)
                summed_levels += 1

            # high branch (to next level, then upsample back)
            if level < self.num_levels - 1:
                H_next, W_next = self.level_shapes[level + 1]
                offset_resized = _bilinear_resize_via_grid_sample(
                    self.device, offset_curr, H_next, W_next, self.offset_dim
                )
                mask_resized = _bilinear_resize_via_grid_sample(self.device, mask_curr, H_next, W_next, self.mask_dim)
                high = self.dcn_high[level](x_list_nhwc[level + 1], offset_resized, mask_resized)
                high = self.gn_high(high)
                if H_curr != H_next or W_curr != W_next:
                    high = _bilinear_resize_via_grid_sample(self.device, high, H_curr, W_curr, self.out_channels)
                scale_w_high = self.scale_attn(high)
                weighted_high = ttnn.multiply(high, scale_w_high, memory_config=lv_mem)
                ttnn.deallocate(scale_w_high)
                ttnn.deallocate(high)
                ttnn.deallocate(offset_resized)
                ttnn.deallocate(mask_resized)
                sum_feat = ttnn.add(sum_feat, weighted_high, memory_config=lv_mem)
                ttnn.deallocate(weighted_high)
                summed_levels += 1

            if summed_levels > 1:
                sum_feat = ttnn.multiply(sum_feat, 1.0 / summed_levels, memory_config=lv_mem)

            out = self.task_attn(sum_feat)
            outs.append(out)
            ttnn.deallocate(sum_feat)
            ttnn.deallocate(mid)

        # Cleanup
        for o in offset_per_level + mask_per_level:
            ttnn.deallocate(o)

        return outs


class TtDyHeadDevice:
    """Multi-block on-device DyHead.

    Composes `num_blocks` TtDyHeadBlockDevice instances. Inputs and outputs are
    lists of NHWC TTNN tensors (one per FPN level) — no host roundtrips.
    """

    def __init__(self, device, pt_dyhead: nn.Module, level_shapes: List[Tuple[int, int]]):
        self.device = device
        self.num_blocks = pt_dyhead.num_blocks
        self.blocks: List[TtDyHeadBlockDevice] = []
        for i in range(self.num_blocks):
            self.blocks.append(TtDyHeadBlockDevice(device, pt_dyhead.dyhead_blocks[i], level_shapes))

    def __call__(self, x_list_nhwc: List["ttnn.Tensor"]) -> List["ttnn.Tensor"]:
        x = x_list_nhwc
        for block in self.blocks:
            x = block(x)
        return x
