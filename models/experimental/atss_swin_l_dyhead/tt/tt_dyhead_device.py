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
        # Average pool over spatial dims (-3, -2)
        pooled = ttnn.mean(feat_nhwc, dim=(-3, -2), keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        pooled = ttnn.reshape(pooled, (B, C))

        # Fused linear (matmul + bias + relu) — saves 2 dispatches per call.
        out = ttnn.linear(
            pooled,
            self.weight,
            bias=self.bias,
            activation="relu",
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_config,
        )
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

        pooled = ttnn.mean(feat_nhwc, dim=(-3, -2), keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        pooled = ttnn.reshape(pooled, (B, C))

        # Fused matmul + bias + relu in one dispatch (was 3 ops).
        h = ttnn.linear(
            pooled,
            self.weight1,
            bias=self.bias1,
            activation="relu",
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_config,
        )
        # Fused matmul + bias in one dispatch (was 2 ops).
        coeffs = ttnn.linear(
            h,
            self.weight2,
            bias=self.bias2,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_config,
        )
        coeffs = ttnn.hardsigmoid(coeffs, memory_config=ttnn.L1_MEMORY_CONFIG)
        coeffs = ttnn.add(coeffs, -0.5, memory_config=ttnn.L1_MEMORY_CONFIG)

        a1, b1, a2, b2 = ttnn.split(coeffs, C, dim=1)
        ttnn.deallocate(coeffs)

        # Reshape for NHWC broadcast: (B, 1, 1, C)
        a1 = ttnn.reshape(a1, (B, 1, 1, C))
        b1 = ttnn.reshape(b1, (B, 1, 1, C))
        a2 = ttnn.reshape(a2, (B, 1, 1, C))
        b2 = ttnn.reshape(b2, (B, 1, 1, C))

        a1 = ttnn.add(
            ttnn.multiply(a1, 2.0, memory_config=ttnn.L1_MEMORY_CONFIG), 1.0, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        a2 = ttnn.multiply(a2, 2.0, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Fused branch_k = b_k + feat * a_k. Broadcast shape (B, 1, 1, C) over (B, H, W, C)
        # is supported by addcmul (unlike the (B,1,1,1) scalar-broadcast case). Deallocate
        # small (a, b) coefficients immediately so L1 has room for the next branch.
        # Inspired by the 1280-branch's commit ba954b9c7d2 ("Improve memory lifecycle").
        branch1 = ttnn.addcmul(b1, feat_nhwc, a1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(a1)
        ttnn.deallocate(b1)
        branch2 = ttnn.addcmul(b2, feat_nhwc, a2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(a2)
        ttnn.deallocate(b2)

        out = ttnn.maximum(branch1, branch2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(branch1)
        ttnn.deallocate(branch2)
        return out


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
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
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
    output = ttnn.sharded_to_interleaved(output, ttnn.L1_MEMORY_CONFIG)
    output = ttnn.reshape(output, (1, H_out, W_out, out_ch))
    return output, weight_tt, bias_tt, H_out, W_out


# ---------------------------------------------------------------------------
# Helper: GroupNorm on NHWC tensor with TTNN backend
# ---------------------------------------------------------------------------


class TtGroupNorm:
    """GroupNorm on NHWC input, using ttnn.group_norm with use_welford=True.

    Welford's algorithm gives stable mean/variance accumulation, recovering most
    of the precision the previous fp32 custom path provided. Replaces ~12 device
    ops with ~3 (reshape + group_norm + reshape), saving ~700 ops/inference of
    dispatch overhead at ~100 us each.

    Per-spatial-size reciprocals tensors are precomputed at __init__ so the hot
    path stays trace-safe.
    """

    @staticmethod
    def _pick_grid(H: int, W: int, max_y: int = 8, max_x: int = 8) -> "ttnn.CoreGrid":
        """Largest core grid where grid_y divides ceil(H*W/32) (= Ht). Required by ttnn.group_norm."""
        Ht = (H * W + 31) // 32
        for gy in range(min(max_y, Ht), 0, -1):
            if Ht % gy == 0:
                return ttnn.CoreGrid(y=gy, x=max_x)
        return ttnn.CoreGrid(y=1, x=1)

    def __init__(
        self,
        device,
        weight: torch.Tensor,
        bias: torch.Tensor,
        num_groups: int,
        level_shapes: List[Tuple[int, int]],
    ):
        self.device = device
        self.num_groups = num_groups
        self.C = weight.shape[0]
        assert self.C % num_groups == 0, f"C={self.C} not divisible by num_groups={num_groups}"
        self.C_per_G = self.C // num_groups

        # Per (H, W): pick a core_grid that fits ttnn.group_norm's divisibility constraint,
        # then prepare the matching gamma/beta/input_mask/reciprocals for that grid.
        self.params: dict = {}
        for H, W in set(level_shapes):
            grid = self._pick_grid(H, W)
            [gamma_t, beta_t], input_mask = ttnn.dram_group_norm_params_from_torch(
                [weight, bias], self.C, num_groups, device, core_grid=grid, return_mask=True, dtype=ttnn.bfloat16
            )
            recip_torch = ttnn.create_group_norm_reciprocals(1, self.C, H, W, num_groups, grid)
            recip_dram = ttnn.from_torch(
                recip_torch,
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            sharded_mem_cfg = ttnn.create_sharded_memory_config(
                shape=recip_dram.shape,
                core_grid=grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            recip = ttnn.to_memory_config(recip_dram, sharded_mem_cfg)
            ttnn.deallocate(recip_dram)
            self.params[(H, W)] = (grid, gamma_t, beta_t, input_mask, recip)

        self.epsilon = 1e-5
        self._compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi3,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

        # fp32 gamma/beta for the small-spatial-size custom GN fallback (P6, P7).
        self._gamma_5d = ttnn.from_torch(
            weight.reshape(1, 1, 1, num_groups, self.C_per_G).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._beta_5d = ttnn.from_torch(
            bias.reshape(1, 1, 1, num_groups, self.C_per_G).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x_nhwc):
        N, H, W, C = x_nhwc.shape
        # ttnn.group_norm only takes BFLOAT16 input; for small spatial sizes (<=100
        # elements like P6 10x10 and P7 5x5) the bf16 quantization + welford reduction
        # compounds badly across 6 DyHead blocks (cent PCC at P7 drops to ~0.92).
        # Fall back to the custom fp32 path for those sizes; use welford for everything
        # bigger where the precision holds and the op-count savings dominate.
        if H * W < 200:
            return self._custom_fp32_gn(x_nhwc)
        grid, gamma_t, beta_t, input_mask, recip = self.params[(H, W)]
        x_4d = ttnn.reshape(x_nhwc, (N, 1, H * W, C))
        if x_4d.layout != ttnn.TILE_LAYOUT:
            x_4d = ttnn.to_layout(x_4d, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.group_norm(
            x_4d,
            num_groups=self.num_groups,
            input_mask=input_mask,
            weight=gamma_t,
            bias=beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=grid,
            inplace=False,
            epsilon=self.epsilon,
            use_welford=True,
            reciprocals=recip,
            compute_kernel_config=self._compute_config,
        )
        return ttnn.reshape(out, (N, H, W, C))

    def _custom_fp32_gn(self, x_nhwc):
        """fp32 GN for small spatial sizes (P6, P7) where welford precision is poor.

        Adapted from gtobarTT's 499fcb3: keep the big tensor 4D throughout (avoid the
        5D reshape that forces real data movement when C/G=16 isn't tile-aligned);
        reduce group statistics on the tiny (N,1,1,C) per-channel stats; fold the
        affine transform into one addcmul on the big tensor.
        """
        N, H, W, C = x_nhwc.shape
        G = self.num_groups
        C_per_G = self.C_per_G
        big_bytes = N * H * W * C * 4
        big_mem = ttnn.L1_MEMORY_CONFIG if big_bytes <= 4 * 1024 * 1024 else ttnn.DRAM_MEMORY_CONFIG
        small_mem = ttnn.L1_MEMORY_CONFIG

        x_fp32 = ttnn.typecast(x_nhwc, ttnn.float32, memory_config=big_mem)
        if x_fp32.layout != ttnn.TILE_LAYOUT:
            x_fp32 = ttnn.to_layout(x_fp32, ttnn.TILE_LAYOUT, memory_config=big_mem)

        # Per-channel stats on the 4D tensor (cheap reduction over H, W).
        mean_c = ttnn.mean(x_fp32, dim=(1, 2), keepdim=True, memory_config=small_mem)  # (N,1,1,C)
        x_sq = ttnn.multiply(x_fp32, x_fp32, memory_config=big_mem)
        mean_x_sq_c = ttnn.mean(x_sq, dim=(1, 2), keepdim=True, memory_config=small_mem)
        ttnn.deallocate(x_sq)

        # Per-group stats on the tiny per-channel tensors. E[E[x|HW]|G] = E[x|G].
        mean_c_5g = ttnn.reshape(mean_c, (N, 1, G, C_per_G))
        mean_x_sq_c_5g = ttnn.reshape(mean_x_sq_c, (N, 1, G, C_per_G))
        mean_g = ttnn.mean(mean_c_5g, dim=(3,), keepdim=True, memory_config=small_mem)
        mean_x_sq_g = ttnn.mean(mean_x_sq_c_5g, dim=(3,), keepdim=True, memory_config=small_mem)
        var_g = ttnn.subtract(
            mean_x_sq_g, ttnn.multiply(mean_g, mean_g, memory_config=small_mem), memory_config=small_mem
        )
        inv_std_g = ttnn.rsqrt(
            ttnn.add(var_g, self.epsilon, memory_config=small_mem),
            fast_and_approximate_mode=True,
            memory_config=small_mem,
        )

        # Fold affine: y = alpha * x + delta where alpha = inv_std * gamma_per_channel
        # and delta = beta - mean * alpha. Both computed in (1, 1, G, C/G) and reshaped
        # back to (1, 1, 1, C) for the broadcast against the (N, H, W, C) tensor.
        gamma_5g = ttnn.reshape(self._gamma_5d, (1, 1, G, C_per_G))
        beta_5g = ttnn.reshape(self._beta_5d, (1, 1, G, C_per_G))
        alpha_5g = ttnn.multiply(inv_std_g, gamma_5g, memory_config=small_mem)
        delta_5g = ttnn.subtract(
            beta_5g, ttnn.multiply(mean_g, alpha_5g, memory_config=small_mem), memory_config=small_mem
        )
        alpha_4d = ttnn.reshape(alpha_5g, (1, 1, 1, C))
        delta_4d = ttnn.reshape(delta_5g, (1, 1, 1, C))

        out_4d = ttnn.addcmul(delta_4d, x_fp32, alpha_4d, memory_config=big_mem)
        return ttnn.typecast(out_4d, ttnn.bfloat16, memory_config=big_mem)


# ---------------------------------------------------------------------------
# Helper: bilinear resize via grid_sample (downsample 2x for cross-level)
# ---------------------------------------------------------------------------


# Cache of pre-built grid_sample grids keyed by (device_id, src_H, src_W, tgt_H, tgt_W).
# This is critical for trace capture: ttnn.from_torch on the hot path would issue a
# host→device write, which is forbidden inside ttnn.begin_trace_capture.
_RESIZE_GRID_CACHE: dict = {}


def _get_resize_grid(device, src_H, src_W, target_H, target_W, padded_C):
    """Return a cached precomputed (1, target_H, target_W, 6) grid for the given resize.

    Reference DyHead uses F.interpolate(mode='bilinear', align_corners=True) for cross-level
    offset/mask/feature resizing. We replicate that sampling position with ttnn.grid_sample.
    The returned grid is in the "precomputed" format produced by
    ttnn.prepare_grid_sample_grid — it bakes the pixel coordinates and the bilinear
    weights, so the runtime grid_sample skips that arithmetic each call.

    align_corners=True target-to-source mapping:
        src_y = ty * (H_in - 1) / (H_out - 1)        for H_out > 1
        src_y = 0                                     for H_out == 1
    Convert to ttnn (align_corners=False) grid:
        ny = (2 * src_y + 1) / H_in - 1
    """
    key = (id(device), src_H, src_W, target_H, target_W, padded_C)
    g = _RESIZE_GRID_CACHE.get(key)
    if g is not None:
        return g
    ty = torch.arange(target_H, dtype=torch.float32)
    tx = torch.arange(target_W, dtype=torch.float32)
    if target_H > 1:
        src_y = ty * (src_H - 1.0) / (target_H - 1.0)
    else:
        src_y = torch.zeros(1, dtype=torch.float32)
    if target_W > 1:
        src_x = tx * (src_W - 1.0) / (target_W - 1.0)
    else:
        src_x = torch.zeros(1, dtype=torch.float32)
    ny = (2 * src_y + 1) / src_H - 1
    nx = (2 * src_x + 1) / src_W - 1
    grid = torch.zeros(1, target_H, target_W, 2, dtype=torch.float32)
    grid[:, :, :, 0] = nx.view(1, 1, target_W)
    grid[:, :, :, 1] = ny.view(1, target_H, 1)
    # prepare_grid_sample_grid expects the input grid on HOST.
    grid_host = ttnn.from_torch(grid, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_prepared = ttnn.prepare_grid_sample_grid(
        grid_host,
        input_shape=[1, src_H, src_W, padded_C],
        padding_mode="zeros",
        output_dtype=ttnn.bfloat16,
    )
    grid_tt = ttnn.from_torch(
        ttnn.to_torch(grid_prepared),
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
            x_nhwc = ttnn.to_layout(x_nhwc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_nhwc = ttnn.pad(x_nhwc, padding=[(0, 0), (0, 0), (0, 0), (0, padded_C - C)], value=0.0)

    grid_tt = _get_resize_grid(device, H, W, target_H, target_W, padded_C)
    if x_nhwc.layout != ttnn.ROW_MAJOR_LAYOUT:
        x_nhwc = ttnn.to_layout(x_nhwc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.grid_sample(
        x_nhwc, grid_tt, mode="bilinear", padding_mode="zeros", align_corners=False, use_precomputed_grid=True
    )
    if needs_pad:
        out = ttnn.slice(out, [0, 0, 0, 0], [1, target_H, target_W, channels], memory_config=ttnn.L1_MEMORY_CONFIG)
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

        # spatial_conv_offset weight/bias (256 -> 27, 3x3 padding=1).
        # Permute the first 18 output channels to swap each (dy_k, dx_k) pair into
        # (dx_k, dy_k), matching our DCNs' offset_layout="xy". Channels 18..26 are
        # the modulation mask and stay in place.
        scoff = pt_block.spatial_conv_offset
        perm = list(range(scoff.weight.shape[0]))  # 27
        for k in range(self.K):
            perm[2 * k], perm[2 * k + 1] = perm[2 * k + 1], perm[2 * k]
        perm_t = torch.as_tensor(perm, dtype=torch.long)
        scoff_w_perm = scoff.weight.data.index_select(0, perm_t).contiguous()
        scoff_b_perm = scoff.bias.data.index_select(0, perm_t).contiguous()
        self.so_weight = self._prep_conv_weight(scoff_w_perm, device)
        self.so_bias = self._prep_conv_bias(scoff_b_perm, device)
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
                    offset_layout="xy",
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
                        offset_layout="xy",
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
                        offset_layout="xy",
                    )
                )
            else:
                self.dcn_high.append(None)

        # GroupNorms (one per conv role; weights shared across all levels).
        # Pass all level_shapes so the welford reciprocals can be precomputed at __init__.
        self.gn_mid = TtGroupNorm(
            device,
            pt_block.spatial_conv_mid.norm.weight.data,
            pt_block.spatial_conv_mid.norm.bias.data,
            num_groups=16,
            level_shapes=level_shapes,
        )
        self.gn_low = TtGroupNorm(
            device,
            pt_block.spatial_conv_low.norm.weight.data,
            pt_block.spatial_conv_low.norm.bias.data,
            num_groups=16,
            level_shapes=level_shapes,
        )
        self.gn_high = TtGroupNorm(
            device,
            pt_block.spatial_conv_high.norm.weight.data,
            pt_block.spatial_conv_high.norm.bias.data,
            num_groups=16,
            level_shapes=level_shapes,
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
            packer_l1_acc=True,
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
        # Slice into offset (first 18 ch) and mask_logit (last 9 ch)
        offset = ttnn.slice(out, [0, 0, 0, 0], [1, H, W, self.offset_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
        mask_logit = ttnn.slice(
            out,
            [0, 0, 0, self.offset_dim],
            [1, H, W, self.offset_dim + self.mask_dim],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(out)
        mask = ttnn.sigmoid(mask_logit, memory_config=ttnn.L1_MEMORY_CONFIG)
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

            # mid branch
            mid = self.dcn_mid[level](x_list_nhwc[level], offset_curr, mask_curr)
            mid = self.gn_mid(mid)
            scale_w_mid = self.scale_attn(mid)
            sum_feat = ttnn.multiply(mid, scale_w_mid, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(scale_w_mid)
            # mid is no longer needed after sum_feat is computed — free its L1/DRAM up-front
            # so the low/high branch tensors have room. (Was being held until end of level.)
            ttnn.deallocate(mid)
            summed_levels = 1

            # low branch (from previous level)
            if level > 0:
                H_prev, W_prev = self.level_shapes[level - 1]
                # Stride-2 conv: offset/mask shape must match output (current level dims).
                # If they already do (the common case), pass directly.
                low = self.dcn_low[level](x_list_nhwc[level - 1], offset_curr, mask_curr)
                low = self.gn_low(low)
                scale_w_low = self.scale_attn(low)
                weighted_low = ttnn.multiply(low, scale_w_low, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(scale_w_low)
                ttnn.deallocate(low)
                sum_feat = ttnn.add(sum_feat, weighted_low, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(weighted_low)
                summed_levels += 1

            # high branch (to next level, then upsample back)
            if level < self.num_levels - 1:
                H_next, W_next = self.level_shapes[level + 1]
                # Resize offset/mask from curr dims to next level dims
                offset_resized = _bilinear_resize_via_grid_sample(
                    self.device, offset_curr, H_next, W_next, self.offset_dim
                )
                mask_resized = _bilinear_resize_via_grid_sample(self.device, mask_curr, H_next, W_next, self.mask_dim)
                high = self.dcn_high[level](x_list_nhwc[level + 1], offset_resized, mask_resized)
                high = self.gn_high(high)
                # Upsample to current level dims via grid_sample (handles non-tile-aligned spatial sizes).
                if H_curr != H_next or W_curr != W_next:
                    high = _bilinear_resize_via_grid_sample(self.device, high, H_curr, W_curr, self.out_channels)
                scale_w_high = self.scale_attn(high)
                weighted_high = ttnn.multiply(high, scale_w_high, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(scale_w_high)
                ttnn.deallocate(high)
                ttnn.deallocate(offset_resized)
                ttnn.deallocate(mask_resized)
                sum_feat = ttnn.add(sum_feat, weighted_high, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(weighted_high)
                summed_levels += 1

            if summed_levels > 1:
                sum_feat = ttnn.multiply(sum_feat, 1.0 / summed_levels, memory_config=ttnn.L1_MEMORY_CONFIG)

            out = self.task_attn(sum_feat)
            outs.append(out)
            ttnn.deallocate(sum_feat)

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
        import tracy

        x = x_list_nhwc
        for i, block in enumerate(self.blocks):
            tracy.signpost(f"dyhead_block_{i}")
            x = block(x)
        return x
