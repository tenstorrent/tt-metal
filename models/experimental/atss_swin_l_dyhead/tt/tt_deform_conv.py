# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
On-device implementation of modulated deformable conv2d (DCNv2).

Composes existing TTNN primitives — grid_sample (bilinear) + matmul + ops —
to replicate torchvision.ops.deform_conv2d semantics with a modulation mask.

Algorithm (matches torchvision.ops.deform_conv2d):
    For each output pixel (i, j) and kernel index k=(ky, kx):
        base_y, base_x = i*sh - ph + ky*dh, j*sw - pw + kx*dw  (raw pixel coords)
        sample_y, sample_x = base + (offset_dy_k, offset_dx_k)
        sampled[k] = bilinear_interpolate(input, sample_y, sample_x)
    output = sum_{c_in, k} weight[c_out, c_in, ky, kx] * mask[k] * sampled[k, c_in]

Composition with TTNN ops:
    1. Build sample grid (precomputed base + offset * scale, then (y,x)→(x,y) swap)
    2. ttnn.grid_sample(X, grid, K=9, batch_output_channels=True)  → (1, H, W, 9*C_in)
    3. mask_expanded = repeat_interleave(mask, C_in, dim=-1)         → (1, H, W, 9*C_in)
    4. modulated = samples * mask_expanded
    5. matmul with weight reshaped to (9*C_in, C_out)                 → (1, H, W, C_out)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch
import ttnn


@lru_cache(maxsize=None)
def _precompute_base_grid_yx_and_scale(
    H_in: int,
    W_in: int,
    H_out: int,
    W_out: int,
    kH: int,
    kW: int,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    align_corners: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute the constant pieces of the sample-grid construction.

    Returns:
        base_grid_yx: (1, H_out, W_out, 2*kH*kW) with channel order
            (y_0, x_0, y_1, x_1, ..., y_{K-1}, x_{K-1}), pre-normalized so that
            final_grid_yx = base_grid_yx + offset_dy_dx * scale_yx
        scale_yx:    (1, 1, 1, 2*kH*kW), (scale_y, scale_x) interleaved K times,
                     for normalizing the (dy, dx) offsets to [-1, 1] grid space.
    """
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    K = kH * kW

    # NOTE: ttnn.grid_sample uses align_corners=False semantics regardless of the
    # align_corners flag passed to it (as of this writing). We always use the
    # align_corners=False normalization so that grid coord = (2*y_pixel + 1)/H - 1.
    # This makes pixel coordinate y_pixel map to the correct grid coordinate for
    # ttnn.grid_sample to sample the same pixel as torchvision's DCN.
    scale_y = 2.0 / H_in
    scale_x = 2.0 / W_in
    bias_y = 1.0 / H_in - 1.0
    bias_x = 1.0 / W_in - 1.0
    # Param kept for API symmetry; not used.
    _ = align_corners

    iy = torch.arange(H_out, dtype=torch.float32)
    ix = torch.arange(W_out, dtype=torch.float32)
    iy_grid, ix_grid = torch.meshgrid(iy, ix, indexing="ij")  # (H_out, W_out)

    ky_idx = torch.arange(kH, dtype=torch.float32).view(kH, 1).expand(-1, kW).flatten()  # (K,)
    kx_idx = torch.arange(kW, dtype=torch.float32).view(1, kW).expand(kH, -1).flatten()  # (K,)

    base_y_raw = iy_grid.unsqueeze(-1) * sh - ph + ky_idx * dh  # (H_out, W_out, K)
    base_x_raw = ix_grid.unsqueeze(-1) * sw - pw + kx_idx * dw  # (H_out, W_out, K)

    base_y_norm = base_y_raw * scale_y + bias_y
    base_x_norm = base_x_raw * scale_x + bias_x

    # Stack interleaved (y, x): (H_out, W_out, K, 2) → (H_out, W_out, 2K)
    base = torch.stack([base_y_norm, base_x_norm], dim=-1).reshape(H_out, W_out, K * 2)
    base = base.unsqueeze(0).contiguous()  # (1, H_out, W_out, 2K)

    scale = torch.zeros(K * 2, dtype=torch.float32)
    scale[0::2] = scale_y
    scale[1::2] = scale_x
    scale = scale.view(1, 1, 1, K * 2).contiguous()

    return base, scale


def prepare_deform_conv_weight(weight: torch.Tensor) -> torch.Tensor:
    """Reshape torchvision-format weight for use in the final 1x1 matmul.

    Input:  (C_out, C_in, kH, kW)        — torchvision convention
    Output: (C_out, kH*kW*C_in)          — channels in k-major order (k=ky*kW+kx),
                                          matching ttnn.grid_sample's batch_output_channels=True
                                          which packs K groups of C_in channels.
    """
    C_out, C_in, kH, kW = weight.shape
    return weight.permute(0, 2, 3, 1).reshape(C_out, kH * kW * C_in).contiguous()


class TtDeformConv2dV2:
    """On-device modulated deformable conv2d (DCNv2).

    Pre-uploads constant grid/scale tensors once, then accepts dynamic (x, offset, mask)
    tensors at inference time. Designed to be allocated per-DyHead-block-per-conv-call-site
    so that the H_in/W_in/H_out/W_out shapes are fixed and the base_grid can be cached.

    Usage:
        dcn = TtDeformConv2dV2(device, weight, bias, C_in, C_out, kH=3, kW=3,
                               H_in, W_in, H_out, W_out, stride, padding, dilation)
        y = dcn(x_nhwc_ttnn, offset_nhwc_ttnn, mask_nhwc_ttnn)
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,  # (C_out, C_in, kH, kW)
        bias: Optional[torch.Tensor],  # (C_out,) or None
        C_in: int,
        C_out: int,
        kH: int,
        kW: int,
        H_in: int,
        W_in: int,
        H_out: int,
        W_out: int,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        align_corners: bool = True,
        offset_layout: str = "yx",
    ):
        self.device = device
        self.C_in = C_in
        self.C_out = C_out
        self.kH = kH
        self.kW = kW
        self.K = kH * kW
        self.H_in = H_in
        self.W_in = W_in
        self.H_out = H_out
        self.W_out = W_out
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.align_corners = align_corners
        assert offset_layout in ("yx", "xy"), f"offset_layout must be 'yx' or 'xy', got {offset_layout}"
        self.offset_layout = offset_layout

        # Precompute base_grid and scale on host, transfer once. When offset_layout is "xy"
        # the base_grid is built in (x_k, y_k) interleaved order so the runtime grid_yx -> grid_xy
        # swap is unneeded (caller is responsible for delivering (dx_k, dy_k) offsets).
        base_grid_torch, scale_torch = _precompute_base_grid_yx_and_scale(
            H_in, W_in, H_out, W_out, kH, kW, stride, padding, dilation, align_corners
        )
        if offset_layout == "xy":
            # Reorder last dim from (y_0, x_0, y_1, x_1, ...) to (x_0, y_0, x_1, y_1, ...).
            base_grid_torch = (
                base_grid_torch.reshape(1, H_out, W_out, kH * kW, 2)
                .flip(-1)
                .reshape(1, H_out, W_out, 2 * kH * kW)
                .contiguous()
            )
            scale_torch = scale_torch.reshape(1, 1, 1, kH * kW, 2).flip(-1).reshape(1, 1, 1, 2 * kH * kW).contiguous()
        self.base_grid = ttnn.from_torch(
            base_grid_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.scale = ttnn.from_torch(
            scale_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Prepare matmul weight in (K*C_in, C_out) layout for the final 1x1 mul.
        # matmul: (1, H*W, K*C_in) @ (K*C_in, C_out) -> (1, H*W, C_out)
        w_prepared = prepare_deform_conv_weight(weight)  # (C_out, K*C_in)
        w_for_matmul = w_prepared.t().contiguous()  # (K*C_in, C_out)
        self.weight_tt = ttnn.from_torch(
            w_for_matmul,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if bias is not None:
            self.bias_tt = ttnn.from_torch(
                bias.view(1, 1, C_out).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bias_tt = None

        # Use fp32 accumulation in the K*C_in=2304 reduction to limit bf16 precision drift.
        self._compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

    def __call__(self, x_nhwc, offset_nhwc, mask_nhwc):
        """
        Args:
            x_nhwc:      (1, H_in, W_in, C_in) NHWC ttnn — feature map.
            offset_nhwc: (1, H_out, W_out, 2*K) NHWC ttnn — channels in (dy_0, dx_0, ..., dy_{K-1}, dx_{K-1}) order.
            mask_nhwc:   (1, H_out, W_out, K)   NHWC ttnn — sigmoided modulation values per kernel cell.

        Returns:
            (1, H_out, W_out, C_out) NHWC ttnn.
        """
        H_out, W_out, K, C_in, C_out = self.H_out, self.W_out, self.K, self.C_in, self.C_out

        # Ensure offset is in ROW_MAJOR layout for downstream grid_sample.
        if offset_nhwc.layout != ttnn.ROW_MAJOR_LAYOUT:
            offset_nhwc = ttnn.to_layout(offset_nhwc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 1. grid = base_grid + offset * scale  (interleaved in self.offset_layout order)
        off_scaled = ttnn.multiply(offset_nhwc, self.scale, memory_config=ttnn.L1_MEMORY_CONFIG)
        grid = ttnn.add(self.base_grid, off_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(off_scaled)

        if self.offset_layout == "yx":
            # 2. Swap interleaved (y_k, x_k) -> (x_k, y_k) for grid_sample.
            grid_5d = ttnn.reshape(grid, (1, H_out, W_out, K, 2))
            y_comp = ttnn.slice(grid_5d, [0, 0, 0, 0, 0], [1, H_out, W_out, K, 1])
            x_comp = ttnn.slice(grid_5d, [0, 0, 0, 0, 1], [1, H_out, W_out, K, 2])
            grid_xy_5d = ttnn.concat([x_comp, y_comp], dim=4)
            grid_xy = ttnn.reshape(grid_xy_5d, (1, H_out, W_out, 2 * K))
            ttnn.deallocate(grid_5d)
            ttnn.deallocate(y_comp)
            ttnn.deallocate(x_comp)
        else:
            # base_grid already in XY-interleaved order — grid is the final sample grid.
            grid_xy = grid

        # Ensure grid is ROW_MAJOR for ttnn.grid_sample (it asserts on this).
        if grid_xy.layout != ttnn.ROW_MAJOR_LAYOUT:
            grid_xy = ttnn.to_layout(grid_xy, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Ensure x is ROW_MAJOR for grid_sample.
        if x_nhwc.layout != ttnn.ROW_MAJOR_LAYOUT:
            x_nhwc = ttnn.to_layout(x_nhwc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 3. grid_sample with K-batching → (1, H_out, W_out, C_in*K) channels k-major
        samples = ttnn.grid_sample(
            x_nhwc,
            grid_xy,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,  # ttnn.grid_sample uses align_corners=False semantics regardless
            batch_output_channels=True,
        )
        ttnn.deallocate(grid_xy)

        # 4. Apply modulation. Channels in `samples` are laid out (K, C_in) with K as the
        # slower-varying axis, so mask values can be broadcast over C_in by reshaping both
        # tensors to 5D and relying on ttnn's last-dim-1 broadcast. This avoids materializing
        # the K*C_in repeat-interleaved mask (29 MB at P3@80x80).
        big_tensor_bytes = H_out * W_out * K * C_in * 2  # bf16
        big_mem_config = ttnn.DRAM_MEMORY_CONFIG if big_tensor_bytes > 4 * 1024 * 1024 else ttnn.L1_MEMORY_CONFIG

        if samples.memory_config().buffer_type != ttnn.BufferType.DRAM and big_mem_config == ttnn.DRAM_MEMORY_CONFIG:
            samples = ttnn.to_memory_config(samples, big_mem_config)
        samples_5d = ttnn.reshape(samples, (1, H_out, W_out, K, C_in))
        mask_5d = ttnn.reshape(mask_nhwc, (1, H_out, W_out, K, 1))
        modulated_5d = ttnn.multiply(samples_5d, mask_5d, memory_config=big_mem_config)

        # 5. Final 1x1 weighted sum via matmul.
        modulated_flat = ttnn.reshape(modulated_5d, (1, H_out * W_out, K * C_in))
        modulated_tiled = ttnn.to_layout(modulated_flat, ttnn.TILE_LAYOUT, memory_config=big_mem_config)

        out_flat = ttnn.matmul(
            modulated_tiled,
            self.weight_tt,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_config,
        )
        if self.bias_tt is not None:
            out_flat = ttnn.add(out_flat, self.bias_tt, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Reshape back to (1, H_out, W_out, C_out)
        out_nhwc = ttnn.reshape(out_flat, (1, H_out, W_out, C_out))
        return out_nhwc
