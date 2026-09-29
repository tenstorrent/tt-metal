# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-side modulated deformable conv 2D, composed from ttnn.grid_sample and matmul.

The ResNet101 backbone has 26 modulated deformable convs per inference (layer3 and
layer4, `stage_with_dcn=(False, False, True, True)`). This module decomposes modulated
deformable conv (Dai et al. 2018) into K*K (= 9 for K=3) sample positions per output
pixel, one `ttnn.grid_sample` over all of them and a matmul over K*K*C_in, so the
whole op stays on device.

Math (per output (h_o, w_o), output channel c_out):

    acc = 0
    for kh in range(K):
      for kw in range(K):
        kk = kh*K + kw
        sy = h_o*stride[0] - padding[0] + kh*dilation[0] + offset_y[b, kk, h_o, w_o]
        sx = w_o*stride[1] - padding[1] + kw*dilation[1] + offset_x[b, kk, h_o, w_o]
        for c_in in range(C_in):
          sampled = bilinear_interp(x[b, c_in, :, :], sy, sx) * mask[b, kk, h_o, w_o]
          acc += weight[c_out, c_in, kh, kw] * sampled
    output[b, c_out, h_o, w_o] = acc + bias[c_out]

`ttnn.grid_sample` expects:
  - input NHWC, shape (N, H_in, W_in, C)
  - grid (N, H_out, W_out, 2) in (x, y) order normalized to [-1, 1]
  - mode="bilinear", align_corners=False. The base grid and the offset scales
    use the align_corners=False normalization, gy = (2*sy + 1)/H - 1; switching
    to align_corners=True needs (H - 1) in both formulas.

The offsets reach this module in pixels but already in grid_sample's (x, y)
channel order: `grid_offset_order` gives the reorder from DCNv2's (y, x), which
the preprocessor folds into the offset conv. The forward scales them to [-1, 1] units
with one multiply and adds the base grid.

x is split along C_in into chunks of at most 256 channels (one chunk for the
layer3 DCNs, C_in = 256; two for layer4, C_in = 512). Each chunk runs the
grid_sample / mask / matmul pipeline and the partial C_out outputs are added,
which equals a single matmul over the full K*K*C_in reduction axis.
ttnn.grid_sample does not need the split, since it tiles wide channel dims
internally; the chunk size bounds the K*K*c_chunk width of the sampled tensor
the matmul reads.
"""

import torch
import ttnn

# Largest C_in chunk sampled at once; see the module docstring.
_GRID_SAMPLE_C_CAP = 256


def grid_offset_order(kernel_positions):
    """The reorder from DCNv2's offset channels (y0, x0, y1, x1, ...) to the (x0, y0, x1, y1, ...)
    the device DCN takes: channel c of its input is DCNv2 channel ``source[c]``."""
    source = []
    for kk in range(kernel_positions):
        source += [2 * kk + 1, 2 * kk]
    return source


class TtModulatedDeformConv2dDevice:
    """Device-side modulated deformable conv 2D.

    Per-instance constants (base grids, weight slices, bias) are uploaded once, at
    construction.
    """

    def __init__(
        self,
        weight,  # torch.Tensor (C_out, C_in, K, K)
        bias,  # torch.Tensor (C_out,) or None
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
        device,
        input_shape,
    ):
        """``input_shape`` is the (batch, height, width) of the NHWC input every call will see.
        It fixes the sampling base grid, which is built and uploaded here so a forward does
        no host work."""
        assert groups == 1, f"device DCN supports groups=1 only, got {groups}"
        assert deform_groups == 1, f"device DCN supports deform_groups=1 only, got {deform_groups}"

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.C_out, self.C_in, self.K, _ = weight.shape
        assert weight.shape[2] == weight.shape[3], "kernel must be square"

        # The DCN C_in values here (256, 512) are multiples of _GRID_SAMPLE_C_CAP,
        # so one chunk size fits every instance; the assert flags a shape that
        # would need ragged chunking.
        self.c_chunk = min(self.C_in, _GRID_SAMPLE_C_CAP)
        assert (
            self.C_in % self.c_chunk == 0
        ), f"C_in={self.C_in} is not a multiple of grid_sample chunk size {self.c_chunk}"
        self.n_c_chunks = self.C_in // self.c_chunk

        # Per-chunk concatenated weight (K*K*c_chunk, C_out): for chunk q, the
        # rows of the matmul's reduction axis are the K*K weight slices
        # restricted to the chunk's C_in range. Per chunk we run a full
        # K*K*c_chunk -> C_out matmul, then sum the partials.
        self.weight_cat_chunks = []
        for q in range(self.n_c_chunks):
            c_start = q * self.c_chunk
            c_end = c_start + self.c_chunk
            chunk_blocks = []
            for kh in range(self.K):
                for kw in range(self.K):
                    chunk_blocks.append(weight[:, c_start:c_end, kh, kw].t().contiguous())  # (c_chunk, C_out)
            wc_chunk = torch.cat(chunk_blocks, dim=0).contiguous()  # (K*K*c_chunk, C_out)
            self.weight_cat_chunks.append(
                ttnn.from_torch(wc_chunk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            )

        if bias is not None:
            self.bias = ttnn.from_torch(bias.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        else:
            self.bias = None

        # HiFi4 + fp32 accumulator + no math approx. ttnn.grid_sample
        # bilinear reads bf16 corner sticks and would normally accumulate
        # in bf16; fp32_dest_acc_en lifts the bilinear pool reduction to
        # fp32. The matmul reuses this config so the K*K*C_in -> C_out
        # reduction also accumulates in fp32.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        batch, H_in, W_in = input_shape
        H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (self.K - 1) - 1) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (self.K - 1) - 1) // self.stride[1] + 1
        self.io_shape = (batch, H_in, W_in, H_out, W_out)
        # Pixel offsets in (x, y) order times these give grid offsets.
        self.grid_scale = ttnn.from_torch(
            torch.tensor([2.0 / W_in, 2.0 / H_in] * (self.K * self.K)).reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.base_grid = ttnn.from_torch(
            self._build_base_grid(H_in, W_in, H_out, W_out, batch),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def _build_base_grid(self, H_in, W_in, H_out, W_out, batch):
        """Build the normalized base grid of every kernel position.

        For each kernel position (kh, kw), the base sample position in input space is:
            sy_base = h_o * stride[0] - padding[0] + kh * dilation[0]
            sx_base = w_o * stride[1] - padding[1] + kw * dilation[1]

        grid_sample with `align_corners=False` maps grid g to image coordinate
        (g + 1) * H / 2 - 0.5, so DCNv2's pixel-space sy is reached at:
            gy = (2 * sy + 1) / H - 1

        Returns a torch tensor of shape (batch, H_out, W_out, 2*K*K) holding
        (gx, gy) for each kernel position in kk order.
        """
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        h_o_coords = torch.arange(H_out, dtype=torch.float32)
        w_o_coords = torch.arange(W_out, dtype=torch.float32)
        h_grid, w_grid = torch.meshgrid(h_o_coords, w_o_coords, indexing="ij")

        channels = []
        for kh in range(self.K):
            for kw in range(self.K):
                sy_base = h_grid * sh - ph + kh * dh
                sx_base = w_grid * sw - pw + kw * dw
                channels.append((2.0 * sx_base + 1.0) / W_in - 1.0)
                channels.append((2.0 * sy_base + 1.0) / H_in - 1.0)
        grid = torch.stack(channels, dim=-1)  # (H_out, W_out, 2*K*K)
        return grid.unsqueeze(0).expand(batch, H_out, W_out, 2 * self.K * self.K).contiguous()

    def __call__(self, x_nhwc, offset_xy_nhwc, mask_nhwc):
        """Forward.

        Args:
          x_nhwc: (B, H_in, W_in, C_in) NHWC TILE on device, bfloat16 or bfloat8_b;
              it is sampled as bfloat16 ROW_MAJOR.
          offset_xy_nhwc: (B, H_out, W_out, 2*K*K) bfloat16 NHWC TILE pixel offsets in
              (x, y) order per kernel position, see `grid_offset_order`.
          mask_nhwc: (B, H_out, W_out, K*K) bfloat16 NHWC — modulation masks
              after sigmoid.

        Returns:
          (B, H_out, W_out, C_out) bfloat16 NHWC TILE_LAYOUT on device.
        """
        B, H_in, W_in, C_in = x_nhwc.shape
        _, H_out, W_out, _ = mask_nhwc.shape
        K = self.K

        assert (B, H_in, W_in, H_out, W_out) == self.io_shape, (
            f"DCN built for (B, H_in, W_in, H_out, W_out) = {self.io_shape}, "
            f"called with {(B, H_in, W_in, H_out, W_out)}"
        )
        # grid_sample's reader expects ROW_MAJOR for both input and grid.
        x_rm = ttnn.to_layout(x_nhwc, ttnn.ROW_MAJOR_LAYOUT)
        grid_offset = ttnn.to_layout(ttnn.multiply(offset_xy_nhwc, self.grid_scale), ttnn.ROW_MAJOR_LAYOUT)
        if mask_nhwc.layout != ttnn.ROW_MAJOR_LAYOUT:
            mask_nhwc = ttnn.to_layout(mask_nhwc, ttnn.ROW_MAJOR_LAYOUT)

        # One (B, H_out, W_out, 2*K*K) grid for all K*K kernel positions. With
        # `batch_output_channels=True`, a single ttnn.grid_sample call samples all of
        # them and returns (B, H_out, W_out, K*K*c_chunk) in kk-major channel order,
        # the row order weight_cat_chunks is built in.
        packed_grid = ttnn.add(self.base_grid, grid_offset)
        mask_b = ttnn.reshape(mask_nhwc, (B, H_out, W_out, K * K, 1))

        # For each channel chunk: one grid_sample call over the K*K kernel
        # positions, multiply by the mask (broadcast across c_chunk), matmul
        # into C_out, accumulate. The matmul's fp32 accumulator covers the
        # K*K*c_chunk reduction; only the C_out-wide bf16 partials are summed
        # across channel chunks.
        output_acc = None
        for q in range(self.n_c_chunks):
            c_start = q * self.c_chunk
            c_end = c_start + self.c_chunk
            x_chunk = x_rm if self.n_c_chunks == 1 else ttnn.slice(x_rm, [0, 0, 0, c_start], [B, H_in, W_in, c_end])

            sampled = ttnn.grid_sample(
                x_chunk,
                packed_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
                batch_output_channels=True,
                compute_kernel_config=self.compute_kernel_config,
            )  # (B, H_out, W_out, K*K*c_chunk), kk-major
            # Apply per-kk mask. Reshape sampled to expose the kk dim,
            # broadcast-multiply by the (B, H_out, W_out, K*K, 1) mask, then
            # reshape back to flatten kk into the channel axis.
            sampled = ttnn.reshape(sampled, (B, H_out, W_out, K * K, self.c_chunk))
            weighted = ttnn.multiply(sampled, mask_b)
            weighted = ttnn.reshape(weighted, (B, H_out, W_out, K * K * self.c_chunk))

            weighted_tile = ttnn.to_layout(weighted, ttnn.TILE_LAYOUT)
            # Fold (B, H_out, W_out) into M so the matmul heuristic sees a
            # large 2-D problem instead of bcast_batch with M=W_out, which
            # pins the kernel to 8 cores on small spatial dims.
            weighted_tile_flat = ttnn.reshape(weighted_tile, (1, 1, B * H_out * W_out, K * K * self.c_chunk))
            partial = ttnn.matmul(
                weighted_tile_flat,
                self.weight_cat_chunks[q],
                compute_kernel_config=self.compute_kernel_config,
            )
            partial = ttnn.reshape(partial, (B, H_out, W_out, partial.shape[-1]))
            output_acc = partial if output_acc is None else ttnn.add(output_acc, partial)

        if self.bias is not None:
            output_acc = ttnn.add(output_acc, self.bias)

        return output_acc
