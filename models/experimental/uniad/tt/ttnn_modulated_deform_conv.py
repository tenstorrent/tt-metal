# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-side modulated deformable conv 2D — starting scaffold using ttnn.grid_sample.

UniAD's ResNet101 backbone has 26 modulated deformable conv calls per
inference (stages 3 & 4 with `stage_with_dcn=(False, False, True, True)`).
The host fallback path in TtModulatedDeformConv2dPack pulls x / offset /
mask to host and calls `torchvision.ops.deform_conv2d` (DCNv2) on CPU —
TT_DCN_TIMING=1 shows the CPU compute dominates that path.

This module decomposes modulated deformable conv (Dai et al. 2018) into
K*K (= 9 for K=3) `ttnn.grid_sample` calls + per-kernel-position
matmuls, so the whole op can stay on device and the surrounding ResNet
can eventually be captured by Metal Trace.

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
  - mode="bilinear", align_corners=False (paired with the matching base
    grid formula; equivalent to align_corners=True for the DCNv2
    reference since both invert to the same pixel-space sample location)

ttnn.grid_sample's input is bounded to C_in <= TILE_WIDTH * 8 = 256
channels. UniAD DCNs hit C_in = 1024 (stage 3) and 2048 (stage 4), so
this module slices x along C_in into ≤256-channel chunks, runs the K*K
grid_sample / mask / matmul pipeline per chunk, and adds the partial
C_out outputs. The summed result is mathematically identical to a single
matmul over the full K*K*C_in reduction axis.
"""

import torch
import ttnn

# ttnn.grid_sample's wide-reduction cap (TILE_WIDTH * max_tiles_per_reduction
# = 32 * 8) on the input channel dimension. Inputs wider than this must be
# split along C before sampling.
_GRID_SAMPLE_C_CAP = 256


class TtModulatedDeformConv2dDevice:
    """Device-side modulated deformable conv 2D.

    Stateful so per-instance constants (base grids, normalized weight
    slices, bias) can be uploaded once and reused across warm calls.
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
    ):
        assert groups == 1, "device DCN prototype only supports groups=1"
        assert deform_groups == 1, "device DCN prototype only supports deform_groups=1"

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.C_out, self.C_in, self.K, _ = weight.shape
        assert weight.shape[2] == weight.shape[3], "kernel must be square"

        # Pick a chunk size that divides C_in cleanly and is ≤ grid_sample's
        # wide-reduction cap. UniAD's C_in values (256, 512, 1024, 2048) all
        # divide _GRID_SAMPLE_C_CAP, so a single chunk size works for every
        # instance; the assert flags any future shape that would need a more
        # general (ragged) chunking.
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
        # fp32. The downstream fused matmul reuses this config so the
        # K*K*C_in -> C_out reduction also stays in fp32.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        # Base grid is built lazily on first call (depends on H_out/W_out,
        # which we only learn at runtime).
        self._base_grid_cache = {}  # keyed by (B, H_in, W_in, H_out, W_out)

    def _build_base_grid(self, H_in, W_in, H_out, W_out, batch):
        """Build the per-kernel-position normalized base grids.

        For each kp (kh, kw), the base sample position in input space is:
            sy_base = h_o * stride[0] - padding[0] + kh * dilation[0]
            sx_base = w_o * stride[1] - padding[1] + kw * dilation[1]

        Normalizing for ttnn.grid_sample with `align_corners=False`
        (image_coord = (grid + 1) * H / 2 - 0.5):
            gy = (2 * sy + 1) / H - 1

        DCNv2 samples at pixel-space `sy` directly; align_corners=False and
        align_corners=True both invert to that same sy, so the two are
        equivalent for the reference. We use False because the formula
        avoids the H=1 edge case and the grid_sample call below mirrors it.

        Returns a list of K*K tuples (gx_base, gy_base), each a torch
        tensor of shape (batch, H_out, W_out) ready for upload.
        """
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        h_o_coords = torch.arange(H_out, dtype=torch.float32)
        w_o_coords = torch.arange(W_out, dtype=torch.float32)
        h_grid, w_grid = torch.meshgrid(h_o_coords, w_o_coords, indexing="ij")

        base_grids = []
        for kh in range(self.K):
            for kw in range(self.K):
                sy_base = h_grid * sh - ph + kh * dh
                sx_base = w_grid * sw - pw + kw * dw

                gy_base = (2.0 * sy_base + 1.0) / H_in - 1.0
                gx_base = (2.0 * sx_base + 1.0) / W_in - 1.0

                gy_base_b = gy_base.unsqueeze(0).expand(batch, H_out, W_out).contiguous()
                gx_base_b = gx_base.unsqueeze(0).expand(batch, H_out, W_out).contiguous()
                base_grids.append((gx_base_b, gy_base_b))

        gy_scale = 2.0 / H_in
        gx_scale = 2.0 / W_in
        return base_grids, gy_scale, gx_scale

    def __call__(self, x_nhwc, offset_yx_nhwc, mask_nhwc):
        """Forward.

        Args:
          x_nhwc: (B, H_in, W_in, C_in) bfloat16 NHWC on device
          offset_yx_nhwc: (B, H_out, W_out, 2*K*K) bfloat16 NHWC, in the
              DCNv2 interleaved channel order — channel `2*kk` is the
              y-offset for kernel position kk and channel `2*kk + 1` is x.
              This is the layout torchvision/mmcv `deform_conv2d` reads
              directly, so the production wrapper's existing `concat(o1, o2)`
              output can be passed through without permutation.
          mask_nhwc: (B, H_out, W_out, K*K) bfloat16 NHWC — modulation masks
              after sigmoid.

        Returns:
          (B, H_out, W_out, C_out) bfloat16 NHWC TILE_LAYOUT on device.
        """
        B, H_in, W_in, C_in = x_nhwc.shape
        _, H_out, W_out, _ = mask_nhwc.shape
        K = self.K

        cache_key = (B, H_in, W_in, H_out, W_out)
        if cache_key not in self._base_grid_cache:
            base_grids, gy_scale, gx_scale = self._build_base_grid(H_in, W_in, H_out, W_out, B)
            # Upload base grids once per (input-shape, output-shape) combo.
            base_grids_dev = []
            for gx_base_b, gy_base_b in base_grids:
                # ttnn.grid_sample wants grid as (B, H_out, W_out, 2) with (x, y) order.
                grid_xy = torch.stack([gx_base_b, gy_base_b], dim=-1)
                base_grids_dev.append(
                    ttnn.from_torch(grid_xy, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
                )
            self._base_grid_cache[cache_key] = (base_grids_dev, gy_scale, gx_scale)

        base_grids_dev, gy_scale, gx_scale = self._base_grid_cache[cache_key]

        # grid_sample's reader expects ROW_MAJOR for both input and grid;
        # convert x, offsets, and mask up front so the per-kp slice + add +
        # concat below stays in row-major layout.
        x_rm = ttnn.to_layout(x_nhwc, ttnn.ROW_MAJOR_LAYOUT)
        if offset_yx_nhwc.layout != ttnn.ROW_MAJOR_LAYOUT:
            offset_yx_nhwc = ttnn.to_layout(offset_yx_nhwc, ttnn.ROW_MAJOR_LAYOUT)
        if mask_nhwc.layout != ttnn.ROW_MAJOR_LAYOUT:
            mask_nhwc = ttnn.to_layout(mask_nhwc, ttnn.ROW_MAJOR_LAYOUT)

        # Pre-build a single packed grid of shape (B, H_out, W_out, 2*K*K)
        # and a packed mask of shape (B, H_out, W_out, K*K). With
        # `batch_output_channels=True`, one ttnn.grid_sample call replaces
        # the K*K per-channel-chunk inner loop the original code ran. The
        # output shape (B, H_out, W_out, c_chunk*K*K) matches the previous
        # concat-of-K*K-blocks shape so the downstream matmul weights are
        # unchanged. offset_yx_nhwc is in the DCNv2 interleaved layout
        # (y0, x0, y1, x1, …) so per kp we slice (2*kk, 2*kk+1) for (y, x).
        packed_grid_parts = []
        packed_mask_parts = []
        for kk in range(K * K):
            oy = offset_yx_nhwc[..., 2 * kk : 2 * kk + 1]
            ox = offset_yx_nhwc[..., 2 * kk + 1 : 2 * kk + 2]
            oy_norm = ttnn.multiply(oy, gy_scale)
            ox_norm = ttnn.multiply(ox, gx_scale)
            gx_base_full = base_grids_dev[kk][..., 0:1]
            gy_base_full = base_grids_dev[kk][..., 1:2]
            gx = ttnn.add(gx_base_full, ox_norm)
            gy = ttnn.add(gy_base_full, oy_norm)
            packed_grid_parts.append(ttnn.concat([gx, gy], dim=-1))  # (B, H_out, W_out, 2)
            packed_mask_parts.append(mask_nhwc[..., kk : kk + 1])  # (B, H_out, W_out, 1)
        # Concat along the last dim: (B, H_out, W_out, 2*K*K) and (B, H_out, W_out, K*K).
        packed_grid = ttnn.concat(packed_grid_parts, dim=-1)
        packed_mask = ttnn.concat(packed_mask_parts, dim=-1)

        # For each channel chunk: one grid_sample call with K coordinate
        # sets, multiply by the mask (broadcast across c_chunk), matmul
        # into C_out, accumulate. The matmul's fp32 accumulator covers the
        # K*K*c_chunk reduction; only c_chunk-wide bf16 partials are summed
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
            )  # (B, H_out, W_out, c_chunk*K*K)
            # Apply per-kk mask. Reshape sampled to expose the kk dim,
            # broadcast-multiply by the (B, H_out, W_out, K*K, 1) mask, then
            # reshape back to flatten kk into the channel axis.
            sampled = ttnn.reshape(sampled, (B, H_out, W_out, K * K, self.c_chunk))
            mask_b = ttnn.reshape(packed_mask, (B, H_out, W_out, K * K, 1))
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
