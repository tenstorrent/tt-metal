# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-side modulated deformable conv 2D (DCNv2): ttnn.experimental.modulated_deform_conv2d,
or ttnn.experimental.modulated_deform_im2col and one matmul.

The ResNet101 backbone has 26 modulated deformable convs per inference (layer3 and
layer4, `stage_with_dcn=(False, False, True, True)`). Per output pixel and kernel position the
input is sampled bilinearly at the offset position and scaled by the mask (DCNv2, Zhu et al. 2019):

    acc = 0
    for kh in range(K):
      for kw in range(K):
        kk = kh*K + kw
        sy = h_o*stride[0] - padding[0] + kh*dilation[0] + offset[b, h_o, w_o, 2*kk]
        sx = w_o*stride[1] - padding[1] + kw*dilation[1] + offset[b, h_o, w_o, 2*kk + 1]
        for c_in in range(C_in):
          sampled = bilinear_interp(x[b, :, :, c_in], sy, sx) * mask[b, h_o, w_o, kk]
          acc += weight[c_out, c_in, kh, kw] * sampled
    output[b, h_o, w_o, c_out] = acc + bias[c_out]

The im2col op emits the masked samples as a (B*H_out*W_out, K*K*C_in) column matrix in
(kk, c_in) order, and a matmul with the weight laid out the same way finishes the conv.
The fused op runs both in one program, so the columns never reach DRAM.
It takes DCNv2's offset conv output as it is: the offsets, in pixels and (y, x) per kernel
position, then the mask logits, whose sigmoid it applies itself.
"""

import math

import ttnn
from models.experimental.bevformer.model_config import DCN_FUSED_CHANNELS, DCN_MATMUL_IN0_BLOCK_W


class TTModulatedDeformConv2dDevice:
    """Device-side modulated deformable conv 2D, groups and deform_groups 1. The weight and bias
    are uploaded at construction."""

    def __init__(
        self,
        weight,
        bias,
        kernel_size,
        stride,
        padding,
        dilation,
        device,
        input_shape,
        relu=False,
    ):
        """``weight`` and ``bias`` are host tensors from ``model_preprocessing``: the
        ``(K*K*C_in, C_out)`` matmul weight, rows in the columns' (kk, c_in) order, and the
        ``(1, C_out)`` bias or None. ``kernel_size`` is the square kernel's K. ``input_shape`` is
        the (batch, height, width) of the NHWC input every call will see. ``relu`` applies a ReLU
        to the output, in the op that ends the conv."""
        self.stride = tuple(stride)
        self.padding = tuple(padding)
        self.dilation = tuple(dilation)
        self.K = kernel_size
        self.C_out = weight.shape[-1]
        self.C_in = weight.shape[0] // (self.K * self.K)

        self.weight = ttnn.to_device(weight, device)
        self.bias = None if bias is None else ttnn.to_device(bias, device)
        self.relu = relu
        self.fused = self.C_in in DCN_FUSED_CHANNELS

        batch, H_in, W_in = input_shape
        H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (self.K - 1) - 1) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (self.K - 1) - 1) // self.stride[1] + 1
        self.io_shape = (batch, H_in, W_in, H_out, W_out)
        if self.fused:
            return

        # The K*K*C_in reduction accumulates in fp32.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        # Each core takes a run of column rows and the whole output width; the weight is
        # multicast, the columns are read once.
        grid = device.compute_with_storage_grid_size()
        row_tiles = math.ceil(batch * H_out * W_out / ttnn.TILE_SIZE)
        out_tiles = self.C_out // ttnn.TILE_SIZE
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=DCN_MATMUL_IN0_BLOCK_W[self.C_in],
            out_subblock_h=1,
            out_subblock_w=math.gcd(out_tiles, 4),
            per_core_M=math.ceil(row_tiles / (grid.x * grid.y)),
            per_core_N=out_tiles,
            fuse_batch=True,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None,
            mcast_in0=False,
        )

    def __call__(self, x_nhwc, offset_mask_nhwc):
        """``x_nhwc`` (B, H_in, W_in, C_in) and ``offset_mask_nhwc`` (B, H_out, W_out, 3*K*K), the
        offset conv's output: the offsets in pixels, (y, x) per kernel position, then the mask
        logits. Both inputs bfloat16, ROW_MAJOR, interleaved. Returns the
        (1, 1, B*H_out*W_out, C_out) bfloat16 TILE output, conv2d's layout."""
        B, H_in, W_in, _ = x_nhwc.shape
        _, H_out, W_out, _ = offset_mask_nhwc.shape
        assert (B, H_in, W_in, H_out, W_out) == self.io_shape, (
            f"DCN built for (B, H_in, W_in, H_out, W_out) = {self.io_shape}, "
            f"called with {(B, H_in, W_in, H_out, W_out)}"
        )
        if self.fused:
            return ttnn.experimental.modulated_deform_conv2d(
                x_nhwc,
                offset_mask_nhwc,
                self.weight,
                self.bias,
                kernel_size=(self.K, self.K),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                relu=self.relu,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        columns = ttnn.experimental.modulated_deform_im2col(
            x_nhwc,
            offset_mask_nhwc,
            kernel_size=(self.K, self.K),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            # The columns are K*K times the input, too large for L1 whatever the input's placement.
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = ttnn.linear(
            columns,
            self.weight,
            bias=self.bias,
            program_config=self.program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(columns)
        return output
