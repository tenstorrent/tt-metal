# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-side modulated deformable conv 2D (DCNv2): ttnn.experimental.modulated_deform_im2col
and one matmul.

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
Offsets are DCNv2's own: pixels, (y, x) per kernel position.
"""

import ttnn


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
        to the output, in the matmul."""
        self.stride = tuple(stride)
        self.padding = tuple(padding)
        self.dilation = tuple(dilation)
        self.K = kernel_size
        self.C_out = weight.shape[-1]
        self.C_in = weight.shape[0] // (self.K * self.K)

        self.weight = ttnn.to_device(weight, device)
        self.bias = None if bias is None else ttnn.to_device(bias, device)
        # The K*K*C_in reduction accumulates in fp32.
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
        self.relu = relu

    def __call__(self, x_nhwc, offset_nhwc, mask_nhwc):
        """``x_nhwc`` (B, H_in, W_in, C_in), ``offset_nhwc`` (B, H_out, W_out, 2*K*K) in pixels,
        (y, x) per kernel position, and ``mask_nhwc`` (B, H_out, W_out, K*K) after its sigmoid,
        all bfloat16. Returns the (1, 1, B*H_out*W_out, C_out) bfloat16 TILE output, conv2d's
        layout."""
        B, H_in, W_in, _ = x_nhwc.shape
        _, H_out, W_out, _ = mask_nhwc.shape
        assert (B, H_in, W_in, H_out, W_out) == self.io_shape, (
            f"DCN built for (B, H_in, W_in, H_out, W_out) = {self.io_shape}, "
            f"called with {(B, H_in, W_in, H_out, W_out)}"
        )
        # The im2col reads every input as ROW_MAJOR sticks.
        columns = ttnn.experimental.modulated_deform_im2col(
            ttnn.to_layout(x_nhwc, ttnn.ROW_MAJOR_LAYOUT),
            ttnn.to_layout(offset_nhwc, ttnn.ROW_MAJOR_LAYOUT),
            ttnn.to_layout(mask_nhwc, ttnn.ROW_MAJOR_LAYOUT),
            kernel_size=(self.K, self.K),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        output = ttnn.linear(
            columns,
            self.weight,
            bias=self.bias,
            activation="relu" if self.relu else None,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(columns)
        return output
