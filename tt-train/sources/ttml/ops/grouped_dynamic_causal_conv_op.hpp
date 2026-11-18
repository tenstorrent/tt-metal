// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// DFlash 2's grouped dynamic causal convolution (z-lab dflash `_grouped_dynamic_convolve`), with autograd for all
// three inputs. Over independent blocks of `block_size` consecutive rows:
//
//   out[t, c] = sum_{o < K} (base_kernel[o, c] + dynamic_kernel[t, o * G + c / group_size]) * input[t - o, c],   G = C
//   / group_size
//
// where input[t - o] is zero when row t - o lies before t's block: the convolution never reads across a block
// boundary. One call is one of upstream's `prepare` or `finish` convolutions; `dynamic_kernel` is that half of the
// `kernel_projection` output and `base_kernel` that half of upstream's `base_kernel` parameter.
//
// input   [B, 1, S, C]     TILE; S a multiple of block_size. [1, 1, blocks * block_size, C] avoids tile padding
//                          when block_size < 32 (b blocks of 8 rows as [b, 1, 8, C] pad every block to 32 rows).
// dynamic_kernel [B, 1, S, K * G] TILE; column o * G + g is offset o's kernel value for group g, upstream's
//                          `view(..., K, G)` order. G must be a multiple of 32.
// base_kernel    [1, 1, K, C]     TILE; K <= block_size.
// Returns [B, 1, S, C].
autograd::TensorPtr grouped_dynamic_causal_conv(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& dynamic_kernel,
    const autograd::TensorPtr& base_kernel,
    uint32_t block_size,
    uint32_t group_size);

}  // namespace ttml::ops
