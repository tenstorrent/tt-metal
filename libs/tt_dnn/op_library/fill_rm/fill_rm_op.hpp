#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// Generates an NCHW row-major tensor and fill it with ones up to hOnes, wOnes in each HW tile
// with the rest padded with zeros. So for H=2, W=3, hFill=1, wFill=2 the following tensor will be generated:
// +------------> W
// | hi hi lo
// | lo lo lo
// |
// v H
//
// H, W are expected to be multiples of 32
// The 'any' Tensor arg is only used to pass the device and resulting tensor dtype
// val_hi/lo are expected to be uint16 encodings of bfloat16 numbers, so 0x3f80 for 1.0 etc.
Tensor fill_rm (int N, int C, int H, int W, int hOnes, int wOnes, const Tensor& any, int val_hi, int val_lo);

inline
Tensor fill_ones_rm (int N, int C, int H, int W, int hOnes, int wOnes, const Tensor& any) {
    // 0x3f80 is 1.0 in BF16
    return fill_rm(N, C, H, W, hOnes, wOnes, any, 0x3F80, 0);
}

}  // namespace ll_buda

}  // namespace tt
