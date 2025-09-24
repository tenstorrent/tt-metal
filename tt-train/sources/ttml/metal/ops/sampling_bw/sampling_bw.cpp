// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sampling_bw.hpp"

namespace ttml::metal::ops::sampling_bw {

ttnn::Tensor SamplingBackwardOperation::invoke(const ttnn::Tensor& input_tensor) {
    TT_FATAL(false, "Sampling backward operation is not implemented yet.");

    return input_tensor;
}
}  // namespace ttml::metal::ops::sampling_bw
