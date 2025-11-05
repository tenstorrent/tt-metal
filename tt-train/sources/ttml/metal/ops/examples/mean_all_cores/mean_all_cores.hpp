// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::examples::mean_all_cores {

struct MeanAllCoresOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input  // input tensor (B, 1, N, H)
    );
};
}  // namespace ttml::metal::ops::examples::mean_all_cores

