// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mean_all_cores.hpp"

#include "device/mean_all_cores_device_operation.hpp"

namespace ttml::metal::ops::examples::mean_all_cores {

ttnn::Tensor MeanAllCoresOperation::invoke(const ttnn::Tensor& input_tensor) {
    auto result = ttnn::prim::ttml_mean_all_cores(input_tensor);
    return result;
}
}  // namespace ttml::metal::ops::examples::mean_all_cores

