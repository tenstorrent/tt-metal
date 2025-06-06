// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw.hpp"

#include "device/rmsnorm_bw_device_operation.hpp"  // if needed

namespace ttml::metal::ops::rmsnorm_bw {

std::vector<std::optional<ttnn::Tensor>> RMSNormBackwardOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float epsilon) {
    // std::cerr << "Within RMSNormBackwardOperation::invoke" << std::endl;
    // dL_dout_tensor.print();
    auto result = ttnn::prim::ttml_rmsnorm_bw(input_tensor, gamma_tensor, rms_tensor, dL_dout_tensor, epsilon);

    // Always return both gradients over input and gamma. Consider parameterizing this in the future and nullopt?
    return {result[0], result[1]};
}

}  // namespace ttml::metal::ops::rmsnorm_bw
