
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cpp/ttnn/decorators.hpp>
#include <cpp/ttnn/tensor/tensor.hpp>
#include <vector>

namespace ttnn::operations::experimental {

struct RMSNormForwardOperation {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& input_tensor,
        const Tensor& gamma_tensor,
        bool return_intermediates = true,
        float epsilon = 1e-6F);

    static std::vector<std::optional<Tensor>> create_async_optional_output_tensors(
        const Tensor& input_tensor,
        const Tensor& gamma_tensor,
        bool return_intermediates = true,
        float epsilon = 1e-6F);
};
}  // namespace ttnn::operations::experimental
namespace ttnn::experimental {
constexpr auto rmsnorm_fw_op = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::rmsnorm_fw_op",
    ttnn::operations::experimental::RMSNormForwardOperation>();
}  // namespace ttnn::experimental
