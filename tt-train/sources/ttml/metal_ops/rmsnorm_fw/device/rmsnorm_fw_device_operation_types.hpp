// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cpp/ttnn/tensor/tensor.hpp>

namespace ttnn::operations::experimental::rmsnorm_fw {

struct operation_attributes_t {
    bool return_intermediates{false};
    float epsilon{1e-6F};
};

struct tensor_args_t {
    const Tensor& input;
    const Tensor& gamma;

    std::optional<Tensor> preallocated_rms;
    std::optional<Tensor> preallocated_output;
};

using tensor_return_value_t = std::vector<Tensor>;

using spec_return_value_t = std::vector<TensorSpec>;

}  // namespace ttnn::operations::experimental::rmsnorm_fw
