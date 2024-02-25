/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_dnn/op_library/operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_nll_loss_backward_impl(
    const Tensor &input, const Tensor &target, const std::optional<const Tensor> weight, const std::optional<const Tensor> divisor, const Tensor &output_grad, const Tensor &input_grad, const int32_t ignore_index, const bool reduction_mean, const CoreRange core_range);

struct MorehNllLossBackward {
    int32_t ignore_index;
    bool reduction_mean;

    const MemoryConfig output_mem_config;
    const CoreRange core_range;  // unused for now

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    static constexpr auto attribute_names = std::make_tuple("output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->output_mem_config));
    }
};

Tensor moreh_nll_loss_backward_(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const Tensor& output_grad_tensor,
    const Tensor& input_grad_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor moreh_nll_loss_backward(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const Tensor& output_grad_tensor,
    const Tensor& input_grad_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary
}  // namespace operations
}  // namespace tt
