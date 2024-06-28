/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_nll_loss_unreduced_backward_impl(
    const Tensor &target,
    const std::optional<const Tensor> weight,
    const Tensor &output_grad,
    const Tensor &input_grad,
    const int32_t ignore_index,
    const CoreRange core_range,
    const DeviceComputeKernelConfig compute_kernel_config);

struct MorehNllLossUnreducedBackward {
    int32_t ignore_index;

    const MemoryConfig input_grad_mem_config;
    const CoreRange core_range;  // unused for now
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("ignore_index", "input_grad_mem_config", "compute_kernel_config");
    const auto attribute_values() const { return std::make_tuple(
        std::cref(this->ignore_index),
        std::cref(this->input_grad_mem_config),
        std::cref(this->compute_kernel_config)
        ); }
};

Tensor moreh_nll_loss_unreduced_backward(
    const Tensor &target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const Tensor &output_grad_tensor,
    const std::optional<const Tensor> input_grad_tensor,
    const int32_t ignore_index,
    const MemoryConfig &input_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
