// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_op.hpp"

namespace tt {

using namespace constants;

namespace operations {
namespace primary {

namespace {

inline void check_tensor(const Tensor &tensor, const std::string &op_name) {
    TT_FATAL(tensor.get_layout() == Layout::TILE, "{} only supports tiled layout.", op_name);
    TT_FATAL(tensor.get_dtype() == DataType::BFLOAT16, "{} only supports bfloat16.", op_name);
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE, "Operands to {} need to be on device!", op_name);
    TT_FATAL(
        tensor.buffer() != nullptr, "Operands to {} need to be allocated in buffers on device!", op_name);
}

inline void check_tensor(std::optional<Tensor> tensor, const std::string &op_name) {
    if (!tensor.has_value()) {
        return;
    }
    check_tensor(tensor.value(), op_name);
}

}  // namespace

////////////////////////////////////////////////////////////////////////////
//                         MorehSumBackward
////////////////////////////////////////////////////////////////////////////
void MorehSumBackward::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto &output_grad = input_tensors.at(0);
    const auto &input = input_tensors.at(1);
    auto &input_grad = output_tensors.at(0);

    // validate tensor
    check_tensor(output_grad, "moreh_sum_backward output_grad");
    check_tensor(input, "moreh_sum_backward input");
    check_tensor(input_grad, "moreh_sum_backward input_grad");

    // validate shape
    // keepdim=true
    const auto &input_shape = input.get_legacy_shape();
    auto input_shape_wo_padding = input_shape.without_padding();
    auto output_grad_shape_wo_padding = output_grad.get_legacy_shape().without_padding();
    for (int i = 0; i < input_shape.rank(); ++i) {
        TT_FATAL(input_shape_wo_padding[i] >= output_grad_shape_wo_padding[i]);
    }

    if (input_grad.has_value()) {
        const auto &input_grad_shape = input_grad.value().get_legacy_shape();
        TT_FATAL(input_shape == input_grad_shape, "both shape between input and input_grad should be the same");
    }
}

std::vector<Shape> MorehSumBackward::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    return {input_tensors.at(1).get_legacy_shape()};
}

std::vector<Tensor> MorehSumBackward::create_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(1).get_dtype(), Layout::TILE, this->input_grad_mem_config);
}

operation::ProgramWithCallbacks MorehSumBackward::create_program(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const {
    auto &output_grad = inputs.at(0);
    auto &input_grad = outputs.at(0);

    return moreh_sum_backward_impl(output_grad, input_grad, this->compute_kernel_config);
}

Tensor moreh_sum_backward(
    const Tensor &output_grad,
    const Tensor &input,
    std::vector<int64_t> &dims,
    const std::optional<const Tensor> input_grad,
    const MemoryConfig &input_grad_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({output_grad, input}))};
    auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config);

    operation::launch_op(
        [dims, input_grad_mem_config, kernel_config_val](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehSumBackward{.dims = dims, .input_grad_mem_config = input_grad_mem_config, .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {output_grad, input},
        output_tensors,
        {},
        {input_grad});

    return output_tensors.at(0);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
