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
    TT_ASSERT(tensor.get_layout() == Layout::TILE, fmt::format("{} only supports tiled layout.", op_name));
    TT_ASSERT(tensor.get_dtype() == DataType::BFLOAT16, fmt::format("{} only supports bfloat16.", op_name));
    TT_ASSERT(
        tensor.storage_type() == StorageType::DEVICE, fmt::format("Operands to {} need to be on device!", op_name));
    TT_ASSERT(
        tensor.buffer() != nullptr, fmt::format("Operands to {} need to be allocated in buffers on device!", op_name));
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

    check_tensor(output_grad, "moreh_sum_backward output_grad");
    check_tensor(input, "moreh_sum_backward input");
    check_tensor(input_grad, "moreh_sum_backward input_grad");

    const auto& input_shape = input.get_legacy_shape();

    if (input_grad.has_value()) {
        const auto& input_grad_shape = input_grad.value().get_legacy_shape();
        TT_ASSERT(input_shape == input_grad_shape, "both shape between input and input_grad should be the same");
    }

    // TODO: add more asserts
}

std::vector<Shape> MorehSumBackward::compute_output_shapes(
    const std::vector<Tensor> &input_tensors) const {
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
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    auto& output_grad = inputs.at(0);
    auto& input_grad = outputs.at(0);

    return moreh_sum_backward_impl(output_grad, input_grad);
}

Tensor moreh_sum_backward(
    const Tensor &output_grad,
    const Tensor &input,
    std::vector<int64_t> &dims,
    const std::optional<const Tensor> input_grad,
    const MemoryConfig &input_grad_mem_config) {
    return operation::run(
               MorehSumBackward{.dims = dims, .input_grad_mem_config = std::move(input_grad_mem_config)},
               {output_grad, input},
               {},
               {input_grad})
        .at(0);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
