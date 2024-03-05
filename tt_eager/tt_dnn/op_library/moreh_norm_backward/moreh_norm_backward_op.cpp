// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_norm_backward/moreh_norm_backward_op.hpp"

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

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
}  // namespace

void MorehNormBackward::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);
    const auto &output_grad = input_tensors.at(2);
    const auto &input_grad = input_tensors.at(3);

    check_tensor(input, "moreh_norm_backward");
    check_tensor(output, "moreh_norm_backward");
    check_tensor(output_grad, "moreh_norm_backward");
    check_tensor(input_grad, "moreh_norm_backward");
}

std::vector<Shape> MorehNormBackward::compute_output_shapes(const std::vector<Tensor> &) const { return {}; }

std::vector<Tensor> MorehNormBackward::create_output_tensors(const std::vector<Tensor> &) const { return {}; }

operation::ProgramWithCallbacks MorehNormBackward::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);
    const auto &output_grad = input_tensors.at(2);
    const auto &input_grad = input_tensors.at(3);

    return moreh_norm_backward_(input, output, output_grad, this->p, input_grad);
}

[[maybe_unused]] Tensor moreh_norm_backward(
    const Tensor &input,
    const Tensor &output,
    const Tensor &output_grad,
    float p,
    const std::optional<std::reference_wrapper<const Tensor>> input_grad,
    const MemoryConfig &input_grad_mem_config) {
    if (input_grad.has_value()) {
        return moreh_norm_backward_impl(input, output, output_grad, p, input_grad->get());
    }

    // Make input_grad
    auto created_input_grad =
        create_device_tensor(input.get_legacy_shape(), input.get_dtype(), Layout::TILE, input.device(), input_grad_mem_config);
    return moreh_norm_backward_impl(input, output, output_grad, p, created_input_grad);
}

Tensor moreh_norm_backward_impl(
    const Tensor &input, const Tensor &output, const Tensor &output_grad, float p, const Tensor &input_grad) {
    operation::run(MorehNormBackward{.p = p}, {input, output, output_grad, input_grad});
    return input_grad;
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
