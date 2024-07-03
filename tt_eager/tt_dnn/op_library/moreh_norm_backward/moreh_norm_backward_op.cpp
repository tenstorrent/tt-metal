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

void MorehNormBackward::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);
    const auto &output_grad = input_tensors.at(2);
    const auto &input_grad = input_tensors.at(3);

    check_tensor(input, "moreh_norm_backward", "input");
    check_tensor(output, "moreh_norm_backward", "output");
    check_tensor(output_grad, "moreh_norm_backward", "output_grad");
    check_tensor(input_grad, "moreh_norm_backward", "input_grad");
}

std::vector<Shape> MorehNormBackward::compute_output_shapes(const std::vector<Tensor> &) const { return {}; }

std::vector<Tensor> MorehNormBackward::create_output_tensors(const std::vector<Tensor> &input_tensors) const { return {input_tensors.at(3)}; }

operation::ProgramWithCallbacks MorehNormBackward::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);
    const auto &output_grad = input_tensors.at(2);
    const auto &input_grad = output_tensors.at(0);

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

// TODO: move input_grad to optional_output_tensors
Tensor moreh_norm_backward_impl(
    const Tensor &input, const Tensor &output, const Tensor &output_grad, float p, const Tensor &input_grad) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input, output, output_grad, input_grad}))};
    operation::launch_op(
        [p](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehNormBackward{.p = p}, input_tensors, optional_input_tensors, optional_output_tensors);
        },
        {input, output, output_grad, input_grad},
        output_tensors,
        {},
        {});

    return output_tensors.at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
