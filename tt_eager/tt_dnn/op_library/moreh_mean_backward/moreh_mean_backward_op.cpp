// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_mean_backward/moreh_mean_backward_op.hpp"

#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace tt {

using namespace constants;

namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         MorehMeanBackward
////////////////////////////////////////////////////////////////////////////
void MorehMeanBackward::validate(const std::vector<Tensor>& inputs) const {
    const auto& output_grad = inputs.at(0);
    const auto& input_grad = inputs.at(1);

    auto output_grad_shape = output_grad.shape();
    const auto& input_grad_shape = input_grad.shape();
    auto output_grad_shape_wo_padding = output_grad.shape().without_padding();
    const auto& input_grad_shape_wo_padding = input_grad.shape().without_padding();
}

operation::ProgramWithCallbacks MorehMeanBackward::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    auto& output_grad = inputs.at(0);
    auto& input_grad = inputs.at(1);

    return moreh_mean_backward_program(output_grad, input_grad);
}

std::vector<Tensor> MorehMeanBackward::create_output_tensors(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

std::vector<Shape> MorehMeanBackward::compute_output_shapes(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

tt::stl::reflection::Attributes MorehMeanBackward::attributes() const { return {}; }

Tensor moreh_mean_backward_(const Tensor& output_grad, const Tensor& input_grad) {
    operation::run(MorehMeanBackward{}, {output_grad, input_grad});
    return input_grad;
}

Tensor moreh_mean_backward(const Tensor& output_grad, const Tensor& input_grad) {
    return moreh_mean_backward_(output_grad, input_grad);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
