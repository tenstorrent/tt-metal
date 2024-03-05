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

    auto output_grad_shape_wo_padding = output_grad.get_legacy_shape().without_padding();
    const auto& input_grad_shape_wo_padding = input_grad.get_legacy_shape().without_padding();

    for (int i = 0; i < output_grad_shape_wo_padding.rank(); ++i) {
        const auto output_grad_dim = output_grad_shape_wo_padding[i];
        const auto input_grad_dim = input_grad_shape_wo_padding[i];
        if (output_grad_dim == input_grad_dim) {
            continue;
        }
        TT_ASSERT(output_grad_dim == 1);
    }

    TT_ASSERT(
        (output_grad.get_layout() == Layout::TILE && input_grad.get_layout() == Layout::TILE),
        "Tensors must be tilized");
    TT_ASSERT(
        output_grad.get_dtype() == DataType::BFLOAT16 || output_grad.get_dtype() == DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_ASSERT(
        input_grad.get_dtype() == DataType::BFLOAT16 || input_grad.get_dtype() == DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_ASSERT(
        output_grad.get_dtype() == input_grad.get_dtype(), "Unsupported data format");
    TT_ASSERT(
        output_grad.storage_type() == StorageType::DEVICE and input_grad.storage_type() == StorageType::DEVICE,
        "Operands to mean backward need to be on device!");
    TT_ASSERT(output_grad.device() == input_grad.device(), "Operands to mean backward need to be on the same device!");
    TT_ASSERT(
        output_grad.buffer() != nullptr and input_grad.buffer() != nullptr,
        "Operands to mean backward need to be allocated in buffers on device!");
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
