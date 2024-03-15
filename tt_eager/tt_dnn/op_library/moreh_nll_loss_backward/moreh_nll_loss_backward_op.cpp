// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward_op.hpp"
#include "tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void MorehNllLossBackward::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 4, "Must have 2 input tensors");
    TT_ASSERT(optional_input_tensors.size() == 2, "Must have 1 optional input tensors");

    auto& input_tensor = input_tensors.at(0);
    auto& target_tensor = input_tensors.at(1);
    auto& output_grad_tensor = input_tensors.at(2);
    auto& input_grad_tensor = input_tensors.at(3);
    auto& weight_tensor = optional_input_tensors.at(0);
    auto& divisor_tensor = optional_input_tensors.at(1);

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.get_layout() == Layout::TILE), "intput_tensor to nll_loss must be tilized");
    TT_ASSERT(input_tensor.get_dtype() == DataType::BFLOAT16);

    TT_ASSERT(target_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_ASSERT(target_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_ASSERT(target_tensor.get_dtype() == DataType::UINT32);

    TT_ASSERT(output_grad_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_ASSERT(output_grad_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((output_grad_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_ASSERT(output_grad_tensor.get_dtype() == DataType::BFLOAT16);

    TT_ASSERT(input_grad_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_ASSERT(input_grad_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((input_grad_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_ASSERT(input_grad_tensor.get_dtype() == DataType::BFLOAT16);

    if (weight_tensor.has_value()) {
        TT_ASSERT(weight_tensor.value().storage_type() == StorageType::DEVICE, "weight_tensor to nll_loss need to be on device!");
        TT_ASSERT(weight_tensor.value().buffer() != nullptr, "weight_tensor to nll_loss need to be allocated in buffers on device!");
        TT_ASSERT((weight_tensor.value().get_layout() == Layout::ROW_MAJOR), "weight_tensor to nll_loss must be in row major layout");
        TT_ASSERT(weight_tensor.value().get_dtype() == DataType::BFLOAT16);
    }

    if (divisor_tensor.has_value()) {
        TT_ASSERT(divisor_tensor.value().storage_type() == StorageType::DEVICE, "divisor_tensor to nll_loss need to be on device!");
        TT_ASSERT(divisor_tensor.value().buffer() != nullptr, "divisor_tensor to nll_loss need to be allocated in buffers on device!");
        TT_ASSERT((divisor_tensor.value().get_layout() == Layout::TILE), "divisor_tensor to nll_loss must be tilized");
        TT_ASSERT(divisor_tensor.value().get_dtype() == DataType::BFLOAT16);
    }
}

std::vector<Shape> MorehNllLossBackward::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {};
}

std::vector<Tensor> MorehNllLossBackward::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return {};
}

operation::ProgramWithCallbacks MorehNllLossBackward::create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const {
    auto& input = input_tensors.at(0);
    auto& target = input_tensors.at(1);
    auto& output_grad = input_tensors.at(2);
    auto& input_grad = input_tensors.at(3);
    auto& weight = optional_input_tensors.at(0);
    auto& divisor = optional_input_tensors.at(1);

    return {moreh_nll_loss_backward_impl(input, target, weight, divisor, output_grad, input_grad, this->ignore_index, this->reduction_mean, this->core_range)};
}

Tensor moreh_nll_loss_backward_(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const Tensor& output_grad_tensor,
    const Tensor& input_grad_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig& output_mem_config) {

    auto device = input_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    operation::run(
               MorehNllLossBackward{
                   .ignore_index = ignore_index,
                   .reduction_mean = reduction_mean,
                   .output_mem_config = output_mem_config,
                   .core_range = all_cores},
               {input_tensor, target_tensor, output_grad_tensor, input_grad_tensor},
               {weight_tensor, divisor_tensor});
    return input_grad_tensor;
}

Tensor moreh_nll_loss_backward(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const Tensor& output_grad_tensor,
    const Tensor& input_grad_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig& output_mem_config) {

    return moreh_nll_loss_backward_(input_tensor, target_tensor, weight_tensor, divisor_tensor, output_grad_tensor, input_grad_tensor, ignore_index, reduction_mean, output_mem_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
