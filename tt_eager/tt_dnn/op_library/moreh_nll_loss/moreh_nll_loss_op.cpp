// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_op.hpp"
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

void MorehNllLossStep1::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 2, "Must have 2 input tensors");
    TT_ASSERT(optional_input_tensors.size() == 1, "Must have 1 optional input tensors");

    auto& input_tensor = input_tensors.at(0);
    auto& target_tensor = input_tensors.at(1);
    auto& weight_tensor = optional_input_tensors.at(0);

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.get_layout() == Layout::TILE), "intput_tensor to nll_loss must be tilized");
    TT_ASSERT(input_tensor.get_dtype() == DataType::BFLOAT16);

    TT_ASSERT(target_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_ASSERT(target_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_ASSERT(target_tensor.get_dtype() == DataType::UINT32);

    if (weight_tensor.has_value()) {
        TT_ASSERT(weight_tensor.value().storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
        TT_ASSERT(weight_tensor.value().buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
        TT_ASSERT((weight_tensor.value().get_layout() == Layout::ROW_MAJOR), "target_tensor to nll_loss must be tilized");
        TT_ASSERT(weight_tensor.value().get_dtype() == DataType::BFLOAT16);
    }
}

std::vector<Shape> MorehNllLossStep1::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto input_shape = input_tensor.get_legacy_shape();
    const Shape output_shape = {input_shape[0], 1, input_shape[2], input_shape[3]};
    return {output_shape};
}

std::vector<Tensor> MorehNllLossStep1::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehNllLossStep1::create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const {
    auto& input = input_tensors.at(0);
    auto& target = input_tensors.at(1);
    auto& weight = optional_input_tensors.at(0);
    auto& output = output_tensors.at(0);

    return {moreh_nll_loss_step1_impl(input, target, weight, output, this->ignore_index, this->reduction_mean, this->core_range)};
}


void MorehNllLossStep2::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 2, "Must have 2 input tensors");
    TT_ASSERT(optional_input_tensors.size() == 2, "Must have 2 optional input tensors");

    auto& input_tensor = input_tensors.at(0);
    auto& target_tensor = input_tensors.at(1);
    auto& weight_tensor = optional_input_tensors.at(0);
    auto& divisor_tensor = optional_input_tensors.at(1);

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "intput_tensor to nll_loss need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "intput_tensor to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.get_layout() == Layout::TILE), "intput_tensor to nll_loss must be tilized");
    TT_ASSERT(input_tensor.get_dtype() == DataType::BFLOAT16);

    TT_ASSERT(target_tensor.storage_type() == StorageType::DEVICE, "target_tensor to nll_loss need to be on device!");
    TT_ASSERT(target_tensor.buffer() != nullptr, "target_tensor to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_ASSERT(target_tensor.get_dtype() == DataType::UINT32);

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

std::vector<Shape> MorehNllLossStep2::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto input_shape = input_tensor.get_legacy_shape();
    const Shape output_shape = {input_shape[0], 1, input_shape[2], input_shape[3]};
    return {output_shape};
}

std::vector<Tensor> MorehNllLossStep2::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehNllLossStep2::create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const {
    auto& input = input_tensors.at(0);
    auto& target = input_tensors.at(1);
    auto& weight = optional_input_tensors.at(0);
    auto& divisor = optional_input_tensors.at(1);
    auto& output = output_tensors.at(0);

    return {moreh_nll_loss_step2_impl(input, target, weight, divisor, output, this->ignore_index, this->reduction_mean, this->core_range)};
}

Tensor moreh_nll_loss_step1(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig& output_mem_config) {

    auto device = input_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    return operation::run(
               MorehNllLossStep1{
                   .ignore_index = ignore_index,
                   .reduction_mean = reduction_mean,
                   .output_mem_config = output_mem_config,
                   .core_range = all_cores},
               {input_tensor, target_tensor},
               {weight_tensor})
        .at(0);
}


Tensor moreh_nll_loss_step2(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig& output_mem_config) {

    auto device = input_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    return operation::run(
               MorehNllLossStep2{
                   .ignore_index = ignore_index,
                   .reduction_mean = reduction_mean,
                   .output_mem_config = output_mem_config,
                   .core_range = all_cores},
               {input_tensor, target_tensor},
               {weight_tensor, divisor_tensor})
        .at(0);
}

Tensor moreh_nll_loss(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const Tensor& output_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig& output_mem_config) {

    if (reduction_mean) {
        TT_ASSERT(divisor_tensor.has_value());

        const Tensor& step1_result = moreh_nll_loss_step1(input_tensor, target_tensor, weight_tensor, ignore_index, reduction_mean, output_mem_config);
        std::vector<int64_t> dims_step1;
        moreh_sum(step1_result, divisor_tensor.value(), dims_step1);

        const Tensor& step2_result = moreh_nll_loss_step2(input_tensor, target_tensor, weight_tensor, divisor_tensor, ignore_index, reduction_mean, output_mem_config);
        std::vector<int64_t> dims_step2;

        moreh_sum(step2_result, output_tensor, dims_step2);
        return output_tensor;
    } else {
        const Tensor& step2_result = moreh_nll_loss_step2(input_tensor, target_tensor, weight_tensor, std::nullopt, ignore_index, reduction_mean, output_mem_config);

        std::vector<int64_t> dims;
        moreh_sum(step2_result, output_tensor, dims);
        return output_tensor;
    }
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
