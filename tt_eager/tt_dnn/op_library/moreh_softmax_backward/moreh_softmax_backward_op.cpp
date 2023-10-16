// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_softmax_backward/moreh_softmax_backward_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void MorehSoftmaxBackward::validate(const std::vector<Tensor> &input_tensors) const {
    TT_ASSERT(input_tensors.size() == 2, "Must have 2 input tensors");
    TT_ASSERT(this->dim >= 0 || this->dim <= 3, "Only dim [0,1,2,3] supported");
    auto& output_tensor = input_tensors.at(0);
    auto& output_grad_tensor = input_tensors.at(1);
    TT_ASSERT(output_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_ASSERT(output_grad_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_ASSERT(output_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_ASSERT(output_grad_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_ASSERT((output_tensor.layout() == Layout::TILE), "Output to softmax must be tilized");
    TT_ASSERT((output_grad_tensor.layout() == Layout::TILE), "Output_grad to softmax must be tilized");
    TT_ASSERT(output_tensor.dtype() == DataType::BFLOAT16 || output_tensor.dtype() == DataType::BFLOAT8_B);
    TT_ASSERT(output_grad_tensor.dtype() == DataType::BFLOAT16 || output_grad_tensor.dtype() == DataType::BFLOAT8_B);
}

std::vector<Shape> MorehSoftmaxBackward::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto& output_tensor = input_tensors.at(0);
    return {output_tensor.shape()};
}

std::vector<Tensor> MorehSoftmaxBackward::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehSoftmaxBackward::create_program(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    auto& output = input_tensors.at(0);
    auto& output_grad = input_tensors.at(1);
    auto& input_grad = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy){
        case MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W:
            return {moreh_softmax_backward_w_small(output, output_grad, input_grad, this->core_range)};
        case MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H:
            return {moreh_softmax_backward_h_small(output, output_grad, input_grad, this->core_range)};
        case MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W:
            return {moreh_softmax_backward_w_large(output, output_grad, input_grad, this->core_range)};
        case MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_H:
            return {moreh_softmax_backward_h_large(output, output_grad, input_grad, this->core_range)};
        case MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C:
            return {moreh_softmax_backward_c_large(output, output_grad, input_grad, this->dim, this->core_range)};
        // default:
        //     break;
    }

    return {moreh_softmax_backward_h_large(output, output_grad, input_grad, this->core_range)};
}

MorehSoftmaxBackwardOpParallelizationStrategy MorehSoftmaxBackward::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    auto& output = input_tensors.at(0);

    if (this->dim == 0 || this->dim == 1) {
        return MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C;
    }

    if (is_moreh_softmax_backward_w_small_available(output) && this->dim == 3) {
        log_info(LogTest, "Small tensor algorithm selected");
        return MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W;
    }
    if (is_moreh_softmax_backward_h_small_available(output) && this->dim == 2) {
        log_info(LogTest, "Small tensor algorithm selected");
        return MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H;
    }

    log_info(LogTest, "Large tensor algorithm selected");
    if (this->dim == 3) {
        return MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W;
    } else {
        return MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_H;
    }
}

tt::stl::reflection::Attributes MorehSoftmaxBackward::attributes() const {
    return {
        {"dim", this->dim},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor moreh_softmax_backward(const Tensor& output_tensor, const Tensor& output_grad_tensor, uint32_t dim, const MemoryConfig& output_mem_config) {
    auto device = output_grad_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores = {.start{0, 0}, .end = {grid_coord.x - 1, grid_coord.y-1}};

    return operation::run(MorehSoftmaxBackward{.dim=dim, .output_mem_config=output_mem_config, .core_range=all_cores}, {output_tensor, output_grad_tensor}, {}).at(0);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
