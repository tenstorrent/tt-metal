// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void MorehSoftmax::validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const
{
    // validate input tensor
    auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.get_layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_ASSERT(input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B);

    // validate parameters
    TT_ASSERT(this->dim >= 0 || this->dim <= 3, "Only dim [0,1,2,3] supported");

    if(output_tensors.empty() || !output_tensors.at(0).has_value()){
        // If the user decided to not use any optional output tensors, then this would be empty or would be a nullptr.
        return;
    }
    TT_ASSERT(input_tensors.size() == 1, "Must have 1 input tensors");
    TT_ASSERT(output_tensors.size() == 1, "Must have 1 output tensors");
}


std::vector<Shape> MorehSoftmax::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> MorehSoftmax::create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if(!output_tensors.empty() && output_tensors.at(0).has_value()){
        return {output_tensors.at(0).value()};
    }
    const auto& output_shape = input_tensors.at(0).get_legacy_shape();

    return {operation::generic_create_output_tensors(*this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config)};
}

operation::ProgramWithCallbacks MorehSoftmax::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    auto& input = input_tensors.at(0);
    auto& output = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy) {
        case MorehSoftmaxOpParallelizationStrategy::SMALL_W:
            return {moreh_softmax_w_small(input, output, this->core_range, this->op)};
        case MorehSoftmaxOpParallelizationStrategy::SMALL_H:
            return {moreh_softmax_h_small(input, output, this->core_range, this->op)};
        case MorehSoftmaxOpParallelizationStrategy::LARGE_W:
            return {moreh_softmax_w_large(input, output, this->core_range, this->op)};
        case MorehSoftmaxOpParallelizationStrategy::LARGE_H:
            return {moreh_softmax_h_large(input, output, this->core_range, this->op)};
        case MorehSoftmaxOpParallelizationStrategy::LARGE_C:
            return {moreh_softmax_c_large(input, output, this->dim, this->core_range, this->op)};
        case MorehSoftmaxOpParallelizationStrategy::NONE:
        default: break;
    }

    return {moreh_softmax_h_large(input, output, this->core_range, this->op)};
}

MorehSoftmaxOpParallelizationStrategy MorehSoftmax::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors.at(0);

    if (this->strategy == MorehSoftmaxOpParallelizationStrategy::NONE) {
        if (this->dim == 0 || this->dim == 1) {
            return MorehSoftmaxOpParallelizationStrategy::LARGE_C;
        }
        if (is_moreh_softmax_w_small_available(input) && this->dim == 3) {
            return MorehSoftmaxOpParallelizationStrategy::SMALL_W;
        }
        if (is_moreh_softmax_h_small_available(input) && this->dim == 2) {
            return MorehSoftmaxOpParallelizationStrategy::SMALL_H;
        }

        if (this->dim == 3) {
            return MorehSoftmaxOpParallelizationStrategy::LARGE_W;
        } else {
            return MorehSoftmaxOpParallelizationStrategy::LARGE_H;
        }
    }

    if (this->dim == 0 || this->dim == 1) {
        TT_ASSERT(
            this->strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_C,
            "Invalid parallelization strategy. large c is for dim 0, 1");
    }
    if (this->dim == 2) {
        TT_ASSERT(
            this->strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_H ||
                this->strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_H,
            fmt::format("Invalid parallelization strategy. {} is not for dim 2", this->strategy));

        if (this->strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_H) {
            TT_ASSERT(
                is_moreh_softmax_h_small_available(input),
                fmt::format("not enough circular buffer memory for {}", this->strategy));
        }
    }
    if (this->dim == 3) {
        TT_ASSERT(
            this->strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_W ||
                this->strategy == MorehSoftmaxOpParallelizationStrategy::LARGE_W,
            fmt::format("Invalid parallelization strategy. {} is not for dim 3", this->strategy));

        if (this->strategy == MorehSoftmaxOpParallelizationStrategy::SMALL_W) {
            TT_ASSERT(
                is_moreh_softmax_w_small_available(input),
                fmt::format("not enough circular buffer memory for {}", this->strategy));
        }
    }

    return this->strategy;
}


Tensor moreh_softmax(
    const Tensor &input_tensor,
    uint32_t dim,
    std::optional<Tensor> output_tensor,
    const MorehSoftmaxOpParallelizationStrategy strategy,
    const MemoryConfig &output_mem_config) {

    auto device = input_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    output_tensor = operation::run(
               MorehSoftmax{
                   .dim = dim,
                   .core_range = all_cores,
                   .op = MorehSoftmaxOp::SOFTMAX,
                   .strategy = strategy,
                   .output_mem_config = output_mem_config},
               {input_tensor},
               {},
               {output_tensor}).at(0);

    return output_tensor.value();
}

Tensor moreh_softmin(
    const Tensor &input_tensor,
    uint32_t dim,
    std::optional<Tensor> output_tensor,
    const MorehSoftmaxOpParallelizationStrategy strategy,
    const MemoryConfig &output_mem_config) {

    auto device = input_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    output_tensor = operation::run(
               MorehSoftmax{
                   .dim = dim,
                   .core_range = all_cores,
                   .op = MorehSoftmaxOp::SOFTMIN,
                   .strategy = strategy,
                   .output_mem_config = output_mem_config},
               {input_tensor},
               {},
               {output_tensor}).at(0);

    return output_tensor.value();
}

Tensor moreh_logsoftmax(
    const Tensor &input_tensor,
    uint32_t dim,
    std::optional<Tensor> output_tensor,
    const MorehSoftmaxOpParallelizationStrategy strategy,
    const MemoryConfig &output_mem_config) {

    auto device = input_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    output_tensor = operation::run(
        MorehSoftmax{
            .dim = dim,
            .core_range = all_cores,
            .op = MorehSoftmaxOp::LOGSOFTMAX,
            .strategy = strategy,
            .output_mem_config = output_mem_config},
        {input_tensor},
        {},
        {output_tensor}).at(0);

    return output_tensor.value();
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
