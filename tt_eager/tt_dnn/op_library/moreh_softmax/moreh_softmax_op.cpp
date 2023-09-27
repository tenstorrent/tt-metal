// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
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

void MorehSoftmax::validate(const std::vector<Tensor> &input_tensors) const {
    TT_ASSERT(input_tensors.size() == 1, "Must have 1 input tensors");
    TT_ASSERT(this->dim == 2 || this->dim == 3, "Only dim 2 or 3 supported");
    auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr , "Operands to softmax need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_ASSERT(input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B);
}

std::vector<Shape> MorehSoftmax::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> MorehSoftmax::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehSoftmax::create_program(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    auto& input = input_tensors.at(0);
    auto& output = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy){
        case MorehSoftmaxParallelizationStrategy::SMALL_W:
            return {moreh_softmax_w_small(input, output, this->core_range)};
        case MorehSoftmaxParallelizationStrategy::SMALL_H:
            return {moreh_softmax_h_small(input, output, this->core_range)};
        case MorehSoftmaxParallelizationStrategy::LARGE_W:
            return {moreh_softmax_w_large(input, output, this->core_range)};
        case MorehSoftmaxParallelizationStrategy::LARGE_H:
            return {moreh_softmax_h_large(input, output, this->core_range)};
        // default:
        //     break;
    }

    return {moreh_softmax_h_large(input, output, this->core_range)};
}

MorehSoftmaxParallelizationStrategy MorehSoftmax::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);

    if (is_moreh_softmax_w_small_available(input) && this->dim == 3) {
        log_info(LogTest, "Small tensor algorithm selected");
        return MorehSoftmaxParallelizationStrategy::SMALL_W;
    }
    if (is_moreh_softmax_h_small_available(input) && this->dim == 2) {
        log_info(LogTest, "Small tensor algorithm selected");
        return MorehSoftmaxParallelizationStrategy::SMALL_H;
    }

    log_info(LogTest, "Large tensor algorithm selected");
    if (this->dim == 3) {
        return MorehSoftmaxParallelizationStrategy::LARGE_W;
    } else {
        return MorehSoftmaxParallelizationStrategy::LARGE_H;
    }
}

tt::stl::reflection::Attributes MorehSoftmax::attributes() const {
    return {
        {"dim", this->dim},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor moreh_softmax(const Tensor& input_tensor, uint32_t dim, const MemoryConfig& output_mem_config) {
    const CoreRange all_cores = {.start{0, 0}, .end = {11, 8}};

    return operation::run(MorehSoftmax{.dim=dim, .output_mem_config=output_mem_config, .core_range=all_cores}, {input_tensor}, {}).at(0);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
