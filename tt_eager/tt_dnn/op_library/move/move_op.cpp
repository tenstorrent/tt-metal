// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/move/move_op.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace move_op_utils {
using namespace tt::tt_metal;

bool can_deallocate(const Tensor &input_tensor) {
    return std::visit(
        [](auto &&storage) {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, DeviceStorage>) {
                return storage.buffer.use_count() == 1;
            } else {
                return false;
            }
        },
        input_tensor.get_storage());
}

} // namespace move_op_utils

namespace tt {

namespace tt_metal {

void Move::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
}

std::vector<Shape> Move::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> Move::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    return {input_tensors.at(1)};
}

operation::ProgramWithCallbacks Move::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->move_op_parallelization_strategy;
    switch (parallelization_strategy){
        case MoveOpParallelizationStrategy::MULTI_CORE:
            return move_multi_core(input_tensor, output_tensor);
        case MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP:
            return move_multi_core_with_overlap(input_tensor, output_tensor);
        case MoveOpParallelizationStrategy::MULTI_CORE_SHARDED:
            return move_multi_core_sharded(input_tensor, output_tensor);
        case MoveOpParallelizationStrategy::SINGLE_CORE:
        default:
            return move_single_core(input_tensor, output_tensor);
    }
}

MoveOpParallelizationStrategy Move::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    return this->move_op_parallelization_strategy;
}

tt::stl::reflection::Attributes Move::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
        {"move_op_parallelization_strategy", this->move_op_parallelization_strategy},
    };
}

}  // namespace tt_metal

}  // namespace tt
