#include "tt_dnn/op_library/move/move_op.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"


namespace tt {

namespace tt_metal {

void Move::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT((input_tensor_a.layout() == Layout::TILE), "Inputs to move must be tilized");
}

std::vector<Shape> Move::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
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
            break;
        case MoveOpParallelizationStrategy::SINGLE_CORE:
        default:
            return move_single_core(input_tensor, output_tensor);
    }
}


tt::stl::reflection::Attributes Move::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
        {"move_op_parallelization_strategy", this->move_op_parallelization_strategy},
    };
}

}  // namespace tt_metal

}  // namespace tt
