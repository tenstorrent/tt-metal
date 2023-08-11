#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


void RotaryEmbedding::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to rotary embedding need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operands to rotary embedding need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");
}

std::vector<Shape> RotaryEmbedding::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.shape();
    shape[-2] = this->seq_len;
    return {shape};
}

std::vector<Tensor> RotaryEmbedding::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = this->compute_output_shapes(input_tensors)[0];
    output_shape[-2] = round_up(output_shape[-2], TILE_HEIGHT);
    return {create_device_tensor(output_shape, input_tensor.dtype(), Layout::TILE, input_tensor.device(), this->output_mem_config)};
}

operation::ProgramWithCallbacks RotaryEmbedding::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& cos = input_tensors.at(1);
    const auto& sin = input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);

    switch (this->get_parallelization_strategy(input_tensors)) {
        case RotaryEmbeddingOpParallelizationStrategy::MULTI_CORE:
            return rotary_embedding_multi_core(input_tensor, cos, sin, output_tensor);
            break;
        case RotaryEmbeddingOpParallelizationStrategy::SINGLE_CORE:
        default: return rotary_embedding_single_core(input_tensor, cos, sin, output_tensor);
    }
}


RotaryEmbeddingOpParallelizationStrategy RotaryEmbedding::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    uint32_t num_rows = input_tensor.volume() / input_tensor.shape()[-1] / TILE_HEIGHT;
    if(num_rows > 1){
           return RotaryEmbeddingOpParallelizationStrategy::MULTI_CORE;
    }
    else{
       return RotaryEmbeddingOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes RotaryEmbedding::attributes() const {
    return {
        {"seq_len", this->seq_len},
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace tt_metal

}  // namespace tt
