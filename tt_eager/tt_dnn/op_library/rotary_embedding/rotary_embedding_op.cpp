// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
    const auto& cos = input_tensors.at(1);
    const auto& sin = input_tensors.at(2);
    TT_ASSERT(input_tensors.size() == 3);
    auto ref_device = input_tensor.device();
    for (const auto& input : input_tensors) {
        TT_ASSERT(input.storage_type() == StorageType::DEVICE, "Operands to rotary embedding need to be on device!");
        TT_ASSERT(input.buffer() != nullptr, "Operands to rotary embedding need to be allocated in buffers on device!");
        TT_ASSERT(input.device() == ref_device, "Operands to rotary embedding need to be on same device!");
        TT_ASSERT((input.layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");
    }

    TT_ASSERT(input_tensor.shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
    uint32_t seq_len = input_tensor.shape()[-2];
    uint32_t B = input_tensor.shape()[0];
    uint32_t X = input_tensor.shape()[-1];
    TT_ASSERT(cos.shape() == sin.shape(), "Cos and Sin dims must match");
    TT_ASSERT(cos.shape()[0] == 1 && cos.shape()[1] == 1 && cos.shape()[-1] == X, "Cos dims must match input dims");
    if (this->token_idx.has_value()) {
        TT_ASSERT(cos.shape()[-2] >= token_idx, "Cos dims must match input dims");
    } else {
        TT_ASSERT(cos.shape()[-2] >= seq_len, "Cos dims must match input dims");
    }
}

std::vector<Shape> RotaryEmbedding::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.shape();
    if (!this->token_idx.has_value()) {
        shape[-2] = this->seq_len;
    }
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
            return rotary_embedding_multi_core(input_tensor, cos, sin, output_tensor, this->token_idx);
            break;
        case RotaryEmbeddingOpParallelizationStrategy::SINGLE_CORE:
        default: return rotary_embedding_single_core(input_tensor, cos, sin, output_tensor, this->token_idx);
    }
}


RotaryEmbeddingOpParallelizationStrategy RotaryEmbedding::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    uint32_t num_rows = input_tensor.volume() / input_tensor.shape()[-1] / TILE_HEIGHT;
    if(num_rows > 1) {
           return RotaryEmbeddingOpParallelizationStrategy::MULTI_CORE;
    }
    else {
       return RotaryEmbeddingOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes RotaryEmbedding::attributes() const {
    return {
        {"seq_len", this->seq_len},
        {"token_idx", this->token_idx},
        {"output_mem_config", this->output_mem_config},
    };
}

const operation::Hash RotaryEmbedding::compute_program_hash(
    const std::vector<Tensor> &input_tensors) const {
    return operation::hash_operation<RotaryEmbedding>(this->seq_len, this->output_mem_config, input_tensors);
}

}  // namespace tt_metal

}  // namespace tt
