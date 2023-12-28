// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt {

namespace tt_metal {

void NlpTM::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor.layout() == Layout::TILE);

    switch (this->nlp_tm_op_type) {
        case NlpTMOpType::CREATE_QKV_HEADS:
            TT_FATAL(input_shape[2] % TILE_HEIGHT == 0);
            TT_FATAL((input_shape == Shape({input_shape[0], 1, input_shape[2], 4672})), "Unsupported input shape");
            TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
            break;
        case NlpTMOpType::CONCAT_HEADS:
            if (input_tensor.is_sharded()) {
                TT_FATAL(input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
                auto shard_spec = input_tensor.shard_spec().value();
                TT_FATAL(shard_spec.shape[1] == input_tensor.shape()[-1]);
                TT_FATAL(shard_spec.shape[0] % input_tensor.shape()[-2] == 0);
                TT_FATAL(input_tensor.shape()[1] % (shard_spec.shape[0] / input_tensor.shape()[-2]) == 0);
                TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
            } else {
                TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
            }
            break;
        default:
            TT_FATAL(false, "Unknown nlp tm op in validate!");
    }
}

std::vector<Shape> NlpTM::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    std::vector<Shape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.shape();
    switch (this->nlp_tm_op_type) {
        case NlpTMOpType::CREATE_QKV_HEADS:
            output_shape_vec = {(Shape) {input_shape[0], 71, input_shape[2], 64}, (Shape) {input_shape[0], 1, input_shape[2], 64}, (Shape) {input_shape[0], 1, input_shape[2], 64}};
            break;
        case NlpTMOpType::CONCAT_HEADS:
            output_shape_vec = {(Shape) {input_shape[0], 1, input_shape[2], input_shape[1] * input_shape[3]}};
            break;
        default:
            TT_ASSERT(false, "Unknown nlp tm op in compute_output_shapes!");
    }
    return output_shape_vec;
}

std::vector<Tensor> NlpTM::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        if (this->nlp_tm_op_type == NlpTMOpType::CONCAT_HEADS) {
            ShardSpec shard_spec = input_tensor.shard_spec().value();
            uint32_t num_cores = shard_spec.num_cores();
            uint32_t heads_per_shard = shard_spec.shape[0] / input_tensor.shape()[-2];
            shard_spec.shape = {shard_spec.shape[0] / heads_per_shard, shard_spec.shape[1] * heads_per_shard};
            return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.dtype(), Layout::TILE, input_tensor.device(), this->output_mem_config, shard_spec)};
        } else {
            TT_ASSERT(false);
            return {};
        }
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks NlpTM::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    switch (this->nlp_tm_op_type) {
        case NlpTMOpType::CREATE_QKV_HEADS:
            return  multi_core_nlp_create_qkv_heads(input_tensor, output_tensors, compute_with_storage_grid_size);
        case NlpTMOpType::CONCAT_HEADS:
            return  multi_core_nlp_concat_heads(input_tensor, output_tensor, compute_with_storage_grid_size);
        default:
            TT_ASSERT(false, "Unknown nlp tm op in create_program!");
    }
    return {};
}

tt::stl::reflection::Attributes NlpTM::attributes() const {
    return {
        {"nlp_tm_op_type", this->nlp_tm_op_type},
        {"output_mem_config", this->output_mem_config},
    };
}

} // namespace tt_metal

} // namespace tt
