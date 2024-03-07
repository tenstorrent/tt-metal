// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt {

namespace tt_metal {

// Hard-coded for Falcon7B
void NlpCreateHeadsFalcon7B::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE);

    TT_FATAL(input_shape[2] % TILE_HEIGHT == 0);
    TT_FATAL((input_shape == Shape({input_shape[0], 1, input_shape[2], 4672})), "Unsupported input shape");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
}

std::vector<Shape> NlpCreateHeadsFalcon7B::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    std::vector<Shape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    output_shape_vec = {(Shape) {input_shape[0], 71, input_shape[2], 64}, (Shape) {input_shape[0], 1, input_shape[2], 64}, (Shape) {input_shape[0], 1, input_shape[2], 64}};
    return output_shape_vec;
}

std::vector<Tensor> NlpCreateHeadsFalcon7B::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        TT_ASSERT(false);
        return {};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks NlpCreateHeadsFalcon7B::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    return  multi_core_nlp_create_qkv_heads_falcon7b(input_tensor, output_tensors, compute_with_storage_grid_size);
}

tt::stl::reflection::Attributes NlpCreateHeadsFalcon7B::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}


// Generic NLP CreateHeads op
void NlpCreateHeads::validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    // NOTE: Checks for head_dim and shape[3] is done in nlp_create_qkv_heads because it's needed to infer head_dim
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE);

    TT_FATAL(input_shape[2] % TILE_HEIGHT == 0, "Unsupported input shape");
    TT_FATAL(input_shape[1] == 1, "Unsupported input shape");
    if (input_tensor.is_sharded()) {
        TT_FATAL(input_tensor.shard_spec().value().shape[0] == input_tensor.volume() / input_tensor.get_legacy_shape()[-1]);
        TT_FATAL(this->output_mem_config.is_sharded() && this->output_mem_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
        TT_FATAL(input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
        auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
        uint32_t num_cores = core_grid.x * core_grid.y;
        // 1 Head Per Core Max for now
        TT_FATAL(this->num_q_heads <= num_cores);
        TT_FATAL(this->num_kv_heads <= num_cores);
        TT_FATAL(this->num_q_heads >= this->num_kv_heads);
        TT_FATAL(this->num_q_heads % input_tensor.shard_spec().value().num_cores() == 0);
        if (optional_input_tensors.at(0).has_value()) {
            TT_FATAL(optional_input_tensors.at(0).value().is_sharded());
            TT_FATAL(input_tensor.shard_spec().value().grid == optional_input_tensors.at(0).value().shard_spec().value().grid);
            TT_FATAL(input_tensor.shard_spec().value().orientation == optional_input_tensors.at(0).value().shard_spec().value().orientation);
            TT_FATAL(input_tensor.shard_spec().value().shape[1] == (this->num_q_heads / this->num_kv_heads) * this->head_dim);
        } else {
            TT_FATAL(this->num_kv_heads % input_tensor.shard_spec().value().num_cores() == 0);
            TT_FATAL(input_tensor.shard_spec().value().shape[1] == (this->num_q_heads / this->num_kv_heads + 2) * this->head_dim);
        }
        TT_FATAL(!this->transpose_k_heads);
    } else {
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }

    if (optional_input_tensors.at(0).has_value()) {
        const auto& input_tensor_kv = optional_input_tensors.at(0).value();
        const auto input_shape_kv = input_tensor_kv.get_legacy_shape();

        TT_FATAL(input_tensor_kv.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
        TT_FATAL(input_tensor_kv.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
        TT_FATAL(input_tensor_kv.get_dtype() == input_tensor.get_dtype(), "KV tensor dtype must be same as Q tensor dtype!");
        TT_FATAL(input_tensor_kv.get_layout() == Layout::TILE);

        TT_FATAL(input_shape_kv[0] == input_shape[0], "KV tensor batch dim must be same as Q tensor batch!");
        TT_FATAL(input_shape_kv[1] == 1, "Unsupported input shape");
        TT_FATAL(input_shape_kv[2] == input_shape[2], "KV tensor seq_len dim must be same as Q tensor seq_len!");
        if (input_tensor_kv.is_sharded()) {
            TT_FATAL(input_tensor.is_sharded());
            TT_FATAL(input_tensor_kv.shard_spec().value().shape[0] == input_tensor_kv.volume() / input_tensor_kv.get_legacy_shape()[-1]);
            TT_FATAL(input_tensor_kv.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
            TT_FATAL(input_tensor_kv.shard_spec().value().shape[1] == 2 * this->head_dim);
            TT_FATAL(this->num_kv_heads % input_tensor_kv.shard_spec().value().num_cores() == 0);
        }
    }
}

std::vector<Shape> NlpCreateHeads::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    std::vector<Shape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    auto sequence_length = input_shape[2];
    auto head_dim = this->head_dim;
    if (sequence_length % TILE_HEIGHT != 0) {
        sequence_length = (sequence_length / TILE_HEIGHT + 1) * TILE_HEIGHT;
    }
    if (head_dim % TILE_WIDTH != 0) {
        head_dim = (head_dim / TILE_WIDTH + 1) * TILE_WIDTH;
    }

    const Shape q_output_shape = {input_shape[0], this->num_q_heads, sequence_length, head_dim};
    const Shape v_output_shape = {input_shape[0], this->num_kv_heads, sequence_length, head_dim};
    const Shape k_output_shape = this->transpose_k_heads
                                     ? (Shape){input_shape[0], this->num_kv_heads, head_dim, sequence_length}
                                     : v_output_shape;
    output_shape_vec = {q_output_shape, k_output_shape, v_output_shape};

    return output_shape_vec;
}

std::vector<Tensor> NlpCreateHeads::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
        auto q_shard_grid = num_cores_to_corerange_set(this->num_q_heads, core_grid, true);
        ShardSpec q_shard_spec{q_shard_grid, {TILE_HEIGHT, this->head_dim}};
        auto q_mem_config = this->output_mem_config;
        q_mem_config.shard_spec = q_shard_spec;
        auto kv_shard_grid = num_cores_to_corerange_set(this->num_kv_heads, core_grid, true);
        ShardSpec kv_shard_spec{kv_shard_grid, {TILE_HEIGHT, this->head_dim}};
        auto kv_mem_config = this->output_mem_config;
        kv_mem_config.shard_spec = kv_shard_spec;
        auto output_shapes = this->compute_output_shapes(input_tensors);
        return {
            create_sharded_device_tensor(output_shapes[0], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), q_mem_config),
            create_sharded_device_tensor(output_shapes[1], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), kv_mem_config),
            create_sharded_device_tensor(output_shapes[2], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), kv_mem_config)
        };

    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks NlpCreateHeads::create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_tensor_kv = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    if (input_tensor.is_sharded()) {
        return  multi_core_nlp_create_qkv_heads_sharded(input_tensor, input_tensor_kv, this->num_q_heads, this->num_kv_heads, this->head_dim, this->transpose_k_heads, output_tensors, compute_with_storage_grid_size);
    } else {
        return  multi_core_nlp_create_qkv_heads(input_tensor, input_tensor_kv, this->num_q_heads, this->num_kv_heads, this->head_dim, this->transpose_k_heads, output_tensors, compute_with_storage_grid_size);
    }
}

tt::stl::reflection::Attributes NlpCreateHeads::attributes() const {
    return {
        {"num_q_heads", this->num_q_heads},
        {"num_kv_heads", this->num_kv_heads},
        {"transpose_k_heads", this->transpose_k_heads},
        {"output_mem_config", this->output_mem_config},
    };
}


// Generic NLP ConcatHeads op
void NlpConcatHeads::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE);

    if (input_tensor.is_sharded()) {
        TT_FATAL(input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
        auto shard_spec = input_tensor.shard_spec().value();
        TT_FATAL(shard_spec.shape[1] == input_tensor.get_legacy_shape()[-1]);
        TT_FATAL(shard_spec.shape[0] % input_tensor.get_legacy_shape()[-2] == 0);
        TT_FATAL(input_tensor.get_legacy_shape()[1] % (shard_spec.shape[0] / input_tensor.get_legacy_shape()[-2]) == 0);
        TT_FATAL(this->output_mem_config.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED);
    } else {
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<Shape> NlpConcatHeads::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    std::vector<Shape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    output_shape_vec = {(Shape) {input_shape[0], 1, input_shape[2], input_shape[1] * input_shape[3]}};

    auto num_heads = input_shape[1];
    auto sequence_length = input_shape[2];
    auto head_dim = input_shape[3];

    auto hidden_dim = num_heads * head_dim;

    const Shape output_shape = {input_shape[0], 1, sequence_length, hidden_dim};
    return {output_shape};
}

std::vector<Tensor> NlpConcatHeads::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        ShardSpec shard_spec = input_tensor.shard_spec().value();
        uint32_t num_cores = shard_spec.num_cores();
        uint32_t heads_per_shard = shard_spec.shape[0] / input_tensor.get_legacy_shape()[-2];
        shard_spec.shape = {shard_spec.shape[0] / heads_per_shard, shard_spec.shape[1] * heads_per_shard};
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = shard_spec;
        return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config)};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks NlpConcatHeads::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    return  multi_core_nlp_concat_heads(input_tensor, output_tensor, compute_with_storage_grid_size);
}

tt::stl::reflection::Attributes NlpConcatHeads::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

} // namespace tt_metal

} // namespace tt
