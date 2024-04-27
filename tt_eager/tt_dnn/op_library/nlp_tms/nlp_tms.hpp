// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"
#include "ttnn/operations/core.hpp"

namespace tt {

namespace tt_metal {

// input_tensor - qkv tensor if kv_tensor is nullopt, q tensor if kv_tensor is populated
// expectation for both interleaved and sharded implementation is that each q, k and v vector are concatenated across the last dimension (|q_i k_i v_i| is each row of the tensor)

// operation::ProgramWithCallbacks multi_core_create_qkv_heads_interleaved(const Tensor &input_tensor_qkv, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, const bool transpose_k_heads, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_create_qkv_heads_sharded(const Tensor &input_tensor_qkv, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, const bool transpose_k_heads, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_create_q_and_kv_heads_sharded(const Tensor &input_tensor_q, const Tensor &input_tensor_kv, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, const bool transpose_k_heads, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size);

struct CreateQKVHeads {
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    bool transpose_k_heads;
    MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline std::tuple<Tensor, Tensor, Tensor> create_qkv_heads(const Tensor &input_tensor, const uint32_t num_q_heads, const std::optional<uint32_t> num_kv_heads, const bool transpose_k_heads, const MemoryConfig& output_mem_config) {
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_q_heads);
    TT_FATAL(input_tensor.get_legacy_shape()[3] % (num_q_heads + (2 * num_kv_heads_val)) == 0, fmt::format("Flattened hidden dimension {} must be a multiple of the combined Q {}, K {} and V {} heads", input_tensor.get_legacy_shape()[3], num_q_heads, num_kv_heads_val, num_kv_heads_val));
    const uint32_t head_dim = input_tensor.get_legacy_shape()[3] / (num_q_heads + (2 * num_kv_heads_val));
    auto output_tensors = operation::run(CreateQKVHeads{num_q_heads, num_kv_heads_val, head_dim, transpose_k_heads, output_mem_config}, {input_tensor});
    return {output_tensors.at(0), output_tensors.at(1), output_tensors.at(2)};
}

struct CreateQKVHeadsSeparateTensors {
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    bool transpose_k_heads;
    MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline std::tuple<Tensor, Tensor, Tensor> create_qkv_heads_from_separate_tensors(const Tensor &input_tensor_q, const Tensor &input_tensor_kv, const uint32_t num_q_heads, const std::optional<uint32_t> num_kv_heads, const bool transpose_k_heads, const MemoryConfig& output_mem_config) {
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_q_heads);
    TT_FATAL(input_tensor_q.get_legacy_shape()[3] % num_q_heads == 0, fmt::format("Flattened Q hidden dimension {} must be a multiple of Q heads {}", input_tensor_q.get_legacy_shape()[3], num_q_heads));
    const uint32_t head_dim = input_tensor_q.get_legacy_shape()[3] / (num_q_heads);
    auto output_tensors = operation::run(CreateQKVHeadsSeparateTensors{num_q_heads, num_kv_heads_val, head_dim, transpose_k_heads, output_mem_config}, {input_tensor_q, input_tensor_kv});
    return {output_tensors.at(0), output_tensors.at(1), output_tensors.at(2)};
}

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_falcon7b(const Tensor &input_tensor_a, std::vector<Tensor> &output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_sharded(const Tensor &input_tensor, std::optional<const Tensor> input_tensor_kv, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, const bool transpose_k_heads, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads(const Tensor &input_tensor, std::optional<const Tensor> input_tensor_kv, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, const bool transpose_k_heads, std::vector<Tensor> &output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_nlp_concat_heads(const Tensor &input_tensor_a, Tensor &output, CoreCoord compute_with_storage_grid_size);

struct NlpCreateHeadsFalcon7B {
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

struct NlpCreateHeads {
    const uint32_t num_q_heads;
    const uint32_t num_kv_heads;
    const uint32_t head_dim;
    const bool transpose_k_heads;
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

struct NlpConcatHeads {
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline std::vector<Tensor> nlp_create_qkv_heads_falcon7b(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
  // TODO: hard-coded for falcon-7b; can delete if we switch to the more generic one (but perf may be worse)
  std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a})), Tensor(operation::get_workers_for_op_output({input_tensor_a})), Tensor(operation::get_workers_for_op_output({input_tensor_a}))};
  operation::launch_op(
    [mem_config] (std::vector<Tensor> input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) mutable -> std::vector<Tensor> {
        return operation::run(NlpCreateHeadsFalcon7B{mem_config}, input_tensors);
    }, {input_tensor_a}, output_tensors);
    return output_tensors;
}
inline std::vector<Tensor> nlp_create_qkv_heads(
    const Tensor &input_tensor, std::optional<const Tensor> input_tensor_kv,
    const uint32_t num_heads, std::optional<const uint32_t> num_kv_heads,
    const bool transpose_k_heads,
    const MemoryConfig& mem_config
) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}, {input_tensor_kv})),
                                          Tensor(operation::get_workers_for_op_output({input_tensor}, {input_tensor_kv})),
                                          Tensor(operation::get_workers_for_op_output({input_tensor}, {input_tensor_kv}))};
    operation::launch_op(
        [num_heads, num_kv_heads, transpose_k_heads, mem_config] (std::vector<Tensor> input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) mutable -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
            auto& input_tensor_kv = optional_input_tensors.at(0);
            const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_heads);
            // Infer head_dim
            uint32_t head_dim;
            if (input_tensor_kv.has_value()) {
                TT_FATAL(input_tensor.get_legacy_shape()[3] % num_heads == 0, "Unsupported input shape");
                TT_FATAL(input_tensor_kv.value().get_legacy_shape()[3] % (2 * num_kv_heads_val) == 0, "Unsupported input shape");
                head_dim = input_tensor.get_legacy_shape()[3] / num_heads;
                TT_FATAL(input_tensor_kv.value().get_legacy_shape()[3] / (2 * num_kv_heads_val) == head_dim, "Head dims must be the same for Q and K, V");
            } else {
                TT_FATAL(input_tensor.get_legacy_shape()[3] % (num_heads + 2 * num_kv_heads_val) == 0, "Unsupported input shape");
                head_dim = input_tensor.get_legacy_shape()[3] / (num_heads + 2 * num_kv_heads_val);
            }
            return operation::run(NlpCreateHeads{num_heads, num_kv_heads_val, head_dim, transpose_k_heads, mem_config}, {input_tensor}, {input_tensor_kv});
        }, {input_tensor}, output_tensors, {input_tensor_kv});
    return output_tensors;
}
inline Tensor nlp_concat_heads(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a}))};
    operation::launch_op(
        [mem_config] (std::vector<Tensor> input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) mutable -> std::vector<Tensor> {
            return operation::run(NlpConcatHeads{mem_config}, input_tensors);
        }, {input_tensor_a}, output_tensors);
    return output_tensors.at(0);
}

}  // namespace tt_metal

}  // namespace tt

namespace ttnn {
namespace operations {
namespace transformer {

struct ConcatenateHeads : public NlpConcatHeads {
    static inline const std::vector<TensorSchema> input_schemas{
        ttnn::TensorSchema{4, 4, {ttnn::bfloat16, ttnn::bfloat8_b}, {ttnn::TILE_LAYOUT}, true, false, false}};

    void validate(const std::vector<Tensor>& input_tensors) const {
        const auto& input_tensor = input_tensors.at(0);
        const auto head_size = input_tensor.get_shape()[-1];
        const auto padded_head_size = input_tensor.get_legacy_shape()[-1];
        TT_FATAL(
            head_size % ttnn::types::TILE_SIZE == 0,
            fmt::format(
                "Head size must be a multiple of {} but was found to be {}! Update matmul that uses the output of this "
                "operation to have the "
                "padding in the weights!",
                ttnn::types::TILE_SIZE,
                head_size));
        TT_FATAL(padded_head_size - head_size == 0, "Head size cannot have tile padding!");

        NlpConcatHeads::validate(input_tensors);
    }

    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
        std::vector<tt::tt_metal::Shape> output_shape_vec;
        const auto& input_tensor = input_tensors.at(0);
        const ttnn::types::Shape input_shape = input_tensor.get_shape();
        const ttnn::types::Shape padded_input_shape = input_shape.with_tile_padding();

        auto batch_size = input_shape[0];
        auto num_heads = input_shape[1];
        auto sequence_size = input_shape[2];
        auto padded_sequence_size = padded_input_shape[2];
        auto head_size = input_shape[3];
        auto padded_head_size = padded_input_shape[3];

        std::array<uint32_t, 3> intended_output_shape = {batch_size, sequence_size, num_heads * head_size};
        std::array<uint32_t, 3> padded_output_shape = {batch_size, padded_sequence_size, num_heads * padded_head_size};
        return {ttnn::types::Shape(intended_output_shape, padded_output_shape).value()};
    }

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const {
        const auto& input_tensor = input_tensors.at(0);
        if (this->output_mem_config.is_sharded()) {
            ShardSpec shard_spec = input_tensor.shard_spec().value();
            uint32_t num_cores = shard_spec.num_cores();
            uint32_t heads_per_shard = shard_spec.shape[0] / input_tensor.get_legacy_shape()[-2];
            shard_spec.shape = {shard_spec.shape[0] / heads_per_shard, shard_spec.shape[1] * heads_per_shard};
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = shard_spec;
            return {create_sharded_device_tensor(
                this->compute_output_shapes(input_tensors).at(0),
                input_tensor.get_dtype(),
                Layout::TILE,
                input_tensor.device(),
                mem_config)};
        } else {
            return operation::generic_create_output_tensors(
                *this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
        }
    }
};

inline Tensor concatenate_heads(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [memory_config](
            std::vector<Tensor> input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors) mutable -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
            return operation::run(
                ConcatenateHeads{memory_config.value_or(input_tensor.memory_config())}, {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace transformer
}  // namespace operations
}  // namespace ttnn
