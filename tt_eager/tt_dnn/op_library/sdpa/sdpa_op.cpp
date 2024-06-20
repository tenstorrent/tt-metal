// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/sdpa/sdpa_op.hpp"

#include <optional>
#include <type_traits>

#include "common/base_types.hpp"
#include "tensor/types.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tt_dnn/op_library/math.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void ScaledDotProductAttention::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 3 and optional_input_tensors.size() == 1, "Must have 3 input tensors and mask");



    for (auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
        TT_FATAL(input_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
        TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to softmax must be tilized");
        TT_FATAL(
            input_tensor.get_dtype() == DataType::BFLOAT16 ||
            input_tensor.get_dtype() == DataType::BFLOAT8_B);

    }

    auto mask = optional_input_tensors.at(0).value();
    TT_FATAL(mask.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(input_tensors.at(0).device() == mask.device());
    TT_FATAL(mask.get_layout() == Layout::TILE);
    TT_FATAL(mask.get_dtype() == DataType::BFLOAT16 || mask.get_dtype() == DataType::BFLOAT8_B);

    TT_FATAL(mask.buffer()->buffer_type() == tt_metal::BufferType::DRAM);

    const auto q_shape = input_tensors.at(0).get_legacy_shape();
    const auto k_shape = input_tensors.at(1).get_legacy_shape();
    const auto v_shape = input_tensors.at(2).get_legacy_shape();
    const auto mask_shape = mask.get_legacy_shape();

    // assert all dataformats are the same
    TT_FATAL(
        input_tensors.at(0).get_dtype() == input_tensors.at(1).get_dtype() &&
        input_tensors.at(0).get_dtype() == input_tensors.at(2).get_dtype() &&
        input_tensors.at(0).get_dtype() == mask.get_dtype());


    if (this->is_causal) {
        // All inputs must be in DRAM
        for (auto& input_tensor : input_tensors) {
            TT_FATAL(input_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
        }
        // Check sequence lengths
        TT_FATAL(q_shape[-2] == k_shape[-2] && q_shape[-2] == v_shape[-2]);
        TT_FATAL(q_shape[-2] == mask_shape[-2] && q_shape[-2] == mask_shape[-1]);

        // Check batch size
        TT_FATAL(q_shape[-4] == k_shape[-4] && q_shape[-4] == v_shape[-4]);
        TT_FATAL(q_shape[-4] == mask_shape[-4]);

        // Check hidden size
        TT_FATAL(q_shape[-1] == k_shape[-1] && q_shape[-1] == v_shape[-1]);

        // Check kv heads
        TT_FATAL(k_shape[-3] == v_shape[-3]);

        // Check qkv heads
        TT_FATAL(q_shape[-3] >= k_shape[-3]);

        TT_FATAL(mask_shape[-3] == 1);

        TT_FATAL(this->output_mem_config.buffer_type == tt_metal::BufferType::DRAM);

    } else {
        // Input 0 must be sharded by height. All other inputs must be in DRAM.
        const auto Q_memcfg = input_tensors.at(0).memory_config();
        TT_FATAL(input_tensors.at(0).is_sharded() == true);
        TT_FATAL(Q_memcfg.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

        for (std::size_t i = 1; i < input_tensors.size(); i++) {
            TT_FATAL(input_tensors.at(i).buffer()->buffer_type() == tt_metal::BufferType::DRAM);
        }
        // Output memconfig must be height sharded, same as input
        TT_FATAL(this->output_mem_config.is_sharded());
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

        // Assert we are in decode mode if not causal
        // Q: [1, B, NH, D]
        // K: [1, B, S, D]
        // V: [1, B, S, D]
        // mask: [1, B, NH, VS]

        // Batch must match
        const auto B = q_shape[1];
        TT_FATAL(k_shape[1] == B);
        TT_FATAL(v_shape[1] == B);
        TT_FATAL(mask_shape[1] == B);
        TT_FATAL(Q_memcfg.shard_spec.value().grid.num_cores() == B, "Q must be height sharded by batch ");

        // NKV must be 1 if we are running in this decode mode
        TT_FATAL(q_shape[0] == 1);
        TT_FATAL(k_shape[0] == 1);
        TT_FATAL(v_shape[0] == 1);
        TT_FATAL(mask_shape[0] == 1);

        // Check sequence lengths
        TT_FATAL(k_shape[-2] == v_shape[-2]);

        // Check hidden size
        const auto D = q_shape[-1];
        TT_FATAL(k_shape[-1] == D);
        TT_FATAL(v_shape[-1] == D);

        // Check NH
        TT_FATAL(q_shape[2] == mask_shape[2]);

        // Check valid seqlen
        TT_FATAL(valid_seq_len.has_value(), "Non-causal SDPA must set valid_seq_len");
        TT_FATAL(valid_seq_len.value() == mask_shape[-1], "Mask sequence dim must match valid_seq_len");
        TT_FATAL(valid_seq_len.value() <= k_shape[-2], "valid_seq_len must be <= K sequence dim");


    }

    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              tt::operations::primary::transformers::SDPAMultiCoreProgramConfig>) {
                auto q_chunk_size = program_config.q_chunk_size;
                auto k_chunk_size = program_config.k_chunk_size;

                TT_FATAL(q_shape[-2] % q_chunk_size == 0);
                TT_FATAL(k_shape[-2] % k_chunk_size == 0);

                if (! this->is_causal) {
                    TT_FATAL(q_chunk_size == q_shape[-2], "Non-causal SDPA must have q_chunk_size == q_shape[-2]");
                    TT_FATAL(this->valid_seq_len.value() % k_chunk_size == 0, "valid_seq_len must be divisible by k_chunk_size");
                }

            } else {
                if (! this->is_causal) {
                    TT_FATAL(false, "Non-causal SDPA must use multi-core program config");
                }
            }
        },
        this->program_config);
}

std::vector<Shape> ScaledDotProductAttention::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> ScaledDotProductAttention::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks ScaledDotProductAttention::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    const auto& attn_mask = optional_input_tensors.at(0);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.get_legacy_shape()[-1]));
    }

    std::size_t q_chunk_size;
    std::size_t k_chunk_size;

    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              tt::operations::primary::transformers::SDPAMultiCoreProgramConfig>) {
                q_chunk_size = program_config.q_chunk_size;
                k_chunk_size = program_config.k_chunk_size;
            } else {
                q_chunk_size = k_chunk_size = 32;
            }
        },
        this->program_config);

    return sdpa_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        output_tensor,
        attn_mask,
        scale,
        this->is_causal,
        q_chunk_size,
        k_chunk_size,
        this->compute_kernel_config,
        this->program_config,
        this->valid_seq_len);
}

tt::stl::reflection::Attributes ScaledDotProductAttention::attributes() const {
    // fill out with everything in struct
    return {
        {"scale", this->scale},
        {"output_mem_config", this->output_mem_config},
        {"program_config", this->program_config},
        {"is_causal", this->is_causal},
        {"compute_kernel_config", this->compute_kernel_config},
        {"valid_seq_len", this->valid_seq_len.value_or(0)}};
}

void ScaledDotProductAttentionDecode::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 3 and optional_input_tensors.size() == 1, "Must have 3 input tensors and mask");



    for (auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
        TT_FATAL(input_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
        TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to softmax must be tilized");
        TT_FATAL(
            input_tensor.get_dtype() == DataType::BFLOAT16 ||
            input_tensor.get_dtype() == DataType::BFLOAT8_B);

    }

    auto mask = optional_input_tensors.at(0).value();
    TT_FATAL(mask.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(input_tensors.at(0).device() == mask.device());
    TT_FATAL(mask.get_layout() == Layout::TILE);
    TT_FATAL(mask.get_dtype() == DataType::BFLOAT16 || mask.get_dtype() == DataType::BFLOAT8_B);

    TT_FATAL(mask.buffer()->buffer_type() == tt_metal::BufferType::DRAM);

    const auto q_shape = input_tensors.at(0).get_legacy_shape();
    const auto k_shape = input_tensors.at(1).get_legacy_shape();
    const auto v_shape = input_tensors.at(2).get_legacy_shape();
    const auto mask_shape = mask.get_legacy_shape();

    // assert all dataformats are the same
    TT_FATAL(
        input_tensors.at(0).get_dtype() == input_tensors.at(1).get_dtype() &&
        input_tensors.at(0).get_dtype() == input_tensors.at(2).get_dtype() &&
        input_tensors.at(0).get_dtype() == mask.get_dtype());


    // Input 0 must be sharded by height. All other inputs must be in DRAM.
    // const auto Q_memcfg = input_tensors.at(0).memory_config();
    // TT_FATAL(input_tensors.at(0).is_sharded() == true);
    // TT_FATAL(Q_memcfg.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

    for (std::size_t i = 1; i < input_tensors.size(); i++) {
        TT_FATAL(input_tensors.at(i).buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    }
    // Output memconfig must be height sharded, same as input
    // TT_FATAL(this->output_mem_config.is_sharded());
    // TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

    // Assert we are in decode mode if not causal
    // Q: [1, B, NH, D]
    // K: [1, B, S, D]
    // V: [1, B, S, D]
    // mask: [1, B, NH, VS]

    // Batch must match
    const auto B = q_shape[1];
    TT_FATAL(k_shape[1] == B);
    TT_FATAL(v_shape[1] == B);
    TT_FATAL(mask_shape[1] == B);
    // TT_FATAL(Q_memcfg.shard_spec.value().grid.num_cores() == B, "Q must be height sharded by batch ");

    // NKV must be 1 if we are running in this decode mode
    TT_FATAL(q_shape[0] == 1);
    TT_FATAL(k_shape[0] == 1);
    TT_FATAL(v_shape[0] == 1);
    TT_FATAL(mask_shape[0] == 1);

    // Check sequence lengths
    TT_FATAL(k_shape[-2] == v_shape[-2]);

    // Check hidden size
    const auto D = q_shape[-1];
    TT_FATAL(k_shape[-1] == D);
    TT_FATAL(v_shape[-1] == D);

    // Check NH
    TT_FATAL(q_shape[2] == mask_shape[2]);

    // Check valid seqlen
    TT_FATAL(valid_seq_len.has_value(), "Non-causal SDPA must set valid_seq_len");
    TT_FATAL(valid_seq_len.value() == mask_shape[-1], "Mask sequence dim must match valid_seq_len");
    TT_FATAL(valid_seq_len.value() <= k_shape[-2], "valid_seq_len must be <= K sequence dim");


    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              tt::operations::primary::transformers::SDPAMultiCoreProgramConfig>) {
                auto q_chunk_size = program_config.q_chunk_size;
                auto k_chunk_size = program_config.k_chunk_size;

                TT_FATAL(q_shape[-2] % q_chunk_size == 0);
                TT_FATAL(k_shape[-2] % k_chunk_size == 0);

                TT_FATAL(q_chunk_size == q_shape[-2], "Non-causal SDPA must have q_chunk_size == q_shape[-2]");
                TT_FATAL(this->valid_seq_len.value() % k_chunk_size == 0, "valid_seq_len must be divisible by k_chunk_size");

                // For now, assert that chunk sizes are the same
                // TT_FATAL(q_chunk_size == k_chunk_size);

                // Ensure that batch * num_heads divides the number of cores
                // auto b_nh = q_shape[-4] * q_shape[-3];
                // auto num_cores = program_config.compute_with_storage_grid_size.x *
                // program_config.compute_with_storage_grid_size.y; TT_FATAL((num_cores / b_nh) * b_nh == num_cores);
            } else {
                TT_FATAL(false, "Non-causal SDPA must use multi-core program config");
            }
        },
        this->program_config);
}

std::vector<Shape> ScaledDotProductAttentionDecode::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> ScaledDotProductAttentionDecode::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks ScaledDotProductAttentionDecode::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    const auto& attn_mask = optional_input_tensors.at(0);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.get_legacy_shape()[-1]));
    }

    // TODO: get this from program_config
    std::size_t q_chunk_size;
    std::size_t k_chunk_size;

    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              tt::operations::primary::transformers::SDPAMultiCoreProgramConfig>) {
                q_chunk_size = program_config.q_chunk_size;
                k_chunk_size = program_config.k_chunk_size;
            } else {
                q_chunk_size = k_chunk_size = 32;
            }
        },
        this->program_config);

    return sdpa_decode_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        output_tensor,
        attn_mask,
        scale,
        k_chunk_size,
        this->compute_kernel_config,
        this->program_config,
        this->valid_seq_len);
}

// What is this?
tt::stl::reflection::Attributes ScaledDotProductAttentionDecode::attributes() const {
    // fill out with everything in struct
    return {
        {"scale", this->scale},
        {"output_mem_config", this->output_mem_config},
        {"program_config", this->program_config},
        {"compute_kernel_config", this->compute_kernel_config}};
}

namespace transformers {
// Function which is bound to the Python API
Tensor scaled_dot_product_attention(
    Tensor& input_tensor_q,
    Tensor& input_tensor_k,
    Tensor& input_tensor_v,
    std::optional<const Tensor> causal_mask,
    const bool is_causal,
    std::optional<float> scale,
    const MemoryConfig& output_mem_config,
    const SDPAProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<const uint32_t> valid_seq_len) {
    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({input_tensor_q, input_tensor_k, input_tensor_v}))};
    operation::launch_op(
        [scale, output_mem_config, program_config, is_causal, compute_kernel_config, valid_seq_len](
            std::vector<Tensor> input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor_q = input_tensors.at(0);
            const auto& input_tensor_k = input_tensors.at(1);
            const auto& input_tensor_v = input_tensors.at(2);
            const auto& causal_mask = optional_input_tensors.at(0);
            auto arch = input_tensor_q.storage_type() == StorageType::DEVICE ? input_tensor_q.device()->arch()
                                                                             : AutoFormat::GetDefaultDevice()->arch();
            auto kernel_config_val = init_device_compute_kernel_config(
                input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);
            return operation::run(
                ScaledDotProductAttention{
                    .scale = scale,
                    .output_mem_config = output_mem_config,
                    .program_config = program_config,
                    .is_causal = is_causal,
                    .compute_kernel_config = kernel_config_val,
                    .valid_seq_len=valid_seq_len},
                {input_tensor_q, input_tensor_k, input_tensor_v},
                {causal_mask});
        },
        {input_tensor_q, input_tensor_k, input_tensor_v},
        output_tensors,
        {causal_mask});
    return output_tensors.at(0);
}

Tensor scaled_dot_product_attention_decode(
    Tensor& input_tensor_q,
    Tensor& input_tensor_k,
    Tensor& input_tensor_v,
    std::optional<const Tensor> mask,
    std::optional<float> scale,
    const MemoryConfig& output_mem_config,
    const SDPAProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<const uint32_t> valid_seq_len) {
    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({input_tensor_q, input_tensor_k, input_tensor_v}))};
    operation::launch_op(
        [scale, output_mem_config, program_config, compute_kernel_config, valid_seq_len](
            std::vector<Tensor> input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor_q = input_tensors.at(0);
            const auto& input_tensor_k = input_tensors.at(1);
            const auto& input_tensor_v = input_tensors.at(2);
            const auto& mask = optional_input_tensors.at(0);
            auto arch = input_tensor_q.storage_type() == StorageType::DEVICE ? input_tensor_q.device()->arch()
                                                                             : AutoFormat::GetDefaultDevice()->arch();
            auto kernel_config_val = init_device_compute_kernel_config(
                input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);
            return operation::run(
                ScaledDotProductAttentionDecode{
                    .scale = scale,
                    .output_mem_config = output_mem_config,
                    .program_config = program_config,
                    .compute_kernel_config = kernel_config_val,
                    .valid_seq_len=valid_seq_len},
                {input_tensor_q, input_tensor_k, input_tensor_v},
                {mask});
        },
        {input_tensor_q, input_tensor_k, input_tensor_v},
        output_tensors,
        {mask});
    return output_tensors.at(0);
}


}  // namespace transformers
}  // namespace primary
}  // namespace operations

}  // namespace tt
