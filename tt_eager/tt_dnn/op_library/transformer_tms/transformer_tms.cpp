// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/transformer_tms/transformer_tms.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt {
namespace operations {
namespace primary {
namespace transformers {

void SplitFusedQKVAndSplitHeads::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.shape()[0];
    // TODO: See issue #1744
    TT_FATAL((input_tensor.shape() == Shape({batch_size, 1, 384, 3072})), "Unsupported input shape");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");

    if (input_tensor.is_sharded() == false) {
        TT_FATAL(batch_size >= 7 && batch_size <= 9, "Input batch size must be between 2 to 9 for bert large TM ops!");
    }
}

std::vector<Shape> SplitFusedQKVAndSplitHeads::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.shape()[0];
    return {Shape{batch_size, 16, 384, 64}, Shape{batch_size, 16, 64, 384}, Shape{batch_size, 16, 384, 64}};
}

std::vector<Tensor> SplitFusedQKVAndSplitHeads::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    if (input_tensor.is_sharded()) {
        // tensor dim
        uint32_t batch = input_tensor.shape()[0]; // 12
        uint32_t num_heads = this->num_heads;
        uint32_t num_output_tensors = 3;
        uint32_t M = input_tensor.shape()[2]; // 384
        uint32_t K = input_tensor.shape()[-1] / num_output_tensors / num_heads; // 64
        // core range
        CoreRangeSet all_cores({});
        ShardOrientation shard_orientation;
        auto num_cores_x = this->compute_with_storage_grid_size.x;
        auto num_cores_y = this->compute_with_storage_grid_size.y;
        all_cores = CoreRangeSet({CoreRange{.start={0, 0}, .end={num_cores_x - 1, num_cores_y - 1}}});
        // shard spec
        uint32_t per_core_M_qv = (num_heads / num_cores_y) * M; // 768
        uint32_t per_core_N_qv = K; // 64
        ShardSpec shard_spec_qv = ShardSpec{.shard_grid=all_cores,
                .shard_shape={per_core_M_qv, per_core_N_qv}, .shard_orientation=ShardOrientation::COL_MAJOR};
        uint32_t per_core_M_k = (num_heads / num_cores_y) * K; // 128
        uint32_t per_core_N_k = M; // 384
        ShardSpec shard_spec_k = ShardSpec{.shard_grid=all_cores,
                .shard_shape={per_core_M_k, per_core_N_k}, .shard_orientation=ShardOrientation::COL_MAJOR};
        // create sharded tensors
        auto out_tensor_q = create_sharded_device_tensor(Shape{batch, num_heads, M, K}, input_tensor.dtype(),
                Layout::TILE, input_tensor.device(), this->output_mem_config, shard_spec_qv);
        auto out_tensor_k = create_sharded_device_tensor(Shape{batch, num_heads, K, M}, input_tensor.dtype(),
                Layout::TILE, input_tensor.device(), this->output_mem_config, shard_spec_k);
        auto out_tensor_v = create_sharded_device_tensor(Shape{batch, num_heads, M, K}, input_tensor.dtype(),
                Layout::TILE, input_tensor.device(), this->output_mem_config, shard_spec_qv);
        return {out_tensor_q, out_tensor_k, out_tensor_v};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
    }

}

operation::ProgramWithCallbacks SplitFusedQKVAndSplitHeads::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto device_compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    TT_ASSERT((this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x && this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y), "Unsupported grid shape");

    if (input_tensor.is_sharded()) {
        return multi_core_split_query_key_value_and_split_heads_sharded(input_tensor, output_tensors, this->compute_with_storage_grid_size);
    } else {
        return multi_core_split_query_key_value_and_split_heads(input_tensor, output_tensors, this->compute_with_storage_grid_size);
    }
}

tt::stl::reflection::Attributes SplitFusedQKVAndSplitHeads::attributes() const {
    return {
        {"compute_with_storage_grid_size", this->compute_with_storage_grid_size.str()},
        {"output_mem_config", this->output_mem_config},
    };
}

void ConcatenateHeads::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.shape()[0];
    // TODO: See issue #1744
    TT_FATAL(batch_size >= 7 && batch_size <= 9, "Input batch size must be between 2 to 9 for bert large TM ops!");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");

    TT_FATAL((input_tensor.shape() == Shape({batch_size, 16, 384, 64})), "Unsupported input shape");
}

std::vector<Shape> ConcatenateHeads::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.shape()[0];
    return {Shape{batch_size, 1, 384, 1024}};
}

std::vector<Tensor> ConcatenateHeads::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks ConcatenateHeads::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    const auto batch_size = input_tensor.shape()[0];

    auto device_compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    TT_ASSERT((this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x && this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y), "Unsupported grid shape");

    return multi_core_concat_heads(input_tensor, output_tensor, this->compute_with_storage_grid_size);
}

tt::stl::reflection::Attributes ConcatenateHeads::attributes() const {
    return {
        {"compute_with_storage_grid_size", this->compute_with_storage_grid_size.str()},
        {"output_mem_config", this->output_mem_config},
    };
}

void AttnMatmul::validate(const std::vector<Tensor>& input_tensors) const {
    // input_a: [q_len, q_heads, batch, head_dim]
    // input_b: [batch, kv_heads, head_dim, kv_len]
    // intermediate: [q_heads, batch, batch, kv_len]
    // output: [q_len, q_heads, batch, kv_len]

    TT_FATAL(input_tensors.size() == 2);
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL((input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE), "Inputs to matmul must be tilized");

    // TODO: Uplift to support BFLOAT8_B and mixed precision
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE, "Operands to matmul need to be on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

    const auto ashape = input_tensor_a.shape();
    const auto bshape = input_tensor_b.shape();
    TT_FATAL((ashape[0] == 1), "Input q_len must be 1!");
    TT_FATAL((bshape[1] == 1), "Number of kv_heads must be 1!"); // TODO: May need to uplift to support falcon-40B
    TT_FATAL((ashape[2] == bshape[0]), "Num of users must match!");

    bool read_from_kv_cache = false;
    if (this->num_tokens.has_value() or this->transpose_hw.has_value()) {
        TT_FATAL((this->num_tokens.has_value() and this->transpose_hw.has_value()), "Must provide num_tokens and transpose_hw flag if we are reading from cache for in1!");
        TT_FATAL(this->num_tokens.value() % 32 == 0, "Number of tokens must be divisble by 32!");
        read_from_kv_cache = true;
    }

    if (read_from_kv_cache) {
        if (this->transpose_hw.value()) {
            TT_FATAL(ashape[3] == bshape[3] && "For pre-attention matmul, dimension K for B is in B.shape[3], so A.shape[3] must match B.shape[3]"); // A.K == B.K
        } else {
            TT_FATAL(ashape[3] == this->num_tokens && "For post-attention matmul, dimension K (A.shape[3]) is the kv_seq_len in this case and must match the length of the cache we read"); // A.K == B.K
        }
    } else {
        TT_FATAL(ashape[3] == bshape[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in attn_matmul op"); // A.K == B.K
    }
}

std::vector<Shape> AttnMatmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // input_a: [q_len, q_heads, batch, head_dim]
    // input_b: [batch, kv_heads, head_dim, kv_len]
    // intermediate: [q_heads, batch, batch, kv_len]
    // output: [q_len, q_heads, batch, kv_len]
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto ashape = input_tensor_a.shape();
    const auto bshape = input_tensor_b.shape();

    uint32_t N = bshape[3];
    if (this->transpose_hw.value_or(false)) {
        N = this->num_tokens.value();
    }

    return {Shape{1, ashape[1], ashape[2], N}};
}

std::vector<Tensor> AttnMatmul::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks AttnMatmul::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    auto device_compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
    TT_ASSERT((this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x && this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y), "Unsupported grid shape");

    return multi_core_attn_matmul(input_tensor_a, input_tensor_b, output_tensor, this->num_tokens, this->transpose_hw, this->compute_with_storage_grid_size);
}

tt::stl::reflection::Attributes AttnMatmul::attributes() const {
    return {
        {"transpose_hw", this->transpose_hw},
        {"compute_with_storage_grid_size", this->compute_with_storage_grid_size.str()},
        {"output_mem_config", this->output_mem_config},
        {"output_dtype", this->output_dtype},
    };
}


const operation::Hash AttnMatmul::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    return operation::hash_operation<AttnMatmul>(
        this->transpose_hw,
        this->output_mem_config,
        this->output_dtype,
        input_tensors.at(0).memory_config(),
        input_tensors.at(0).dtype(),
        input_tensors.at(1).memory_config(),
        input_tensors.at(1).dtype());
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
