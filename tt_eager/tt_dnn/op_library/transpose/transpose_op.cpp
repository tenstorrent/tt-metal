// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace tt {

namespace tt_metal {

void Transpose::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to transpose need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr , "Operands to transpose need to be allocated in buffers on device!");
    const auto shape = input_tensor.get_legacy_shape();
    uint32_t W = shape[3], H = shape[2], C = shape[1], N = shape[0];
    uint32_t HW = H*W;
    TT_FATAL(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_FATAL(input_tensor.volume() % TILE_HW == 0);
    if (this->dim == TransposeOpDim::WH) {
        if (input_tensor.is_sharded()) {
            TT_FATAL(input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
            const auto shard_spec = input_tensor.shard_spec().value();
            TT_FATAL(shard_spec.shape[1] == W);
            TT_FATAL(shard_spec.shape[0] % H == 0);
            TT_FATAL(this->output_mem_config.is_sharded());
            TT_FATAL(this->output_mem_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
        } else {
            TT_FATAL(!this->output_mem_config.is_sharded());
        }
    } else {
        TT_FATAL(!input_tensor.is_sharded());
        TT_FATAL(!this->output_mem_config.is_sharded());
    }
    if (this->dim == TransposeOpDim::HC) {
        TT_FATAL(C % TILE_HEIGHT == 0);
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16);
    } else if (this->dim == TransposeOpDim::CW) {
        TT_FATAL(C % TILE_WIDTH == 0);
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16);
    } else if (this->dim == TransposeOpDim::NH) {
        TT_FATAL(N % TILE_HEIGHT == 0);
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16);
    } else if (this->dim == TransposeOpDim::NW) {
        TT_FATAL(N % TILE_WIDTH == 0);
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16);
    }
}


std::vector<Shape> Transpose::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto out_shape = input_tensor.get_legacy_shape();
    switch (this->dim){
        case TransposeOpDim::CN:
            out_shape[0] = input_tensor.get_legacy_shape()[1];
            out_shape[1] = input_tensor.get_legacy_shape()[0];
            break;
        case TransposeOpDim::HC:
            out_shape[1] = input_tensor.get_legacy_shape()[2];
            out_shape[2] = input_tensor.get_legacy_shape()[1];
            break;
        case TransposeOpDim::WH:
            out_shape[2] = input_tensor.get_legacy_shape()[3];
            out_shape[3] = input_tensor.get_legacy_shape()[2];
            break;
        case TransposeOpDim::NH:
            out_shape[0] = input_tensor.get_legacy_shape()[2];
            out_shape[2] = input_tensor.get_legacy_shape()[0];
            break;
        case TransposeOpDim::NW:
            out_shape[3] = input_tensor.get_legacy_shape()[0];
            out_shape[0] = input_tensor.get_legacy_shape()[3];
            break;
        case TransposeOpDim::CW:
            out_shape[1] = input_tensor.get_legacy_shape()[3];
            out_shape[3] = input_tensor.get_legacy_shape()[1];
            break;
    }
    return {out_shape};
}


std::vector<Tensor> Transpose::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    // This is only for WH
    if (this->output_mem_config.is_sharded()) {
        if (this->dim == TransposeOpDim::WH) {
            ShardSpec shard_spec = input_tensor.shard_spec().value();
            shard_spec.shape[0] = shard_spec.shape[0] / input_tensor.get_legacy_shape()[-2] * input_tensor.get_legacy_shape()[-1];
            shard_spec.shape[1] = input_tensor.get_legacy_shape()[-2];
            const auto output_shape = this->compute_output_shapes(input_tensors)[0];
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = shard_spec;
            return {create_sharded_device_tensor(
                output_shape,
                input_tensor.get_dtype(),
                input_tensor.get_layout(),
                input_tensor.device(),
                mem_config)};
        } else {
            TT_ASSERT(false, "Unsupported sharding");
        }
    }
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Transpose::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy) {
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            if (input_tensor.is_sharded()) {
                return transpose_wh_multi_core_sharded(input_tensor, output_tensor);
            } else {
                return transpose_wh_multi_core(input_tensor, output_tensor);
            }
            break;
        case TransposeOpParallelizationStrategy::MULTI_CORE_HC:
            return transpose_hc_multi_core(input_tensor, output_tensor);
            break;
        default:
            return transpose_single_core(input_tensor, output_tensor, this->dim);
    }
}

TransposeOpParallelizationStrategy Transpose::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto ashape = input_tensor.get_legacy_shape();
    uint32_t num_tiles = input_tensor.volume() / TILE_HW;
    if (this->dim == TransposeOpDim::WH && (num_tiles > 1 || input_tensor.is_sharded())) {
        return TransposeOpParallelizationStrategy::MULTI_CORE_WH;
    } else if (this->dim == TransposeOpDim::HC && num_tiles > 1) { // Always true for legal shape until requirement on tile size IO is no longer required
        return TransposeOpParallelizationStrategy::MULTI_CORE_HC;
    } else if (this->dim == TransposeOpDim::CN && num_tiles > 1) {
        return TransposeOpParallelizationStrategy::MULTI_CORE_CN;
    } else {
        return TransposeOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes Transpose::attributes() const {
    return {
        {"dim", this->dim},
    };
}

const operation::Hash Transpose::compute_program_hash(
    const std::vector<Tensor> &input_tensors) const {
    auto input_tensor = input_tensors.at(0);
    auto input_mem_config = input_tensor.memory_config();
    auto output_mem_config = this->output_mem_config;
    auto dtype = input_tensor.get_dtype();
    return operation::hash_operation<Transpose>(
        input_mem_config, output_mem_config, dtype, this->dim, get_parallelization_strategy(input_tensors));
}

inline Tensor transpose_(const Tensor &a, TransposeOpDim transpose_dim, const MemoryConfig& output_mem_config) {
    bool pad_c = false;
    bool pad_n = false;

    switch (transpose_dim) {
        case TransposeOpDim::HC:
            pad_c = true;
            break;
        case TransposeOpDim::NH:
            return permute(a, {2, 1, 0, 3}, output_mem_config);
            pad_n = true;
            break;
        case TransposeOpDim::NW:
            return permute(a, {3, 1, 2, 0}, output_mem_config);
            pad_n = true;
            break;
        case TransposeOpDim::CW:
            return permute(a, {0, 3, 2, 1}, output_mem_config);
            pad_c = true;
            break;
        default:
            break;
    }

    // TODO: Add pad_n to run_with_autoformat when needed
    return operation::run_with_autoformat(Transpose{transpose_dim, output_mem_config}, {a}, {}, 0, pad_c /*, pad_n */).at(0);
}

Tensor transpose(const Tensor &a, std::int64_t dim1, std::int64_t dim2, const MemoryConfig& output_mem_config) {
    uint32_t normalized_dim1 = a.get_legacy_shape().get_normalized_index(dim1);
    uint32_t normalized_dim2 = a.get_legacy_shape().get_normalized_index(dim2);

    TT_FATAL( normalized_dim1 <= 3, "dimension have to be 0-3 only corresponding to N,C,H,W");
    TT_FATAL(normalized_dim2 <= 3, "dimension have to be 0-3 only corresponding to N,C,H,W");

    if (
        (normalized_dim1 == normalized_dim2) ||
        (a.get_legacy_shape()[normalized_dim1] == 1 && a.get_legacy_shape()[normalized_dim2] == 1)
    ) {
        return AutoFormat::move_tensor_to_mem_config(a, output_mem_config);
    }

    if ( normalized_dim1 > normalized_dim2 ) {
        std::swap(normalized_dim1, normalized_dim2);
    }

    TransposeOpDim transpose_dim = TransposeOpDim::NW;

    if ( normalized_dim2 == 3 && normalized_dim1 == 0 ) {
        transpose_dim = TransposeOpDim::NW;
    } else if (normalized_dim2 == 3 && normalized_dim1 == 1) {
       transpose_dim = TransposeOpDim::CW;
    } else if (normalized_dim2 == 3 && normalized_dim1 == 2) {
        transpose_dim = TransposeOpDim::WH;
    } else if (normalized_dim2 == 2 && normalized_dim1 == 0) {
        transpose_dim = TransposeOpDim::NH;
    } else if (normalized_dim2 == 2 && normalized_dim1 == 1) {
        transpose_dim = TransposeOpDim::HC;
    } else if (normalized_dim2 == 1 && normalized_dim1 == 0) {
        transpose_dim = TransposeOpDim::CN;
    } else {
        TT_ASSERT(false, "Unsupported transpose dims");
    }
    return transpose_(a, transpose_dim, output_mem_config);
}


}  // namespace tt_metal

}  // namespace tt
