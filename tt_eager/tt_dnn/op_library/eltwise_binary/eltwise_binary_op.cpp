// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

std::map<string, string> get_defines(BinaryOpType op_type, const std::optional<std::vector<UnaryWithParam>> fused_activations) {
    std::map<string, string> defines;
    string op_name = "sub_tiles";
    string op_code = "1";
    string idst = "i";

    switch (op_type) {
        case BinaryOpType::ADD:
            op_name = "add_tiles";
            op_code = "0";
            break;
        case BinaryOpType::SUB:
            op_name = "sub_tiles";
            op_code = "1";
            break;
        case BinaryOpType::MUL:
            op_name = "mul_tiles";
            op_code = "2";
            break;
        case BinaryOpType::GT: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst)); break;
        case BinaryOpType::LT: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LTZ, std::nullopt, "0", idst)); break;
        case BinaryOpType::GTE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GEZ, std::nullopt, "0", idst)); break;
        case BinaryOpType::LTE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LEZ, std::nullopt, "0", idst)); break;
        case BinaryOpType::EQ: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EQZ, std::nullopt, "0", idst)); break;
        case BinaryOpType::NE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst)); break;
        case BinaryOpType::SQUARED_DIFFERENCE: defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::SQUARE, std::nullopt, "0", idst)); break;
        case BinaryOpType::LOGICAL_AND:
            op_name = "mul_tiles";
            op_code = "2";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::BIAS_GELU:
            op_name = "add_tiles";
            op_code = "0";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GELU, 0, "0", idst));
            break;
        case BinaryOpType::LOGADDEXP:
            // PRE_IN0_0 ===> Applies prescaling for first input
            // PRE_IN1_0 ====> Applies prescaling for second input
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP, false, "PRE_IN0_0"));
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP, false, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_code = "0";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LOG, std::nullopt, "0", idst));
            break;
        case BinaryOpType::LOGICAL_OR:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0"));
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_code = "0";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst));
	    break;
        case BinaryOpType::LDEXP:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0"));
            op_name = "mul_tiles";
            op_code = "2";
            break;
        case BinaryOpType::LOGADDEXP2:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN0_0"));
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_code = "0";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LOG2, std::nullopt, "0", idst));
            break;
        default: TT_ASSERT(false && "Undefined op type");
    }

    defines["ELTWISE_OP"] = op_name.c_str();
    defines["ELTWISE_OP_CODE"] = op_code.c_str();
    if (fused_activations.has_value()) {
        if (op_type == BinaryOpType::ADD and fused_activations.value().size() == 1 and fused_activations.value().at(0).op_type == UnaryOpType::RELU) {
            defines["PACK_RELU"] = "1";
        } else {
            defines.merge(eltwise_unary_op_utils::get_block_defines(fused_activations.value(), "0", idst));
        }
    }

    return defines;
}



}  // namespace eltwise_binary_op_utils

namespace tt {

namespace tt_metal {


void EltwiseBinary::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL(input_tensor_a.get_legacy_shape() == input_tensor_b.get_legacy_shape(), "Input shapes must be the same!");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE, "Operands to eltwise binary need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to eltwise binary need to be on the same device!");
    TT_FATAL((input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE), "Inputs to eltwise binary must be tilized");
    if (this->in_place) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
        TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type);
        TT_FATAL(input_tensor_a.get_dtype() == this->output_dtype);
    }
    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            // If we aren't height sharded, we require all sharding schemes to match until we add blocked reader/writers for width and block sharding
            TT_FATAL((input_tensor_b.memory_config().is_sharded()));
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
        }
        if (input_tensor_b.memory_config().is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == input_tensor_b.memory_config().memory_layout);
            TT_FATAL(input_tensor_a.shard_spec().value() == input_tensor_b.shard_spec().value());
        }
        if (this->output_mem_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
        } else {
            TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else if (input_tensor_b.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (this->output_mem_config.is_sharded()) {
            TT_FATAL(input_tensor_b.memory_config().memory_layout == this->output_mem_config.memory_layout);
        } else {
            TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (this->output_mem_config.is_sharded()) {
            TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
            uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_cores = core_grid.x * core_grid.y;
            TT_FATAL(num_blocks < num_cores || num_blocks % num_cores == 0);

        } else {
            TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    }
}

std::vector<Shape> EltwiseBinary::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> EltwiseBinary::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    if (this->in_place) {
        return {};
    }
    if (this->output_mem_config.is_sharded()) {
        ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
        if (input_tensor_a.memory_config().is_sharded()) {
            shard_spec = input_tensor_a.shard_spec().value();
        } else if (input_tensor_b.memory_config().is_sharded()) {
            shard_spec = input_tensor_b.shard_spec().value();
        } else {
            uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_grid_cores = core_grid.x * core_grid.y;
            uint32_t target_num_cores = num_blocks < num_grid_cores ? num_blocks : num_grid_cores;
            shard_spec.grid = num_cores_to_corerange_set(target_num_cores, core_grid, true);
            shard_spec.shape = {num_blocks / target_num_cores * TILE_HEIGHT, input_tensor_a.get_legacy_shape()[-1]};
            shard_spec.orientation = ShardOrientation::ROW_MAJOR;
        }
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = shard_spec;
        return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), this->output_dtype, Layout::TILE, input_tensor_a.device(), mem_config)};
    }
    return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks EltwiseBinary::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& output_tensor = this->in_place ? input_tensor_a : output_tensors.at(0);

    switch (this->get_parallelization_strategy(input_tensors)) {
        case BinaryOpParallelizationStrategy::MULTI_CORE:
            return eltwise_binary_multi_core(input_tensor_a, input_tensor_b, output_tensor, this->op_type, this->fused_activations);
            break;
        case BinaryOpParallelizationStrategy::SINGLE_CORE:
        default: return eltwise_binary_single_core(input_tensor_a, input_tensor_b, output_tensor, this->op_type, this->fused_activations);
    }
}


BinaryOpParallelizationStrategy EltwiseBinary::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(0);
    uint32_t num_tiles = input_tensor_a.volume() / TILE_HW;
    if(num_tiles > 1 || input_tensor_a.memory_config().is_sharded() || input_tensor_b.memory_config().is_sharded() || this->output_mem_config.is_sharded()){
        return BinaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
       return BinaryOpParallelizationStrategy::SINGLE_CORE;
    }
}

const operation::Hash EltwiseBinary::compute_program_hash(
    const std::vector<Tensor> &input_tensors) const {
    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    operation::Hash hash = tt::stl::hash::hash_objects(
        0,
        typeid(*this).hash_code(),
        this->op_type,
        parallelization_strategy,
        input_tensor_a.get_dtype(),
        input_tensor_a.memory_config(),
        input_tensor_b.get_dtype(),
        input_tensor_b.memory_config(),
        this->output_dtype,
        this->output_mem_config,
        this->in_place);

    if (this->fused_activations.has_value()) {
        for (const auto& unary_with_param_op : this->fused_activations.value()) {
            hash = tt::stl::hash::hash_objects(hash, static_cast<uint16_t>(unary_with_param_op.op_type));
            if (unary_with_param_op.param.has_value()) {
                hash = tt::stl::hash::hash_objects(hash, static_cast<uint16_t>(unary_with_param_op.param.value()));
            }
        }
    }
    return hash;
}

}  // namespace tt_metal

}  // namespace tt
