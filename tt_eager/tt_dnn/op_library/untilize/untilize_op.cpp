// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tensor/tensor_utils.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void Untilize::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to untilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Can only untilize tile major data");

    TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);

    if (input_tensor_a.memory_config().is_sharded()) {
        if (this->output_mem_config.is_sharded()) {
            TT_FATAL(this->output_mem_config.memory_layout == input_tensor_a.memory_config().memory_layout);
        }
        if (input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
        }
        TT_FATAL(this->use_multicore == true);
    } else if (this->output_mem_config.is_sharded()) {
        TT_FATAL(this->use_multicore == true);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        uint32_t ntiles = input_tensor_a.volume() / TILE_HW;
        uint32_t ntiles_per_block = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;
        uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
        auto num_cores = untilize_helpers::get_num_cores(input_tensor_a.device()->compute_with_storage_grid_size(), nblocks);
        uint32_t fused_height = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
        TT_FATAL(fused_height % num_cores == 0);
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<Shape> Untilize::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return {input_tensor_a.get_legacy_shape()};
}

std::vector<Tensor> Untilize::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    DataType output_dtype = input_tensor.get_dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.get_dtype();
    if (output_mem_config.is_sharded()) {
        if (input_tensor.memory_config().is_sharded()) {
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = input_tensor.memory_config().shard_spec;
            return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), output_dtype, Layout::ROW_MAJOR, input_tensor.device(), mem_config)};
        } else {
            uint32_t ntiles = input_tensor.volume() / TILE_HW;
            uint32_t ntiles_per_block = input_tensor.get_legacy_shape()[-1] / TILE_WIDTH;
            uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
            auto num_cores = untilize_helpers::get_num_cores(input_tensor.device()->compute_with_storage_grid_size(), nblocks);
            auto shard_grid = num_cores_to_corerange_set(num_cores, input_tensor.device()->compute_with_storage_grid_size(), true);
            uint32_t fused_height = input_tensor.volume() / input_tensor.get_legacy_shape()[-1];
            std::array<uint32_t, 2> shard_shape = {fused_height / num_cores, input_tensor.get_legacy_shape()[-1]};
            ShardSpec shard_spec{shard_grid, shard_shape, ShardOrientation::ROW_MAJOR};
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = shard_spec;
            return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), output_dtype, Layout::ROW_MAJOR, input_tensor.device(), mem_config)};
        }
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, output_dtype, Layout::ROW_MAJOR, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Untilize::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    switch (this->get_parallelization_strategy(input_tensors)) {
        case UntilizeOpParallelizationStrategy::MULTI_CORE:
            return untilize_multi_core(input_tensor_a, output_tensor, use_pack_untilize, this->fp32_dest_acc_en);
        case UntilizeOpParallelizationStrategy::SINGLE_CORE:
        default:
            return untilize_single_core(input_tensor_a, output_tensor, use_pack_untilize, this->fp32_dest_acc_en);
    }
}

UntilizeOpParallelizationStrategy Untilize::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    if (this->use_multicore) {
        return UntilizeOpParallelizationStrategy::MULTI_CORE;
    } else {
        return UntilizeOpParallelizationStrategy::SINGLE_CORE;
    }
}

Tensor untilize(const Tensor &input_tensor_a, const MemoryConfig& output_mem_config, bool use_multicore, bool use_pack_untilize) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.get_layout() == Layout::ROW_MAJOR) {
        log_warning("Perf warning: Trying to untilize non-tilized data.");
        return AutoFormat::move_tensor_to_mem_config(input_tensor_a, output_mem_config);
    }
    bool fp32_dest_acc_en = input_tensor_a.get_dtype() == DataType::UINT32;            // MT: Currently only uint32 is moved to DST directly, fp32 is converted to fp16b
    return operation::run_without_autoformat(Untilize{output_mem_config, use_multicore, use_pack_untilize, fp32_dest_acc_en}, {input_tensor_a}).at(0);
}


void UntilizeWithUnpadding::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operandsneed to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Can only untilize tile major data");

    TT_FATAL(
        (this->output_tensor_start[0] == 0 && this->output_tensor_start[1] == 0 && this->output_tensor_start[2] == 0 && this->output_tensor_start[3] == 0),
        "On device unpadding only supports unpadding at end of dims"
    );

    TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);
    for (uint32_t i = 0; i < input_tensor_a.get_legacy_shape().rank(); i++) {
        TT_FATAL(this->output_tensor_start[i] < input_tensor_a.get_legacy_shape()[i]);
        TT_FATAL(this->output_tensor_end[i] < input_tensor_a.get_legacy_shape()[i]);

        // Check if start shape is <= end shape
        TT_FATAL(this->output_tensor_start[i] <= this->output_tensor_end[i]);
    }

    TT_FATAL(((this->output_tensor_end[-1] - this->output_tensor_start[-1] + 1) % 2 == 0), "Can only unpad to row major tensor of even width");

    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
            TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
            TT_FATAL(input_tensor_a.volume() / (input_tensor_a.get_legacy_shape()[-2] * input_tensor_a.get_legacy_shape()[-1]) == 1, "Can only write unbatched output interleaved");
        } else if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            // TODO ...
        } else if(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            auto output_shape = this->compute_output_shapes(input_tensors).at(0);
            // Minor host code changes required to remove this restriction
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
            for (uint32_t i = 0; i < output_shape.rank() - 2; i++) {
                TT_FATAL(input_tensor_a.get_legacy_shape()[i] == output_shape[i]);
            }
            if (output_mem_config.is_sharded()) {
                TT_FATAL(this->output_mem_config.memory_layout == input_tensor_a.memory_config().memory_layout);
                TT_FATAL(input_tensor_a.get_legacy_shape()[-1] == output_shape[-1]);
            } else {
                TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
                TT_FATAL(input_tensor_a.volume() / (input_tensor_a.get_legacy_shape()[-2] * input_tensor_a.get_legacy_shape()[-1]) == 1, "Can only write unbatched output interleaved");
                TT_FATAL(input_tensor_a.get_legacy_shape()[-1] - output_shape[-1] < input_tensor_a.shard_spec().value().shape[1]);
            }
        } else {
            TT_FATAL(false, "Unsupported sharding scheme");
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}
std::vector<Shape> UntilizeWithUnpadding::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    std::vector<uint32_t> out_shape;
    auto rank = input_tensors[0].get_legacy_shape().rank();
    out_shape.reserve(rank);
    for (uint32_t i = 0; i < rank; i++) {
        out_shape.push_back(this->output_tensor_end[i] - this->output_tensor_start[i] + 1);
    }
    Shape output_tensor_shape(out_shape);
    return {output_tensor_shape};
}
std::vector<Tensor> UntilizeWithUnpadding::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    DataType output_dtype = input_tensor_a.get_dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor_a.get_dtype();
    if (input_tensor_a.memory_config().is_sharded() && this->output_mem_config.is_sharded()) {
        auto output_shape = this->compute_output_shapes(input_tensors).at(0);
        uint32_t fused_height = tt_metal::compute_volume(output_shape) / output_shape[-1];
        uint32_t num_cores = input_tensor_a.shard_spec().value().num_cores();
        std::array<uint32_t, 2> shard_shape;
        ShardSpec shard_spec = input_tensor_a.shard_spec().value();
        if (input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            shard_shape = {div_up(fused_height, num_cores), output_shape[-1]};
        } else {
            shard_shape = {fused_height, output_shape[-1] / num_cores};
        }
        shard_spec.shape = shard_shape;
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = shard_spec;
        return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), output_dtype, Layout::ROW_MAJOR, input_tensor_a.device(), mem_config)};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, output_dtype, Layout::ROW_MAJOR, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks UntilizeWithUnpadding::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    switch (this->get_parallelization_strategy(input_tensors)) {
        case UntilizeWithUnpaddingOpParallelizationStrategy::MULTI_CORE:
            return untilize_with_unpadding_multi_core(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end, use_pack_untilize, this->fp32_dest_acc_en);
            break;
        case UntilizeWithUnpaddingOpParallelizationStrategy::SINGLE_CORE:
        default: return untilize_with_unpadding_single_core(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end, use_pack_untilize, this->fp32_dest_acc_en);
    }
}

UntilizeWithUnpaddingOpParallelizationStrategy UntilizeWithUnpadding::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    if (input_tensors.at(0).memory_config().is_sharded()) {
        return UntilizeWithUnpaddingOpParallelizationStrategy::MULTI_CORE;
    } else {
        return UntilizeWithUnpaddingOpParallelizationStrategy::SINGLE_CORE;
    }
}

Tensor untilize_with_unpadding(const Tensor &input_tensor_a, const Shape &output_tensor_start, const Shape &output_tensor_end, const MemoryConfig& output_mem_config, bool use_pack_untilize) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    const Shape output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };
    if (input_tensor_a.get_layout() != Layout::TILE) {
        if (input_tensor_a.get_legacy_shape() == output_tensor_shape) {
            log_warning("Perf warning: Untilize with unpadding called on already untilized tensor of target shape");
            return AutoFormat::move_tensor_to_mem_config(input_tensor_a, output_mem_config);
        } else {
            TT_FATAL(false, "Cannot untilize and unpad input which is not tilized");
        }
    }
    bool fp32_dest_acc_en = input_tensor_a.get_dtype() == DataType::UINT32;            // MT: Currently only uint32 is moved to DST directly, fp32 is converted to fp16b
    return operation::run_without_autoformat(UntilizeWithUnpadding{output_tensor_start, output_tensor_end, output_mem_config, use_pack_untilize, fp32_dest_acc_en}, {input_tensor_a}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
