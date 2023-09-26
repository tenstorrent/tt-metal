// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void Untilize::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to untilize need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_ASSERT(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        if (this->output_mem_config.is_sharded()) {
            TT_ASSERT(this->output_mem_config == input_tensor_a.memory_config());
        }
        TT_ASSERT(this->use_multicore == true);
    } else if (this->output_mem_config.is_sharded()) {
        TT_ASSERT(this->use_multicore == true);
        uint32_t ntiles = input_tensor_a.volume() / TILE_HW;
        uint32_t ntiles_per_block = input_tensor_a.shape()[3] / TILE_WIDTH;
        uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
        auto num_cores = untilize_helpers::get_num_cores(input_tensor_a.device()->compute_with_storage_grid_size(), nblocks);
        uint32_t fused_height = input_tensor_a.volume() / input_tensor_a.shape()[-1] / TILE_HEIGHT;
        TT_ASSERT(fused_height % num_cores == 0);
    } else {
        TT_ASSERT(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_ASSERT(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<Shape> Untilize::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return {input_tensor_a.shape()};
}

std::vector<Tensor> Untilize::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (output_mem_config.is_sharded()) {
        if (input_tensor.memory_config().is_sharded()) {
            return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.dtype(), Layout::ROW_MAJOR, input_tensor.device(), this->output_mem_config, input_tensor.shard_spec().value())};
        } else {
            uint32_t ntiles = input_tensor.volume() / TILE_HW;
            uint32_t ntiles_per_block = input_tensor.shape()[-1] / TILE_WIDTH;
            uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
            auto num_cores = untilize_helpers::get_num_cores(input_tensor.device()->compute_with_storage_grid_size(), nblocks);
            auto shard_grid = num_cores_to_corerange_set(num_cores, input_tensor.device()->compute_with_storage_grid_size(), true);
            uint32_t fused_height = input_tensor.volume() / input_tensor.shape()[-1];
            std::array<uint32_t, 2> shard_shape = {fused_height / num_cores, input_tensor.shape()[-1]};
            ShardSpec shard_spec{.shard_grid=shard_grid, .shard_shape=shard_shape};
            return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.dtype(), Layout::ROW_MAJOR, input_tensor.device(), this->output_mem_config, shard_spec)};
        }
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::ROW_MAJOR, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Untilize::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (this->use_multicore) {
        return {untilize_multi_core(input_tensor_a, output_tensor)};
    } else {
        return {untilize_single_core(input_tensor_a, output_tensor)};
    }
}

tt::stl::reflection::Attributes Untilize::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor untilize(const Tensor &input_tensor_a, const MemoryConfig& mem_config, bool use_multicore) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        log_warning("Perf warning: Trying to untilize non-tilized data.");
        return input_tensor_a;
    }
    return operation::run_without_autoformat(Untilize{mem_config, use_multicore}, {input_tensor_a}).at(0);
}


void UntilizeWithUnpadding::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operandsneed to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_ASSERT(
        (this->output_tensor_start[0] == 0 && this->output_tensor_start[1] == 0 && this->output_tensor_start[2] == 0 && this->output_tensor_start[3] == 0),
        "On device unpadding only supports unpadding at end of dims"
    );

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
    TT_ASSERT(this->output_tensor_start[0] < input_tensor_a.shape()[0]);
    TT_ASSERT(this->output_tensor_end[0] < input_tensor_a.shape()[0]);
    TT_ASSERT(this->output_tensor_start[1] < input_tensor_a.shape()[1]);
    TT_ASSERT(this->output_tensor_end[1] < input_tensor_a.shape()[1]);
    TT_ASSERT(this->output_tensor_start[2] < input_tensor_a.shape()[2]);
    TT_ASSERT(this->output_tensor_end[2] < input_tensor_a.shape()[2]);
    TT_ASSERT(this->output_tensor_start[3] < input_tensor_a.shape()[3]);
    TT_ASSERT(this->output_tensor_end[3] < input_tensor_a.shape()[3]);

    // Check if start shape is <= end shape
    TT_ASSERT(this->output_tensor_start[0] <= this->output_tensor_end[0]);
    TT_ASSERT(this->output_tensor_start[1] <= this->output_tensor_end[1]);
    TT_ASSERT(this->output_tensor_start[2] <= this->output_tensor_end[2]);
    TT_ASSERT(this->output_tensor_start[3] <= this->output_tensor_end[3]);

    TT_ASSERT(((this->output_tensor_end[3] - this->output_tensor_start[3] + 1) % 2 == 0), "Can only unpad to row major tensor of even width");

}
std::vector<Shape> UntilizeWithUnpadding::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    Shape output_tensor_shape = {
        this->output_tensor_end[0] - this->output_tensor_start[0] + 1,
        this->output_tensor_end[1] - this->output_tensor_start[1] + 1,
        this->output_tensor_end[2] - this->output_tensor_start[2] + 1,
        this->output_tensor_end[3] - this->output_tensor_start[3] + 1,
    };
    return {output_tensor_shape};
}
std::vector<Tensor> UntilizeWithUnpadding::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.dtype(), Layout::ROW_MAJOR, this->output_mem_config);
}

operation::ProgramWithCallbacks UntilizeWithUnpadding::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {untilize_with_unpadding_single_core(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end)};
}

tt::stl::reflection::Attributes UntilizeWithUnpadding::attributes() const {
    return {
        {"output_tensor_start", this->output_tensor_start},
        {"output_tensor_end", this->output_tensor_end},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor untilize_with_unpadding(const Tensor &input_tensor_a, const Shape &output_tensor_start, const Shape &output_tensor_end, const MemoryConfig& mem_config) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    const Shape output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };
    if (input_tensor_a.layout() != Layout::TILE) {
        if (input_tensor_a.shape() == output_tensor_shape) {
            log_warning("Perf warning: Untilize with unpadding called on already untilized tensor of target shape");
            return input_tensor_a;
        } else {
            TT_ASSERT(false, "Cannot untilize and unpad input which is not tilized");
        }
    }
    return operation::run_without_autoformat(UntilizeWithUnpadding{output_tensor_start, output_tensor_end, mem_config}, {input_tensor_a}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
