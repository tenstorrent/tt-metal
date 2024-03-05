// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tensor/tensor_utils.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/math.hpp"

#include "tt_metal/tt_stl/reflection.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void Tilize::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to tilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Can only tilize row major data");

    TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);

    auto width = input_tensor_a.get_legacy_shape()[-1];
    uint32_t stick_s =  width;
    uint32_t num_sticks = input_tensor_a.volume() / width;
    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16);

    uint32_t stick_size = stick_s * input_tensor_a.element_size(); // Assuming bfloat16 dataformat

    TT_FATAL((stick_size % 2) == 0, "Stick size must be divisible by 2");

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(this->output_mem_config.memory_layout == input_tensor_a.memory_config().memory_layout);
        TT_FATAL(this->use_multicore == true);
        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<Shape> Tilize::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto output_shape = input_tensor_a.get_legacy_shape();
    return {output_shape};
}

std::vector<Tensor> Tilize::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.memory_config().is_sharded()) {
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = input_tensor.memory_config().shard_spec;
        return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), this->output_dtype, Layout::TILE, input_tensor.device(), mem_config)};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Tilize::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    switch (this->get_parallelization_strategy(input_tensors)) {
        case TilizeOpParallelizationStrategy::MULTI_CORE:
            return tilize_multi_core(input_tensor_a, output_tensor);
        case TilizeOpParallelizationStrategy::SINGLE_CORE:
        default:
            return tilize_single_core(input_tensor_a, output_tensor);
    }
}

TilizeOpParallelizationStrategy Tilize::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    if (this->use_multicore) {
        return TilizeOpParallelizationStrategy::MULTI_CORE;
    } else {
        return TilizeOpParallelizationStrategy::SINGLE_CORE;
    }
}

Tensor tilize(const Tensor &input_tensor_a, const MemoryConfig& output_mem_config, std::optional<const DataType> output_dtype, bool use_multicore) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.get_layout() == Layout::TILE) {
        log_warning("Perf warning: tilize called on already tilized tensor.");
        return AutoFormat::move_tensor_to_mem_config(input_tensor_a, output_mem_config);
    }
    return operation::run_without_autoformat(Tilize{output_mem_config, output_dtype.value_or(input_tensor_a.get_dtype()), use_multicore}, {input_tensor_a}).at(0);
}

void TilizeWithValPadding::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Can only tilize row major data");
    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16);

    TT_FATAL(input_tensor_a.get_legacy_shape()[0] + this->input_tensor_start[0] <= this->output_tensor_shape[0]);
    TT_FATAL(input_tensor_a.get_legacy_shape()[1] + this->input_tensor_start[1] <= this->output_tensor_shape[1]);
    TT_FATAL(input_tensor_a.get_legacy_shape()[2] + this->input_tensor_start[2] <= this->output_tensor_shape[2]);
    TT_FATAL(input_tensor_a.get_legacy_shape()[3] + this->input_tensor_start[3] <= this->output_tensor_shape[3]);
    TT_FATAL((this->input_tensor_start[0] == 0 && this->input_tensor_start[1] == 0 && this->input_tensor_start[2] == 0 && this->input_tensor_start[3] == 0), "On device padding only supports padding at end of dims");

    uint32_t num_rows = this->output_tensor_shape[2];
    uint32_t inner_dim = this->output_tensor_shape[3];
    TT_FATAL(num_rows % TILE_HEIGHT == 0, "Output shape must be tilizable");
    TT_FATAL(inner_dim % TILE_WIDTH == 0, "Output shape must be tilizable");

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
        TT_FATAL(this->output_mem_config.memory_layout == input_tensor_a.memory_config().memory_layout);
        for (uint32_t i = 0; i < input_tensor_a.get_legacy_shape().rank(); i++) {
            if (i != input_tensor_a.get_legacy_shape().rank() - 2) {
                TT_FATAL(input_tensor_a.get_legacy_shape()[i] == this->output_tensor_shape[i]);
            }
        }
    }
}
std::vector<Shape> TilizeWithValPadding::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto input_shape = input_tensors.at(0).get_legacy_shape();
    auto dimensions_pads = std::vector<Padding::PadDimension>();
    for (auto index = 0; index < input_shape.rank(); index++) {
        auto front = this->input_tensor_start[index];
        auto back = this->output_tensor_shape[index] - (this->input_tensor_start[index] + input_shape[index]);
        dimensions_pads.push_back(Padding::PadDimension{.front=front, .back=back});
    }
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    return {Shape(this->output_tensor_shape, padding)};
}
std::vector<Tensor> TilizeWithValPadding::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    if (input_tensor_a.memory_config().is_sharded()) {
        auto output_shape = this->compute_output_shapes(input_tensors).at(0);
        auto shard_spec = input_tensor_a.shard_spec().value();
        shard_spec.shape[0] = tt_metal::compute_volume(output_shape) / output_shape[-1];
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = shard_spec;
        return {create_sharded_device_tensor(output_shape, this->output_dtype, Layout::TILE, input_tensor_a.device(), mem_config)};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
    }
}

// TODO: If pad is called on a tile and output is not tile, we could untilize then pad, and output is RM
// Currently calling pad on a tile requires the output pad shape to be tile
operation::ProgramWithCallbacks TilizeWithValPadding::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    switch (this->get_parallelization_strategy(input_tensors)) {
        case TilizeWithValPaddingOpParallelizationStrategy::MULTI_CORE:
            return tilize_with_val_padding_multi_core(input_tensor_a, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
        case TilizeWithValPaddingOpParallelizationStrategy::SINGLE_CORE:
        default:
            return tilize_with_val_padding_single_core(input_tensor_a, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
    }
}

TilizeWithValPaddingOpParallelizationStrategy TilizeWithValPadding::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    if (input_tensors.at(0).memory_config().is_sharded()) {
        return TilizeWithValPaddingOpParallelizationStrategy::MULTI_CORE;
    } else {
        return TilizeWithValPaddingOpParallelizationStrategy::SINGLE_CORE;
    }
}

Tensor tilize_with_val_padding(const Tensor &input_tensor_a, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value, const MemoryConfig& output_mem_config, std::optional<const DataType> output_dtype) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    if (input_tensor_a.get_layout() == Layout::TILE) {
        if (output_tensor_shape == input_tensor_a.get_legacy_shape()) {
            log_warning("Perf warning: tilize with padding called on already tilized tensor of target shape.");
            return input_tensor_a;
        } else {
            TT_FATAL(false, "Cannot tilize and pad tensor that is already tilized");
        }
    }
    return operation::run_without_autoformat(TilizeWithValPadding{output_tensor_shape, input_tensor_start, pad_value, output_mem_config, output_dtype.value_or(input_tensor_a.get_dtype())}, {input_tensor_a}).at(0);

}

Tensor tilize_with_zero_padding(const Tensor &input_tensor_a, const MemoryConfig& output_mem_config, std::optional<const DataType> output_dtype) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.get_layout() == Layout::TILE) {
        log_warning("Perf warning: tilize called on already tilized tensor.");
        return AutoFormat::move_tensor_to_mem_config(input_tensor_a, output_mem_config);
    }
    auto shape = input_tensor_a.get_legacy_shape();


    shape[2] = round_up(shape[2], TILE_HEIGHT);
    shape[3] = round_up(shape[3], TILE_WIDTH);
    return tilize_with_val_padding(input_tensor_a, shape, {0, 0, 0, 0}, 0, output_mem_config, output_dtype);
}

}  // namespace tt_metal

}  // namespace tt
