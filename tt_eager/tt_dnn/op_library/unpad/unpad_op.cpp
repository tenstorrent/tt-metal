// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/unpad/unpad_op.hpp"

#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

uint32_t get_upper_dims_compressed(const Tensor &tensor) {
    uint32_t upper_dims_compressed = 1;

    for (uint32_t dim = 0; dim < tensor.shape().rank() - 2; dim++) {
        upper_dims_compressed *= tensor.shape()[dim];
    }

    return upper_dims_compressed;
}

uint32_t get_upper_start_offset(const Tensor &tensor, const Shape &output_tensor_start) {
    // offset for every dim except last 2
    uint32_t start_offset = 0;

    uint32_t num_pages = tensor.volume();
    if (tensor.layout() == Layout::TILE) {
        num_pages /= (TILE_HW);
    } else {
        uint32_t page_width = tensor.shape()[-1];
        num_pages /= page_width;
    }

    for (uint32_t dim_outer = 0; dim_outer < tensor.shape().rank() - 2; dim_outer++) {
        uint32_t compressed_dims = 1;
        for (uint32_t dim_inner = 0; dim_inner <= dim_outer; dim_inner++) {
            compressed_dims *= tensor.shape()[dim_inner];
        }
        start_offset += (num_pages / compressed_dims) * output_tensor_start[dim_outer];
    }
    return start_offset;
}

uint32_t get_tiled_start_offset(const Tensor &input_tensor, const Shape &output_tensor_start) {
    uint32_t num_input_pages = input_tensor.volume() / (TILE_HW);

    uint32_t upper_dims_compressed = get_upper_dims_compressed(input_tensor);
    uint32_t num_pages_width = num_input_pages / (upper_dims_compressed * (input_tensor.shape()[-2] / TILE_HEIGHT));

    // offset for every dim except last 2
    uint32_t start_offset = get_upper_start_offset(input_tensor, output_tensor_start);

    start_offset += output_tensor_start[-2] / TILE_HEIGHT * num_pages_width + output_tensor_start[-1] / TILE_WIDTH;
    return start_offset;
}

uint32_t get_rm_start_offset(const Tensor &tensor, const Shape &output_tensor_start) {
    uint32_t start_offset = 0;

    if (tensor.shape().rank() >= 2) {
        uint32_t num_pages = tensor.volume() / tensor.shape()[-1];
        uint32_t upper_dims_compressed = get_upper_dims_compressed(tensor);
        start_offset = get_upper_start_offset(tensor, output_tensor_start);
        start_offset += output_tensor_start[-2];
    }

    return start_offset;
}

void Unpad::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR);

    for (uint32_t i = 0; i < input_tensor_a.shape().rank(); i++) {
        TT_FATAL(this->output_tensor_start[i] < input_tensor_a.shape()[i]);
        TT_FATAL(this->output_tensor_end[i] < input_tensor_a.shape()[i]);

        // Check if start shape is <= end shape
        TT_FATAL(this->output_tensor_start[i] <= this->output_tensor_end[i]);
    }

    Shape output_tensor_shape = this->compute_output_shapes(input_tensors)[0];

    if (input_tensor_a.layout() == Layout::TILE) {
        TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);
        TT_FATAL(
            (output_tensor_shape[-2] % TILE_HEIGHT == 0) && (this->output_tensor_start[-2] % TILE_HEIGHT == 0),
            "Can only unpad tilized tensor with full tiles");
        TT_FATAL(
            (output_tensor_shape[-1] % TILE_WIDTH == 0) && (this->output_tensor_start[-1] % TILE_WIDTH == 0),
            "Can only unpad tilized tensor with full tiles");
    } else if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            (output_tensor_shape[-1] * input_tensor_a.element_size() % sizeof(uint32_t) == 0) &&
                (this->output_tensor_start[-1] * input_tensor_a.element_size() % sizeof(uint32_t) == 0),
            "RM unpadding requires output X size to be packable");
    }

    auto input_mem_config = input_tensor_a.memory_config();
    TT_FATAL(not input_tensor_a.is_sharded(), "Input sharding not supported");
    if (this->output_mem_config.is_sharded()) {
        TT_FATAL(
            this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
            "Currently only supporting height sharded on output");
        if (this->output_mem_config.shard_spec.has_value()) {
            TT_FATAL(
                div_up(
                    tt::tt_metal::compute_volume(output_tensor_shape) / output_tensor_shape[-1],
                    this->output_mem_config.shard_spec.value().shape[0]) ==
                    this->output_mem_config.shard_spec.value().num_cores(),
                "Output sharding only supports uneven shards on last core");
        }
    }
}

std::vector<Shape> Unpad::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    std::vector<uint32_t> out_shape;
    auto rank = input_tensors[0].shape().rank();
    out_shape.reserve(rank);
    for (uint32_t i = 0; i < rank; i++) {
        out_shape.push_back(this->output_tensor_end[i] - this->output_tensor_start[i] + 1);
    }
    Shape output_tensor_shape(out_shape);
    return {output_tensor_shape};
}

std::vector<Tensor> Unpad::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor_a = input_tensors.at(0);
    auto input_shape = input_tensor_a.shape();
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);

    if (this->output_mem_config.is_sharded()) {
        if (not this->output_mem_config.shard_spec.has_value()) {
            auto compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();

            uint32_t output_volume = tt::tt_metal::compute_volume(output_shape);

            uint32_t page_width;
            uint32_t page_height;
            if (input_tensor_a.layout() == Layout::TILE) {
                page_height = TILE_HEIGHT;
                page_width = TILE_WIDTH;
            } else {
                page_height = 1;
                page_width = output_shape[-1];
            }

            bool row_wise = true;
            ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;

            uint32_t units_to_divide = output_volume / (output_shape[-1] * page_height);
            auto [all_cores, units_per_core] =
                shard_split_work_to_cores(compute_with_storage_grid_size, units_to_divide, row_wise);

            std::array<uint32_t, 2> shard_shape = {units_per_core * page_height, output_shape[-1]};

            auto shard_spec =
                ShardSpec{.shard_grid = all_cores, .shard_shape = shard_shape, .shard_orientation = shard_orientation};
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = shard_spec;
            return {create_sharded_device_tensor(
                output_shape, input_tensor_a.dtype(), input_tensor_a.layout(), input_tensor_a.device(), mem_config)};
        } else {
            return {create_sharded_device_tensor(
                output_shape,
                input_tensor_a.dtype(),
                input_tensor_a.layout(),
                input_tensor_a.device(),
                this->output_mem_config)};
        }
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, input_tensor_a.dtype(), input_tensor_a.layout(), this->output_mem_config);
    }
}

// TODO: If unpad is called on a tile and output is not tile, we could untilize then unpad, and output is RM
// Currently calling unpad on a tile requires the output unpad shape to be tile
operation::ProgramWithCallbacks Unpad::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input_tensor_a = input_tensors.at(0);
    auto &output_tensor = output_tensors.at(0);
    switch (this->get_parallelization_strategy(input_tensors)) {
        case UnpadOpParallelizationStrategy::MULTI_CORE:
            return unpad_impl::multi_core::get_program(
                input_tensor_a, output_tensor, output_tensor_start, output_tensor_end);
        case UnpadOpParallelizationStrategy::SINGLE_CORE:
        default: return unpad_single_core(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end);
    };
}

UnpadOpParallelizationStrategy Unpad::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor_a = input_tensors.at(0);
    uint32_t num_units;
    auto shape = this->compute_output_shapes(input_tensors).at(0);
    if (input_tensor_a.layout() == Layout::TILE) {
        num_units = tt::tt_metal::compute_volume(shape) / TILE_HW;
    } else {
        num_units = tt::tt_metal::compute_volume(shape) / shape[-1];
    }
    if (num_units > 1 || input_tensor_a.is_sharded() || this->output_mem_config.is_sharded()) {
        return UnpadOpParallelizationStrategy::MULTI_CORE;
    } else {
        return UnpadOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes Unpad::attributes() const {
    return {
        {"output_tensor_start", this->output_tensor_start},
        {"output_tensor_end", this->output_tensor_end},
        {"output_mem_config", this->output_mem_config},
    };
}

const operation::Hash Unpad::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    auto input_tensor = input_tensors.at(0);
    tt_metal::Device *device = input_tensor.device();
    auto input_mem_config = input_tensor.memory_config();
    auto input_device_id = input_tensor.device()->id();
    auto output_mem_config = this->output_mem_config;
    auto dtype = input_tensor.dtype();
    auto num_dims = input_tensor.shape().rank();

    auto str = operation::hash_operation<Unpad>(
        num_dims,
        input_tensor.layout(),
        input_mem_config.memory_layout,
        input_mem_config.buffer_type,
        input_device_id,
        output_mem_config.memory_layout,
        output_mem_config.buffer_type,
        dtype,
        get_parallelization_strategy(input_tensors));
    return str;
}

Tensor unpad(
    const Tensor &input_tensor_a,
    const Shape &output_tensor_start,
    const Shape &output_tensor_end,
    const MemoryConfig &output_mem_config) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    auto input_tensor_shape = input_tensor_a.shape();
    const Shape output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };
    if (input_tensor_a.shape() == output_tensor_shape) {
        return AutoFormat::move_tensor_to_mem_config(input_tensor_a, output_mem_config);
    }

    return operation::run_without_autoformat(
               Unpad{
                   output_tensor_start, output_tensor_end, output_mem_config, output_tensor_shape, input_tensor_shape},
               {input_tensor_a})
        .at(0);
}

void UnpadOnHost::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() != StorageType::DEVICE);
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR);

    TT_FATAL(this->output_tensor_start[0] < input_tensor.shape()[0]);
    TT_FATAL(this->output_tensor_end[0] < input_tensor.shape()[0]);
    TT_FATAL(this->output_tensor_start[1] < input_tensor.shape()[1]);
    TT_FATAL(this->output_tensor_end[1] < input_tensor.shape()[1]);
    TT_FATAL(this->output_tensor_start[2] < input_tensor.shape()[2]);
    TT_FATAL(this->output_tensor_end[2] < input_tensor.shape()[2]);
    TT_FATAL(this->output_tensor_start[3] < input_tensor.shape()[3]);
    TT_FATAL(this->output_tensor_end[3] < input_tensor.shape()[3]);

    // Check if start shape is <= end shape
    TT_FATAL(this->output_tensor_start[0] <= this->output_tensor_end[0]);
    TT_FATAL(this->output_tensor_start[1] <= this->output_tensor_end[1]);
    TT_FATAL(this->output_tensor_start[2] <= this->output_tensor_end[2]);
    TT_FATAL(this->output_tensor_start[3] <= this->output_tensor_end[3]);
}
std::vector<Shape> UnpadOnHost::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    Shape output_tensor_shape = {
        this->output_tensor_end[0] - this->output_tensor_start[0] + 1,
        this->output_tensor_end[1] - this->output_tensor_start[1] + 1,
        this->output_tensor_end[2] - this->output_tensor_start[2] + 1,
        this->output_tensor_end[3] - this->output_tensor_start[3] + 1,
    };
    return {output_tensor_shape};
}
std::vector<Tensor> UnpadOnHost::compute_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    if (input_tensor.shape() == UnpadOnHost::compute_output_shapes(input_tensors).at(0)) {
        return {input_tensor};
    } else {
        return {input_tensor.unpad(this->output_tensor_start, this->output_tensor_end)};
    }
}

tt::stl::reflection::Attributes UnpadOnHost::attributes() const {
    return {
        {"output_tensor_start", this->output_tensor_start},
        {"output_tensor_end", this->output_tensor_end},
    };
}

Tensor unpad_on_host(
    const Tensor &input_tensor,
    const Shape &output_tensor_start,
    const Shape &output_tensor_end,
    const MemoryConfig &mem_config) {
    return operation::run(UnpadOnHost{output_tensor_start, output_tensor_end}, {input_tensor}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
