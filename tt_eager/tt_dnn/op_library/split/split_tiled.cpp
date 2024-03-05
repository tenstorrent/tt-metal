// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/split/split_tiled.hpp"

#include <iostream>

#include "common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

void SplitTiled::boiler_plate_asserts(const Tensor &a) const {
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        a.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || a.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Split does not currently support sharding");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Split does not currently support sharding");
}

void SplitTiled::shape_asserts(const Tensor &a) const {
    int chunk_size = a.get_legacy_shape()[dim] / num_chunks;
    TT_FATAL(a.get_legacy_shape()[0] == 1, "Only batch 1 implemented");
    TT_FATAL(a.get_legacy_shape()[dim] % num_chunks == 0, "Incorrect shape on last dim");
    TT_FATAL(dim <= a.get_legacy_shape().rank() && dim >= 0, "Improper dims");
    TT_FATAL(a.get_legacy_shape().rank() == 4, "W,Z,Y,X tensor");
    TT_FATAL(a.get_layout() == Layout::TILE, "Currently only tile layout support");
    TT_FATAL((a.get_legacy_shape()[2] % TILE_HEIGHT == 0), "Shape not divisible by tile");
    TT_FATAL((a.get_legacy_shape()[3] % TILE_WIDTH == 0), "Shape not divisible by tile");
    if (dim == 3)
        TT_FATAL((chunk_size % TILE_WIDTH == 0), "Chunk not divisible by tile");
    else if (dim == 2)
        TT_FATAL((chunk_size % TILE_HEIGHT == 0), "Chunk not divisible by tile");
}

inline bool is_dram(const Tensor &a) { return a.memory_config().buffer_type == BufferType::DRAM; }

Shape SplitTiled::get_single_output_shape(const Shape &input_shape) const {
    auto output_shape = input_shape;
    output_shape[dim] /= num_chunks;
    return output_shape;
}

tt::DataFormat get_data_format(const Tensor &a) {
    tt::DataFormat cb_data_format = tt::DataFormat::Bfp8_b;
    if (a.get_dtype() == tt::tt_metal::DataType::BFLOAT16) {
        cb_data_format = tt::DataFormat::Float16_b;
    }
    return cb_data_format;
}

void SplitTiled::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    tt_metal::Buffer *in0_buffer = input_tensor.buffer();
    auto cb_data_format = get_data_format(input_tensor);
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    TT_FATAL(in0_buffer->size() % single_tile_size == 0);
    boiler_plate_asserts((const Tensor &)input_tensor);
    shape_asserts((const Tensor &)input_tensor);
}

std::vector<Shape> SplitTiled::compute_output_shapes(
    const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = get_single_output_shape(input_tensor.get_legacy_shape());
    // split last dim in half
    return {output_shape, output_shape};
}

std::vector<Tensor> SplitTiled::create_output_tensors(
    const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks SplitTiled::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    return {};
}

tt::stl::reflection::Attributes SplitTiled::attributes() const {
    return {
        {"dim", this->dim},
        {"num_chunks", this->num_chunks},
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace tt_metal

}  // namespace tt
