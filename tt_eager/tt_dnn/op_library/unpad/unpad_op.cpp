// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tensor/tensor_utils.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_dnn/op_library/work_split.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


void Unpad::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR);

    TT_FATAL(
        (this->output_tensor_start[0] == 0 && this->output_tensor_start[1] == 0 && output_tensor_start[2] == 0 && output_tensor_start[3] == 0),
        "On device unpadding only supports unpadding at end of dims"
    );

    for (uint32_t i = 0; i < input_tensor_a.shape().rank(); i++) {
        TT_FATAL(this->output_tensor_start[i] < input_tensor_a.shape()[i]);
        TT_FATAL(this->output_tensor_end[i] < input_tensor_a.shape()[i]);

        // Check if start shape is <= end shape
        TT_FATAL(this->output_tensor_start[i] <= this->output_tensor_end[i]);
    }

    Shape output_tensor_shape = this->compute_output_shapes(input_tensors)[0];

    if (input_tensor_a.layout() == Layout::TILE) {
        TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);
        TT_FATAL((output_tensor_shape[-2] % TILE_HEIGHT == 0), "Can only unpad tilized tensor with full tiles");
        TT_FATAL((output_tensor_shape[-1] % TILE_WIDTH == 0), "Can only unpad tilized tensor with full tiles");
    } else if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(output_tensor_shape[-1] * input_tensor_a.element_size() % sizeof(uint32_t) == 0, "RM unpadding requires output X size to be packable");
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
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.dtype(), input_tensor_a.layout(), this->output_mem_config);
}

// TODO: If unpad is called on a tile and output is not tile, we could untilize then unpad, and output is RM
// Currently calling unpad on a tile requires the output unpad shape to be tile
operation::ProgramWithCallbacks Unpad::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    switch(this->get_parallelization_strategy(input_tensors)) {
        case UnpadOpParallelizationStrategy::MULTI_CORE:
            return unpad_multi_core(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end);
        case UnpadOpParallelizationStrategy::SINGLE_CORE:
        default:
            return unpad_single_core(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end);
    };
}

UnpadOpParallelizationStrategy Unpad::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    uint32_t num_units;
    auto shape = this->compute_output_shapes(input_tensors).at(0);
    if (input_tensor_a.layout() == Layout::TILE) {
        num_units = tt::tt_metal::compute_volume(shape) / TILE_HW;
    } else {
        num_units = tt::tt_metal::compute_volume(shape) / shape[-1];
    }
    if (num_units > 1) {
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

const operation::Hash Unpad::compute_program_hash (
    const std::vector<Tensor> &input_tensors) const {
    auto input_tensor = input_tensors.at(0);
    tt_metal::Device *device = input_tensor.device();
    auto input_mem_config = input_tensor.memory_config();
    auto output_mem_config = this->output_mem_config;
    auto dtype = input_tensor.dtype();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_dims = input_tensor.shape().rank();


    std::string rm_width = "TILE";
    if(input_tensor.layout() == Layout::ROW_MAJOR){
        rm_width = fmt::format("{}", input_tensor.shape()[3]);
    }

    auto str = operation::hash_operation<Unpad>(
        num_dims,
        input_tensor.layout(),
        input_mem_config,
        output_mem_config,
        dtype,
        get_parallelization_strategy(input_tensors),
        rm_width

    );
    return str;

}

Tensor unpad(const Tensor &input_tensor_a, const Shape &output_tensor_start, const Shape &output_tensor_end, const MemoryConfig& output_mem_config) {
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

    return operation::run_without_autoformat(Unpad{output_tensor_start, output_tensor_end, output_mem_config, output_tensor_shape, input_tensor_shape}, {input_tensor_a}).at(0);

}

void UnpadOnHost::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
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
std::vector<Tensor> UnpadOnHost::compute_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
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

Tensor unpad_on_host(const Tensor &input_tensor, const Shape &output_tensor_start, const Shape &output_tensor_end, const MemoryConfig& mem_config) {
    return operation::run(UnpadOnHost{output_tensor_start, output_tensor_end}, {input_tensor}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
