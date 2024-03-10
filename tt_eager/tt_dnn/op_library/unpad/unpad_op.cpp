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


uint32_t get_upper_dims_compressed(const Tensor & tensor) {

    uint32_t upper_dims_compressed = 1;

    for(uint32_t dim=0; dim<tensor.get_legacy_shape().rank() - 2; dim++) {
        upper_dims_compressed *= tensor.get_legacy_shape()[dim];
    }

    return upper_dims_compressed;
}

uint32_t get_upper_start_offset(const Tensor & tensor, const Shape & output_tensor_start) {

    // offset for every dim except last 2
    uint32_t start_offset = 0;

    uint32_t num_pages = tensor.volume();
    if(tensor.get_layout() == Layout::TILE) {
        num_pages /= (TILE_HW);
    }
    else {
        uint32_t page_width = tensor.get_legacy_shape()[-1];
        num_pages /= page_width;
    }

    for(uint32_t dim_outer=0; dim_outer<tensor.get_legacy_shape().rank() - 2; dim_outer++) {
        uint32_t compressed_dims = 1;
        for(uint32_t dim_inner = 0; dim_inner <= dim_outer; dim_inner++) {
            compressed_dims*=tensor.get_legacy_shape()[dim_inner];
        }
        start_offset += (num_pages/compressed_dims) * output_tensor_start[dim_outer];
    }
    return start_offset;

}



uint32_t get_tiled_start_offset(const Tensor &input_tensor,
                        const Shape &output_tensor_start
 ) {
    uint32_t num_input_pages = input_tensor.volume() / (TILE_HW);

    uint32_t upper_dims_compressed = get_upper_dims_compressed(input_tensor);
    uint32_t num_pages_width = num_input_pages / (upper_dims_compressed * (input_tensor.get_legacy_shape()[-2]/TILE_HEIGHT));


    // offset for every dim except last 2
    uint32_t start_offset = get_upper_start_offset(input_tensor, output_tensor_start);

    start_offset += output_tensor_start[-2]/TILE_HEIGHT*num_pages_width
                                + output_tensor_start[-1]/TILE_WIDTH;
    return start_offset;

}

uint32_t get_rm_start_offset(
                        const Tensor &tensor,
                        const Shape &output_tensor_start
 ) {

    uint32_t start_offset = 0;

    if (tensor.get_legacy_shape().rank() >= 2) {
        uint32_t num_pages = tensor.volume() / tensor.get_legacy_shape()[-1];
        uint32_t upper_dims_compressed = get_upper_dims_compressed(tensor);
        start_offset = get_upper_start_offset(tensor, output_tensor_start);
        start_offset += output_tensor_start[-2];
    }

    return start_offset;
}


void Unpad::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE || input_tensor_a.get_layout() == Layout::ROW_MAJOR);

    for (uint32_t i = 0; i < input_tensor_a.get_legacy_shape().rank(); i++) {
        TT_FATAL(this->output_tensor_start[i] < input_tensor_a.get_legacy_shape()[i]);
        TT_FATAL(this->output_tensor_end[i] < input_tensor_a.get_legacy_shape()[i]);

        // Check if start shape is <= end shape
        TT_FATAL(this->output_tensor_start[i] <= this->output_tensor_end[i]);
    }

    Shape output_tensor_shape = this->compute_output_shapes(input_tensors)[0];

    if (input_tensor_a.get_layout() == Layout::TILE) {
        TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);
        TT_FATAL((output_tensor_shape[-2] % TILE_HEIGHT == 0) &&
                 (this->output_tensor_start[-2] % TILE_HEIGHT == 0),
                "Can only unpad tilized tensor with full tiles");
        TT_FATAL((output_tensor_shape[-1] % TILE_WIDTH == 0) &&
                 (this->output_tensor_start[-1] % TILE_WIDTH == 0),
                "Can only unpad tilized tensor with full tiles");
    } else if (input_tensor_a.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL(( output_tensor_shape[-1] * input_tensor_a.element_size() % sizeof(uint32_t) == 0) &&
                  (this->output_tensor_start[-1] * input_tensor_a.element_size() % sizeof(uint32_t) == 0)
                    ,"RM unpadding requires output X size to be packable");
    }
}
std::vector<Shape> Unpad::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    std::vector<uint32_t> out_shape;
    auto rank = input_tensors[0].get_legacy_shape().rank();
    out_shape.reserve(rank);
    for (uint32_t i = 0; i < rank; i++) {
        out_shape.push_back(this->output_tensor_end[i] - this->output_tensor_start[i] + 1);
    }
    Shape output_tensor_shape(out_shape);
    return {output_tensor_shape};
}
std::vector<Tensor> Unpad::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.get_dtype(), input_tensor_a.get_layout(), this->output_mem_config);
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
    if (input_tensor_a.get_layout() == Layout::TILE) {
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
    auto input_mem_config = input_tensor.memory_config();
    auto output_mem_config = this->output_mem_config;
    auto dtype = input_tensor.get_dtype();
    auto num_dims = input_tensor.get_legacy_shape().rank();


    std::string rm_width = "TILE";
    if(input_tensor.get_layout() == Layout::ROW_MAJOR){
        rm_width = fmt::format("{}", input_tensor.get_legacy_shape()[3]);
    }

    auto str = operation::hash_operation<Unpad>(
        num_dims,
        input_tensor.get_layout(),
        input_mem_config.memory_layout,
        input_mem_config.buffer_type,
        output_mem_config.memory_layout,
        output_mem_config.buffer_type,
        dtype,
        get_parallelization_strategy(input_tensors),
        rm_width

    );
    return str;

}

Tensor unpad(const Tensor &input_tensor_a, const Shape &output_tensor_start, const Shape &output_tensor_end, const MemoryConfig& output_mem_config) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    auto input_tensor_shape = input_tensor_a.get_legacy_shape();
    const Shape output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };
    if (input_tensor_a.get_legacy_shape() == output_tensor_shape) {
        return AutoFormat::move_tensor_to_mem_config(input_tensor_a, output_mem_config);
    }

    return operation::run_without_autoformat(Unpad{output_tensor_start, output_tensor_end, output_mem_config, output_tensor_shape, input_tensor_shape}, {input_tensor_a}).at(0);

}

void UnpadOnHost::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() != StorageType::DEVICE);
    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR);

    TT_FATAL(this->output_tensor_start[0] < input_tensor.get_legacy_shape()[0]);
    TT_FATAL(this->output_tensor_end[0] < input_tensor.get_legacy_shape()[0]);
    TT_FATAL(this->output_tensor_start[1] < input_tensor.get_legacy_shape()[1]);
    TT_FATAL(this->output_tensor_end[1] < input_tensor.get_legacy_shape()[1]);
    TT_FATAL(this->output_tensor_start[2] < input_tensor.get_legacy_shape()[2]);
    TT_FATAL(this->output_tensor_end[2] < input_tensor.get_legacy_shape()[2]);
    TT_FATAL(this->output_tensor_start[3] < input_tensor.get_legacy_shape()[3]);
    TT_FATAL(this->output_tensor_end[3] < input_tensor.get_legacy_shape()[3]);

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
    if (input_tensor.get_legacy_shape() == UnpadOnHost::compute_output_shapes(input_tensors).at(0)) {
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
