// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/concat/concat_op.hpp"

#include "tensor/tensor_utils.hpp"

#include "tt_dnn/op_library/auto_format.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

ConcatOpParallelizationStrategy Concat::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    uint32_t num_tiles = tt_metal::compute_volume(this->compute_output_shapes(input_tensors).at(0)) / TILE_HW;
    if (num_tiles > 1) {
        return ConcatOpParallelizationStrategy::MULTI_CORE;
    } else {
        return ConcatOpParallelizationStrategy::SINGLE_CORE;
    }
}

void Concat::validate(const std::vector<Tensor> &input_tensors) const {

    tt::tt_metal::Shape shape_first = input_tensors[0].shape();
    shape_first[dim] = 0;

    for (const Tensor &in_ref : input_tensors) {
        TT_ASSERT((in_ref.layout() == Layout::TILE) && "Only tile layout supported.");
        TT_ASSERT(in_ref.device() && "Operand to concat needs to be on device.");
        TT_ASSERT(in_ref.buffer() && "Operand to concat needs to be allocated in a buffer on device.");
        TT_ASSERT(in_ref.layout() == input_tensors.at(0).layout() && "All Tensors should have same layouts.");
        tt::tt_metal::Shape curr_shape = in_ref.shape();
        curr_shape[dim] = 0;
        TT_ASSERT(curr_shape == shape_first, "concat tensors differ in shape across non-concat dimensions.");
    }
}

std::vector<tt::tt_metal::Shape> Concat::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    tt::tt_metal::Shape shape_out = input_tensors[0].shape();
    shape_out[dim] = 0;
    for (const Tensor &in_ref : input_tensors) {
        tt::tt_metal::Shape curr_shape = in_ref.shape();
        shape_out[dim] += curr_shape[dim];
    }
    return {shape_out};
}

std::vector<Tensor> Concat::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const Tensor &ref_in_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, ref_in_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

tt::stl::reflection::Attributes Concat::attributes() const {
    return {
        {"dim", this->dim},
        {"output_mem_config", this->output_mem_config},
    };
}

operation::ProgramWithCallbacks Concat::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    switch(this->get_parallelization_strategy(input_tensors)) {
        case ConcatOpParallelizationStrategy::MULTI_CORE: return concat_multi_core(input_tensors, this->dim, output_tensors[0]);
        case ConcatOpParallelizationStrategy::SINGLE_CORE:
        default:
            return concat_single_core(input_tensors, this->dim, output_tensors[0]);
    };
}

Tensor concat(std::vector<Tensor> &input_tensors, std::int64_t dim, const MemoryConfig& output_mem_config) {
    TT_ASSERT(input_tensors.size() > 0, "need 1 or more tensors");
    uint32_t ref_rank = input_tensors[0].shape().rank();
    uint32_t normalized_dim =  input_tensors[0].shape().get_normalized_index(dim);
    if (normalized_dim == ref_rank - 1) {
        for (const auto& input_tensor : input_tensors) {
            TT_ASSERT(input_tensor.shape()[dim] % TILE_WIDTH == 0);
        }
    } else if (normalized_dim == ref_rank - 2) {
        for (const auto& input_tensor : input_tensors) {
            TT_ASSERT(input_tensor.shape()[dim] % TILE_HEIGHT == 0);
        }
    }
    return operation::run_with_autoformat(Concat{normalized_dim}, {input_tensors}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
