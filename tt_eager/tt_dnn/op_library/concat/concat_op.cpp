// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/concat/concat_op.hpp"

#include "tt_dnn/op_library/copy/copy_op.hpp"

#include "tensor/tensor_utils.hpp"

#include "tt_dnn/op_library/auto_format.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

ConcatOpParallelizationStrategy Concat::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {

    if(input_tensors[0].is_sharded()) {
        return ConcatOpParallelizationStrategy::SHARDED_MULTI_CORE;
    }
    else {
        uint32_t num_tiles = tt_metal::compute_volume(this->compute_output_shapes(input_tensors).at(0)) / TILE_HW;
        if (num_tiles > 1) {
            return ConcatOpParallelizationStrategy::MULTI_CORE;
        } else {
            return ConcatOpParallelizationStrategy::SINGLE_CORE;
        }
    }
}

void Concat::validate(const std::vector<Tensor> &input_tensors) const {

    tt::tt_metal::Shape shape_first = input_tensors[0].shape();
    shape_first[dim] = 0;
    bool shard_first = input_tensors[0].is_sharded();

    for (const Tensor &in_ref : input_tensors) {
        TT_FATAL(in_ref.device() && "Operand to concat needs to be on device.");
        TT_FATAL(in_ref.buffer() && "Operand to concat needs to be allocated in a buffer on device.");
        TT_FATAL(in_ref.layout() == input_tensors.at(0).layout() && "All Tensors should have same layouts.");
        tt::tt_metal::Shape curr_shape = in_ref.shape();
        curr_shape[dim] = 0;
        TT_FATAL(curr_shape == shape_first, "concat tensors differ in shape across non-concat dimensions.");
        TT_FATAL(in_ref.is_sharded() == shard_first, "All tensors must be sharded or all must be interleaved");
        if(shard_first) {
            TT_FATAL((in_ref.layout() == Layout::ROW_MAJOR) && "Only row major  supported for sharded concat.");
        }
        else {
            TT_FATAL((in_ref.layout() == Layout::TILE) && "Only tile layout supported.");
        }
    }
    if(shard_first) {
        TT_FATAL(dim == 3, "Only width concat on sharded tensors");
        TT_FATAL(this->output_mem_config.is_sharded(), "output must be sharded if input is sharded");
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


    if(this->output_mem_config.is_sharded()) {
        return {create_sharded_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            ref_in_tensor.dtype(),
            ref_in_tensor.layout(),
            ref_in_tensor.device(),
            this->output_mem_config
            )};
    }
    else {
        return operation::generic_create_output_tensors(*this, input_tensors, ref_in_tensor.dtype(), ref_in_tensor.layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Concat::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    switch(this->get_parallelization_strategy(input_tensors)) {
        case ConcatOpParallelizationStrategy::SHARDED_MULTI_CORE: return sharded_concat_multi_core(input_tensors, this->dim, output_tensors[0]);
        case ConcatOpParallelizationStrategy::MULTI_CORE: return concat_multi_core(input_tensors, this->dim, output_tensors[0]);
        case ConcatOpParallelizationStrategy::SINGLE_CORE:
        default:
            return concat_single_core(input_tensors, this->dim, output_tensors[0]);
    };
}

Tensor concat(std::vector<Tensor> &input_tensors, std::int64_t dim, const MemoryConfig& output_mem_config) {
    TT_FATAL(input_tensors.size() > 0, "need 1 or more tensors");
    if (input_tensors.size() == 1) {
        return AutoFormat::move_tensor_to_mem_config(input_tensors[0], output_mem_config);
    }
    uint32_t ref_rank = input_tensors[0].shape().rank();
    uint32_t normalized_dim =  input_tensors[0].shape().get_normalized_index(dim);

    if(input_tensors[0].is_sharded()) {
        return operation::run(Concat{normalized_dim, output_mem_config}, {input_tensors}).at(0);
    }
    else {
        return operation::run_with_autoformat(Concat{normalized_dim}, {input_tensors}).at(0);
    }
}

}  // namespace tt_metal

}  // namespace tt
