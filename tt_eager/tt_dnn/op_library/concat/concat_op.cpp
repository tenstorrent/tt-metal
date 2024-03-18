// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/concat/concat_op.hpp"

#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

ConcatOpParallelizationStrategy Concat::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    if (input_tensors[0].is_sharded()) {
        return ConcatOpParallelizationStrategy::SHARDED_MULTI_CORE;
    } else {
        uint32_t num_pages = tt_metal::compute_volume(this->compute_output_shapes(input_tensors).at(0));
        if (input_tensors[0].get_layout() == Layout::ROW_MAJOR) {
            num_pages /= input_tensors[0].get_legacy_shape()[-1];
        } else {
            num_pages /= TILE_HW;
        }
        if (num_pages > 1) {
            return ConcatOpParallelizationStrategy::MULTI_CORE;
        } else {
            return ConcatOpParallelizationStrategy::SINGLE_CORE;
        }
    }
}

void Concat::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &first_input = input_tensors[0];
    tt::tt_metal::Shape shape_first = first_input.get_legacy_shape();
    TT_FATAL(this->dim < shape_first.rank(), "Concat dim specified is larger than input tensor rank.");
    shape_first[this->dim] = 0;
    bool shard_first = input_tensors[0].is_sharded();

    for (const Tensor &in_ref : input_tensors) {
        TT_FATAL(in_ref.buffer(), "Operand to concat needs to be allocated in a buffer on device.");
        TT_FATAL(in_ref.device(), "Operand to concat needs to be on device.");
        TT_FATAL(in_ref.device() == first_input.device(), "Operands to concat need to be on the same device.");
        TT_FATAL(in_ref.get_layout() == first_input.get_layout(), "All Tensors should have same layouts.");
        TT_FATAL(in_ref.get_dtype() == first_input.get_dtype(), "All Tensors should have same dtypes.");
        tt::tt_metal::Shape curr_shape = in_ref.get_legacy_shape();
        TT_FATAL(curr_shape.rank() == shape_first.rank(), "Input tensor ranks must be equal");
        curr_shape[this->dim] = 0;
        TT_FATAL(curr_shape == shape_first, "concat tensors differ in shape across non-concat dimensions.");
        if (in_ref.get_layout() == Layout::ROW_MAJOR && this->dim == shape_first.rank() - 1) {
            TT_FATAL(
                (in_ref.get_legacy_shape()[this->dim] * in_ref.element_size()) % ADDRESS_ALIGNMENT == 0,
                "Current concat implementation requires aligned last dim when concatting on last dim");
        }
        TT_FATAL(in_ref.is_sharded() == shard_first, "All tensors must be sharded or all must be interleaved");
        if (shard_first) {
            TT_FATAL((in_ref.get_layout() == Layout::ROW_MAJOR), "Only row major supported for sharded concat.");
        }
    }
    if (shard_first) {
        TT_FATAL(this->dim == shape_first.rank() - 1, "Only width concat on sharded tensors");
        TT_FATAL(this->output_mem_config.is_sharded(), "Output must be sharded if input is sharded");
    }
}

std::vector<tt::tt_metal::Shape> Concat::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    tt::tt_metal::Shape shape_out = input_tensors[0].get_legacy_shape();
    shape_out[this->dim] = 0;
    for (const Tensor &in_ref : input_tensors) {
        tt::tt_metal::Shape curr_shape = in_ref.get_legacy_shape();
        shape_out[this->dim] += curr_shape[this->dim];
    }
    return {shape_out};
}

std::vector<Tensor> Concat::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const Tensor &ref_in_tensor = input_tensors.at(0);

    if (this->output_mem_config.is_sharded()) {
        return {create_sharded_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            ref_in_tensor.get_dtype(),
            ref_in_tensor.get_layout(),
            ref_in_tensor.device(),
            this->output_mem_config)};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, ref_in_tensor.get_dtype(), ref_in_tensor.get_layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Concat::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    switch (this->get_parallelization_strategy(input_tensors)) {
        case ConcatOpParallelizationStrategy::SHARDED_MULTI_CORE:
            return sharded_concat_multi_core(input_tensors, this->dim, output_tensors[0]);
        case ConcatOpParallelizationStrategy::MULTI_CORE:
            return concat_multi_core(input_tensors, this->dim, output_tensors[0]);
        case ConcatOpParallelizationStrategy::SINGLE_CORE:
        default: return concat_single_core(input_tensors, this->dim, output_tensors[0]);
    };
}

Tensor concat(std::vector<Tensor> &input_tensors, const std::int64_t dim, const MemoryConfig &output_mem_config) {
    TT_FATAL(input_tensors.size() > 0, "need 1 or more tensors");
    if (input_tensors.size() == 1) {
        return AutoFormat::move_tensor_to_mem_config(input_tensors[0], output_mem_config);
    }
    uint32_t ref_rank = input_tensors[0].get_legacy_shape().rank();
    uint32_t normalized_dim = input_tensors[0].get_legacy_shape().get_normalized_index(dim);

    if (input_tensors[0].is_sharded()) {
        return operation::run(Concat{normalized_dim, output_mem_config}, {input_tensors}).at(0);
    } else {
        if (input_tensors[0].get_layout() == Layout::ROW_MAJOR && normalized_dim == ref_rank - 1) {
            for (const auto &input_tensor : input_tensors) {
                TT_FATAL(
                    (input_tensor.get_legacy_shape()[dim] * input_tensor.element_size()) % ADDRESS_ALIGNMENT == 0,
                    "Current concat implementation requires aligned last dim when concatting on last dim");
            }
        }
        Layout target_layout = Layout::TILE;
        for (const auto &input_tensor : input_tensors) {
            if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
                const auto &input_shape = input_tensor.get_legacy_shape();
                if (input_shape.rank() < 2 || input_shape[-2] % TILE_HEIGHT != 0 || input_shape[-1] % TILE_WIDTH != 0) {
                    target_layout = Layout::ROW_MAJOR;
                    break;
                }
            }
        }
        std::vector<FormatParams> input_format_params;
        input_format_params.reserve(input_tensors.size());
        for (const auto &input_tensor : input_tensors) {
            if (target_layout == Layout::ROW_MAJOR) {
                input_format_params.push_back(FormatParams{
                    .pad_shape = input_tensor.get_legacy_shape(), .pad_value = 0.0, .target_layout = target_layout});
            } else {
                Shape pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
                input_format_params.push_back(
                    FormatParams{.pad_shape = pad_shape, .pad_value = 0.0, .target_layout = target_layout});
            }
        }

        return operation::run_with_autoformat(
                   Concat{normalized_dim, output_mem_config}, {input_tensors}, {input_format_params}, {target_layout})
            .at(0);
    }
}

}  // namespace tt_metal

}  // namespace tt
