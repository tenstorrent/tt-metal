// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"

#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         MorehSum
////////////////////////////////////////////////////////////////////////////
namespace {

Tensor _moreh_sum(
    const Tensor& input,
    const int64_t& dim,
    const std::optional<const Tensor>& output,
    const MemoryConfig& output_mem_config) {
    return operation::run(MorehSum{.dim = dim, .output_mem_config = output_mem_config}, {input}, {}, {output}).at(0);
}
}  // namespace

void MorehSum::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_ASSERT((dim >= 0 && dim <= 3), "dim should be 0 - 3");
    const auto& input = input_tensors.at(0);
    auto& output = output_tensors.at(0);

    auto input_shape = input.get_legacy_shape();
    auto input_shape_wo_padding = input.get_legacy_shape().without_padding();

    if (output.has_value()) {
        const auto& output_shape = output.value().get_legacy_shape();
        const auto& output_shape_wo_padding = output.value().get_legacy_shape().without_padding();

        if (dim == 0 || dim == 1) {
            input_shape[dim] = 1;
            input_shape_wo_padding[dim] = 1;
        } else {
            input_shape[dim] = TILE_HEIGHT;
            input_shape_wo_padding[dim] = 1;
        }

        for (int i = 0; i < input_shape.rank(); ++i) {
            TT_ASSERT(input_shape[i] == output_shape[i]);
            TT_ASSERT(input_shape_wo_padding[i] == output_shape_wo_padding[i]);
        }
    }
}

std::vector<Shape> MorehSum::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.get_legacy_shape();

    auto output_shape = input_shape;
    auto padding = output_shape.padding();
    switch (dim) {
        case 0:
        case 1: output_shape[dim] = 1; break;
        case 2:
            output_shape[dim] = TILE_HEIGHT;
            padding[dim] = Padding::PadDimension{0, 31};
            break;
        case 3:
            output_shape[dim] = TILE_WIDTH;
            padding[dim] = Padding::PadDimension{0, 31};
            break;
    }

    output_shape = Shape(output_shape, padding);
    log_debug(LogOp, "{}:{} dim {}", __func__, __LINE__, dim);
    log_debug(LogOp, "{}:{} output_shape {}", __func__, __LINE__, output_shape);
    return {output_shape};
}

std::vector<Tensor> MorehSum::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        log_debug(LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {output_tensors.at(0).value()};
    }

    log_debug(LogOp, "{}:{} create output tensor", __func__, __LINE__);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehSum::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    TT_ASSERT((dim >= 0 && dim <= 3), "dim should be 0 - 3");
    auto& input = inputs.at(0);
    auto& output = outputs.at(0);

    if (dim == 0 || dim == 1) {
        return moreh_sum_nc_impl(input, output, dim);
    } else if (dim == 2) {
        return moreh_sum_h_impl(input, output);
    } else {
        return moreh_sum_w_impl(input, output);
    }
}

Tensor moreh_sum(
    const Tensor& input,
    std::vector<int64_t>& dims,
    const std::optional<const Tensor> output,
    const MemoryConfig& output_mem_config) {
    // reduce for all dims
    if (dims.empty()) {
        dims = {0, 1, 2, 3};
    }

    std::vector<int64_t> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());

    auto temp_input = input;
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        log_debug(LogOp, "{}:{} dim {}", __func__, __LINE__, sorted_dims[i]);
        auto temp_output = _moreh_sum(temp_input, sorted_dims[i], std::nullopt, output_mem_config);
        temp_input = temp_output;
    }
    log_debug(LogOp, "{}:{} dim {}", __func__, __LINE__, sorted_dims.front());
    return _moreh_sum(temp_input, sorted_dims.front(), output, output_mem_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
