// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_op.hpp"

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {

inline Shape compute_output_shape(const Shape &input_shape, int64_t dim) {
    const auto input_rank = static_cast<decltype(dim)>(input_shape.rank());
    auto output_shape = input_shape;
    auto output_padding = input_shape.padding();

    if (dim == input_rank - 1) {
        output_shape[input_rank - 1] = TILE_WIDTH;
        output_padding[input_rank - 1] = Padding::PadDimension{0, TILE_WIDTH - 1};
    } else if (dim == input_rank - 2) {
        output_shape[input_rank - 2] = TILE_HEIGHT;
        output_padding[input_rank - 2] = Padding::PadDimension{0, TILE_HEIGHT - 1};
    } else {
        output_shape[dim] = 1;
    }

    return Shape(output_shape, output_padding);
}

inline Tensor create_output_tensor(const Tensor &input, int64_t dim) {
    const auto output_shape = compute_output_shape(input.shape(), dim);
    const auto &output = create_device_tensor(output_shape, input.dtype(), Layout::TILE, input.device());
    return std::move(output);
}

}  // namespace

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
    auto floored_p = std::floor(p);
    auto decimal = p - floored_p;
    const bool p_is_negative = floored_p < 0.0f;
    if (p_is_negative) {
        floored_p = -floored_p;
    }
    return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
}

void MorehNorm::validate(const std::vector<Tensor> &input_tensors) const {}

std::vector<Shape> MorehNorm::compute_output_shapes(const std::vector<Tensor> &) const { return {}; }

std::vector<Tensor> MorehNorm::create_output_tensors(const std::vector<Tensor> &) const { return {}; }

Tensor moreh_norm(const Tensor &input, float p, std::variant<int64_t, std::vector<int64_t>> dims) {
    if (std::holds_alternative<int64_t>(dims)) {
        return moreh_norm_impl(input, p, std::get<int64_t>(dims));
    }
    return moreh_norm_impl(input, p, std::get<std::vector<int64_t>>(dims).at(0));
}

Tensor moreh_norm_impl(const Tensor &input, float p, int64_t dim) {
    const auto &output = create_output_tensor(input, dim);
    operation::run(MorehNorm{.p = p, .dim = dim}, {input, output});
    return std::move(output);
}

operation::ProgramWithCallbacks MorehNorm::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);

    const auto dim = this->dim;
    const auto input_rank = static_cast<decltype(dim)>(input.shape().rank());

    if (dim == input_rank - 1) {
        return moreh_norm_w_impl(input, this->p, output);
    } else if (dim == input_rank - 2) {
        return moreh_norm_h_impl(input, this->p, output);
    } else {
        return moreh_norm_other_impl(input, this->p, dim, output);
    }
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
