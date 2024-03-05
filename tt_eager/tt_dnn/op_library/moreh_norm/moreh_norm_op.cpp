// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_op.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
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

inline void check_tensor(const Tensor &tensor, const std::string &op_name) {
    TT_ASSERT(tensor.get_layout() == Layout::TILE, fmt::format("{} only supports tiled layout.", op_name));
    TT_ASSERT(tensor.get_dtype() == DataType::BFLOAT16, fmt::format("{} only supports bfloat16.", op_name));
    TT_ASSERT(
        tensor.storage_type() == StorageType::DEVICE, fmt::format("Operands to {} need to be on device!", op_name));
    TT_ASSERT(
        tensor.buffer() != nullptr, fmt::format("Operands to {} need to be allocated in buffers on device!", op_name));
}

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

inline Tensor create_output_tensor(const Tensor &input, int64_t dim, const MemoryConfig &output_mem_config) {
    const auto output_shape = compute_output_shape(input.get_legacy_shape(), dim);
    const auto &output =
        create_device_tensor(output_shape, input.get_dtype(), Layout::TILE, input.device(), output_mem_config);
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

void MorehNorm::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);

    check_tensor(input, "moreh_norm");
    check_tensor(output, "moreh_norm");
}

std::vector<Shape> MorehNorm::compute_output_shapes(const std::vector<Tensor> &) const { return {}; }

std::vector<Tensor> MorehNorm::create_output_tensors(const std::vector<Tensor> &) const { return {}; }

[[maybe_unused]] Tensor moreh_norm(
    const Tensor &input,
    float p,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const std::optional<std::reference_wrapper<const Tensor>> output,
    const MemoryConfig &output_mem_config) {
    if (dim == std::nullopt) {
        std::vector<int64_t> dims(input.get_legacy_shape().rank());
        std::iota(dims.begin(), dims.end(), 0);
        dim = std::make_optional(dims);
    }

    if (std::holds_alternative<int64_t>(dim.value())) {
        const auto d = std::get<int64_t>(dim.value());
        if (output.has_value()) {
            return moreh_norm_impl(input, p, d, output->get());
        }
        return moreh_norm_impl(input, p, d, create_output_tensor(input, d, output_mem_config));
    }

    auto dims = std::get<std::vector<int64_t>>(dim.value());

    if (dims.empty()) {
        std::vector<int64_t> all_dims(input.get_legacy_shape().rank());
        std::iota(all_dims.begin(), all_dims.end(), 0);
        dims = all_dims;
    }

    if (dims.size() == 1) {
        const auto d = dims[0];
        if (output.has_value()) {
            return moreh_norm_impl(input, p, d, output->get());
        }
        return moreh_norm_impl(input, p, d, create_output_tensor(input, d, output_mem_config));
    }

    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
    const auto innermost_dim = dims[0];
    const auto outermost_dim = dims[dims.size() - 1];

    auto tmp_output =
        moreh_norm_impl(input, p, innermost_dim, create_output_tensor(input, innermost_dim, output_mem_config));

    using idx_t = decltype(dims.size());
    for (idx_t idx = 1; idx < dims.size() - 1; ++idx) {
        tmp_output =
            moreh_norm_impl(tmp_output, p, dims[idx], create_output_tensor(tmp_output, dims[idx], output_mem_config));
    }

    if (output.has_value()) {
        return moreh_norm_impl(tmp_output, p, outermost_dim, output->get());
    }
    return moreh_norm_impl(
        tmp_output, p, outermost_dim, create_output_tensor(tmp_output, outermost_dim, output_mem_config));
}

Tensor moreh_norm_impl(const Tensor &input, float p, int64_t dim, const Tensor &output) {
    operation::run(MorehNorm{.p = p, .dim = dim}, {input, output});
    return output;
}

operation::ProgramWithCallbacks MorehNorm::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);

    const auto dim = this->dim;
    const auto input_rank = static_cast<decltype(dim)>(input.get_legacy_shape().rank());

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
