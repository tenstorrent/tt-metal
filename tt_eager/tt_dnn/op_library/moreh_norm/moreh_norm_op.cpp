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

void MorehNorm::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto &input = input_tensors.at(0);

    check_tensor(input, "moreh_norm", "input");

    const auto &output = output_tensors.at(0);
    if (output.has_value()) {
        check_tensor(output, "moreh_norm", "output");
    }
}

std::vector<Shape> MorehNorm::compute_output_shapes(const std::vector<Tensor> &) const { return {}; }

std::vector<Tensor> MorehNorm::create_output_tensors(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors
) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    auto input = input_tensors.at(0);
    auto device = input.device();
    const auto output_shape = compute_output_shape(input.get_legacy_shape(), this->dim);

    return {create_device_tensor(output_shape, input.get_dtype(), input.get_layout(), device, this->memory_config)};
}

Tensor moreh_norm(
    const Tensor &input,
    float p,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const std::optional<const Tensor> output,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    if (dim == std::nullopt) {
        std::vector<int64_t> dims(input.get_legacy_shape().rank());
        std::iota(dims.begin(), dims.end(), 0);
        dim = std::make_optional(dims);
    }

    if (std::holds_alternative<int64_t>(dim.value())) {
        const auto d = std::get<int64_t>(dim.value());
        return moreh_norm_impl(input, p, d, output, memory_config, compute_kernel_config);
    }

    auto dims = std::get<std::vector<int64_t>>(dim.value());

    if (dims.empty()) {
        std::vector<int64_t> all_dims(input.get_legacy_shape().rank());
        std::iota(all_dims.begin(), all_dims.end(), 0);
        dims = all_dims;
    }

    if (dims.size() == 1) {
        const auto d = dims[0];
        return moreh_norm_impl(input, p, d, output, memory_config, compute_kernel_config);
    }

    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
    const auto innermost_dim = dims[0];
    const auto outermost_dim = dims[dims.size() - 1];

    auto tmp_output =
        moreh_norm_impl(input, p, innermost_dim, std::nullopt, memory_config, compute_kernel_config);

    using idx_t = decltype(dims.size());
    for (idx_t idx = 1; idx < dims.size() - 1; ++idx) {
        tmp_output =
            moreh_norm_impl(tmp_output, p, dims[idx], std::nullopt, memory_config, compute_kernel_config);
    }

    return moreh_norm_impl(
        tmp_output, p, outermost_dim, output, memory_config, compute_kernel_config);
}

Tensor moreh_norm_impl(const Tensor &input, float p, int64_t dim,
    const std::optional<const Tensor> output,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config
) {
    auto device = input.device();

    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input}))};
    operation::launch_op(
        [p, dim, memory_config, kernel_config_val](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehNorm{.p = p, .dim = dim,
                    .memory_config = memory_config.value_or(input_tensors.at(0).memory_config()),
                    .compute_kernel_config = kernel_config_val},
                    input_tensors,
                    optional_input_tensors,
                    optional_output_tensors);
        },
        {input},
        output_tensors,
        {},
        {output});

    return output_tensors.at(0);
}

operation::ProgramWithCallbacks MorehNorm::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = output_tensors.at(0);

    const auto dim = this->dim;
    const auto input_rank = static_cast<decltype(dim)>(input.get_legacy_shape().rank());

    if (dim == input_rank - 1) {
        return moreh_norm_w_impl(input, this->p, output, this->compute_kernel_config);
    } else if (dim == input_rank - 2) {
        return moreh_norm_h_impl(input, this->p, output, this->compute_kernel_config);
    } else {
        return moreh_norm_other_impl(input, this->p, dim, output, this->compute_kernel_config);
    }
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
