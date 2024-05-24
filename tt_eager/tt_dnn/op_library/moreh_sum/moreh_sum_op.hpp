// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

inline std::tuple<uint32_t, uint32_t, uint32_t> extract_spatial_dims(const Shape &shape) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t W = shape[-1];
    uint32_t H = shape[-2];

    uint32_t other_dims_product = 1;
    for (auto i = 0; i < rank - 2; ++i) {
        other_dims_product *= shape[i];
    }

    return {W, H, other_dims_product};
}

struct MorehSum {
    int64_t dim;
    MemoryConfig output_mem_config;
    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;

    static constexpr auto attribute_names = std::forward_as_tuple("dim", "output_mem_config");
    const auto attribute_values() const { return std::forward_as_tuple(this->dim, this->output_mem_config); }
};

operation::ProgramWithCallbacks moreh_sum_nc_impl(const Tensor &input, const Tensor &output, int64_t dim);
// revised from reduce_op
operation::ProgramWithCallbacks moreh_sum_w_impl(const Tensor &a, const Tensor &output);
operation::ProgramWithCallbacks moreh_sum_h_impl(const Tensor &a, const Tensor &output);

Tensor moreh_sum(
    const Tensor &input,
    std::vector<int64_t> &dims,
    const std::optional<const Tensor> output = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
