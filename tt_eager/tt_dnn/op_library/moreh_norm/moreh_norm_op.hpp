// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p);

operation::ProgramWithCallbacks moreh_norm_h_impl(const Tensor &input, float p, const Tensor &output);
operation::ProgramWithCallbacks moreh_norm_w_impl(const Tensor &input, float p, const Tensor &output);
operation::ProgramWithCallbacks moreh_norm_other_impl(const Tensor &input, float p, int64_t dim, const Tensor &output);

struct MorehNorm {
    float p;
    int64_t dim;
    // MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    // static constexpr auto attribute_names = std::make_tuple("p", "dim", "output_mem_config");
    // const auto attribute_values() const {
    //     return std::make_tuple(std::cref(this->p), std::cref(this->dim), std::cref(this->output_mem_config));
    // }

    static constexpr auto attribute_names = std::make_tuple("p", "dim");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->p), std::cref(this->dim)); }
};

Tensor moreh_norm(
    const Tensor &input,
    float p,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim = std::nullopt,
    const std::optional<std::reference_wrapper<const Tensor>> output = std::nullopt);

Tensor moreh_norm_impl(const Tensor &input, float p, int64_t dim);

// [[maybe_unused]] Tensor moreh_norm_impl(const Tensor &input, float p, int64_t dim, const Tensor &output);

}  // namespace primary

}  // namespace operations

}  // namespace tt
