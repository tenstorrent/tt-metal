// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehCumSum {
    int64_t dim;
    bool flip;
    void validate(const std::vector<Tensor> &inputs) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &inputs) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &inputs) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;
    stl::reflection::Attributes attributes() const;
    static constexpr auto attribute_names = std::make_tuple("dim", "flip");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->dim), std::cref(this->flip)); }
};

operation::ProgramWithCallbacks moreh_cumsum_nc(const Tensor &input, const Tensor &output, const int64_t &dim, const bool &flip);

Tensor moreh_cumsum_backward(const Tensor &output_grad, const Tensor &input_grad, const int64_t &dim);

Tensor moreh_cumsum(const Tensor &input, const Tensor &output, const int64_t &dim);

}  // namespace primary

}  // namespace operations

}  // namespace tt
