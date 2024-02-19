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

struct Prod {
    int64_t dim;
    void validate(const std::vector<Tensor> &inputs) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &inputs) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &inputs) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;
    stl::reflection::Attributes attributes() const;
    static constexpr auto attribute_names = std::make_tuple("dim");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->dim)); }
};

operation::ProgramWithCallbacks prod_nc_format(const Tensor &input, const Tensor &output, int64_t dim);

Tensor prod_(
    const Tensor &input,
    std::optional<std::reference_wrapper<const Tensor>> output,
    const int64_t &dim,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor prod_nc(
    const Tensor &input,
    const Tensor &output,
    std::vector<int64_t> &dims,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
