/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <optional>

#include "tt_dnn/op_library/operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_sgd_(
    const Tensor& param_in,
    const Tensor& grad,
    std::optional<const Tensor> momentum_buffer_in,
    const Tensor& param_out,
    std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const CoreRange core_range);

struct MorehSGD {
    float lr;
    float momentum;
    float dampening;
    float weight_decay;
    bool nesterov;
    bool momentum_initialized;
    const CoreRange core_range;  // unused for now

    void validate(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names =
        std::make_tuple("lr", "momentum", "dampening", "weight_decay", "nesterov", "momentum_initialized");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->lr),
            std::cref(this->momentum),
            std::cref(this->dampening),
            std::cref(this->weight_decay),
            std::cref(this->nesterov),
            std::cref(this->momentum_initialized));
    }
};

void moreh_sgd(
    const Tensor &param_in,
    const Tensor &grad,
    std::optional<const Tensor> momentum_buffer_in,
    const Tensor &param_out,
    std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized);

}  // namespace primary
}  // namespace operations
}  // namespace tt
