// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

struct MorehAdam {
    bool inplace;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    uint32_t step;
    bool amsgrad;

    MemoryConfig output_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;

    static constexpr auto attribute_names = std::make_tuple(
        "inplace",
        "lr",
        "beta1",
        "beta2",
        "eps",
        "weight_decay",
        "step",
        "amsgrad",
        "output_mem_config",
        "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::ref(this->inplace),
            std::ref(this->lr),
            std::ref(this->beta1),
            std::ref(this->beta2),
            std::ref(this->eps),
            std::ref(this->weight_decay),
            std::ref(this->step),
            std::ref(this->amsgrad),
            std::ref(this->output_mem_config),
            std::cref(this->compute_kernel_config));
    }
};

operation::ProgramWithCallbacks moreh_adam_(
    const Tensor &param,
    const Tensor &grad,
    const Tensor &exp_avg,
    const Tensor &exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    uint32_t step,
    bool amsgrad,
    const DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<std::reference_wrapper<const Tensor>> max_exp_avg_sq = std::nullopt);

std::vector<std::optional<Tensor>> moreh_adam(
    const Tensor &param,
    const Tensor &grad,
    const Tensor &exp_avg,
    const Tensor &exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    uint32_t step,
    bool amsgrad,
    const std::optional<std::reference_wrapper<const Tensor>> max_exp_avg_sq = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
