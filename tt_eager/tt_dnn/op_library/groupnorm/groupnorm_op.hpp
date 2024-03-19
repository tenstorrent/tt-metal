// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tt_eager/tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

/**
Ref: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
>>> input = torch.randn(20, 6, 10, 10)
>>> # Separate 6 channels into 3 groups
>>> m = nn.GroupNorm(3, 6)
>>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
>>> m = nn.GroupNorm(6, 6)
>>> # Put all 6 channels into a single group (equivalent with LayerNorm)
>>> m = nn.GroupNorm(1, 6)
>>> # Activating the module
>>> output = m(input)
*/

// FIXME: special case for group_size = 1 is only supported at this time
Tensor groupnorm(
    const Tensor& a,
    uint32_t group_size,
    float eps,
    std::optional<const Tensor> gamma = std::nullopt,
    std::optional<const Tensor> beta = std::nullopt,
    const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal


namespace operations {

    using namespace tt_metal;
namespace primary {
struct GroupNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    MathFidelity math_fidelity;
    DataType im_data_format;
    DataType out_data_format;
    bool inplace;

    tt::stl::reflection::Attributes attributes() const;
};

struct GroupNorm {
    float eps;
    uint32_t num_groups;
    MemoryConfig output_mem_config;
    tt::operations::primary::GroupNormShardedMultiCoreProgramConfig program_config;

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline Tensor groupnorm(const Tensor &a, const uint32_t num_groups, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, std::optional<const Tensor> input_mask = std::nullopt, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const GroupNormShardedMultiCoreProgramConfig& program_config = GroupNormShardedMultiCoreProgramConfig{}) {
    return operation::run(GroupNorm{.eps=eps, .num_groups=num_groups, .output_mem_config=output_mem_config, .program_config=program_config}, {a}, {gamma, beta, input_mask}).at(0);
}


}   // namespace primary
}   // namespace operations
}  // namespace tt
