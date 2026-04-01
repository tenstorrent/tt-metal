// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_dispatch_program_factory.hpp"
#include "moe_dispatch_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::moe_dispatch {

struct MoeDispatchDeviceOperation {
    using operation_attributes_t = MoeDispatchParams;
    using tensor_args_t = MoeDispatchTensorArgs;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;
    using program_factory_t = std::variant<MoeDispatchMeshWorkloadFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::moe_dispatch

namespace ttnn::prim {

std::vector<ttnn::Tensor> ttml_moe_dispatch(
    const ttnn::Tensor& sorted_hidden,
    const ttnn::Tensor& w_up,
    uint32_t cluster_axis,
    const std::vector<std::vector<uint32_t>>& expert_offsets_per_device,
    const std::vector<std::vector<uint32_t>>& expert_counts_per_device,
    uint32_t E_local);

}  // namespace ttnn::prim
