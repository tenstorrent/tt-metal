// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "all_gather_minimal_matmul_outer_dim_async_device_operation_types.hpp"
#include "all_gather_minimal_matmul_outer_dim_async_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal::ops::all_gather_minimal_matmul_outer_dim_async {
struct AllGatherMinimalMatmulOuterDimAsyncDeviceOperation {
    using operation_attributes_t = AllGatherMinimalMatmulOuterDimAsyncParams;
    using tensor_args_t = AllGatherMinimalMatmulOuterDimAsyncInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<AllGatherMinimalMatmulOuterDimAsyncProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::vector<tt::tt_metal::TensorTopology> compute_output_topologies(
        const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttml::metal::ops::all_gather_minimal_matmul_outer_dim_async

namespace ttnn::prim {
ttnn::Tensor ttml_all_gather_minimal_matmul_outer_dim_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gathered_tensor,
    const ttnn::Tensor& persistent_gather_buffer,
    const std::vector<ttnn::GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> cluster_axis,
    ttnn::ccl::Topology topology,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t num_links,
    bool transpose_b);
}
