// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "all_gather_weight_minimal_matmul_async_device_operation_types.hpp"
#include "all_gather_weight_minimal_matmul_async_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {
struct AllGatherWeightMinimalMatmulAsyncDeviceOperation {
    using operation_attributes_t = AllGatherWeightMinimalMatmulAsyncParams;
    using tensor_args_t = AllGatherWeightMinimalMatmulAsyncInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<AllGatherWeightMinimalMatmulAsyncProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async

namespace ttnn::prim {
ttnn::Tensor ttml_all_gather_weight_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& persistent_weight_buffer,
    const std::vector<ttnn::GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> cluster_axis,
    ttnn::ccl::Topology topology,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t num_links,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel,
    bool transpose_b);
}
