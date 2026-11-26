// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "minimal_matmul_reduce_scatter_async_device_operation_types.hpp"
#include "minimal_matmul_reduce_scatter_async_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async {
struct MinimalMatmulReduceScatterAsyncDeviceOperation {
    using operation_attributes_t = MinimalMatmulReduceScatterAsyncParams;
    using tensor_args_t = MinimalMatmulReduceScatterAsyncInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<MinimalMatmulReduceScatterAsyncProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async

namespace ttnn::prim {
ttnn::Tensor ttml_minimal_matmul_reduce_scatter_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& persistent_buffer,
    const std::vector<ttnn::GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> num_links,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    bool transpose_a,
    bool transpose_b,
    bool bidirectional);
}
