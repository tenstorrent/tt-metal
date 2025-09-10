// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_ops.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed::distributed {

tt::tt_metal::Tensor all_gather(const tt::tt_metal::Tensor& tensor, int dim) {
    auto* current_device = &ttml::autograd::ctx().get_device();
    auto num_devices = current_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("All gather should not be called for a single device case");
    }
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();

    return ttnn::experimental::all_gather_async(
        tensor,
        dim,
        ccl_resources.get_all_gather_semaphore(),
        /* num_links */ 1,
        /* memory_config */ std::nullopt,
        ttnn::ccl::Topology::Linear,
        /* subdevice_id */ std::nullopt,
        /* use_optimal_ccl_for_llama */ false,
        /* barrier_semaphore */ ccl_resources.get_barrier_semaphore());
}

tt::tt_metal::Tensor all_reduce(const tt::tt_metal::Tensor& tensor) {
    auto* current_device = &ttml::autograd::ctx().get_device();
    auto num_devices = current_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("All reduce should not be called for a single device case");
    }

    auto shape = tensor.logical_shape();
    if (shape.rank() != 4U) {
        throw std::logic_error("All reduce supports only 4D tensors");
    }

    auto reshaped_tensor = ttnn::reshape(tensor, ttnn::Shape({1, shape[0] * shape[1], shape[2], shape[3]}));
    auto gathered_tensor = all_gather(reshaped_tensor, 0);
    auto reduced_tensor = ttnn::moreh_sum(
        gathered_tensor,
        0,
        /* keep_dim */ true,
        /* output */ std::nullopt,
        /* memory_config */ std::nullopt,
        core::ComputeKernelConfig::precise());
    reduced_tensor = ttnn::reshape(reduced_tensor, shape);
    return reduced_tensor;
}

tt::tt_metal::Tensor reduce_scatter(const tt::tt_metal::Tensor& tensor, int dim) {
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    return ttnn::experimental::reduce_scatter_minimal_async(
        tensor,
        /* persistent_output_buffers */ std::nullopt,
        dim,
        ccl_resources.get_reduce_scatter_semaphores(),
        ccl_resources.get_barrier_semaphore(),
        /* num_links */ 1U,
        /* memory_config */ std::nullopt,
        /* intermediate_memory_config */ std::nullopt,
        ttnn::ccl::Topology::Linear);
}

}  // namespace ttml::ttnn_fixed::distributed
