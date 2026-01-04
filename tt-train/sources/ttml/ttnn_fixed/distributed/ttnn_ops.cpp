// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_ops.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed::distributed {

tt::tt_metal::Tensor all_gather(const tt::tt_metal::Tensor& tensor, int dim, std::optional<uint32_t> cluster_axis) {
    auto* mesh_device = &ttml::autograd::ctx().get_device();
    auto num_devices = mesh_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("All gather should not be called for a single device case");
    }
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(
        *mesh_device, /* cluster_axis */ cluster_axis);

    if (cluster_axis.has_value()) {
        // Use cluster_axis overload for 2D mesh
        return ttnn::experimental::all_gather_async(
            tensor,
            dim,
            cluster_axis.value(),
            *mesh_device,
            ttnn::ccl::Topology::Linear,
            ccl_resources.get_all_gather_semaphore(),
            /* persistent_output_tensor */ std::nullopt,
            /* memory_config */ std::nullopt,
            std::optional<size_t>(num_links),
            /* subdevice_id */ std::nullopt,
            /* use_optimal_ccl_for_llama */ false,
            /* barrier_semaphore */ ccl_resources.get_barrier_semaphore(),
            /* reverse_order */ false,
            /* sub_core_grid */ std::nullopt);
    } else {
        // Use original overload for 1D mesh or when cluster_axis is not specified
        return ttnn::experimental::all_gather_async(
            tensor,
            dim,
            ccl_resources.get_all_gather_semaphore(),
            num_links,
            /* memory_config */ std::nullopt,
            ttnn::ccl::Topology::Linear,
            /* subdevice_id */ std::nullopt,
            /* use_optimal_ccl_for_llama */ false,
            /* barrier_semaphore */ ccl_resources.get_barrier_semaphore());
    }
}

tt::tt_metal::Tensor all_reduce(const tt::tt_metal::Tensor& tensor, std::optional<uint32_t> cluster_axis) {
    auto* mesh_device = &ttml::autograd::ctx().get_device();
    auto num_devices = mesh_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("All reduce should not be called for a single device case");
    }

    auto shape = tensor.logical_shape();
    if (shape.rank() != 4U) {
        throw std::logic_error("All reduce supports only 4D tensors");
    }

    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    auto all_reduce_barrier_semaphores = ccl_resources.get_all_reduce_barrier_semaphores();
    auto all_gather_semaphores = ccl_resources.get_all_gather_semaphore();
    auto reduce_scatter_semaphores = ccl_resources.get_reduce_scatter_semaphores();

    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(
        *mesh_device, /* cluster_axis */ cluster_axis);

    if (cluster_axis.has_value()) {
        // Use cluster_axis overload for 2D mesh
        return ttnn::experimental::all_reduce_async(
            tensor,
            cluster_axis,
            *mesh_device,
            /* barrier_semaphores */ std::nullopt,
            /* rs_global_semaphores */ std::nullopt,
            /* ag_global_semaphores */ std::nullopt,
            ttnn::operations::reduction::ReduceType::Sum,
            /* memory_config */ std::nullopt,
            ttnn::ccl::Topology::Linear,
            std::optional<size_t>(num_links),
            /* worker_subdevice_id_opt */ std::nullopt);
    } else {
        // Use original overload for 1D mesh
        return ttnn::experimental::all_reduce_async(
            tensor,
            num_devices,
            all_reduce_barrier_semaphores,
            reduce_scatter_semaphores,
            all_gather_semaphores,
            ttnn::operations::reduction::ReduceType::Sum,
            /* memory_config */ std::nullopt,
            /* topology */ ttnn::ccl::Topology::Linear,
            /* num_preferred_links */ num_links);
    }
}

tt::tt_metal::Tensor reduce_scatter(const tt::tt_metal::Tensor& tensor, int dim, std::optional<uint32_t> cluster_axis) {
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources();
    auto& mesh_device = ttml::autograd::ctx().get_device();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(
        mesh_device, /* cluster_axis */ cluster_axis);
    return ttnn::experimental::reduce_scatter_minimal_async(
        tensor,
        /* persistent_output_buffers */ std::nullopt,
        dim,
        ccl_resources.get_reduce_scatter_semaphores(),
        ccl_resources.get_barrier_semaphore(),
        num_links,
        /* memory_config */ std::nullopt,
        /* intermediate_memory_config */ std::nullopt,
        ttnn::ccl::Topology::Linear,
        /* subdevice_id */ std::nullopt,
        /* cluster_axis */ cluster_axis);
}

}  // namespace ttml::ttnn_fixed::distributed
