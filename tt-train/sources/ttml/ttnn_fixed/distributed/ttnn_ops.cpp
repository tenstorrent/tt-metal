// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_ops.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed::distributed {

tt::tt_metal::Tensor all_reduce(const tt::tt_metal::Tensor& tensor) {
    auto* current_device = &ttml::autograd::ctx().get_device();
    auto num_devices = current_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("All reduce should not be called for a single device case");
    }

    auto shape = tensor.get_logical_shape();
    if (shape.rank() != 4U) {
        throw std::logic_error("All reduce supports only 4D tensors");
    }

    auto reshaped_tensor = ttnn::reshape(tensor, core::create_shape({1, shape[0] * shape[1], shape[2], shape[3]}));
    auto gathered_tensor = ttnn::all_gather(reshaped_tensor, 0);

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

tt::tt_metal::Tensor scatter(const tt::tt_metal::Tensor& tensor, int dim) {
    auto* current_device = &ttml::autograd::ctx().get_device();
    auto num_devices = current_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("Scatter should not be called for a single device case");
    }

    auto device_grid_shape = current_device->shape();
    const auto& storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
    auto num_tensor_buffers = storage.num_buffers();

    if (num_devices != num_tensor_buffers) {
        throw std::logic_error(fmt::format(
            "Number of buffers should be equal to the number of devices. Tensor is not properly replicated."
            " Number of devices: {}, number of buffers: {}",
            num_devices,
            num_tensor_buffers));
    }

    auto tensor_shape = tensor.get_logical_shape();
    auto tensor_rank = tensor_shape.rank();
    if (tensor_rank != 4U) {
        throw std::logic_error(
            fmt::format("Scatter supports only 4D tensors. Shape {} Rank {}", tensor_shape, tensor_rank));
    }
    auto split_axis_size = tensor_shape[dim];
    if (split_axis_size % num_devices != 0) {
        throw std::logic_error(fmt::format(
            "Split axis size should be divisible by number of devices. Split axis size: {}, number of devices: {}",
            split_axis_size,
            num_devices));
    }
    auto split_size_per_device = split_axis_size / num_devices;
    if (split_size_per_device % 32 != 0) {
        throw std::logic_error(fmt::format(
            "ttnn::slice does not support output dimension that is not divisible by 32."
            "Requested output dimension: {}",
            split_size_per_device));
    }

    ttnn::SmallVector<uint32_t> start{0, 0, 0, 0};
    ttnn::SmallVector<uint32_t> end{tensor_shape[0], tensor_shape[1], tensor_shape[2], tensor_shape[3]};
    ttnn::SmallVector<uint32_t> stride{1U, 1U, 1U, 1U};

    std::vector<tt::tt_metal::Tensor> scattered_tensors;
    scattered_tensors.reserve(num_tensor_buffers);
    for (size_t device_index = 0; device_index < num_tensor_buffers; ++device_index) {
        auto device = storage.get_buffer_for_device_id(device_index)->device();
        auto tensor_on_device =
            tt::tt_metal::Tensor(storage.get_buffer_for_device(device), storage.get_tensor_spec_for_device(device));

        start[dim] = split_size_per_device * device_index;
        end[dim] = split_size_per_device * (device_index + 1);

        auto sliced_tensor = ttnn::slice(tensor_on_device, start, end, stride);
        scattered_tensors.push_back(sliced_tensor);
    }
    auto distributed_tensor = ttnn::distributed::create_multi_device_tensor(
        scattered_tensors, ttnn::StorageType::MULTI_DEVICE, storage.strategy);
    return distributed_tensor;
}

}  // namespace ttml::ttnn_fixed::distributed
