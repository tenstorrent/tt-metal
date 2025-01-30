// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "trivial_ttnn_ops.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_dim(const tt::tt_metal::Tensor& t, uint32_t dim) {
    return sum_ttnn(t, dim, /* keepdim */ true);
}

tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t) {
    return sum_over_dim(t, /* dim */ 0);
}

// Stable log-softmax implementation
tt::tt_metal::Tensor log_softmax(const tt::tt_metal::Tensor& t, int dim) {
    auto t_max = ttnn::max(t, dim, /* keepdim */ true);
    auto t_sub_max = ttnn::subtract(t, t_max);

    auto t_sub_max_exp = ttnn::exp(t_sub_max);
    auto t_sum_over_dim = sum_over_dim(t_sub_max_exp, dim);

    auto log_t_sum_over_dim = ttnn::log(t_sum_over_dim);
    return ttnn::subtract(t_sub_max, log_t_sum_over_dim);
}

// Stable softmax implementation
// ttnn::softmax also exists, but it is not stable (even after max subtraction optimization)
tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim) {
    return ttnn::softmax(
        t,
        /* dim */ dim,
        /*memory_config */ std::nullopt,
        ttml::core::ComputeKernelConfig::softmax(),
        /*stable*/ true);
}

tt::tt_metal::Tensor divide(const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b) {
    auto inv_b = ttnn::reciprocal(/* queue_id */ 0, b);
    return ttnn::multiply(a, inv_b);
}

tt::tt_metal::Tensor mean_moreh(const tt::tt_metal::Tensor& t, int dim, bool keep_dim) {
    auto res = ttnn::moreh_mean(
        t,
        dim,
        keep_dim,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
    return res;
}
tt::tt_metal::Tensor mean_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim) {
    return ttnn::mean(t, dim, keep_dim, std::nullopt, core::ComputeKernelConfig::precise());
}

tt::tt_metal::Tensor sum_moreh(const tt::tt_metal::Tensor& t, int dim, bool keep_dim) {
    auto res = ttnn::moreh_sum(
        t,
        dim,
        keep_dim,
        std::nullopt,
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
    return res;
}
tt::tt_metal::Tensor sum_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim) {
    return ttnn::sum(t, dim, keep_dim, std::nullopt, core::ComputeKernelConfig::precise());
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

    auto tensor_shape = tensor.get_shape();
    auto tensor_rank = tensor_shape.rank();
    if (tensor_rank != 4U) {
        throw std::logic_error("Scatter supports only 4D tensors");
    }
    auto split_axis_size = tensor_shape[dim];
    if (split_axis_size % num_devices != 0) {
        throw std::logic_error("Slice dimension should be divisible by number of devices");
    }
    auto split_size_per_device = split_axis_size / num_devices;
    if (split_size_per_device % 32 != 0) {
        throw std::logic_error(fmt::format(
            "ttnn::slice does not support output dimension that is not divisible by 32."
            "Requested output dimension: {}",
            split_size_per_device));
    }

    SmallVector<uint32_t> start{0, 0, 0, 0};
    SmallVector<uint32_t> end{tensor_shape[0], tensor_shape[1], tensor_shape[2], tensor_shape[3]};
    SmallVector<uint32_t> step{1U, 1U, 1U, 1U};

    std::vector<tt::tt_metal::Tensor> scattered_tensors;
    scattered_tensors.reserve(num_tensor_buffers);
    for (size_t device_index = 0; device_index < num_tensor_buffers; ++device_index) {
        auto device = storage.get_buffer_for_device_id(device_index)->device();
        auto tensor_on_device =
            tt::tt_metal::Tensor(storage.get_buffer_for_device(device), storage.get_tensor_spec_for_device(device));

        start[dim] = split_size_per_device * device_index;
        end[dim] = split_size_per_device * (device_index + 1);

        auto sliced_tensor = ttnn::slice(tensor_on_device, start, end, step);
        scattered_tensors.push_back(sliced_tensor);
    }
    auto distributed_tensor = ttnn::distributed::create_multi_device_tensor(
        scattered_tensors, ttnn::StorageType::MULTI_DEVICE, storage.strategy);
    return distributed_tensor;
}

}  // namespace ttml::ttnn_fixed
