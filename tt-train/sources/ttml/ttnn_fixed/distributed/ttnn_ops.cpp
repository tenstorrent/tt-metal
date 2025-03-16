// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_ops.hpp"

#include <core/ttnn_all_includes.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/tensor/tensor.hpp>

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
    const auto& storage = std::get<tt::tt_metal::DeviceStorage>(tensor.get_storage());
    const auto num_tensor_buffers = storage.specs.size();

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

    // ttnn::SmallVector<uint32_t> shape{tensor_shape[0], tensor_shape[1], tensor_shape[2], tensor_shape[3]};
    // shape[dim] = split_size_per_device;

    // ttnn::Tensor scattered_tensor = tt::tt_metal::allocate_tensor_on_mesh(
    //     ttnn::TensorSpec(ttnn::Shape(shape), tensor.get_tensor_spec().tensor_layout()), current_device);


    MeshContainer<ttnn::SmallVector<uint32_t>> start_args;
    MeshContainer<ttnn::SmallVector<uint32_t>> end_args;
    size_t idx = 0;
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(device_grid_shape)) {
        start[dim] = split_size_per_device * idx;
        end[dim] = split_size_per_device * (idx + 1);
        start_args.at(coord) = start;
        end_args.at(coord) = end;
        ++idx;
    }    

  auto  scattered_tensor = ttnn::slice(tensor, start_args, end_args, stride, std::nullopt);

    // Brainstorming:
    // 1. Append 'mesh' to the name - confusing to distinguish between slice and slice_mesh
    // ttnn::slice_mesh(...)

    // Existing version:
    // Not ideal: 
    // Need to specify `DropoutOp` and ttnn::prim::dropout twice,
    // Need to work with `DrouputAttrs`,
    // Need to specify `seed` that will be overridden anyways,
    // Hard to know which args can be overridden and which cannot.
    // 
    // return ttnn::launch_mesh_workload<DropoutOp>(
    //     [seed, prob, scale](const auto& coord, auto* device) -> DropoutAttrs {
    //         auto seed_offset = device->get_device(coord)->id();
    //         return DropoutAttrs{
    //             .output_dtype = DataType::BFLOAT16,
    //             .output_memory_config = MemoryConfig{},
    //             .seed = seed + seed_offset,
    //             .prob = prob,
    //             .scale = scale};
    //     },
    //     ttnn::prim::dropout,
    //     input_tensor,
    //     prob,
    //     scale,
    //     seed,
    //     DataType::BFLOAT16);


    // size_t idx = 0;
    // for (const auto& tensor_shard : ttnn::distributed::get_device_tensors(tensor)) {
    //     start[dim] = split_size_per_device * idx;
    //     end[dim] = split_size_per_device * (idx + 1);

    //     ttnn::slice(tensor_shard, start, end, stride, std::nullopt, scattered_tensors[idx]);
    //     ++idx;
    // }
    return scattered_tensor;
}

}  // namespace ttml::ttnn_fixed::distributed
