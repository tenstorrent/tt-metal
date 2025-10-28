// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "tt-metalium/tensor/types.hpp"
#include <tt-metalium/common/queue_id.hpp>  //  TODO: Do we need this here?
#include <tt_stl/optional_reference.hpp>
#include <tt-metalium/tensor/layout/layout.hpp>

namespace tt::tt_metal {
class Tensor;
class MemoryConfig;
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace tt::tt_metal::tensor_ops {

Tensor tensor_to_device(
    const Tensor& input_tensor,
    distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const MemoryConfig> mem_config,
    std::optional<QueueId> cq_id);

Tensor tensor_to_layout(const Tensor& input_tensor, Layout target_layout);

Tensor tensor_cpu(const Tensor& input_tensor, bool blocking, std::optional<QueueId> cq_id);

void tensor_print(const Tensor& input_tensor);

Tensor tensor_pad(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value);

Tensor tensor_unpad(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end);

Tensor tensor_pad_to_tile(const Tensor& input_tensor, float pad_value);

Tensor tensor_unpad_from_tile(const Tensor& input_tensor, const tt::tt_metal::Shape& output_tensor_shape);

Tensor tensor_reshape(const Tensor& input_tensor, const tt::tt_metal::Shape& new_shape);
Tensor tensor_reshape(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

}  // namespace tt::tt_metal::tensor_ops
