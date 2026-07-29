// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>

/**
 * Private buffer allocation helpers for Runtime Tensors.
 *
 * Not part of the installed public API. Prefer HostTensor::allocate_for_overwrite /
 * MeshTensor::allocate_on_device at call sites outside this directory.
 */

namespace tt::tt_metal::tensor_impl {

std::shared_ptr<distributed::MeshBuffer> allocate_device_buffer(
    distributed::MeshDevice* mesh_device, const TensorSpec& tensor_spec);

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec);

}  // namespace tt::tt_metal::tensor_impl
