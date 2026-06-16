// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/float8.hpp>

#include <tt_stl/assert.hpp>

#include <memory>

/**
 * Buffer allocation and DataType dispatch utilities for Runtime Tensors.
 *
 * These remain exported in the public API area as a transient state, pending dispersal into the
 * public Runtime Tensor APIs (see tensor_apis.hpp / host_tensor.hpp) or relocation to tt_metal
 * internals. The data-conversion helpers that previously lived here are now private to tt_metal
 * (tt_metal/impl/tensor/tensor_impl_private.hpp).
 */

namespace tt::tt_metal::tensor_impl {

std::shared_ptr<distributed::MeshBuffer> allocate_device_buffer(
    distributed::MeshDevice* mesh_device, const TensorSpec& tensor_spec);

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec);

// Empty structs to facilitate Tensor template logic.
struct bfloat4_b {};
struct bfloat8_b {};

// Utility to convert runtime DataType to compile-time constant and dispatch the function call
template <typename Func, typename... Args>
auto dispatch(DataType dtype, Func&& func, Args&&... args) {
    switch (dtype) {
        case DataType::BFLOAT16:
            return (std::forward<Func>(func)).template operator()<bfloat16>(std::forward<Args>(args)...);
        case DataType::FLOAT32:
            return (std::forward<Func>(func)).template operator()<float>(std::forward<Args>(args)...);
        case DataType::INT32:
            return (std::forward<Func>(func)).template operator()<int32_t>(std::forward<Args>(args)...);
        case DataType::UINT32:
            return (std::forward<Func>(func)).template operator()<uint32_t>(std::forward<Args>(args)...);
        case DataType::UINT16:
            return (std::forward<Func>(func)).template operator()<uint16_t>(std::forward<Args>(args)...);
        case DataType::UINT8:
            return (std::forward<Func>(func)).template operator()<uint8_t>(std::forward<Args>(args)...);
        case DataType::BFLOAT8_B:
            return (std::forward<Func>(func)).template operator()<bfloat8_b>(std::forward<Args>(args)...);
        case DataType::BFLOAT4_B:
            return (std::forward<Func>(func)).template operator()<bfloat4_b>(std::forward<Args>(args)...);
        case DataType::FP8_E4M3:
            return (std::forward<Func>(func)).template operator()<float8_e4m3>(std::forward<Args>(args)...);
        default: TT_THROW("Unsupported data type");
    }
}

}  // namespace tt::tt_metal::tensor_impl
