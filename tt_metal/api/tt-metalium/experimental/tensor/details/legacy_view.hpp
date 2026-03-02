// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/shape.hpp>

namespace tt::tt_metal::tensor_impl {

// ======================================================================================
//                                  .view()
//        These maybe replaced by dedicated view types, See: #38093
// ======================================================================================

HostTensor view(const HostTensor& tensor, const Shape& new_logical_shape, const Shape& new_padded_shape);

MeshTensor view(const MeshTensor& tensor, const Shape& new_logical_shape, const Shape& new_padded_shape);

}  // namespace tt::tt_metal::tensor_impl
