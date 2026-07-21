// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/shape.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace test_utils {

// Allocates a runtime MeshTensor for the given shape/layout, performs a host->device->host
// byte round-trip, and asserts the read-back data and padded shape match expectations.
void test_tensor_on_device(
    const tt::tt_metal::Shape& input_shape,
    const tt::tt_metal::TensorLayout& layout,
    tt::tt_metal::distributed::MeshDevice& device);

// Same as above, but spins up its own single-device unit mesh for the round-trip.
void test_tensor_on_device(const tt::tt_metal::Shape& input_shape, const tt::tt_metal::TensorLayout& layout);

}  // namespace test_utils
