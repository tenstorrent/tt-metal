// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {
class IDevice;
class Shape;
class TensorLayout;
}  // namespace tt::tt_metal

namespace test_utils {
// All tests must pass an explicit device. The previous no-device overload silently called
// MeshDevice::create_unit_mesh(0) per call, which defeats the suite-shared device pattern;
// callers should derive from a *Shared fixture (e.g. TTNNUnitMeshCQSharedFixture) and pass
// `device_` instead.
void test_tensor_on_device(
    const ttnn::Shape& input_shape,
    const tt::tt_metal::TensorLayout& layout,
    tt::tt_metal::distributed::MeshDevice* device);
}  // namespace test_utils
