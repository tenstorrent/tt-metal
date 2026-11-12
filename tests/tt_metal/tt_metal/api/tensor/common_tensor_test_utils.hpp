// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <string>

#include <gtest/gtest.h>

#include <tt-metalium/shape.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace test_utils {

// Every MX DataType. MX tensors are TILE-only and can only be placed on a Quasar device.
inline constexpr std::array kMxDataTypes = {
    tt::tt_metal::DataType::MXFP8_E4M3,
    tt::tt_metal::DataType::MXFP8_E5M2,
    tt::tt_metal::DataType::MXFP6_E2M3,
    tt::tt_metal::DataType::MXFP6_E3M2,
    tt::tt_metal::DataType::MXFP4,
    tt::tt_metal::DataType::MXINT8,
    tt::tt_metal::DataType::MXINT4,
    tt::tt_metal::DataType::MXINT2,
};

// gtest parameter name for a DataType-parametrized test, e.g. "MXFP8_E4M3".
std::string dtype_param_name(const ::testing::TestParamInfo<tt::tt_metal::DataType>& info);

// Allocates a runtime MeshTensor for the given shape/layout, performs a host->device->host
// byte round-trip, and asserts the read-back data and padded shape match expectations.
void test_tensor_on_device(
    const tt::tt_metal::Shape& input_shape,
    const tt::tt_metal::TensorLayout& layout,
    tt::tt_metal::distributed::MeshDevice& device);

// Same as above, but spins up its own single-device unit mesh for the round-trip.
void test_tensor_on_device(const tt::tt_metal::Shape& input_shape, const tt::tt_metal::TensorLayout& layout);

}  // namespace test_utils
