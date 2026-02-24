// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "dispatch_core_common.hpp"
#include "hostdevcommon/common_values.hpp"
#include "impl/context/context_descriptor.hpp"
#include "impl/context/metallium_object.hpp"
#include "impl/device/mock_device_common.hpp"

namespace tt::tt_metal {

TEST(MetalliumObject, Physical) {
    auto descriptor = std::make_shared<ContextDescriptor>(
        /*num_cqs=*/1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, DEFAULT_WORKER_L1_SIZE);
    auto metallium_object = MetalliumObject(descriptor);
}

TEST(MetalliumObject, Mock) {
    // Multiple mocks can be created without a hang
    auto mock_path = experimental::get_mock_cluster_desc_for_config(tt::ARCH::WORMHOLE_B0, 1).value();
    auto descriptor = std::make_shared<ContextDescriptor>(
        /*num_cqs=*/1,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        DEFAULT_WORKER_L1_SIZE,
        DispatchCoreConfig{},
        tt::stl::Span<const std::uint32_t>{},
        mock_path);
    auto metallium_object_1 = MetalliumObject(descriptor);
    auto metallium_object_2 = MetalliumObject(descriptor);
}

TEST(MetalliumObject, OnePhysicalMultipleMock) {
    // Check a physical can be created alongside multiple mocks without a hang
    auto descriptor = std::make_shared<ContextDescriptor>(
        /*num_cqs=*/1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, DEFAULT_WORKER_L1_SIZE);

    auto mock_path = experimental::get_mock_cluster_desc_for_config(tt::ARCH::BLACKHOLE, 2).value();
    auto mock_descriptor = std::make_shared<ContextDescriptor>(
        /*num_cqs=*/1,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        DEFAULT_WORKER_L1_SIZE,
        DispatchCoreConfig{},
        tt::stl::Span<const std::uint32_t>{},
        mock_path);

    auto metallium_object_1 = MetalliumObject(descriptor);
    auto metallium_object_2 = MetalliumObject(mock_descriptor);
    auto metallium_object_3 = MetalliumObject(mock_descriptor);
}

}  // namespace tt::tt_metal
