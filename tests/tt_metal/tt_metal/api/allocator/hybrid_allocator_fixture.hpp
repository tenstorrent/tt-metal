// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "tests/tt_metal/tt_metal/common/device_fixture.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

// Unit meshes with the HYBRID allocator, which is the only mode where per-core and lockstep
// allocations coexist and therefore the only one where either narrowing has an effect.
class HybridAllocatorTest : public MeshDeviceSingleCardBufferFixture {
protected:
    void SetUp() override {
        // Enable HYBRID allocator mode before device creation.
        setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", /*overwrite=*/1);

        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
            ids.push_back(id);
        }
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            ids, l1_small_size_, trace_region_size_, 1, dispatch_core_config, {}, DEFAULT_WORKER_L1_SIZE);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
        init_max_cbs();
    }

    void TearDown() override {
        MeshDeviceSingleCardBufferFixture::TearDown();
        unsetenv("TT_METAL_ALLOCATOR_MODE_HYBRID");
    }
};

// Safely above all alignment requirements: FreeListOpt internally uses DRAM alignment, which may
// be larger than L1's.
inline constexpr DeviceAddr HYBRID_TEST_PAGE_SIZE = 1024;

}  // namespace tt::tt_metal
