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
    // Bytes of L1 the allocator manages per core, from the top of L1 down; the rest above KERNEL_CONFIG
    // is the kernel-config window. A subclass may set it before SetUp.
    size_t worker_l1_size_ = DEFAULT_WORKER_L1_SIZE;

    void SetUp() override {
        // Enable HYBRID allocator mode before device creation.
        setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", /*overwrite=*/1);
        previous_allocator_mode_hybrid_ =
            tt::tt_metal::MetalContext::instance().rtoptions().get_allocator_mode_hybrid();
        tt::tt_metal::MetalContext::instance().rtoptions().set_allocator_mode_hybrid(true);

        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
            ids.push_back(id);
        }
        const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().resolve_dispatch_core_config();
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            ids, l1_small_size_, trace_region_size_, 1, dispatch_core_config, {}, worker_l1_size_);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
        init_max_dfbs();
    }

    void TearDown() override {
        MeshDeviceSingleCardBufferFixture::TearDown();
        unsetenv("TT_METAL_ALLOCATOR_MODE_HYBRID");
        tt::tt_metal::MetalContext::instance().rtoptions().set_allocator_mode_hybrid(previous_allocator_mode_hybrid_);
    }

private:
    bool previous_allocator_mode_hybrid_ = false;
};

// Safely above all alignment requirements: FreeListOpt internally uses DRAM alignment, which may
// be larger than L1's.
inline constexpr DeviceAddr HYBRID_TEST_PAGE_SIZE = 1024;

}  // namespace tt::tt_metal
