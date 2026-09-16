// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <enchantum/enchantum.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>

#include "common/mesh_dispatch_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "jit_build/build_env_manager.hpp"

namespace tt::tt_metal {

// Device-free JIT-build fixture: the shared mesh device is a mock Blackhole, so every compile
// is pure host-side JIT and no silicon is touched.
//
// Lives in the named namespace: gtest TEST_F classes derive from it with external linkage, and
// an anonymous-namespace base trips -Werror=subobject-linkage under gcc Unity builds
// (merge-queue build-sweeps).
class MockBlackholeMeshDispatchFixture : public MeshDispatchFixture {
protected:
    // Mock mode must be registered BEFORE the base fixture opens its shared devices — and that
    // happens at suite scope (MeshDispatchFixture::SetUpTestSuite), not in SetUp. Overriding the
    // suite hooks keeps the whole suite off silicon.
    static void SetUpTestSuite() {
        experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
        MeshDispatchFixture::SetUpTestSuite();
    }
    static void TearDownTestSuite() {
        MeshDispatchFixture::TearDownTestSuite();
        experimental::disable_mock_mode();
    }

    // Build state for one of the kernel's processors (compute: 0=unpack, 1=math, 2=pack;
    // data-movement: kernel.get_kernel_processor_type(0)). Core and class are derived from
    // the kernel, so this works for any kernel type.
    const JitBuildState& kernel_build_state(const Kernel& kernel, uint32_t processor_id) {
        distributed::MeshDevice* device = devices_.at(0).get();
        const auto& hal = MetalContext::instance(kernel.get_context_id()).hal();
        const uint32_t core_idx = hal.get_programmable_core_type_index(kernel.get_kernel_programmable_core_type());
        const uint32_t class_idx = enchantum::to_underlying(kernel.get_kernel_processor_class());
        return BuildEnvManager::get_instance(kernel.get_context_id())
            .get_kernel_build_state(device->build_id(), core_idx, class_idx, processor_id);
    }
};

}  // namespace tt::tt_metal
