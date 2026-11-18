// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "common/mesh_dispatch_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/metal2_host_api/program_spec/program_spec_internal.hpp"
#include "jit_build/build_env_manager.hpp"
#include "llrt/get_platform_architecture.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {

// JIT-build fixture that follows the runner's architecture:
//  - Quasar runner (ttsim or silicon): opens the real Quasar device. JIT compile is a no-op on a mock
//    Quasar (Program::compile returns early), so a mock would test nothing.
//  - Any other runner (Wormhole, Blackhole, CPU-only): a mock Blackhole, so every compile is pure
//    host-side JIT and no silicon is touched.
// Kernels are compiled, never dispatched.
//
// Lives in the named namespace: gtest TEST_F classes derive from it with external linkage, and
// an anonymous-namespace base trips -Werror=subobject-linkage under gcc Unity builds
// (merge-queue build-sweeps).
class JitBuildDeviceFixture : public MeshDispatchFixture {
protected:
    // Device setup happens at suite scope (MeshDispatchFixture::SetUpTestSuite), so the mock must be
    // registered there, before the base fixture opens its shared devices.
    static void SetUpTestSuite() {
        suite_skip_reason().clear();
        // A local RunTimeOptions: probing through MetalContext::instance() would bring up the default
        // context (and its cluster) before the mock is registered.
        const llrt::RunTimeOptions rtoptions;
        if (runner_is_quasar(rtoptions)) {
            const auto target = rtoptions.get_target_device();
            if (target == tt::TargetDevice::Mock || target == tt::TargetDevice::Emule) {
                suite_skip_reason() =
                    "JIT compile is a no-op on a mock/emulated Quasar (Program::compile returns early); "
                    "run on a Quasar simulator or device";
                return;
            }
            MeshDispatchFixture::SetUpTestSuite();
            return;
        }
        experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
        MeshDispatchFixture::SetUpTestSuite();
    }
    static void TearDownTestSuite() {
        MeshDispatchFixture::TearDownTestSuite();
        experimental::disable_mock_mode();  // No-op when no mock was configured.
    }

    void SetUp() override {
        if (!suite_skip_reason().empty()) {
            GTEST_SKIP() << suite_skip_reason();
        }
        MeshDispatchFixture::SetUp();
    }
    void TearDown() override {
        if (!suite_skip_reason().empty()) {
            return;
        }
        MeshDispatchFixture::TearDown();
    }

    bool is_quasar() const { return devices_.at(0)->arch() == tt::ARCH::QUASAR; }

    // A second logical worker core besides {0, 0}, from the queried compute grid; nullopt on a 1-core grid.
    std::optional<CoreCoord> second_worker_core() const {
        const CoreCoord grid = devices_.at(0)->compute_with_storage_grid_size();
        if (grid.x > 1) {
            return CoreCoord{1, 0};
        }
        if (grid.y > 1) {
            return CoreCoord{0, 1};
        }
        return std::nullopt;
    }

    // HAL processor type of each binary the kernel builds (BH compute: unpack, math, pack; Quasar
    // compute: one per TRISC slot in use). Pass these to kernel_build_state().
    static std::vector<uint32_t> binary_processor_types(const Kernel& kernel) {
        std::vector<uint32_t> types;
        for (int i = 0; i < kernel.expected_num_binaries(); i++) {
            types.push_back(kernel.get_kernel_processor_type(i));
        }
        return types;
    }

    // Metal 2.0 Program built from a spec but not compiled, so tests can drive compile themselves.
    Program build_program_from_spec(const ProgramSpec& spec) {
        return BuildProgramFromSpec(*devices_.at(0), spec);
    }

    // Build state for one of the kernel's processors (a value from binary_processor_types()). Core
    // and class are derived from the kernel, so this works for any kernel type and architecture.
    const JitBuildState& kernel_build_state(const Kernel& kernel, uint32_t processor_id) {
        distributed::MeshDevice* device = devices_.at(0).get();
        const auto& hal = MetalContext::instance(kernel.get_context_id()).hal();
        const uint32_t core_idx = hal.get_programmable_core_type_index(kernel.get_kernel_programmable_core_type());
        const uint32_t class_idx = enchantum::to_underlying(kernel.get_kernel_processor_class());
        return MetalContext::instance(kernel.get_context_id())
            .build_env_manager()
            .get_kernel_build_state(device->build_id(), core_idx, class_idx, processor_id);
    }

private:
    // True only when the runner is a Quasar simulator or device (or a mock/emulated Quasar, which the
    // caller skips). Built to leave non-Quasar runners as they were before this fixture existed:
    //  - Simulator: decided from the soc_descriptor.yaml a Quasar simulator must ship, never by loading
    //    the simulator image (get_simulator_architecture would load libttsim for a WH/BH .so).
    //  - Any probe failure (bad simulator path, unreadable mock descriptor, mixed-arch PCI cards)
    //    falls back to the mock Blackhole, as before.
    static bool runner_is_quasar(const llrt::RunTimeOptions& rtoptions) {
        try {
            switch (rtoptions.get_target_device()) {
                case tt::TargetDevice::Simulator: {
                    const std::string soc_desc = find_simulator_soc_descriptor_path(rtoptions.get_simulator_path());
                    return !soc_desc.empty() &&
                           umd::SocDescriptor::get_arch_from_soc_descriptor_path(soc_desc) == tt::ARCH::QUASAR;
                }
                case tt::TargetDevice::Mock:
                case tt::TargetDevice::Emule:
                case tt::TargetDevice::Silicon: return get_platform_architecture(rtoptions) == tt::ARCH::QUASAR;
                default: return false;
            }
        } catch (const std::exception&) {
            return false;
        }
    }

    static std::string& suite_skip_reason() {
        static std::string reason;
        return reason;
    }
};

// GTEST_SKIP()s the current test on Quasar with the given reason.
#define JIT_BUILD_SKIP_ON_QUASAR(reason)        \
    do {                                        \
        if (this->is_quasar()) {                \
            GTEST_SKIP() << "Quasar: " << (reason); \
        }                                       \
    } while (0)

}  // namespace tt::tt_metal
