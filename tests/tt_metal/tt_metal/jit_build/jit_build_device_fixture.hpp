// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "common/mesh_dispatch_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/metal2_host_api/program_spec/program_spec_internal.hpp"
#include "impl/program/program_impl.hpp"
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
        // A local RunTimeOptions: probing through MetalContext::instance() would bring up the default
        // context (and its cluster) before the mock is registered.
        const llrt::RunTimeOptions rtoptions;
        const bool quasar = runner_is_quasar(rtoptions);
        const auto target = rtoptions.get_target_device();
        skip_suite_ = quasar && (target == tt::TargetDevice::Mock || target == tt::TargetDevice::Emule);
        if (skip_suite_) {
            return;
        }
        if (!quasar) {
            experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
        }
        MeshDispatchFixture::SetUpTestSuite();
    }
    static void TearDownTestSuite() {
        MeshDispatchFixture::TearDownTestSuite();
        experimental::disable_mock_mode();  // No-op when no mock was configured.
    }

    void SetUp() override {
        if (skip_suite_) {
            GTEST_SKIP() << "JIT compile is a no-op on a mock/emulated Quasar (Program::compile returns early); "
                            "run on a Quasar simulator or device";
        }
        MeshDispatchFixture::SetUp();
    }
    void TearDown() override {
        if (skip_suite_) {
            return;
        }
        MeshDispatchFixture::TearDown();
    }

    bool is_quasar() const { return devices_.at(0)->arch() == tt::ARCH::QUASAR; }

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

    // Builds `kernel` alone on core {0, 0} through Metal 2.0, compiles it, and returns it. Plain compile is
    // the first step of MakeProgramFromSpec's compile_and_allocate; allocation is not needed to read recipes.
    std::shared_ptr<Kernel> compile_single_kernel_spec(KernelSpec kernel) {
        const KernelSpecName name = kernel.unique_id;
        Program program = build_program_from_spec(ProgramSpec{
            .name = "jit_build_single_kernel",
            .kernels = {std::move(kernel)},
            .work_units = {WorkUnitSpec{.name = "main", .kernels = {name}, .target_nodes = CoreCoord{0, 0}}},
        });
        program.impl().compile(devices_.at(0).get());
        return program.impl().get_kernel(0);
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

    // The kernel's exported compile recipe cflags for one of its processors (a binary_processor_types() value).
    std::string recipe_cflags(const std::shared_ptr<Kernel>& kernel, uint32_t processor_id) {
        return kernel_build_state(*kernel, processor_id).export_target_recipe(kernel.get()).cflags;
    }

private:
    // True when the runner is a Quasar simulator, device, or mock/emulated Quasar (which the caller skips).
    //  - Simulator: decided from the soc_descriptor.yaml a Quasar simulator must ship, never by loading
    //    the simulator image (get_simulator_architecture would load libttsim for a WH/BH .so).
    //  - Any probe failure (bad simulator path, unreadable mock descriptor, mixed-arch PCI cards)
    //    falls back to the mock Blackhole.
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

    // Set per suite by SetUpTestSuite: the runner is a mock/emulated Quasar, where compile is a no-op.
    inline static bool skip_suite_ = false;
};

// GTEST_SKIP()s the current test on Quasar with the given reason.
#define JIT_BUILD_SKIP_ON_QUASAR(reason)        \
    do {                                        \
        if (this->is_quasar()) {                \
            GTEST_SKIP() << "Quasar: " << (reason); \
        }                                       \
    } while (0)

}  // namespace tt::tt_metal
