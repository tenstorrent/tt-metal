// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Direct coverage of the two host APIs the runtime binary reload captures an image with:
// experimental::ConfigureProgramWithoutLaunch lands a program's binary, runtime args and launch
// message on its cores without a go signal, and experimental::ReadKernelConfig reads that launch message
// back. The Python-side host-support test covers the same path through generic_op; this one calls
// the C++ API on one device.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"

namespace tt::tt_metal {

namespace {

constexpr uint32_t MARKER = 0xC0FFEE01;
// One data-movement kernel that writes its second runtime arg to the L1 address in its first.
constexpr const char* MARKER_KERNEL = "tests/tt_metal/tt_metal/test_kernels/misc/write_l1_marker.cpp";

uint32_t read_word(IDevice* device, const CoreCoord& core, uint32_t addr) {
    std::vector<uint32_t> words;
    detail::ReadFromDeviceL1(device, core, addr, sizeof(uint32_t), words);
    return words.at(0);
}

}  // namespace

// Slow dispatch only: the fixture skips otherwise.
TEST_F(MeshDeviceFixture, TensixConfigureProgramWithoutLaunchInstallsButDoesNotRun) {
    for (const auto& mesh : devices_) {
        IDevice* device = mesh->get_devices().at(0);
        const CoreCoord core{0, 0};
        // Nothing is allocated in this test, so the allocator's base is free scratch.
        const auto addr = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
        std::vector<uint32_t> zero{0};
        detail::WriteToDeviceL1(device, core, addr, zero);

        Program program = CreateProgram();
        KernelHandle kernel = CreateKernel(
            program,
            MARKER_KERNEL,
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program, kernel, core, {addr, MARKER});

        // Configured: the launch message names a binary, but the kernel has not run.
        experimental::ConfigureProgramWithoutLaunch(device, program);
        EXPECT_EQ(read_word(device, core, addr), 0u) << "configure-only must not run the kernel";
        const auto cfg = experimental::ReadKernelConfig(device, core);
        EXPECT_NE(cfg.enables, 0u) << "the launch message enables the configured kernel's RISC";
        EXPECT_GT(*std::max_element(cfg.kernel_text_size.begin(), cfg.kernel_text_size.end()), 0u)
            << "the configured program's binary is on the core";
        EXPECT_EQ(cfg.kernel_text_offset.size(), cfg.kernel_text_size.size());

        // Configuring again on the same device is allowed: it overwrites the previous configuration.
        experimental::ConfigureProgramWithoutLaunch(device, program);
        EXPECT_EQ(read_word(device, core, addr), 0u);
        const auto again = experimental::ReadKernelConfig(device, core);
        EXPECT_EQ(again.kernel_config_base, cfg.kernel_config_base);
        EXPECT_EQ(again.kernel_text_offset, cfg.kernel_text_offset);

        // The same program, launched for real, runs.
        detail::LaunchProgram(device, program);
        EXPECT_EQ(read_word(device, core, addr), MARKER) << "the same program launched normally writes the marker";
    }
}

}  // namespace tt::tt_metal
