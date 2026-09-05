// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The kernel-exit epilogue requires an idle NoC: a non-posted atomic still in flight
// when kernel_main returns races the next kernel's counter re-init. These tests pin
// that contract for the multicast-atomic credit shape used by persistent service and
// mcast-receiver kernels -- issue the credit, then return.

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <cstdint>
#include "debug_tools_fixture.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {
namespace {

constexpr const char* kKernel = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_atomic_at_exit.cpp";
constexpr const char* kExpected = "missing NOC non-posted atomics flushed barrier";

class NonPostedAtomicAtExitFixture : public MeshWatcherFixture {
protected:
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    IDevice* device{nullptr};
    std::uint32_t l1_unreserved_base{0};

    void SetUp() override {
        if (MetalContext::instance().rtoptions().watcher_assert_disabled()) {
            GTEST_SKIP() << "This test requires watcher asserts to be enabled";
        }
        MeshWatcherFixture::SetUp();
        if (arch_ == tt::ARCH::QUASAR) {
            // The Quasar kernel epilogue does not yet run the NoC-idle asserts.
            GTEST_SKIP() << "Kernel-exit NoC-idle asserts are not enabled on Quasar";
        }
        mesh_device = devices_[0];
        device = mesh_device->get_devices()[0];
        l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    }

    // Multicasts a non-posted atomic increment from (0,0) to every other storage-grid
    // core, then returns. `drain` selects the trailing atomic barrier.
    void Run(std::uint32_t drain) {
        const CoreCoord grid = device->compute_with_storage_grid_size();
        ASSERT_GE(grid.x, 2u);
        const CoreCoord sender{0, 0};
        const CoreRange dests(CoreCoord(1, 0), CoreCoord(grid.x - 1, grid.y - 1));
        const std::uint32_t num_dests = (grid.x - 1) * grid.y;

        std::vector<std::uint32_t> zero{0};
        for (const auto& c : dests) {
            tt::tt_metal::detail::WriteToDeviceL1(device, c, l1_unreserved_base, zero);
        }

        const CoreCoord start = device->worker_core_from_logical_core(dests.start_coord);
        const CoreCoord end = device->worker_core_from_logical_core(dests.end_coord);

        Program program;
        auto kernel = CreateKernel(
            program,
            kKernel,
            sender,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, kernel, sender, {l1_unreserved_base, start.x, start.y, end.x, end.y, num_dests, drain});

        distributed::MeshWorkload workload;
        const distributed::MeshCoordinate zero_coord{0, 0};
        const distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};
        workload.add_program(device_range, std::move(program));
        try {
            RunProgram(mesh_device, workload);
        } catch (const std::runtime_error& e) {
            log_info(tt::LogTest, "Caught exception: {}", e.what());
        }
    }

    std::string WaitForException(std::chrono::milliseconds timeout) {
        const auto start = std::chrono::steady_clock::now();
        std::string exception;
        while (std::chrono::steady_clock::now() - start < timeout) {
            exception = MetalContext::instance().watcher_server()->exception_message();
            if (!exception.empty()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return exception;
    }
};

// Control: draining with noc_async_atomic_barrier keeps the NoC idle at exit.
TEST_F(NonPostedAtomicAtExitFixture, DrainedMulticastCredit) {
    Run(/*drain=*/1);
    EXPECT_EQ(WaitForException(std::chrono::milliseconds(1500)), "");
}

// Returning with the multicast atomic still in flight trips the epilogue assert.
// The assert halts the core; the fixture reopens the device for the next test.
TEST_F(NonPostedAtomicAtExitFixture, UndrainedMulticastCredit) {
    Run(/*drain=*/0);
    const std::string exception = WaitForException(std::chrono::milliseconds(5000));
    EXPECT_NE(exception.find(kExpected), std::string::npos) << "actual: " << exception;
}

}  // namespace
}  // namespace tt::tt_metal
