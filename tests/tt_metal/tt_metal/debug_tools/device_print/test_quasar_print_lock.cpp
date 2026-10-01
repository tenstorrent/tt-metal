// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar DM print-lock checks that do not need the print server: the lock has a 64-byte cache
// line of its own, and a released lock reads as free to another hart's atomic.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"
#include "hostdev/device_print_common.h"
#include "impl/context/metal_context.hpp"
#include "llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {
namespace {

constexpr const char* kLineProbeKernel =
    "tests/tt_metal/tt_metal/test_kernels/device_print/dm_print_lock_line_probe.cpp";
constexpr const char* kReleaseProbeKernel =
    "tests/tt_metal/tt_metal/test_kernels/device_print/dm_print_lock_release_probe.cpp";

// The Quasar Tensix print region holds the TRISC sub-buffer first, then the DM one
// (tt_metal/hw/inc/internal/tt-2xx/quasar/device_print_mem.h).
constexpr uint32_t kQuasarTriscPrintBufferBytes = 3264;

constexpr uint32_t kLineDoneMarker = 0x4C4F434Bu;
constexpr uint32_t kNumHeaderWords = 16;
constexpr uint32_t kReleaseDoneMarker = 0x52454C53u;

const char* header_word_name(uint32_t i) {
    static const char* names[] = {"wpos", "rpos", "risc_state[0..3]", "risc_state[4..7]"};
    return i < 4 ? names[i] : "data";
}

class QuasarPrintLockFixture : public MeshDispatchFixture {
protected:
    static constexpr experimental::NodeCoord core = {0, 0};
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    uint32_t l1_unreserved_base{0};
    std::vector<uint32_t> result;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        if (arch_ != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "The cached-alias print lock is Quasar-only";
        }
        mesh_device_ = devices_[0];
        l1_unreserved_base = mesh_device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    }

    void run_kernel(
        const std::string& kernel_src,
        uint32_t num_threads,
        const std::vector<std::string>& arg_names,
        std::initializer_list<std::pair<std::string, uint32_t>> args,
        bool prints_enabled_build) {
        const experimental::KernelSpecName name{"print_lock_probe"};
        experimental::KernelSpec kernel_spec{
            .unique_id = name,
            .source = std::filesystem::path{kernel_src},
            .num_threads = num_threads,
            .runtime_arg_schema = {.runtime_arg_names = arg_names},
            .hw_config = experimental::DataMovementHardwareConfig{},
        };
        if (prints_enabled_build) {
            kernel_spec.compiler_options.defines = {{"DEBUG_PRINT_ENABLED", "1"}};
        }
        experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {name}, .target_nodes = core};
        experimental::ProgramSpec spec{.name = "quasar_print_lock", .kernels = {kernel_spec}, .work_units = {main_wu}};
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        experimental::ProgramRunArgs params;
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = name, .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, args)});
        experimental::SetProgramRunArgs(program, params);

        distributed::MeshWorkload workload;
        distributed::MeshCoordinate zero_coord{0, 0};
        workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
        RunProgram(mesh_device_, workload);
    }

    void zero_l1(uint32_t addr, uint32_t words) {
        std::vector<uint32_t> zeros(words, 0);
        slow_dispatch::WriteToL1(*mesh_device_, core, addr, zeros);
    }

    void read_l1(uint32_t addr, uint32_t words) {
        slow_dispatch::ReadFromL1(*mesh_device_, core, addr, words * sizeof(uint32_t), result);
        ASSERT_EQ(result.size(), words);
    }

    uint32_t dm_print_lock_addr() const {
        const auto& hal = MetalContext::instance().hal();
        return static_cast<uint32_t>(
                   hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DPRINT_BUFFERS)) +
               kQuasarTriscPrintBufferBytes + DEVICE_PRINT_QUASAR_LOCK_LINE_BYTES;
    }
};

// Writing the print lock's cache line back must not change the header words written through the
// uncached alias. Fails when the lock shares a line with them (padding removed in device_print_common.h).
TEST_F(QuasarPrintLockFixture, PrintLockLineIsolation) {
    const uint32_t report = (l1_unreserved_base + 63u) & ~63u;
    zero_l1(report, 32);

    run_kernel(kLineProbeKernel, 1, {"report_addr"}, {{"report_addr", report}}, false);

    read_l1(report, 2 + kNumHeaderWords);
    ASSERT_EQ(result[0], kLineDoneMarker) << "the probe kernel did not finish";
    for (uint32_t i = 0; i < kNumHeaderWords; i++) {
        EXPECT_EQ(result[2 + i], 0xA5A50000u + i)
            << header_word_name(i) << " (word " << i << ") was overwritten by the lock line's write-back";
    }
    EXPECT_EQ(result[1], 0u) << result[1] << " header words were overwritten by the lock line's write-back";

    // The lock's 64-byte line holds nothing but padding and the lock.
    const uint32_t lock_addr = dm_print_lock_addr();
    const uint32_t buffer_addr = lock_addr - DEVICE_PRINT_QUASAR_LOCK_LINE_BYTES;
    const uint32_t line_start = lock_addr & ~63u;
    EXPECT_GE(line_start, buffer_addr + 2 * sizeof(uint32_t) + 8) << "the lock's line reaches the header words";
    EXPECT_LE(line_start + 64u, buffer_addr + DEVICE_PRINT_QUASAR_AUX_BYTES) << "the lock's line reaches the data";
    read_l1(lock_addr, 1);
    EXPECT_EQ(result[0], 0u) << "the print lock was left taken";
}

// Thread 0 holds the print lock while thread 1 spins on it the way acquire_lock() does, then releases
// it through the production code; thread 1 must get the lock.
TEST_F(QuasarPrintLockFixture, PrintLockReleaseVisibleToOtherHart) {
    const uint32_t report = (l1_unreserved_base + 63u) & ~63u;
    const uint32_t flag = report + 64u;
    zero_l1(report, 32);
    const bool prints_on =
        MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint);

    run_kernel(
        kReleaseProbeKernel,
        2,
        {"report_addr", "flag_addr", "prints_off"},
        {{"report_addr", report}, {"flag_addr", flag}, {"prints_off", prints_on ? 0u : 1u}},
        true);

    read_l1(report, 5);
    ASSERT_EQ(result[0], kReleaseDoneMarker) << "thread 0 did not finish";
    ASSERT_NE(result[1], 2u) << "thread 1 timed out waiting for thread 0 to take the lock";
    EXPECT_NE(result[2], result[3]) << "both threads ran on the same hart";
    EXPECT_EQ(result[1], 1u) << "hart " << result[3] << " never saw the lock released by hart " << result[2] << " ("
                             << result[4] << " attempts)";
    log_info(tt::LogTest, "hart {} got the lock after {} attempts", result[3], result[4]);
}

}  // namespace
}  // namespace CMAKE_UNIQUE_NAMESPACE
