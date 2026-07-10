// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Metadata_*"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/allocator.hpp>
#include "impl/context/metal_context.hpp"
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

// Program's static CB region (allocated upward from l1_unreserved_base)
// runs past the lowest L1 address occupied by an allocated buffer. The
// underlying detection is validate_circular_buffer_region in ProgramImpl; our
// wrapper in ConfigureDeviceWithProgram translates its TT_THROW into an
// ASAN-style abort for emule.
TEST_F(MeshDeviceFixture, Metadata_CB_Tensor_Clash_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto& mesh_device = this->devices_.at(0);
    auto* device = mesh_device->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    // Pin a low lowest_occupied_compute_l1_address by allocating a 1 MB L1
    // mesh buffer. The allocator places L1 buffers from the top of L1
    // downward, so this leaves only a small unreserved window below the
    // buffer for CBs to grow into.
    constexpr uint32_t l1_buffer_size = 1024 * 1024;  // 1 MB
    distributed::DeviceLocalBufferConfig l1_local_config{
        .page_size = l1_buffer_size, .buffer_type = BufferType::L1, .bottom_up = false};
    distributed::ReplicatedBufferConfig l1_buf_config{.size = l1_buffer_size};
    auto l1_buffer = distributed::MeshBuffer::create(l1_buf_config, l1_local_config, mesh_device.get());

    // CB sized to overrun the buffer's lower edge.
    Program program = CreateProgram();
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t cb_num_pages = 512;
    constexpr uint32_t cb_page_size = 2048;
    CircularBufferConfig cb_config =
        CircularBufferConfig(cb_num_pages * cb_page_size, {{cb_id, tt::DataFormat::Float16_b}})
            .set_page_size(cb_id, cb_page_size);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {}
    )";
    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(detail::LaunchProgram(device, program), ".*\\[ASAN ERROR\\] Metadata Overflow.*");
}

// Directly exercises the emule-only static KERNEL_CONFIG-window check
// (check_program_metadata_size). The Clash test above trips a DIFFERENT throw
// first — validate_circular_buffer_region, which needs an L1 tensor pinning the
// lowest occupied address — so the emule window check itself is never reached
// there. Here NO L1 tensor is allocated (so the CB-region validator passes) and
// the program's static config is inflated with runtime args until it exceeds the
// KERNEL_CONFIG window, tripping check_program_metadata_size.
//
// Reachability is arch-dependent: LaunchProgram runs finalize_offsets first,
// whose TT_FATAL bounds the config against the allocator's ring-buffer size
// (l1_unreserved_base - KERNEL_CONFIG). The emule check bounds against the HAL's
// DEFAULT_UNRESERVED - KERNEL_CONFIG window. The emule check is only INDEPENDENTLY
// reachable when the ring buffer is larger than that window (config can land in
// the band between them); otherwise the finalize FATAL fires first and this check
// is defensive. The test self-calibrates and SKIPs when the band is unusable.
TEST_F(MeshDeviceFixture, Metadata_KernelConfigWindow_Overflow_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    const auto& hal = MetalContext::instance().hal();

    uint32_t kc_base =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG));
    uint32_t default_unreserved =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED));
    uint32_t alloc_unreserved = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    uint32_t window_size = default_unreserved - kc_base;    // check_program_metadata_size bound
    uint32_t ringbuffer_size = alloc_unreserved - kc_base;  // finalize_offsets TT_FATAL bound

    // Aim ~1 KB past the window (so the check fires) while staying under the ring
    // buffer (so the finalize FATAL does not). Each runtime arg is 4 bytes of
    // KERNEL_CONFIG; the TENSIX rt-arg count ceiling is 4096 (validated at
    // SetRuntimeArgs).
    constexpr uint32_t kOverBy = 1024;
    constexpr uint32_t kMaxArgs = 4000;
    fprintf(
        stderr,
        "EMULE METADATA DIAG: kc=0x%x def_unres=0x%x alloc_unres=0x%x window=%u ring=%u\n",
        kc_base,
        default_unreserved,
        alloc_unreserved,
        window_size,
        ringbuffer_size);

    // Skip BEFORE any LaunchProgram so, when the band is unusable (the common
    // case — on WH/BH the allocator's l1_unreserved_base equals the HAL's
    // DEFAULT_UNRESERVED, so window == ring), no launch runs and the EXPECT_DEATH
    // below is never reached. That also keeps the parent single-threaded: a prior
    // non-death LaunchProgram would poison the EXPECT_DEATH fork (see the
    // per-process split notes in the regression runner).
    if (ringbuffer_size <= window_size + 2 * kOverBy) {
        ::unsetenv("TT_METAL_EMULE_ASAN");
        GTEST_SKIP() << "KERNEL_CONFIG window (" << window_size << ") >= the finalize ring buffer (" << ringbuffer_size
                     << ") on this arch: check_program_metadata_size is dominated by the "
                        "finalize TT_FATAL and is not independently reachable via a real program.";
    }
    // Minimal-kernel static config is ~0 on emule, so N runtime args ≈ N*4 bytes
    // of config; target just over the window. (Only reached on an arch where the
    // ring buffer exceeds the window — none today.)
    uint32_t n_args = (window_size + kOverBy) / static_cast<uint32_t>(sizeof(uint32_t));
    if (n_args > kMaxArgs) {
        ::unsetenv("TT_METAL_EMULE_ASAN");
        GTEST_SKIP() << "window (" << window_size << ") needs " << n_args
                     << " runtime args to overflow, above the TENSIX rt-arg ceiling; not reachable with this lever.";
    }

    Program program = CreateProgram();
    std::string minimal = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {}
    )";
    auto kernel = CreateKernelFromString(
        program,
        minimal,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::vector<uint32_t> args(n_args, 0u);
    SetRuntimeArgs(program, kernel, logical_core, args);

    EXPECT_DEATH(detail::LaunchProgram(device, program), ".*\\[ASAN ERROR\\] Metadata Overflow.*");

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// Positive control: a small CB that comfortably fits below the lowest occupied
// L1 address must configure and launch without aborting. Guards against the
// overflow check firing on well-sized programs.
//
// ORDERING: kept after the death tests above. A non-death LaunchProgram leaves
// the emule fiber worker pool alive in the parent; a later EXPECT_DEATH fork()s
// and the child inherits its locked state without the worker threads, hanging
// until the watchdog aborts. Keeping death tests first lets each fork cleanly.
// (Today the window-overflow death test SKIPs on WH/BH, but this ordering keeps
// the bare `Metadata_*` glob safe on any arch where it does fire.)
TEST_F(MeshDeviceFixture, Metadata_CB_Tensor_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};

    // A tiny 2-page CB leaves the vast majority of L1 free; no clash possible.
    Program program = CreateProgram();
    constexpr uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * 1024, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {}
    )";
    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Must NOT abort.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
