// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run:
// $ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.OOB_Tensor_*"

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

// Sanity check for the L1 out-of-bounds-tensor sanitizer in
// __emule_local_l1_to_ptr. Allocates one small L1 buffer, then has the kernel
// translate an L1 address that is comfortably below the buffer (still well
// within the L1 mmap and well above l1_unreserved_base, so it is not inside
// any system region and not inside any allocated tensor). The sanitizer is
// expected to abort with an Out-of-Bounds Write ASAN message.
TEST_F(MeshDeviceFixture, OOB_Tensor_Gap_L1_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // One small L1 buffer near the top of L1 (allocator is top-down for L1).
    constexpr uint32_t buffer_size = 1024;
    auto buf = Buffer::create(device, buffer_size, buffer_size, BufferType::L1);

    // Pick a target that sits 64 KB below the buffer's start. That lands in
    // user-allocatable L1 (above l1_unreserved_base on a fresh device with a
    // single small buffer), but no allocation covers it.
    constexpr uint32_t gap_distance = 64 * 1024;
    uint32_t oob_addr = static_cast<uint32_t>(buf->address()) - gap_distance;

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* bad_ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(addr);
            *bad_ptr = 0x666;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {oob_addr});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Out-of-Bounds Write: Attempted to access address.*not part of any allocated tensor.*");
}

// Sanity check for the DRAM out-of-bounds-tensor sanitizer inside the
// extern "C" __emule_dram_ptr bridge function. Allocates one DRAM buffer,
// then has the kernel translate a DRAM offset that lives above
// dram_unreserved_base but does not fall inside the allocated buffer. The
// sanitizer is expected to abort with an Out-of-Bounds Write ASAN message
// naming DRAM.
TEST_F(MeshDeviceFixture, OOB_Tensor_Gap_DRAM_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // One small DRAM buffer. DRAM allocates bottom-up from dram_unreserved_base
    // by default, so the buffer's address is close to that base.
    constexpr uint32_t buffer_size = 1024;
    auto buf = Buffer::create(device, buffer_size, buffer_size, BufferType::DRAM);

    // Target a DRAM offset 1 MB above the buffer — well above the buffer's
    // end and above dram_unreserved_base, but not inside any allocated DRAM
    // tensor.
    constexpr uint32_t gap_distance = 1024 * 1024;
    uint32_t oob_addr = static_cast<uint32_t>(buf->address()) + gap_distance;

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* bad_ptr = (volatile uint32_t*)__emule_dram_ptr(addr);
            *bad_ptr = 0x777;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {oob_addr});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Out-of-Bounds Write: Attempted to access DRAM address.*not part of any allocated tensor.*");
}

// Precision control (death): the host-poke fallback is NOT a blanket whitelist. A
// write just PAST the poked [addr, addr+size) region (still in the gap, in no
// tensor) must still abort — a kernel overrunning a raw-L1 region is a real bug
// the check must keep catching. Kept among the death tests (before the non-death
// controls below) so its EXPECT_DEATH forks from a single-threaded parent.
TEST_F(MeshDeviceFixture, OOB_Tensor_HostPoke_JustPast_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t buffer_size = 1024;
    auto anchor = Buffer::create(device, buffer_size, buffer_size, BufferType::L1);
    constexpr uint32_t gap_distance = 64 * 1024;
    uint32_t poke_addr = static_cast<uint32_t>(anchor->address()) - gap_distance;

    std::vector<uint32_t> host_data(32, 0xABCDABCDu);  // poked region = [poke_addr, poke_addr+128)
    detail::WriteToDeviceL1(device, logical_core, poke_addr, host_data);

    // One word past the poked region: not in the poke range, not in any tensor.
    uint32_t past_addr = poke_addr + static_cast<uint32_t>(host_data.size() * sizeof(uint32_t));

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(addr);
            *ptr = 0x666;
        }
    )";
    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {past_addr});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Out-of-Bounds Write: Attempted to access address.*not part of any allocated tensor.*");
}

// Positive control (L1): resolving and writing an address INSIDE an allocated L1
// buffer must NOT abort. Guards the OOB check from flagging legitimate in-bounds
// accesses (e.g. an off-by-one in the range comparison or the offset
// normalization).
TEST_F(MeshDeviceFixture, OOB_Tensor_InBounds_L1_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t buffer_size = 1024;
    auto buf = Buffer::create(device, buffer_size, buffer_size, BufferType::L1);
    // A word comfortably inside the buffer.
    uint32_t in_addr = static_cast<uint32_t>(buf->address()) + 64;

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(addr);
            *ptr = 0x1234;
        }
    )";
    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {in_addr});

    // Must NOT abort — the address is inside an allocated tensor.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// Positive control (DRAM): resolving a DRAM offset INSIDE an allocated DRAM
// buffer via __emule_dram_ptr must NOT abort.
TEST_F(MeshDeviceFixture, OOB_Tensor_InBounds_DRAM_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t buffer_size = 1024;
    auto buf = Buffer::create(device, buffer_size, buffer_size, BufferType::DRAM);
    uint32_t in_addr = static_cast<uint32_t>(buf->address()) + 64;

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* ptr = (volatile uint32_t*)__emule_dram_ptr(addr);
            *ptr = 0x5678;
        }
    )";
    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {in_addr});

    // Must NOT abort — the offset is inside an allocated DRAM tensor.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

// Positive control for the host-poke fallback. Raw L1 that the host designated
// via WriteToDeviceL1 (no Buffer allocated) is valid data outside the tensor
// allocator; a kernel access into that exact [addr, addr+size) region must NOT
// abort even though it matches no live tensor. This guards the fallback scan in
// __emule_asan_check_oob_tensor (the DM-microbenchmark raw-L1 false-positive fix)
// — if that scan regresses, legitimate raw-L1 reads/writes start aborting and
// only the separate unit_tests_data_movement binary would notice.
TEST_F(MeshDeviceFixture, OOB_Tensor_HostPoke_Accept_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Anchor a valid L1 address, then poke a raw region 64 KB below it (in the
    // gap: above l1_unreserved_base, inside no allocated tensor) via
    // WriteToDeviceL1 — which registers the host-poke extent when ASAN is on.
    constexpr uint32_t buffer_size = 1024;
    auto anchor = Buffer::create(device, buffer_size, buffer_size, BufferType::L1);
    constexpr uint32_t gap_distance = 64 * 1024;
    uint32_t poke_addr = static_cast<uint32_t>(anchor->address()) - gap_distance;

    std::vector<uint32_t> host_data(32, 0xABCDABCDu);  // 128-byte poked region
    detail::WriteToDeviceL1(device, logical_core, poke_addr, host_data);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(addr);
            *ptr = 0x1234;   // inside the host-poked region -> must be accepted
        }
    )";
    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {poke_addr});

    // Must NOT abort — the address is inside a host-designated raw-L1 region.
    detail::LaunchProgram(device, program);
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
