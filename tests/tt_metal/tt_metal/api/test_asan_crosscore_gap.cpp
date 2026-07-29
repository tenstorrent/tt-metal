// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Coverage-gap probes for the emule L1 sanitizers, built as a controlled
// experiment around test_write_outside_tensor.cpp's
// OOB_Tensor_Gap_L1_SanityCheck.
//
// That test establishes the POSITIVE control: a *local* write to an L1 address
// that is (a) above l1_unreserved_base and (b) inside no allocated tensor
// aborts with "Out-of-Bounds Write".
//
// The tests here hold the bad address constant and vary exactly one factor at a
// time, to isolate which factor makes the check stop seeing it:
//
//   GapAddr_LocalWrite_Detected           same addr, local   -> DIES   (control)
//   GapAddr_CrossCoreNocWrite_NotDetected same addr, via NOC -> SILENT (gap 1)
//   GapAddr_McastWrite_NotDetected        same addr, mcast   -> SILENT (gap 1)
//   BelowUnreservedBase_LocalWrite_NotDetected  local, addr < base -> SILENT (gap 2)
//
// Gap 1 (cross-core): __emule_local_l1_to_ptr is the only chokepoint that runs
// the OOB/padding/semaphore checks, and cross-core traffic never reaches it —
// it goes through __emule_resolve_noc_addr / __emule_multicast_write, which do a
// core-map lookup and hand back a pointer with no liveness validation. The only
// cross-core checks are __emule_check_noc_{read,write}_alignment.
//
// Gap 2 (reserved region): __emule_asan_check_oob_tensor early-returns for
// l1_off < san.l1_unreserved_base ("below this is firmware/system, passes
// through" — docs/ASAN.md). On Blackhole the launch mailbox is
// MEM_MAILBOX_BASE=96 / MEM_MAILBOX_SIZE=12912, i.e. [0x60, 0x3270), entirely
// inside that unchecked band.
//
// Together the two gaps are why a Blaze cross-core Gather whose dst CB is not
// backed on the receiver can clobber the receiver's launch mailbox
// (run_mailbox=0xbf) with the sanitizers fully armed and silent.
//
// A "NotDetected" test PASSES when nothing aborts: an abort would kill the test
// process and gtest would report the crash. These are diagnostic probes of
// current behavior, not assertions that the behavior is correct.
//
// To run (all checks armed, single fiber worker so death tests don't fork a
// 64-worker pool):
//   TT_METAL_EMULE_ASAN=1 TT_EMULE_FIBER_WORKERS=1 \
//     build_emule/test/tt_metal/unit_tests_api \
//     --gtest_death_test_style=threadsafe --gtest_filter='*GapAddr*:*UnreservedBase*'

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <string>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

namespace {

constexpr uint32_t kBufSize = 1024;
constexpr uint32_t kGapDistance = 64 * 1024;

// The shared bad address: kGapDistance below a freshly allocated L1 buffer.
// L1 allocates top-down, so this lands above l1_unreserved_base but inside no
// allocation — identical construction to OOB_Tensor_Gap_L1_SanityCheck.
uint32_t gap_addr_from(const std::shared_ptr<Buffer>& buf) {
    uint32_t addr = static_cast<uint32_t>(buf->address()) - kGapDistance;
    return addr & ~0xFu;  // 16B-align so NOC alignment checks can't be what fires
}

}  // namespace

// POSITIVE CONTROL (mirrors OOB_Tensor_Gap_L1_SanityCheck, using the shared
// helper so the address is provably the same one the negative tests use).
// Local write to the gap -> Out-of-Bounds Write abort.
TEST_F(MeshDeviceFixture, GapAddr_LocalWrite_Detected) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    auto buf = Buffer::create(device, kBufSize, kBufSize, BufferType::L1);
    const uint32_t bad_addr = gap_addr_from(buf);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* p = (volatile uint32_t*)__emule_local_l1_to_ptr(addr);
            *p = 0x666;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {bad_addr});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Out-of-Bounds Write: Attempted to access address.*not part of any allocated tensor.*");
}

// GAP 1 — the decisive comparison. The SAME gap address, reached by a NOC write
// from (0,0) to a different core instead of locally. Nothing aborts: the
// cross-core path never reaches the OOB check.
TEST_F(MeshDeviceFixture, GapAddr_CrossCoreNocWrite_NotDetected) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord sender_logical = {0, 0};
    CoreCoord target_logical = {1, 1};
    Program program = CreateProgram();

    // src must sit inside a live extent so the *local* read side of the write
    // isn't what trips the check — we want the destination to be the only
    // suspect thing about this transfer.
    auto src_buf = Buffer::create(device, kBufSize, kBufSize, BufferType::L1);
    const uint32_t src_addr = static_cast<uint32_t>(src_buf->address());
    const uint32_t bad_addr = gap_addr_from(src_buf);

    const CoreCoord target_noc = device->worker_core_from_logical_core(target_logical);

    std::printf(
        "[PROBE] cross-core NOC write: (0,0) -> logical(%u,%u)/noc(%u,%u) dst=0x%x src=0x%x\n",
        static_cast<unsigned>(target_logical.x),
        static_cast<unsigned>(target_logical.y),
        static_cast<unsigned>(target_noc.x),
        static_cast<unsigned>(target_noc.y),
        bad_addr,
        src_addr);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t dst_x   = get_arg_val<uint32_t>(0);
            uint32_t dst_y   = get_arg_val<uint32_t>(1);
            uint32_t dst_off = get_arg_val<uint32_t>(2);
            uint32_t src_off = get_arg_val<uint32_t>(3);
            uint64_t dst = get_noc_addr(dst_x, dst_y, dst_off);
            noc_async_write(src_off, dst, 16);
            noc_async_write_barrier();
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        sender_logical,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, sender_logical, {target_noc.x, target_noc.y, bad_addr, src_addr});

    // No EXPECT_DEATH: passing means the sanitizers stayed silent.
    detail::LaunchProgram(device, program);
    std::printf("[PROBE] cross-core NOC write completed with NO ASAN abort\n");
}

// GAP 1, multicast flavour — the shape of the "reduced mcast target buffer" bug:
// the NOC multicast rectangle is delivered to every worker core in it, while the
// target buffer may be allocated on only a subset. __emule_multicast_write
// memcpys into each core in the rectangle with no liveness check.
TEST_F(MeshDeviceFixture, GapAddr_McastWrite_NotDetected) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord sender_logical = {0, 0};
    Program program = CreateProgram();

    auto src_buf = Buffer::create(device, kBufSize, kBufSize, BufferType::L1);
    const uint32_t src_addr = static_cast<uint32_t>(src_buf->address());
    const uint32_t bad_addr = gap_addr_from(src_buf);

    // A 2x2 rectangle of receivers, none of which has anything allocated at
    // bad_addr. Mirrors "mcast targeting the corners of a rectangle while a core
    // inside it is excluded from the target CB allocation".
    const CoreCoord start_noc = device->worker_core_from_logical_core(CoreCoord{1, 1});
    const CoreCoord end_noc = device->worker_core_from_logical_core(CoreCoord{2, 2});
    const uint32_t num_dests = 4;

    std::printf(
        "[PROBE] mcast write: rect noc(%u,%u)..(%u,%u) dst=0x%x src=0x%x\n",
        static_cast<unsigned>(start_noc.x),
        static_cast<unsigned>(start_noc.y),
        static_cast<unsigned>(end_noc.x),
        static_cast<unsigned>(end_noc.y),
        bad_addr,
        src_addr);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t xs      = get_arg_val<uint32_t>(0);
            uint32_t ys      = get_arg_val<uint32_t>(1);
            uint32_t xe      = get_arg_val<uint32_t>(2);
            uint32_t ye      = get_arg_val<uint32_t>(3);
            uint32_t dst_off = get_arg_val<uint32_t>(4);
            uint32_t src_off = get_arg_val<uint32_t>(5);
            uint32_t ndest   = get_arg_val<uint32_t>(6);
            uint64_t dst = get_noc_multicast_addr(xs, ys, xe, ye, dst_off);
            noc_async_write_multicast(src_off, dst, 16, ndest);
            noc_async_write_barrier();
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        sender_logical,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(
        program,
        kernel,
        sender_logical,
        {start_noc.x, start_noc.y, end_noc.x, end_noc.y, bad_addr, src_addr, num_dests});

    detail::LaunchProgram(device, program);
    std::printf("[PROBE] mcast write completed with NO ASAN abort\n");
}

// GAP 2 — the reserved-region blind spot, probed at its exact boundary. A LOCAL
// write (so it does reach the chokepoint) to an address just below
// l1_unreserved_base is waved through by the OOB check's early return, even
// though no tensor covers it. The Blackhole launch mailbox [0x60, 0x3270) sits
// far below this boundary, in the same unchecked band.
TEST_F(MeshDeviceFixture, BelowUnreservedBase_LocalWrite_NotDetected) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    const uint32_t unreserved_base =
        static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    // 16 bytes below the boundary: unambiguously in the "passes through" band,
    // while staying as close to legitimate user space as possible so this probe
    // disturbs as little firmware state as it can.
    const uint32_t below_addr = (unreserved_base - 16) & ~0xFu;

    std::printf(
        "[PROBE] l1_unreserved_base=0x%x; probing local write at 0x%x (below base); "
        "BH launch mailbox = [0x60,0x3270)\n",
        unreserved_base,
        below_addr);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* p = (volatile uint32_t*)__emule_local_l1_to_ptr(addr);
            *p = 0x666;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {below_addr});

    detail::LaunchProgram(device, program);
    std::printf("[PROBE] sub-unreserved-base local write completed with NO ASAN abort\n");
}

// --- §13 Launch-Mailbox Clobber -----------------------------------------------
// The three tests below are the regression fence for the new check. They use the
// launch-mailbox region rather than the `gap_addr_from` helper: the mailbox is
// reserved firmware state, so it is the one sub-unreserved-base region a
// kernel-initiated write must never touch.
//
// Offset choice: MAILBOX_BASE + 0x40 is inside the region on every arch
// (BH base 96 / WH+Q base 16, all sizes >= 12912) and 16B-aligned so the NOC
// alignment check (§10) can't be what fires. The check reads its bounds from the
// per-launch armed state (san.mailbox_l1_range_*, from the HAL), so the tests
// don't hardcode per-arch numbers.
namespace {
constexpr uint32_t kMailboxProbeOffset = 96 + 0x40;
}  // namespace

// Local write into the mailbox -> must abort. Also proves the check runs BEFORE
// cb_resolve is irrelevant here (no CB involved) but establishes the message.
TEST_F(MeshDeviceFixture, Mailbox_LocalWrite_Detected) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    ::unsetenv("TT_METAL_EMULE_ASAN_CHECK_MAILBOX_CLOBBER");

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* p = (volatile uint32_t*)__emule_local_l1_to_ptr(addr);
            *p = 0xbf;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {kMailboxProbeOffset});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Launch-Mailbox Clobber: local write to offset.*reserved launch-mailbox region.*");
}

// THE bug-A shape: a cross-core NOC write landing on another core's run message.
// Before §13 this was silent (see GapAddr_CrossCoreNocWrite_NotDetected).
TEST_F(MeshDeviceFixture, Mailbox_CrossCoreNocWrite_Detected) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    ::unsetenv("TT_METAL_EMULE_ASAN_CHECK_MAILBOX_CLOBBER");

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord sender_logical = {0, 0};
    Program program = CreateProgram();

    auto src_buf = Buffer::create(device, kBufSize, kBufSize, BufferType::L1);
    const uint32_t src_addr = static_cast<uint32_t>(src_buf->address());
    const CoreCoord target_noc = device->worker_core_from_logical_core(CoreCoord{1, 1});

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t dst_x   = get_arg_val<uint32_t>(0);
            uint32_t dst_y   = get_arg_val<uint32_t>(1);
            uint32_t dst_off = get_arg_val<uint32_t>(2);
            uint32_t src_off = get_arg_val<uint32_t>(3);
            uint64_t dst = get_noc_addr(dst_x, dst_y, dst_off);
            noc_async_write(src_off, dst, 16);
            noc_async_write_barrier();
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        sender_logical,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, sender_logical, {target_noc.x, target_noc.y, kMailboxProbeOffset, src_addr});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Launch-Mailbox Clobber: NOC write to offset.*reserved launch-mailbox region.*");
}

// The opt-out must actually opt out, so a sweep can isolate this check.
TEST_F(MeshDeviceFixture, Mailbox_SkipGate_Respected) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    ::setenv("TT_METAL_EMULE_ASAN_CHECK_MAILBOX_CLOBBER", "0", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile uint32_t* p = (volatile uint32_t*)__emule_local_l1_to_ptr(addr);
            *p = 0xbf;
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {kMailboxProbeOffset});

    detail::LaunchProgram(device, program);
    std::printf("[PROBE] skip gate honored: mailbox write completed with NO abort\n");
    ::unsetenv("TT_METAL_EMULE_ASAN_CHECK_MAILBOX_CLOBBER");
}

}  // namespace tt::tt_metal
