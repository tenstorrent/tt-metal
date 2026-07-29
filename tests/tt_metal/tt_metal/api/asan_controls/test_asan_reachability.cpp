// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ASAN reachability controls — see README.md in this directory.
//
// Every kernel here uses ONLY the public kernel API (cb_reserve_back,
// get_write_ptr, noc_async_write, ...) and never names an __emule_* internal.
// That is the whole point: the existing per-check death tests call the
// sanitizer's host function directly, which proves the comparison works but not
// that a real kernel can reach it.
//
// To run:
//   TT_METAL_EMULE_ASAN=1 TT_EMULE_FIBER_WORKERS=1 \
//     build_emule/test/tt_metal/unit_tests_api \
//     --gtest_death_test_style=threadsafe --gtest_filter='*Reach*'

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

// Two adjacent CBs, packed back-to-back the way Blaze's arena packs scratch CBs
// (`_compute_arena_offsets`: "the lowest offset free on ALL of its cores").
// Returns the program with cb 0 and cb 1 created on `core`.
struct TwoCbs {
    static constexpr uint32_t kPageSize = 64;
    static constexpr uint32_t kPages = 4;
};

}  // namespace

// §8 CB Reservation Overflow — public API, must abort.
// This is the positive control for the whole file: it proves that a violation
// expressed purely through the public API does reach a check and does report.
// If this ever stops aborting, the harness itself is broken and every
// `_Unreachable` result below becomes meaningless.
TEST_F(MeshDeviceFixture, Reach_CbReservationOverflow_Reachable) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    CircularBufferConfig cb_config =
        CircularBufferConfig(TwoCbs::kPages * TwoCbs::kPageSize, {{0, tt::DataFormat::Float16_b}})
            .set_page_size(0, TwoCbs::kPageSize);
    CreateCircularBuffer(program, logical_core, cb_config);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // 4-page CB, ask for 5 — public API only.
            cb_reserve_back(0, 5);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(detail::LaunchProgram(device, program), ".*CB Reservation Overflow.*");
}

// §4 OOB-L1 vs the Blaze arena-adjacency gap — public API, currently NOT detected.
//
// Two CBs are created back-to-back. The kernel reserves cb 0, then writes one
// page PAST cb 0's end using only `get_write_ptr(0)` + arithmetic. On silicon
// that is a straightforward buffer overrun into a neighbour.
//
// Nothing fires. `__emule_asan_cb_resolve` searches for *whichever* CB contains
// the address, finds cb 1, and returns before §4 OOB runs — CB memory is
// deliberately absent from LiveL1Ranges. §7 CB-Boundary is then evaluated
// against cb 1 (the wrong CB), where the access looks like an ordinary
// no-active-window scratch write.
//
// This is exactly the Blaze failure mode: `_compute_arena_offsets` packs scratch
// CBs densely, so an overrun lands in a neighbouring CB rather than in
// unallocated space.
//
// To fix: carry the *intended* cb_id to the access site and compare the access
// against THAT CB's extent, instead of searching for a containing CB.
// Flip to EXPECT_DEATH when that lands.
TEST_F(MeshDeviceFixture, Reach_OobL1_AcrossAdjacentCb_Unreachable) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // cb 0 and cb 1, same geometry, created together so the allocator packs them
    // adjacently — mirroring Blaze's arena.
    CircularBufferConfig cb0 =
        CircularBufferConfig(TwoCbs::kPages * TwoCbs::kPageSize, {{0, tt::DataFormat::Float16_b}})
            .set_page_size(0, TwoCbs::kPageSize);
    CreateCircularBuffer(program, logical_core, cb0);
    CircularBufferConfig cb1 =
        CircularBufferConfig(TwoCbs::kPages * TwoCbs::kPageSize, {{1, tt::DataFormat::Float16_b}})
            .set_page_size(1, TwoCbs::kPageSize);
    CreateCircularBuffer(program, logical_core, cb1);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            constexpr uint32_t kPageSize = 64;
            constexpr uint32_t kPages = 4;
            cb_reserve_back(0, kPages);
            uint32_t base = get_write_ptr(0);
            // One full CB past cb 0's base: beyond cb 0 entirely.
            volatile tt_l1_ptr uint32_t* past_end =
                (volatile tt_l1_ptr uint32_t*)(base + kPages * kPageSize);
            *past_end = 0xdeadbeef;
            cb_push_back(0, kPages);
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Passing == nothing aborted == the gap is still open.
    detail::LaunchProgram(device, program);
    std::printf("[CONTROL] wrote past cb0 into adjacent cb1 — NO abort (OOB-L1/CB-Boundary gap)\n");
}

//
// The kernel writes to a DRAM bank address far past the end of the only allocated
// DRAM buffer, using the public bank addrgen + noc_async_write. Nothing names an
// __emule_* internal, so this proves the check is reachable from real kernel code —
// the property that was missing while the check sat in the orphaned
// __emule_dram_ptr (zero call sites) and its own death test called that function
// directly.
TEST_F(MeshDeviceFixture, Reach_OobDram_ViaPublicApi_Reachable) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    ::unsetenv("TT_METAL_EMULE_ASAN_CHECK_OOB_DRAM");

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t kDramBufSize = 1024;
    auto dram_buf = Buffer::create(device, kDramBufSize, kDramBufSize, BufferType::DRAM);
    auto l1_src = Buffer::create(device, kDramBufSize, kDramBufSize, BufferType::L1);

    // 8 MB past the buffer: outside any allocated DRAM extent, still inside the
    // emulated bank's backing store.
    const uint32_t bad_dram_addr = static_cast<uint32_t>(dram_buf->address()) + (8u << 20);

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t src      = get_arg_val<uint32_t>(0);
            uint32_t dram_off = get_arg_val<uint32_t>(1);
            uint64_t dst = get_noc_addr_from_bank_id<true>(0, dram_off);
            noc_async_write(src, dst, 16);
            noc_async_write_barrier();
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {static_cast<uint32_t>(l1_src->address()), bad_dram_addr});

    EXPECT_DEATH(
        detail::LaunchProgram(device, program),
        ".*Out-of-Bounds Write: Attempted to access DRAM address.*not part of any allocated tensor.*");
}

// §14 in-bounds NoViolation control — must stay SILENT.
//
// The single most important test for this check. §14 compares an allocator address
// against the live-extent list, but the kernel puts an *in-bank NOC offset* on the
// wire, which already includes the dram view's base offset. The check subtracts that
// offset to get back to allocator space. If the sign, the key, or the view lookup is
// wrong, a perfectly legal DRAM write starts aborting — and this test is what catches
// it. A passing Reachable test above says nothing about that, because a check that
// fires on *everything* also fires on the bad address.
TEST_F(MeshDeviceFixture, Reach_OobDram_InBounds_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    ::unsetenv("TT_METAL_EMULE_ASAN_CHECK_OOB_DRAM");

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    constexpr uint32_t kDramBufSize = 2048;
    auto dram_buf = Buffer::create(device, kDramBufSize, kDramBufSize, BufferType::DRAM);
    auto l1_src = Buffer::create(device, kDramBufSize, kDramBufSize, BufferType::L1);

    // Squarely inside the buffer.
    const uint32_t good_dram_addr = static_cast<uint32_t>(dram_buf->address());

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t src      = get_arg_val<uint32_t>(0);
            uint32_t dram_off = get_arg_val<uint32_t>(1);
            uint64_t dst = get_noc_addr_from_bank_id<true>(0, dram_off);
            noc_async_write(src, dst, 16);
            noc_async_write_barrier();
        }
    )";

    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {static_cast<uint32_t>(l1_src->address()), good_dram_addr});

    detail::LaunchProgram(device, program);
    std::printf("[CONTROL] in-bounds DRAM write completed with no abort (normalization is sane)\n");
}

}  // namespace tt::tt_metal
