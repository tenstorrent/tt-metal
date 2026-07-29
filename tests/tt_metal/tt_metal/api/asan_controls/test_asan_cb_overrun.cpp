// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Controls for §4 arm 2 (CB overrun). See README.md and SANITIZER_CHECKS.md §4.
//
// Four behaviours are pinned here, because a bounds check is only trustworthy if all
// four hold:
//   Positive  — legal in-bounds CB access stays silent
//   Death     — overrun into a gap between CBs aborts
//   FP        — a legal access at the very last byte of a CB stays silent (off-by-one)
//   FN        — overrun that lands inside a neighbouring CB is NOT caught (documented)
//
// All kernels use the public API only.
//
// To run:
//   TT_METAL_EMULE_ASAN=1 TT_EMULE_FIBER_WORKERS=1 \
//     build_emule/test/tt_metal/unit_tests_api \
//     --gtest_death_test_style=threadsafe --gtest_filter='*CbOverrun*'

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

namespace {

constexpr uint32_t kPageSize = 64;
constexpr uint32_t kPages = 4;
constexpr uint32_t kCbBytes = kPages * kPageSize;

// One CB on `core`, at buffer index `idx`.
void make_cb(Program& program, const CoreCoord& core, uint32_t idx) {
    CircularBufferConfig cfg =
        CircularBufferConfig(kCbBytes, {{idx, tt::DataFormat::Float16_b}}).set_page_size(idx, kPageSize);
    CreateCircularBuffer(program, core, cfg);
}

// A kernel that reserves cb 0 and writes at `byte_offset` from its write pointer.
std::string offset_write_kernel() {
    return R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t off = get_arg_val<uint32_t>(0);
            cb_reserve_back(0, 4);
            uint32_t base = get_write_ptr(0);
            volatile tt_l1_ptr uint32_t* p = (volatile tt_l1_ptr uint32_t*)(base + off);
            *p = 0xa5a5a5a5;
            cb_push_back(0, 4);
        }
    )";
}

void run_offset_write(IDevice* device, uint32_t byte_offset, uint32_t num_cbs, bool expect_death) {
    CoreCoord core = {0, 0};
    Program program = CreateProgram();
    for (uint32_t i = 0; i < num_cbs; ++i) {
        make_cb(program, core, i);
    }
    auto kernel = CreateKernelFromString(
        program,
        offset_write_kernel(),
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, core, {byte_offset});

    if (expect_death) {
        EXPECT_DEATH(
            detail::LaunchProgram(device, program),
            ".*Out-of-Bounds Write: offset .* past the end of CB 0 .* a CB overrun.*");
    } else {
        detail::LaunchProgram(device, program);
    }
}

void arm() {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    ::unsetenv("TT_METAL_EMULE_ASAN_CHECK_OOB_L1");
}

}  // namespace

// POSITIVE — a legal write inside cb 0 must stay silent. If this ever aborts the check
// is over-firing and everything else here is meaningless.
TEST_F(MeshDeviceFixture, CbOverrun_InBounds_NoViolation) {
    arm();
    auto* device = this->devices_.at(0)->get_devices()[0];
    run_offset_write(device, /*byte_offset=*/0, /*num_cbs=*/2, /*expect_death=*/false);
    std::printf("[CONTROL] in-bounds CB write: no abort\n");
}

// FP GUARD — the last 4 bytes of cb 0 are still cb 0. An off-by-one in the extent
// arithmetic (using <= instead of <, or start+size-1) would abort here.
TEST_F(MeshDeviceFixture, CbOverrun_LastByteInBounds_NoViolation) {
    arm();
    auto* device = this->devices_.at(0)->get_devices()[0];
    run_offset_write(device, /*byte_offset=*/kCbBytes - 4, /*num_cbs=*/2, /*expect_death=*/false);
    std::printf("[CONTROL] last-word-of-CB write: no abort\n");
}

// DEATH — one CB only, so the bytes past its end are inside no CB. With a second CB
// created the same overrun would land in that neighbour and go undetected (see the FN
// control below), which is exactly why this uses a single CB.
TEST_F(MeshDeviceFixture, CbOverrun_PastEnd_Detected) {
    arm();
    auto* device = this->devices_.at(0)->get_devices()[0];
    run_offset_write(device, /*byte_offset=*/kCbBytes, /*num_cbs=*/1, /*expect_death=*/true);
}

// FN — the documented limitation. Two CBs are created adjacently; an overrun of exactly
// one CB's worth lands inside the neighbour, which IS a live CB, so arm 2 cannot see it
// (nor can arm 1 — CB memory is not in LiveL1Ranges). Detection requires a gap between
// slots; that is what the arena redzone is for. Flip to EXPECT_DEATH when redzones land
// and this offset falls inside one.
TEST_F(MeshDeviceFixture, CbOverrun_IntoAdjacentCb_NotDetected) {
    arm();
    auto* device = this->devices_.at(0)->get_devices()[0];
    run_offset_write(device, /*byte_offset=*/kCbBytes, /*num_cbs=*/2, /*expect_death=*/false);
    std::printf("[CONTROL] overrun into an adjacent CB: NO abort (needs an inter-slot gap)\n");
}

}  // namespace tt::tt_metal
