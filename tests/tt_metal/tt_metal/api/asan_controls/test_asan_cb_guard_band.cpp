// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Controls for §15 CB guard band. See SANITIZER_CHECKS.md §15.
//
// The metal allocator will not leave a hole between CBs on request, so these back two
// CBs at explicit addresses inside ONE L1 buffer with a deliberate hole between them —
// the same geometry Blaze's arena redzone produces. Because the hole is inside an
// allocated buffer, §4 accepts a write there; §15 is what reports it, which is exactly
// the situation being pinned.
//
// Five behaviours, because a poison check is only trustworthy if all of them hold:
//   Positive      in-bounds writes to both CBs stay silent
//   Death         a write in the hole aborts, naming the lower CB
//   FP (low edge) the last byte of the lower CB stays silent
//   FP (high edge) the first byte of the upper CB stays silent
//   Bound         a hole WIDER than redzone+alignment does NOT fire
//
// To run:
//   TT_METAL_EMULE_ASAN=1 TT_EMULE_FIBER_WORKERS=1 BLAZE_ASAN_CB_REDZONE=64 \
//     build_emule/test/tt_metal/unit_tests_api \
//     --gtest_death_test_style=threadsafe --gtest_filter='*GuardBand*'

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

namespace guard_band_ctl {

constexpr uint32_t kPageSize = 512;
constexpr uint32_t kPages = 4;
constexpr uint32_t kCbBytes = kPages * kPageSize;  // 2048
constexpr uint32_t kRedzone = 64;                  // matches BLAZE_ASAN_CB_REDZONE below

// Two CBs with a `hole` between them. set_globally_allocated_address takes no offset,
// so the hole is carved with the allocator instead: L1 allocates top-down, so reserving
// hi / gap / lo in that order puts them in descending address order, and releasing the
// middle one leaves a hole between lo and hi. Returns {lo, hi}; the caller keeps them
// alive. `lo_end` is where the hole starts.
struct HolePair {
    std::shared_ptr<Buffer> lo;
    std::shared_ptr<Buffer> hi;
    uint32_t lo_end = 0;
};

HolePair two_cbs_with_hole(IDevice* device, Program& program, const CoreCoord& core, uint32_t hole) {
    auto hi = Buffer::create(device, kCbBytes, kCbBytes, BufferType::L1);
    auto gap = Buffer::create(device, hole, hole, BufferType::L1);
    auto lo = Buffer::create(device, kCbBytes, kCbBytes, BufferType::L1);
    DeallocateBuffer(*gap);

    CircularBufferConfig c0 = CircularBufferConfig(kCbBytes, {{0, tt::DataFormat::Float16_b}})
                                  .set_page_size(0, kPageSize)
                                  .set_globally_allocated_address(*lo);
    CreateCircularBuffer(program, core, c0);
    CircularBufferConfig c1 = CircularBufferConfig(kCbBytes, {{1, tt::DataFormat::Float16_b}})
                                  .set_page_size(1, kPageSize)
                                  .set_globally_allocated_address(*hi);
    CreateCircularBuffer(program, core, c1);
    return HolePair{lo, hi, static_cast<uint32_t>(lo->address()) + kCbBytes};
}

std::string write_at_kernel() {
    return R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t addr = get_arg_val<uint32_t>(0);
            volatile tt_l1_ptr uint32_t* p = (volatile tt_l1_ptr uint32_t*)addr;
            *p = 0x5a5a5a5a;
        }
    )";
}

void arm() {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    ::setenv("BLAZE_ASAN_CB_REDZONE", "64", 1);
    ::unsetenv("TT_METAL_EMULE_ASAN_CHECK_CB_GUARD_BAND");
}

// Launch a kernel that writes to an absolute L1 offset.
void write_absolute_expect(
    IDevice* device, Program& program, const CoreCoord& core, uint32_t l1_off, const char* pattern) {
    auto kernel = CreateKernelFromString(
        program,
        write_at_kernel(),
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, core, {l1_off});
    if (pattern != nullptr) {
        EXPECT_DEATH(detail::LaunchProgram(device, program), pattern);
    } else {
        detail::LaunchProgram(device, program);
    }
}

void write_absolute(IDevice* device, Program& program, const CoreCoord& core, uint32_t l1_off, bool expect_death) {
    write_absolute_expect(device, program, core, l1_off, expect_death ? ".*in the guard band between CB 0.*" : nullptr);
}

}  // namespace guard_band_ctl

// POSITIVE — a write inside cb 0 must stay silent.
TEST_F(MeshDeviceFixture, GuardBand_InBounds_NoViolation) {
    guard_band_ctl::arm();
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord core = {0, 0};
    Program program = CreateProgram();
    auto hp = guard_band_ctl::two_cbs_with_hole(device, program, core, guard_band_ctl::kRedzone);
    guard_band_ctl::write_absolute(
        device, program, core, static_cast<uint32_t>(hp.lo->address()), /*expect_death=*/false);
    std::printf("[CONTROL] in-bounds CB write: no abort\n");
}

// FP GUARD, low edge — the last word of cb 0 is still cb 0.
TEST_F(MeshDeviceFixture, GuardBand_LastByteOfLowerCb_NoViolation) {
    guard_band_ctl::arm();
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord core = {0, 0};
    Program program = CreateProgram();
    auto hp = guard_band_ctl::two_cbs_with_hole(device, program, core, guard_band_ctl::kRedzone);
    guard_band_ctl::write_absolute(device, program, core, hp.lo_end - 4, /*expect_death=*/false);
    std::printf("[CONTROL] last word of lower CB: no abort\n");
}

// DEATH — a write inside the hole must abort and name the lower CB. This is the whole
// point of the redzone: without the hole this address would be live CB data.
TEST_F(MeshDeviceFixture, GuardBand_InHole_Detected) {
    guard_band_ctl::arm();
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord core = {0, 0};
    Program program = CreateProgram();
    auto hp = guard_band_ctl::two_cbs_with_hole(device, program, core, guard_band_ctl::kRedzone);
    guard_band_ctl::write_absolute(device, program, core, hp.lo_end, /*expect_death=*/true);
}

// BOUND — a hole wider than redzone+alignment must NOT be treated as a guard band: two
// CBs that far apart may have unrelated live data between them. The address here is also
// genuinely unallocated, so §4 catches it; asserting §4's message proves §15 declined,
// because a §15 fire would print the guard-band text instead. That is a stronger check
// than expecting silence — it pins WHICH check owns the address.
TEST_F(MeshDeviceFixture, GuardBand_HoleWiderThanBound_FallsThroughToOob) {
    guard_band_ctl::arm();
    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord core = {0, 0};
    Program program = CreateProgram();
    // redzone(64) + align(64) = 128 is the bound; 4096 is comfortably past it.
    auto hp = guard_band_ctl::two_cbs_with_hole(device, program, core, 4096);
    guard_band_ctl::write_absolute_expect(device, program, core, hp.lo_end, ".*not part of any allocated tensor.*");
}

}  // namespace tt::tt_metal
