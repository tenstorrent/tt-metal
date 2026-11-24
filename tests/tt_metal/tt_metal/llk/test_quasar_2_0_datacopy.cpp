// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 compute API datacopy over the Dest-sync cases S1-S8 (Phase 1 matrix; S9, S1 under LLK asserts, is a
// build flag, not a case) x SyncHalf/SyncFull x 16/32-bit Dest, with Float16_b L1 (FPU datacopy at either Dest
// width) and, at 32-bit Dest, Float32 L1 (fp32in: unpack-to-dest, all 32 bits). Checked bit-exact against the input;
// outputs start as a sentinel. Test names: <S>_<sync>_<dest>, e.g. S3_full_fp32, S3_half_fp32in.
// _lta: S2 with in/out as resident L1 tensors read through LocalTensorAccessor (no DFBs, no DM kernels), per L1
// format; the Dest sync is the same as the DFB runs, so SyncHalf only.

#include "quasar_2_0_test_common.hpp"

namespace tt::tt_metal {
namespace {

using namespace quasar_2_0;

struct CopyCase {
    std::uint32_t s;  // sync case S1..S8; S12-S14 early returns
    bool full_sync;
    bool fp32;
    bool fp32_in = false;  // Float32 L1 in and out (fp32 Dest only)
    bool lta = false;      // operands via LocalTensorAccessor instead of DFBs
};

std::string copy_case_name(const ::testing::TestParamInfo<CopyCase>& info) {
    const CopyCase& c = info.param;
    return "S" + std::to_string(c.s) + (c.full_sync ? "_full" : "_half") +
           (c.fp32_in ? "_fp32in"
            : c.fp32  ? "_fp32"
                      : "_fp16b") +
           (c.lta ? "_lta" : "");
}

std::vector<CopyCase> copy_cases() {
    std::vector<CopyCase> cases;
    for (bool full : {false, true}) {
        for (bool fp32 : {false, true}) {
            for (std::uint32_t s = 1; s <= 8; ++s) {
                cases.push_back({s, full, fp32});
                if (fp32) {
                    cases.push_back({s, full, true, true});
                }
            }
            for (std::uint32_t s = 12; s <= 14 && !fp32; ++s) {
                cases.push_back({s, full, fp32});
            }
        }
    }
    cases.push_back({2, false, false, false, true});
    cases.push_back({2, false, true, false, true});
    cases.push_back({2, false, true, true, true});
    return cases;
}

class Quasar2_0Datacopy : public LLKQuasarMeshDeviceSingleCardFixture,
                          public ::testing::WithParamInterface<CopyCase> {};

TEST_P(Quasar2_0Datacopy, Run) {
    const CopyCase& c = GetParam();
    const std::uint32_t cap = dest_capacity(c.full_sync, c.fp32);
    // tiles per section, total tiles, mode (see datacopy_2_0_quasar.cpp)
    std::uint32_t tps = 1, n = 32, mode = 0;
    switch (c.s) {
        case 2:
            tps = cap;
            n = 4 * cap;
            break;
        case 3:
            tps = 3;
            n = 33;
            break;
        case 4:
            tps = cap < 8 ? cap : 8;
            n = 37;
            break;  // full blocks + a runtime tail
        case 5:
            tps = 2;
            n = 32;
            break;  // DFB depth 4 < N
        case 6:
            tps = 2;
            n = 22;
            mode = 6;
            break;  // 3 + 5 + 3 sections, re-init between
        case 7:
            tps = 2;
            n = 32;
            mode = 7;
            break;  // two DFB pairs
        case 12:
        case 13:
        case 14:
            tps = 2;
            n = 8;
            mode = 12;
            break;       // an early return, then a checked run
        default: break;  // S1, S8 (S1 twice)
    }
    // Skip per run (datacopy_2_0_quasar.cpp mode 12): S12 returns before compute_kernel_hw_startup, S13 right after it,
    // S14 with Dest left unpacked; the next run is checked. A run with skip != 0 is not checked. Two programs per test:
    // more do not fit one emulator session.
    const std::vector<std::uint32_t> skips = c.s >= 12  ? std::vector<std::uint32_t>{c.s - 11, 0}
                                             : c.s == 8 ? std::vector<std::uint32_t>{0, 0}
                                                        : std::vector<std::uint32_t>{0};
    for (std::uint32_t run = 0; run < skips.size(); ++run) {
        const std::uint32_t skip = skips[run];
        ProgramIo io;
        io.compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/datacopy_2_0_quasar.cpp";
        io.fp32_dest = c.fp32;
        io.full_sync = c.full_sync;
        io.fp32_l1 = c.fp32_in;
        io.lta = c.lta;
        const std::uint32_t halves = c.fp32_in ? 2 : 1;  // uint16 halves per element
        io.ct_args = {{"num_tiles", n}, {"tiles_per_section", tps}, {"mode", mode}};
        io.rt_args = {{"skip", skip}};
        const std::uint32_t pairs = mode == 7 ? 2 : 1;
        for (std::uint32_t p = 0; p < pairs; ++p) {
            // Early returns move only what the kernel consumes and packs: a DFB left with entries hangs the DM kernels.
            const std::uint32_t in_tiles = skip == 0 ? n / pairs : skip == 3 ? 2 * tps : 0;
            io.inputs.push_back(random_tiles(halves * in_tiles, 100 * c.s + 10 * run + p));
            io.out_tiles.push_back(skip == 0 ? n / pairs : skip == 3 ? tps : 0);
            io.in_depth.push_back(2 * tps);
            io.out_depth.push_back(2 * tps);
        }
        const std::vector<Tiles> outs = run_program(*this->devices_.at(0), io);
        if (skip != 0) {
            continue;
        }
        for (std::uint32_t p = 0; p < pairs; ++p) {
            EXPECT_TRUE(tiles_equal(outs[p], io.inputs[p])) << "run " << run << " output " << p;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Quasar2_0, Quasar2_0Datacopy, ::testing::ValuesIn(copy_cases()), copy_case_name);

}  // namespace
}  // namespace tt::tt_metal
