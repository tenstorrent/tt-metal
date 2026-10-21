// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 compute API datacopy over the Dest-sync cases S1-S9 (Phase 1 matrix): both copy paths (FPU datacopy,
// unpack-to-dest) x SyncHalf/SyncFull x 16/32-bit Dest. Checked bit-exact against the input; outputs start as a
// sentinel. Test names: <S>_<path>_<sync>_<dest>, e.g. S3_todest_full_fp32.

#include "quasar_2_0_test_common.hpp"

namespace tt::tt_metal {
namespace {

using namespace quasar_2_0;

struct CopyCase {
    std::uint32_t s;  // sync case S1..S9
    bool to_dest;
    bool full_sync;
    bool fp32;
};

std::string copy_case_name(const ::testing::TestParamInfo<CopyCase>& info) {
    const CopyCase& c = info.param;
    return "S" + std::to_string(c.s) + (c.s == 6 ? "_chain" : (c.to_dest ? "_todest" : "_fpu")) +
           (c.full_sync ? "_full" : "_half") + (c.fp32 ? "_fp32" : "_fp16b");
}

std::vector<CopyCase> copy_cases() {
    std::vector<CopyCase> cases;
    for (bool full : {false, true}) {
        for (bool fp32 : {false, true}) {
            for (std::uint32_t s = 1; s <= 9; ++s) {
                for (bool to_dest : {false, true}) {
                    if (s == 6 && to_dest) {
                        continue;  // S6 runs both paths in one kernel
                    }
                    cases.push_back({s, to_dest, full, fp32});
                }
            }
        }
    }
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
        case 2: tps = cap; n = 4 * cap; break;
        case 3: tps = 3; n = 33; break;
        case 4: tps = cap < 8 ? cap : 8; n = 37; break;  // full blocks + a runtime tail
        case 5: tps = 2; n = 32; break;                   // DFB depth 4 < N
        case 6: tps = 2; n = 22; mode = 6; break;         // 3 + 5 + 3 sections
        case 7: tps = 2; n = 32; mode = 7; break;         // two DFB pairs
        default: break;                                   // S1, S8 (S1 twice), S9 (S1 under LLK asserts)
    }
    const std::uint32_t runs = c.s == 8 ? 2 : 1;
    for (std::uint32_t run = 0; run < runs; ++run) {
        ProgramIo io;
        io.compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/datacopy_2_0_quasar.cpp";
        io.fp32_dest = c.fp32;
        io.full_sync = c.full_sync;
        io.ct_args = {{"num_tiles", n}, {"tiles_per_section", tps}, {"to_dest", c.to_dest}, {"mode", mode}};
        const std::uint32_t pairs = mode == 7 ? 2 : 1;
        for (std::uint32_t p = 0; p < pairs; ++p) {
            io.inputs.push_back(random_tiles(n / pairs, 100 * c.s + 10 * run + p));
            io.out_tiles.push_back(n / pairs);
            io.in_depth.push_back(2 * tps);
            io.out_depth.push_back(2 * tps);
        }
        const std::vector<Tiles> outs = run_program(*this->devices_.at(0), io);
        for (std::uint32_t p = 0; p < pairs; ++p) {
            EXPECT_TRUE(tiles_equal(outs[p], io.inputs[p])) << "run " << run << " output " << p;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Quasar2_0, Quasar2_0Datacopy, ::testing::ValuesIn(copy_cases()), copy_case_name);

}  // namespace
}  // namespace tt::tt_metal
