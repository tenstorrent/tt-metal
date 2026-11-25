// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 compute API datacopy over the Dest-sync cases S1-S8 (Phase 1 matrix; S9, S1 under LLK asserts, is a
// build flag, not a case) x SyncHalf/SyncFull x 16/32-bit Dest, with Float16_b L1 (FPU datacopy at either Dest
// width) and, at 32-bit Dest, Float32 L1 (fp32in: unpack-to-dest, all 32 bits). Checked bit-exact against the input;
// outputs start as a sentinel. Test names: <S>_<sync>_<dest>, e.g. S3_full_fp32, S3_half_fp32in.
// _lta: S2 with in/out as resident L1 tensors read through LocalTensorAccessor (no DFBs, no DM kernels), per L1
// format; the Dest sync is the same as the DFB runs, so SyncHalf only. _sem: _lta, but a DM kernel writes in from
// DRAM and then signals the compute kernel through a shared semaphore.
// T_<rd><ci>_<co><wr>_r<R>c<C>w<W>: a DFB topology, Float16_b, SyncHalf, 16-bit Dest, 64 tiles, blocks of 4. R DM
// readers -> in -> C compute threads -> out -> W DM writers; <rd><ci> are the in DFB's producer / consumer patterns and
// <co><wr> the out DFB's (S = STRIDED, B = BLOCKED, A = ALL). 1:N, N:1 and N:M thread counts on every pairing.
// _relu / _l1acc: S2 with the pack features: ReLU (negatives -> 0) and L1 accumulation (each section packed twice,
// out = 2 x in).

#include "quasar_2_0_test_common.hpp"

namespace tt::tt_metal {
namespace {

using namespace quasar_2_0;

struct CopyCase {
    std::uint32_t s;  // sync case S1..S8; S12-S14 early returns
    bool full_sync;
    bool fp32;
    bool fp32_in = false;    // Float32 L1 in and out (fp32 Dest only)
    bool lta = false;        // operands via LocalTensorAccessor instead of DFBs
    bool sem = false;        // lta, with in written by a DM kernel that signals compute through a semaphore
    std::uint32_t pack = 0;  // kernel mode for a pack feature: 20 = ReLU, 21 = L1 accumulation
    int family = -1;         // >= 0: a topology case FAMILIES[family] with threads r / c / w (S is unused)
    std::uint8_t r = 1, c = 1, w = 1;
};

// DFB pattern pairings of a topology case: reader -> compute on in, compute -> writer on out.
struct Family {
    const char* name;
    DFBAccessPattern reader, in, out, writer;
};
constexpr DFBAccessPattern S_ = DFBAccessPattern::STRIDED, B_ = DFBAccessPattern::BLOCKED, A_ = DFBAccessPattern::ALL;
constexpr Family FAMILIES[] = {
    {"SS_SS", S_, S_, S_, S_},
    {"SB_BS", S_, B_, B_, S_},
    {"SA_SS", S_, A_, S_, S_},
    {"BB_BB", B_, B_, B_, B_},
    {"BA_SS", B_, A_, S_, S_},
    {"BS_SB", B_, S_, S_, B_},
    {"SS_SA", S_, S_, S_, A_},
    {"SB_BA", S_, B_, B_, A_},
};
constexpr std::uint32_t TOPO_BLOCK = 4;

std::string copy_case_name(const ::testing::TestParamInfo<CopyCase>& info) {
    const CopyCase& c = info.param;
    if (c.family >= 0) {
        return std::string("T_") + FAMILIES[c.family].name + "_r" + std::to_string(c.r) + "c" + std::to_string(c.c) +
               "w" + std::to_string(c.w);
    }
    return "S" + std::to_string(c.s) + (c.full_sync ? "_full" : "_half") +
           (c.fp32_in ? "_fp32in"
            : c.fp32  ? "_fp32"
                      : "_fp16b") +
           (c.sem          ? "_sem"
            : c.lta        ? "_lta"
            : c.pack == 20 ? "_relu"
            : c.pack == 21 ? "_l1acc"
                           : "");
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
    cases.push_back({2, false, false, false, true, true});
    // Readers + writers <= 6 DM threads, ALL consumers <= 4.
    constexpr std::uint8_t TOPOS[][3] = {{1, 2, 1}, {1, 4, 1}, {2, 1, 2}, {4, 1, 2}, {2, 2, 2}, {2, 4, 2}, {4, 2, 2}};
    for (int f = 0; f < static_cast<int>(std::size(FAMILIES)); ++f) {
        for (const auto& t : TOPOS) {
            cases.push_back({.s = 0, .full_sync = false, .fp32 = false, .family = f, .r = t[0], .c = t[1], .w = t[2]});
        }
    }
    for (std::uint32_t pack : {20u, 21u}) {
        cases.push_back({.s = 2, .full_sync = false, .fp32 = false, .pack = pack});
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
            break;  // 37 is prime: block 1
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
    if (c.pack != 0) {
        mode = c.pack;
    }
    const Family* fam = c.family >= 0 ? &FAMILIES[c.family] : nullptr;
    if (fam) {
        // A compute op covers its whole share of a block: the block on a BLOCKED side and on ALL facing a BLOCKED
        // reader, block / threads on STRIDED facing a BLOCKED DM side, any size (4) between STRIDED / ALL sides.
        const bool dm_blocked = fam->reader == B_ || fam->writer == B_;
        tps = fam->in == B_ || fam->out == B_ || (fam->in == A_ && fam->reader == B_) ? TOPO_BLOCK
              : dm_blocked                                                            ? TOPO_BLOCK / c.c
                                                                                      : 4;
        n = 64;
    }
    const bool all = fam && fam->in == A_;
    const std::uint32_t threads = fam ? c.c : 1;
    // Copies of each input tile in the output, in a row: one per ALL compute thread, then one per ALL writer.
    const std::uint32_t reps = (all ? threads : 1) * (fam && fam->writer == A_ ? c.w : 1);
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
        io.sem = c.sem;
        const std::uint32_t halves = c.fp32_in ? 2 : 1;  // uint16 halves per element
        if (fam) {
            io.threads = c.c;
            io.readers = c.r;
            io.writers = c.w;
            io.block = TOPO_BLOCK;
            io.reader_pattern = fam->reader;
            io.pattern = fam->in;
            io.out_pattern = fam->out;
            io.writer_pattern = fam->writer;
        }
        // num_tiles is per compute thread: an ALL thread copies every tile, the others share them.
        io.ct_args = {{"num_tiles", all ? n : n / threads}, {"tiles_per_section", tps}, {"mode", mode}};
        io.rt_args = {{"skip", skip}};
        const std::uint32_t pairs = mode == 7 ? 2 : 1;
        for (std::uint32_t p = 0; p < pairs; ++p) {
            // Early returns move only what the kernel consumes and packs: a DFB left with entries hangs the DM kernels.
            const std::uint32_t in_tiles = skip == 0 ? n / pairs : skip == 3 ? 2 * tps : 0;
            io.inputs.push_back(random_tiles(halves * in_tiles, 100 * c.s + 10 * run + p));
            io.out_tiles.push_back(skip == 0 ? (all ? n * threads : n / pairs) : skip == 3 ? tps : 0);
            io.in_depth.push_back(fam ? 32 : 2 * tps);
            io.out_depth.push_back(fam ? 32 : 2 * tps);
        }
        const std::vector<Tiles> outs = run_program(*this->devices_.at(0), io);
        if (skip != 0) {
            continue;
        }
        for (std::uint32_t p = 0; p < pairs; ++p) {
            Tiles want;
            for (std::size_t k = 0; k < io.inputs[p].size() / TILE_ELEMS / halves; ++k) {
                for (std::uint32_t t = 0; t < reps; ++t) {
                    want.insert(
                        want.end(),
                        io.inputs[p].begin() + k * halves * TILE_ELEMS,
                        io.inputs[p].begin() + (k + 1) * halves * TILE_ELEMS);
                }
            }
            for (std::uint16_t& v : want) {
                if (c.pack == 20 && (v & 0x8000)) {
                    v = 0;  // ReLU
                } else if (c.pack == 21) {
                    v = float_to_bf16(2.0f * bf16_to_float(v));  // exact: a doubling
                }
            }
            EXPECT_TRUE(tiles_equal(outs[p], want)) << "run " << run << " output " << p;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Quasar2_0, Quasar2_0Datacopy, ::testing::ValuesIn(copy_cases()), copy_case_name);

}  // namespace
}  // namespace tt::tt_metal
