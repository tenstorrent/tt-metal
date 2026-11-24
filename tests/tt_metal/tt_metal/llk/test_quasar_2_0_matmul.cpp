// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 compute API matmul over the Dest-sync cases S1-S10 (Phase 1 matrix): SyncHalf/SyncFull x 16/32-bit
// Dest. S6 switches unpack-to-dest copy <-> matmul mid-kernel; S10 accumulates kt = 4 inside a full-capacity block;
// S11 copies a tile, then matmul accumulates on it in the same section (copy, FPU, PACK). fp32mix (S6, S11, 32-bit
// Dest): the copy source is a Float32 in2, so the copy unpacks to Dest (UNPACK -> PACK and FPU -> PACK uses on the same
// regions in S6, UNPACK -> FPU -> PACK in S11) while the matmul operands stay Float16_b. S12 is S1 with each in1 tile
// transposed (matmul_init transpose = 1, SDPA Q x K^T).
// The readers stream each section's operand tiles in consumption order. Checked per element against a host
// reference; outputs start as a sentinel. Test names: <S>_<sync>_<dest>.

#include "quasar_2_0_test_common.hpp"

namespace tt::tt_metal {
namespace {

using namespace quasar_2_0;

struct MatmulCase {
    std::uint32_t s;  // sync case S1..S12
    bool full_sync;
    bool fp32;
    bool mix = false;  // Float32 copy source in2 (S6, S11; fp32 Dest only)
};

std::string matmul_case_name(const ::testing::TestParamInfo<MatmulCase>& info) {
    const MatmulCase& c = info.param;
    return "S" + std::to_string(c.s) + (c.full_sync ? "_full" : "_half") +
           (c.mix    ? "_fp32mix"
            : c.fp32 ? "_fp32"
                     : "_fp16b");
}

std::vector<MatmulCase> matmul_cases() {
    std::vector<MatmulCase> cases;
    for (bool full : {false, true}) {
        for (bool fp32 : {false, true}) {
            for (std::uint32_t s = 1; s <= 12; ++s) {
                cases.push_back({s, full, fp32});
                if (fp32 && (s == 6 || s == 11)) {
                    cases.push_back({s, full, true, true});
                }
            }
        }
    }
    return cases;
}

// out += a x b (b^T when tb) for one 32x32 tile pair.
void tile_matmul(
    std::vector<float>& out,
    std::size_t o,
    const Tiles& a,
    std::size_t ai,
    const Tiles& b,
    std::size_t bi,
    bool tb = false) {
    for (std::uint32_t r = 0; r < 32; ++r) {
        for (std::uint32_t c = 0; c < 32; ++c) {
            float acc = 0.0f;
            for (std::uint32_t k = 0; k < 32; ++k) {
                acc += bf16_to_float(a[ai * TILE_ELEMS + tile_elem(r, k)]) *
                       bf16_to_float(b[bi * TILE_ELEMS + (tb ? tile_elem(c, k) : tile_elem(k, c))]);
            }
            out[o * TILE_ELEMS + tile_elem(r, c)] += acc;
        }
    }
}

class Quasar2_0Matmul : public LLKQuasarMeshDeviceSingleCardFixture,
                        public ::testing::WithParamInterface<MatmulCase> {};

TEST_P(Quasar2_0Matmul, Run) {
    const MatmulCase& c = GetParam();
    const std::uint32_t cap = dest_capacity(c.full_sync, c.fp32);
    const std::uint32_t blk_ct = cap >= 16 ? 4 : (cap >= 8 ? 4 : 2), blk_rt = cap / blk_ct;  // ct x rt = capacity
    // tile mode: tps output tiles per section; block mode: ct x rt per section (see matmul_2_0_quasar.cpp)
    std::uint32_t tps = 1, n = 32, mode = 0, block = 0, ct = 1, rt = 1, kt = 1, transpose = 0;
    switch (c.s) {
        case 2:
            block = 1;
            ct = blk_ct;
            rt = blk_rt;
            n = 4 * cap;
            break;
        case 3:
            tps = 3;
            n = 33;
            break;
        case 4:
            block = 1;
            ct = cap < 8 ? cap : 8;
            n = 5 * ct;
            break;  // 1 x ct blocks
        case 5:
            tps = 2;
            n = 32;
            break;  // DFB depth 4 < N
        case 6:
            tps = 2;
            n = 22;
            mode = 6;
            break;  // copy 3, matmul 5, copy 3 sections
        case 7:
            tps = 2;
            n = 32;
            mode = 7;
            break;  // in0 x in1 -> out0, in1 x in0 -> out1
        case 11:
            n = 8;
            mode = 11;
            break;  // copy + matmul per section
        case 10:
            block = 1;
            ct = blk_ct;
            rt = blk_rt;
            kt = 4;
            n = 3 * cap;
            break;  // kt accumulation
        case 12:
            transpose = 1;
            n = 8;
            break;       // in1 transpose
        default: break;  // S1, S8 (S1 twice), S9 (S1 under LLK asserts)
    }
    // Operand tiles each reader streams, and the reference.
    std::uint32_t a_tiles = n * kt, b_tiles = n * kt;
    if (block) {
        a_tiles = n / ct * kt;  // n / (ct * rt) blocks of rt x kt in0 and kt x ct in1 tiles
        b_tiles = n / rt * kt;
    } else if (mode == 6) {
        a_tiles = c.mix ? 5 * tps : n;
        b_tiles = 5 * tps;
    } else if (mode == 11) {
        a_tiles = c.mix ? n : 2 * n;
    }
    const std::uint32_t c_tiles = mode == 6 ? 6 * tps : n;  // copy tiles from in2 (mix)
    const std::uint32_t runs = c.s == 8 ? 2 : 1;
    for (std::uint32_t run = 0; run < runs; ++run) {
        ProgramIo io;
        io.compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_2_0_quasar.cpp";
        io.fp32_dest = c.fp32;
        io.full_sync = c.full_sync;
        io.ct_args = {
            {"num_tiles", n},
            {"tiles_per_section", tps},
            {"kt", kt},
            {"ct", ct},
            {"rt", rt},
            {"block", block},
            {"mode", mode},
            {"transpose", transpose}};
        io.inputs = {random_tiles(a_tiles, 3000 + 100 * c.s + run), random_tiles(b_tiles, 4000 + 100 * c.s + run)};
        if (block) {
            io.in_depth = {2 * rt * kt, 2 * kt * ct};
        } else if (mode == 11) {
            io.in_depth = {c.mix ? 2u : 4u, 2};
        } else {
            io.in_depth = {2 * tps * kt, 2 * tps * kt};
        }
        if (c.mix) {  // Float32 in2: (lo, hi) halves per element
            io.inputs.push_back(random_tiles(2 * c_tiles, 5000 + 100 * c.s + run));
            io.in_depth.push_back(2 * tps);
            io.fp32_in_mask = 1u << 2;
        }
        const Tiles& a = io.inputs[0];
        const Tiles& b = io.inputs[1];
        // Copy source element e of tile t: in0 (bf16), or in2 (Float32) when mixed.
        auto copy_val = [&](std::uint32_t t, std::uint32_t e) {
            if (!c.mix) {
                return bf16_to_float(a[t * TILE_ELEMS + e]);
            }
            const std::size_t i = 2 * (static_cast<std::size_t>(t) * TILE_ELEMS + e);
            const std::uint32_t u = io.inputs[2][i] | (static_cast<std::uint32_t>(io.inputs[2][i + 1]) << 16);
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        };
        std::vector<std::vector<float>> want;
        if (block) {
            std::vector<float> w(n * TILE_ELEMS, 0.0f);
            std::uint32_t o = 0, ai = 0, bi = 0;
            while (o < n) {
                for (std::uint32_t r = 0; r < rt; ++r) {
                    for (std::uint32_t cc = 0; cc < ct; ++cc) {
                        for (std::uint32_t k = 0; k < kt; ++k) {
                            tile_matmul(w, o + r * ct + cc, a, ai + r * kt + k, b, bi + k * ct + cc);
                        }
                    }
                }
                o += ct * rt;
                ai += rt * kt;
                bi += kt * ct;
            }
            want = {w};
        } else if (mode == 6) {
            std::vector<float> w(n * TILE_ELEMS, 0.0f);
            for (std::uint32_t t = 0; t < n; ++t) {
                if (t >= 3 * tps && t < 8 * tps) {
                    tile_matmul(w, t, a, c.mix ? t - 3 * tps : t, b, t - 3 * tps);
                } else {
                    const std::uint32_t ci = c.mix && t >= 8 * tps ? t - 5 * tps : t;  // in2 skips the matmul tiles
                    for (std::uint32_t e = 0; e < TILE_ELEMS; ++e) {
                        w[t * TILE_ELEMS + e] = copy_val(ci, e);
                    }
                }
            }
            want = {w};
        } else if (mode == 11) {
            std::vector<float> w(n * TILE_ELEMS, 0.0f);
            for (std::uint32_t t = 0; t < n; ++t) {
                for (std::uint32_t e = 0; e < TILE_ELEMS; ++e) {
                    w[t * TILE_ELEMS + e] = copy_val(c.mix ? t : 2 * t, e);
                }
                tile_matmul(w, t, a, c.mix ? t : 2 * t + 1, b, t);
            }
            want = {w};
        } else if (mode == 7) {
            std::vector<float> w0(n / 2 * TILE_ELEMS, 0.0f), w1(n / 2 * TILE_ELEMS, 0.0f);
            for (std::uint32_t t = 0; t < n / 2; ++t) {
                tile_matmul(w0, t, a, t, b, t);
                tile_matmul(w1, t, b, n / 2 + t, a, n / 2 + t);
            }
            want = {w0, w1};
        } else {
            std::vector<float> w(n * TILE_ELEMS, 0.0f);
            for (std::uint32_t t = 0; t < n; ++t) {
                for (std::uint32_t k = 0; k < kt; ++k) {
                    tile_matmul(w, t, a, t * kt + k, b, t * kt + k, transpose != 0);
                }
            }
            want = {w};
        }
        for (const auto& w : want) {
            io.out_tiles.push_back(w.size() / TILE_ELEMS);
            io.out_depth.push_back(2 * (block ? ct * rt : tps));
        }
        const std::vector<Tiles> outs = run_program(*this->devices_.at(0), io);
        for (std::size_t p = 0; p < want.size(); ++p) {
            EXPECT_TRUE(tiles_close(outs[p], want[p], 0.05f, 0.15f)) << "run " << run << " output " << p;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Quasar2_0, Quasar2_0Matmul, ::testing::ValuesIn(matmul_cases()), matmul_case_name);

}  // namespace
}  // namespace tt::tt_metal
