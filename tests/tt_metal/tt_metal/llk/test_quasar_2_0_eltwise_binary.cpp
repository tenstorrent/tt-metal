// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 compute API eltwise binary (add / sub / mul) over the Dest-sync cases S1-S8 (Phase 1 matrix):
// SyncHalf/SyncFull x 16/32-bit Dest. S6 switches unpack-to-dest copy <-> add mid-kernel. Checked per element
// against a host reference (rtol/atol 0.02); outputs start as a sentinel. Test names: <S>_<op>_<sync>_<dest>.

#include "quasar_2_0_test_common.hpp"

namespace tt::tt_metal {
namespace {

using namespace quasar_2_0;

struct BinaryCase {
    std::uint32_t s;   // sync case S1..S8
    std::uint32_t op;  // 0 add, 1 sub, 2 mul
    bool full_sync;
    bool fp32;
};

constexpr const char* OP_NAMES[] = {"add", "sub", "mul"};

std::string binary_case_name(const ::testing::TestParamInfo<BinaryCase>& info) {
    const BinaryCase& c = info.param;
    return "S" + std::to_string(c.s) + "_" + (c.s == 6 ? "chain" : OP_NAMES[c.op]) + (c.full_sync ? "_full" : "_half") +
           (c.fp32 ? "_fp32" : "_fp16b");
}

std::vector<BinaryCase> binary_cases() {
    std::vector<BinaryCase> cases;
    for (bool full : {false, true}) {
        for (bool fp32 : {false, true}) {
            for (std::uint32_t s = 1; s <= 8; ++s) {
                for (std::uint32_t op = 0; op < 3; ++op) {
                    if (s == 6 && op != 0) {
                        continue;  // S6 = unpack-to-dest copy <-> add
                    }
                    cases.push_back({s, op, full, fp32});
                }
            }
        }
    }
    return cases;
}

float binary_ref(std::uint32_t op, float a, float b) { return op == 0 ? a + b : (op == 1 ? a - b : a * b); }

class Quasar2_0EltwiseBinary : public LLKQuasarMeshDeviceSingleCardFixture,
                               public ::testing::WithParamInterface<BinaryCase> {};

TEST_P(Quasar2_0EltwiseBinary, Run) {
    const BinaryCase& c = GetParam();
    const std::uint32_t cap = dest_capacity(c.full_sync, c.fp32);
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
            tps = cap < 5 ? 3 : 5;
            n = 15;
            break;  // no tails: the block divides the dim (15 = 3 x 5)
        case 5:
            tps = 2;
            n = 32;
            break;  // DFB depth 4 < N
        case 6:
            tps = 2;
            n = 22;
            mode = 6;
            break;  // copy 3, add 5, copy 3 sections
        case 7:
            tps = 2;
            n = 32;
            mode = 7;
            break;       // A op B -> out0, B op A -> out1
        default: break;  // S1, S8 (S1 twice)
    }
    const std::uint32_t b_tiles = mode == 6 ? 5 * tps : n;
    const std::uint32_t runs = c.s == 8 ? 2 : 1;
    for (std::uint32_t run = 0; run < runs; ++run) {
        ProgramIo io;
        io.compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary_2_0_quasar.cpp";
        io.fp32_dest = c.fp32;
        io.full_sync = c.full_sync;
        io.ct_args = {{"num_tiles", n}, {"tiles_per_section", tps}, {"op", c.op}, {"mode", mode}};
        io.inputs = {
            random_tiles(n, 1000 + 100 * c.s + 10 * run + c.op), random_tiles(b_tiles, 2000 + 100 * c.s + run)};
        io.in_depth = {2 * tps, 2 * tps};
        const Tiles& a = io.inputs[0];
        const Tiles& b = io.inputs[1];
        std::vector<std::vector<float>> want;
        auto elem = [&](const Tiles& x, std::uint32_t i) { return bf16_to_float(x[i]); };
        if (mode == 6) {
            std::vector<float> w(n * TILE_ELEMS);
            const std::uint32_t lo = 3 * tps * TILE_ELEMS, hi = 8 * tps * TILE_ELEMS;
            for (std::uint32_t i = 0; i < w.size(); ++i) {
                w[i] = (i >= lo && i < hi) ? binary_ref(c.op, elem(a, i), elem(b, i - lo)) : elem(a, i);
            }
            want = {w};
        } else if (mode == 7) {
            const std::uint32_t half = n / 2 * TILE_ELEMS;
            std::vector<float> w0(half), w1(half);
            for (std::uint32_t i = 0; i < half; ++i) {
                w0[i] = binary_ref(c.op, elem(a, i), elem(b, i));
                w1[i] = binary_ref(c.op, elem(b, half + i), elem(a, half + i));
            }
            want = {w0, w1};
        } else {
            std::vector<float> w(n * TILE_ELEMS);
            for (std::uint32_t i = 0; i < w.size(); ++i) {
                w[i] = binary_ref(c.op, elem(a, i), elem(b, i));
            }
            want = {w};
        }
        for (const auto& w : want) {
            io.out_tiles.push_back(w.size() / TILE_ELEMS);
            io.out_depth.push_back(2 * tps);
        }
        const std::vector<Tiles> outs = run_program(*this->devices_.at(0), io);
        for (std::size_t p = 0; p < want.size(); ++p) {
            EXPECT_TRUE(tiles_close(outs[p], want[p], 0.02f, 0.02f)) << "run " << run << " output " << p;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Quasar2_0, Quasar2_0EltwiseBinary, ::testing::ValuesIn(binary_cases()), binary_case_name);

// Broadcast (bcast.h), SyncHalf / 16-bit Dest, one tile per section. op: 3 sub COL, 4 mul COL, 5 mul ROW (binary:
// A op B[r][0] / B[0][c]); 6 unary ROW, 7 unary COL (A[0][c] / A[r][0]; in1 is consumed but unused).
constexpr const char* BCAST_NAMES[] = {"sub_col", "mul_col", "mul_row", "unary_row", "unary_col"};

class Quasar2_0Bcast : public LLKQuasarMeshDeviceSingleCardFixture,
                       public ::testing::WithParamInterface<std::uint32_t> {};

TEST_P(Quasar2_0Bcast, Run) {
    const std::uint32_t op = GetParam();
    const std::uint32_t n = 8;
    ProgramIo io;
    io.compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary_2_0_quasar.cpp";
    io.ct_args = {{"num_tiles", n}, {"tiles_per_section", 1}, {"op", op}, {"mode", 0}};
    io.inputs = {random_tiles(n, 3000 + op), random_tiles(n, 4000 + op)};
    io.in_depth = {2, 2};
    const Tiles& a = io.inputs[0];
    const Tiles& b = io.inputs[1];
    std::vector<float> w(n * TILE_ELEMS);
    for (std::uint32_t t = 0; t < n; ++t) {
        auto at = [&](const Tiles& x, std::uint32_t r, std::uint32_t c) {
            return bf16_to_float(x[t * TILE_ELEMS + tile_elem(r, c)]);
        };
        for (std::uint32_t r = 0; r < 32; ++r) {
            for (std::uint32_t c = 0; c < 32; ++c) {
                const float x = at(a, r, c);
                float v = 0;
                switch (op) {
                    case 3: v = x - at(b, r, 0); break;
                    case 4: v = x * at(b, r, 0); break;
                    case 5: v = x * at(b, 0, c); break;
                    case 6: v = at(a, 0, c); break;
                    default: v = at(a, r, 0); break;
                }
                w[t * TILE_ELEMS + tile_elem(r, c)] = v;
            }
        }
    }
    io.out_tiles = {n};
    io.out_depth = {2};
    const std::vector<Tiles> outs = run_program(*this->devices_.at(0), io);
    EXPECT_TRUE(tiles_close(outs[0], w, 0.02f, 0.02f));
}

INSTANTIATE_TEST_SUITE_P(
    Quasar2_0, Quasar2_0Bcast, ::testing::Range(3u, 8u), [](const ::testing::TestParamInfo<std::uint32_t>& info) {
        return std::string(BCAST_NAMES[info.param - 3]);
    });

}  // namespace
}  // namespace tt::tt_metal
