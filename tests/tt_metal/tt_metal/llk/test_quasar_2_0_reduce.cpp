// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 compute API reduce (reduce_init / reduce_tile / reduce_block / reduce_uninit) with Float16_b L1 at 16-bit
// Dest: data in DFB in0, one scaler tile in DFB in1. Checked against a host float reference: the reduced column (ROW),
// row (COL) or element (SCALAR) of each output tile, zero elsewhere (the packer edge mask). Test names:
// <dim>_<pool>_<tile|block><tiles per output>, e.g. row_max_tile2.

#include <algorithm>
#include <cmath>

#include "quasar_2_0_test_common.hpp"

namespace tt::tt_metal {
namespace {

using namespace quasar_2_0;

enum Pool : std::uint32_t { MAX, SUM, AVG };
enum Dim : std::uint32_t { ROW, COL, SCALAR };

struct ReduceCase {
    Dim dim;
    Pool pool;
    bool block;             // reduce_block (each tile to its own DST slot), else reduce_tile accumulating into DST 0
    std::uint32_t n;        // data tiles per output (tile), or per reduce_block call (block)
    std::uint32_t num_out;  // output tiles
};

std::string reduce_case_name(const ::testing::TestParamInfo<ReduceCase>& info) {
    const ReduceCase& c = info.param;
    static const char* dims[] = {"row", "col", "scalar"};
    static const char* pools[] = {"max", "sum", "avg"};
    return std::string(dims[c.dim]) + "_" + pools[c.pool] + (c.block ? "_block" : "_tile") + std::to_string(c.n);
}

// Host reference: output tile o reduces data tiles [o * n, (o + 1) * n) (tile mode) or tile o alone (block mode).
std::vector<float> reduce_ref(const ReduceCase& c, const Tiles& in, float scaler) {
    const std::uint32_t per_out = c.block ? 1 : c.n;
    std::vector<float> want(c.num_out * TILE_ELEMS, 0.0f);
    for (std::uint32_t o = 0; o < c.num_out; ++o) {
        auto x = [&](std::uint32_t t, std::uint32_t r, std::uint32_t col) {
            return bf16_to_float(in[(o * per_out + t) * TILE_ELEMS + tile_elem(r, col)]);
        };
        float* w = &want[o * TILE_ELEMS];
        const std::uint32_t outer = c.dim == SCALAR ? 1 : 32;
        for (std::uint32_t k = 0; k < outer; ++k) {
            float acc = c.pool == MAX ? -INFINITY : 0.0f;
            for (std::uint32_t t = 0; t < per_out; ++t) {
                for (std::uint32_t r = 0; r < 32; ++r) {
                    for (std::uint32_t col = 0; col < 32; ++col) {
                        const bool in_line = c.dim == SCALAR || (c.dim == ROW ? r == k : col == k);
                        if (!in_line) {
                            continue;
                        }
                        const float v = x(t, r, col);
                        acc = c.pool == MAX ? std::max(acc, v) : acc + v * scaler;
                    }
                }
            }
            w[c.dim == SCALAR ? 0 : c.dim == ROW ? tile_elem(k, 0) : tile_elem(0, k)] = acc;
        }
    }
    return want;
}

std::vector<ReduceCase> reduce_cases() {
    return {
        {ROW, MAX, false, 2, 2},  // SDPA row max
        {ROW, SUM, false, 2, 2},  // RMSNorm row sum
        {COL, MAX, false, 2, 2},  // ResNet max pool
        {COL, AVG, false, 2, 2},  // ResNet average pool
        {ROW, AVG, false, 1, 2},
        {COL, SUM, false, 1, 2},
        {SCALAR, MAX, false, 1, 2},
        {SCALAR, SUM, false, 1, 2},
        {ROW, MAX, true, 4, 8},
        {COL, AVG, true, 4, 8},
    };
}

class Quasar2_0Reduce : public LLKQuasarMeshDeviceSingleCardFixture,
                        public ::testing::WithParamInterface<ReduceCase> {};

TEST_P(Quasar2_0Reduce, Run) {
    const ReduceCase& c = GetParam();
    const float scaler = c.pool == AVG ? 1.0f / 32 : 1.0f;
    ProgramIo io;
    io.compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_2_0_quasar.cpp";
    io.ct_args = {{"pool", c.pool}, {"dim", c.dim}, {"num_out", c.num_out}, {"tiles_per_out", c.n}, {"block", c.block}};
    const std::uint32_t in_tiles = c.block ? c.num_out : c.num_out * c.n;
    io.inputs.push_back(random_tiles(in_tiles, 7 * c.dim + c.pool + (c.block ? 50 : 0)));
    io.inputs.push_back(Tiles(TILE_ELEMS, float_to_bf16(scaler)));
    io.out_tiles = {c.num_out};
    io.in_depth = {2 * c.n, 1};
    io.out_depth = {2 * (c.block ? c.n : 1)};
    const std::vector<Tiles> outs = run_program(*this->devices_.at(0), io);
    const bool scalar_sum = c.dim == SCALAR && c.pool != MAX;  // 1024 bf16 partial sums
    EXPECT_TRUE(tiles_close(
        outs[0], reduce_ref(c, io.inputs[0], scaler), scalar_sum ? 0.05f : 0.02f, scalar_sum ? 0.5f : 0.08f));
}

INSTANTIATE_TEST_SUITE_P(Quasar2_0, Quasar2_0Reduce, ::testing::ValuesIn(reduce_cases()), reduce_case_name);

}  // namespace
}  // namespace tt::tt_metal
