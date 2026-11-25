// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 compute API tilize (tilize_block) and untilize (copy_block + pack_untilize_dest_block) at SyncHalf,
// 16/32-bit Dest, Float16_b L1, two tile rows each. Checked bit-exact against a host re-layout of the input; outputs
// start as a sentinel. Test names: tilize_b<block>_<dest>, untilize_b<block>_f<full>_<dest>.

#include "quasar_2_0_test_common.hpp"

namespace tt::tt_metal {
namespace {

using namespace quasar_2_0;

struct TilizeCase {
    bool untilize;
    std::uint32_t block;
    std::uint32_t full;  // row width in tiles (tilize: == block)
    bool fp32;
};

std::string tilize_case_name(const ::testing::TestParamInfo<TilizeCase>& info) {
    const TilizeCase& c = info.param;
    return std::string(c.untilize ? "untilize" : "tilize") + "_b" + std::to_string(c.block) +
           (c.untilize ? "_f" + std::to_string(c.full) : "") + (c.fp32 ? "_fp32" : "_fp16b");
}

constexpr std::uint32_t ROWS = 2;  // tile rows

// Row-major element (R, C) of a tensor `width` tiles wide <-> element of its tile (R / 32) * width + C / 32.
std::uint32_t tiled_index(std::uint32_t row, std::uint32_t col, std::uint32_t width) {
    return ((row / 32) * width + col / 32) * TILE_ELEMS + tile_elem(row % 32, col % 32);
}

class Quasar2_0Tilize : public LLKQuasarMeshDeviceSingleCardFixture,
                        public ::testing::WithParamInterface<TilizeCase> {};

TEST_P(Quasar2_0Tilize, Run) {
    const TilizeCase& c = GetParam();
    const std::uint32_t n = ROWS * c.full;
    ProgramIo io;
    io.compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/tilize_untilize_2_0_quasar.cpp";
    io.fp32_dest = c.fp32;
    io.ct_args = {{"op", c.untilize ? 1u : 0u}, {"rows", ROWS}, {"block", c.block}, {"full", c.full}};
    io.inputs.push_back(random_tiles(n, 7 * c.block + c.full + (c.fp32 ? 100 : 0)));
    io.out_tiles.push_back(n);
    io.in_depth.push_back(2 * c.block);
    io.out_depth.push_back(2 * c.full);
    const std::vector<Tiles> outs = run_program(*this->devices_.at(0), io);

    const Tiles& in = io.inputs[0];
    Tiles want(in.size());
    const std::uint32_t width = c.full * 32;
    for (std::uint32_t row = 0; row < ROWS * 32; ++row) {
        for (std::uint32_t col = 0; col < width; ++col) {
            const std::uint32_t rm = row * width + col, tiled = tiled_index(row, col, c.full);
            if (c.untilize) {
                want[rm] = in[tiled];
            } else {
                want[tiled] = in[rm];
            }
        }
    }
    EXPECT_TRUE(tiles_equal(outs[0], want));
}

INSTANTIATE_TEST_SUITE_P(
    Quasar2_0,
    Quasar2_0Tilize,
    ::testing::Values(
        TilizeCase{false, 1, 1, false},
        TilizeCase{false, 4, 4, false},
        TilizeCase{false, 4, 4, true},
        TilizeCase{true, 1, 1, false},
        TilizeCase{true, 4, 4, false},
        TilizeCase{true, 4, 8, false},
        TilizeCase{true, 4, 4, true}),
    tilize_case_name);

}  // namespace
}  // namespace tt::tt_metal
