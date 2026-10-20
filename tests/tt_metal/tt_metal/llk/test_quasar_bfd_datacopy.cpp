// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar BFD re-architecture POC: buffer descriptors are no longer programmed in
// hw_configure (coupled 1:1 with DFB ids); instead each op's llk_*_init allocates an
// id from its TRISC's partition (T0: [0,16), T2: [16,24), T3: [24,32)) via a
// bump-and-wrap counter and programs the table entry itself. This test copies three
// input DFBs to one output DFB the realistic way -- init once per operand, then a block
// loop of tiles -- and wraps the whole three-operand sequence in an outer loop of
// NUM_LOOPS. Each operand switch bump-allocates a fresh unpack BFD, so NUM_INPUTS*NUM_LOOPS
// (= 30) inits per run wrap the 16-entry unpack partition and reuse ids mid-run. The readers
// re-stream the same TILES_PER_INPUT tiles NUM_LOOPS times (the Quasar tile-counter model
// consumes a tile per unpack, so re-copying requires re-streaming). Every copy, including the
// post-wrap ones, is bit-exact-checked, so a wrapped id that programs the wrong input's
// descriptor is caught.

#include "llk_device_fixture.hpp"
#include "single_node_dfb_program.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "impl/data_format/bfloat16_utils.hpp"
#include "impl/program/program_impl.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr std::uint32_t NUM_INPUTS = 3;
constexpr std::uint32_t TILES_PER_INPUT = 16;  // unique tiles per input, copied as one block
constexpr std::uint32_t NUM_LOOPS = 10;        // stream + copy the same tiles this many times
// NUM_CYCLES = one pass over all inputs (in0..in2, TILES_PER_INPUT each = 48). The compute wraps this in an
// outer loop of NUM_LOOPS, re-initing the unpack BFD once per operand (NUM_INPUTS*NUM_LOOPS = 30 inits > the
// 16-entry unpack partition, so it wraps). The readers re-stream the same TILES_PER_INPUT tiles NUM_LOOPS
// times: the same 16 tiles/input are copied 10x -> TILES_STREAMED_PER_INPUT (160) per input, TOTAL_TILES (480).
constexpr std::uint32_t NUM_CYCLES = NUM_INPUTS * TILES_PER_INPUT;
constexpr std::uint32_t TILES_STREAMED_PER_INPUT = TILES_PER_INPUT * NUM_LOOPS;
constexpr std::uint32_t TOTAL_TILES = NUM_CYCLES * NUM_LOOPS;
}  // namespace

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarBfdDatacopy) {
    namespace single_node_dfb = tt::tt_metal::unit_tests::llk::single_node_dfb;
    const std::uint32_t single_tile_size = 2 * 1024;  // Float16_b 32x32 tile

    // Stimulus: TILES_PER_INPUT distinct random tiles per input, repeated NUM_LOOPS times in DRAM so the
    // readers stream (and the compute copies) the SAME tiles over and over.
    auto make_streamed = [&](std::uint32_t seed) {
        std::vector<std::uint32_t> unique =
            create_random_vector_of_bfloat16(single_tile_size * TILES_PER_INPUT, 1.0f, seed);
        std::vector<std::uint32_t> streamed;
        streamed.reserve(unique.size() * NUM_LOOPS);
        for (std::uint32_t l = 0; l < NUM_LOOPS; ++l) {
            streamed.insert(streamed.end(), unique.begin(), unique.end());
        }
        return streamed;
    };
    std::vector<std::uint32_t> src0_vec = make_streamed(0xC0FFEE);
    std::vector<std::uint32_t> src1_vec = make_streamed(0xDEAD01);
    std::vector<std::uint32_t> src2_vec = make_streamed(0xBEEF02);

    auto input = [](const std::vector<std::uint32_t>& data, const char* accessor) {
        return single_node_dfb::Input{
            .data = data,
            .format = tt::DataFormat::Float16_b,
            .num_tiles = TILES_STREAMED_PER_INPUT,
            .accessor = accessor,
            .num_entries = 2};
    };
    std::vector<std::uint32_t> result_vec = single_node_dfb::run(
        *this->devices_.at(0),
        {.name = "bfd_datacopy",
         .inputs = {input(src0_vec, "in0"), input(src1_vec, "in1"), input(src2_vec, "in2")},
         .output = {.format = tt::DataFormat::Float16_b, .num_tiles = TOTAL_TILES, .accessor = "out", .num_entries = 4},
         .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/bfd_datacopy_quasar.cpp",
         .compile_time_args = {{"num_cycles", NUM_CYCLES}, {"num_loops", NUM_LOOPS}}});

    // Output order = pack order: per outer loop, in0's whole block, then in1's, then in2's (blocked).
    // Within a loop, cycle i copies tile (i % TILES_PER_INPUT) of input (i / TILES_PER_INPUT), taken from
    // that loop's slice of the stream, i.e. stream tile (outer*TILES_PER_INPUT + i % TILES_PER_INPUT).
    const std::vector<std::uint32_t>* srcs[NUM_INPUTS] = {&src0_vec, &src1_vec, &src2_vec};
    const size_t words_per_tile = single_tile_size / sizeof(std::uint32_t);
    std::vector<std::uint32_t> golden;
    golden.reserve(words_per_tile * TOTAL_TILES);
    for (std::uint32_t outer = 0; outer < NUM_LOOPS; ++outer) {
        for (std::uint32_t i = 0; i < NUM_CYCLES; ++i) {
            const std::vector<std::uint32_t>& src = *srcs[i / TILES_PER_INPUT];
            const std::uint32_t tile_idx = outer * TILES_PER_INPUT + (i % TILES_PER_INPUT);
            golden.insert(
                golden.end(), src.begin() + tile_idx * words_per_tile, src.begin() + (tile_idx + 1) * words_per_tile);
        }
    }

    auto comparison_function = [](float a, float b) { return a == b; };  // datacopy is bit-exact
    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, golden, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}
