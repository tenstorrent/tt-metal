// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar routes a 32-bit integer operand to DEST from its format alone: an Int32 / UInt32 DFB consumed by copy_tile
// with enable_32_bit_dest and NO unpack_modes entry must still pack bit-identical output. On origin/main the Int32
// case hangs (the operand is lowered to SrcA, which Quasar rejects) and UInt32 is refused by the host.

#include "llk_device_fixture.hpp"
#include "single_node_dfb_program.hpp"

#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr std::uint32_t NUM_TILES = 4;
constexpr std::uint32_t WORDS_PER_TILE = 32 * 32;  // 32-bit datums

// Random 32-bit words over the full range: Int32 covers negatives (two's complement), UInt32 the top bit.
std::vector<std::uint32_t> make_stimulus(tt::DataFormat format, std::uint32_t seed) {
    std::vector<std::uint32_t> words(NUM_TILES * WORDS_PER_TILE);
    std::mt19937 rng(seed);
    if (format == tt::DataFormat::Int32) {
        std::uniform_int_distribution<std::int32_t> dist(
            std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max());
        for (auto& w : words) {
            w = static_cast<std::uint32_t>(dist(rng));
        }
    } else {
        std::uniform_int_distribution<std::uint32_t> dist(0u, std::numeric_limits<std::uint32_t>::max());
        for (auto& w : words) {
            w = dist(rng);
        }
    }
    // Pin a few sentinels so sign and magnitude extremes are always covered.
    words[0] = 0x80000000u;
    words[1] = 0x7FFFFFFFu;
    words[2] = 0xFFFFFFFFu;
    words[3] = 0u;
    return words;
}

void run_copy_no_mode_entry(distributed::MeshDevice& device, tt::DataFormat format, bool dst_full_sync_en) {
    namespace single_node_dfb = tt::tt_metal::unit_tests::llk::single_node_dfb;
    const std::vector<std::uint32_t> src = make_stimulus(format, 0x1D32 + static_cast<std::uint32_t>(format));

    // No unpack_modes entry on purpose: the route must come from the operand format.
    const std::vector<std::uint32_t> result = single_node_dfb::run(
        device,
        {.name = "int32_copy_no_mode",
         .inputs = {{.data = src, .format = format, .num_tiles = NUM_TILES, .accessor = "in", .num_entries = 2}},
         .output = {.format = format, .num_tiles = NUM_TILES, .accessor = "out", .num_entries = 2},
         .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/int32_copy_no_mode.cpp",
         .compile_time_args = {{"num_tiles", NUM_TILES}},
         .hw_config = experimental::ComputeHardwareConfig{
             .enable_32_bit_dest = true,
             .double_buffer_dest = !dst_full_sync_en,
         }});

    ASSERT_EQ(result.size(), src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(result[i], src[i]) << "mismatch at word " << i << " (tile " << i / WORDS_PER_TILE << ")";
    }
}

}  // namespace

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarInt32CopyNoModeEntry) {
    for (const bool dst_full_sync_en : {false, true}) {
        SCOPED_TRACE(dst_full_sync_en ? "dst_full_sync_en=true (SyncFull)" : "dst_full_sync_en=false (SyncHalf)");
        run_copy_no_mode_entry(*this->devices_.at(0), tt::DataFormat::Int32, dst_full_sync_en);
    }
}

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarUInt32CopyNoModeEntry) {
    for (const bool dst_full_sync_en : {false, true}) {
        SCOPED_TRACE(dst_full_sync_en ? "dst_full_sync_en=true (SyncFull)" : "dst_full_sync_en=false (SyncHalf)");
        run_copy_no_mode_entry(*this->devices_.at(0), tt::DataFormat::UInt32, dst_full_sync_en);
    }
}
