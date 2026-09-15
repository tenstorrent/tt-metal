// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-logger/tt-logger.hpp>

#include "llk_device_fixture.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {

using std::vector;

namespace {

constexpr const char* kKernel = "tests/tt_metal/tt_metal/test_kernels/compute/copy_dest_values_occupied.cpp";

// One bfloat16 tile is 2048 bytes, i.e. 512 packed uint32 words.
constexpr std::uint32_t kWordsPerTile = 512;

struct CopyDestResult {
    vector<std::uint32_t> source_slot;       // DST[0], the copy source
    vector<std::uint32_t> destination_slot;  // DST[1], the copy destination
};

// Splits run_binary's two output tiles into the two DST slots they came from.
CopyDestResult split(const vector<std::uint32_t>& packed) {
    TT_FATAL(
        packed.size() == 2 * kWordsPerTile,
        "expected two bfloat16 tiles ({} words), got {}",
        2 * kWordsPerTile,
        packed.size());
    return CopyDestResult{
        vector<std::uint32_t>(packed.begin(), packed.begin() + kWordsPerTile),
        vector<std::uint32_t>(packed.begin() + kWordsPerTile, packed.end())};
}

CopyDestResult run(distributed::MeshDevice& device, int preceding_mode, bool skip_copy) {
    const std::uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    auto src0 = create_random_vector_of_bfloat16(tile_bytes, /*rand_max_float=*/4, /*seed=*/1101, /*offset=*/-2.0f);
    auto src1 = create_random_vector_of_bfloat16(tile_bytes, /*rand_max_float=*/4, /*seed=*/2202, /*offset=*/-2.0f);

    std::map<std::string, std::string> defines = {{"PRECEDING_MODE", std::to_string(preceding_mode)}};
    if (skip_copy) {
        defines["SKIP_COPY"] = "1";
    }

    auto packed = unit_tests::llk::single_core::run_binary(
        device,
        src0,
        src1,
        /*num_tiles=*/1,
        kKernel,
        defines,
        /*cb_depth_tiles=*/2,
        /*out_tiles=*/2);
    return split(packed);
}

// Grades one preceding-op mode. Two device runs: one performs the copy, one
// skips it. That makes the check model-free -- there is no host-side golden for
// a matmul, and none is needed, because the contract is a relation between two
// DST slots that both came off the same datapath.
void check_mode(distributed::MeshDevice& device, int preceding_mode, const char* description) {
    const auto with_copy = run(device, preceding_mode, /*skip_copy=*/false);
    const auto without_copy = run(device, preceding_mode, /*skip_copy=*/true);

    // Self-check, and it must come first: if the destination slot already held
    // the source slot's value before the copy, then the contract check below
    // would pass no matter what copy_dest_values did. This is what stops a
    // degenerate setup from looking like a passing test.
    ASSERT_NE(without_copy.destination_slot, with_copy.source_slot)
        << description << ": the destination slot already matched the source slot BEFORE the copy, so this "
        << "variant cannot discriminate. Fix the test, do not read anything into the result.";

    // The source slot must be untouched by the copy -- it is the value the
    // caller still intends to pack.
    EXPECT_EQ(with_copy.source_slot, without_copy.source_slot)
        << description << ": copy_dest_values modified the SOURCE slot DST[0]";

    // The contract.
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < kWordsPerTile; ++i) {
        if (with_copy.destination_slot[i] != with_copy.source_slot[i]) {
            ++mismatches;
        }
    }
    const bool stale = with_copy.destination_slot == without_copy.destination_slot;
    EXPECT_EQ(mismatches, 0u)
        << description << ": after copy_dest_values(0 -> 1), DST[1] differs from DST[0] in " << mismatches << " of "
        << kWordsPerTile << " packed words"
        << (stale ? "; it is bit-identical to what it held before the copy, so the store did not land" : "");
}

}  // namespace

// copy_dest_value is defined for Wormhole and Blackhole alike, but these bars
// have only been measured on Blackhole.
TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesOccupiedByDatacopy) {
    check_mode(*this->devices_.at(0), 0, "preceding op = datacopy into both slots");
}

TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesOccupiedByMatmul) {
    check_mode(*this->devices_.at(0), 1, "preceding op = matmul into both slots");
}

TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesStaleFromTwoOpsBack) {
    check_mode(*this->devices_.at(0), 2, "destination written by a datacopy two ops back, then a matmul wrote DST[0]");
}

}  // namespace tt::tt_metal
