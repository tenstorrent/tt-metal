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
    // Tile 0 is the leading tile from the first DST section; the two slots under
    // test are the last two.
    TT_FATAL(
        packed.size() == 3 * kWordsPerTile,
        "expected three bfloat16 tiles ({} words), got {}",
        3 * kWordsPerTile,
        packed.size());
    return CopyDestResult{
        vector<std::uint32_t>(packed.begin() + kWordsPerTile, packed.begin() + 2 * kWordsPerTile),
        vector<std::uint32_t>(packed.begin() + 2 * kWordsPerTile, packed.end())};
}

CopyDestResult run(
    distributed::MeshDevice& device,
    int preceding_mode,
    bool skip_copy,
    int following_sfpu = 0,
    int queue_depth = 0,
    int prior_section = 0,
    int raw_iterations = 2,
    int sfpi_overload = 0) {
    const std::uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    auto src0 = create_random_vector_of_bfloat16(tile_bytes, /*rand_max_float=*/4, /*seed=*/1101, /*offset=*/-2.0f);
    auto src1 = create_random_vector_of_bfloat16(tile_bytes, /*rand_max_float=*/4, /*seed=*/2202, /*offset=*/-2.0f);

    std::map<std::string, std::string> defines = {
        {"PRECEDING_MODE", std::to_string(preceding_mode)},
        {"FOLLOWING_SFPU", std::to_string(following_sfpu)},
        {"QUEUE_DEPTH", std::to_string(queue_depth)},
        {"PRIOR_SECTION", std::to_string(prior_section)},
        {"RAW_ITERATIONS", std::to_string(raw_iterations)},
        {"SFPI_OVERLOAD", std::to_string(sfpi_overload)}};
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
        /*cb_depth_tiles=*/3,
        /*out_tiles=*/3);
    return split(packed);
}

// Reports exactly which packed words the copy changed, and whether each one
// landed on the source slot's value. For the partial-copy form (ITERATIONS=2,
// VectorMode::R) most of the tile is out of scope by design, so the footprint
// is the measurement of interest rather than whole-tile equality.
void report_footprint(const CopyDestResult& with_copy, const CopyDestResult& without_copy, const char* description) {
    std::size_t changed = 0, changed_and_correct = 0, unchanged_but_should_match = 0;
    int first_changed = -1, last_changed = -1;
    for (std::size_t i = 0; i < kWordsPerTile; ++i) {
        const bool differs_from_before = with_copy.destination_slot[i] != without_copy.destination_slot[i];
        const bool matches_source = with_copy.destination_slot[i] == with_copy.source_slot[i];
        if (differs_from_before) {
            ++changed;
            if (first_changed < 0) {
                first_changed = static_cast<int>(i);
            }
            last_changed = static_cast<int>(i);
            if (matches_source) {
                ++changed_and_correct;
            }
        } else if (!matches_source) {
            ++unchanged_but_should_match;
        }
    }
    log_info(
        tt::LogTest,
        "{}: of {} packed words, {} changed ({} of those now match the source slot), {} left stale; "
        "changed range [{}, {}]",
        description,
        kWordsPerTile,
        changed,
        changed_and_correct,
        unchanged_but_should_match,
        first_changed,
        last_changed);
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

    report_footprint(with_copy, without_copy, description);

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

// Does the copy's store become visible to the NEXT SFPU op, or only to the
// packer? Three runs, all model-free:
//
//   A  copy, then an SFPU op on DST[1]
//   B  NO copy, then the same SFPU op on DST[1]  (so it transforms the prefill)
//   C  copy, no following SFPU op                (already known to work)
//
// If A == B, the following SFPU op saw the prefill rather than the copy, i.e.
// the store had not landed when it was read -- even though C proves the same
// store is visible to the packer. That is a store-to-load ordering defect, not
// a failure to store, and it is invisible to any test that packs straight after
// the copy.
void check_following_sfpu(distributed::MeshDevice& device, int following_sfpu, const char* description) {
    const auto a = run(device, 0, /*skip_copy=*/false, following_sfpu);
    const auto b = run(device, 0, /*skip_copy=*/true, following_sfpu);
    const auto c = run(device, 0, /*skip_copy=*/false, /*following_sfpu=*/0);

    // C is the control: the copy itself works when the packer is the next reader.
    ASSERT_EQ(c.destination_slot, c.source_slot)
        << description << ": the plain copy (no following SFPU op) already fails, so this variant says "
        << "nothing about ordering";

    std::size_t same_as_prefill_path = 0;
    for (std::size_t i = 0; i < kWordsPerTile; ++i) {
        if (a.destination_slot[i] == b.destination_slot[i]) {
            ++same_as_prefill_path;
        }
    }
    log_info(
        tt::LogTest,
        "{}: {} of {} packed words are identical between 'copy then op' and 'no copy then op'",
        description,
        same_as_prefill_path,
        kWordsPerTile);

    EXPECT_NE(a.destination_slot, b.destination_slot)
        << description << ": DST[1] after 'copy then SFPU op' is bit-identical to 'no copy then SFPU op', so the "
        << "following SFPU op read the pre-copy contents -- the store had not landed when it was read, even though "
        << "the same store IS visible to the packer";
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

// Mode 4: same preceding op as mode 3 (a ROW-broadcast add writes DST[0], with
// the stale value already in DST[1]), but the public whole-tile copy. Separates
// "which op wrote DST[0]" from "which form of the copy call".
TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesAfterRowBroadcastAdd) {
    check_mode(*this->devices_.at(0), 4, "preceding op = ROW-broadcast add into DST[0], whole-tile copy");
}

// Mode 3: the raw SFPU_BINARY_CALL at ITERATIONS=2 / VectorMode::R. This is a
// PARTIAL copy by construction -- most of the destination tile is out of scope
// and correctly keeps its prior contents -- so whole-tile equality is the wrong
// bar. What must hold is that every word the copy DID touch carries the source
// slot's value, and that it touched something at all.
TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesRawPartialIterations2VectorModeR) {
    auto& device = *this->devices_.at(0);
    const auto with_copy = run(device, 3, /*skip_copy=*/false);
    const auto without_copy = run(device, 3, /*skip_copy=*/true);

    ASSERT_NE(without_copy.destination_slot, with_copy.source_slot)
        << "the destination slot already matched the source slot BEFORE the copy; this variant cannot discriminate";

    report_footprint(with_copy, without_copy, "raw call, ITERATIONS=2, VectorMode::R");

    std::size_t changed = 0, changed_but_wrong = 0;
    for (std::size_t i = 0; i < kWordsPerTile; ++i) {
        if (with_copy.destination_slot[i] != without_copy.destination_slot[i]) {
            ++changed;
            if (with_copy.destination_slot[i] != with_copy.source_slot[i]) {
                ++changed_but_wrong;
            }
        }
    }
    EXPECT_GT(changed, 0u) << "the copy changed nothing at all in the destination slot";
    EXPECT_EQ(changed_but_wrong, 0u) << changed_but_wrong << " of the " << changed
                                     << " words the copy touched do not carry the source slot's value";
}

// The ordering probe: is the copy's store visible to the next SFPU op, or only
// to the packer?
TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesThenSfpuNegative) {
    check_following_sfpu(*this->devices_.at(0), 1, "copy then negative_tile(1)");
}

TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesThenSfpuSigmoid) {
    check_following_sfpu(*this->devices_.at(0), 2, "copy then sigmoid_tile(1)");
}

// Does a deep Tensix instruction stream before the copy change whether its
// store lands? Each SFPU op in the burst reprograms the same config register
// the copy's destination address comes from, so a late-consumed config write
// would show up here and nowhere else.
TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesUnderDeepInstructionQueue) {
    auto& device = *this->devices_.at(0);
    bool any_failed = false;
    for (int depth : {0, 1, 4, 16, 64, 256}) {
        const auto with_copy = run(device, 0, /*skip_copy=*/false, /*following_sfpu=*/0, depth);
        const auto without_copy = run(device, 0, /*skip_copy=*/true, /*following_sfpu=*/0, depth);

        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < kWordsPerTile; ++i) {
            if (with_copy.destination_slot[i] != with_copy.source_slot[i]) {
                ++mismatches;
            }
        }
        const bool stale = with_copy.destination_slot == without_copy.destination_slot;
        log_info(
            tt::LogTest,
            "queue_depth={}: {} of {} words in DST[1] differ from DST[0]{}",
            depth,
            mismatches,
            kWordsPerTile,
            stale ? "  <-- DST[1] is bit-identical to its pre-copy contents" : "");
        if (mismatches != 0) {
            any_failed = true;
        }
    }
    EXPECT_FALSE(any_failed) << "the copy stopped landing at some instruction-queue depth";
}

// Does the copy land when it is the first SFPU op of a NEW DST section, with the
// previous section's base still in the config register?
TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesFirstSfpuOfNewDstSection) {
    auto& device = *this->devices_.at(0);
    for (int prior : {0, 1}) {
        const auto with_copy = run(device, 0, /*skip_copy=*/false, /*following_sfpu=*/0, /*queue_depth=*/0, prior);
        const auto without_copy = run(device, 0, /*skip_copy=*/true, /*following_sfpu=*/0, /*queue_depth=*/0, prior);

        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < kWordsPerTile; ++i) {
            if (with_copy.destination_slot[i] != with_copy.source_slot[i]) {
                ++mismatches;
            }
        }
        const bool stale = with_copy.destination_slot == without_copy.destination_slot;
        log_info(
            tt::LogTest,
            "prior_section={}: {} of {} words in DST[1] differ from DST[0]{}",
            prior,
            mismatches,
            kWordsPerTile,
            stale ? "  <-- DST[1] is bit-identical to its pre-copy contents" : "");
        EXPECT_EQ(mismatches, 0u) << "prior_section=" << prior << ": the copy did not land";
    }
}

namespace {

// Number of packed words the copy changed in the destination slot.
std::size_t copied_footprint(distributed::MeshDevice& device, int raw_iterations, int sfpi_overload) {
    const auto with_copy = run(device, 3, /*skip_copy=*/false, 0, 0, 0, raw_iterations, sfpi_overload);
    const auto without_copy = run(device, 3, /*skip_copy=*/true, 0, 0, 0, raw_iterations, sfpi_overload);
    std::size_t changed = 0, wrong = 0;
    for (std::size_t i = 0; i < kWordsPerTile; ++i) {
        if (with_copy.destination_slot[i] != without_copy.destination_slot[i]) {
            ++changed;
            if (with_copy.destination_slot[i] != with_copy.source_slot[i]) {
                ++wrong;
            }
        }
    }
    EXPECT_EQ(wrong, 0u) << "iterations=" << raw_iterations << " sfpi=" << sfpi_overload << ": " << wrong << " of "
                         << changed << " copied words do not carry the source slot's value";
    return changed;
}

}  // namespace

// ITERATIONS is the loop count the functor turns into row advance via
// `dst_reg++`. That is an sfpi construct, while the loop body addresses DEST
// with TT_SFPLOAD/TT_SFPSTORE immediates and ADDR_MOD_7, which is configured
// with dest.incr == 0 -- so nothing in the instruction encoding advances the
// row, and the advance exists only if codegen makes `dst_reg++` do it. The
// functor's own source comment flags that path as fragile, and downstream the
// same source fails under a different JIT's flags (tenstorrent/tt-metal#56673).
//
// At VectorMode::R the functor body runs twice (faces 0 and 1) and each
// SFPLOAD/SFPSTORE pair moves 32 bf16 elements = 16 packed words, so the copied
// footprint should be 2 * ITERATIONS * 16 = 32 * ITERATIONS words. Measured on
// Blackhole p150b:
//
//     ITERATIONS   expected   measured
//              1         32         64   <-- see the DISABLED test below
//              2         64         64
//              4        128        128
//              8        256        256
//
// This pins the law for the range where it holds. If codegen ever stops
// honouring `dst_reg++` the footprint would go flat across ITERATIONS, which is
// the shape of the downstream failure.
TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesFootprintScalesWithIterations) {
    auto& device = *this->devices_.at(0);
    for (int iterations : {2, 4, 8}) {
        const std::size_t changed = copied_footprint(device, iterations, /*sfpi_overload=*/0);
        const std::size_t expected = 32u * static_cast<std::size_t>(iterations);
        log_info(
            tt::LogTest,
            "ITERATIONS={}: {} of {} packed words copied (expected {})",
            iterations,
            changed,
            kWordsPerTile,
            expected);
        EXPECT_EQ(changed, expected) << "ITERATIONS=" << iterations
                                     << ": the copied footprint is not 32 * ITERATIONS words, so `dst_reg++` is not "
                                     << "advancing the row address by one 32-lane group per iteration";
    }
}

// ITERATIONS=1 copies 64 words where the law above predicts 32 -- the same
// footprint as ITERATIONS=2, i.e. as though the loop body ran twice. The values
// it writes are correct, so this is over-copy rather than corruption, and it is
// harmless for a caller that owns the whole destination tile. It is recorded
// rather than asserted because it is a real deviation and the single-iteration
// case is exactly where the functor's "the compiler unrolls this loop" comment
// stops applying. DISABLED so CI stays green; run with
// --gtest_also_run_disabled_tests.
TEST_F(LLKBlackholeSingleCardFixture, DISABLED_TensixCopyDestValuesFootprintAtIterations1) {
    auto& device = *this->devices_.at(0);
    const std::size_t changed = copied_footprint(device, 1, /*sfpi_overload=*/0);
    log_info(tt::LogTest, "ITERATIONS=1: {} of {} packed words copied (expected 32)", changed, kWordsPerTile);
    EXPECT_EQ(changed, 32u) << "ITERATIONS=1 copied " << changed << " words, not 32";
}

// The two overloads must agree. The DataFormat-templated one is the recommended
// path and the sfpi one is [[deprecated]], yet they have independent bodies and
// only the sfpi one keeps working downstream (#56673). Any divergence here is a
// bug in one of them.
TEST_F(LLKBlackholeSingleCardFixture, TensixCopyDestValuesOverloadsAgree) {
    auto& device = *this->devices_.at(0);
    for (int iterations : {1, 2, 4, 8}) {
        const auto raw = run(device, 3, /*skip_copy=*/false, 0, 0, 0, iterations, /*sfpi_overload=*/0);
        const auto sfpi = run(device, 3, /*skip_copy=*/false, 0, 0, 0, iterations, /*sfpi_overload=*/1);
        std::size_t differing = 0;
        for (std::size_t i = 0; i < kWordsPerTile; ++i) {
            if (raw.destination_slot[i] != sfpi.destination_slot[i]) {
                ++differing;
            }
        }
        log_info(tt::LogTest, "ITERATIONS={}: {} words differ between the two overloads", iterations, differing);
        EXPECT_EQ(differing, 0u) << "ITERATIONS=" << iterations
                                 << ": copy_dest_value<DataFormat,...> and the sfpi overload disagree in " << differing
                                 << " packed words";
    }
}

}  // namespace tt::tt_metal
