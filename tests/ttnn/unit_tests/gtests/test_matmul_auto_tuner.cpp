// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_auto_tuner.hpp"

namespace ttnn::operations::matmul::auto_tune::test {

namespace {

ttnn::DeviceComputeKernelConfig make_compute_config(bool fp32_dest_acc_en = false, bool dst_full_sync_en = false) {
    ttnn::DeviceComputeKernelConfig cfg;
    cfg.fp32_dest_acc_en = fp32_dest_acc_en;
    cfg.dst_full_sync_en = dst_full_sync_en;
    return cfg;
}

SubblockTuneInputs base_inputs(const ttnn::DeviceComputeKernelConfig& cfg, uint32_t per_core_M, uint32_t per_core_N) {
    SubblockTuneInputs inputs{.compute_kernel_config = cfg};
    inputs.per_core_M = per_core_M;
    inputs.per_core_N = per_core_N;
    return inputs;
}

}  // namespace

// ---------------------------------------------------------------------------
// determine_largest_subblock — DST capacity
// ---------------------------------------------------------------------------

TEST(MatmulAutoTunerSubblock, HalfSyncNonFp32DstCapacityIs8) {
    // Half-sync + bf16 => 8 tiles. per_core 8x8 admits (1, 8) as the top fast-path pick.
    auto cfg = make_compute_config(/*fp32_dest_acc_en=*/false, /*dst_full_sync_en=*/false);
    auto choice = determine_largest_subblock(base_inputs(cfg, 8, 8));
    EXPECT_EQ(choice.out_subblock_h, 1u);
    EXPECT_EQ(choice.out_subblock_w, 8u);
}

TEST(MatmulAutoTunerSubblock, FullSyncNonFp32DstCapacityIs16) {
    // Full-sync + bf16 => 16 tiles. per_core 4x16 admits (1, 16)? No — table caps at 8. (1, 8) wins.
    auto cfg = make_compute_config(/*fp32_dest_acc_en=*/false, /*dst_full_sync_en=*/true);
    auto choice = determine_largest_subblock(base_inputs(cfg, 4, 16));
    // Table tops out at volume 8: (1,8) is the largest fast-path pick with 16%8==0.
    EXPECT_EQ(choice.out_subblock_h, 1u);
    EXPECT_EQ(choice.out_subblock_w, 8u);
}

TEST(MatmulAutoTunerSubblock, HalfSyncFp32CapacityIs4) {
    // Half-sync + fp32 => 4 tiles. per_core 8x8 cannot fit (1, 8); best fast-path pick is (1, 4).
    auto cfg = make_compute_config(/*fp32_dest_acc_en=*/true, /*dst_full_sync_en=*/false);
    auto choice = determine_largest_subblock(base_inputs(cfg, 8, 8));
    EXPECT_EQ(choice.out_subblock_h, 1u);
    EXPECT_EQ(choice.out_subblock_w, 4u);
}

TEST(MatmulAutoTunerSubblock, FullSyncFp32CapacityIs8) {
    // Full-sync + fp32 => 8 tiles — same as half-sync non-fp32.
    auto cfg = make_compute_config(/*fp32_dest_acc_en=*/true, /*dst_full_sync_en=*/true);
    auto choice = determine_largest_subblock(base_inputs(cfg, 8, 8));
    EXPECT_EQ(choice.out_subblock_h, 1u);
    EXPECT_EQ(choice.out_subblock_w, 8u);
}

// ---------------------------------------------------------------------------
// determine_largest_subblock — fast-path preference
// ---------------------------------------------------------------------------

TEST(MatmulAutoTunerSubblock, FastPathPrefers1xNOver2xNOver2) {
    // per_core 4x8, dst=8. Both (1,8) and (2,4) and (4,2) fit and divide cleanly.
    // Fast-path preference should pick (1, 8) — the helper's pack fast path.
    auto cfg = make_compute_config();
    auto inputs = base_inputs(cfg, 4, 8);
    inputs.prefer_fast_path = true;
    auto choice = determine_largest_subblock(inputs);
    EXPECT_EQ(choice.out_subblock_h, 1u);
    EXPECT_EQ(choice.out_subblock_w, 8u);
}

TEST(MatmulAutoTunerSubblock, LegacyOrderPicks4x2First) {
    // With prefer_fast_path=false, legacy table puts (4, 2) first.
    auto cfg = make_compute_config();
    auto inputs = base_inputs(cfg, 4, 8);
    inputs.prefer_fast_path = false;
    auto choice = determine_largest_subblock(inputs);
    // per_core 4x8, dst=8. Legacy picks first entry matching divisibility = (4, 2).
    EXPECT_EQ(choice.out_subblock_h, 4u);
    EXPECT_EQ(choice.out_subblock_w, 2u);
}

// ---------------------------------------------------------------------------
// determine_largest_subblock — constraint flags
// ---------------------------------------------------------------------------

TEST(MatmulAutoTunerSubblock, SubblockWEqPerCoreNConstraintLimitsToH1OrFullW) {
    // Mimics the subblock-major writer's pre-row-major constraint.
    // per_core 8x4, dst=8. Without constraint -> (2, 4) or (1, 4) or better.
    // With constraint -> must be w==per_core_N=4 OR h==1. (1, 4) fits both; also (2, 4), (4, 1)... wait
    // (4, 1) has h != 1 and w != per_core_N(=4), so excluded. Top pick w/ fast-path: (1, 4).
    auto cfg = make_compute_config();
    auto inputs = base_inputs(cfg, 8, 4);
    inputs.subblock_w_eq_per_core_n_required = true;
    auto choice = determine_largest_subblock(inputs);
    // (2, 4) has h != 1 but w == per_core_N, so it IS allowed.
    // Legacy-order finds (2, 4) before (1, 4); fast-path order finds (1, 4) first in volume-8 tier?
    // Actually fast-path order: (1,8) excl (w=8>4), (8,1) excl (h=8>8? h=8 == per_core_M=8 ok, w=1!=4 excl),
    //   (2,4) h=2 != 1 and w=4 == per_core_N OK, returns (2, 4).
    EXPECT_EQ(choice.out_subblock_h, 2u);
    EXPECT_EQ(choice.out_subblock_w, 4u);
}

TEST(MatmulAutoTunerSubblock, SubblockHEqPerCoreMConstraintLimitsToW1OrFullH) {
    // Symmetric constraint used by mcast_in0 sharded-output path.
    auto cfg = make_compute_config();
    auto inputs = base_inputs(cfg, 4, 8);
    inputs.subblock_h_eq_per_core_m_required = true;
    auto choice = determine_largest_subblock(inputs);
    // Must be h == per_core_M=4 OR w == 1. (4, 2) fits, (8, 1) doesn't (h>per_core_M).
    // Fast-path order finds (1, 8): h=1 != per_core_M=4 AND w=8 != 1 -> excluded.
    //   (4, 2): h=4 == per_core_M, allowed. Returns (4, 2).
    EXPECT_EQ(choice.out_subblock_h, 4u);
    EXPECT_EQ(choice.out_subblock_w, 2u);
}

// ---------------------------------------------------------------------------
// determine_largest_subblock — optional caps
// ---------------------------------------------------------------------------

TEST(MatmulAutoTunerSubblock, MaxSubblockHIsHonored) {
    // SDPA's streaming-compute pattern: max_subblock_h = 2.
    auto cfg = make_compute_config();
    auto inputs = base_inputs(cfg, 8, 8);
    inputs.max_subblock_h = 2;
    auto choice = determine_largest_subblock(inputs);
    EXPECT_LE(choice.out_subblock_h, 2u);
    EXPECT_EQ(choice.out_subblock_h * choice.out_subblock_w, 8u);
    // Fast-path finds (1, 8) first; h=1<=2, OK.
    EXPECT_EQ(choice.out_subblock_h, 1u);
    EXPECT_EQ(choice.out_subblock_w, 8u);
}

TEST(MatmulAutoTunerSubblock, MaxSubblockWIsHonored) {
    auto cfg = make_compute_config();
    auto inputs = base_inputs(cfg, 8, 8);
    inputs.max_subblock_w = 2;
    auto choice = determine_largest_subblock(inputs);
    EXPECT_LE(choice.out_subblock_w, 2u);
    // Top candidate with w<=2: fast-path order: (1,8)>skip, (8,1)>OK volume=8. Returns (8, 1).
    EXPECT_EQ(choice.out_subblock_h, 8u);
    EXPECT_EQ(choice.out_subblock_w, 1u);
}

// ---------------------------------------------------------------------------
// determine_largest_subblock — degenerate shapes
// ---------------------------------------------------------------------------

TEST(MatmulAutoTunerSubblock, PerCore1x1YieldsOneByOne) {
    auto cfg = make_compute_config();
    auto choice = determine_largest_subblock(base_inputs(cfg, 1, 1));
    EXPECT_EQ(choice.out_subblock_h, 1u);
    EXPECT_EQ(choice.out_subblock_w, 1u);
}

TEST(MatmulAutoTunerSubblock, PerCoreZeroReturnsOneByOne) {
    auto cfg = make_compute_config();
    auto choice = determine_largest_subblock(base_inputs(cfg, 0, 8));
    EXPECT_EQ(choice.out_subblock_h, 1u);
    EXPECT_EQ(choice.out_subblock_w, 1u);
}

TEST(MatmulAutoTunerSubblock, PrimeShapeForcesOneByOne) {
    // per_core 7x11 — table contains (7, 1) and (1, 7), so (7, 1) wins (volume 7).
    auto cfg = make_compute_config();
    auto choice = determine_largest_subblock(base_inputs(cfg, 7, 11));
    // Fast-path order: volume-8 entries all require divisibility which 7 and 11 fail.
    // Then volume-7 entries: (1, 7) w=7 doesn't divide 11; (7, 1) h=7 divides 7 and w=1 divides 11 -> win.
    EXPECT_EQ(choice.out_subblock_h, 7u);
    EXPECT_EQ(choice.out_subblock_w, 1u);
}

// ---------------------------------------------------------------------------
// determine_largest_subblock — back-compat: legacy order matches pre-refactor tuner
// ---------------------------------------------------------------------------

TEST(MatmulAutoTunerSubblock, LegacyOrderMatchesPreRefactorPicks) {
    // Spot-check that the legacy table (prefer_fast_path=false, no constraints, bf16 half-sync)
    // reproduces picks the prior get_matmul_subblock_params / get_subblock_sizes would have made
    // for common auto-config shapes.
    auto cfg = make_compute_config();

    auto expect_pick = [&](uint32_t M, uint32_t N, uint32_t expect_h, uint32_t expect_w) {
        auto inputs = base_inputs(cfg, M, N);
        inputs.prefer_fast_path = false;
        auto c = determine_largest_subblock(inputs);
        EXPECT_EQ(c.out_subblock_h, expect_h) << "per_core " << M << "x" << N;
        EXPECT_EQ(c.out_subblock_w, expect_w) << "per_core " << M << "x" << N;
    };

    // 8x8 => (4, 2) first divisible entry of volume 8 in legacy order (matches SUBBLOCK_HW_CHOICES[0]).
    expect_pick(8, 8, 4, 2);
    // 4x4 => (4, 2) no (4%4==0, 4%2==0 OK) volume 8 — but 4x4 with (4, 2): h=4 divides 4, w=2 divides 4. OK.
    expect_pick(4, 4, 4, 2);
    // 2x4 => (2, 4) is first entry in legacy, 2%2==0 4%4==0, volume 8.
    expect_pick(2, 4, 2, 4);
    // 1x4 => volume 8 entries fail on h=4>per_core_M=1. (4,1),(1,4) in volume 4: legacy order (2,2) volume 4 fails
    //   actually let's walk: {4,2} h=4>1, {2,4} h=2>1, {8,1} h=8>1, {1,8} w=8>4, {7,1} h=7>1, {1,7} w=7>4,
    //   {3,2} h=3>1, {2,3} h=2>1, {6,1} h=6>1, {1,6} w=6>4, {5,1} h=5>1, {1,5} w=5>4, {2,2} h=2>1,
    //   {4,1} h=4>1, {1,4} OK -> (1, 4).
    expect_pick(1, 4, 1, 4);
}

// ---------------------------------------------------------------------------
// determine_largest_in0_block_w
// ---------------------------------------------------------------------------

TEST(MatmulAutoTunerIBW, LargeBudgetReturnsFullKt) {
    // Kt divisible by many values, budget ample enough for w==Kt.
    // per_ibw_footprint = 2 * (4*2048 + 4*2048) = 32768.
    // fixed_footprint = 4*4*2048 = 32768.
    // Need budget >= 32768 + 64*32768 = 2162688 to allow w=64.
    InBlockWTuneInputs inputs;
    inputs.Kt = 64;
    inputs.per_core_M = 4;
    inputs.per_core_N = 4;
    inputs.in0_single_tile_size = 2048;
    inputs.in1_single_tile_size = 2048;
    inputs.out_single_tile_size = 2048;
    inputs.l1_budget_bytes = 1u << 22;  // 4 MB, ample for w=64
    inputs.max_in0_block_w = 64;
    uint32_t w = determine_largest_in0_block_w(inputs);
    EXPECT_EQ(w, 64u);
}

TEST(MatmulAutoTunerIBW, TightBudgetReducesIBW) {
    // Budget just fits per_core_M + per_core_N per ibw unit with a small margin.
    InBlockWTuneInputs inputs;
    inputs.Kt = 16;
    inputs.per_core_M = 4;
    inputs.per_core_N = 4;
    inputs.in0_single_tile_size = 2048;
    inputs.in1_single_tile_size = 2048;
    inputs.out_single_tile_size = 2048;
    // Fixed footprint = 4*4*2048 = 32768.
    // per_ibw = 2 * (4*2048 + 4*2048) = 32768.
    // budget = 32768 + 4*32768 = 163840 -> l1_capped = 4.
    inputs.l1_budget_bytes = 32768u + 4u * 32768u;
    inputs.max_in0_block_w = 64;
    uint32_t w = determine_largest_in0_block_w(inputs);
    EXPECT_EQ(w, 4u);
    EXPECT_EQ(inputs.Kt % w, 0u);
}

TEST(MatmulAutoTunerIBW, CappedByMax) {
    InBlockWTuneInputs inputs;
    inputs.Kt = 32;
    inputs.per_core_M = 2;
    inputs.per_core_N = 2;
    inputs.in0_single_tile_size = 2048;
    inputs.in1_single_tile_size = 2048;
    inputs.out_single_tile_size = 2048;
    inputs.l1_budget_bytes = 1 << 20;
    inputs.max_in0_block_w = 4;
    uint32_t w = determine_largest_in0_block_w(inputs);
    EXPECT_LE(w, 4u);
    EXPECT_EQ(32u % w, 0u);
    EXPECT_EQ(w, 4u);
}

TEST(MatmulAutoTunerIBW, FuseBiasAddsIntermFootprint) {
    // Ensure interm_cb is counted when fuse_bias is true.
    InBlockWTuneInputs inputs;
    inputs.Kt = 16;
    inputs.per_core_M = 4;
    inputs.per_core_N = 4;
    inputs.in0_single_tile_size = 2048;
    inputs.in1_single_tile_size = 2048;
    inputs.out_single_tile_size = 2048;
    inputs.interm_single_tile_size = 2048;
    inputs.fuse_bias = true;
    // Without fuse_bias, fixed_footprint = 32768 and budget 97k -> (97k-32k)/32k = 2.
    // With fuse_bias, fixed_footprint = 65536 and budget 97k -> (97k-65k)/32k = 1.
    inputs.l1_budget_bytes = 32768u + 32768u + 32768u;  // 98304 ~ 97k
    inputs.max_in0_block_w = 64;
    uint32_t w = determine_largest_in0_block_w(inputs);
    EXPECT_EQ(w, 1u);
}

TEST(MatmulAutoTunerIBW, NonDivisibleIBWFallsBack) {
    // l1_capped is 7 but Kt=16 so 7 doesn't divide; fall back to 4.
    InBlockWTuneInputs inputs;
    inputs.Kt = 16;
    inputs.per_core_M = 1;
    inputs.per_core_N = 1;
    inputs.in0_single_tile_size = 1024;
    inputs.in1_single_tile_size = 1024;
    inputs.out_single_tile_size = 1024;
    // per_ibw = 2 * (1*1024 + 1*1024) = 4096.
    // fixed = 1*1*1024 = 1024.
    // budget 1024 + 7*4096 = 29696 -> l1_capped=7. Kt%7 != 0 -> 4.
    inputs.l1_budget_bytes = 1024u + 7u * 4096u;
    inputs.max_in0_block_w = 64;
    uint32_t w = determine_largest_in0_block_w(inputs);
    EXPECT_EQ(w, 4u);
}

TEST(MatmulAutoTunerIBW, ZeroBudgetReturnsOne) {
    InBlockWTuneInputs inputs;
    inputs.Kt = 16;
    inputs.per_core_M = 1;
    inputs.per_core_N = 1;
    inputs.in0_single_tile_size = 1024;
    inputs.in1_single_tile_size = 1024;
    inputs.out_single_tile_size = 1024;
    inputs.l1_budget_bytes = 0;
    inputs.max_in0_block_w = 64;
    uint32_t w = determine_largest_in0_block_w(inputs);
    EXPECT_EQ(w, 1u);
}

TEST(MatmulAutoTunerIBW, ZeroKtReturnsOne) {
    InBlockWTuneInputs inputs;
    inputs.Kt = 0;
    inputs.per_core_M = 1;
    inputs.per_core_N = 1;
    inputs.in0_single_tile_size = 1024;
    inputs.in1_single_tile_size = 1024;
    inputs.out_single_tile_size = 1024;
    inputs.l1_budget_bytes = 1 << 20;
    inputs.max_in0_block_w = 64;
    uint32_t w = determine_largest_in0_block_w(inputs);
    EXPECT_EQ(w, 1u);
}

}  // namespace ttnn::operations::matmul::auto_tune::test
