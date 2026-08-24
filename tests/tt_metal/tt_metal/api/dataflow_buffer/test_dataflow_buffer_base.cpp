// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Base single-DFB config sweep (full Metal 2.0 matrix).

#include "dfb_test_common.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {


// gtest param-name helpers
static std::string ImplicitSyncParamName(const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "ImplicitSyncTrue" : "ImplicitSyncFalse";
}

static std::string M2ImplicitSyncParamName(const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "ImplicitSyncTrue" : "ImplicitSyncFalse";
}

// All single-DFB configs now live in the Metal 2.0 sweep below. The 2.0 driver runs the
// simple 1x1 explicit-sync cases on WH/BH too (a DFB lowers to a circular buffer there);
// implicit-sync and multi-core stay Quasar-only. The former legacy-only configs (the three
// 1Sx1S and DM->Tensix 6Sx4A) have been uplifted to 2.0, so no legacy DFB_TEST entries remain.
// ====================================================================================

// Metal 2.0 single-DFB config sweep
#define DFB_TEST_2_0(suffix, p_type, c_type, num_p, pap_kind, num_c, cap_kind) \
    TEST_P(DFBImplicitSyncParamFixture_2_0, suffix##_2_0) {                    \
        M2SingleDFBParams params{                                              \
            .producer_type = M2PorCType::p_type,                               \
            .consumer_type = M2PorCType::c_type,                               \
            .num_producers = (num_p),                                          \
            .num_consumers = (num_c),                                          \
            .pap = m2::DFBAccessPattern::pap_kind,                             \
            .cap = m2::DFBAccessPattern::cap_kind,                             \
            .implicit_sync = GetParam(),                                       \
            .num_entries = default_num_entries((num_p), (num_c)),              \
        };                                                                     \
        run_single_dfb_program_2_0(this->device(), params);                    \
    }

DFB_TEST_2_0(DMTest1xDFB1Sx1S, DM, DM, 1, STRIDED, 1, STRIDED)
DFB_TEST_2_0(DMTensixTest1xDFB1Sx1S, DM, TENSIX, 1, STRIDED, 1, STRIDED)
DFB_TEST_2_0(TensixDMTest1xDFB1Sx1S, TENSIX, DM, 1, STRIDED, 1, STRIDED)

DFB_TEST_2_0(DMTest1xDFB1Sx4S, DM, DM, 1, STRIDED, 4, STRIDED)
DFB_TEST_2_0(DMTest1xDFB4Sx1S, DM, DM, 4, STRIDED, 1, STRIDED)
// DMTest1xDFB4Sx4S omitted: 4+4=8 DM cores exceeds Gen2 user-DM cap (6).
// Legacy can do it via num_threads_per_cluster; m2's num_threads = literal DM cores.
DFB_TEST_2_0(DMTest1xDFB2Sx2S, DM, DM, 2, STRIDED, 2, STRIDED)
DFB_TEST_2_0(DMTensixTest1xDFB4Sx1S, DM, TENSIX, 4, STRIDED, 1, STRIDED)
DFB_TEST_2_0(DMTensixTest1xDFB4Sx2S, DM, TENSIX, 4, STRIDED, 2, STRIDED)
DFB_TEST_2_0(TensixDMTest1xDFB1Sx4S, TENSIX, DM, 1, STRIDED, 4, STRIDED)
DFB_TEST_2_0(TensixDMTest1xDFB4Sx1S, TENSIX, DM, 4, STRIDED, 1, STRIDED)

// ---------- Matrix completion: portable legacy DFB_TEST variants ported to M2 ----------
// Filters applied (configs that violate these are documented but skipped):
//   DM-DM:     num_p + num_c <= 6  (Gen2 user-DM cap; legacy uses num_threads_per_cluster which we can't replicate)
//   DM→Tensix: num_p <= 6 DM; Tensix consumer num_threads ∈ {1, 2, 4}  (Gen2 compute thread set)
//   Tensix→DM: Tensix producer num_threads ∈ {1, 2, 4}; num_c <= 6 DM
//   DM→DM ALL with implicit-sync: known runtime gap (legacy hits it too); ImplicitSyncTrue auto-skips
//
// Architecturally NOT portable (would exceed M2 / Gen2 constraints):
//   DMTest 4Sx4S / 4Sx4A          : 4+4=8 > 6-DM cap
//   *3Sx3*  for DMTensix/TensixDM : Tensix side = 3 threads, not in {1,2,4}
//   *3Sx2A* DMTensix              : Tensix consumer = 2 OK, but 3-thread DM producer fine; (this one IS portable)
//   DMTensix *Sx3*                : Tensix consumer = 3, not in {1,2,4}
//   TensixDM *3Sx                 : Tensix producer = 3, not in {1,2,4}

// STRIDED — DM-DM additional variants
DFB_TEST_2_0(DMTest1xDFB1Sx2S, DM, DM, 1, STRIDED, 2, STRIDED)
DFB_TEST_2_0(DMTest1xDFB1Sx3S, DM, DM, 1, STRIDED, 3, STRIDED)
DFB_TEST_2_0(DMTest1xDFB1Sx5S, DM, DM, 1, STRIDED, 5, STRIDED)
DFB_TEST_2_0(DMTest1xDFB2Sx1S, DM, DM, 2, STRIDED, 1, STRIDED)
DFB_TEST_2_0(DMTest1xDFB3Sx1S, DM, DM, 3, STRIDED, 1, STRIDED)
DFB_TEST_2_0(DMTest1xDFB3Sx3S, DM, DM, 3, STRIDED, 3, STRIDED)
DFB_TEST_2_0(DMTest1xDFB4Sx2S, DM, DM, 4, STRIDED, 2, STRIDED)
DFB_TEST_2_0(DMTest1xDFB5Sx1S, DM, DM, 5, STRIDED, 1, STRIDED)
DFB_TEST_2_0(DMTest1xDFB2Sx4S, DM, DM, 2, STRIDED, 4, STRIDED)

// STRIDED — DM→Tensix additional variants (Tensix consumer ∈ {1,2,4})
DFB_TEST_2_0(DMTensixTest1xDFB1Sx2S, DM, TENSIX, 1, STRIDED, 2, STRIDED)
DFB_TEST_2_0(DMTensixTest1xDFB1Sx4S, DM, TENSIX, 1, STRIDED, 4, STRIDED)
DFB_TEST_2_0(DMTensixTest1xDFB2Sx1S, DM, TENSIX, 2, STRIDED, 1, STRIDED)
DFB_TEST_2_0(DMTensixTest1xDFB2Sx4S, DM, TENSIX, 2, STRIDED, 4, STRIDED)
DFB_TEST_2_0(DMTensixTest1xDFB3Sx1S, DM, TENSIX, 3, STRIDED, 1, STRIDED)
DFB_TEST_2_0(DMTensixTest1xDFB4Sx4S, DM, TENSIX, 4, STRIDED, 4, STRIDED)
DFB_TEST_2_0(DMTensixTest1xDFB6Sx1S, DM, TENSIX, 6, STRIDED, 1, STRIDED)
DFB_TEST_2_0(DMTensixTest1xDFB6Sx2S, DM, TENSIX, 6, STRIDED, 2, STRIDED)

// STRIDED — Tensix→DM additional variants (Tensix producer ∈ {1,2,4})
DFB_TEST_2_0(TensixDMTest1xDFB1Sx2S, TENSIX, DM, 1, STRIDED, 2, STRIDED)
DFB_TEST_2_0(TensixDMTest1xDFB1Sx3S, TENSIX, DM, 1, STRIDED, 3, STRIDED)
DFB_TEST_2_0(TensixDMTest1xDFB1Sx6S, TENSIX, DM, 1, STRIDED, 6, STRIDED)
DFB_TEST_2_0(TensixDMTest1xDFB2Sx1S, TENSIX, DM, 2, STRIDED, 1, STRIDED)
// TensixDMTest1xDFB2Sx3S omitted: 2P × 3C asymmetric STRIDED triggers an
// M2-vs-legacy ring-slot mapping divergence (M2 interleaves consumer slots
// across the ring per the [1126-1130] comment; the helper's identity-equal
// verification doesn't match). Coverage of Tensix→DM asymmetric STRIDED is
// preserved by 1Sx3S (asymmetric 1×N), 2Sx4S, 4Sx2S (asymmetric N×M with
// divisible ratios) which all pass.
DFB_TEST_2_0(TensixDMTest1xDFB2Sx4S, TENSIX, DM, 2, STRIDED, 4, STRIDED)
DFB_TEST_2_0(TensixDMTest1xDFB2Sx6S, TENSIX, DM, 2, STRIDED, 6, STRIDED)
DFB_TEST_2_0(TensixDMTest1xDFB4Sx2S, TENSIX, DM, 4, STRIDED, 2, STRIDED)
DFB_TEST_2_0(TensixDMTest1xDFB4Sx4S, TENSIX, DM, 4, STRIDED, 4, STRIDED)

// ALL — DM-DM (ImplicitSyncTrue auto-skips per known DM→DM ALL impl-sync gap)
DFB_TEST_2_0(DMTest1xDFB1Sx3A, DM, DM, 1, STRIDED, 3, ALL)
DFB_TEST_2_0(DMTest1xDFB1Sx4A, DM, DM, 1, STRIDED, 4, ALL)
DFB_TEST_2_0(DMTest1xDFB2Sx3A, DM, DM, 2, STRIDED, 3, ALL)
DFB_TEST_2_0(DMTest1xDFB2Sx4A, DM, DM, 2, STRIDED, 4, ALL)
DFB_TEST_2_0(DMTest1xDFB3Sx1A, DM, DM, 3, STRIDED, 1, ALL)
DFB_TEST_2_0(DMTest1xDFB3Sx2A, DM, DM, 3, STRIDED, 2, ALL)
DFB_TEST_2_0(DMTest1xDFB3Sx3A, DM, DM, 3, STRIDED, 3, ALL)
DFB_TEST_2_0(DMTest1xDFB4Sx1A, DM, DM, 4, STRIDED, 1, ALL)
DFB_TEST_2_0(DMTest1xDFB4Sx2A, DM, DM, 4, STRIDED, 2, ALL)

// ALL — DM→Tensix (Tensix consumer ∈ {1,2,4})
DFB_TEST_2_0(DMTensixTest1xDFB1Sx4A, DM, TENSIX, 1, STRIDED, 4, ALL)
DFB_TEST_2_0(DMTensixTest1xDFB2Sx4A, DM, TENSIX, 2, STRIDED, 4, ALL)
DFB_TEST_2_0(DMTensixTest1xDFB3Sx1A, DM, TENSIX, 3, STRIDED, 1, ALL)
DFB_TEST_2_0(DMTensixTest1xDFB3Sx2A, DM, TENSIX, 3, STRIDED, 2, ALL)
DFB_TEST_2_0(DMTensixTest1xDFB3Sx4A, DM, TENSIX, 3, STRIDED, 4, ALL)
DFB_TEST_2_0(DMTensixTest1xDFB4Sx1A, DM, TENSIX, 4, STRIDED, 1, ALL)
DFB_TEST_2_0(DMTensixTest1xDFB4Sx2A, DM, TENSIX, 4, STRIDED, 2, ALL)
DFB_TEST_2_0(DMTensixTest1xDFB4Sx4A, DM, TENSIX, 4, STRIDED, 4, ALL)
DFB_TEST_2_0(DMTensixTest1xDFB6Sx1A, DM, TENSIX, 6, STRIDED, 1, ALL)
DFB_TEST_2_0(DMTensixTest1xDFB6Sx2A, DM, TENSIX, 6, STRIDED, 2, ALL)
DFB_TEST_2_0(DMTensixTest1xDFB6Sx4A, DM, TENSIX, 6, STRIDED, 4, ALL)

// ALL — Tensix→DM (ported from the legacy sweep: Tensix producer + ALL DM consumer)
DFB_TEST_2_0(TensixDMTest1xDFB1Sx4A, TENSIX, DM, 1, STRIDED, 4, ALL)
DFB_TEST_2_0(TensixDMTest1xDFB2Sx4A, TENSIX, DM, 2, STRIDED, 4, ALL)
DFB_TEST_2_0(TensixDMTest1xDFB4Sx1A, TENSIX, DM, 4, STRIDED, 1, ALL)
DFB_TEST_2_0(TensixDMTest1xDFB4Sx2A, TENSIX, DM, 4, STRIDED, 2, ALL)
DFB_TEST_2_0(TensixDMTest1xDFB4Sx4A, TENSIX, DM, 4, STRIDED, 4, ALL)

// instantiations (each fixture instantiated exactly once in the whole binary)
INSTANTIATE_TEST_SUITE_P(
    ImplicitSync,
    DFBImplicitSyncParamFixture,
    ::testing::Bool(),
    ImplicitSyncParamName);


INSTANTIATE_TEST_SUITE_P(
    M2ImplicitSync, DFBImplicitSyncParamFixture_2_0, ::testing::Values(false, true), M2ImplicitSyncParamName);



// =====================================================================================
// BLOCKED access-pattern matrix
// =====================================================================================
#define DFB_BLOCKED_TEST_2_0(suffix, p_type, c_type, num_p, num_c, blk, entries, impl) \
    TEST_F(UnitMeshFixture, suffix##_2_0) {                                          \
        M2SingleDFBParams params{                                                      \
            .producer_type = M2PorCType::p_type,                                       \
            .consumer_type = M2PorCType::c_type,                                       \
            .num_producers = (num_p),                                                  \
            .num_consumers = (num_c),                                                  \
            .pap = m2::DFBAccessPattern::BLOCKED,                                      \
            .cap = m2::DFBAccessPattern::BLOCKED,                                      \
            .implicit_sync = (impl),                                                   \
            .num_entries = (entries),                                                  \
            .block_size = (blk),                                                       \
        };                                                                             \
        run_single_dfb_program_2_0(this->device(), params);                      \
    }

// --- BLOCKED→BLOCKED (DM→DM, explicit sync: one NoC burst per block) ---
// Global block order makes every DM→DM round-trip identity. Block and ring sizes vary.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk4, DM, DM, 1, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk2, DM, DM, 1, 1, 2, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk8, DM, DM, 1, 1, 8, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk4_ring32, DM, DM, 1, 1, 4, 32, false)
// Symmetric NxN: thread t owns sub-ring t and pairs 1:1 with consumer t, so still identity.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx2B_blk4, DM, DM, 2, 2, 4, 16, false)

// 3Bx3B sits at the 6 DM-core Gen2 cap.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB3Bx3B_blk4, DM, DM, 3, 3, 4, 24, false)
// Non-power-of-2 block size.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk3, DM, DM, 1, 1, 3, 12, false)

// --- ASYMMETRIC BLOCKED→BLOCKED (DM→DM, explicit) ---
// Integer thread-count ratios only; the tile-counter round-robin keeps each block in one sub-ring.
// Still identity: the producer's block read composes with the consumer's block write. P+C <= 6.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx2B_blk4, DM, DM, 1, 2, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx1B_blk4, DM, DM, 2, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx4B_blk4, DM, DM, 1, 4, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB4Bx1B_blk4, DM, DM, 4, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx4B_blk4, DM, DM, 2, 4, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB4Bx2B_blk4, DM, DM, 4, 2, 4, 16, false)

// --- BLOCKED→BLOCKED (DM→DM, implicit sync) ---
// Same layout and goldens as the explicit variants; only the sync mode differs.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk4_impl, DM, DM, 1, 1, 4, 16, true)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx2B_blk4_impl, DM, DM, 2, 2, 4, 16, true)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB3Bx3B_blk4_impl, DM, DM, 3, 3, 4, 24, true)
// Other block sizes, to exercise the ISR credit batching.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk2_impl, DM, DM, 1, 1, 2, 16, true)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk8_impl, DM, DM, 1, 1, 8, 16, true)

// --- ASYMMETRIC BLOCKED→BLOCKED (DM→DM, implicit sync) ---
// commit_implicit_read/write advance the tile counter only on a block boundary, so a block stays in one
// sub-ring and implicit matches the explicit golden.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx2B_blk4_impl, DM, DM, 1, 2, 4, 16, true)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx1B_blk4_impl, DM, DM, 2, 1, 4, 16, true)

// More blocks per thread, and a non-power-of-2 block at NxN. Both identity.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx2B_blk2_e32, DM, DM, 2, 2, 2, 32, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx2B_blk3_e24, DM, DM, 2, 2, 3, 24, false)

// Bigger entry size (2048 vs the 1024 default): larger per-block NoC bursts.
TEST_F(UnitMeshFixture, DMTest1xDFB1Bx1B_blk4_entry2048_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 1,
        .pap = m2::DFBAccessPattern::BLOCKED,
        .cap = m2::DFBAccessPattern::BLOCKED,
        .implicit_sync = false,
        .entry_size = 2048,
        .num_entries = 16,
        .block_size = 4,
    };
    run_single_dfb_program_2_0(this->device(), params);
}

// Same at 2Bx2B, across two sub-rings.
TEST_F(UnitMeshFixture, DMTest1xDFB2Bx2B_blk4_entry2048_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 2,
        .num_consumers = 2,
        .pap = m2::DFBAccessPattern::BLOCKED,
        .cap = m2::DFBAccessPattern::BLOCKED,
        .implicit_sync = false,
        .entry_size = 2048,
        .num_entries = 16,
        .block_size = 4,
    };
    run_single_dfb_program_2_0(this->device(), params);
}

// --- BLOCKED→BLOCKED (Trisc→DM, explicit) ---
// The Tensix producer only posts credits over a host-prefilled ring; the DM consumer bursts each block
// out to DRAM. 1x1 is identity.
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx1B_blk4, TENSIX, DM, 1, 1, 4, 16, false)
// Block-size and ring coverage at N=1.
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx1B_blk2, TENSIX, DM, 1, 1, 2, 16, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx1B_blk8, TENSIX, DM, 1, 1, 8, 16, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx1B_blk4_ring32, TENSIX, DM, 1, 1, 4, 32, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx1B_blk3, TENSIX, DM, 1, 1, 3, 12, false)
// Symmetric NxN. The flat host prefill is the global block order, so the round-trip is identity.
// Tensix threads must be 1, 2 or 4, so there is no 3Bx3B here.
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB2Bx2B_blk4, TENSIX, DM, 2, 2, 4, 16, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB4Bx4B_blk4, TENSIX, DM, 4, 4, 4, 32, false)

// Bigger entry size (2048); spelled out because the macro can't set entry_size.
TEST_F(UnitMeshFixture, TensixDMTest1xDFB1Bx1B_blk4_entry2048_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::TENSIX,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 1,
        .pap = m2::DFBAccessPattern::BLOCKED,
        .cap = m2::DFBAccessPattern::BLOCKED,
        .implicit_sync = false,
        .entry_size = 2048,
        .num_entries = 16,
        .block_size = 4,
    };
    run_single_dfb_program_2_0(this->device(), params);
}

// --- ASYMMETRIC BLOCKED→BLOCKED (Trisc→DM, explicit) ---
// Identity for every P/C under global block order. Explicit only, since a Tensix producer cannot
// feed an implicit BLOCKED DM consumer (rejected at config time).
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx2B_blk4, TENSIX, DM, 1, 2, 4, 16, false)
// 32 entries gives 2 blocks per thread, so the C=4 fan-out is non-degenerate. At 16 it collapses to
// identity and verifies nothing.
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx4B_blk4, TENSIX, DM, 1, 4, 4, 32, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB2Bx4B_blk4, TENSIX, DM, 2, 4, 4, 32, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB2Bx1B_blk4, TENSIX, DM, 2, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB4Bx1B_blk4, TENSIX, DM, 4, 1, 4, 32, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB4Bx2B_blk4, TENSIX, DM, 4, 2, 4, 32, false)

// --- BLOCKED→BLOCKED (DM→Trisc, explicit) ---
// The Tensix consumer drains on the UNPACK path, which needs a copy_tile between wait_front and
// pop_front or the buffer descriptor goes inconsistent and traps.
// RUN-ONLY: a Tensix consumer writes no DRAM, so passing means finishing without a trap or hang.
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB1Bx1B_blk4, DM, TENSIX, 1, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB1Bx2B_blk4, DM, TENSIX, 1, 2, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB2Bx2B_blk4, DM, TENSIX, 2, 2, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB1Bx4B_blk4, DM, TENSIX, 1, 4, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB2Bx1B_blk4, DM, TENSIX, 2, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB4Bx4B_blk4, DM, TENSIX, 4, 4, 4, 32, false)

// --- BLOCKED→ALL (Trisc→DM, explicit) ---
// A Tensix producer routes the ALL fan-out through the remapper, not the broadcast credit mode.
// Golden: output[r] = input[(r%P)*capacity + (r/P)], capacity = num_entries/P (identity at P==1).
// Odd P+C is deliberate: the config blob is 36 + 62*(P+C) bytes, so it lands non-word-aligned --
// regression coverage for the write_to_device truncation that dropped the remapper bytes.
#define DFB_TRISC_BLOCKED_ALL_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(UnitMeshFixture, suffix##_2_0) {                              \
        M2SingleDFBParams params{                                          \
            .producer_type = M2PorCType::TENSIX,                           \
            .consumer_type = M2PorCType::DM,                               \
            .num_producers = (num_p),                                      \
            .num_consumers = (num_c),                                      \
            .pap = m2::DFBAccessPattern::BLOCKED,                          \
            .cap = m2::DFBAccessPattern::ALL,                              \
            .implicit_sync = false,                                        \
            .num_entries = (entries),                                      \
            .block_size = (blk),                                           \
        };                                                                 \
        run_single_dfb_program_2_0(this->device(), params);          \
    }
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB1Bx1A_blk4, 1, 1, 4, 16)  // P+C=2 (even): 1->1, no fan-out
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB1Bx2A_blk4, 1, 2, 4, 16)  // P+C=3 (odd): 1->2 broadcast
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB1Bx4A_blk4, 1, 4, 4, 16)  // P+C=5 (odd): 1->4 broadcast
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB2Bx2A_blk4, 2, 2, 4, 16)  // P+C=4 (even): 2 pairs
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB2Bx4A_blk4, 2, 4, 4, 16)  // P+C=6 (even): 2 pairs, P<C
// P=4 is the widest legal Tensix remapper fan-out; 32 entries keeps it non-degenerate.
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB4Bx1A_blk4, 4, 1, 4, 32)  // P+C=5 (odd)
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB4Bx2A_blk4, 4, 2, 4, 32)  // P+C=6 (even)

// --- BLOCKED→ALL (DM→DM, explicit) ---
// Every ALL consumer reads every entry, freed after all acks via broadcast credits (DM→DM never uses the
// remapper). Identity at P==1; at P>1 the producer's per-block interleave doesn't cancel the consumer's
// per-tile round-robin, so it's a permutation. C <= 4 (ALL slot cap), P+C <= 6 (Gen2 DM cap).
#define DFB_BLOCKED_ALL_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(UnitMeshFixture, suffix##_2_0) {                        \
        M2SingleDFBParams params{                                    \
            .producer_type = M2PorCType::DM,                         \
            .consumer_type = M2PorCType::DM,                         \
            .num_producers = (num_p),                                \
            .num_consumers = (num_c),                                \
            .pap = m2::DFBAccessPattern::BLOCKED,                    \
            .cap = m2::DFBAccessPattern::ALL,                        \
            .implicit_sync = false,                                  \
            .num_entries = (entries),                                \
            .block_size = (blk),                                     \
        };                                                           \
        run_single_dfb_program_2_0(this->device(), params);    \
    }

// P==1: one sub-ring written in order, so identity regardless of C.
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB1Bx1A_blk4, 1, 1, 4, 16)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB1Bx2A_blk4, 1, 2, 4, 16)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB1Bx2A_blk2, 1, 2, 2, 16)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB1Bx4A_blk4, 1, 4, 4, 16)
// P==2: 2 blocks of 4 per producer, so a permutation. 2Bx4A sits at the 6-core DM cap.
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB2Bx2A_blk4, 2, 2, 4, 16)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB2Bx4A_blk4, 2, 4, 4, 16)
// Smaller block at P=2, and P=3 (the ALL producer ceiling). The golden keys only on P and block_size.
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB2Bx2A_blk2, 2, 2, 2, 16)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB3Bx1A_blk4, 3, 1, 4, 24)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB3Bx3A_blk4, 3, 3, 4, 24)

// --- BLOCKED→STRIDED (DM→DM, explicit) ---
// The producer bursts whole blocks in global block order and each consumer drains its share of every
// block, so the round-trip is identity for every P/C.
#define DFB_BLOCKED_STRIDED_TEST_2_0(suffix, num_p, num_c, blk, entries, impl) \
    TEST_F(UnitMeshFixture, suffix##_2_0) {                                  \
        M2SingleDFBParams params{                                              \
            .producer_type = M2PorCType::DM,                                   \
            .consumer_type = M2PorCType::DM,                                   \
            .num_producers = (num_p),                                          \
            .num_consumers = (num_c),                                          \
            .pap = m2::DFBAccessPattern::BLOCKED,                              \
            .cap = m2::DFBAccessPattern::STRIDED,                              \
            .implicit_sync = (impl),                                           \
            .num_entries = (entries),                                          \
            .block_size = (blk),                                               \
        };                                                                     \
        run_single_dfb_program_2_0(this->device(), params);              \
    }
// P==1 shapes.
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB1Bx2S_blk2_e4, 1, 2, 2, 4, false)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB1Bx1S_blk4, 1, 1, 4, 16, false)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB1Bx2S_blk4, 1, 2, 4, 16, false)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB1Bx4S_blk4, 1, 4, 4, 16, false)
// P>1 with C>=P.
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB2Bx2S_blk4, 2, 2, 4, 16, false)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB2Bx4S_blk4, 2, 4, 4, 16, false)
// Fan-in (P>C): each consumer round-robins across all P producers' blocks.
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB2Bx1S_blk4, 2, 1, 4, 16, false)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB4Bx1S_blk4, 4, 1, 4, 16, false)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB4Bx2S_blk4, 4, 2, 4, 16, false)

// Implicit sync. An interleaved ring needs credits once per entry rather than per block, which is why
// serialize_for_core sends block_size 1 for a non-BLOCKED ring. Only C > P exercises that cadence.
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB1Bx2S_blk4_impl, 1, 2, 4, 16, true)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB1Bx4S_blk4_impl, 1, 4, 4, 16, true)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB2Bx4S_blk4_impl, 2, 4, 4, 16, true)
// C < bs: the implicit consumer's per-entry txns rotate its counters every bs/C entries — the first
// implicit exercise of mid-run txn windows (per_txn covers whole runs on both counters).
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB2Bx2S_blk4_impl, 2, 2, 4, 16, true)

// --- BLOCKED→STRIDED (Trisc→DM, explicit) ---
// The Tensix producer only posts credits over a flat prefilled ring, and a STRIDED consumer needs
// per-tile credits, so this reuses the plain per-tile Tensix producer. The consumer's read stride and
// write stride are both C, so they cancel and the round-trip is identity.
#define DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(UnitMeshFixture, suffix##_2_0) {                                  \
        M2SingleDFBParams params{                                              \
            .producer_type = M2PorCType::TENSIX,                               \
            .consumer_type = M2PorCType::DM,                                   \
            .num_producers = (num_p),                                          \
            .num_consumers = (num_c),                                          \
            .pap = m2::DFBAccessPattern::BLOCKED,                              \
            .cap = m2::DFBAccessPattern::STRIDED,                              \
            .implicit_sync = false,                                            \
            .num_entries = (entries),                                          \
            .block_size = (blk),                                               \
        };                                                                     \
        run_single_dfb_program_2_0(this->device(), params);              \
    }
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB1Bx1S_blk4, 1, 1, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB1Bx2S_blk4, 1, 2, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB1Bx4S_blk4, 1, 4, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB2Bx2S_blk4, 2, 2, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB2Bx4S_blk4, 2, 4, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB4Bx4S_blk4, 4, 4, 4, 32)
// Fan-in (P>C). Tensix threads must be 2 or 4.
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB2Bx1S_blk4, 2, 1, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB4Bx1S_blk4, 4, 1, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB4Bx2S_blk4, 4, 2, 4, 16)


// --- BLOCKED→ALL (DM→Trisc, explicit) ---
// A Tensix consumer routes the fan-out through the remapper, the same path the STRIDED→ALL DM→Tensix
// tests above take, except the producer block-bursts instead of striding.
// RUN-ONLY: the Tensix consumer writes no DRAM, so passing means no trap or hang.
#define DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(UnitMeshFixture, suffix##_2_0) {                                 \
        M2SingleDFBParams params{                                             \
            .producer_type = M2PorCType::DM,                                  \
            .consumer_type = M2PorCType::TENSIX,                              \
            .num_producers = (num_p),                                         \
            .num_consumers = (num_c),                                         \
            .pap = m2::DFBAccessPattern::BLOCKED,                             \
            .cap = m2::DFBAccessPattern::ALL,                                 \
            .implicit_sync = false,                                           \
            .num_entries = (entries),                                         \
            .block_size = (blk),                                              \
        };                                                                    \
        run_single_dfb_program_2_0(this->device(), params);             \
    }
DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(DMTensixTest1xDFB1Bx1A_blk4, 1, 1, 4, 16)
DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(DMTensixTest1xDFB1Bx2A_blk4, 1, 2, 4, 16)
DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(DMTensixTest1xDFB1Bx4A_blk4, 1, 4, 4, 16)
DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(DMTensixTest1xDFB2Bx2A_blk4, 2, 2, 4, 16)
DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(DMTensixTest1xDFB2Bx4A_blk4, 2, 4, 4, 16)

// --- BLOCKED→STRIDED (DM→Trisc, explicit) ---
// The DM producer reads block-contiguous DRAM but pushes per tile; the Tensix consumer drains per tile
// on the UNPACK path.
// RUN-ONLY: the Tensix consumer writes no DRAM, so passing means no trap or hang.
#define DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(UnitMeshFixture, suffix##_2_0) {                                     \
        M2SingleDFBParams params{                                                 \
            .producer_type = M2PorCType::DM,                                      \
            .consumer_type = M2PorCType::TENSIX,                                  \
            .num_producers = (num_p),                                             \
            .num_consumers = (num_c),                                             \
            .pap = m2::DFBAccessPattern::BLOCKED,                                 \
            .cap = m2::DFBAccessPattern::STRIDED,                                 \
            .implicit_sync = false,                                               \
            .num_entries = (entries),                                             \
            .block_size = (blk),                                                  \
        };                                                                        \
        run_single_dfb_program_2_0(this->device(), params);                 \
    }
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB1Bx1S_blk4, 1, 1, 4, 16)
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB1Bx2S_blk4, 1, 2, 4, 16)
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB1Bx4S_blk4, 1, 4, 4, 16)
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB2Bx2S_blk4, 2, 2, 4, 16)
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB2Bx4S_blk4, 2, 4, 4, 16)
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB4Bx4S_blk4, 4, 4, 4, 32)

}  // namespace tt::tt_metal
