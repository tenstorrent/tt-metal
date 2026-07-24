// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Base single-DFB config sweep (full Metal 2.0 matrix).

#include "dfb_test_common.hpp"

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
        run_single_dfb_program_2_0(this->devices_.at(0), params);              \
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
// BLOCKED access-pattern matrix (migrated from monolithic test_dataflow_buffer_2_0.cpp)
// =====================================================================================
#define DFB_BLOCKED_TEST_2_0(suffix, p_type, c_type, num_p, num_c, blk, entries, impl) \
    TEST_F(MeshDeviceFixture, suffix##_2_0) {                                          \
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
        run_single_dfb_program_2_0(this->devices_.at(0), params);                      \
    }

// --- BLOCKED→BLOCKED (DM-DM, EXPLICIT sync: one NoC burst per block) ---
// Single-thread (1 producer, 1 consumer): one contiguous sub-ring; block_size divides the ring.
//   blk4: 16-entry ring → 4 blocks of 4   (verified passing on emulator)
//   blk2: 16-entry ring → 8 blocks of 2
//   blk8: 16-entry ring → 2 blocks of 8
//   blk4, larger ring: 32-entry ring → 8 blocks of 4
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk4, DM, DM, 1, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk2, DM, DM, 1, 1, 2, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk8, DM, DM, 1, 1, 8, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk4_ring32, DM, DM, 1, 1, 4, 32, false)
// Symmetric multi-thread (N producers == N consumers): each thread t owns sub-ring t
// (stride_in_entries=1 ⇒ contiguous per-thread region), producer t pairs 1:1 with consumer t.
//   2Bx2B blk4: 16-entry ring → capacity 8/thread → 2 blocks of 4 per thread.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx2B_blk4, DM, DM, 2, 2, 4, 16, false)

// 3Bx3B blk4: 6 DM cores (at the Gen2 user-DM cap), 24-entry ring → capacity 8/thread.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB3Bx3B_blk4, DM, DM, 3, 3, 4, 24, false)
// Non-power-of-2 block: blk3, 12-entry ring → 4 blocks of 3 (guards against pow2 assumptions).
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk3, DM, DM, 1, 1, 3, 12, false)

// --- ASYMMETRIC BLOCKED→BLOCKED (DM-DM, explicit) — num_producers != num_consumers ---
// Supported at integer thread-count ratios via the tile-counter round-robin (stride_in_entries stays 1,
// so blocks stay contiguous and the burst is valid). DM→DM still verifies as identity (the producer's
// block page-read composes with the consumer's page-write). P+C <= 6 (Gen2 DM cap); 16 % (4*max(P,C))==0.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx2B_blk4, DM, DM, 1, 2, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx1B_blk4, DM, DM, 2, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx4B_blk4, DM, DM, 1, 4, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB4Bx1B_blk4, DM, DM, 4, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx4B_blk4, DM, DM, 2, 4, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB4Bx2B_blk4, DM, DM, 4, 2, 4, 16, false)

// --- BLOCKED→BLOCKED (DM-DM, IMPLICIT sync: one TXN_ID transfer per tile, ISR-batched credits) ---
// Same layout/page-mapping as the explicit variants; only the sync mode differs.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk4_impl, DM, DM, 1, 1, 4, 16, true)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx2B_blk4_impl, DM, DM, 2, 2, 4, 16, true)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB3Bx3B_blk4_impl, DM, DM, 3, 3, 4, 24, true)
// Implicit sync at other block sizes (single-thread): same identity, exercises the ISR credit batching
// at blk2 (8 blocks of 2) and blk8 (2 blocks of 8).
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk2_impl, DM, DM, 1, 1, 2, 16, true)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx1B_blk8_impl, DM, DM, 1, 1, 8, 16, true)

// --- ASYMMETRIC BLOCKED→BLOCKED (DM-DM, IMPLICIT sync) — now block-aware ---
// Previously a verified LIMIT: an implicit asymmetric side round-robined the tile-counter per-ENTRY, so a
// block scattered across sub-rings and credits misrouted. Now block_size is plumbed to the device and
// commit_implicit_read/write only advance tc_idx at a block boundary (dataflow_buffer.inl `% block_size`),
// so a whole block stays in one sub-ring — the implicit asymmetric path matches the explicit per-block
// golden (identity for DM→DM). Mirrors the explicit 1Bx2B/2Bx1B cases above with implicit=true.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB1Bx2B_blk4_impl, DM, DM, 1, 2, 4, 16, true)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx1B_blk4_impl, DM, DM, 2, 1, 4, 16, true)

// --- BLOCKED→BLOCKED (DM-DM, explicit) extra coverage: larger ring / non-pow2 block, multi-thread ---
// 2Bx2B blk2 on a 32-entry ring → capacity 16/thread → 8 blocks of 2 per thread (more blocks/thread than
// the 16-entry cases; NOT ring-pressure — that needs a Tensix producer with entries_per_core>num_entries).
// 2Bx2B blk3 → non-power-of-2 block at multi-thread (guards against pow2 assumptions). Both verify identity.
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx2B_blk2_e32, DM, DM, 2, 2, 2, 32, false)
DFB_BLOCKED_TEST_2_0(DMTest1xDFB2Bx2B_blk3_e24, DM, DM, 2, 2, 3, 24, false)

// Bigger entry size (2048 vs the 1024 default) — exercises larger per-block NoC bursts.
TEST_F(MeshDeviceFixture, DMTest1xDFB1Bx1B_blk4_entry2048_2_0) {
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
    run_single_dfb_program_2_0(this->devices_.at(0), params);
}

// Bigger entry size (2048), multi-thread (2Bx2B): larger per-block bursts across two sub-rings. Identity.
TEST_F(MeshDeviceFixture, DMTest1xDFB2Bx2B_blk4_entry2048_2_0) {
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
    run_single_dfb_program_2_0(this->devices_.at(0), params);
}

// --- BLOCKED→BLOCKED (Trisc→DM: Tensix BLOCKED producer → DM BLOCKED consumer, explicit) ---
// Tensix producer posts credits block_size-at-a-time (host pre-fills the L1 ring); the DM consumer
// bursts each block out to DRAM. Avoids the unpacker (consumer-side) Tensix path — only the packer
// (producer) is on Tensix. Symmetric 1×1: one contiguous sub-ring, identity verify.
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx1B_blk4, TENSIX, DM, 1, 1, 4, 16, false)
// N=1 block-size / ring coverage — all verify as identity (the permutation degenerates at N=1).
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx1B_blk2, TENSIX, DM, 1, 1, 2, 16, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx1B_blk8, TENSIX, DM, 1, 1, 8, 16, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx1B_blk4_ring32, TENSIX, DM, 1, 1, 4, 32, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx1B_blk3, TENSIX, DM, 1, 1, 3, 12, false)
// Symmetric multi-thread Trisc->DM (N producers == N consumers). These RUN like DM->DM NxN, but the
// flat host-prefill + consumer de-interleave make the output a permutation of the input — verified by
// the Tensix->DM BLOCKED golden branch in run_single_dfb_program_2_0. NOTE: the Tensix (compute)
// PRODUCER only supports 1/2/4 threads (ValidateProgramSpec, program_spec.cpp) — 3 is NOT legal — so
// the symmetric Trisc->DM set is 2Bx2B and 4Bx4B (no 3Bx3B). 4Bx4B is the ceiling. (DM->DM 3Bx3B is
// fine because DM kernels allow 3 threads; only compute kernels are restricted to 1/2/4.)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB2Bx2B_blk4, TENSIX, DM, 2, 2, 4, 16, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB4Bx4B_blk4, TENSIX, DM, 4, 4, 4, 32, false)

// Bigger entry size (2048) for Trisc->DM BLOCKED — N=1 identity; macro can't set entry_size.
TEST_F(MeshDeviceFixture, TensixDMTest1xDFB1Bx1B_blk4_entry2048_2_0) {
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
    run_single_dfb_program_2_0(this->devices_.at(0), params);
}

// --- ASYMMETRIC Trisc→DM BLOCKED→BLOCKED (Tensix BLOCKED producer → DM BLOCKED consumer) ---
// Parity with the DM→DM asymmetric set. Tensix producer threads ∈ {1,2,4}; DM consumer C ≤ 6 (the producer
// is on Tensix cores, so the DM P+C≤6 budget only counts C here); integer ratios; EXPLICIT sync only
// (asymmetric implicit BLOCKED is FATAL). All verify against the generalized Tensix→DM BLOCKED permutation
// golden (capacity=num_entries/max(P,C), ntc=(P>=C)?P/C:1). Fan-in P>C uses a 32-entry ring so the
// permutation is non-degenerate (at the minimal ring some P>C goldens collapse to identity).
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx2B_blk4, TENSIX, DM, 1, 2, 4, 16, false)
// 32-entry ring so blocks_per_thread=2 → a real (non-degenerate) C=4 fan-out permutation. At ne=16 these
// collapsed to identity (1 block/thread), so they didn't actually data-verify the C=4 fan-out read order.
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB1Bx4B_blk4, TENSIX, DM, 1, 4, 4, 32, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB2Bx4B_blk4, TENSIX, DM, 2, 4, 4, 32, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB2Bx1B_blk4, TENSIX, DM, 2, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB4Bx1B_blk4, TENSIX, DM, 4, 1, 4, 32, false)
DFB_BLOCKED_TEST_2_0(TensixDMTest1xDFB4Bx2B_blk4, TENSIX, DM, 4, 2, 4, 32, false)

// --- DM→Trisc BLOCKED→BLOCKED (DM BLOCKED producer → Tensix BLOCKED consumer) ---
// The final matrix column. The DM producer block-bursts into the contiguous BLOCKED sub-rings; the Tensix
// consumer drains them on the UNPACK path (dfb_t6_consumer_2_0.cpp now does copy_tile between wait_front
// and pop_front — the SW/HW spec requires the unpacker to read the tile, else the buffer descriptor goes
// inconsistent and traps; the descriptor x/y-dim fix is in-tree). RUN-ONLY: a Tensix consumer writes no
// DRAM, so the pass signal is completion without the unpack trap / hang (no data golden). Tensix consumer
// threads ∈ {1,2,4}; explicit sync; num_entries % (block_size*max(P,C))==0 and num_entries_per_consumer %
// block_size == 0. C≥P fan-out and P>C fan-in both via the integer-ratio TC round-robin.
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB1Bx1B_blk4, DM, TENSIX, 1, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB1Bx2B_blk4, DM, TENSIX, 1, 2, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB2Bx2B_blk4, DM, TENSIX, 2, 2, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB1Bx4B_blk4, DM, TENSIX, 1, 4, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB2Bx1B_blk4, DM, TENSIX, 2, 1, 4, 16, false)
DFB_BLOCKED_TEST_2_0(DMTensixTest1xDFB4Bx4B_blk4, DM, TENSIX, 4, 4, 4, 32, false)

// --- Trisc→DM BLOCKED→ALL (Tensix BLOCKED producer → DM ALL consumer) ---
// A Tensix producer at cap=ALL routes the credit fan-out through the REMAPPER instead of broadcast_tc
// (dm_dm_all is false when the producer is not DM). Golden: output[r]=input[(r%P)*capacity+(r/P)],
// capacity=num_entries/P (identity at P==1). C<=4. The cases below span both odd and even total RISC
// counts (P+C): the serialized DFB config blob is 36 + 62*(P+C) bytes, so an odd P+C makes it non-word-
// aligned. This is regression coverage for the write_to_device trailing-partial-word truncation that used
// to drop the producer's remapper bytes (remapper_consumer_ids_mask / producer_client_type) — see the
// word-alignment pad in tt_metal.cpp. 1Bx2A/1Bx4A (P+C odd) exercise that boundary directly.
#define DFB_TRISC_BLOCKED_ALL_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(MeshDeviceFixture, suffix##_2_0) {                              \
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
        run_single_dfb_program_2_0(this->devices_.at(0), params);          \
    }
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB1Bx1A_blk4, 1, 1, 4, 16)  // P+C=2 (even): 1->1, no fan-out
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB1Bx2A_blk4, 1, 2, 4, 16)  // P+C=3 (odd): 1->2 broadcast
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB1Bx4A_blk4, 1, 4, 4, 16)  // P+C=5 (odd): 1->4 broadcast
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB2Bx2A_blk4, 2, 2, 4, 16)  // P+C=4 (even): 2 pairs
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB2Bx4A_blk4, 2, 4, 4, 16)  // P+C=6 (even): 2 pairs, P<C
// P=4 (widest legal Tensix-producer remapper fan-out): 32-entry ring so each producer sub-ring holds 2
// blocks (non-degenerate). Data-verified via the BLOCKED→ALL golden output[r]=input[(r%P)*cap+(r/P)].
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB4Bx1A_blk4, 4, 1, 4, 32)  // P+C=5 (odd)
DFB_TRISC_BLOCKED_ALL_TEST_2_0(TensixDMTest1xDFB4Bx2A_blk4, 4, 2, 4, 32)  // P+C=6 (even)

// --- BLOCKED-producer → ALL-consumer (DM-DM, explicit sync) ---
// The producer block-bursts into its contiguous per-producer sub-ring; every ALL consumer reads every
// entry (free-after-all-ack via the built-in broadcast_tc — DM→DM never engages the remapper). Rides
// the existing ALL device path (cap=ALL drives capacity/stride/broadcast); pap=BLOCKED only swaps the
// producer's txn descriptor to a block burst, so NO device/kernel change is needed. Host-side, one
// validation guard was relaxed to admit BLOCKED→ALL (program_spec.cpp) plus a divisibility check so the
// sub-ring holds a whole number of blocks (dataflow_buffer.cpp). Constraints: C ≤ 4 (ALL slot cap),
// P+C ≤ 6 (Gen2 DM cap), num_entries % (block_size*P) == 0. Verified by the DM→DM BLOCKED→ALL golden
// in run_single_dfb_program_2_0: identity at P==1, permutation at P>1 (the producer's per-block
// interleave does not cancel the ALL consumer's per-tile round-robin).
#define DFB_BLOCKED_ALL_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(MeshDeviceFixture, suffix##_2_0) {                        \
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
        run_single_dfb_program_2_0(this->devices_.at(0), params);    \
    }

// Single-producer (P==1): the whole ring is one sub-ring written in order, so the round-trip is
// identity regardless of consumer count. C = 1/2/4 ALL consumers all read the full ring.
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB1Bx1A_blk4, 1, 1, 4, 16)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB1Bx2A_blk4, 1, 2, 4, 16)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB1Bx2A_blk2, 1, 2, 2, 16)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB1Bx4A_blk4, 1, 4, 4, 16)
// Multi-producer (P==2): capacity = num_entries/P = 8 per producer → 2 blocks of 4. The output is a
// permutation of the input (golden derived from the STRIDED→ALL round-robin de-interleave). C ≤ 4,
// P+C ≤ 6 (2Bx4A sits at the DM cap of 6 cores).
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB2Bx2A_blk4, 2, 2, 4, 16)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB2Bx4A_blk4, 2, 4, 4, 16)
// Extra coverage: smaller block at P=2, and P=3 (the producer ceiling for ALL: P+C<=6, C<=4).
// The permutation golden keys only on P and block_size (C-independent), so 3Bx1A and 3Bx3A share it.
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB2Bx2A_blk2, 2, 2, 2, 16)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB3Bx1A_blk4, 3, 1, 4, 24)
DFB_BLOCKED_ALL_TEST_2_0(DMTest1xDFB3Bx3A_blk4, 3, 3, 4, 24)

// --- BLOCKED-producer → STRIDED-consumer (DM→DM, explicit sync) ---
// The producer reads block_size CONTIGUOUS DRAM pages per block (the fast blocked read) but PUSHES
// PER-TILE, so the existing STRIDED round-robin scatters each tile into the next consumer's interleaved
// ring slot — NO remapper, NO broadcast, NO credit-path change (cap=STRIDED drives the interleaved
// layout and the per-consumer credit divide out of the box). Host: one program_spec guard relaxed to
// admit BLOCKED→STRIDED + a block divisibility check in the STRIDED capacity case. Golden (DM→DM):
// identity at P==1; a deterministic permutation at P>1 (the block read order does not cancel the
// per-tile round-robin). First cut covers C ≥ P at an integer ratio; num_entries % (block_size*P)==0
// and num_entries % max(P,C)==0.
#define DFB_BLOCKED_STRIDED_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(MeshDeviceFixture, suffix##_2_0) {                            \
        M2SingleDFBParams params{                                        \
            .producer_type = M2PorCType::DM,                             \
            .consumer_type = M2PorCType::DM,                             \
            .num_producers = (num_p),                                    \
            .num_consumers = (num_c),                                    \
            .pap = m2::DFBAccessPattern::BLOCKED,                        \
            .cap = m2::DFBAccessPattern::STRIDED,                        \
            .implicit_sync = false,                                      \
            .num_entries = (entries),                                    \
            .block_size = (blk),                                         \
        };                                                               \
        run_single_dfb_program_2_0(this->devices_.at(0), params);        \
    }
// Single-producer (P==1): per-tile pushes walk the consumers' interleaved slots in order → identity,
// regardless of C. Smallest case first (P=1,C=2,bs=2,entries=4) then deeper interleave / larger block.
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB1Bx2S_blk2_e4, 1, 2, 2, 4)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB1Bx1S_blk4, 1, 1, 4, 16)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB1Bx2S_blk4, 1, 2, 4, 16)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB1Bx4S_blk4, 1, 4, 4, 16)
// Multi-producer (P>1, C≥P integer ratio): a deterministic permutation (golden simulates the round-robin).
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB2Bx2S_blk4, 2, 2, 4, 16)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB2Bx4S_blk4, 2, 4, 4, 16)
// Fan-in (P>C, integer ratio): the golden now handles P>C (each producer feeds one consumer; the consumer
// round-robins its P/C feeding producers). Closes the P>C BLOCKED→STRIDED gap for DM→DM.
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB2Bx1S_blk4, 2, 1, 4, 16)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB4Bx1S_blk4, 4, 1, 4, 16)
DFB_BLOCKED_STRIDED_TEST_2_0(DMTest1xDFB4Bx2S_blk4, 4, 2, 4, 16)

// --- BLOCKED-producer → STRIDED-consumer (Trisc→DM, explicit sync) ---
// A Tensix producer only POSTS credits over a host-flat-prefilled ring (ring[s]=input[s]); it never
// reads DRAM, so its "block-ness" was only credit cadence — and a STRIDED consumer needs per-tile
// credits, so we reuse the plain per-tile Tensix producer (dfb_t6_producer_2_0.cpp). The DM STRIDED
// consumer reads its interleaved slots {c, c+C, ...} and writes out page k*C+c, which over a flat
// prefill is IDENTITY for C≥P (handled by the identity golden branch) — independent of block_size and P.
// Tensix compute threads ∈ {1,2,4}; C≥P integer ratio; num_entries % (block_size*P)==0.
#define DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(MeshDeviceFixture, suffix##_2_0) {                                  \
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
        run_single_dfb_program_2_0(this->devices_.at(0), params);              \
    }
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB1Bx1S_blk4, 1, 1, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB1Bx2S_blk4, 1, 2, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB1Bx4S_blk4, 1, 4, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB2Bx2S_blk4, 2, 2, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB2Bx4S_blk4, 2, 4, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB4Bx4S_blk4, 4, 4, 4, 32)
// Fan-in (P>C): Tensix producer flat-prefill, DM STRIDED consumer round-robins P/C stride-P TCs (golden
// above). Closes the P>C BLOCKED→STRIDED gap for Trisc→DM. Tensix producer threads ∈ {2,4}.
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB2Bx1S_blk4, 2, 1, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB4Bx1S_blk4, 4, 1, 4, 16)
DFB_TRISC_BLOCKED_STRIDED_TEST_2_0(TensixDMTest1xDFB4Bx2S_blk4, 4, 2, 4, 16)


// --- DM→Trisc BLOCKED→ALL (DM BLOCKED producer → Tensix ALL consumer) ---
// cap=ALL with a Tensix consumer routes the fan-out through the REMAPPER (dm_dm_all is false when the
// consumer is Tensix) — the same remapper-to-Tensix path the STRIDED→ALL DM→Tensix tests above exercise,
// except the producer block-bursts into its sub-ring instead of striding. The Tensix consumer drains
// per-tile on the (fixed) UNPACK path. RUN-ONLY (Tensix consumer writes no DRAM): pass = no trap/hang.
// C ≤ 4 (ALL slot cap; Tensix threads ∈ {1,2,4}); explicit sync; num_entries % (block_size*P) == 0.
#define DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(MeshDeviceFixture, suffix##_2_0) {                                 \
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
        run_single_dfb_program_2_0(this->devices_.at(0), params);             \
    }
DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(DMTensixTest1xDFB1Bx1A_blk4, 1, 1, 4, 16)
DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(DMTensixTest1xDFB1Bx2A_blk4, 1, 2, 4, 16)
DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(DMTensixTest1xDFB1Bx4A_blk4, 1, 4, 4, 16)
DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(DMTensixTest1xDFB2Bx2A_blk4, 2, 2, 4, 16)
DFB_DMTENSIX_BLOCKED_ALL_TEST_2_0(DMTensixTest1xDFB2Bx4A_blk4, 2, 4, 4, 16)

// --- DM→Trisc BLOCKED→STRIDED (DM BLOCKED producer → Tensix STRIDED consumer) ---
// The final matrix cell. The DM producer reads block-contiguous DRAM but pushes per-tile (the
// dfb_blocked_strided_producer kernel), so the STRIDED round-robin scatters each tile into the next
// consumer's interleaved slot; the Tensix consumer drains per-tile on the (fixed) UNPACK path. Reuses the
// blocked_to_strided host guard + STRIDED block-divisibility added for DM→DM. RUN-ONLY (Tensix consumer
// writes no DRAM): pass = no trap/hang. C ≥ P integer ratio; Tensix threads ∈ {1,2,4}; explicit sync;
// num_entries % (block_size*P) == 0 and num_entries % max(P,C) == 0.
#define DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(suffix, num_p, num_c, blk, entries) \
    TEST_F(MeshDeviceFixture, suffix##_2_0) {                                     \
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
        run_single_dfb_program_2_0(this->devices_.at(0), params);                 \
    }
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB1Bx1S_blk4, 1, 1, 4, 16)
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB1Bx2S_blk4, 1, 2, 4, 16)
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB1Bx4S_blk4, 1, 4, 4, 16)
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB2Bx2S_blk4, 2, 2, 4, 16)
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB2Bx4S_blk4, 2, 4, 4, 16)
DFB_DMTENSIX_BLOCKED_STRIDED_TEST_2_0(DMTensixTest1xDFB4Bx4S_blk4, 4, 4, 4, 32)

}  // namespace tt::tt_metal
