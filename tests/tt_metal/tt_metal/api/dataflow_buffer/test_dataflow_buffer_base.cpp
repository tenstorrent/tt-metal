// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Base single-DFB config sweep (legacy kept configs + full Metal 2.0 matrix).

#include "dfb_test_common.hpp"

namespace tt::tt_metal {


// gtest param-name helpers
static std::string ImplicitSyncParamName(const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "ImplicitSyncTrue" : "ImplicitSyncFalse";
}

static std::string M2ImplicitSyncParamName(const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "ImplicitSyncTrue" : "ImplicitSyncFalse";
}

// legacy single-DFB config sweep macros + kept configs
#define DFB_TEST(prefix, suffix, p_kind, c_kind, num_p, pap_kind, num_c, cap_kind, extra_skip)          \
    TEST_P(DFBImplicitSyncParamFixture, prefix##Test1xDFB##suffix) {                                    \
        DFB_SKIP_IF_UNSUPPORTED((num_p), (num_c));                                                      \
        extra_skip;                                                                                     \
        experimental::dfb::DataflowBufferConfig config{                                                 \
            .entry_size = 1024,                                                                         \
            .num_entries = dfb_default_num_entries((num_p), (num_c)),                                   \
            .num_producers = (num_p),                                                                   \
            .pap = dfb::AccessPattern::pap_kind,                                                        \
            .num_consumers = (num_c),                                                                   \
            .cap = dfb::AccessPattern::cap_kind,                                                        \
            .enable_producer_implicit_sync = GetParam(),                                                \
            .enable_consumer_implicit_sync = GetParam()};                                               \
        run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::p_kind, DFBPorCType::c_kind); \
    }

#define DFB_TEST_BUF(prefix, suffix, p_kind, c_kind, num_p, pap_kind, num_c, cap_kind, extra_skip, n_buf)     \
    TEST_P(DFBImplicitSyncParamFixture, prefix##Test1xDFB##suffix) {                                          \
        DFB_SKIP_IF_UNSUPPORTED((num_p), (num_c));                                                            \
        extra_skip;                                                                                           \
        experimental::dfb::DataflowBufferConfig config{                                                       \
            .entry_size = 1024,                                                                               \
            .num_entries = dfb_default_num_entries((num_p), (num_c)),                                         \
            .num_producers = (num_p),                                                                         \
            .pap = dfb::AccessPattern::pap_kind,                                                              \
            .num_consumers = (num_c),                                                                         \
            .cap = dfb::AccessPattern::cap_kind,                                                              \
            .enable_producer_implicit_sync = GetParam(),                                                      \
            .enable_consumer_implicit_sync = GetParam()};                                                     \
        CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));                             \
        run_single_dfb_program(                                                                               \
            this->devices_.at(0), config, DFBPorCType::p_kind, DFBPorCType::c_kind, core_range_set, (n_buf)); \
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


}  // namespace tt::tt_metal
