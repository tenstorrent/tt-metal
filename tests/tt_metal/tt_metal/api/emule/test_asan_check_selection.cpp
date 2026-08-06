// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="*Asan_Checks_*"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"
#include "impl/emulation/host_sanitizers.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

// TT_METAL_EMULE_ASAN_CHECKS narrows which sanitizers a run enables, so an
// investigation can isolate one check instead of taking whichever violation the
// whole suite happens to hit first. The master switch still gates everything; an
// absent/empty/"all" list means every check, so existing runs are unchanged.
//
// Two properties are worth guarding, and they fail in opposite directions:
//   * a named check must still fire  — otherwise selection silently disables the
//     sweep and a clean report means nothing;
//   * an unnamed check must NOT fire — otherwise selection does nothing and the
//     isolation it promises is a lie.
//
// Everything below goes through the by-name query rather than the bit values: it
// is the contract a caller actually uses, and it keeps this test free of a
// tt-emule include (the test target has no jit_hw include path).
//
// Death tests are deliberately not used. death_test_env.cpp forces "threadsafe"
// style, which re-exec's the binary, and its global SetUp re-forces CHECKS="all"
// in the child — so a narrowed list set in a test body would not survive into a
// death-test child. In-process assertions do.

namespace {
// Restores whatever the suite-wide default is (see death_test_env.cpp) so one
// test's narrowed list cannot leak into the next test in this process.
class CheckListGuard {
public:
    ~CheckListGuard() { ::setenv("TT_METAL_EMULE_ASAN_CHECKS", "all", 1); }
};
}  // namespace

TEST(Asan_Checks_Env, AbsentOrEmptyOrAllMeansEveryCheck) {
    CheckListGuard guard;
    for (const char* v : {static_cast<const char*>(nullptr), "", "all"}) {
        if (v == nullptr) {
            ::unsetenv("TT_METAL_EMULE_ASAN_CHECKS");
        } else {
            ::setenv("TT_METAL_EMULE_ASAN_CHECKS", v, 1);
        }
        EXPECT_TRUE(emule::asan_check_enabled_by_name("oob")) << "value: " << (v ? v : "<unset>");
        EXPECT_TRUE(emule::asan_check_enabled_by_name("dirty_cb")) << "value: " << (v ? v : "<unset>");
        EXPECT_TRUE(emule::asan_check_enabled_by_name("object_intent")) << "value: " << (v ? v : "<unset>");
    }
}

TEST(Asan_Checks_Env, NamedSubsetSelectsExactlyThose) {
    CheckListGuard guard;
    ::setenv("TT_METAL_EMULE_ASAN_CHECKS", "oob,dirty_cb", 1);
    EXPECT_TRUE(emule::asan_check_enabled_by_name("oob"));
    EXPECT_TRUE(emule::asan_check_enabled_by_name("dirty_cb"));
    EXPECT_FALSE(emule::asan_check_enabled_by_name("padding"));
    EXPECT_FALSE(emule::asan_check_enabled_by_name("object_intent"));
    EXPECT_FALSE(emule::asan_check_enabled_by_name("noc_align"));
}

TEST(Asan_Checks_Env, SeparatorsAreInterchangeable) {
    CheckListGuard guard;
    for (const char* v : {"oob, dirty_cb", ",,oob,,dirty_cb,", "oob dirty_cb"}) {
        ::setenv("TT_METAL_EMULE_ASAN_CHECKS", v, 1);
        EXPECT_TRUE(emule::asan_check_enabled_by_name("oob")) << "value: " << v;
        EXPECT_TRUE(emule::asan_check_enabled_by_name("dirty_cb")) << "value: " << v;
        EXPECT_FALSE(emule::asan_check_enabled_by_name("padding")) << "value: " << v;
    }
}

// A list that names nothing recognizable falls back to every check rather than to
// none: a typo must never be able to silence the sanitizers and report a clean run.
TEST(Asan_Checks_Env, UnrecognizedListFallsBackToEveryCheck) {
    CheckListGuard guard;
    // Whole-word matching: a prefix or an over-long name is a typo, not a selection.
    for (const char* v : {"bogus", "oo", "oobx"}) {
        ::setenv("TT_METAL_EMULE_ASAN_CHECKS", v, 1);
        EXPECT_TRUE(emule::asan_check_enabled_by_name("oob")) << "value: " << v;
        EXPECT_TRUE(emule::asan_check_enabled_by_name("dirty_cb")) << "value: " << v;
    }
    // A typo alongside a valid name keeps the valid selection rather than widening.
    ::setenv("TT_METAL_EMULE_ASAN_CHECKS", "oob,bogus", 1);
    EXPECT_TRUE(emule::asan_check_enabled_by_name("oob"));
    EXPECT_FALSE(emule::asan_check_enabled_by_name("dirty_cb"));
}

// CB Reservation Overflow is load-bearing: gating it would turn an over-reserve
// into a silent deadlock on the space wait instead of a clear report, so it stays
// enabled even when the list names something else. Documented in docs/ASAN.md.
TEST(Asan_Checks_Env, ReservationOverflowStaysOnWhenDeselected) {
    CheckListGuard guard;
    ::setenv("TT_METAL_EMULE_ASAN_CHECKS", "padding", 1);
    EXPECT_TRUE(emule::asan_check_enabled_by_name("cb_reservation"));
    EXPECT_FALSE(emule::asan_check_enabled_by_name("dirty_cb"));
}

TEST(Asan_Checks_Env, UnknownNameQueryIsFalse) {
    CheckListGuard guard;
    ::setenv("TT_METAL_EMULE_ASAN_CHECKS", "all", 1);
    EXPECT_FALSE(emule::asan_check_enabled_by_name("not_a_check"));
    EXPECT_FALSE(emule::asan_check_enabled_by_name(nullptr));
}

// The reader must track the environment per call (no static cache) — the same
// contract emule_asan_enabled() keeps, so a combined gtest run can toggle it.
TEST(Asan_Checks_Env, ReaderTracksEnvironmentPerCall) {
    CheckListGuard guard;
    ::setenv("TT_METAL_EMULE_ASAN_CHECKS", "oob", 1);
    EXPECT_TRUE(emule::asan_check_enabled_by_name("oob"));
    EXPECT_FALSE(emule::asan_check_enabled_by_name("dirty_cb"));
    ::setenv("TT_METAL_EMULE_ASAN_CHECKS", "dirty_cb", 1);
    EXPECT_FALSE(emule::asan_check_enabled_by_name("oob"));
    EXPECT_TRUE(emule::asan_check_enabled_by_name("dirty_cb"));
}

// ---- Behavioural: a deselected check must not fire -------------------------

// The Dirty_CB_ReserveWithoutPush program, run with a list that does NOT name
// dirty_cb. It must complete: if the mask never reached the check, the process
// would abort here and take the binary down — which is how the other NoViolation
// controls in this directory report a wrongly-firing check too.
TEST_F(MeshDeviceFixture, Asan_Checks_DeselectedCheckDoesNotFire) {
    CheckListGuard guard;
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);
    ::unsetenv("TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB");
    // Name a different check, so Dirty CB is off for this launch.
    ::setenv("TT_METAL_EMULE_ASAN_CHECKS", "oob", 1);

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    uint32_t cb_id = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(2 * 1024, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, 1024);
    CreateCircularBuffer(program, logical_core, cb_config);

    // Reserve without the matching push — a genuine Dirty CB violation.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            cb_reserve_back(0, 1);
            // MISSING: cb_push_back(0, 1);
        }
    )";
    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    detail::LaunchProgram(device, program);
}

}  // namespace tt::tt_metal
