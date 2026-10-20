// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/gtest_dispatch.hpp -- glue macros between a suite descriptor and
// gtest so a case's TEST_P body is a single call.
//
// A characterization case uses this file like:
//
//   using MyRunner = x280::harness::Runner<MySuite::Axis>;
//   TEST_P(MyRunner, ...) { X280_HARNESS_DISPATCH(MySuite); }
//
// The macro references the base class's fixture() and spec() members, so
// the body is exactly one call.

#ifndef X280_HARNESS_GTEST_DISPATCH_HPP
#define X280_HARNESS_GTEST_DISPATCH_HPP

#include "harness/runner.hpp"

// A suite descriptor `Descriptor` provides a static function
// `run_axis(fixture, spec, axis)` that carries out the whole test.
#define X280_HARNESS_DISPATCH(Descriptor)                                       \
    do {                                                                        \
        if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) \
            return;                                                             \
        Descriptor::run_axis(fixture(), spec(), GetParam());                    \
    } while (0)

#endif  // X280_HARNESS_GTEST_DISPATCH_HPP
