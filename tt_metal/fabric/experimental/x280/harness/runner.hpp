// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/runner.hpp -- the RunSpec each characterization case fills in and
// a gtest fixture template that consumes it.
//
// A characterization case's host side is:
//
//   class MyRunner : public x280::harness::Runner<MySuite::Axis> {
//   protected:
//     x280::harness::RunSpec make_spec() override { return {...}; }
//   };
//   TEST_P(MyRunner, ...) { X280_HARNESS_DISPATCH(MySuite); }
//
// This header carries only the shared plumbing. Suite-specific work
// (layout, args block, harvest, report, JSONL) lives in the suite
// descriptor under `tests/device/suites/<suite>/*.hpp`.

#ifndef X280_HARNESS_RUNNER_HPP
#define X280_HARNESS_RUNNER_HPP

#include "harness/session/session.hpp"

#include <gtest/gtest.h>

#include <string>

namespace x280::harness {

// The settings a case wants at construction time, populated by the derived
// Runner's make_spec() override: how the session opens, and the label the
// JSONL writer keys on -- typically the suite name ("p2p_latency",
// "transport_bandwidth") plus the row/direction. Suite descriptors also read
// `open.kernel_dir` to resolve their suite-owned kernel paths and JIT
// include paths.
struct RunSpec {
    session::OpenConfig open;
    std::string suite_label;
};

// Base test fixture for a characterization case. Owns the Fixture and
// takes care of setup/teardown; the derived class overrides `make_spec()`
// to provide the per-case configuration and exposes the axis type via
// GetParam().
template <typename Axis>
class Runner : public ::testing::TestWithParam<Axis> {
public:
    session::Fixture& fixture() { return fixture_; }
    const session::Fixture& fixture() const { return fixture_; }
    const RunSpec& spec() const { return spec_; }

protected:
    // Derived classes provide the per-case configuration.
    virtual RunSpec make_spec() = 0;

    void SetUp() override {
        spec_ = make_spec();
        session::open_or_skip(fixture_, spec_.open);
    }

    void TearDown() override { fixture_.close(); }

private:
    session::Fixture fixture_;
    RunSpec spec_;
};

}  // namespace x280::harness

#endif  // X280_HARNESS_RUNNER_HPP
