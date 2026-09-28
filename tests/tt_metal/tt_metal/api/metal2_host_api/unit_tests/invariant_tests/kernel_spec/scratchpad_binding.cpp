// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelSpec::ScratchpadBinding and KernelSpec::scratchpad_bindings (kernel_spec.hpp):
// accessor names, and each scratchpad bound at most once per kernel.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadBoundTwiceInOneKernelFails) {
    // One kernel binds the SAME scratchpad twice under two different accessor_names. Illegal: a kernel
    // may bind a given scratchpad at most once (two bindings would request two separate per-node
    // allocations under one name). This is a structural input error with no node-set dependency, so
    // it is caught up front during collection rather than by the placement census.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024}};
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s_a"},
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s_b"},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("binds scratchpad 'scratch_0' more than once")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DuplicateScratchpadAccessorNameFails) {
    // One kernel with two scratchpad_bindings to two DIFFERENT scratchpads but sharing the same
    // accessor_name. The accessor_name is the kernel-local C++ symbol, so it must be unique per
    // kernel (the per-kernel duplicate check fires before the bound-more-than-once check, since
    // the two scratchpads are distinct).
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024},
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_1"}, .size_per_node = 1024},
    };
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "dup"},
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_1"}, .accessor_name = "dup"},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("duplicate scratchpad accessor_name")));
}

TEST_F(ProgramSpecTestQuasar, CPU_InvalidScratchpadAccessorNameFails) {
    // The accessor_name becomes a C++ identifier in the generated kernel_bindings header, so it must
    // be a valid C++ identifier and fit in MAX_ACCESSOR_NAME_LENGTH. (Mirrors
    // InvalidLocalAccessorNameFails / the semaphore-accessor equivalent; here we just spot-check a
    // couple of clearly-invalid names plus the too-long case.)
    const std::vector<std::string> invalid_names = {
        "1bad",       // leading digit
        "has space",  // whitespace
    };

    for (const auto& bad_name : invalid_names) {
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024}};
        spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = bad_name}};

        EXPECT_THAT(
            [&] { MakeProgramFromSpec(*mesh_device_, spec); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be a valid C++ identifier")))
            << "Expected rejection for scratchpad accessor_name: '" << bad_name << "'";
    }

    const std::string too_long(MAX_ACCESSOR_NAME_LENGTH + 1, 'a');
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024}};
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = too_long}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be at most")));
}

TEST_F(ProgramSpecTestQuasar, CPU_MultipleScratchpadsEachBoundToOwnKernelSucceeds) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "scratchpad_multi";

    // Two independent scratchpads, each bound by its own kernel on its own node — the simplest
    // multi-scratchpad case (distinct from binding one shared scratchpad across disjoint nodes).
    auto kernel_a = MakeMinimalGen2DMKernel("kernel_a");
    kernel_a.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_a"}, .accessor_name = "scratch"});
    auto kernel_b = MakeMinimalGen2DMKernel("kernel_b");
    kernel_b.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_b"}, .accessor_name = "scratch"});

    spec.kernels = {kernel_a, kernel_b};
    spec.scratchpads = {
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_a"}, .size_per_node = 1024},
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_b"}, .size_per_node = 2048},
    };
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_a", node0, {"kernel_a"}),
        MakeMinimalWorkUnit("wu_b", node1, {"kernel_b"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

}  // namespace
}  // namespace tt::tt_metal::experimental
