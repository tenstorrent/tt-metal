// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of ScratchpadSpec (scratchpad_spec.hpp).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_ValidScratchpadSucceeds) {
    // Positive baseline: one ScratchpadSpec bound by exactly one kernel.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024}};
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ZeroSizeScratchpadFails) {
    // A ScratchpadSpec with size_per_node == 0 (the default) reserves no L1, so the device-side
    // accessor's operator[] would be out of bounds on first use. Bound to a kernel here so the
    // size check — not the unbound check — is what fires.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}}};  // size_per_node defaults to 0
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("size_per_node == 0")));
}

TEST_F(ProgramSpecTestQuasar, ScratchpadFormatUnsupportedOnArchFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"scratch_0"},
        .size_per_node = 1024,
        .data_format_metadata = tt::DataFormat::Bfp8,
    }};
    spec.kernels[1].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "pad"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'scratch_0' has data format")));
}

TEST_F(ProgramSpecTestQuasar, ScratchpadTileWithoutFormatFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"scratch_0"},
        .size_per_node = 1024,
        .tile_format_metadata = Tile{{32, 32}},
    }};
    spec.kernels[1].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "pad"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("no data_format_metadata")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
