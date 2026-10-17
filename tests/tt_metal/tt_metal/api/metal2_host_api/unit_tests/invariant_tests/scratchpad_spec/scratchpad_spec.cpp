// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ScratchpadSpec invariants and shared allocation behavior (scratchpad_spec.hpp).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestBlackhole;
using test_helpers::ProgramSpecTestGen1;
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

void CheckSharedScratchpad(distributed::MeshDevice& device, ProgramSpec spec) {
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"shared"},
        .size_per_node = 1024,
        .advanced_options = {.allow_multi_kernel_binding = true},
    }};
    spec.kernels[0].scratchpad_bindings = {
        {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "a"}};
    spec.kernels[1].scratchpad_bindings = {
        {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "b"}};

    Program program = MakeProgramFromSpec(device, spec);
    auto dm = program.impl().get_kernel_by_spec_name("dm_kernel");
    auto compute = program.impl().get_kernel_by_spec_name("compute_kernel");
    const auto& a = dm->scratchpad_binding_handles().at(0);
    const auto& b = compute->scratchpad_binding_handles().at(0);
    auto check_addresses = [&] {
        ASSERT_NE(a.allocated_address, 0u);
        EXPECT_EQ(a.allocated_address, b.allocated_address);
        EXPECT_EQ(dm->common_runtime_args_data()[a.addr_crta_word], a.allocated_address);
        EXPECT_EQ(compute->common_runtime_args_data()[b.addr_crta_word], a.allocated_address);
    };
    check_addresses();
    const uint32_t original_address = a.allocated_address;

    // Rebuilding CRTAs for binding-only kernels must preserve both shared addresses.
    SetProgramRunArgs(program, ProgramRunArgs{});
    check_addresses();

    // Growing the DFB moves the shared allocation and must refresh every binder's CRTA.
    ProgramRunArgs args;
    args.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .num_entries = 8});
    SetProgramRunArgs(program, args);
    program.impl().allocate_dataflow_buffers(&device);
    program.impl().allocate_scratchpads(&device);
    EXPECT_GT(a.allocated_address, original_address);
    check_addresses();
    SetProgramRunArgs(program, ProgramRunArgs{});
    check_addresses();
}

TEST_F(ProgramSpecTestQuasar, CPU_SharedScratchpadAllocationAndReallocation) {
    CheckSharedScratchpad(*mesh_device_, MakeMinimalValidProgramSpec());
}

TEST_F(ProgramSpecTestGen1, CPU_SharedScratchpadAllocationAndReallocation) {
    CheckSharedScratchpad(*mesh_device_, MakeMinimalGen1ValidProgramSpec());
}

TEST_F(ProgramSpecTestBlackhole, CPU_SharedScratchpadAllocationAndReallocation) {
    CheckSharedScratchpad(*mesh_device_, MakeMinimalGen1ValidProgramSpec());
}

TEST_F(ProgramSpecTestQuasar, CPU_SharedScratchpadUsesEntireBinderNodeUnion) {
    // The DFB occupies only nodes 1 and 2. The shared scratchpad must clear it even
    // if the first binding kernel's node set does not include node 2.
    for (bool reverse_order : {false, true}) {
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        auto extra = MakeMinimalGen2DMKernel("extra");
        spec.kernels.push_back(extra);
        spec.work_units = {
            MakeMinimalWorkUnit("left", NodeCoord{0, 0}, {"extra"}),
            MakeMinimalWorkUnit("overlap", NodeCoord{1, 0}, {"extra", "dm_kernel", "compute_kernel"}),
            MakeMinimalWorkUnit("right", NodeCoord{2, 0}, {"dm_kernel", "compute_kernel"}),
        };
        spec.scratchpads = {
            ScratchpadSpec{
                .unique_id = ScratchpadSpecName{"shared"},
                .size_per_node = 1024,
                .advanced_options = {.allow_multi_kernel_binding = true}},
            ScratchpadSpec{.unique_id = ScratchpadSpecName{"private"}, .size_per_node = 512},
            ScratchpadSpec{
                .unique_id = ScratchpadSpecName{"other_shared"},
                .size_per_node = 256,
                .advanced_options = {.allow_multi_kernel_binding = true}},
        };
        for (auto& kernel : spec.kernels) {
            kernel.scratchpad_bindings = {
                {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "pad"},
                {.scratchpad_spec_name = ScratchpadSpecName{"other_shared"}, .accessor_name = "other"}};
        }
        spec.kernels[2].scratchpad_bindings.push_back(
            {.scratchpad_spec_name = ScratchpadSpecName{"private"}, .accessor_name = "local"});
        if (reverse_order) {
            std::reverse(spec.kernels.begin(), spec.kernels.end());
        }
        Program program = MakeProgramFromSpec(*mesh_device_, spec);
        auto dm = program.impl().get_kernel_by_spec_name("dm_kernel");
        auto compute = program.impl().get_kernel_by_spec_name("compute_kernel");
        auto left = program.impl().get_kernel_by_spec_name("extra");
        auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
        for (size_t i = 0; i < 2; ++i) {
            const auto& shared = dm->scratchpad_binding_handles().at(i);
            EXPECT_NE(shared.allocated_address, 0u);
            EXPECT_EQ(shared.allocated_address, left->scratchpad_binding_handles().at(i).allocated_address);
            EXPECT_EQ(shared.allocated_address, compute->scratchpad_binding_handles().at(i).allocated_address);
            EXPECT_GE(shared.allocated_address, dfb->core_lookup_.at(CoreCoord{2, 0}).second + dfb->total_size());
        }
        const auto& handles = left->scratchpad_binding_handles();
        for (size_t i = 0; i < handles.size(); ++i) {
            for (size_t j = i + 1; j < handles.size(); ++j) {
                EXPECT_TRUE(
                    handles[i].allocated_address + handles[i].size_bytes <= handles[j].allocated_address ||
                    handles[j].allocated_address + handles[j].size_bytes <= handles[i].allocated_address);
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental
