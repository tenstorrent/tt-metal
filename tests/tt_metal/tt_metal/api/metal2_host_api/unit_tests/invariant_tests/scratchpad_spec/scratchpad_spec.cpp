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

#include "impl/allocator/allocator.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalGen2ComputeKernel;
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

TEST_F(ProgramSpecTestQuasar, CPU_SharedScratchpadDMAndComputeFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"shared"},
        .size_per_node = 1024,
        .advanced_options = {.allow_instance_multi_binding = true},
    }};
    spec.kernels[0].scratchpad_bindings = {
        {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "a"}};
    spec.kernels[1].scratchpad_bindings = {
        {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "b"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("only between DM kernels on Gen2")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadMultiBindingFlagWithSingleBinderSucceeds) {
    for (size_t binder_index : {0u, 1u}) {
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        spec.scratchpads = {ScratchpadSpec{
            .unique_id = ScratchpadSpecName{"pad"},
            .size_per_node = 1024,
            .advanced_options = {.allow_instance_multi_binding = true},
        }};
        // A single DM or compute binder does not share memory with another kernel.
        spec.kernels[binder_index].scratchpad_bindings = {
            {.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
        EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
    }
}

TEST_F(ProgramSpecTestQuasar, CPU_SharedScratchpadBetweenDMKernelsSucceeds) {
    ProgramSpec spec;
    spec.name = "shared_scratchpad_dm_only";
    spec.kernels = {MakeMinimalGen2DMKernel("dm_a"), MakeMinimalGen2DMKernel("dm_b")};
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
        .advanced_options = {.allow_instance_multi_binding = true},
    }};
    for (auto& kernel : spec.kernels) {
        kernel.scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    }
    spec.work_units = {MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_a", "dm_b"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    auto dm_a = program.impl().get_kernel_by_spec_name("dm_a");
    auto dm_b = program.impl().get_kernel_by_spec_name("dm_b");
    const auto& a = dm_a->scratchpad_binding_handles().at(0);
    const auto& b = dm_b->scratchpad_binding_handles().at(0);
    EXPECT_NE(a.allocated_address, 0u);
    EXPECT_EQ(a.allocated_address, b.allocated_address);
    EXPECT_EQ(dm_a->common_runtime_args_data()[a.addr_crta_word], a.allocated_address);
    EXPECT_EQ(dm_b->common_runtime_args_data()[b.addr_crta_word], a.allocated_address);
}

TEST_F(ProgramSpecTestQuasar, CPU_SharedScratchpadGen2RestrictionIsPerNode) {
    ProgramSpec spec;
    spec.name = "shared_scratchpad_per_node";
    spec.kernels = {
        MakeMinimalGen2DMKernel("dm_a"),
        MakeMinimalGen2DMKernel("dm_b"),
        MakeMinimalGen2ComputeKernel("compute"),
    };
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
        .advanced_options = {.allow_instance_multi_binding = true},
    }};
    for (auto& kernel : spec.kernels) {
        kernel.scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    }
    spec.work_units = {
        MakeMinimalWorkUnit("dm_work", NodeCoord{0, 0}, {"dm_a", "dm_b"}),
        MakeMinimalWorkUnit("compute_work", NodeCoord{1, 0}, {"compute"}),
    };
    // Sharing on node 0 is DM-only; the compute kernel has a private instance on node 1.
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));

    // Extending one DM kernel onto node 1 introduces unsafe DM/compute sharing there.
    spec.work_units[1].kernels.push_back(KernelSpecName{"dm_a"});
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("only between DM kernels on Gen2")));
}

TEST_F(ProgramSpecTestGen1, CPU_SharedScratchpadAllocationAndReallocation) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"shared"},
        .size_per_node = 1024,
        .advanced_options = {.allow_instance_multi_binding = true},
    }};
    spec.kernels[0].scratchpad_bindings = {
        {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "a"}};
    spec.kernels[1].scratchpad_bindings = {
        {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "b"}};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
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
    program.impl().prepare_ephemeral_allocation(mesh_device_.get());
    program.impl().allocate_dataflow_buffers(mesh_device_.get());
    program.impl().allocate_scratchpads(mesh_device_.get());
    EXPECT_GT(a.allocated_address, original_address);
    check_addresses();
    SetProgramRunArgs(program, ProgramRunArgs{});
    check_addresses();
}

TEST_F(ProgramSpecTestBlackhole, CPU_SharedScratchpadAllocationAndReallocation) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"shared"},
        .size_per_node = 1024,
        .advanced_options = {.allow_instance_multi_binding = true},
    }};
    spec.kernels[0].scratchpad_bindings = {
        {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "a"}};
    spec.kernels[1].scratchpad_bindings = {
        {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "b"}};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
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
    program.impl().prepare_ephemeral_allocation(mesh_device_.get());
    program.impl().allocate_dataflow_buffers(mesh_device_.get());
    program.impl().allocate_scratchpads(mesh_device_.get());
    EXPECT_GT(a.allocated_address, original_address);
    check_addresses();
    SetProgramRunArgs(program, ProgramRunArgs{});
    check_addresses();
}

TEST_F(ProgramSpecTestGen1, CPU_SharedScratchpadReallocationWithUnequalPersistentBases) {
    auto& arena = mesh_device_->allocator_impl()->persistent_l1();
    const auto alignment = mesh_device_->allocator()->get_alignment(BufferType::DRAM);
    const auto persistent = arena.allocate(CoreRangeSet(CoreRange({1, 0})), 4096, alignment);
    const auto lower_base = arena.high_water_mark(CoreRangeSet(CoreRange({0, 0})));
    const auto higher_base = arena.high_water_mark(CoreRangeSet(CoreRange({1, 0})));
    ASSERT_GT(higher_base, lower_base);

    {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        auto left = MakeMinimalGen1DMKernel("left");
        auto right = MakeMinimalGen1DMKernel("right");
        left.scratchpad_bindings = {
            {.scratchpad_spec_name = ScratchpadSpecName{"private"}, .accessor_name = "local"},
            {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "pad"},
        };
        right.scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "pad"}};
        spec.kernels.push_back(left);
        spec.kernels.push_back(right);
        spec.scratchpads = {
            ScratchpadSpec{.unique_id = ScratchpadSpecName{"private"}, .size_per_node = 512},
            ScratchpadSpec{
                .unique_id = ScratchpadSpecName{"shared"},
                .size_per_node = 1024,
                .advanced_options = {.allow_instance_multi_binding = true}},
        };
        spec.work_units = {
            MakeMinimalWorkUnit("left_work", NodeCoord{0, 0}, {"left"}),
            MakeMinimalWorkUnit("right_work", NodeCoord{1, 0}, {"right"}),
            // Keep DFB allocation from populating the scratchpad allocators after invalidation.
            MakeMinimalWorkUnit("dfb_work", NodeCoord{2, 0}, {"dm_kernel", "compute_kernel"}),
        };

        Program program = MakeProgramFromSpec(*mesh_device_, spec);
        auto left_kernel = program.impl().get_kernel_by_spec_name("left");
        auto right_kernel = program.impl().get_kernel_by_spec_name("right");
        const auto& private_pad = left_kernel->scratchpad_binding_handles().at(0);
        const auto& shared_left = left_kernel->scratchpad_binding_handles().at(1);
        const auto& shared_right = right_kernel->scratchpad_binding_handles().at(0);
        auto check_addresses = [&] {
            EXPECT_NE(private_pad.allocated_address, 0u);
            EXPECT_GE(private_pad.allocated_address, lower_base);
            EXPECT_GE(shared_left.allocated_address, higher_base);
            EXPECT_EQ(shared_left.allocated_address, shared_right.allocated_address);
            EXPECT_TRUE(
                private_pad.allocated_address + private_pad.size_bytes <= shared_left.allocated_address ||
                shared_left.allocated_address + shared_left.size_bytes <= private_pad.allocated_address);
            EXPECT_EQ(
                left_kernel->common_runtime_args_data()[private_pad.addr_crta_word], private_pad.allocated_address);
            EXPECT_EQ(
                left_kernel->common_runtime_args_data()[shared_left.addr_crta_word], shared_left.allocated_address);
            EXPECT_EQ(
                right_kernel->common_runtime_args_data()[shared_right.addr_crta_word], shared_right.allocated_address);
        };
        check_addresses();

        // The first shared allocation creates an allocator spanning adjacent nodes 0–1.
        // Resize clears its regions but retains that range. Reallocating the private pad
        // on node 0 must account for node 1's higher base when marking this empty allocator.
        ProgramRunArgs args;
        args.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .num_entries = 8});
        SetProgramRunArgs(program, args);
        ASSERT_NO_THROW(program.impl().prepare_ephemeral_allocation(mesh_device_.get()));
        ASSERT_NO_THROW(program.impl().allocate_dataflow_buffers(mesh_device_.get()));
        ASSERT_NO_THROW(program.impl().allocate_scratchpads(mesh_device_.get()));
        check_addresses();
    }
    arena.deallocate(persistent.id);
}

TEST_F(ProgramSpecTestGen1, CPU_SharedScratchpadWithoutDfbsOnDisjointNodes) {
    ProgramSpec spec;
    spec.name = "shared_scratchpad_without_dfbs";
    spec.kernels = {MakeMinimalGen1DMKernel("left"), MakeMinimalGen1DMKernel("right")};
    spec.work_units = {
        MakeMinimalWorkUnit("left_work", NodeCoord{0, 0}, {"left"}),
        MakeMinimalWorkUnit("right_work", NodeCoord{2, 0}, {"right"}),
    };
    spec.scratchpads = {
        ScratchpadSpec{
            .unique_id = ScratchpadSpecName{"shared"},
            .size_per_node = 1024,
            .advanced_options = {.allow_instance_multi_binding = true}},
        ScratchpadSpec{
            .unique_id = ScratchpadSpecName{"other_shared"},
            .size_per_node = 256,
            .advanced_options = {.allow_instance_multi_binding = true}},
    };
    for (auto& kernel : spec.kernels) {
        kernel.scratchpad_bindings = {
            {.scratchpad_spec_name = ScratchpadSpecName{"shared"}, .accessor_name = "pad"},
            {.scratchpad_spec_name = ScratchpadSpecName{"other_shared"}, .accessor_name = "other"},
        };
    }
    // Different binding orders prevent independent per-kernel allocations from
    // accidentally producing matching addresses on otherwise empty nodes.
    std::reverse(spec.kernels[1].scratchpad_bindings.begin(), spec.kernels[1].scratchpad_bindings.end());

    // No DFBs or private scratchpads seed the allocators: shared allocation must
    // create them on both disjoint nodes and reserve space for both scratchpads.
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    ASSERT_TRUE(program.impl().dataflow_buffers().empty());
    auto left = program.impl().get_kernel_by_spec_name("left");
    auto right = program.impl().get_kernel_by_spec_name("right");
    const auto& left_handles = left->scratchpad_binding_handles();
    const auto& right_handles = right->scratchpad_binding_handles();
    ASSERT_EQ(left_handles.size(), 2u);
    ASSERT_EQ(right_handles.size(), 2u);
    auto check_addresses = [&] {
        for (size_t i = 0; i < left_handles.size(); ++i) {
            const auto& a = left_handles[i];
            const auto& b = right_handles[right_handles.size() - 1 - i];
            EXPECT_NE(a.allocated_address, 0u);
            EXPECT_EQ(a.allocated_address, b.allocated_address);
            EXPECT_EQ(left->common_runtime_args_data()[a.addr_crta_word], a.allocated_address);
            EXPECT_EQ(right->common_runtime_args_data()[b.addr_crta_word], a.allocated_address);
        }
        const auto& a = left_handles[0];
        const auto& b = left_handles[1];
        EXPECT_TRUE(
            a.allocated_address + a.size_bytes <= b.allocated_address ||
            b.allocated_address + b.size_bytes <= a.allocated_address);
    };
    check_addresses();
    SetProgramRunArgs(program, ProgramRunArgs{});
    check_addresses();
}

TEST_F(ProgramSpecTestGen1, CPU_SharedScratchpadUsesEntireBinderNodeUnion) {
    // The DFB occupies only nodes 1 and 2. The shared scratchpad must clear it even
    // if the first binding kernel's node set does not include node 2.
    for (bool reverse_order : {false, true}) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        auto extra = MakeMinimalGen1DMKernel("extra", DataMovementProcessor::RISCV_1);
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
                .advanced_options = {.allow_instance_multi_binding = true}},
            ScratchpadSpec{.unique_id = ScratchpadSpecName{"private"}, .size_per_node = 512},
            ScratchpadSpec{
                .unique_id = ScratchpadSpecName{"other_shared"},
                .size_per_node = 256,
                .advanced_options = {.allow_instance_multi_binding = true}},
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
