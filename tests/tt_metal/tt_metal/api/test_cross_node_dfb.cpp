// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/core_coord.hpp>
#include "impl/dataflow_buffer/cross_node_dfb.hpp"
#include "impl/dataflow_buffer/dataflow_buffer.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <exception>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>

// Access to internal API: ProgramImpl::finalize_offsets, get_sem_base_addr
#include "impl/program/program_impl.hpp"
#include "impl/program/dispatch.hpp"
#include "impl/context/context_types.hpp"
#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/api/cross_node_dfb_test_utils.hpp"

namespace tt::tt_metal {

// CrossNode device API is WH/BH-only for now; Quasar support is a follow-up.
class CrossNodeDFBFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        MeshDispatchFixture::SetUp();
        if (this->arch_ == tt::ARCH::QUASAR) {
            GTEST_SKIP() << "CrossNodeDFB is not supported on Quasar yet";
        }
    }
};

// ---------------------------------------------------------------------------
// Group 1: Direct mirrors of test_global_circular_buffers.cpp
// ---------------------------------------------------------------------------

TEST_F(CrossNodeDFBFixture, CreateCrossNodeDFBs) {
    CoreRangeSet cores(CoreRange({1, 1}, {1, 1}));
    CoreRangeSet cores2(CoreRange({1, 1}, {2, 2}));
    auto mesh_device = devices_[0];

    // Valid 1:1 mapping - should not throw.
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), cores}};
        EXPECT_NO_THROW(experimental::CrossNodeDFB(mesh_device.get(), mapping, /*entry_size=*/256, /*num_entries=*/4));
    }
    // Sender core appears in its own receiver CoreRangeSet (sender-receiver overlap).
    {
        CoreRangeSet overlap_cores(CoreRange({0, 0}, {0, 0}));
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), overlap_cores}};
        EXPECT_THROW(experimental::CrossNodeDFB(mesh_device.get(), mapping, 256, 4), std::exception);
    }
    // Two senders share a receiver core (receiver sets overlap across senders).
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), cores}, {CoreCoord(0, 1), cores2}};
        EXPECT_THROW(experimental::CrossNodeDFB(mesh_device.get(), mapping, 256, 4), std::exception);
    }
}

TEST_F(CrossNodeDFBFixture, ProgramCrossNodeDFBsAPI) {
    CoreCoord sender_core = CoreCoord(0, 0);
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    CoreRangeSet receiver_cores(CoreRange({1, 1}, {2, 2}));
    CoreRangeSet dummy_receiver_cores(CoreRange({3, 3}, {3, 3}));
    auto all_cores = sender_cores.merge(receiver_cores).merge(dummy_receiver_cores);

    auto mesh_device = devices_[0];

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};

    // Valid: create into program on all mapping cores.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});

        const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, mesh_device.get(), mapping, 256, 4);
        EXPECT_EQ(remote_dfb_id, 0u);

        program.impl().compile(mesh_device.get());
        program.impl().finalize_offsets(mesh_device.get());

        const auto& hal = MetalContext::instance().hal();
        uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        EXPECT_FALSE(program.impl().get_per_core_cross_node_dfbs().empty());
        const auto& kernel_groups = program.impl().get_kernel_groups(index);
        ASSERT_FALSE(kernel_groups.empty());
        EXPECT_NE(
            kernel_groups[0]->launch_msg.view().kernel_config().cross_node_dfb_offset(), CROSS_NODE_DFB_OFFSET_NONE);
    }
    // UpdateDynamicCrossNodeDFBAddress: valid case - retargets to a distinct matching buffer.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, mesh_device.get(), mapping, 256, 4);
        program.impl().compile(mesh_device.get());
        program.impl().finalize_offsets(mesh_device.get());

        const uint32_t original_data_addr = program.impl().get_cross_node_dfb(remote_dfb_id).buffer_address();
        const uint32_t original_page_size = program.impl().get_cross_node_dfb(remote_dfb_id).config_page_size();

        // Shard grid must match the CrossNodeDFB's cores (sender ∪ receivers), which
        // excludes the dummy core the kernel also runs on.
        auto replacement = cross_node_dfb_test::make_cross_node_data_buffer(
            mesh_device.get(), sender_cores.merge(receiver_cores), /*entry_size=*/256, /*num_entries=*/4);
        ASSERT_NE(static_cast<uint32_t>(replacement->address()), original_data_addr);

        EXPECT_NO_THROW(experimental::UpdateDynamicCrossNodeDFBAddress(program, remote_dfb_id, *replacement));
        EXPECT_EQ(
            program.impl().get_cross_node_dfb(remote_dfb_id).buffer_address(),
            static_cast<uint32_t>(replacement->address()));
        EXPECT_EQ(program.impl().get_cross_node_dfb(remote_dfb_id).config_page_size(), original_page_size);
    }
    // UpdateDynamicCrossNodeDFBAddress: invalid case - throws when gdfb does not match.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, mesh_device.get(), mapping, 256, 4);
        const CoreRangeSet dummy_all_cores = CoreRangeSet(CoreRange(CoreCoord(0, 0))).merge(dummy_receiver_cores);
        auto dummy_data = cross_node_dfb_test::make_cross_node_data_buffer(
            mesh_device.get(), dummy_all_cores, /*entry_size=*/256, /*num_entries=*/4);
        program.impl().compile(mesh_device.get());
        program.impl().finalize_offsets(mesh_device.get());
        EXPECT_THROW(
            experimental::UpdateDynamicCrossNodeDFBAddress(program, remote_dfb_id, *dummy_data), std::exception);
    }
    // No CrossNodeDFBs created: cross_node_dfb_offset must be CROSS_NODE_DFB_OFFSET_NONE.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        program.impl().compile(mesh_device.get());
        program.impl().finalize_offsets(mesh_device.get());
        const auto& hal = MetalContext::instance().hal();
        uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        EXPECT_EQ(program.impl().get_program_config(index).cross_node_dfb_offset, CROSS_NODE_DFB_OFFSET_NONE);
    }
}

// MeshWorkload finalizes offsets across all programs together. Programs can share logical
// core coordinates across device ranges; a program with no CrossNodeDFB must keep
// CROSS_NODE_DFB_OFFSET_NONE even when another program in the workload has participants on
// the same logical cores.
TEST_F(CrossNodeDFBFixture, MeshWorkload_CrossNodeOffsetUsesPerProgramParticipants) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {2, 0}));
    auto all_cores = CoreRangeSet(CoreRange(sender_core)).merge(receiver_cores);
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};

    auto make_blank_program = [&]() {
        Program program = CreateProgram();
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        return program;
    };

    Program program_with_cn = make_blank_program();
    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program_with_cn, mesh_device.get(), mapping, 256, 4);
    EXPECT_EQ(remote_dfb_id, 0u);

    Program program_without_cn = make_blank_program();
    EXPECT_TRUE(program_without_cn.impl().get_per_core_cross_node_dfbs().empty());

    // Match MeshWorkload: compile/allocate each program, then finalize offsets once across all.
    program_with_cn.impl().compile_and_allocate(mesh_device.get(), /*force_slow_dispatch=*/false);
    program_without_cn.impl().compile_and_allocate(mesh_device.get(), /*force_slow_dispatch=*/false);

    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(mesh_device.get()));
    const auto& hal = metal_ctx.hal();
    const uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    ASSERT_FALSE(program_with_cn.impl().get_kernel_groups(index).empty());
    ASSERT_FALSE(program_without_cn.impl().get_kernel_groups(index).empty());

    std::array<detail::ProgramImpl*, 2> programs = {&program_with_cn.impl(), &program_without_cn.impl()};
    // Any L1-aligned offset that is not the NONE sentinel.
    constexpr uint32_t kSharedRegionOffset = 256;
    program_dispatch::finalize_cross_node_dfbs(
        metal_ctx, index, ttsl::Span<detail::ProgramImpl*>(programs.data(), programs.size()), kSharedRegionOffset);

    for (const auto& kg : program_with_cn.impl().get_kernel_groups(index)) {
        EXPECT_EQ(
            kg->launch_msg.view().kernel_config().cross_node_dfb_offset(), static_cast<uint16_t>(kSharedRegionOffset));
    }
    for (const auto& kg : program_without_cn.impl().get_kernel_groups(index)) {
        EXPECT_EQ(kg->launch_msg.view().kernel_config().cross_node_dfb_offset(), CROSS_NODE_DFB_OFFSET_NONE);
    }
}

// Kernel groups are formed from which kernels are placed on cores, independent of
// CrossNodeDFB participation. A single shared kernel over sender + receivers (+ extras)
// therefore lands in one kernel-group range even though:
//   - CreateCrossNodeRelay stamps relay_dfb_id only on receiver cores (sender stays INVALID)
//   - cores outside all_cores() are non-participants
// Dispatch must partition that range by identical payload before multicasting.
TEST_F(CrossNodeDFBFixture, DispatchPartitionsHeterogeneousKernelGroupByPayload) {
    auto mesh_device = devices_[0];

    const CoreCoord sender_core(0, 0);
    const CoreRangeSet receiver_cores(CoreRange({1, 0}, {2, 0}));
    const CoreCoord non_participant_core(3, 0);
    // One kernel covers participant and non-participant cores → one heterogeneous kernel group.
    const CoreRangeSet kernel_cores(CoreRange({0, 0}, {3, 0}));
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        kernel_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, mesh_device.get(), mapping, 256, 4);
    experimental::dfb::DataflowBufferConfig relay_config{.entry_size = 256, .num_entries = 4};
    const uint32_t relay_host_id =
        experimental::CreateCrossNodeRelayDataflowBuffer(program, receiver_cores, relay_config, remote_dfb_id);
    const uint8_t relay_device_slot =
        static_cast<uint8_t>(program.impl().get_dataflow_buffer(relay_host_id)->device_slot);

    program.impl().compile_and_allocate(mesh_device.get(), /*force_slow_dispatch=*/false);

    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(mesh_device.get()));
    const auto& hal = metal_ctx.hal();
    const uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const auto& kernel_groups = program.impl().get_kernel_groups(index);
    ASSERT_EQ(kernel_groups.size(), 1u);
    EXPECT_TRUE(kernel_groups[0]->core_ranges.contains(kernel_cores));

    const auto& participants = program.impl().get_per_core_cross_node_dfbs();
    ASSERT_TRUE(participants.count(sender_core));
    ASSERT_TRUE(participants.count(CoreCoord(1, 0)));
    ASSERT_TRUE(participants.count(CoreCoord(2, 0)));
    EXPECT_FALSE(participants.count(non_participant_core));
    EXPECT_EQ(participants.at(sender_core).at(0).relay_dfb_id, std::numeric_limits<uint8_t>::max());
    EXPECT_EQ(participants.at(CoreCoord(1, 0)).at(0).relay_dfb_id, relay_device_slot);
    EXPECT_EQ(participants.at(CoreCoord(2, 0)).at(0).relay_dfb_id, relay_device_slot);

    const auto groups = program_dispatch::partition_cores_by_cross_node_dfb_payload(
        kernel_groups[0]->core_ranges, participants, program.impl().num_cross_node_dfb_slots());

    // Sender (INVALID relay) and receivers (shared relay slot) must not share a multicast.
    ASSERT_EQ(groups.size(), 2u);
    uint32_t total_cores = 0;
    for (const auto& group : groups) {
        EXPECT_FALSE(group.cores.contains(non_participant_core));
        total_cores += group.cores.num_cores();
        const uint32_t relay_word = group.payload[CROSS_NODE_DFB_REGION_HEADER_WORDS + 2];
        if (group.cores.contains(sender_core)) {
            EXPECT_EQ(group.cores.num_cores(), 1u);
            EXPECT_EQ(relay_word, std::numeric_limits<uint8_t>::max());
        } else {
            EXPECT_EQ(group.cores.num_cores(), 2u);
            EXPECT_EQ(group.cores.size(), 1u);  // merged into one multicast rectangle
            EXPECT_EQ(relay_word, relay_device_slot);
        }
        EXPECT_EQ(group.payload[0], 1u);
        EXPECT_TRUE(group.cores.contains(group.representative_core));
    }
    EXPECT_EQ(total_cores, 3u);
}

// ---------------------------------------------------------------------------
// Group 2: DFB-specific host-API tests (no kernel execution)
// ---------------------------------------------------------------------------

TEST_F(CrossNodeDFBFixture, CreateCrossNodeDFBs_MultiSender) {
    auto mesh_device = devices_[0];

    // Valid M:N: 2 independent senders, each with a disjoint CoreRangeSet.
    {
        CoreRangeSet recv0(CoreRange({2, 0}, {3, 0}));
        CoreRangeSet recv1(CoreRange({2, 1}, {3, 1}));
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), recv0}, {CoreCoord(1, 0), recv1}};
        EXPECT_NO_THROW(experimental::CrossNodeDFB(mesh_device.get(), mapping, 256, 4));
    }
    // Single sender with 2 receivers: creates without error.
    {
        CoreRangeSet recv(CoreRange({1, 0}, {2, 0}));
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), recv}};
        EXPECT_NO_THROW(experimental::CrossNodeDFB(mesh_device.get(), mapping, 256, 4));
    }
}

TEST_F(CrossNodeDFBFixture, ProgramCrossNodeDFBsAPI_SlotAssignment) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {2, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    auto all_cores = sender_cores.merge(receiver_cores);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};

    tt_metal::Program program = CreateProgram();
    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, mesh_device.get(), mapping, 256, 4);
    EXPECT_EQ(remote_dfb_id, 0u);

    detail::ProgramImpl& impl = program.impl();
    auto it = impl.get_per_core_cross_node_dfbs().find(CoreCoord(1, 0));
    ASSERT_NE(it, impl.get_per_core_cross_node_dfbs().end());
    ASSERT_FALSE(it->second.empty());
    EXPECT_EQ(it->second[0].remote_dfb_id, 0u);
    EXPECT_TRUE(impl.get_per_core_cross_node_dfbs().count(sender_core));
}

TEST_F(CrossNodeDFBFixture, ProgramCrossNodeDFBsAPI_IndependentTopologiesUseProgramWideSlots) {
    auto mesh_device = devices_[0];

    // Program-wide remote_dfb_id holes: two independent topologies in one program.
    // Host participant records stay sparse; device payloads stay dense with
    // config_page_addr==0 holes.
    const CoreCoord sender0(0, 0);
    const CoreCoord receiver0(1, 0);
    const CoreCoord sender1(0, 1);
    const CoreCoord receiver1(1, 1);
    const CoreRangeSet topology0_cores(std::vector<CoreRange>{CoreRange(sender0), CoreRange(receiver0)});
    const CoreRangeSet topology1_cores(std::vector<CoreRange>{CoreRange(sender1), CoreRange(receiver1)});
    const CoreRangeSet all_cores = topology0_cores.merge(topology1_cores);

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    const uint8_t slot0 = experimental::CreateCrossNodeDFB(
        program,
        mesh_device.get(),
        {{sender0, CoreRangeSet(CoreRange(receiver0))}},
        /*entry_size=*/256,
        /*num_entries=*/4);
    const uint8_t slot1 = experimental::CreateCrossNodeDFB(
        program,
        mesh_device.get(),
        {{sender1, CoreRangeSet(CoreRange(receiver1))}},
        /*entry_size=*/256,
        /*num_entries=*/4);
    EXPECT_EQ(slot0, 0u);
    EXPECT_EQ(slot1, 1u);
    EXPECT_EQ(program.impl().num_cross_node_dfb_slots(), 2u);

    // Host storage is sparse: each core only lists the slots it participates in.
    // Each participant points at its slot's dedicated sharded config Buffer.
    const auto& per_core = program.impl().get_per_core_cross_node_dfbs();
    for (const CoreCoord& core : corerange_to_cores(topology0_cores)) {
        const auto& slots = per_core.at(core);
        ASSERT_EQ(slots.size(), 1u);
        EXPECT_EQ(slots[0].remote_dfb_id, 0u);
        EXPECT_EQ(slots[0].config_page_addr, program.impl().get_cross_node_dfb(0).config_address());
    }
    for (const CoreCoord& core : corerange_to_cores(topology1_cores)) {
        const auto& slots = per_core.at(core);
        ASSERT_EQ(slots.size(), 1u);
        EXPECT_EQ(slots[0].remote_dfb_id, 1u);
        EXPECT_EQ(slots[0].config_page_addr, program.impl().get_cross_node_dfb(1).config_address());
    }

    // Device payloads are dense over the program-wide slot space.
    const auto groups = program_dispatch::partition_cores_by_cross_node_dfb_payload(
        all_cores, per_core, program.impl().num_cross_node_dfb_slots());
    ASSERT_EQ(groups.size(), 2u);
    for (const auto& group : groups) {
        ASSERT_EQ(group.payload[0], 2u);
        ASSERT_EQ(group.payload.size(), cross_node_dfb_config_region_words(2));
        const bool is_topology0 = group.cores.contains(sender0);
        const uint32_t participant_slot = is_topology0 ? 0u : 1u;
        const uint32_t non_participant_slot = is_topology0 ? 1u : 0u;
        const uint32_t participant_base =
            CROSS_NODE_DFB_REGION_HEADER_WORDS + participant_slot * CROSS_NODE_DFB_CONFIG_WORDS;
        const uint32_t non_participant_base =
            CROSS_NODE_DFB_REGION_HEADER_WORDS + non_participant_slot * CROSS_NODE_DFB_CONFIG_WORDS;
        EXPECT_EQ(
            group.payload[participant_base], program.impl().get_cross_node_dfb(participant_slot).config_address());
        EXPECT_EQ(group.payload[participant_base + 1], 256u);
        EXPECT_EQ(group.payload[non_participant_base], 0u);
        EXPECT_EQ(group.payload[non_participant_base + 1], 0u);
        EXPECT_EQ(group.cores.num_cores(), 2u);
    }

    // finalize sizes only the shared dense index from the program-wide slot count (2),
    // not from any one core's sparse participant count (1).
    program.impl().compile_and_allocate(mesh_device.get(), /*force_slow_dispatch=*/false);
    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(mesh_device.get()));
    const auto& hal = metal_ctx.hal();
    const uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    detail::ProgramImpl* programs[] = {&program.impl()};
    constexpr uint32_t kBase = 256;
    const uint32_t next = program_dispatch::finalize_cross_node_dfbs(
        metal_ctx, index, ttsl::Span<detail::ProgramImpl*>(programs, 1), kBase);
    const uint32_t l1_align = hal.get_alignment(HalMemType::L1);
    const uint32_t region_bytes = cross_node_dfb_config_region_words(2) * sizeof(uint32_t);
    const uint32_t expected_next = (kBase + region_bytes + l1_align - 1) & ~(l1_align - 1);
    EXPECT_EQ(next, expected_next);
    for (const auto& kg : program.impl().get_kernel_groups(index)) {
        EXPECT_EQ(kg->launch_msg.view().kernel_config().cross_node_dfb_offset(), static_cast<uint16_t>(kBase));
    }
    // Finalize does not move the dedicated config Buffers.
    for (const CoreCoord& core : corerange_to_cores(all_cores)) {
        const auto& slots = program.impl().get_per_core_cross_node_dfbs().at(core);
        ASSERT_EQ(slots.size(), 1u);
        EXPECT_EQ(
            slots[0].config_page_addr, program.impl().get_cross_node_dfb(slots[0].remote_dfb_id).config_address());
    }
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_RelayDFB_HostRelationshipValidation) {
    auto mesh_device = devices_[0];
    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {2, 0}));
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};

    {
        Program program = CreateProgram();
        const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, mesh_device.get(), mapping, 256, 4);
        experimental::dfb::DataflowBufferConfig config{.entry_size = 256, .num_entries = 4};
        const uint32_t relay_host_id =
            experimental::CreateCrossNodeRelayDataflowBuffer(program, receiver_cores, config, remote_dfb_id);
        const uint8_t expected_slot =
            static_cast<uint8_t>(program.impl().get_dataflow_buffer(relay_host_id)->device_slot);
        for (const CoreCoord& core : corerange_to_cores(receiver_cores)) {
            const auto& participant = program.impl().get_per_core_cross_node_dfbs().at(core).at(0);
            EXPECT_EQ(participant.relay_dfb_id, expected_slot);
        }
        EXPECT_EQ(
            program.impl().get_per_core_cross_node_dfbs().at(sender_core).at(0).relay_dfb_id,
            std::numeric_limits<uint8_t>::max());

        auto replacement = cross_node_dfb_test::make_cross_node_data_buffer(
            mesh_device.get(),
            CoreRangeSet(CoreRange(sender_core)).merge(receiver_cores),
            /*entry_size=*/256,
            /*num_entries=*/4);
        experimental::UpdateDynamicCrossNodeDFBAddress(program, remote_dfb_id, *replacement);
        EXPECT_EQ(
            program.impl().get_dataflow_buffer(relay_host_id)->borrowed_addr_,
            static_cast<uint32_t>(replacement->address()));
    }

    {
        Program program = CreateProgram();
        const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, mesh_device.get(), mapping, 256, 4);
        experimental::dfb::DataflowBufferConfig wrong_size{.entry_size = 128, .num_entries = 4};
        EXPECT_THROW(
            experimental::CreateCrossNodeRelayDataflowBuffer(program, receiver_cores, wrong_size, remote_dfb_id),
            std::exception);
    }

    {
        Program program = CreateProgram();
        experimental::dfb::DataflowBufferConfig config{.entry_size = 256, .num_entries = 4};
        EXPECT_THROW(
            experimental::CreateCrossNodeRelayDataflowBuffer(program, receiver_cores, config, /*remote_dfb_id=*/0),
            std::exception);
    }
}

// ---------------------------------------------------------------------------
// Group 3: Kernel execution tests (hybrid Metal 2.0 + Metal 1.0)
// ---------------------------------------------------------------------------

static distributed::MeshCoordinateRange unit_mesh_device_range() {
    auto coord = distributed::MeshCoordinate(0, 0);
    return distributed::MeshCoordinateRange(coord, coord);
}

// Enqueue a compiled program on the unit mesh coordinate and block until done.
// Returns a reference to the program stored in workload_out (valid while workload_out lives).
static Program& run_on_mesh_device(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    Program program,
    distributed::MeshWorkload& workload_out) {
    const auto device_range = unit_mesh_device_range();
    workload_out = distributed::MeshWorkload{};
    workload_out.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_out, false);
    distributed::Finish(mesh_device->mesh_command_queue());
    return workload_out.get_programs().at(device_range);
}

// Log a compact per-entry summary of ring/staging bytes (first byte + uniform check).
static void log_cross_node_dfb_byte_summary(
    const char* label, uint32_t entry_size, uint32_t num_entries, const std::vector<uint8_t>& bytes) {
    log_info(tt::LogTest, "{}: {} entries x {} B ({} B total)", label, num_entries, entry_size, bytes.size());
    for (uint32_t i = 0; i < num_entries; ++i) {
        const uint32_t off = i * entry_size;
        if (off >= bytes.size()) {
            log_info(tt::LogTest, "  entry[{}]: (out of range)", i);
            continue;
        }
        const uint8_t first = bytes[off];
        const uint32_t check_len = std::min(entry_size, static_cast<uint32_t>(bytes.size() - off));
        const bool uniform = std::all_of(
            bytes.begin() + off + 1, bytes.begin() + off + check_len, [first](uint8_t b) { return b == first; });
        if (uniform) {
            log_info(tt::LogTest, "  entry[{}]: 0x{:02x} (all {} B)", i, first, entry_size);
        } else {
            std::string prefix;
            const uint32_t preview = std::min(check_len, 16u);
            for (uint32_t j = 0; j < preview; ++j) {
                prefix += fmt::format("{:02x} ", bytes[off + j]);
            }
            log_info(tt::LogTest, "  entry[{}]: non-uniform, first {} B: {}", i, preview, prefix);
        }
    }
}

static void log_cross_node_dfb_mismatch(
    uint32_t entry_size, const std::vector<uint8_t>& expected, const std::vector<uint8_t>& received) {
    const uint32_t compare_len = static_cast<uint32_t>(std::min(expected.size(), received.size()));
    uint32_t mismatch_count = 0;
    for (uint32_t i = 0; i < compare_len; ++i) {
        if (expected[i] != received[i]) {
            if (mismatch_count < 8) {
                log_info(
                    tt::LogTest,
                    "  byte mismatch at offset {} (entry {} + {}): expected 0x{:02x}, received 0x{:02x}",
                    i,
                    entry_size > 0 ? i / entry_size : 0,
                    entry_size > 0 ? i % entry_size : i,
                    expected[i],
                    received[i]);
            }
            mismatch_count++;
        }
    }
    if (expected.size() != received.size()) {
        log_info(tt::LogTest, "  size mismatch: expected {} B, received {} B", expected.size(), received.size());
    }
    if (mismatch_count > 8) {
        log_info(tt::LogTest, "  ... {} more byte mismatches", mismatch_count - 8);
    }
}

// Helper: build and run a 1-sender N-receiver program with a given write_primitive.
// Returns the number of receivers whose CrossNodeDFB ring matches the expected pattern on host.
static uint32_t run_1toN_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& sender_core,
    const CoreRangeSet& receiver_cores,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t write_primitive,
    uint32_t counter_base = 0) {
    IDevice* device = mesh_device.get();
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};

    const uint32_t num_receivers = static_cast<uint32_t>(corerange_to_cores(receiver_cores).size());
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(write_primitive);
    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, num_entries, num_receivers);
    tt_metal::Program program = CreateProgram();
    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, num_entries);
    const auto& gdfb = program.impl().get_cross_node_dfb(remote_dfb_id);

    tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, write_primitive, data_pattern, 0u}});

    auto recvs = corerange_to_cores(receiver_cores);
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        CoreRangeSet single = CoreRangeSet(CoreRange(recvs[ri]));
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_receiver.cpp",
            single,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, num_entries, ri}});
    }

    // Runtime args must be set before run_on_mesh_device (which internally calls
    // finalize_offsets).  finalize_offsets sizes the RTA region from the currently-set
    // args; if it runs with 0 args the CrossNodeDFB config ends up at the same offset
    // as the RTA slot and overwrites it during dispatch.
    cross_node_dfb_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, gdfb, staging_size);

    const auto sender_staging_bytes = cross_node_dfb_test::build_sender_staging_bytes(
        data_pattern, entry_size, num_entries, num_receivers, counter_base);
    cross_node_dfb_test::write_sender_l1_staging(
        *mesh_device, sender_cores, gdfb, data_pattern, entry_size, num_entries, num_receivers, counter_base);

    log_info(
        tt::LogTest,
        "run_1toN_program: sender=({},{}), receivers={}, write_primitive={}, data_pattern={}, "
        "entry_size={}, num_entries={}, counter_base={}, dfb_buffer=0x{:x}, config_page_size={}",
        sender_core.x,
        sender_core.y,
        num_receivers,
        write_primitive,
        data_pattern,
        entry_size,
        num_entries,
        counter_base,
        gdfb.buffer_address(),
        gdfb.config_page_size());
    log_info(
        tt::LogTest,
        "sender L1 staging=0x{:x} (size {} B, host-written)",
        cross_node_dfb_test::sender_l1_staging_address(gdfb, staging_size),
        staging_size);
    log_cross_node_dfb_byte_summary(
        "sender staging (host-written L1 payload)", entry_size, num_entries, sender_staging_bytes);

    distributed::MeshWorkload workload;
    Program& program_ref = run_on_mesh_device(mesh_device, std::move(program), workload);

    uint32_t pass_count = 0;
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        const auto expected = cross_node_dfb_test::expected_receiver_ring_bytes(
            data_pattern, entry_size, num_entries, ri, num_receivers, counter_base);
        const auto received = cross_node_dfb_test::read_receiver_ring_bytes(
            *mesh_device, gdfb, recvs[ri], static_cast<uint32_t>(expected.size()));
        const bool match = (received == expected);

        log_info(
            tt::LogTest,
            "receiver[{}] core=({},{}): ring verify {}",
            ri,
            recvs[ri].x,
            recvs[ri].y,
            match ? "PASS" : "FAIL");
        log_cross_node_dfb_byte_summary("  expected ring", entry_size, num_entries, expected);
        log_cross_node_dfb_byte_summary("  received ring", entry_size, num_entries, received);
        if (!match) {
            log_cross_node_dfb_mismatch(entry_size, expected, received);
        }

        if (match) {
            pass_count++;
        }
    }

    const bool credits_ok = cross_node_dfb_test::verify_credits_drained(
        *mesh_device, program_ref, remote_dfb_id, sender_core, receiver_cores, entry_size, num_entries);
    log_info(tt::LogTest, "credits drained verify {}", credits_ok ? "PASS" : "FAIL");
    if (!credits_ok) {
        return 0;
    }
    return pass_count;
}

static uint32_t relay_expected_checksum(uint32_t total_entries) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < total_entries; ++i) {
        checksum += static_cast<uint32_t>(static_cast<uint8_t>(i)) * 0x01010101u;
    }
    return checksum;
}

// Full host→DM→TRISC relay path. Returns the number of TRISC consumers that
// reported the expected entry count and payload checksum.
static uint32_t run_relay_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& sender_core,
    const CoreRangeSet& receiver_cores,
    uint32_t entry_size,
    uint32_t ring_depth,
    uint32_t total_entries,
    uint32_t batch_size,
    uint32_t trisc_delay_iterations = 0) {
    TT_FATAL(total_entries % batch_size == 0, "Relay test total_entries must be divisible by batch_size");
    TT_FATAL(ring_depth % batch_size == 0, "Relay test ring_depth must be divisible by batch_size");

    IDevice* device = mesh_device.get();
    CoreRangeSet sender_cores{CoreRange(sender_core)};
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};

    // Reserve tracked L1 allocations for sender payload and TRISC results.
    auto staging_buffer =
        cross_node_dfb_test::make_cross_node_data_buffer(mesh_device.get(), sender_cores, entry_size, total_entries);
    constexpr uint32_t result_page_size = 32;
    auto result_buffer =
        cross_node_dfb_test::make_cross_node_data_buffer(mesh_device.get(), receiver_cores, result_page_size, 1);

    const uint32_t num_receivers = receiver_cores.num_cores();
    const auto staging_bytes = cross_node_dfb_test::build_sender_staging_bytes(
        static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter),
        entry_size,
        total_entries,
        num_receivers);
    std::vector<uint32_t> staging_words((staging_bytes.size() + sizeof(uint32_t) - 1) / sizeof(uint32_t), 0);
    std::memcpy(staging_words.data(), staging_bytes.data(), staging_bytes.size());
    slow_dispatch::WriteToL1(
        *mesh_device, sender_core, static_cast<uint32_t>(staging_buffer->address()), staging_words, CoreType::WORKER);

    Program program = CreateProgram();
    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, ring_depth);

    experimental::dfb::DataflowBufferConfig relay_config{
        .entry_size = entry_size,
        .num_entries = ring_depth,
    };
    const uint32_t relay_host_id =
        experimental::CreateCrossNodeRelayDataflowBuffer(program, receiver_cores, relay_config, remote_dfb_id);

    const uint32_t relay_device_slot = program.impl().get_dataflow_buffer(relay_host_id)->device_slot;

    const KernelHandle sender_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_relay_sender.cpp",
        sender_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, total_entries, batch_size}});
    const KernelHandle receiver_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_relay_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, total_entries, batch_size}});
    const KernelHandle trisc_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_relay_trisc.cpp",
        receiver_cores,
        ComputeConfig{.compile_args = {relay_device_slot, total_entries, batch_size, trisc_delay_iterations}});

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, relay_host_id, receiver_kernel, trisc_kernel);

    SetRuntimeArgs(program, sender_kernel, sender_cores, {static_cast<uint32_t>(staging_buffer->address())});
    SetRuntimeArgs(program, trisc_kernel, receiver_cores, {static_cast<uint32_t>(result_buffer->address())});

    distributed::MeshWorkload workload;
    Program& program_ref = run_on_mesh_device(mesh_device, std::move(program), workload);

    const uint32_t expected_checksum = relay_expected_checksum(total_entries);
    uint32_t pass_count = 0;
    for (const CoreCoord& receiver_core : corerange_to_cores(receiver_cores)) {
        std::vector<uint32_t> result(2, 0);
        slow_dispatch::ReadFromL1(
            *mesh_device,
            receiver_core,
            static_cast<uint32_t>(result_buffer->address()),
            std::span<uint8_t>(reinterpret_cast<uint8_t*>(result.data()), result.size() * sizeof(uint32_t)),
            CoreType::WORKER);
        if (result[0] == total_entries && result[1] == expected_checksum) {
            ++pass_count;
        } else {
            log_error(
                tt::LogTest,
                "Relay result mismatch on core {}: count {} (expected {}), checksum 0x{:08x} (expected 0x{:08x})",
                receiver_core.str(),
                result[0],
                total_entries,
                result[1],
                expected_checksum);
        }
    }

    const bool credits_ok = cross_node_dfb_test::verify_credits_drained(
        *mesh_device, program_ref, remote_dfb_id, sender_core, receiver_cores, entry_size, total_entries);
    return credits_ok ? pass_count : 0;
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_BasicPushPop_1to1) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    // write_primitive=2: write_to_receiver(0) then push_back (1:1 uses receiver index 0).
    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 2);
    EXPECT_EQ(pass, 1u) << "1:1 basic push/pop failed";
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_RelayDFB_DMToCompute_1to1) {
    auto mesh_device = devices_[0];
    const uint32_t pass = run_relay_program(
        mesh_device,
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        /*entry_size=*/256,
        /*ring_depth=*/4,
        /*total_entries=*/4,
        /*batch_size=*/1);
    EXPECT_EQ(pass, 1u);
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_RelayDFB_Broadcast_1to4) {
    auto mesh_device = devices_[0];
    const uint32_t pass = run_relay_program(
        mesh_device,
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {4, 0})),
        /*entry_size=*/256,
        /*ring_depth=*/4,
        /*total_entries=*/4,
        /*batch_size=*/1,
        /*trisc_delay_iterations=*/1000);
    EXPECT_EQ(pass, 4u);
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_RelayDFB_Backpressure_NoOverwrite) {
    auto mesh_device = devices_[0];
    const uint32_t pass = run_relay_program(
        mesh_device,
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        /*entry_size=*/256,
        /*ring_depth=*/2,
        /*total_entries=*/8,
        /*batch_size=*/1,
        /*trisc_delay_iterations=*/10000);
    EXPECT_EQ(pass, 1u);
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_RelayDFB_MultiEntryPush) {
    auto mesh_device = devices_[0];
    const uint32_t pass = run_relay_program(
        mesh_device,
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        /*entry_size=*/256,
        /*ring_depth=*/4,
        /*total_entries=*/8,
        /*batch_size=*/2);
    EXPECT_EQ(pass, 1u);
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_WriteBroadcast_1to4) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 0);
    EXPECT_EQ(pass, 4u) << "write_broadcast 1:4 failed";
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_WriteStrided_1to4) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    // write_primitive=1: sender writes interleaved, each receiver verifies its index pattern.
    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 1);
    EXPECT_EQ(pass, 4u) << "write_strided 1:4 failed";
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_WriteToReceiver_ReceiverContiguous) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    // write_primitive=2: write_to_receiver N times then collective push_back.
    // Each receiver gets its unique data (receiver_idx pattern).
    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 2);
    EXPECT_EQ(pass, 4u) << "write_to_receiver (receiver-contiguous) 1:4 failed";
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_RoundRobinPushBackToReceiver) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 1;

    // write_primitive=3: write_to_receiver + push_back_to_receiver per iteration.
    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 3);
    EXPECT_EQ(pass, 4u) << "write_to_receiver + push_back_to_receiver round-robin failed";
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_PerReceiverCreditInterleaved_RingDepth4) {
    // Per-receiver credit with a ring deeper than one entry, pushed entry-major so the
    // receivers are never in lockstep: receiver 0 is credited entry i while receiver 1 is
    // still one entry behind. Every receiver must still see entry i at slot i, which only
    // holds if the sender keeps an independent write position per receiver (derived from
    // that receiver's entries_sent credits). A single shared cursor places receiver r's
    // entry i at slot (i * num_receivers + r) % depth, so the rings come back rotated and
    // partly overwritten.
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {2, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 5);
    EXPECT_EQ(pass, 2u) << "Interleaved push_back_to_receiver must keep an independent write position per receiver";
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_DecoupledWriteThenCredit) {
    // Layered-API check: one reserve(n) + write_broadcast(n) + flush + push_back(n).
    // Credits must stay invisible until the collective push (write_primitive=4).
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 4);
    EXPECT_EQ(pass, 4u) << "Decoupled write-then-credit (write_broadcast(n) then push_back(n)) failed";
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_MultipleSenders_MtoN) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    // 2 independent 1:2 channels, disjoint receivers.
    CoreCoord sender0(0, 0), sender1(0, 1);
    CoreRangeSet recv0(CoreRange({1, 0}, {2, 0}));
    CoreRangeSet recv1(CoreRange({1, 1}, {2, 1}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender0)).merge(CoreRangeSet(CoreRange(sender1)));
    CoreRangeSet receiver_cores = recv0.merge(recv1);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender0, recv0}, {sender1, recv1}};
    tt_metal::Program program = CreateProgram();
    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, num_entries);
    const auto& gdfb = program.impl().get_cross_node_dfb(remote_dfb_id);

    auto recvs = corerange_to_cores(receiver_cores);
    const uint32_t num_receivers = static_cast<uint32_t>(recvs.size());
    constexpr uint32_t data_pattern = static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, num_entries, num_receivers);
    tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u, data_pattern, 0u}});
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        CoreRangeSet single = CoreRangeSet(CoreRange(recvs[ri]));
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_receiver.cpp",
            single,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, num_entries, ri}});
    }

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        *mesh_device, sender_cores, gdfb, data_pattern, entry_size, num_entries, num_receivers);

    distributed::MeshWorkload workload;
    Program& program_ref = run_on_mesh_device(mesh_device, std::move(program), workload);

    uint32_t pass_count = 0;
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        if (cross_node_dfb_test::verify_receiver_ring(
                *mesh_device, gdfb, recvs[ri], data_pattern, entry_size, num_entries, ri, num_receivers)) {
            pass_count++;
        }
    }
    EXPECT_EQ(pass_count, 4u) << "Not all M:N receivers received expected data";
    EXPECT_TRUE(cross_node_dfb_test::verify_credits_drained(
        *mesh_device, program_ref, remote_dfb_id, sender0, recv0, entry_size, num_entries))
        << "M:N sender0 credits not drained";
    EXPECT_TRUE(cross_node_dfb_test::verify_credits_drained(
        *mesh_device, program_ref, remote_dfb_id, sender1, recv1, entry_size, num_entries))
        << "M:N sender1 credits not drained";
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_ProgramInitResetsPointers) {
    // CrossNode is same-program only: program init resets fifo ptrs to fifo_start_addr.
    // Two programs share one borrowed data ring. Program1 writes with counter_base=0;
    // Program2 writes with counter_base=0xA0. Finding 0xA0 at ring start proves the
    // second launch did not continue from a persisted wr_ptr.
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);
    const CoreRangeSet all_cores = sender_cores.merge(receiver_cores);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 8;
    const uint32_t entries_per_program = 4;
    constexpr uint32_t program2_counter_base = 0xA0;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto user_data =
        cross_node_dfb_test::make_cross_node_data_buffer(mesh_device.get(), all_cores, entry_size, num_entries);

    auto build_and_run = [&](uint32_t counter_base) {
        tt_metal::Program program = CreateProgram();
        const uint8_t remote_dfb_id =
            experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, num_entries, *user_data);
        // Copy keeps shared buffer refs alive for host verify after the program is moved.
        experimental::CrossNodeDFB gdfb = program.impl().get_cross_node_dfb(remote_dfb_id);

        constexpr uint32_t data_pattern =
            static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
        const uint32_t staging_size =
            cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, entries_per_program, 1);
        tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_sender.cpp",
            sender_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {
                    remote_dfb_id,
                    entry_size,
                    entries_per_program,
                    0u,  // write_broadcast
                    data_pattern,
                    0u}});
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_receiver.cpp",
            receiver_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, entries_per_program, 0u}});
        cross_node_dfb_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, gdfb, staging_size);
        cross_node_dfb_test::write_sender_l1_staging(
            *mesh_device, sender_cores, gdfb, data_pattern, entry_size, entries_per_program, 1, counter_base);

        distributed::MeshWorkload workload;
        Program& program_ref = run_on_mesh_device(mesh_device, std::move(program), workload);
        EXPECT_TRUE(cross_node_dfb_test::verify_credits_drained(
            *mesh_device, program_ref, remote_dfb_id, sender_core, receiver_cores, entry_size, entries_per_program))
            << "Program-init reset: credits not drained";
        return gdfb;
    };

    // Build/run/verify sequentially: write_sender_l1_staging mutates the same L1
    // scratch, so building program2 before program1 runs would clobber program1's pattern.
    {
        experimental::CrossNodeDFB gdfb = build_and_run(0);
        EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
            *mesh_device,
            gdfb,
            receiver_core,
            static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter),
            entry_size,
            entries_per_program,
            0,
            1,
            0))
            << "Program-init reset: first program data mismatch";
    }

    {
        experimental::CrossNodeDFB gdfb = build_and_run(program2_counter_base);
        EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
            *mesh_device,
            gdfb,
            receiver_core,
            static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter),
            entry_size,
            entries_per_program,
            0,
            1,
            program2_counter_base))
            << "Program-init reset: second program should write from fifo_start (counter_base=0xA0 at ring start)";
    }
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_UpdateDynamicCrossNodeDFBAddressFunctional) {
    // Create on ring A, then UpdateDynamic to ring B before launch.
    // Data must land in B; A must stay untouched. Config sideband address is unchanged.
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);
    const CoreRangeSet all_cores = sender_cores.merge(receiver_cores);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;
    constexpr uint32_t data_pattern = static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    constexpr uint32_t counter_base = 0x40;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto data_b =
        cross_node_dfb_test::make_cross_node_data_buffer(mesh_device.get(), all_cores, entry_size, num_entries);
    tt_metal::Program program = CreateProgram();
    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, num_entries);
    // Keep a copy of the original ring for "untouched" checks after UpdateDynamic.
    experimental::CrossNodeDFB gdfb_a = program.impl().get_cross_node_dfb(remote_dfb_id);
    const uint32_t page_size_before = gdfb_a.config_page_size();
    ASSERT_NE(gdfb_a.buffer_address(), static_cast<uint32_t>(data_b->address()))
        << "Expected distinct data rings for UpdateDynamic functional coverage";

    experimental::UpdateDynamicCrossNodeDFBAddress(program, remote_dfb_id, *data_b);
    experimental::CrossNodeDFB gdfb = program.impl().get_cross_node_dfb(remote_dfb_id);
    EXPECT_EQ(gdfb.buffer_address(), static_cast<uint32_t>(data_b->address()));
    EXPECT_EQ(gdfb.config_page_size(), page_size_before) << "UpdateDynamic should keep host config page geometry";

    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, num_entries, 1);
    tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u, data_pattern, 0u}});
    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_receiver.cpp",
        receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u}});

    // Staging / RTAs must track the live addresses after UpdateDynamic.
    cross_node_dfb_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        *mesh_device, sender_cores, gdfb, data_pattern, entry_size, num_entries, 1, counter_base);

    // Rings are not zeroed by Create; clear both so "a untouched" is meaningful and b starts clean.
    cross_node_dfb_test::zero_receiver_ring(*mesh_device, gdfb_a, receiver_core);
    cross_node_dfb_test::zero_receiver_ring(*mesh_device, gdfb, receiver_core);

    distributed::MeshWorkload workload;
    Program& program_ref = run_on_mesh_device(mesh_device, std::move(program), workload);

    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        *mesh_device, gdfb, receiver_core, data_pattern, entry_size, num_entries, 0, 1, counter_base))
        << "UpdateDynamic: expected payload in retargeted ring";
    EXPECT_TRUE(cross_node_dfb_test::verify_credits_drained(
        *mesh_device, program_ref, remote_dfb_id, sender_core, receiver_cores, entry_size, num_entries))
        << "UpdateDynamic: retargeted credits not drained";

    const auto a_ring =
        cross_node_dfb_test::read_receiver_ring_bytes(*mesh_device, gdfb_a, receiver_core, entry_size * num_entries);
    const auto expected_payload = cross_node_dfb_test::expected_receiver_ring_bytes(
        data_pattern, entry_size, num_entries, /*receiver_idx=*/0, /*num_receivers=*/1, counter_base);
    const bool a_untouched = std::all_of(a_ring.begin(), a_ring.end(), [](uint8_t b) { return b == 0; });
    EXPECT_TRUE(a_untouched) << "UpdateDynamic: gdfb_a ring should remain zero after redirect to gdfb_b";
    EXPECT_NE(a_ring, expected_payload) << "UpdateDynamic: gdfb_a must not contain the redirected payload";
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_BarrierCompletesAll) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    tt_metal::Program program = CreateProgram();
    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, num_entries);
    const auto& gdfb = program.impl().get_cross_node_dfb(remote_dfb_id);

    constexpr uint32_t data_pattern = static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, num_entries, 4);
    tt::tt_metal::KernelHandle send_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u, data_pattern, 1u}});

    auto recvs = corerange_to_cores(receiver_cores);
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        CoreRangeSet single = CoreRangeSet(CoreRange(recvs[ri]));
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_receiver.cpp",
            single,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, num_entries, ri}});
    }

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(program, send_k, sender_cores, gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        *mesh_device, sender_cores, gdfb, data_pattern, entry_size, num_entries, 4);

    distributed::MeshWorkload workload;
    Program& program_ref = run_on_mesh_device(mesh_device, std::move(program), workload);

    // barrier() blocks the sender until every receiver has acked all pushed entries.
    // Completion plus host ring/credit verification is sufficient; no device semaphore needed.
    uint32_t recv_pass_count = 0;
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        if (cross_node_dfb_test::verify_receiver_ring(
                *mesh_device, gdfb, recvs[ri], data_pattern, entry_size, num_entries, ri, 4)) {
            recv_pass_count++;
        }
    }
    EXPECT_EQ(recv_pass_count, 4u) << "Not all receivers received expected data in barrier test";
    EXPECT_TRUE(cross_node_dfb_test::verify_credits_drained(
        *mesh_device, program_ref, remote_dfb_id, sender_core, receiver_cores, entry_size, num_entries))
        << "BarrierCompletesAll: credits not drained after barrier";
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_BorrowedMemoryPushPop_1to1) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);
    const CoreRangeSet all_cores = sender_cores.merge(receiver_cores);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;
    constexpr uint32_t data_pattern =
        static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::PerReceiverConstant);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto user_data =
        cross_node_dfb_test::make_cross_node_data_buffer(mesh_device.get(), all_cores, entry_size, num_entries);
    tt_metal::Program program = CreateProgram();
    const uint8_t remote_dfb_id =
        experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, num_entries, *user_data);
    const auto& gdfb = program.impl().get_cross_node_dfb(remote_dfb_id);

    EXPECT_EQ(gdfb.buffer_address(), static_cast<uint32_t>(user_data->address()));

    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, num_entries, 1);
    tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            // write_primitive=2: write_to_receiver(0) for 1:1
            .compile_args = {remote_dfb_id, entry_size, num_entries, 2u, data_pattern, 0u}});
    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_receiver.cpp",
        receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u}});

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        *mesh_device, sender_cores, gdfb, data_pattern, entry_size, num_entries, 1);
    cross_node_dfb_test::zero_receiver_ring(*mesh_device, gdfb, receiver_core);

    distributed::MeshWorkload workload;
    Program& program_ref = run_on_mesh_device(mesh_device, std::move(program), workload);

    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        *mesh_device, gdfb, receiver_core, data_pattern, entry_size, num_entries, 0, 1))
        << "BorrowedMemoryPushPop_1to1: payload mismatch in user data ring";
    EXPECT_TRUE(cross_node_dfb_test::verify_credits_drained(
        *mesh_device, program_ref, remote_dfb_id, sender_core, receiver_cores, entry_size, num_entries))
        << "BorrowedMemoryPushPop_1to1: credits not drained";
}

TEST_F(CrossNodeDFBFixture, CreateCrossNodeDFB_BorrowedMismatch_PageSize) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();
    CoreRangeSet all_cores = CoreRangeSet(CoreRange({0, 0}, {0, 0})).merge(CoreRangeSet(CoreRange({1, 0}, {1, 0})));
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};

    auto bad = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = 128 * 2,
        .page_size = 128,  // should be entry_size * num_entries = 256 * 4
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = ShardSpecBuffer(all_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {2, 1}),
    });
    EXPECT_THROW(experimental::CrossNodeDFB(device, mapping, 256, 4, *bad), std::exception);
}

TEST_F(CrossNodeDFBFixture, CreateCrossNodeDFB_BorrowedMismatch_Cores) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();
    // Mapping uses (0,0)->(1,0); buffer covers a different pair of cores.
    CoreRangeSet wrong_cores = CoreRangeSet(CoreRange({2, 0}, {2, 0})).merge(CoreRangeSet(CoreRange({3, 0}, {3, 0})));
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    auto bad = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = 256 * 4 * 2,
        .page_size = 256 * 4,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = ShardSpecBuffer(wrong_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {2, 1}),
    });
    EXPECT_THROW(experimental::CrossNodeDFB(device, mapping, 256, 4, *bad), std::exception);
}

TEST_F(CrossNodeDFBFixture, CreateCrossNodeDFB_BorrowedMismatch_BufferType) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};

    auto bad = CreateBuffer(InterleavedBufferConfig{
        .device = device,
        .size = 256 * 4,
        .page_size = 256 * 4,
        .buffer_type = BufferType::DRAM,
    });
    EXPECT_THROW(experimental::CrossNodeDFB(device, mapping, 256, 4, *bad), std::exception);
}

TEST_F(CrossNodeDFBFixture, CreateCrossNodeDFB_BorrowedMismatch_Size) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();
    CoreRangeSet all_cores = CoreRangeSet(CoreRange({0, 0}, {0, 0})).merge(CoreRangeSet(CoreRange({1, 0}, {1, 0})));
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};

    // page_size and grid match, but size is larger than page_size * num_all_cores.
    const uint32_t ring_size = 256 * 4;
    auto bad = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = ring_size * 4,
        .page_size = ring_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = ShardSpecBuffer(all_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {2, 1}),
    });
    EXPECT_THROW(experimental::CrossNodeDFB(device, mapping, 256, 4, *bad), std::exception);
}

TEST_F(CrossNodeDFBFixture, CrossNodeDFB_BorrowedUpdateDynamic) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);
    const CoreRangeSet all_cores = sender_cores.merge(receiver_cores);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;
    constexpr uint32_t data_pattern = static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    constexpr uint32_t counter_base = 0x40;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto user_a =
        cross_node_dfb_test::make_cross_node_data_buffer(mesh_device.get(), all_cores, entry_size, num_entries);
    auto user_b =
        cross_node_dfb_test::make_cross_node_data_buffer(mesh_device.get(), all_cores, entry_size, num_entries);
    tt_metal::Program program = CreateProgram();
    const uint8_t remote_dfb_id =
        experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, num_entries, *user_a);
    experimental::CrossNodeDFB gdfb_a = program.impl().get_cross_node_dfb(remote_dfb_id);
    ASSERT_NE(gdfb_a.buffer_address(), static_cast<uint32_t>(user_b->address()));
    experimental::UpdateDynamicCrossNodeDFBAddress(program, remote_dfb_id, *user_b);
    experimental::CrossNodeDFB gdfb = program.impl().get_cross_node_dfb(remote_dfb_id);

    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, num_entries, 1);
    tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u, data_pattern, 0u}});
    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_receiver.cpp",
        receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u}});

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        *mesh_device, sender_cores, gdfb, data_pattern, entry_size, num_entries, 1, counter_base);
    cross_node_dfb_test::zero_receiver_ring(*mesh_device, gdfb_a, receiver_core);
    cross_node_dfb_test::zero_receiver_ring(*mesh_device, gdfb, receiver_core);

    distributed::MeshWorkload workload;
    Program& program_ref = run_on_mesh_device(mesh_device, std::move(program), workload);

    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        *mesh_device, gdfb, receiver_core, data_pattern, entry_size, num_entries, 0, 1, counter_base))
        << "BorrowedUpdateDynamic: expected payload in user_b ring";
    EXPECT_TRUE(cross_node_dfb_test::verify_credits_drained(
        *mesh_device, program_ref, remote_dfb_id, sender_core, receiver_cores, entry_size, num_entries))
        << "BorrowedUpdateDynamic: retargeted credits not drained";

    const auto a_ring =
        cross_node_dfb_test::read_receiver_ring_bytes(*mesh_device, gdfb_a, receiver_core, entry_size * num_entries);
    EXPECT_TRUE(std::all_of(a_ring.begin(), a_ring.end(), [](uint8_t b) { return b == 0; }))
        << "BorrowedUpdateDynamic: gdfb_a / user_a ring should remain zero";
}

// Create/UpdateDynamic must be host-only: rebuilding config pages must not eagerly write
// device L1 (would race with an in-flight launch). Materialization happens on the next
// program launch via the worker config ringbuffer.
TEST_F(CrossNodeDFBFixture, UpdateDynamicDoesNotEagerlyWriteDeviceConfig) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreRangeSet all_cores = sender_cores.merge(receiver_cores);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};

    auto data_b =
        cross_node_dfb_test::make_cross_node_data_buffer(mesh_device.get(), all_cores, entry_size, num_entries);
    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, num_entries);
    const uint32_t addr_a = program.impl().get_cross_node_dfb(remote_dfb_id).buffer_address();
    ASSERT_NE(addr_a, static_cast<uint32_t>(data_b->address()));

    // Launch once so the dedicated config Buffer holds fifo_start = addr_a.
    distributed::MeshWorkload workload1;
    Program& program_ref = run_on_mesh_device(mesh_device, std::move(program), workload1);
    const uint32_t page_addr = cross_node_dfb_test::cross_node_config_page_l1_address(program_ref, remote_dfb_id);
    std::vector<uint32_t> page_words(
        program_ref.impl().get_cross_node_dfb(remote_dfb_id).config_page_size() / sizeof(uint32_t), 0);
    slow_dispatch::ReadFromL1(
        *mesh_device,
        sender_core,
        page_addr,
        std::span<uint8_t>(reinterpret_cast<uint8_t*>(page_words.data()), page_words.size() * sizeof(uint32_t)),
        CoreType::WORKER);
    EXPECT_EQ(page_words[2], addr_a) << "First launch should materialize fifo_start = ring A";

    // UpdateDynamic rebuilds host pages only; the device config Buffer must still show addr_a.
    experimental::UpdateDynamicCrossNodeDFBAddress(program_ref, remote_dfb_id, *data_b);
    EXPECT_EQ(
        program_ref.impl().get_cross_node_dfb(remote_dfb_id).buffer_address(),
        static_cast<uint32_t>(data_b->address()));
    EXPECT_EQ(
        program_ref.impl().get_cross_node_dfb(remote_dfb_id).config_page(sender_core)[2],
        static_cast<uint32_t>(data_b->address()))
        << "Host config image must reflect the retargeted fifo_start";

    slow_dispatch::ReadFromL1(
        *mesh_device,
        sender_core,
        page_addr,
        std::span<uint8_t>(reinterpret_cast<uint8_t*>(page_words.data()), page_words.size() * sizeof(uint32_t)),
        CoreType::WORKER);
    EXPECT_EQ(page_words[2], addr_a)
        << "UpdateDynamic must not eagerly rewrite device config (would race with in-flight work)";

    // Re-launching the same workload refreshes the dedicated config Buffer from the
    // host page image. Fast dispatch inserts a pre-config stall on this CQ before
    // rewriting the shared program-owned page.
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload1, false);
    distributed::Finish(mesh_device->mesh_command_queue());
    slow_dispatch::ReadFromL1(
        *mesh_device,
        sender_core,
        page_addr,
        std::span<uint8_t>(reinterpret_cast<uint8_t*>(page_words.data()), page_words.size() * sizeof(uint32_t)),
        CoreType::WORKER);
    EXPECT_EQ(page_words[2], static_cast<uint32_t>(data_b->address()))
        << "Next launch must materialize the host-only UpdateDynamic image";
}

// Same program enqueued back-to-back nonblocking on one CQ. The second launch
// rewrites the program-owned config Buffer (zeroing credits), so dispatch must wait
// for the first launch to finish before the rewrite. Without that wait, credits of
// the in-flight first launch would be clobbered (hang or lost entries).
TEST_F(CrossNodeDFBFixture, BackToBackRelaunchOrdersConfigRewrite) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;
    constexpr uint32_t data_pattern = static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    constexpr uint32_t counter_base = 0x40;
    constexpr uint32_t num_launches = 4;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    tt_metal::Program program = CreateProgram();
    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, num_entries);
    const experimental::CrossNodeDFB gdfb = program.impl().get_cross_node_dfb(remote_dfb_id);

    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, num_entries, 1);
    tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u, data_pattern, 0u}});
    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_receiver.cpp",
        receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u}});

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        *mesh_device, sender_cores, gdfb, data_pattern, entry_size, num_entries, 1, counter_base);
    cross_node_dfb_test::zero_receiver_ring(*mesh_device, gdfb, receiver_core);

    distributed::MeshWorkload workload;
    const auto device_range = unit_mesh_device_range();
    workload.add_program(device_range, std::move(program));
    for (uint32_t i = 0; i < num_launches; ++i) {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    }
    distributed::Finish(mesh_device->mesh_command_queue());
    Program& program_ref = workload.get_programs().at(device_range);

    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        *mesh_device, gdfb, receiver_core, data_pattern, entry_size, num_entries, 0, 1, counter_base))
        << "Receiver ring corrupted across back-to-back relaunches";
    EXPECT_TRUE(cross_node_dfb_test::verify_credits_drained(
        *mesh_device, program_ref, remote_dfb_id, sender_core, receiver_cores, entry_size, num_entries))
        << "Credits not drained after back-to-back relaunches";
}

// ---------------------------------------------------------------------------
// Group 4: Mesh-trace coverage for CrossNode config rewrite + capture snapshot
// ---------------------------------------------------------------------------

class CrossNodeDFBTraceFixture : public MeshDispatchFixture {
protected:
    static constexpr size_t kTraceRegionSize = 2 << 20;  // 2 MiB

    CrossNodeDFBTraceFixture() : MeshDispatchFixture(DEFAULT_L1_SMALL_SIZE, kTraceRegionSize) {}

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        if (this->arch_ == tt::ARCH::QUASAR) {
            GTEST_SKIP() << "CrossNodeDFB is not supported on Quasar yet";
        }
        if (this->IsSlowDispatch()) {
            GTEST_SKIP() << "Mesh trace requires fast dispatch";
        }
    }
};

namespace {

struct CrossNodeTracePushPop {
    CoreCoord sender_core{0, 0};
    CoreRangeSet receiver_cores{CoreRange({1, 0}, {1, 0})};
    CoreRangeSet sender_cores{CoreRange(sender_core)};
    CoreCoord receiver_core{1, 0};
    uint32_t entry_size = 256;
    uint32_t num_entries = 4;
    uint32_t data_pattern = static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    uint32_t counter_base = 0x40;
    uint8_t remote_dfb_id = 0;
    std::optional<experimental::CrossNodeDFB> gdfb;
    distributed::MeshWorkload workload;
};

// Build a 1:1 CrossNode push/pop MeshWorkload. Staging is host-written once and
// reused across warmup / capture / replay / relaunch.
CrossNodeTracePushPop make_1to1_trace_push_pop(distributed::MeshDevice& mesh_device) {
    CrossNodeTracePushPop ctx;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{ctx.sender_core, ctx.receiver_cores}};

    Program program = CreateProgram();
    ctx.remote_dfb_id =
        experimental::CreateCrossNodeDFB(program, &mesh_device, mapping, ctx.entry_size, ctx.num_entries);
    ctx.gdfb = program.impl().get_cross_node_dfb(ctx.remote_dfb_id);

    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(ctx.data_pattern, ctx.entry_size, ctx.num_entries, 1);
    KernelHandle sender_k = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_sender.cpp",
        ctx.sender_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {ctx.remote_dfb_id, ctx.entry_size, ctx.num_entries, 0u, ctx.data_pattern, 0u}});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_receiver.cpp",
        ctx.receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {ctx.remote_dfb_id, ctx.entry_size, ctx.num_entries, 0u}});

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(
        program, sender_k, ctx.sender_cores, *ctx.gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        mesh_device,
        ctx.sender_cores,
        *ctx.gdfb,
        ctx.data_pattern,
        ctx.entry_size,
        ctx.num_entries,
        1,
        ctx.counter_base);

    ctx.workload.add_program(unit_mesh_device_range(), std::move(program));
    return ctx;
}

void expect_1to1_ring_and_credits(
    distributed::MeshDevice& mesh_device, CrossNodeTracePushPop& ctx, const char* failure_context) {
    Program& program_ref = ctx.workload.get_programs().at(unit_mesh_device_range());
    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        mesh_device,
        *ctx.gdfb,
        ctx.receiver_core,
        ctx.data_pattern,
        ctx.entry_size,
        ctx.num_entries,
        0,
        1,
        ctx.counter_base))
        << failure_context << ": receiver ring mismatch";
    EXPECT_TRUE(cross_node_dfb_test::verify_credits_drained(
        mesh_device,
        program_ref,
        ctx.remote_dfb_id,
        ctx.sender_core,
        ctx.receiver_cores,
        ctx.entry_size,
        ctx.num_entries))
        << failure_context << ": credits not drained";
}

}  // namespace

// Capture/replay must bake CrossNode config pages into TraceNode (like CBs) and
// order config-Buffer rewrites within the trace via SimpleTraceAllocator.
TEST_F(CrossNodeDFBTraceFixture, CaptureReplay_OrdersConfigRewrite) {
    auto mesh_device = devices_[0];
    auto ctx = make_1to1_trace_push_pop(*mesh_device);
    auto& cq = mesh_device->mesh_command_queue();

    // Warmup: capture cannot load binaries.
    distributed::EnqueueMeshWorkload(cq, ctx.workload, true);

    cross_node_dfb_test::zero_receiver_ring(*mesh_device, *ctx.gdfb, ctx.receiver_core);
    const auto tid = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    distributed::EnqueueMeshWorkload(cq, ctx.workload, false);
    // Second enqueue of the same CrossNode program inside the capture: the
    // allocator must stall_first on the prior launch before rewriting credits.
    distributed::EnqueueMeshWorkload(cq, ctx.workload, false);
    mesh_device->end_mesh_trace(cq.id(), tid);

    cross_node_dfb_test::zero_receiver_ring(*mesh_device, *ctx.gdfb, ctx.receiver_core);
    mesh_device->replay_mesh_trace(cq.id(), tid, true);
    expect_1to1_ring_and_credits(*mesh_device, ctx, "CaptureReplay");
    mesh_device->release_mesh_trace(tid);
}

// After replay, update_worker_state_post_trace_execution marks the config buffer
// completely full, so the next non-traced relaunch stalls on the last program in
// the trace before rewriting CrossNode config
TEST_F(CrossNodeDFBTraceFixture, ReplayThenRelaunch_OrdersConfigRewrite) {
    auto mesh_device = devices_[0];
    auto ctx = make_1to1_trace_push_pop(*mesh_device);
    auto& cq = mesh_device->mesh_command_queue();

    distributed::EnqueueMeshWorkload(cq, ctx.workload, true);

    const auto tid = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    distributed::EnqueueMeshWorkload(cq, ctx.workload, false);
    mesh_device->end_mesh_trace(cq.id(), tid);

    cross_node_dfb_test::zero_receiver_ring(*mesh_device, *ctx.gdfb, ctx.receiver_core);
    mesh_device->replay_mesh_trace(cq.id(), tid, true);
    expect_1to1_ring_and_credits(*mesh_device, ctx, "Replay");

    cross_node_dfb_test::zero_receiver_ring(*mesh_device, *ctx.gdfb, ctx.receiver_core);
    distributed::EnqueueMeshWorkload(cq, ctx.workload, false);
    distributed::Finish(cq);
    expect_1to1_ring_and_credits(*mesh_device, ctx, "PostReplayRelaunch");
    mesh_device->release_mesh_trace(tid);
}

// UpdateDynamic is host-only until the next launch. Trace capture must snapshot
// the capture-time host config pages so replay materializes the retargeted
// fifo_start (not the original Create-time ring).
TEST_F(CrossNodeDFBTraceFixture, CaptureReplay_UpdateDynamicSnapshot) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);
    const CoreRangeSet all_cores = sender_cores.merge(receiver_cores);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;
    constexpr uint32_t data_pattern = static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    constexpr uint32_t counter_base = 0x40;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto data_b =
        cross_node_dfb_test::make_cross_node_data_buffer(mesh_device.get(), all_cores, entry_size, num_entries);

    Program program = CreateProgram();
    const uint8_t remote_dfb_id = experimental::CreateCrossNodeDFB(program, device, mapping, entry_size, num_entries);
    experimental::CrossNodeDFB gdfb_a = program.impl().get_cross_node_dfb(remote_dfb_id);
    ASSERT_NE(gdfb_a.buffer_address(), static_cast<uint32_t>(data_b->address()));

    // Retarget before any launch (same order as the non-trace UpdateDynamic functional test).
    experimental::UpdateDynamicCrossNodeDFBAddress(program, remote_dfb_id, *data_b);
    const experimental::CrossNodeDFB gdfb_b = program.impl().get_cross_node_dfb(remote_dfb_id);
    EXPECT_EQ(gdfb_b.buffer_address(), static_cast<uint32_t>(data_b->address()));

    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, num_entries, 1);
    KernelHandle sender_k = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_sender.cpp",
        sender_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u, data_pattern, 0u}});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cross_node_dfb_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries, 0u}});

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, gdfb_b, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        *mesh_device, sender_cores, gdfb_b, data_pattern, entry_size, num_entries, 1, counter_base);

    distributed::MeshWorkload workload;
    workload.add_program(unit_mesh_device_range(), std::move(program));
    Program& program_ref = workload.get_programs().at(unit_mesh_device_range());
    auto& cq = mesh_device->mesh_command_queue();

    distributed::EnqueueMeshWorkload(cq, workload, true);

    cross_node_dfb_test::zero_receiver_ring(*mesh_device, gdfb_a, receiver_core);
    cross_node_dfb_test::zero_receiver_ring(*mesh_device, gdfb_b, receiver_core);

    const auto tid = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    distributed::EnqueueMeshWorkload(cq, workload, false);
    mesh_device->end_mesh_trace(cq.id(), tid);

    cross_node_dfb_test::zero_receiver_ring(*mesh_device, gdfb_a, receiver_core);
    cross_node_dfb_test::zero_receiver_ring(*mesh_device, gdfb_b, receiver_core);
    mesh_device->replay_mesh_trace(cq.id(), tid, true);

    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        *mesh_device, gdfb_b, receiver_core, data_pattern, entry_size, num_entries, 0, 1, counter_base))
        << "Trace replay must use capture-time UpdateDynamic fifo_start (ring B)";
    EXPECT_TRUE(cross_node_dfb_test::verify_credits_drained(
        *mesh_device, program_ref, remote_dfb_id, sender_core, receiver_cores, entry_size, num_entries))
        << "Credits not drained after UpdateDynamic trace replay";

    const auto a_ring =
        cross_node_dfb_test::read_receiver_ring_bytes(*mesh_device, gdfb_a, receiver_core, entry_size * num_entries);
    EXPECT_TRUE(std::all_of(a_ring.begin(), a_ring.end(), [](uint8_t b) { return b == 0; }))
        << "Ring A must remain untouched when the traced snapshot targets ring B";

    mesh_device->release_mesh_trace(tid);
}

}  // namespace tt::tt_metal
