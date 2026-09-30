// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// AdvancedProgramRunArgs::prefetcher_pipe_args: binding PrefetcherPipe objects to a Program's pipe
// parameters, and rebinding / lifetime rules.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/distributed.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/prefetcher_pipe_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindPipe;
using test_helpers::BindPipes;
using test_helpers::both_sender_nodes;
using test_helpers::KernelNamed;
using test_helpers::MakeFullPipeSpec;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeOtherPipeParameter;
using test_helpers::MakePipeParameter;
using test_helpers::MakeSenderOnlySpec;
using test_helpers::MakeTwoPipeReceiverSpec;
using test_helpers::other_param_name;
using test_helpers::other_receiver_nodes;
using test_helpers::other_sender_node;
using test_helpers::ParticipantOn;
using test_helpers::pipe_param_name;
using test_helpers::pipe_receiver_nodes;
using test_helpers::pipe_ring_size;
using test_helpers::pipe_sender_node;
using test_helpers::PipeSpaceOwner;
using test_helpers::PrefetcherPipeSpecTestQuasar;
using test_helpers::relay_dfb_name;
using test_helpers::ScopedSlowDispatchOverride;

ProgramRunArgs PipeArgs(std::vector<std::pair<PrefetcherPipeParamName, PrefetcherPipe*>> pipes) {
    ProgramRunArgs params;
    for (auto& [name, pipe] : pipes) {
        params.advanced_options.prefetcher_pipe_args.insert({name, PrefetcherPipeArgument{*pipe}});
    }
    return params;
}

#define EXPECT_THROWS_WITH(stmt, substr) \
    EXPECT_THAT([&] { stmt; }, ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(substr)))

// Persistent pipes are created before the programs that use them: placing a program's DFBs on a
// core seals that core's persistent L1 arena (allocate_dataflow_buffers), so a pipe created
// afterwards on those cores is rejected. Tests below follow that order.

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SetRunArgsBindsPipeToEverySlotAndRelay) {
    PrefetcherPipe pipe = MakeWeightsPipe(*mesh_device_);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeFullPipeSpec(/*receiver_threads=*/2));
    EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);

    SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}}));

    const auto& impl = program.impl();
    const auto* binding = impl.get_prefetcher_pipe_parameter(pipe_param_name.get());
    ASSERT_NE(binding, nullptr);
    EXPECT_EQ(binding->bound_pipe, &pipe.impl());
    for (const auto& slot_cores : binding->slots) {
        for (const CoreCoord& core : corerange_to_cores(slot_cores.cores)) {
            const auto* participant = ParticipantOn(program, core, slot_cores.prefetcher_pipe_id);
            ASSERT_NE(participant, nullptr);
            EXPECT_EQ(participant->pipe, &pipe.impl());
            EXPECT_EQ(participant->config_page_addr, pipe.config_address());
        }
    }
    // Receiver bind armed the receiver kernel's lane count on the persistent pipe.
    EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
    // The relay now aliases the ring.
    auto relay = impl.get_dataflow_buffer(impl.get_dfb_handle(relay_dfb_name.get()));
    EXPECT_EQ(relay->borrowed_addr_, pipe.buffer_address());
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SetRunArgsSenderOnlyLeavesLanesAlone) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSenderOnlySpec());
    PrefetcherPipe pipe = MakeWeightsPipe(*mesh_device_);
    SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}}));
    EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);
    const auto* binding = program.impl().get_prefetcher_pipe_parameter(pipe_param_name.get());
    ASSERT_EQ(binding->slots.size(), 1u);
    const auto* participant = ParticipantOn(program, pipe_sender_node, binding->slots[0].prefetcher_pipe_id);
    ASSERT_NE(participant, nullptr);
    EXPECT_EQ(participant->pipe, &pipe.impl());
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SetRunArgsMissingPipeFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSenderOnlySpec());
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, ProgramRunArgs{}),
        "PrefetcherPipeParameter 'weights' is declared in the Program but has no PrefetcherPipeArgument entry");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SetRunArgsUnknownParameterFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSenderOnlySpec());
    PrefetcherPipe pipe = MakeWeightsPipe(*mesh_device_);
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}, {PrefetcherPipeParamName{"nope"}, &pipe}})),
        "PrefetcherPipe argument for 'nope', but the Program declares no PrefetcherPipeParameter of that name");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SetRunArgsSenderNotOnSenderKernelFails) {
    // The spec does not name a sender; a Program that runs the sender kernel learns the sender
    // from the pipe and requires the kernel to be placed there.
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSenderOnlySpec());
    PrefetcherPipe pipe = MakePipeFor(*mesh_device_, MakePipeParameter(), CoreCoord{2, 0});
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}})),
        "is bound by a sender kernel on nodes {[0-0 - 0-0]}, but the supplied pipe's sender is (2,0)");
    EXPECT_EQ(program.impl().get_prefetcher_pipe_parameter(pipe_param_name.get())->bound_pipe, nullptr);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ReceiverOnlyProgramAcceptsAnySender) {
    // A consumer Program never states the sender: pipes from different senders bind alike.
    PrefetcherPipe pipe = MakePipeFor(*mesh_device_, MakePipeParameter(), CoreCoord{2, 0});
    ProgramSpec spec = MakeFullPipeSpec();
    spec.kernels.erase(spec.kernels.begin());  // drop "sender"
    spec.work_units.erase(spec.work_units.begin());
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}})));
    EXPECT_EQ(program.impl().get_prefetcher_pipe_parameter(pipe_param_name.get())->bound_pipe, &pipe.impl());
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SetRunArgsWrongReceiversFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSenderOnlySpec());
    PrefetcherPipeParameter param = MakePipeParameter();
    param.receivers = NodeRangeSet(NodeRange{NodeCoord{0, 1}, NodeCoord{0, 2}});  // 2 of the 3
    PrefetcherPipe pipe = MakePipeFor(*mesh_device_, param, pipe_sender_node);
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}})), "supplies a pipe whose receiver nodes");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SetRunArgsWrongRingSizeFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSenderOnlySpec());
    PrefetcherPipeParameter param = MakePipeParameter();
    param.ring_size = pipe_ring_size * 2;
    PrefetcherPipe pipe = MakePipeFor(*mesh_device_, param, pipe_sender_node);
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}})),
        "supplies a pipe with ring_size 16384 but the parameter declares ring_size 8192");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RebindingDifferentPipeFails) {
    // Sticky identity: a Program binds a parameter to one pipe object for its lifetime.
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSenderOnlySpec());
    PrefetcherPipe first = MakeWeightsPipe(*mesh_device_);
    PrefetcherPipe second = MakeWeightsPipe(*mesh_device_);
    SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &first}}));
    EXPECT_THROWS_WITH(
        UpdateProgramRunArgs(program, PipeArgs({{pipe_param_name, &second}})),
        "supplies a different PrefetcherPipe object than the one this Program is bound to");
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &second}})),
        "supplies a different PrefetcherPipe object than the one this Program is bound to");
    EXPECT_EQ(program.impl().get_prefetcher_pipe_parameter(pipe_param_name.get())->bound_pipe, &first.impl());
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RebindingSamePipeIsNoOpAndUpdateMayOmitIt) {
    PrefetcherPipe pipe = MakeWeightsPipe(*mesh_device_);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeFullPipeSpec());
    SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}}));
    EXPECT_NO_THROW(SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}})));
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}})));
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, ProgramRunArgs{}));
    EXPECT_EQ(program.impl().get_prefetcher_pipe_parameter(pipe_param_name.get())->bound_pipe, &pipe.impl());
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ResizingRelayDFBFails) {
    // The relay's geometry is the pipe's; per-run DFB size overrides do not apply to it.
    PrefetcherPipe pipe = MakeWeightsPipe(*mesh_device_);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeFullPipeSpec());
    ProgramRunArgs params = PipeArgs({{pipe_param_name, &pipe}});
    params.dfb_run_overrides.push_back({.dfb = relay_dfb_name, .num_entries = 2});
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, params), "resizes DFB 'weights_relay', which relays a PrefetcherPipe");
    SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &pipe}}));
    ProgramRunArgs update;
    update.dfb_run_overrides.push_back({.dfb = relay_dfb_name, .entry_size = 1024});
    EXPECT_THROWS_WITH(
        UpdateProgramRunArgs(program, update), "resizes DFB 'weights_relay', which relays a PrefetcherPipe");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeAccessorResolvesOnePipePerNode) {
    // One sender kernel on two sender nodes binding {weights, other} through accessor "out":
    // one slot, and each node's record points at the pipe present on that node.
    ProgramSpec spec;
    spec.name = "two_senders";
    auto sender = MakeMinimalGen2DMKernel("sender");
    sender.advanced_options.prefetcher_pipe_bindings.push_back(BindPipes({pipe_param_name, other_param_name}, "out"));
    spec.kernels = {sender};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter(), MakeOtherPipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit("sender_wu", both_sender_nodes, {"sender"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    ASSERT_EQ(program.impl().num_prefetcher_pipe_slots(), 1u);

    PrefetcherPipe weights = MakeWeightsPipe(*mesh_device_);
    PrefetcherPipe other = MakeOtherPipe(*mesh_device_);
    SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &weights}, {other_param_name, &other}}));

    const auto* on_weights_sender = ParticipantOn(program, pipe_sender_node, 0);
    const auto* on_other_sender = ParticipantOn(program, other_sender_node, 0);
    ASSERT_NE(on_weights_sender, nullptr);
    ASSERT_NE(on_other_sender, nullptr);
    EXPECT_EQ(on_weights_sender->pipe, &weights.impl());
    EXPECT_EQ(on_other_sender->pipe, &other.impl());
    EXPECT_EQ(on_weights_sender->config_page_addr, weights.config_address());
    EXPECT_EQ(on_other_sender->config_page_addr, other.config_address());
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeAccessorTwoPipesOneSenderNodeFails) {
    // Both pipes were carved with (0,0) as sender: the sender kernel's second node (1,0) would be
    // left without a pipe and (0,0) would host two. Rejected as a whole; nothing is bound.
    ProgramSpec spec;
    spec.name = "two_senders";
    auto sender = MakeMinimalGen2DMKernel("sender");
    sender.advanced_options.prefetcher_pipe_bindings.push_back(BindPipes({pipe_param_name, other_param_name}, "out"));
    spec.kernels = {sender};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter(), MakeOtherPipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit("sender_wu", both_sender_nodes, {"sender"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    PrefetcherPipe weights = MakeWeightsPipe(*mesh_device_);
    PrefetcherPipe other = MakePipeFor(*mesh_device_, MakeOtherPipeParameter(), pipe_sender_node);
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &weights}, {other_param_name, &other}})),
        "is claimed by two different PrefetcherPipe objects in one SetProgramRunArgs");
    EXPECT_EQ(program.impl().get_prefetcher_pipe_parameter(pipe_param_name.get())->bound_pipe, nullptr);
    EXPECT_EQ(ParticipantOn(program, pipe_sender_node, 0)->pipe, nullptr);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeAccessorMissingOnePipeFails) {
    PrefetcherPipe weights = MakeWeightsPipe(*mesh_device_);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeTwoPipeReceiverSpec());
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &weights}})),
        "PrefetcherPipeParameter 'other' is declared in the Program but has no PrefetcherPipeArgument entry");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeRelayWithSharedRingBinds) {
    // The persistent arena is per core, so two pipes on disjoint cores created back to back land at
    // one ring address: the relay DFB over both can alias it. (Phase 3's PrefetcherPipeSpace makes
    // this guarantee explicit.)
    PrefetcherPipe weights = MakeWeightsPipe(*mesh_device_);
    PrefetcherPipe other = MakeOtherPipe(*mesh_device_);
    ASSERT_EQ(weights.buffer_address(), other.buffer_address());
    Program program = MakeProgramFromSpec(*mesh_device_, MakeTwoPipeReceiverSpec(/*receiver_threads=*/2));
    SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &weights}, {other_param_name, &other}}));

    const auto& impl = program.impl();
    auto relay = impl.get_dataflow_buffer(impl.get_dfb_handle(relay_dfb_name.get()));
    EXPECT_EQ(relay->borrowed_addr_, weights.buffer_address());
    EXPECT_EQ(weights.impl().num_credit_lanes(), 2u);
    EXPECT_EQ(other.impl().num_credit_lanes(), 2u);
    for (const CoreCoord& core : corerange_to_cores(pipe_receiver_nodes)) {
        EXPECT_EQ(ParticipantOn(program, core, 0)->pipe, &weights.impl()) << core.str();
    }
    for (const CoreCoord& core : corerange_to_cores(other_receiver_nodes)) {
        EXPECT_EQ(ParticipantOn(program, core, 0)->pipe, &other.impl()) << core.str();
    }
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeRelayWithDifferentRingsFails) {
    // Pipes at different ring addresses cannot share one relay DFB. `filler` occupies the first
    // arena slot on other's cores so `other` lands above `weights`.
    PrefetcherPipe weights = MakeWeightsPipe(*mesh_device_);
    PrefetcherPipe filler = MakeOtherPipe(*mesh_device_);
    PrefetcherPipe other = MakeOtherPipe(*mesh_device_);
    ASSERT_NE(weights.buffer_address(), other.buffer_address());
    ASSERT_EQ(weights.buffer_address(), filler.buffer_address());
    Program program = MakeProgramFromSpec(*mesh_device_, MakeTwoPipeReceiverSpec(/*receiver_threads=*/2));
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &weights}, {other_param_name, &other}})),
        "relays several pipes through DFB");

    // All-or-nothing: the ring mismatch is only detectable once both pipes are in hand, and it
    // must not leave `weights` (checked first) bound, its receivers armed, or the relay pointed.
    const auto& impl = program.impl();
    EXPECT_EQ(impl.get_prefetcher_pipe_parameter(pipe_param_name.get())->bound_pipe, nullptr);
    EXPECT_EQ(impl.get_prefetcher_pipe_parameter(other_param_name.get())->bound_pipe, nullptr);
    for (const CoreCoord& core :
         corerange_to_cores(NodeRangeSet(pipe_receiver_nodes).merge(NodeRangeSet(other_receiver_nodes)))) {
        EXPECT_EQ(ParticipantOn(program, core, 0)->pipe, nullptr) << core.str();
    }
    EXPECT_EQ(weights.impl().num_credit_lanes(), 1u);
    EXPECT_EQ(other.impl().num_credit_lanes(), 1u);
    EXPECT_EQ(impl.get_dataflow_buffer(impl.get_dfb_handle(relay_dfb_name.get()))->borrowed_addr_, 0u);

    // So the caller can retry with a pipe that does share the ring.
    EXPECT_NO_THROW(SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &weights}, {other_param_name, &filler}})));
    EXPECT_EQ(impl.get_prefetcher_pipe_parameter(pipe_param_name.get())->bound_pipe, &weights.impl());
    EXPECT_EQ(impl.get_prefetcher_pipe_parameter(other_param_name.get())->bound_pipe, &filler.impl());
    EXPECT_EQ(weights.impl().num_credit_lanes(), 2u);
    EXPECT_EQ(filler.impl().num_credit_lanes(), 2u);
    EXPECT_EQ(
        impl.get_dataflow_buffer(impl.get_dfb_handle(relay_dfb_name.get()))->borrowed_addr_, weights.buffer_address());
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeReceiverWithoutRelayBindsBothPipes) {
    // No relay: the pipes may live at different addresses; each receiver node resolves its own.
    ProgramSpec spec = MakeTwoPipeReceiverSpec(/*receiver_threads=*/2);
    spec.kernels.pop_back();  // drop "compute"
    spec.dataflow_buffers.clear();
    KernelNamed(spec, "receiver").dfb_bindings.clear();
    spec.work_units[0].kernels = {KernelSpecName{"receiver"}};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    PrefetcherPipe weights = MakeWeightsPipe(*mesh_device_);
    PrefetcherPipe other = MakeOtherPipe(*mesh_device_);
    SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &weights}, {other_param_name, &other}}));

    EXPECT_EQ(weights.impl().num_credit_lanes(), 2u);
    EXPECT_EQ(other.impl().num_credit_lanes(), 2u);
    for (const CoreCoord& core : corerange_to_cores(pipe_receiver_nodes)) {
        const auto* participant = ParticipantOn(program, core, 0);
        ASSERT_NE(participant, nullptr) << core.str();
        EXPECT_EQ(participant->pipe, &weights.impl()) << core.str();
    }
    for (const CoreCoord& core : corerange_to_cores(other_receiver_nodes)) {
        const auto* participant = ParticipantOn(program, core, 0);
        ASSERT_NE(participant, nullptr) << core.str();
        EXPECT_EQ(participant->pipe, &other.impl()) << core.str();
    }
}

// A space must outlive its pipes. Dropping the space first detaches the pipe: the L1 is gone,
// so every later use of the pipe is rejected instead of reading freed memory.
TEST_F(PrefetcherPipeSpecTestQuasar, CPU_PipeOutlivingSpaceIsDetached) {
    std::optional<PrefetcherPipeSpace> space = CreatePrefetcherPipeSpace(
        *mesh_device_,
        PrefetcherPipeSpaceConfig{
            .sender_cores = CoreRangeSet(CoreRange(pipe_sender_node)),
            .receiver_domain = CoreRangeSet(pipe_receiver_nodes),
            .ring_size = pipe_ring_size,
            .max_receivers_per_pipe = CoreRangeSet(pipe_receiver_nodes).num_cores(),
        });
    PrefetcherPipe pipe = space->create_pipe(pipe_sender_node, CoreRangeSet(pipe_receiver_nodes));
    EXPECT_EQ(pipe.ring_size(), pipe_ring_size);

    space.reset();  // logs the violation; must not throw

    EXPECT_THROWS_WITH(pipe.buffer_address(), "the PrefetcherPipeSpace it was carved from has been destroyed");
    EXPECT_THROWS_WITH(pipe.config_address(), "the PrefetcherPipeSpace it was carved from has been destroyed");
    // Geometry recorded on the pipe itself is still readable, and destroying the pipe is safe.
    EXPECT_EQ(pipe.sender_core(), pipe_sender_node);
}

// ============================================================================
// Two chips (Wormhole N300 mock): a pipe binds only to a Program on its own mesh
// ============================================================================

class PrefetcherPipeSpecTestGen1TwoChips : public ::testing::Test, protected PipeSpaceOwner {
protected:
    void SetUp() override {
        slow_dispatch_override_.emplace();
        experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 2);
        auto meshes = distributed::MeshDevice::create_unit_meshes({0, 1});
        ASSERT_EQ(meshes.size(), 2u);
        mesh_a_ = meshes.at(0);
        mesh_b_ = meshes.at(1);
    }
    void TearDown() override {
        ReleasePipeSpaces();
        for (auto* mesh : {&mesh_a_, &mesh_b_}) {
            if (*mesh) {
                (*mesh)->close();
                mesh->reset();
            }
        }
        experimental::disable_mock_mode();
        slow_dispatch_override_.reset();
    }

    std::shared_ptr<distributed::MeshDevice> mesh_a_;
    std::shared_ptr<distributed::MeshDevice> mesh_b_;
    std::optional<ScopedSlowDispatchOverride> slow_dispatch_override_;
};

TEST_F(PrefetcherPipeSpecTestGen1TwoChips, CPU_PipeFromAnotherMeshFails) {
    // The pipe's ring and config pages are L1 on mesh B; a Program built for mesh A would pack
    // those addresses into its own dispatch payload and touch uninitialized L1.
    ProgramSpec spec;
    spec.name = "gen1_sender_only";
    auto sender = MakeMinimalGen1DMKernel("sender", DataMovementProcessor::RISCV_0);
    sender.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels = {sender};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit("sender_wu", pipe_sender_node, {"sender"})};

    PrefetcherPipe on_a = MakeWeightsPipe(*mesh_a_);
    PrefetcherPipe on_b = MakeWeightsPipe(*mesh_b_);
    Program program = MakeProgramFromSpec(*mesh_a_, spec);
    EXPECT_THROWS_WITH(
        SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &on_b}})),
        "supplies a pipe allocated on a different MeshDevice than the one this Program was built for");
    EXPECT_EQ(program.impl().get_prefetcher_pipe_parameter(pipe_param_name.get())->bound_pipe, nullptr);
    EXPECT_NO_THROW(SetProgramRunArgs(program, PipeArgs({{pipe_param_name, &on_a}})));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
