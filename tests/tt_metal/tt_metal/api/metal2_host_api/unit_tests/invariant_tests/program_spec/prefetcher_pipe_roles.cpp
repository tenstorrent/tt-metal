// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariants on PrefetcherPipe parameters and bindings (program_spec.hpp /
// prefetcher_pipe_parameter.hpp): references, usage, sender/receiver roles and accessor groups.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

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
using test_helpers::pipe_param_name;
using test_helpers::pipe_receiver_nodes;
using test_helpers::pipe_ring_size;
using test_helpers::pipe_sender_node;
using test_helpers::PrefetcherPipeSpecTestGen1;
using test_helpers::PrefetcherPipeSpecTestQuasar;

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_FullSpecPassesValidation) {
    ProgramSpec spec = MakeFullPipeSpec();
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SenderOnlySpecPassesValidation) {
    ProgramSpec spec = MakeSenderOnlySpec();
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ReceiverOnlyRelaySpecPassesValidation) {
    // Receivers + compute only; the sender is in a different Program. The parameter is used by the
    // relay and the receiver binding.
    ProgramSpec spec = MakeFullPipeSpec();
    spec.kernels.erase(spec.kernels.begin());  // drop "sender"
    spec.work_units.erase(spec.work_units.begin());
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiSenderOneKernelPasses) {
    // One prefetcher kernel on two sender cores, each owning its own 1:N pipe, through one accessor.
    ProgramSpec spec;
    spec.name = "two_senders";
    auto sender = MakeMinimalGen2DMKernel("sender");
    sender.advanced_options.prefetcher_pipe_bindings.push_back(BindPipes({pipe_param_name, other_param_name}, "out"));
    spec.kernels = {sender};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter(), MakeOtherPipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit("sender_wu", both_sender_nodes, {"sender"})};
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeRelayTilingReceiversPasses) {
    // Two pipes whose receivers tile the receiver kernel's nodes and the relay's nodes:
    // (0,1)..(0,3) + (1,1)..(1,3).
    ProgramSpec spec = MakeTwoPipeReceiverSpec();
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeReceiverWithoutRelayPasses) {
    ProgramSpec spec = MakeTwoPipeReceiverSpec();
    spec.kernels.pop_back();  // drop "compute"
    spec.dataflow_buffers.clear();
    KernelNamed(spec, "receiver").dfb_bindings.clear();
    spec.work_units[0].kernels = {KernelSpecName{"receiver"}};
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_FullSpecWithSeparateAccessorsPasses) {
    // Different pipes under different accessors on the same kernel are fine when each group
    // tiles the kernel's nodes on its own; here both are single-pipe groups on one sender node.
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters.push_back(
        MakeOtherPipeParameter());  // different receivers, same sender node
    KernelNamed(spec, "sender")
        .advanced_options.prefetcher_pipe_bindings.push_back(BindPipes({other_param_name}, "other"));
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_DuplicateParameterNameFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters.push_back(MakePipeParameter());
    EXPECT_SPEC_REJECTED(spec, "Duplicate PrefetcherPipeParameter name 'weights'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_BindingUnknownParameterFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings[0].pipe_parameter_names = {
        PrefetcherPipeParamName{"nope"}};
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' accessor 'weights' references unknown PrefetcherPipeParameter 'nope'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayUnknownParameterFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].advanced_options.prefetcher_pipe_relays = {PrefetcherPipeParamName{"nope"}};
    EXPECT_SPEC_REJECTED(spec, "DFB 'weights_relay' relays unknown PrefetcherPipeParameter 'nope'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_UnusedParameterFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings.clear();
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' is defined but not bound by any kernel or relay DFB");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_BindingKernelOnAnyNonReceiverNodeIsSender) {
    // The spec names no sender, so a one-node kernel anywhere outside the receivers is the sender
    // role; whether the pipe's sender is really there is settled when the pipe is supplied
    // (CPU_SetRunArgsSenderNotOnSenderKernelFails).
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.work_units[0].target_nodes = NodeCoord{2, 2};
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_BindingKernelCoversPartialReceiversFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.work_units[1].target_nodes = NodeRange{NodeCoord{0, 1}, NodeCoord{0, 2}};  // 2 of 3 receivers
    EXPECT_SPEC_REJECTED(
        spec,
        "Kernel 'receiver' accessor 'weights' (1 pipe(s)): the kernel's WorkUnitSpec nodes must equal either the "
        "union of the group's receiver nodes (receiver role) or be 1 node(s) outside them, one per pipe (sender "
        "role). Kernel covers 2 node(s): 2 of the 3 receiver node(s), 0 outside the receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_BindingKernelOnSenderAndReceiversFails) {
    // One kernel spanning both ends of one pipe: mixed role.
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.work_units[0].target_nodes =
        NodeRangeSet(std::vector<NodeRange>{NodeRange{pipe_sender_node, pipe_sender_node}, pipe_receiver_nodes});
    EXPECT_SPEC_REJECTED(spec, "Kernel covers 4 node(s): 3 of the 3 receiver node(s), 1 outside the receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_TwoKernelsBindingOnSenderFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    auto second = MakeMinimalGen2DMKernel("sender2");
    second.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels.push_back(second);
    spec.work_units[0].kernels.push_back(KernelSpecName{"sender2"});
    EXPECT_SPEC_REJECTED(
        spec, "Kernels 'sender' and 'sender2' both bind PrefetcherPipeParameter 'weights' as its sender");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_TwoKernelsBindingOnReceiversFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    auto second = MakeMinimalGen2DMKernel("receiver2");
    second.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels.push_back(second);
    spec.work_units[1].kernels.push_back(KernelSpecName{"receiver2"});
    EXPECT_SPEC_REJECTED(
        spec, "Kernels 'receiver' and 'receiver2' both bind PrefetcherPipeParameter 'weights' as its receiver");
}

TEST_F(PrefetcherPipeSpecTestGen1, CPU_TwoDMKernelsOnSenderFails) {
    // WH/BH: only one DM may own the sender credits on a node.
    ProgramSpec spec;
    spec.name = "gen1_dual_sender";
    auto brisc = MakeMinimalGen1DMKernel("brisc", DataMovementProcessor::RISCV_0);
    brisc.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    auto ncrisc = MakeMinimalGen1DMKernel("ncrisc", DataMovementProcessor::RISCV_1);
    ncrisc.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels = {brisc, ncrisc};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit("sender_wu", pipe_sender_node, {"brisc", "ncrisc"})};
    EXPECT_SPEC_REJECTED(
        spec, "Kernels 'brisc' and 'ncrisc' both bind PrefetcherPipeParameter 'weights' as its sender");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_AccessorGroupMixedRolesFails) {
    // Kernel on weights' SENDER and other's RECEIVERS, binding both under one accessor.
    ProgramSpec spec;
    spec.name = "mixed";
    auto kernel = MakeMinimalGen2DMKernel("mixed");
    kernel.advanced_options.prefetcher_pipe_bindings.push_back(BindPipes({pipe_param_name, other_param_name}, "io"));
    spec.kernels = {kernel};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter(), MakeOtherPipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit(
        "wu",
        NodeRangeSet(std::vector<NodeRange>{NodeRange{pipe_sender_node, pipe_sender_node}, other_receiver_nodes}),
        {"mixed"})};
    EXPECT_SPEC_REJECTED(
        spec,
        "Kernel 'mixed' accessor 'io' (2 pipe(s)): the kernel's WorkUnitSpec nodes must equal either the union of "
        "the group's receiver nodes (receiver role) or be 2 node(s) outside them, one per pipe (sender role). Kernel "
        "covers 4 node(s): 3 of the 6 receiver node(s), 1 outside the receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_AccessorGroupPartialTilingFails) {
    // Receiver kernel binds both pipes but only runs on weights' receivers.
    ProgramSpec spec = MakeTwoPipeReceiverSpec();
    spec.work_units[0].target_nodes = pipe_receiver_nodes;
    EXPECT_SPEC_REJECTED(spec, "Kernel covers 3 node(s): 3 of the 6 receiver node(s), 0 outside the receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_AccessorGroupSpillsOutsideFails) {
    // Receiver kernel covers both receiver sets plus one unrelated node.
    ProgramSpec spec = MakeTwoPipeReceiverSpec();
    spec.work_units[0].target_nodes = NodeRangeSet(
        std::vector<NodeRange>{pipe_receiver_nodes, other_receiver_nodes, NodeRange{NodeCoord{3, 3}, NodeCoord{3, 3}}});
    EXPECT_SPEC_REJECTED(spec, "Kernel covers 7 node(s): 6 of the 6 receiver node(s), 1 outside the receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_AccessorGroupOverlappingReceiversFails) {
    ProgramSpec spec = MakeTwoPipeReceiverSpec();
    spec.advanced_options.prefetcher_pipe_parameters[1].receivers = pipe_receiver_nodes;  // same receivers as "weights"
    EXPECT_SPEC_REJECTED(
        spec,
        "Kernel 'receiver' accessor 'in' names PrefetcherPipeParameter 'other' whose receiver nodes overlap another "
        "pipe's in the same accessor");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_AccessorGroupSenderKernelOnReceiverFails) {
    // The two-pipe sender kernel runs on (0,0) and (0,2); (0,2) is one of weights' receivers, so
    // that node would host two pipes. Neither role fits.
    ProgramSpec spec;
    spec.name = "sender_in_receivers";
    auto sender = MakeMinimalGen2DMKernel("sender");
    sender.advanced_options.prefetcher_pipe_bindings.push_back(BindPipes({pipe_param_name, other_param_name}, "out"));
    spec.kernels = {sender};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter(), MakeOtherPipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit(
        "sender_wu",
        NodeRangeSet(std::vector<NodeRange>{NodeRange{pipe_sender_node, pipe_sender_node}, NodeRange{{0, 2}, {0, 2}}}),
        {"sender"})};
    EXPECT_SPEC_REJECTED(spec, "Kernel covers 2 node(s): 1 of the 6 receiver node(s), 1 outside the receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_AccessorGroupSenderNodeCountMismatchFails) {
    // Two pipes under one sender accessor need exactly two sender nodes.
    ProgramSpec spec;
    spec.name = "three_sender_nodes";
    auto sender = MakeMinimalGen2DMKernel("sender");
    sender.advanced_options.prefetcher_pipe_bindings.push_back(BindPipes({pipe_param_name, other_param_name}, "out"));
    spec.kernels = {sender};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter(), MakeOtherPipeParameter()};
    spec.work_units = {
        MakeMinimalWorkUnit("sender_wu", NodeRangeSet(NodeRange{NodeCoord{0, 0}, NodeCoord{2, 0}}), {"sender"})};
    EXPECT_SPEC_REJECTED(spec, "Kernel covers 3 node(s): 0 of the 6 receiver node(s), 3 outside the receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_AccessorGroupGeometryMismatchFails) {
    ProgramSpec spec = MakeTwoPipeReceiverSpec();
    spec.advanced_options.prefetcher_pipe_parameters[1].ring_size = pipe_ring_size * 2;
    EXPECT_SPEC_REJECTED(
        spec,
        "Kernel 'receiver' accessor 'in' names PrefetcherPipeParameters 'weights' (ring_size 8192, entry_size 2048) "
        "and 'other' (ring_size 16384, entry_size 2048); pipes sharing an accessor must share ring_size and "
        "entry_size");
}

}  // namespace
}  // namespace tt::tt_metal::experimental
