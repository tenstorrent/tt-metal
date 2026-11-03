// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariants on PrefetcherPipe relay DFBs (program_spec.hpp /
// prefetcher_pipe_parameter.hpp): relay geometry, node set and producer bindings.

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include "metal2_host_api/test_helpers/prefetcher_pipe_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindPipes;
using test_helpers::KernelNamed;
using test_helpers::MakeFullPipeSpec;
using test_helpers::MakeOtherPipeParameter;
using test_helpers::MakeTwoPipeReceiverSpec;
using test_helpers::other_param_name;
using test_helpers::pipe_entry_size;
using test_helpers::pipe_num_entries;
using test_helpers::pipe_param_name;
using test_helpers::pipe_receiver_nodes;
using test_helpers::pipe_ring_size;
using test_helpers::PrefetcherPipeSpecTestQuasar;
using test_helpers::relay_dfb_name;

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayPagingEachEntryFinerPasses) {
    // Two relay entries per pipe entry, from a single-threaded relay producer.
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].entry_size = pipe_entry_size / 2;
    spec.dataflow_buffers[0].num_entries = pipe_num_entries * 2;
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayPagingEachEntryFinerWithLanesFails) {
    ProgramSpec spec = MakeFullPipeSpec(/*receiver_threads=*/2);
    spec.dataflow_buffers[0].entry_size = pipe_entry_size / 2;
    spec.dataflow_buffers[0].num_entries = pipe_num_entries * 2;
    EXPECT_SPEC_REJECTED(spec, "needs a single-threaded relay producer, but kernel 'receiver' has 2 threads");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayEntryNotDividingPipeEntryFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].entry_size = pipe_entry_size * 3 / 4;
    EXPECT_SPEC_REJECTED(spec, "entry_size 1536 must divide relayed PrefetcherPipeParameter 'weights' entry_size 2048");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayNotCoveringRingFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].num_entries = pipe_num_entries - 1;
    EXPECT_SPEC_REJECTED(spec, "bytes of whole entries in relayed PrefetcherPipeParameter 'weights'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayNodesNotReceiversFails) {
    // Relay DFB (and its kernels) on 2 of the 3 receivers, without binding the pipe directly.
    ProgramSpec spec = MakeFullPipeSpec();
    KernelNamed(spec, "receiver").advanced_options.prefetcher_pipe_bindings.clear();
    spec.work_units[1].target_nodes = NodeRange{NodeCoord{0, 1}, NodeCoord{0, 2}};
    EXPECT_SPEC_REJECTED(spec, "receiver nodes do not match the DFB's node set");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayComputeProducerFails) {
    // Swap roles: compute produces, DM consumes. Compute cannot bind the pipe, so it cannot be the
    // relay's producer.
    ProgramSpec spec = MakeFullPipeSpec();
    KernelNamed(spec, "receiver").dfb_bindings = {ConsumerOf(relay_dfb_name, "relay")};
    KernelNamed(spec, "compute").dfb_bindings = {ProducerOf(relay_dfb_name, "relay")};
    EXPECT_SPEC_REJECTED(
        spec,
        "Kernel 'compute' is a PRODUCER of relay DFB 'weights_relay' but has no PrefetcherPipe accessor naming "
        "exactly the relayed pipe set");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayDFBBlockedProducerFails) {
    auto spec = MakeFullPipeSpec();
    KernelNamed(spec, "receiver").dfb_bindings = {BlockedProducerOf(relay_dfb_name, "relay", /*block_size=*/4)};
    EXPECT_SPEC_REJECTED(spec, "relay DFB does not support the BLOCKED access pattern");
}
TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayDFBBlockedConsumerFails) {
    auto spec = MakeFullPipeSpec();
    KernelNamed(spec, "compute").dfb_bindings = {BlockedConsumerOf(relay_dfb_name, "relay", /*block_size=*/4)};
    EXPECT_SPEC_REJECTED(spec, "relay DFB does not support the BLOCKED access pattern");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayProducerNotBindingPipeFails) {
    // The relay's producer must be the pipe's receiver kernel; without the binding it cannot drive
    // the pipe protocol the relay depends on.
    ProgramSpec spec = MakeFullPipeSpec();
    KernelNamed(spec, "receiver").advanced_options.prefetcher_pipe_bindings.clear();
    EXPECT_SPEC_REJECTED(
        spec,
        "Kernel 'receiver' is a PRODUCER of relay DFB 'weights_relay' but has no PrefetcherPipe accessor naming "
        "exactly the relayed pipe set (1 pipe(s), first 'weights')");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayProducerBindsDifferentGroupFails) {
    // The relay's nodes match "weights"' (single) receiver node, but the producer kernel there
    // binds a different pipe ("other", as its SENDER) instead of "weights".
    ProgramSpec spec = MakeFullPipeSpec();
    spec.kernels.erase(spec.kernels.begin());  // drop "sender"
    spec.work_units.erase(spec.work_units.begin());
    spec.advanced_options.prefetcher_pipe_parameters[0].receivers = NodeCoord{0, 1};
    spec.advanced_options.prefetcher_pipe_parameters.push_back(
        MakeOtherPipeParameter());  // its sender is the (0,1) kernel
    KernelNamed(spec, "receiver").advanced_options.prefetcher_pipe_bindings = {BindPipes({other_param_name}, "out")};
    spec.work_units[0].target_nodes = NodeCoord{0, 1};
    EXPECT_SPEC_REJECTED(
        spec,
        "Kernel 'receiver' is a PRODUCER of relay DFB 'weights_relay' but has no PrefetcherPipe accessor naming "
        "exactly the relayed pipe set (1 pipe(s), first 'weights')");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeRelayMismatchedGeometryFails) {
    // Relay names both pipes but the producer binds only "weights": the relay-side geometry
    // check fires (the producer-binding check comes after it).
    ProgramSpec spec = MakeFullPipeSpec();
    PrefetcherPipeParameter other = MakeOtherPipeParameter();
    other.ring_size = pipe_ring_size * 2;
    spec.advanced_options.prefetcher_pipe_parameters.push_back(other);
    spec.dataflow_buffers[0].advanced_options.prefetcher_pipe_relays.push_back(other.unique_id);
    EXPECT_SPEC_REJECTED(spec, "every pipe relayed by one DFB must share ring_size and entry_size");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeRelayOverlappingReceiversFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    PrefetcherPipeParameter other = MakeOtherPipeParameter();
    other.receivers = pipe_receiver_nodes;  // same receivers as "weights"
    spec.advanced_options.prefetcher_pipe_parameters.push_back(other);
    spec.dataflow_buffers[0].advanced_options.prefetcher_pipe_relays.push_back(other.unique_id);
    EXPECT_SPEC_REJECTED(spec, "relayed pipes must have disjoint receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeRelayProducerBindsSubsetFails) {
    // Relay names {weights, other}; the producer's accessor names only {weights} and so the producer
    // also fails tiling (it runs on both receiver sets). The tiling rule fires first.
    ProgramSpec spec = MakeTwoPipeReceiverSpec();
    KernelNamed(spec, "receiver").advanced_options.prefetcher_pipe_bindings[0].pipe_parameter_names = {pipe_param_name};
    EXPECT_SPEC_REJECTED(spec, "Kernel covers 6 node(s): 3 of the 3 receiver node(s), 3 outside the receivers");
}

}  // namespace
}  // namespace tt::tt_metal::experimental
