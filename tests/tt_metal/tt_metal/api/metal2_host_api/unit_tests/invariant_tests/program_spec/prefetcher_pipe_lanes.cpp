// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariants on PrefetcherPipe credit lanes (program_spec.hpp /
// prefetcher_pipe_parameter.hpp): the receiver kernel's num_threads against the pipe geometry.

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "hostdev/remote_dfb_config_layout.h"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/prefetcher_pipe_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindPipe;
using test_helpers::KernelNamed;
using test_helpers::MakeFullPipeSpec;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakePipeParameter;
using test_helpers::MakeSenderOnlySpec;
using test_helpers::MakeTwoPipeReceiverSpec;
using test_helpers::pipe_entry_size;
using test_helpers::pipe_num_entries;
using test_helpers::pipe_param_name;
using test_helpers::pipe_receiver_nodes;
using test_helpers::pipe_sender_node;
using test_helpers::PrefetcherPipeSpecTestGen1;
using test_helpers::PrefetcherPipeSpecTestQuasar;
using test_helpers::relay_dfb_name;

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiThreadedReceiverDividingRingPasses) {
    // 4 entries, 2 lanes: OK.
    ProgramSpec spec = MakeFullPipeSpec(/*receiver_threads=*/2);
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeReceiverMultiLanePasses) {
    // The group's receiver kernel has 2 threads: both pipes get P = 2 (4 entries each).
    ProgramSpec spec = MakeTwoPipeReceiverSpec(/*receiver_threads=*/2);
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EntrySizeNotDividingRingPassesForSingleLane) {
    // P == 1 tolerates a trailing gap in the ring (the device checkpoints the wrap).
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].ring_size = pipe_entry_size * 3 + 64;
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestGen1, CPU_SingleLanePipePassesValidation) {
    ProgramSpec spec;
    spec.name = "gen1_pipe";

    auto sender = MakeMinimalGen1DMKernel("sender", DataMovementProcessor::RISCV_0);
    sender.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());

    auto receiver = MakeMinimalGen1DMKernel("receiver", DataMovementProcessor::RISCV_1);
    receiver.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    receiver.dfb_bindings.push_back(ProducerOf(relay_dfb_name, "relay"));

    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.dfb_bindings.push_back(ConsumerOf(relay_dfb_name, "relay"));

    auto relay = MakeMinimalDFB(relay_dfb_name.get(), pipe_entry_size, pipe_num_entries);
    relay.data_format_metadata = tt::DataFormat::Float16_b;
    relay.advanced_options.prefetcher_pipe_relays = {pipe_param_name};

    spec.kernels = {sender, receiver, compute};
    spec.dataflow_buffers = {relay};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter()};
    spec.work_units = {
        MakeMinimalWorkUnit("sender_wu", pipe_sender_node, {"sender"}),
        MakeMinimalWorkUnit("receiver_wu", pipe_receiver_nodes, {"receiver", "compute"}),
    };
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_AccessorGroupLanesApplyToEveryPipeFails) {
    // 3 lanes divide neither pipe's 4 entries; the per-pipe lane check still runs for grouped pipes.
    ProgramSpec spec = MakeTwoPipeReceiverSpec(/*receiver_threads=*/3);
    EXPECT_SPEC_REJECTED(spec, "ring holds 4 entries of 2048 bytes, which is not a multiple of 3 credit lanes");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_LanesNotDividingEntriesFails) {
    // 4 entries, 3 lanes.
    ProgramSpec spec = MakeFullPipeSpec(/*receiver_threads=*/3);
    EXPECT_SPEC_REJECTED(spec, "ring holds 4 entries of 2048 bytes, which is not a multiple of 3 credit lanes");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_LanesExceedCapacityFails) {
    static_assert(PREFETCHER_PIPE_MAX_CREDIT_LANES == 4);
    ProgramSpec spec = MakeSenderOnlySpec();
    auto receiver = MakeMinimalGen2DMKernel("receiver", 5);
    receiver.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels.push_back(receiver);
    spec.work_units.push_back(MakeMinimalWorkUnit("receiver_wu", pipe_receiver_nodes, {"receiver"}));
    spec.advanced_options.prefetcher_pipe_parameters[0].ring_size = pipe_entry_size * 10;
    EXPECT_SPEC_REJECTED(spec, "has 5 threads, but a pipe supports at most 4 credit lanes on this architecture");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiLaneEntrySizeNotDividingRingFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    // Receiver-side kernel with 2 threads in a second program-half; ring not a multiple of entry.
    auto receiver = MakeMinimalGen2DMKernel("receiver", 2);
    receiver.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels.push_back(receiver);
    spec.work_units.push_back(MakeMinimalWorkUnit("receiver_wu", pipe_receiver_nodes, {"receiver"}));
    spec.advanced_options.prefetcher_pipe_parameters[0].ring_size = pipe_entry_size * 4 + 64;
    EXPECT_SPEC_REJECTED(spec, "with 2 credit lanes requires entry_size 2048 to divide ring_size");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ReceiverAndRelayProducerThreadMismatchFails) {
    // Receiver binds the pipe with 2 threads; a separate 1-thread DM kernel produces the relay.
    ProgramSpec spec = MakeFullPipeSpec(/*receiver_threads=*/2);
    KernelNamed(spec, "receiver").dfb_bindings.clear();
    auto forwarder = MakeMinimalGen2DMKernel("forwarder", 1);
    forwarder.dfb_bindings.push_back(ProducerOf(relay_dfb_name, "relay"));
    spec.kernels.push_back(forwarder);
    spec.work_units[1].kernels.push_back(KernelSpecName{"forwarder"});
    EXPECT_SPEC_REJECTED(
        spec, "receiver-side kernels disagree on thread count: 'receiver' has 2 threads, 'forwarder' has 1");
}

}  // namespace
}  // namespace tt::tt_metal::experimental
