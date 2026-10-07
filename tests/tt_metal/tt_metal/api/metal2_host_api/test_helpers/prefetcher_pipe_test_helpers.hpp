// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared PrefetcherPipe test geometry, spec builders and fixtures for the Metal 2.0 Host API
// unit tests. All tests use a mock device (Quasar and Wormhole).

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/program/program_impl.hpp"

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental::test_helpers {

// Owns the PrefetcherPipeSpaces the tests carve pipes from. A space must outlive its pipes (a
// pipe points at its space without owning it), so the carve helpers park each space here and
// the fixtures release them in TearDown, after the test body's pipes are gone and before the
// mock device closes.
class PipeSpaceOwner {
protected:
    // One pipe carved from a space sized exactly for it, with `sender` as its sender node (the
    // spec does not carry one).
    PrefetcherPipe MakePipeFor(
        distributed::MeshDevice& device, const PrefetcherPipeParameter& param, const CoreCoord& sender);
    // The test geometry's pipes carved where their sender kernels run.
    PrefetcherPipe MakeWeightsPipe(distributed::MeshDevice& device);
    PrefetcherPipe MakeOtherPipe(distributed::MeshDevice& device);
    void ReleasePipeSpaces() { spaces_.clear(); }

private:
    std::vector<PrefetcherPipeSpace> spaces_;
};

class PrefetcherPipeSpecTestQuasar : public MockMeshDeviceFixture<tt::ARCH::QUASAR>, protected PipeSpaceOwner {
protected:
    void TearDown() override {
        ReleasePipeSpaces();
        MockMeshDeviceFixture::TearDown();
    }
};

class PrefetcherPipeSpecTestGen1 : public MockMeshDeviceFixture<tt::ARCH::WORMHOLE_B0>, protected PipeSpaceOwner {
protected:
    void TearDown() override {
        ReleasePipeSpaces();
        MockMeshDeviceFixture::TearDown();
    }
};

// Geometry shared by every test: one sender at (0,0), three receivers (0,1)..(0,3). The spec
// names only the receivers; the sender node is where the sender kernel's work unit runs, and
// where the test carves the pipe.
inline const NodeCoord pipe_sender_node{0, 0};
inline const NodeRange pipe_receiver_nodes{NodeCoord{0, 1}, NodeCoord{0, 3}};
inline constexpr uint32_t pipe_entry_size = 2048;
inline constexpr uint32_t pipe_num_entries = 4;
inline constexpr uint32_t pipe_ring_size = pipe_entry_size * pipe_num_entries;

inline const PrefetcherPipeParamName pipe_param_name{"weights"};
inline const DFBSpecName relay_dfb_name{"weights_relay"};

inline PrefetcherPipeParameter MakePipeParameter() {
    return PrefetcherPipeParameter{
        .unique_id = pipe_param_name,
        .receivers = pipe_receiver_nodes,
        .ring_size = pipe_ring_size,
        .entry_size = pipe_entry_size,
    };
}

inline PrefetcherPipeBinding BindPipes(std::vector<PrefetcherPipeParamName> pipes, std::string accessor = "weights") {
    return PrefetcherPipeBinding{.pipe_parameter_names = std::move(pipes), .accessor_name = std::move(accessor)};
}

inline PrefetcherPipeBinding BindPipe(std::string accessor = "weights") {
    return BindPipes({pipe_param_name}, std::move(accessor));
}

// A second pipe on the next column: sender (1,0), receivers (1,1)..(1,3). Disjoint from "weights",
// so the two can share an accessor (and a relay) whose nodes they tile.
inline const PrefetcherPipeParamName other_param_name{"other"};
inline const NodeCoord other_sender_node{1, 0};
inline const NodeRange other_receiver_nodes{NodeCoord{1, 1}, NodeCoord{1, 3}};

inline PrefetcherPipeParameter MakeOtherPipeParameter() {
    return PrefetcherPipeParameter{
        .unique_id = other_param_name,
        .receivers = other_receiver_nodes,
        .ring_size = pipe_ring_size,
        .entry_size = pipe_entry_size,
    };
}

inline const NodeRangeSet both_sender_nodes(std::vector<NodeRange>{
    NodeRange{pipe_sender_node, pipe_sender_node}, NodeRange{other_sender_node, other_sender_node}});
inline const NodeRangeSet both_receiver_nodes(std::vector<NodeRange>{pipe_receiver_nodes, other_receiver_nodes});

// Sender DM on the sender node; receiver DM + compute on the receivers, joined by a relay DFB.
// `receiver_threads` is the receiver kernel's num_threads (the pipe's credit lane count).
inline ProgramSpec MakeFullPipeSpec(uint32_t receiver_threads = 1) {
    ProgramSpec spec;
    spec.name = "pipe_spec";

    auto sender = MakeMinimalGen2DMKernel("sender");
    sender.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());

    auto receiver = MakeMinimalGen2DMKernel("receiver", receiver_threads);
    receiver.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    receiver.dfb_bindings.push_back(ProducerOf(relay_dfb_name, "relay"));

    auto compute = MakeMinimalGen2ComputeKernel("compute");
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
    return spec;
}

// Sender-only program: the consumer side lives in another Program.
inline ProgramSpec MakeSenderOnlySpec() {
    ProgramSpec spec;
    spec.name = "sender_only";
    auto sender = MakeMinimalGen2DMKernel("sender");
    sender.advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels = {sender};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit("sender_wu", pipe_sender_node, {"sender"})};
    return spec;
}

// Receiver side of two pipes through one accessor: a receiver DM kernel on both receiver sets
// binding {weights, other}, relaying to compute through one DFB that names both pipes. The senders
// live in another Program.
inline ProgramSpec MakeTwoPipeReceiverSpec(uint32_t receiver_threads = 1) {
    ProgramSpec spec;
    spec.name = "two_pipe_receiver";

    auto receiver = MakeMinimalGen2DMKernel("receiver", receiver_threads);
    receiver.advanced_options.prefetcher_pipe_bindings.push_back(BindPipes({pipe_param_name, other_param_name}, "in"));
    receiver.dfb_bindings.push_back(ProducerOf(relay_dfb_name, "relay"));

    auto compute = MakeMinimalGen2ComputeKernel("compute");
    compute.dfb_bindings.push_back(ConsumerOf(relay_dfb_name, "relay"));

    auto relay = MakeMinimalDFB(relay_dfb_name.get(), pipe_entry_size, pipe_num_entries);
    relay.data_format_metadata = tt::DataFormat::Float16_b;
    relay.advanced_options.prefetcher_pipe_relays = {pipe_param_name, other_param_name};

    spec.kernels = {receiver, compute};
    spec.dataflow_buffers = {relay};
    spec.advanced_options.prefetcher_pipe_parameters = {MakePipeParameter(), MakeOtherPipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit("receiver_wu", both_receiver_nodes, {"receiver", "compute"})};
    return spec;
}

inline KernelSpec& KernelNamed(ProgramSpec& spec, const std::string& name) {
    for (auto& kernel : spec.kernels) {
        if (kernel.unique_id.get() == name) {
            return kernel;
        }
    }
    throw std::runtime_error("no kernel " + name);
}

#define EXPECT_SPEC_REJECTED(spec, substr)                   \
    EXPECT_THAT(                                             \
        [&] { MakeProgramFromSpec(*mesh_device_, (spec)); }, \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(substr)))

// A spec that passes validation builds a Program with its pipe slots reserved (no pipe bound yet).
#define EXPECT_SPEC_VALID(spec) EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, (spec)); })

inline PrefetcherPipe PipeSpaceOwner::MakePipeFor(
    distributed::MeshDevice& device, const PrefetcherPipeParameter& param, const CoreCoord& sender) {
    const CoreRangeSet receivers = std::visit(
        [](const auto& nodes) -> CoreRangeSet {
            if constexpr (std::is_same_v<std::decay_t<decltype(nodes)>, CoreRangeSet>) {
                return nodes;
            } else {
                return CoreRangeSet(CoreRange(nodes));
            }
        },
        param.receivers);
    spaces_.push_back(CreatePrefetcherPipeSpace(
        device,
        PrefetcherPipeSpaceConfig{
            .sender_cores = CoreRangeSet(CoreRange(sender)),
            .receiver_domain = receivers,
            .ring_size = param.ring_size,
            .max_receivers_per_pipe = receivers.num_cores(),
        }));
    return spaces_.back().create_pipe(sender, receivers);
}

inline PrefetcherPipe PipeSpaceOwner::MakeWeightsPipe(distributed::MeshDevice& device) {
    return MakePipeFor(device, MakePipeParameter(), pipe_sender_node);
}
inline PrefetcherPipe PipeSpaceOwner::MakeOtherPipe(distributed::MeshDevice& device) {
    return MakePipeFor(device, MakeOtherPipeParameter(), other_sender_node);
}

// The participant record for `prefetcher_pipe_id` on `core`, or nullptr when the core has no slot.
inline const detail::ProgramImpl::PrefetcherPipeParticipant* ParticipantOn(
    const Program& program, const CoreCoord& core, uint8_t prefetcher_pipe_id) {
    const auto& per_core = program.impl().get_per_core_prefetcher_pipes();
    auto it = per_core.find(core);
    if (it == per_core.end()) {
        return nullptr;
    }
    for (const auto& participant : it->second) {
        if (participant.prefetcher_pipe_id == prefetcher_pipe_id) {
            return &participant;
        }
    }
    return nullptr;
}

}  // namespace tt::tt_metal::experimental::test_helpers
