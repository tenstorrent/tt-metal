// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for the Metal 2.0 Host API: PrefetcherPipeParameter, relay DFBs and
// PrefetcherPipeBinding validation in ProgramSpec.
//
// All tests use a mock device (Quasar and Wormhole) and exercise spec validation
// only; no PrefetcherPipe object exists. Hardware coverage of the bound path lives
// with the PrefetcherPipe device tests.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt_stl/reflection.hpp>
#include "hostdev/remote_dfb_config_layout.h"

#include "test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ScopedSlowDispatchOverride;

// ============================================================================
// Reflection: the new spec types stay hashable (see test_program_spec.cpp)
// ============================================================================

// (Distinct names from test_program_spec.cpp: unity builds put both files in one TU.)
template <typename T>
ttsl::hash::hash_t pipe_spec_hash_one(const T& value) {
    return ttsl::hash::hash_objects_with_default_seed(value);
}
template <typename T>
inline constexpr bool pipe_spec_hashable_v = (static_cast<void>(&pipe_spec_hash_one<T>), true);

static_assert(
    pipe_spec_hashable_v<PrefetcherPipeParameter>, "PrefetcherPipeParameter must be hashable via ttsl reflection");
static_assert(
    pipe_spec_hashable_v<PrefetcherPipeBinding>, "PrefetcherPipeBinding must be hashable via ttsl reflection");
static_assert(pipe_spec_hashable_v<ProgramSpec>, "ProgramSpec must stay hashable with prefetcher_pipe_parameters");

// ============================================================================
// Fixtures
// ============================================================================

class PrefetcherPipeSpecTestQuasar : public ::testing::Test {
protected:
    void SetUp() override {
        slow_dispatch_override_.emplace();
        experimental::configure_mock_mode(tt::ARCH::QUASAR, 1);
        mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    }
    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
        experimental::disable_mock_mode();
        slow_dispatch_override_.reset();
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    std::optional<ScopedSlowDispatchOverride> slow_dispatch_override_;
};

class PrefetcherPipeSpecTestGen1 : public ::testing::Test {
protected:
    void SetUp() override {
        slow_dispatch_override_.emplace();
        experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1);
        mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    }
    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
        experimental::disable_mock_mode();
        slow_dispatch_override_.reset();
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    std::optional<ScopedSlowDispatchOverride> slow_dispatch_override_;
};

// ============================================================================
// Spec builders
// ============================================================================

// Geometry shared by every test: one sender at (0,0), three receivers (0,1)..(0,3).
const NodeCoord pipe_sender_node{0, 0};
const NodeRange pipe_receiver_nodes{NodeCoord{0, 1}, NodeCoord{0, 3}};
constexpr uint32_t pipe_entry_size = 2048;
constexpr uint32_t pipe_num_entries = 4;
constexpr uint32_t pipe_ring_size = pipe_entry_size * pipe_num_entries;

const PrefetcherPipeParamName pipe_param_name{"weights"};
const DFBSpecName relay_dfb_name{"weights_relay"};

PrefetcherPipeParameter MakePipeParameter() {
    return PrefetcherPipeParameter{
        .unique_id = pipe_param_name,
        .sender = pipe_sender_node,
        .receivers = pipe_receiver_nodes,
        .ring_size = pipe_ring_size,
        .entry_size = pipe_entry_size,
    };
}

PrefetcherPipeBinding BindPipe(std::string accessor = "weights") {
    return PrefetcherPipeBinding{.pipe_parameter_name = pipe_param_name, .accessor_name = std::move(accessor)};
}

// Sender DM on the sender node; receiver DM + compute on the receivers, joined by a relay DFB.
// `receiver_threads` is the receiver kernel's num_threads (the pipe's credit lane count).
ProgramSpec MakeFullPipeSpec(uint32_t receiver_threads = 1) {
    ProgramSpec spec;
    spec.name = "pipe_spec";

    auto sender = MakeMinimalGen2DMKernel("sender");
    sender.prefetcher_pipe_bindings.push_back(BindPipe());

    auto receiver = MakeMinimalGen2DMKernel("receiver", receiver_threads);
    receiver.prefetcher_pipe_bindings.push_back(BindPipe());
    receiver.dfb_bindings.push_back(ProducerOf(relay_dfb_name, "relay"));

    auto compute = MakeMinimalGen2ComputeKernel("compute");
    compute.dfb_bindings.push_back(ConsumerOf(relay_dfb_name, "relay"));

    auto relay = MakeMinimalDFB(relay_dfb_name.get(), pipe_entry_size, pipe_num_entries);
    relay.data_format_metadata = tt::DataFormat::Float16_b;
    relay.prefetcher_pipe_relays = {pipe_param_name};

    spec.kernels = {sender, receiver, compute};
    spec.dataflow_buffers = {relay};
    spec.prefetcher_pipe_parameters = {MakePipeParameter()};
    spec.work_units = {
        MakeMinimalWorkUnit("sender_wu", pipe_sender_node, {"sender"}),
        MakeMinimalWorkUnit("receiver_wu", pipe_receiver_nodes, {"receiver", "compute"}),
    };
    return spec;
}

// Sender-only program: the consumer side lives in another Program.
ProgramSpec MakeSenderOnlySpec() {
    ProgramSpec spec;
    spec.name = "sender_only";
    auto sender = MakeMinimalGen2DMKernel("sender");
    sender.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels = {sender};
    spec.prefetcher_pipe_parameters = {MakePipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit("sender_wu", pipe_sender_node, {"sender"})};
    return spec;
}

KernelSpec& KernelNamed(ProgramSpec& spec, const std::string& name) {
    for (auto& kernel : spec.kernels) {
        if (kernel.unique_id.get() == name) {
            return kernel;
        }
    }
    throw std::runtime_error("no kernel " + name);
}

// Until MakeProgramFromSpec binds pipes, a spec that passes validation is rejected with this
// message (and nothing earlier). Validation failures surface as their own messages first.
constexpr const char* not_yet_bound_msg = "PrefetcherPipe binding through MakeProgramFromSpec is not yet implemented";

#define EXPECT_SPEC_REJECTED(spec, substr)                   \
    EXPECT_THAT(                                             \
        [&] { MakeProgramFromSpec(*mesh_device_, (spec)); }, \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(substr)))

#define EXPECT_SPEC_VALID(spec) EXPECT_SPEC_REJECTED(spec, not_yet_bound_msg)

// ============================================================================
// Accepted geometries
// ============================================================================

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

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayWithoutReceiverBindingPassesValidation) {
    // A relay alone is a legal use of the parameter (the producer kernel need not bind the pipe).
    ProgramSpec spec = MakeFullPipeSpec();
    KernelNamed(spec, "receiver").prefetcher_pipe_bindings.clear();
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiThreadedReceiverDividingRingPasses) {
    // 4 entries, 2 lanes: OK.
    ProgramSpec spec = MakeFullPipeSpec(/*receiver_threads=*/2);
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EntrySizeNotDividingRingPassesForSingleLane) {
    // P == 1 tolerates a trailing gap in the ring (the device checkpoints the wrap).
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.prefetcher_pipe_parameters[0].ring_size = pipe_entry_size * 3 + 64;
    EXPECT_SPEC_VALID(spec);
}

// ============================================================================
// Structural rejections (CollectSpecData)
// ============================================================================

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_DuplicateParameterNameFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.prefetcher_pipe_parameters.push_back(MakePipeParameter());
    EXPECT_SPEC_REJECTED(spec, "Duplicate PrefetcherPipeParameter name 'weights'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_BindingUnknownParameterFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").prefetcher_pipe_bindings[0].pipe_parameter_name = PrefetcherPipeParamName{"nope"};
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' references unknown PrefetcherPipeParameter 'nope'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayUnknownParameterFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].prefetcher_pipe_relays = {PrefetcherPipeParamName{"nope"}};
    EXPECT_SPEC_REJECTED(spec, "DFB 'weights_relay' relays unknown PrefetcherPipeParameter 'nope'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayListsSamePipeTwiceFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].prefetcher_pipe_relays = {pipe_param_name, pipe_param_name};
    EXPECT_SPEC_REJECTED(spec, "lists PrefetcherPipeParameter 'weights' more than once");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_UnusedParameterFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").prefetcher_pipe_bindings.clear();
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' is defined but not bound by any kernel or relay DFB");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_DuplicateAccessorNameFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.prefetcher_pipe_parameters.push_back(MakePipeParameter());
    spec.prefetcher_pipe_parameters[1].unique_id = PrefetcherPipeParamName{"other"};
    auto& sender = KernelNamed(spec, "sender");
    sender.prefetcher_pipe_bindings.push_back(
        PrefetcherPipeBinding{.pipe_parameter_name = PrefetcherPipeParamName{"other"}, .accessor_name = "weights"});
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' has duplicate PrefetcherPipe accessor_name 'weights'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_InvalidAccessorNameFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").prefetcher_pipe_bindings[0].accessor_name = "1weights";
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipe accessor_name '1weights' must be a valid C++ identifier");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SamePipeBoundTwiceInOneKernelFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").prefetcher_pipe_bindings.push_back(BindPipe("weights_again"));
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' binds PrefetcherPipeParameter 'weights' more than once");
}

// ============================================================================
// Geometry rejections (ValidateProgramSpec)
// ============================================================================

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SenderAmongReceiversFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.prefetcher_pipe_parameters[0].receivers = NodeRange{NodeCoord{0, 0}, NodeCoord{0, 3}};
    EXPECT_SPEC_REJECTED(spec, "lists its sender node (0,0) among its receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EmptyReceiversFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.prefetcher_pipe_parameters[0].receivers = NodeRangeSet{};
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' has no receiver nodes");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ZeroRingSizeFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.prefetcher_pipe_parameters[0].ring_size = 0;
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' has ring_size = 0");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ZeroEntrySizeFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.prefetcher_pipe_parameters[0].entry_size = 0;
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' has entry_size = 0");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_UnalignedEntrySizeFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.prefetcher_pipe_parameters[0].entry_size = 2048 + 4;
    EXPECT_SPEC_REJECTED(spec, "entry_size 2052 must be a multiple of the L1 alignment");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EntrySizeLargerThanRingFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.prefetcher_pipe_parameters[0].entry_size = pipe_ring_size * 2;
    EXPECT_SPEC_REJECTED(spec, "exceeds ring_size");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_OutOfBoundsReceiverFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.prefetcher_pipe_parameters[0].receivers = NodeCoord{1000, 1000};
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' targets node (1000,1000), which is out of bounds");
}

// ============================================================================
// Binding role rejections
// ============================================================================

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ComputeKernelBindingPipeFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    KernelNamed(spec, "compute").prefetcher_pipe_bindings.push_back(BindPipe());
    EXPECT_SPEC_REJECTED(spec, "Kernel 'compute' binds PrefetcherPipeParameter 'weights' but is a compute kernel");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_BindingKernelOutsidePipeNodesFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.work_units[0].target_nodes = NodeCoord{2, 2};  // neither sender nor receiver
    EXPECT_SPEC_REJECTED(spec, "place it on nodes outside the pipe's sender + receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_BindingKernelCoversPartialReceiversFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.work_units[1].target_nodes = NodeRange{NodeCoord{0, 1}, NodeCoord{0, 2}};  // 2 of 3 receivers
    EXPECT_SPEC_REJECTED(spec, "covers only 2 of its 3 receiver nodes");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_TwoKernelsBindingOnSenderFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    auto second = MakeMinimalGen2DMKernel("sender2");
    second.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels.push_back(second);
    spec.work_units[0].kernels.push_back(KernelSpecName{"sender2"});
    EXPECT_SPEC_REJECTED(spec, "both bind PrefetcherPipeParameter 'weights' on its sender node (0,0)");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_TwoKernelsBindingOnReceiversFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    auto second = MakeMinimalGen2DMKernel("receiver2");
    second.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels.push_back(second);
    spec.work_units[1].kernels.push_back(KernelSpecName{"receiver2"});
    EXPECT_SPEC_REJECTED(spec, "both bind PrefetcherPipeParameter 'weights' on its receiver nodes");
}

// ============================================================================
// Credit lane (num_threads) rejections
// ============================================================================

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_LanesNotDividingEntriesFails) {
    // 4 entries, 3 lanes.
    ProgramSpec spec = MakeFullPipeSpec(/*receiver_threads=*/3);
    EXPECT_SPEC_REJECTED(spec, "ring holds 4 entries of 2048 bytes, which is not a multiple of 3 credit lanes");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_LanesExceedCapacityFails) {
    static_assert(PREFETCHER_PIPE_MAX_CREDIT_LANES == 4);
    ProgramSpec spec = MakeSenderOnlySpec();
    auto receiver = MakeMinimalGen2DMKernel("receiver", 5);
    receiver.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels.push_back(receiver);
    spec.work_units.push_back(MakeMinimalWorkUnit("receiver_wu", pipe_receiver_nodes, {"receiver"}));
    spec.prefetcher_pipe_parameters[0].ring_size = pipe_entry_size * 10;
    EXPECT_SPEC_REJECTED(spec, "has 5 threads, but a pipe supports at most 4 credit lanes on this architecture");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiLaneEntrySizeNotDividingRingFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    // Receiver-side kernel with 2 threads in a second program-half; ring not a multiple of entry.
    auto receiver = MakeMinimalGen2DMKernel("receiver", 2);
    receiver.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels.push_back(receiver);
    spec.work_units.push_back(MakeMinimalWorkUnit("receiver_wu", pipe_receiver_nodes, {"receiver"}));
    spec.prefetcher_pipe_parameters[0].ring_size = pipe_entry_size * 4 + 64;
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

// ============================================================================
// Relay DFB rejections
// ============================================================================

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayWithBorrowedFromFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.tensor_parameters = {test_helpers::MakeMinimalTensorParameter("t", BufferType::L1)};
    spec.dataflow_buffers[0].borrowed_from = TensorParamName{"t"};
    EXPECT_SPEC_REJECTED(spec, "sets both prefetcher_pipe_relays and borrowed_from");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayEntrySizeMismatchFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].entry_size = pipe_entry_size / 2;
    spec.dataflow_buffers[0].num_entries = pipe_num_entries * 2;
    EXPECT_SPEC_REJECTED(
        spec, "entry_size 1024 differs from relayed PrefetcherPipeParameter 'weights' entry_size 2048");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayNotCoveringRingFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].num_entries = pipe_num_entries - 1;
    EXPECT_SPEC_REJECTED(spec, "must exactly cover relayed PrefetcherPipeParameter 'weights' ring_size");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayNodesNotReceiversFails) {
    // Relay DFB (and its kernels) on 2 of the 3 receivers, without binding the pipe directly.
    ProgramSpec spec = MakeFullPipeSpec();
    KernelNamed(spec, "receiver").prefetcher_pipe_bindings.clear();
    spec.work_units[1].target_nodes = NodeRange{NodeCoord{0, 1}, NodeCoord{0, 2}};
    EXPECT_SPEC_REJECTED(spec, "receiver nodes do not match the DFB's node set");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayComputeProducerFails) {
    // Swap roles: compute produces, DM consumes.
    ProgramSpec spec = MakeFullPipeSpec();
    KernelNamed(spec, "receiver").dfb_bindings = {ConsumerOf(relay_dfb_name, "relay")};
    KernelNamed(spec, "compute").dfb_bindings = {ProducerOf(relay_dfb_name, "relay")};
    EXPECT_SPEC_REJECTED(spec, "is a PRODUCER of relay DFB 'weights_relay' but is a compute kernel");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeRelayMismatchedGeometryFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    PrefetcherPipeParameter other = MakePipeParameter();
    other.unique_id = PrefetcherPipeParamName{"other"};
    other.sender = NodeCoord{1, 0};
    other.receivers = NodeRange{NodeCoord{1, 1}, NodeCoord{1, 3}};
    other.ring_size = pipe_ring_size * 2;
    spec.prefetcher_pipe_parameters.push_back(other);
    spec.dataflow_buffers[0].prefetcher_pipe_relays.push_back(other.unique_id);
    EXPECT_SPEC_REJECTED(spec, "every pipe relayed by one DFB must share ring_size and entry_size");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeRelayOverlappingReceiversFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    PrefetcherPipeParameter other = MakePipeParameter();
    other.unique_id = PrefetcherPipeParamName{"other"};
    other.sender = NodeCoord{1, 0};  // same receivers as "weights"
    spec.prefetcher_pipe_parameters.push_back(other);
    spec.dataflow_buffers[0].prefetcher_pipe_relays.push_back(other.unique_id);
    EXPECT_SPEC_REJECTED(spec, "relayed pipes must have disjoint receivers");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeRelayTilingReceiversPasses) {
    // Two pipes whose receivers tile the relay's nodes: one relay DFB over (0,1)..(0,3) + (1,1)..(1,3).
    ProgramSpec spec = MakeFullPipeSpec();
    KernelNamed(spec, "receiver").prefetcher_pipe_bindings.clear();  // per-pipe binding is single-pipe
    PrefetcherPipeParameter other = MakePipeParameter();
    other.unique_id = PrefetcherPipeParamName{"other"};
    other.sender = NodeCoord{1, 0};
    other.receivers = NodeRange{NodeCoord{1, 1}, NodeCoord{1, 3}};
    spec.prefetcher_pipe_parameters.push_back(other);
    spec.dataflow_buffers[0].prefetcher_pipe_relays.push_back(other.unique_id);
    spec.work_units[1].target_nodes =
        NodeRangeSet(std::vector<NodeRange>{pipe_receiver_nodes, NodeRange{{1, 1}, {1, 3}}});
    EXPECT_SPEC_VALID(spec);
}

// ============================================================================
// Gen1 (Wormhole mock)
// ============================================================================

TEST_F(PrefetcherPipeSpecTestGen1, CPU_SingleLanePipePassesValidation) {
    ProgramSpec spec;
    spec.name = "gen1_pipe";

    auto sender = MakeMinimalGen1DMKernel("sender", DataMovementProcessor::RISCV_0);
    sender.prefetcher_pipe_bindings.push_back(BindPipe());

    auto receiver = MakeMinimalGen1DMKernel("receiver", DataMovementProcessor::RISCV_1);
    receiver.prefetcher_pipe_bindings.push_back(BindPipe());
    receiver.dfb_bindings.push_back(ProducerOf(relay_dfb_name, "relay"));

    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.dfb_bindings.push_back(ConsumerOf(relay_dfb_name, "relay"));

    auto relay = MakeMinimalDFB(relay_dfb_name.get(), pipe_entry_size, pipe_num_entries);
    relay.data_format_metadata = tt::DataFormat::Float16_b;
    relay.prefetcher_pipe_relays = {pipe_param_name};

    spec.kernels = {sender, receiver, compute};
    spec.dataflow_buffers = {relay};
    spec.prefetcher_pipe_parameters = {MakePipeParameter()};
    spec.work_units = {
        MakeMinimalWorkUnit("sender_wu", pipe_sender_node, {"sender"}),
        MakeMinimalWorkUnit("receiver_wu", pipe_receiver_nodes, {"receiver", "compute"}),
    };
    EXPECT_SPEC_VALID(spec);
}

TEST_F(PrefetcherPipeSpecTestGen1, CPU_TwoDMKernelsOnSenderFails) {
    // WH/BH: only one DM may own the sender credits on a node.
    ProgramSpec spec;
    spec.name = "gen1_dual_sender";
    auto brisc = MakeMinimalGen1DMKernel("brisc", DataMovementProcessor::RISCV_0);
    brisc.prefetcher_pipe_bindings.push_back(BindPipe());
    auto ncrisc = MakeMinimalGen1DMKernel("ncrisc", DataMovementProcessor::RISCV_1);
    ncrisc.prefetcher_pipe_bindings.push_back(BindPipe());
    spec.kernels = {brisc, ncrisc};
    spec.prefetcher_pipe_parameters = {MakePipeParameter()};
    spec.work_units = {MakeMinimalWorkUnit("sender_wu", pipe_sender_node, {"brisc", "ncrisc"})};
    EXPECT_SPEC_REJECTED(spec, "both bind PrefetcherPipeParameter 'weights' on its sender node (0,0)");
}

}  // namespace
}  // namespace tt::tt_metal::experimental
