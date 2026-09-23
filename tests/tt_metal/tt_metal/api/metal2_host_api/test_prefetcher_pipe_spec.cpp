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
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt_stl/reflection.hpp>
#include "hostdev/remote_dfb_config_layout.h"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

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

class PrefetcherPipeSpecTestQuasar : public ::testing::Test, protected PipeSpaceOwner {
protected:
    void SetUp() override {
        // Mock mode clears the simulator target, and the Quasar descriptor has no dispatch
        // cores. That combination asserts during device init. Skip only the LLK-assert nightly,
        // which is where this shows up; other simulator runs still execute these tests.
        if (const char* llk_asserts = std::getenv("TT_METAL_LLK_ASSERTS");
            llk_asserts != nullptr && std::string(llk_asserts) == "1") {
            GTEST_SKIP() << "Quasar mock device init asserts on empty dispatch cores when LLK asserts are enabled";
        }
        slow_dispatch_override_.emplace();
        experimental::configure_mock_mode(tt::ARCH::QUASAR, 1);
        mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    }
    void TearDown() override {
        ReleasePipeSpaces();
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

class PrefetcherPipeSpecTestGen1 : public ::testing::Test, protected PipeSpaceOwner {
protected:
    void SetUp() override {
        slow_dispatch_override_.emplace();
        experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1);
        mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    }
    void TearDown() override {
        ReleasePipeSpaces();
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

// Geometry shared by every test: one sender at (0,0), three receivers (0,1)..(0,3). The spec
// names only the receivers; the sender node is where the sender kernel's work unit runs, and
// where the test carves the pipe.
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
        .receivers = pipe_receiver_nodes,
        .ring_size = pipe_ring_size,
        .entry_size = pipe_entry_size,
    };
}

PrefetcherPipeBinding BindPipes(std::vector<PrefetcherPipeParamName> pipes, std::string accessor = "weights") {
    return PrefetcherPipeBinding{.pipe_parameter_names = std::move(pipes), .accessor_name = std::move(accessor)};
}

PrefetcherPipeBinding BindPipe(std::string accessor = "weights") {
    return BindPipes({pipe_param_name}, std::move(accessor));
}

// A second pipe on the next column: sender (1,0), receivers (1,1)..(1,3). Disjoint from "weights",
// so the two can share an accessor (and a relay) whose nodes they tile.
const PrefetcherPipeParamName other_param_name{"other"};
const NodeCoord other_sender_node{1, 0};
const NodeRange other_receiver_nodes{NodeCoord{1, 1}, NodeCoord{1, 3}};

PrefetcherPipeParameter MakeOtherPipeParameter() {
    return PrefetcherPipeParameter{
        .unique_id = other_param_name,
        .receivers = other_receiver_nodes,
        .ring_size = pipe_ring_size,
        .entry_size = pipe_entry_size,
    };
}

const NodeRangeSet both_sender_nodes(std::vector<NodeRange>{
    NodeRange{pipe_sender_node, pipe_sender_node}, NodeRange{other_sender_node, other_sender_node}});
const NodeRangeSet both_receiver_nodes(std::vector<NodeRange>{pipe_receiver_nodes, other_receiver_nodes});

// Sender DM on the sender node; receiver DM + compute on the receivers, joined by a relay DFB.
// `receiver_threads` is the receiver kernel's num_threads (the pipe's credit lane count).
ProgramSpec MakeFullPipeSpec(uint32_t receiver_threads = 1) {
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
ProgramSpec MakeSenderOnlySpec() {
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
ProgramSpec MakeTwoPipeReceiverSpec(uint32_t receiver_threads = 1) {
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

KernelSpec& KernelNamed(ProgramSpec& spec, const std::string& name) {
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

#define EXPECT_THROWS_WITH(stmt, substr) \
    EXPECT_THAT([&] { stmt; }, ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(substr)))

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

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiThreadedReceiverDividingRingPasses) {
    // 4 entries, 2 lanes: OK.
    ProgramSpec spec = MakeFullPipeSpec(/*receiver_threads=*/2);
    EXPECT_SPEC_VALID(spec);
}

// ---- One accessor, several pipes ----

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

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MultiPipeReceiverMultiLanePasses) {
    // The group's receiver kernel has 2 threads: both pipes get P = 2 (4 entries each).
    ProgramSpec spec = MakeTwoPipeReceiverSpec(/*receiver_threads=*/2);
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

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EntrySizeNotDividingRingPassesForSingleLane) {
    // P == 1 tolerates a trailing gap in the ring (the device checkpoints the wrap).
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].ring_size = pipe_entry_size * 3 + 64;
    EXPECT_SPEC_VALID(spec);
}

// ============================================================================
// Structural rejections (CollectSpecData)
// ============================================================================

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

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EmptyAccessorGroupFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings[0].pipe_parameter_names.clear();
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' PrefetcherPipe accessor 'weights' names no PrefetcherPipeParameter");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SamePipeTwiceInOneAccessorFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings[0].pipe_parameter_names = {
        pipe_param_name, pipe_param_name};
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' binds PrefetcherPipeParameter 'weights' more than once");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayUnknownParameterFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].advanced_options.prefetcher_pipe_relays = {PrefetcherPipeParamName{"nope"}};
    EXPECT_SPEC_REJECTED(spec, "DFB 'weights_relay' relays unknown PrefetcherPipeParameter 'nope'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayListsSamePipeTwiceFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].advanced_options.prefetcher_pipe_relays = {pipe_param_name, pipe_param_name};
    EXPECT_SPEC_REJECTED(spec, "lists PrefetcherPipeParameter 'weights' more than once");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_UnusedParameterFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings.clear();
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' is defined but not bound by any kernel or relay DFB");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_DuplicateAccessorNameFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters.push_back(MakeOtherPipeParameter());
    KernelNamed(spec, "sender")
        .advanced_options.prefetcher_pipe_bindings.push_back(BindPipes({other_param_name}, "weights"));
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' has duplicate PrefetcherPipe accessor_name 'weights'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_InvalidAccessorNameFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings[0].accessor_name = "1weights";
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipe accessor_name '1weights' must be a valid C++ identifier");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SamePipeBoundTwiceInOneKernelFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings.push_back(BindPipe("weights_again"));
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' binds PrefetcherPipeParameter 'weights' more than once");
}

// ============================================================================
// Geometry rejections (ValidateProgramSpec)
// ============================================================================

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EmptyReceiversFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].receivers = NodeRangeSet{};
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' has no receiver nodes");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ZeroRingSizeFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].ring_size = 0;
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' has ring_size = 0");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ZeroEntrySizeFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].entry_size = 0;
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' has entry_size = 0");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_UnalignedEntrySizeFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].entry_size = 2048 + 4;
    EXPECT_SPEC_REJECTED(spec, "entry_size 2052 must be a multiple of the L1 alignment");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EntrySizeLargerThanRingFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].entry_size = pipe_ring_size * 2;
    EXPECT_SPEC_REJECTED(spec, "exceeds ring_size");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_OutOfBoundsReceiverFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters[0].receivers = NodeCoord{1000, 1000};
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipeParameter 'weights' targets node (1000,1000), which is out of bounds");
}

// ============================================================================
// Binding role rejections
// ============================================================================

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ComputeKernelBindingPipeFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    KernelNamed(spec, "compute").advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    EXPECT_SPEC_REJECTED(
        spec, "Kernel 'compute' binds PrefetcherPipeParameter(s) (accessor 'weights') but is a compute kernel");
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

// ---- Accessor group (several pipes under one accessor) rejections ----

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

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_AccessorGroupLanesApplyToEveryPipeFails) {
    // 3 lanes divide neither pipe's 4 entries; the per-pipe lane check still runs for grouped pipes.
    ProgramSpec spec = MakeTwoPipeReceiverSpec(/*receiver_threads=*/3);
    EXPECT_SPEC_REJECTED(spec, "ring holds 4 entries of 2048 bytes, which is not a multiple of 3 credit lanes");
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

// ============================================================================
// Slot reservation (MakeProgramFromSpec) and pipe binding (SetProgramRunArgs)
// ============================================================================

PrefetcherPipe PipeSpaceOwner::MakePipeFor(
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

PrefetcherPipe PipeSpaceOwner::MakeWeightsPipe(distributed::MeshDevice& device) {
    return MakePipeFor(device, MakePipeParameter(), pipe_sender_node);
}
PrefetcherPipe PipeSpaceOwner::MakeOtherPipe(distributed::MeshDevice& device) {
    return MakePipeFor(device, MakeOtherPipeParameter(), other_sender_node);
}

// The participant record for `prefetcher_pipe_id` on `core`, or nullptr when the core has no slot.
const detail::ProgramImpl::PrefetcherPipeParticipant* ParticipantOn(
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

ProgramRunArgs PipeArgs(std::vector<std::pair<PrefetcherPipeParamName, PrefetcherPipe*>> pipes) {
    ProgramRunArgs params;
    for (auto& [name, pipe] : pipes) {
        params.advanced_options.prefetcher_pipe_args.insert({name, PrefetcherPipeArgument{*pipe}});
    }
    return params;
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MakeProgramReservesOneSlotPerAccessorGroup) {
    // Sender accessor + receiver accessor -> two slots. No pipe is bound; the slot carries the
    // geometry the kernels compile against.
    Program program = MakeProgramFromSpec(*mesh_device_, MakeFullPipeSpec(/*receiver_threads=*/2));
    const auto& impl = program.impl();
    ASSERT_EQ(impl.num_prefetcher_pipe_slots(), 2u);

    const auto* binding = impl.get_prefetcher_pipe_parameter(pipe_param_name.get());
    ASSERT_NE(binding, nullptr);
    EXPECT_EQ(binding->bound_pipe, nullptr);
    EXPECT_EQ(binding->receivers.num_cores(), 3u);
    EXPECT_EQ(binding->ring_size, pipe_ring_size);
    ASSERT_EQ(binding->slots.size(), 2u);

    uint32_t sender_slots = 0;
    uint32_t receiver_slots = 0;
    for (const auto& slot_cores : binding->slots) {
        const auto& slot = impl.get_prefetcher_pipe_slot(slot_cores.prefetcher_pipe_id);
        EXPECT_EQ(slot.ring_size, pipe_ring_size);
        EXPECT_EQ(slot.entry_size, pipe_entry_size);
        if (slot.receiver_cores.num_cores() == 0) {
            ++sender_slots;
            EXPECT_TRUE(slot_cores.sender_role);
            EXPECT_TRUE(slot_cores.cores.contains(pipe_sender_node));
            EXPECT_EQ(slot.cores.num_cores(), 1u);
            EXPECT_EQ(slot.num_credit_lanes, 1u);
            EXPECT_FALSE(slot.relay_dfb_host_id.has_value());
        } else {
            ++receiver_slots;
            EXPECT_EQ(slot.cores.num_cores(), 3u);
            EXPECT_EQ(slot.num_credit_lanes, 2u);  // receiver kernel num_threads
            EXPECT_TRUE(slot.relay_dfb_host_id.has_value());
        }
        for (const CoreCoord& core : corerange_to_cores(slot_cores.cores)) {
            const auto* participant = ParticipantOn(program, core, slot_cores.prefetcher_pipe_id);
            ASSERT_NE(participant, nullptr);
            EXPECT_EQ(participant->pipe, nullptr);
            EXPECT_EQ(participant->entry_size, pipe_entry_size);
        }
    }
    EXPECT_EQ(sender_slots, 1u);
    EXPECT_EQ(receiver_slots, 1u);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MakeProgramCreatesRelayDFBWithoutAddress) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeFullPipeSpec());
    const auto& impl = program.impl();
    const uint32_t relay_id = impl.get_dfb_handle(relay_dfb_name.get());
    auto relay = impl.get_dataflow_buffer(relay_id);
    ASSERT_NE(relay, nullptr);
    EXPECT_TRUE(relay->borrows_memory());
    EXPECT_TRUE(relay->config.is_relay);
    EXPECT_EQ(relay->borrowed_addr_, 0u);  // pointed at the ring when the pipe binds
    EXPECT_TRUE(impl.get_prefetcher_pipe_id_for_relay(relay_id).has_value());
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_KernelsGetOnePipeAccessorTokenPerBinding) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeFullPipeSpec());
    const auto& impl = program.impl();
    std::vector<uint8_t> slot_ids;
    for (const char* name : {"sender", "receiver"}) {
        const auto& handles = impl.get_kernel_by_spec_name(name)->prefetcher_pipe_binding_handles();
        ASSERT_EQ(handles.size(), 1u) << name;
        EXPECT_EQ(handles[0].accessor_name, "weights") << name;
        slot_ids.push_back(handles[0].prefetcher_pipe_id);
    }
    EXPECT_NE(slot_ids[0], slot_ids[1]);  // sender and receiver accessors are different slots
    EXPECT_TRUE(impl.get_kernel_by_spec_name("compute")->prefetcher_pipe_binding_handles().empty());
}

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

// ============================================================================
// Gen1 (Wormhole mock)
// ============================================================================

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
