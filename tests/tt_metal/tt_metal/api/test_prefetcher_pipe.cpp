// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe hardware tests on the Metal 2.0 host API.
//
// A PrefetcherPipeSpace reserves persistent L1, pipes are carved from it, and every Program is
// built from a ProgramSpec that declares a PrefetcherPipeParameter with the pipe's geometry.
// Kernels reach the pipe through pipe::<accessor>; relay DFBs alias the ring through
// DFBAdvancedOptions::prefetcher_pipe_relays; the live pipe object arrives through
// AdvancedProgramRunArgs::prefetcher_pipe_args.

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/sub_device.hpp>

#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/dispatch.hpp"
#include "impl/context/metal_context.hpp"
#include "mesh_dispatch_fixture.hpp"
#include "tests/tt_metal/tt_metal/api/cross_node_dfb_test_utils.hpp"
#include "hostdev/remote_dfb_config_layout.h"
#include "hostdev/remote_dfb_constants.h"
#include "tests/tt_metal/tt_metal/api/prefetcher_pipe_test_utils.hpp"

namespace tt::tt_metal {

class PrefetcherPipeFixture : public MeshDispatchFixture {
protected:
    void SetUp() override { MeshDispatchFixture::SetUp(); }
    void TearDown() override {
        // Test-body pipes are gone by now; release the spaces they were carved from.
        spaces_.clear();
        MeshDispatchFixture::TearDown();
    }

    // A space sized for exactly one pipe, and that pipe carved from it. The space must outlive
    // the pipe (the pipe does not own it), so it is parked on the fixture until TearDown.
    experimental::PrefetcherPipe make_pipe(
        distributed::MeshDevice* device, CoreCoord sender, const CoreRangeSet& receivers, uint32_t ring_size);

    bool is_fast_dispatch() const { return MetalContext::instance().rtoptions().get_fast_dispatch(); }

    bool is_quasar() const { return this->arch_ == tt::ARCH::QUASAR; }

    // Returns a skip reason if the worker grid is too small for the test mapping; empty otherwise.
    // Callers must GTEST_SKIP() << reason in the test body (GTEST_SKIP in a helper only returns
    // from the helper).
    std::string insufficient_worker_grid_reason(uint32_t min_x, uint32_t min_y = 1) const {
        const CoreCoord grid = devices_[0]->compute_with_storage_grid_size();
        if (grid.x < min_x || grid.y < min_y) {
            return "Requires worker grid >= " + std::to_string(min_x) + "x" + std::to_string(min_y) + " (got " +
                   std::to_string(grid.x) + "x" + std::to_string(grid.y) + ")";
        }
        return {};
    }

private:
    std::vector<experimental::PrefetcherPipeSpace> spaces_;
};

namespace {

namespace m2 = tt::tt_metal::experimental;

bool is_quasar_arch() { return MetalContext::instance().get_cluster().arch() == tt::ARCH::QUASAR; }

constexpr const char* sender_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_sender.cpp";
constexpr const char* receiver_kernel_path =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_receiver.cpp";
constexpr const char* relay_receiver_kernel_path =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_relay_receiver.cpp";
constexpr const char* relay_trisc_kernel_path =
    "tests/tt_metal/tt_metal/test_kernels/compute/prefetcher_pipe_relay_trisc.cpp";
constexpr const char* blank_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp";

// The one PrefetcherPipeParameter of a single-pipe Program.
const m2::PrefetcherPipeParamName pipe_param{"pipe"};

// ============================================================================
// Persistent resources
// ============================================================================

m2::PrefetcherPipeSpaceConfig one_pipe_space_config(
    CoreCoord sender, const CoreRangeSet& receivers, uint32_t ring_size) {
    return m2::PrefetcherPipeSpaceConfig{
        .sender_cores = CoreRangeSet(CoreRange(sender)),
        .receiver_domain = receivers,
        .ring_size = ring_size,
        .max_receivers_per_pipe = receivers.num_cores(),
    };
}

// ============================================================================
// ProgramSpec building blocks
// ============================================================================

// DM / compute KernelSpecs for the running arch. Gen1 pins every DM kernel to RISCV_0/NOC_0, which
// is fine here because the pipe tests never place two DM kernels on one node.
m2::KernelSpec make_dm_kernel(const std::string& name, const std::string& source, uint32_t num_threads = 1) {
    m2::KernelSpec kernel{
        .unique_id = m2::KernelSpecName{name},
        .source = std::filesystem::path{source},
        .num_threads = num_threads,
    };
    if (is_quasar_arch()) {
        kernel.hw_config = m2::DataMovementGen2Config{};
    } else {
        TT_FATAL(num_threads == 1, "Non-Quasar PrefetcherPipe tests only support 1 DM thread");
        kernel.hw_config = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    }
    return kernel;
}

m2::KernelSpec make_compute_kernel(const std::string& name, const std::string& source, uint32_t num_threads = 1) {
    m2::KernelSpec kernel{
        .unique_id = m2::KernelSpecName{name},
        .source = std::filesystem::path{source},
        .num_threads = num_threads,
    };
    if (is_quasar_arch()) {
        kernel.hw_config = m2::ComputeGen2Config{};
    } else {
        TT_FATAL(num_threads == 1, "Non-Quasar PrefetcherPipe tests only support 1 compute thread");
        kernel.hw_config = m2::ComputeGen1Config{};
    }
    return kernel;
}

m2::WorkUnitSpec work_unit(const std::string& name, const std::vector<std::string>& kernels, m2::Nodes nodes) {
    m2::WorkUnitSpec wu{.name = name, .kernels = {}, .target_nodes = std::move(nodes)};
    for (const std::string& kernel : kernels) {
        wu.kernels.push_back(m2::KernelSpecName{kernel});
    }
    return wu;
}

m2::PrefetcherPipeParameter pipe_parameter(
    const m2::PrefetcherPipeParamName& name, const CoreRangeSet& receivers, uint32_t ring_size, uint32_t entry_size) {
    return m2::PrefetcherPipeParameter{
        .unique_id = name,
        .receivers = receivers,
        .ring_size = ring_size,
        .entry_size = entry_size,
    };
}

// The parameter describing `pipe`, consumed at `entry_size`.
m2::PrefetcherPipeParameter pipe_parameter(
    const m2::PrefetcherPipeParamName& name, const m2::PrefetcherPipe& pipe, uint32_t entry_size) {
    return pipe_parameter(name, pipe.receiver_cores(), pipe.ring_size(), entry_size);
}

void bind_pipe(m2::KernelSpec& kernel, std::vector<m2::PrefetcherPipeParamName> pipes, const std::string& accessor) {
    kernel.advanced_options.prefetcher_pipe_bindings.push_back(
        m2::PrefetcherPipeBinding{.pipe_parameter_names = std::move(pipes), .accessor_name = accessor});
}

struct SenderKernelParams {
    uint32_t entry_size = 256;
    uint32_t num_entries = 4;
    // See prefetcher_pipe_sender.cpp: 0=broadcast, 1=strided, 2=write_to_receiver + push_back,
    // 3=per-receiver credit, 4=decoupled broadcast, 5=entry-major per-receiver credit.
    uint32_t write_primitive = 0;
    uint32_t do_barrier = 0;
    uint32_t num_threads = 1;

    uint32_t data_pattern() const { return cross_node_dfb_test::data_pattern_for_write_primitive(write_primitive); }
};

// prefetcher_pipe_sender.cpp with accessor "out" naming `pipes`. Staging address is a per-node RTA.
m2::KernelSpec sender_kernel_spec(
    const std::string& name, std::vector<m2::PrefetcherPipeParamName> pipes, const SenderKernelParams& p) {
    m2::KernelSpec kernel = make_dm_kernel(name, sender_kernel_path, p.num_threads);
    bind_pipe(kernel, std::move(pipes), "out");
    kernel.compile_time_args = {
        {"entry_size", p.entry_size},
        {"num_entries", p.num_entries},
        {"write_primitive", p.write_primitive},
        {"data_pattern", p.data_pattern()},
        {"do_barrier", p.do_barrier}};
    kernel.runtime_arg_schema.runtime_arg_names = {"staging_addr"};
    return kernel;
}

// prefetcher_pipe_receiver.cpp with accessor "in" naming `pipes`; `num_threads` is the pipe's P.
m2::KernelSpec receiver_kernel_spec(
    const std::string& name,
    std::vector<m2::PrefetcherPipeParamName> pipes,
    uint32_t num_entries,
    uint32_t num_threads = 1) {
    m2::KernelSpec kernel = make_dm_kernel(name, receiver_kernel_path, num_threads);
    bind_pipe(kernel, std::move(pipes), "in");
    kernel.compile_time_args = {{"num_entries", num_entries}};
    return kernel;
}

// Staging RTA for kernel `kernel` on every sender core of `pipe`.
m2::ProgramRunArgs::KernelRunArgs sender_staging_args(const std::string& kernel, const m2::PrefetcherPipe& pipe) {
    m2::ProgramRunArgs::KernelRunArgs args{.kernel = m2::KernelSpecName{kernel}};
    for (const CoreCoord& core : corerange_to_cores(pipe.sender_cores())) {
        m2::AddRuntimeArgsForNode(
            args.runtime_arg_values, core, {{"staging_addr", prefetcher_pipe_test::sender_l1_staging_address(pipe)}});
    }
    return args;
}

m2::ProgramRunArgs pipe_run_args(
    m2::PrefetcherPipe& pipe, std::vector<m2::ProgramRunArgs::KernelRunArgs> kernels = {}) {
    m2::ProgramRunArgs args;
    args.kernel_run_args = std::move(kernels);
    args.advanced_options.prefetcher_pipe_args.emplace(pipe_param, m2::PrefetcherPipeArgument{pipe});
    return args;
}

// ============================================================================
// Ready-to-enqueue Programs
// ============================================================================

// Sender-only Program: one sender kernel on the pipe's sender core. Staging is written and the
// pipe bound before return.
Program make_sender_program(distributed::MeshDevice& device, m2::PrefetcherPipe& pipe, const SenderKernelParams& p) {
    m2::ProgramSpec spec{
        .name = "pipe_sender",
        .kernels = {sender_kernel_spec("sender", {pipe_param}, p)},
        .work_units = {work_unit("sender_wu", {"sender"}, pipe.sender_cores())},
        .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, p.entry_size)}},
    };
    Program program = m2::MakeProgramFromSpec(device, spec);
    prefetcher_pipe_test::write_sender_l1_staging(
        device,
        pipe.sender_cores(),
        pipe,
        p.data_pattern(),
        p.entry_size,
        p.num_entries,
        pipe.receiver_cores().num_cores());
    m2::SetProgramRunArgs(program, pipe_run_args(pipe, {sender_staging_args("sender", pipe)}));
    return program;
}

// Receiver-only Program: one receiver kernel (P = num_threads) on the pipe's receiver cores.
Program make_receiver_program(
    distributed::MeshDevice& device,
    m2::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_threads = 1) {
    m2::ProgramSpec spec{
        .name = "pipe_receiver",
        .kernels = {receiver_kernel_spec("receiver", {pipe_param}, num_entries, num_threads)},
        .work_units = {work_unit("receiver_wu", {"receiver"}, pipe.receiver_cores())},
        .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, entry_size)}},
    };
    Program program = m2::MakeProgramFromSpec(device, spec);
    m2::SetProgramRunArgs(program, pipe_run_args(pipe));
    return program;
}

// ============================================================================
// Dispatch helpers
// ============================================================================

distributed::MeshCoordinateRange persistent_unit_mesh_device_range() {
    return distributed::MeshCoordinateRange({0, 0}, {0, 0});
}

Program& persistent_run_on_mesh_device(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    Program program,
    distributed::MeshWorkload& workload_out) {
    const auto device_range = persistent_unit_mesh_device_range();
    workload_out = distributed::MeshWorkload{};
    workload_out.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_out, false);
    distributed::Finish(mesh_device->mesh_command_queue());
    return workload_out.get_programs().at(device_range);
}

// Overlap producer/consumer across two programs (cross-program shape). Prefer a single Program
// when both ends can share one LaunchProgram — more reliable on RTL sim.
// Slow dispatch: default Finish only tracks the last enqueue's cores; async SD merges the wait
// set. Fast dispatch already runs the two programs back to back.
void persistent_run_overlapping_programs(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, Program first_program, Program second_program) {
    const bool fast_dispatch = MetalContext::instance().rtoptions().get_fast_dispatch();
    if (!fast_dispatch) {
        m2::DispatchContext::get().enable_asynchronous_slow_dispatch(mesh_device.get());
    }
    {
        const auto device_range = persistent_unit_mesh_device_range();
        distributed::MeshWorkload first_workload;
        first_workload.add_program(device_range, std::move(first_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), first_workload, false);
        distributed::MeshWorkload second_workload;
        second_workload.add_program(device_range, std::move(second_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), second_workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
    }
    if (!fast_dispatch) {
        m2::DispatchContext::get().disable_asynchronous_slow_dispatch(mesh_device.get());
    }
}

// Run `num_pushes` entries of credit through the pipe with no payload, so both endpoints reach
// a counter state that would otherwise take that many entries of real traffic. Producer and
// consumer sit in one program so they run at the same time: the producer can only get a ring
// ahead, so this costs one NoC round trip per lap. Every counter and cursor moves the way it
// does under real traffic, because it is the same credit path that moves it.
void spin_pipe_credits(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    m2::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_pushes) {
    const auto spin_kernel = [&](const std::string& name, uint32_t is_sender) {
        m2::KernelSpec kernel =
            make_dm_kernel(name, "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_credit_spin.cpp");
        bind_pipe(kernel, {pipe_param}, "pipe");
        kernel.compile_time_args = {{"num_ops", num_pushes}, {"entries_per_op", 1u}, {"is_sender", is_sender}};
        return kernel;
    };
    m2::ProgramSpec spec{
        .name = "pipe_credit_spin",
        .kernels = {spin_kernel("spin_sender", 1u), spin_kernel("spin_receiver", 0u)},
        .work_units =
            {work_unit("sender_wu", {"spin_sender"}, pipe.sender_cores()),
             work_unit("receiver_wu", {"spin_receiver"}, pipe.receiver_cores())},
        .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, entry_size)}},
    };
    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);
    m2::SetProgramRunArgs(program, pipe_run_args(pipe));

    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);
}

uint32_t run_persistent_sender_push(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    m2::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_entries) {
    Program program = make_sender_program(
        *mesh_device, pipe, {.entry_size = entry_size, .num_entries = num_entries, .write_primitive = 2});
    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);
    return 1u;
}

uint32_t run_persistent_receiver_pop(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    m2::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_entries) {
    distributed::MeshDevice& device = *mesh_device;
    Program program = make_receiver_program(device, pipe, entry_size, num_entries);
    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(2);
    const CoreCoord receiver = corerange_to_cores(pipe.receiver_cores()).front();
    return prefetcher_pipe_test::verify_receiver_ring(
               device, pipe, receiver, data_pattern, entry_size, num_entries, 0, 1)
               ? 1u
               : 0u;
}

// Cross-program 1:N: Program A pushes on the sender core; Program B pops on the receivers.
// Returns the number of receivers whose ring holds the expected bytes.
uint32_t run_persistent_1toN_cross_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    m2::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t write_primitive,
    bool simultaneous_subdevices = false,
    uint32_t num_sender_threads = 1) {
    distributed::MeshDevice& device = *mesh_device;
    const auto receivers = corerange_to_cores(pipe.receiver_cores());
    const uint32_t num_receivers = static_cast<uint32_t>(receivers.size());
    const SenderKernelParams sender_params{
        .entry_size = entry_size,
        .num_entries = num_entries,
        .write_primitive = write_primitive,
        .num_threads = num_sender_threads};

    Program sender_program = make_sender_program(device, pipe, sender_params);
    Program receiver_program = make_receiver_program(device, pipe, entry_size, num_entries);

    if (simultaneous_subdevices) {
        // Same pattern as remote-CB sub-device sync: launch sender, stall receiver SD so FD
        // can enqueue the consumer while the producer is live, then let both drain.
        distributed::MeshWorkload sender_workload;
        sender_workload.add_program(persistent_unit_mesh_device_range(), std::move(sender_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), sender_workload, false);

        mesh_device->set_sub_device_stall_group({{SubDeviceId{1}}});
        distributed::MeshWorkload receiver_workload;
        receiver_workload.add_program(persistent_unit_mesh_device_range(), std::move(receiver_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), receiver_workload, false);
        mesh_device->reset_sub_device_stall_group();
        distributed::Finish(mesh_device->mesh_command_queue());
    } else {
        // Two programs so producer/consumer overlap when the ring would otherwise fill before
        // the consumer is launched. (Do not merge into one Program when sender/receiver use
        // different num_threads — GO enables disagree.)
        persistent_run_overlapping_programs(mesh_device, std::move(sender_program), std::move(receiver_program));
    }

    uint32_t pass_count = 0;
    for (uint32_t ri = 0; ri < num_receivers; ++ri) {
        if (prefetcher_pipe_test::verify_receiver_ring(
                device,
                pipe,
                receivers[ri],
                sender_params.data_pattern(),
                entry_size,
                num_entries,
                ri,
                num_receivers)) {
            ++pass_count;
        }
    }
    return pass_count;
}

// Quasar multi-DM sender (partition-R via get_num_threads): same Flow C kernel as
// single-threaded, launched with num_threads > 1. APIs skip non-owned receivers.
uint32_t run_persistent_mt_sender_partition_r(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    m2::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_sender_threads,
    uint32_t write_primitive = 3) {
    TT_FATAL(is_quasar_arch(), "Multi-DM PrefetcherPipe sender tests require Quasar");
    TT_FATAL(num_sender_threads >= 2, "Multi-DM sender test requires num_sender_threads >= 2");
    TT_FATAL(pipe.receiver_cores().num_cores() >= 1, "Multi-DM sender test needs at least one receiver");
    return run_persistent_1toN_cross_program(
        mesh_device,
        pipe,
        entry_size,
        num_entries,
        write_primitive,
        /*simultaneous_subdevices=*/false,
        num_sender_threads);
}

// ============================================================================
// Relay DFB (pipe -> TRISC) helpers
// ============================================================================

uint32_t prefetcher_pipe_relay_expected_checksum(uint32_t total_entries) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < total_entries; ++i) {
        checksum += static_cast<uint32_t>(static_cast<uint8_t>(i)) * 0x01010101u;
    }
    return checksum;
}

// PrefetcherPipe relay e2e params. Multi-thread surface is the local relay DFB: the receiver
// DM kernel's num_threads is the relay's num_producers and the pipe's credit lanes (P > 1
// activates lane credits, so wait_front(n)/pop_front(n) are n owned strides); the compute
// kernel's num_threads is num_consumers with access pattern `cap`. Optional num_sender_threads>1
// partitions senders.
struct PrefetcherPipeRelayParams {
    uint32_t entry_size = 256;
    uint32_t ring_depth = 4;
    uint32_t total_entries = 4;
    uint32_t batch_size = 1;
    uint32_t num_producers = 1;
    uint32_t num_consumers = 1;
    m2::DFBAccessPattern cap = m2::DFBAccessPattern::STRIDED;
    std::optional<uint32_t> receiver_entry_size_override = std::nullopt;
    uint32_t trisc_delay_iterations = 0;
    // When true: one program with sender + relay receiver + TRISC (backpressure).
    // Supports num_sender_threads>1 (Quasar).
    bool same_program = false;
    // Quasar: >1 = stock sender with num_threads (Flows A–D partition-R).
    uint32_t num_sender_threads = 1;
};

// Receiver DM (relay producer) + TRISC (relay consumer) kernel specs and the relay DFB naming
// `pipes`.
struct RelayConsumerSpecs {
    m2::KernelSpec receiver;
    m2::KernelSpec compute;
    m2::DataflowBufferSpec relay;
};

struct RelayConsumerParams {
    uint32_t entry_size = 256;
    uint32_t ring_depth = 4;
    uint32_t pipe_total_entries = 4;    // pipe entries the receiver DM consumes over the run
    uint32_t pipe_batch_size = 1;       // pipe entries per wait_front / pop_front
    uint32_t entries_per_consumer = 4;  // relay entries each TRISC thread pops
    uint32_t trisc_batch_size = 1;
    uint32_t trisc_delay_iterations = 0;
    uint32_t num_producers = 1;  // receiver DM threads = pipe credit lanes
    uint32_t num_consumers = 1;  // TRISC threads
    m2::DFBAccessPattern cap = m2::DFBAccessPattern::STRIDED;
};

RelayConsumerSpecs make_relay_consumer(std::vector<m2::PrefetcherPipeParamName> pipes, const RelayConsumerParams& p) {
    const m2::DFBSpecName relay_name{"relay"};
    RelayConsumerSpecs c{
        .receiver = make_dm_kernel("receiver", relay_receiver_kernel_path, p.num_producers),
        .compute = make_compute_kernel("compute", relay_trisc_kernel_path, p.num_consumers),
        .relay =
            m2::DataflowBufferSpec{
                .unique_id = relay_name,
                .entry_size = p.entry_size,
                .num_entries = p.ring_depth,
                .data_format_metadata = tt::DataFormat::Float16_b,
                .advanced_options = {.prefetcher_pipe_relays = pipes},
            },
    };
    bind_pipe(c.receiver, std::move(pipes), "in");
    c.receiver.dfb_bindings.push_back(m2::ProducerOf(relay_name, "relay"));
    c.receiver.compile_time_args = {{"total_entries", p.pipe_total_entries}, {"batch_size", p.pipe_batch_size}};
    c.compute.dfb_bindings.push_back(
        p.cap == m2::DFBAccessPattern::ALL ? m2::AllConsumerOf(relay_name, "relay")
                                           : m2::StridedConsumerOf(relay_name, "relay"));
    c.compute.compile_time_args = {
        {"entries_this_thread", p.entries_per_consumer},
        {"batch_size", p.trisc_batch_size},
        {"delay_iterations", p.trisc_delay_iterations}};
    c.compute.runtime_arg_schema.runtime_arg_names = {"result_addr"};
    return c;
}

m2::ProgramRunArgs::KernelRunArgs compute_result_args(const CoreRangeSet& receiver_cores, uint32_t result_addr) {
    m2::ProgramRunArgs::KernelRunArgs args{.kernel = m2::KernelSpecName{"compute"}};
    for (const CoreCoord& core : corerange_to_cores(receiver_cores)) {
        m2::AddRuntimeArgsForNode(args.runtime_arg_values, core, {{"result_addr", result_addr}});
    }
    return args;
}

// Per-receiver-core check of the TRISC relay consumer output. ALL: every thread saw every entry;
// STRIDED: the threads partition the entries. Returns the number of receiver cores that match.
uint32_t verify_relay_results(
    distributed::MeshDevice& device,
    const CoreRangeSet& receiver_cores,
    uint32_t result_addr,
    uint32_t num_consumers,
    m2::DFBAccessPattern cap,
    uint32_t entries_per_consumer,
    uint32_t total_entries) {
    const uint32_t expected_checksum = prefetcher_pipe_relay_expected_checksum(total_entries);
    uint32_t pass_count = 0;
    for (const CoreCoord& core : corerange_to_cores(receiver_cores)) {
        std::vector<uint32_t> result(num_consumers * 2, 0);
        slow_dispatch::ReadFromL1(
            device,
            core,
            result_addr,
            std::span<uint8_t>(reinterpret_cast<uint8_t*>(result.data()), result.size() * sizeof(uint32_t)),
            CoreType::WORKER);

        bool ok = true;
        uint32_t got_entries = 0;
        uint32_t got_checksum = 0;
        for (uint32_t tid = 0; tid < num_consumers; ++tid) {
            const bool count_ok = result[tid * 2 + 0] == entries_per_consumer;
            const bool checksum_ok = cap != m2::DFBAccessPattern::ALL || result[tid * 2 + 1] == expected_checksum;
            if (!count_ok || !checksum_ok) {
                ok = false;
                log_error(
                    tt::LogTest,
                    "PrefetcherPipe relay mismatch on {} tid {}: count {} (expected {}), checksum 0x{:08x}",
                    core.str(),
                    tid,
                    result[tid * 2 + 0],
                    entries_per_consumer,
                    result[tid * 2 + 1]);
            }
            got_entries += result[tid * 2 + 0];
            got_checksum += result[tid * 2 + 1];
        }
        if (cap != m2::DFBAccessPattern::ALL && (got_entries != total_entries || got_checksum != expected_checksum)) {
            ok = false;
            log_error(
                tt::LogTest,
                "PrefetcherPipe relay STRIDED aggregate mismatch on {}: count {} (expected {}), checksum 0x{:08x} "
                "(expected 0x{:08x})",
                core.str(),
                got_entries,
                total_entries,
                got_checksum,
                expected_checksum);
        }
        if (ok) {
            ++pass_count;
        }
    }
    return pass_count;
}

// Prog A (or same-program): sender push. Prog B: receiver DM bind_relay + TRISC consume.
// Returns number of receiver cores whose per-thread results match the expected pattern.
uint32_t run_prefetcher_pipe_relay(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    m2::PrefetcherPipe& pipe,
    const PrefetcherPipeRelayParams& params) {
    TT_FATAL(params.total_entries % params.batch_size == 0, "Relay test total_entries must be divisible by batch_size");
    TT_FATAL(params.ring_depth % params.batch_size == 0, "Relay test ring_depth must be divisible by batch_size");
    TT_FATAL(params.num_producers >= 1, "num_producers must be >= 1");
    TT_FATAL(params.num_consumers >= 1, "num_consumers must be >= 1");
    if (params.cap == m2::DFBAccessPattern::STRIDED) {
        TT_FATAL(
            params.total_entries % params.num_consumers == 0,
            "STRIDED: total_entries must be divisible by num_consumers");
    }
    TT_FATAL(
        params.ring_depth % std::max(params.num_producers, params.num_consumers) == 0,
        "ring_depth must be divisible by max(P,C)");
    TT_FATAL(params.num_sender_threads >= 1, "num_sender_threads must be >= 1");
    if (params.num_producers > 1) {
        TT_FATAL(is_quasar_arch(), "Multi-producer PrefetcherPipe relay requires Quasar");
        TT_FATAL(
            params.total_entries % params.num_producers == 0,
            "Multi-producer: total_entries must be divisible by num_producers");
        TT_FATAL(
            (params.total_entries / params.num_producers) % params.batch_size == 0,
            "Multi-producer: per-hart entries must be divisible by batch_size");
        TT_FATAL(
            pipe.impl().credit_lane_capacity() >= params.num_producers,
            "Multi-producer: PrefetcherPipe credit_lane_capacity {} < num_producers {}",
            pipe.impl().credit_lane_capacity(),
            params.num_producers);
    }
    if (params.num_sender_threads > 1) {
        TT_FATAL(is_quasar_arch(), "Multi-DM PrefetcherPipe requires Quasar");
    }

    distributed::MeshDevice& device = *mesh_device;
    const CoreRangeSet& receiver_cores = pipe.receiver_cores();
    const uint32_t recv_entry_size = params.receiver_entry_size_override.value_or(params.entry_size);
    const uint32_t recv_num_entries = pipe.ring_size() / recv_entry_size;
    TT_FATAL(pipe.ring_size() % recv_entry_size == 0, "receiver entry size must divide ring");
    TT_FATAL(
        recv_num_entries == params.ring_depth || params.receiver_entry_size_override.has_value(),
        "ring_depth must match pipe.ring_size/entry_size unless overriding recv entry size");

    const uint32_t recv_total_entries = (params.total_entries * params.entry_size) / recv_entry_size;
    TT_FATAL(
        (params.total_entries * params.entry_size) % recv_entry_size == 0,
        "pushed bytes must be divisible by recv entry size");
    TT_FATAL(recv_total_entries % params.batch_size == 0, "recv_total_entries must be divisible by batch_size");

    const uint32_t entries_per_consumer =
        (params.cap == m2::DFBAccessPattern::ALL) ? recv_total_entries : (recv_total_entries / params.num_consumers);
    TT_FATAL(entries_per_consumer % params.batch_size == 0, "entries_per_consumer must be divisible by batch_size");

    // Pipe-side batch for the relay receiver DM. The receiver publishes one relay entry per
    // push, round-robin over its consumer TCs (STRIDED cap with C > P gives each producer
    // C / P TCs). A TRISC batch of b on one TC therefore needs b * (C / P) pipe entries per
    // iteration, or the DM would block in pop_front waiting for consumers that are still
    // waiting for their batch. batch_size == 1 never needs the factor.
    uint32_t pipe_batch_size = params.batch_size;
    if (params.batch_size > 1 && params.cap == m2::DFBAccessPattern::STRIDED &&
        params.num_consumers > params.num_producers) {
        TT_FATAL(
            params.num_consumers % params.num_producers == 0,
            "STRIDED relay: num_consumers must be a multiple of num_producers");
        pipe_batch_size *= params.num_consumers / params.num_producers;
    }
    TT_FATAL(
        (recv_total_entries / params.num_producers) % pipe_batch_size == 0,
        "per-hart pipe entries {} must be divisible by pipe batch {}",
        recv_total_entries / params.num_producers,
        pipe_batch_size);
    TT_FATAL(
        (recv_num_entries / params.num_producers) >= pipe_batch_size,
        "per-hart ring depth {} must hold a pipe batch of {}",
        recv_num_entries / params.num_producers,
        pipe_batch_size);

    const uint32_t result_words = params.num_consumers * 2u;
    const uint32_t result_page_size = std::max(32u, result_words * static_cast<uint32_t>(sizeof(uint32_t)));
    auto result_buffer = cross_node_dfb_test::make_cross_node_data_buffer(device, receiver_cores, result_page_size, 1);
    const uint32_t result_addr = static_cast<uint32_t>(result_buffer->address());

    const SenderKernelParams sender_params{
        .entry_size = params.entry_size,
        .num_entries = params.total_entries,
        .write_primitive = 0,
        .num_threads = params.num_sender_threads};
    RelayConsumerSpecs consumer = make_relay_consumer(
        {pipe_param},
        RelayConsumerParams{
            .entry_size = recv_entry_size,
            .ring_depth = recv_num_entries,
            .pipe_total_entries = recv_total_entries,
            .pipe_batch_size = pipe_batch_size,
            .entries_per_consumer = entries_per_consumer,
            .trisc_batch_size = params.batch_size,
            .trisc_delay_iterations = params.trisc_delay_iterations,
            .num_producers = params.num_producers,
            .num_consumers = params.num_consumers,
            .cap = params.cap,
        });

    if (params.same_program) {
        TT_FATAL(!params.receiver_entry_size_override.has_value(), "same-program relay uses one entry size");
        m2::ProgramSpec spec{
            .name = "pipe_relay_same_program",
            .kernels = {sender_kernel_spec("sender", {pipe_param}, sender_params), consumer.receiver, consumer.compute},
            .dataflow_buffers = {consumer.relay},
            .work_units =
                {work_unit("sender_wu", {"sender"}, pipe.sender_cores()),
                 work_unit("receiver_wu", {"receiver", "compute"}, receiver_cores)},
            .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, params.entry_size)}},
        };
        Program program = m2::MakeProgramFromSpec(device, spec);
        // The relay DFB has no address until the pipe binds.
        EXPECT_EQ(program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("relay"))->borrowed_addr_, 0u);
        prefetcher_pipe_test::write_sender_l1_staging(
            device,
            pipe.sender_cores(),
            pipe,
            sender_params.data_pattern(),
            params.entry_size,
            params.total_entries,
            1);
        m2::SetProgramRunArgs(
            program,
            pipe_run_args(
                pipe, {sender_staging_args("sender", pipe), compute_result_args(receiver_cores, result_addr)}));
        EXPECT_EQ(
            program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("relay"))->borrowed_addr_,
            pipe.buffer_address());

        distributed::MeshWorkload workload;
        persistent_run_on_mesh_device(mesh_device, std::move(program), workload);
    } else {
        // Cross-program shape: bind the consumer first so its receiver kernel arms the pipe's
        // lanes before the sender's slot payload is built at enqueue.
        m2::ProgramSpec consumer_spec{
            .name = "pipe_relay_consumer",
            .kernels = {consumer.receiver, consumer.compute},
            .dataflow_buffers = {consumer.relay},
            .work_units = {work_unit("receiver_wu", {"receiver", "compute"}, receiver_cores)},
            .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, recv_entry_size)}},
        };
        Program program_consumer = m2::MakeProgramFromSpec(device, consumer_spec);
        m2::SetProgramRunArgs(
            program_consumer, pipe_run_args(pipe, {compute_result_args(receiver_cores, result_addr)}));

        Program program_sender = make_sender_program(device, pipe, sender_params);
        persistent_run_overlapping_programs(mesh_device, std::move(program_sender), std::move(program_consumer));
    }

    return verify_relay_results(
        device,
        receiver_cores,
        result_addr,
        params.num_consumers,
        params.cap,
        entries_per_consumer,
        recv_total_entries);
}

// 1P1C relay with an optional receiver-side entry size that differs from the sender's.
uint32_t run_prefetcher_pipe_relay_cross_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    m2::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t ring_depth,
    uint32_t total_entries,
    uint32_t batch_size,
    std::optional<uint32_t> receiver_entry_size_override = std::nullopt,
    uint32_t trisc_delay_iterations = 0) {
    PrefetcherPipeRelayParams params{
        .entry_size = entry_size,
        .ring_depth = ring_depth,
        .total_entries = total_entries,
        .batch_size = batch_size,
        .receiver_entry_size_override = receiver_entry_size_override,
        .trisc_delay_iterations = trisc_delay_iterations,
    };
    // When entry size is overridden, ring_depth is in sender units; recv depth is recomputed inside.
    if (receiver_entry_size_override.has_value()) {
        params.ring_depth = pipe.ring_size() / *receiver_entry_size_override;
    }
    return run_prefetcher_pipe_relay(mesh_device, pipe, params);
}

// A Program whose receiver kernel runs `num_threads` on the pipe's receivers, optionally with a
// relay DFB (produced by that kernel, consumed by a TRISC kernel with `num_consumers` threads and
// access pattern `cap`). Built but not bound.
struct ReceiverProgramSpecParams {
    uint32_t entry_size = 256;
    uint32_t num_threads = 1;
    bool with_relay = false;
    uint32_t num_consumers = 1;
    m2::DFBAccessPattern cap = m2::DFBAccessPattern::STRIDED;
};

m2::ProgramSpec receiver_program_spec(const m2::PrefetcherPipe& pipe, const ReceiverProgramSpecParams& p) {
    const uint32_t ring_depth = pipe.ring_size() / p.entry_size;
    if (!p.with_relay) {
        return m2::ProgramSpec{
            .name = "pipe_receiver_spec",
            .kernels = {receiver_kernel_spec("receiver", {pipe_param}, ring_depth, p.num_threads)},
            .work_units = {work_unit("receiver_wu", {"receiver"}, pipe.receiver_cores())},
            .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, p.entry_size)}},
        };
    }
    RelayConsumerSpecs consumer = make_relay_consumer(
        {pipe_param},
        RelayConsumerParams{
            .entry_size = p.entry_size,
            .ring_depth = ring_depth,
            .pipe_total_entries = ring_depth,
            .entries_per_consumer = ring_depth,
            .num_producers = p.num_threads,
            .num_consumers = p.num_consumers,
            .cap = p.cap,
        });
    return m2::ProgramSpec{
        .name = "pipe_relay_receiver_spec",
        .kernels = {consumer.receiver, consumer.compute},
        .dataflow_buffers = {consumer.relay},
        .work_units = {work_unit("receiver_wu", {"receiver", "compute"}, pipe.receiver_cores())},
        .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, p.entry_size)}},
    };
}

// The slot's relay word (relay id | credit lanes) as dispatch packs it for `core`.
uint32_t slot_relay_word(const Program& program, const CoreCoord& core) {
    const auto payload = program_dispatch::build_prefetcher_pipe_config_payload(
        program.impl(), program.impl().get_per_core_prefetcher_pipes().at(core));
    return payload[REMOTE_DFB_REGION_HEADER_WORDS + 2];
}

}  // namespace

m2::PrefetcherPipe PrefetcherPipeFixture::make_pipe(
    distributed::MeshDevice* device, CoreCoord sender, const CoreRangeSet& receivers, uint32_t ring_size) {
    spaces_.push_back(m2::CreatePrefetcherPipeSpace(*device, one_pipe_space_config(sender, receivers, ring_size)));
    return spaces_.back().create_pipe(sender, receivers);
}

// ============================================================================
// PrefetcherPipeSpace: reservation and carving
// ============================================================================

TEST_F(PrefetcherPipeFixture, PrefetcherPipeSpace_CarveRejects) {
    if (const auto reason = insufficient_worker_grid_reason(3); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const CoreCoord sender(0, 0);
    const CoreRangeSet domain(CoreRange({1, 0}, {2, 0}));
    auto space = m2::CreatePrefetcherPipeSpace(
        *mesh_device,
        m2::PrefetcherPipeSpaceConfig{
            .sender_cores = CoreRangeSet(CoreRange(sender)),
            .receiver_domain = domain,
            .ring_size = 1024,
            .max_receivers_per_pipe = 1,
        });
    EXPECT_EQ(space.reservation_cores().num_cores(), 3u);
    EXPECT_EQ(space.unclaimed_cores().num_cores(), 3u);

    // Sender outside sender_cores; receivers outside the domain; receivers containing the
    // sender; no receivers; more receivers than the space was sized for.
    EXPECT_THROW(space.create_pipe(CoreCoord(1, 0), CoreRangeSet(CoreRange({2, 0}))), std::exception);
    EXPECT_THROW(space.create_pipe(sender, CoreRangeSet(CoreRange({0, 1}))), std::exception);
    EXPECT_THROW(space.create_pipe(sender, CoreRangeSet(CoreRange({0, 0}, {1, 0}))), std::exception);
    EXPECT_THROW(space.create_pipe(sender, CoreRangeSet{}), std::exception);
    EXPECT_THROW(space.create_pipe(sender, domain), std::exception);
    EXPECT_EQ(space.unclaimed_cores().num_cores(), 3u);

    {
        auto pipe = space.create_pipe(sender, CoreRangeSet(CoreRange({1, 0})));
        EXPECT_EQ(space.unclaimed_cores().num_cores(), 1u);
        EXPECT_TRUE(space.unclaimed_cores().contains(CoreCoord(2, 0)));
        // The sender and (1,0) are claimed by a live pipe.
        EXPECT_THROW(space.create_pipe(sender, CoreRangeSet(CoreRange({2, 0}))), std::exception);
    }
    // Destroying the pipe returns its cores.
    EXPECT_EQ(space.unclaimed_cores().num_cores(), 3u);
    EXPECT_NO_THROW(space.create_pipe(sender, CoreRangeSet(CoreRange({2, 0}))));
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipeSpace_ConfigRejects) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const auto config = one_pipe_space_config(CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);

    auto with = [&](auto&& mutate) {
        auto c = config;
        mutate(c);
        return c;
    };
    EXPECT_THROW(m2::CreatePrefetcherPipeSpace(*mesh_device, with([](auto& c) { c.ring_size = 0; })), std::exception);
    EXPECT_THROW(m2::CreatePrefetcherPipeSpace(*mesh_device, with([](auto& c) { c.ring_size = 33; })), std::exception);
    EXPECT_THROW(
        m2::CreatePrefetcherPipeSpace(*mesh_device, with([](auto& c) { c.buffer_type = BufferType::DRAM; })),
        std::exception);
    EXPECT_THROW(
        m2::CreatePrefetcherPipeSpace(*mesh_device, with([](auto& c) { c.max_receivers_per_pipe = 0; })),
        std::exception);
    // A pipe cannot have more receivers than the domain it is carved from.
    EXPECT_THROW(
        m2::CreatePrefetcherPipeSpace(
            *mesh_device, with([](auto& c) { c.max_receivers_per_pipe = c.receiver_domain.num_cores() + 1; })),
        std::exception);
    // No receiver domain: nothing could ever be carved.
    EXPECT_THROW(
        m2::CreatePrefetcherPipeSpace(*mesh_device, with([](auto& c) { c.receiver_domain = CoreRangeSet{}; })),
        std::exception);
    EXPECT_THROW(
        m2::CreatePrefetcherPipeSpace(*mesh_device, with([](auto& c) {
            c.sender_cores = CoreRangeSet{};
            c.receiver_domain = CoreRangeSet{};
        })),
        std::exception);
    // DRAM-sender capacity is not carvable yet (tt-metal#55285).
    EXPECT_THROW(
        m2::CreatePrefetcherPipeSpace(*mesh_device, with([](auto& c) { c.num_dram_senders = 1; })), std::exception);
    EXPECT_NO_THROW(m2::CreatePrefetcherPipeSpace(*mesh_device, config));
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipeSpace_PipesShareAddressesByContract) {
    if (const auto reason = insufficient_worker_grid_reason(4); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    // One space over two disjoint sender/receiver pairs: both pipes sit at the space's addresses.
    auto space = m2::CreatePrefetcherPipeSpace(
        *mesh_device,
        m2::PrefetcherPipeSpaceConfig{
            .sender_cores = CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}), CoreRange({2, 0})}),
            .receiver_domain = CoreRangeSet(std::vector<CoreRange>{CoreRange({1, 0}), CoreRange({3, 0})}),
            .ring_size = 1024,
            .max_receivers_per_pipe = 1,
        });
    auto pipe0 = space.create_pipe(CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})));
    auto pipe1 = space.create_pipe(CoreCoord(2, 0), CoreRangeSet(CoreRange({3, 0})));
    EXPECT_EQ(pipe0.buffer_address(), space.buffer_address());
    EXPECT_EQ(pipe1.buffer_address(), space.buffer_address());
    EXPECT_EQ(pipe0.config_address(), space.config_address());
    EXPECT_EQ(pipe1.config_address(), space.config_address());
    EXPECT_EQ(pipe0.config_page_size(), space.config_page_size());
}

TEST_F(PrefetcherPipeFixture, PersistentArenaSharesAddressesAcrossDisjointSpaces) {
    if (const auto reason = insufficient_worker_grid_reason(4); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    // Two spaces on disjoint cores: the persistent arena places both at the same per-core L1.
    auto pipe0 = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);
    auto pipe1 = make_pipe(mesh_device.get(), CoreCoord(2, 0), CoreRangeSet(CoreRange({3, 0})), 1024);

    EXPECT_EQ(pipe0.buffer_address(), pipe1.buffer_address());
    EXPECT_EQ(pipe0.config_address(), pipe1.config_address());
}

TEST_F(PrefetcherPipeFixture, MultipleDisjointOneToNPipesShareL1Address) {
    if (const auto reason = insufficient_worker_grid_reason(3, 3); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t num_entries = 4;
    constexpr uint32_t ring_bytes = entry_size * num_entries;

    // M=3 independent 1:N pipes (N=2) carved from one space in one batch. Their cores are
    // disjoint, so every ring/config sits at the same per-core L1 address.
    auto space = m2::CreatePrefetcherPipeSpace(
        *mesh_device,
        m2::PrefetcherPipeSpaceConfig{
            .sender_cores = CoreRangeSet(CoreRange({0, 0}, {0, 2})),
            .receiver_domain = CoreRangeSet(CoreRange({1, 0}, {2, 2})),
            .ring_size = ring_bytes,
            .max_receivers_per_pipe = 2,
        });
    const std::vector<std::pair<CoreCoord, CoreRangeSet>> mappings = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {2, 0}))},
        {CoreCoord(0, 1), CoreRangeSet(CoreRange({1, 1}, {2, 1}))},
        {CoreCoord(0, 2), CoreRangeSet(CoreRange({1, 2}, {2, 2}))},
    };
    std::vector<m2::PrefetcherPipe> pipes = space.create_pipes(mappings);
    ASSERT_EQ(pipes.size(), 3u);
    EXPECT_EQ(space.unclaimed_cores().num_cores(), 0u);

    for (const auto& pipe : pipes) {
        EXPECT_EQ(pipe.buffer_address(), space.buffer_address());
        EXPECT_EQ(pipe.config_address(), space.config_address());
    }
    for (auto& pipe : pipes) {
        EXPECT_EQ(
            run_persistent_1toN_cross_program(mesh_device, pipe, entry_size, num_entries, /*write_primitive=*/0), 2u);
    }
}

TEST_F(PrefetcherPipeFixture, PersistentArenaSerializesOverlappingSpacesAndReusesFreedSpace) {
    if (const auto reason = insufficient_worker_grid_reason(3); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    uint32_t first_ring_address = 0;
    uint32_t first_config_address = 0;
    {
        // Spaces are scoped here rather than parked on the fixture: the replacement below can only
        // land back at the first addresses once both spaces have released their persistent L1.
        // Each space is declared before its pipe so the pipe dies first.
        const CoreRangeSet shared_receiver = CoreRangeSet(CoreRange({1, 0}));
        auto space0 =
            m2::CreatePrefetcherPipeSpace(*mesh_device, one_pipe_space_config(CoreCoord(0, 0), shared_receiver, 1024));
        auto pipe0 = space0.create_pipe(CoreCoord(0, 0), shared_receiver);
        first_ring_address = pipe0.buffer_address();
        first_config_address = pipe0.config_address();

        // A second space sharing (1,0) cannot alias the first one's L1 there.
        auto space1 =
            m2::CreatePrefetcherPipeSpace(*mesh_device, one_pipe_space_config(CoreCoord(2, 0), shared_receiver, 1024));
        auto pipe1 = space1.create_pipe(CoreCoord(2, 0), shared_receiver);
        EXPECT_GE(pipe1.buffer_address(), pipe0.config_address() + pipe0.config_page_size());
    }

    auto replacement = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);
    EXPECT_EQ(replacement.buffer_address(), first_ring_address);
    EXPECT_EQ(replacement.config_address(), first_config_address);
}

// ============================================================================
// ProgramSpec / ProgramRunArgs: slot reservation and binding
// ============================================================================

TEST_F(PrefetcherPipeFixture, ProgramSpec_EntrySizeRejects) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);

    // 0, not L1-aligned, larger than the ring.
    for (const uint32_t entry_size : {0u, 33u, 1280u}) {
        m2::ProgramSpec spec{
            .name = "bad_entry_size",
            .kernels = {sender_kernel_spec("sender", {pipe_param}, {.entry_size = entry_size})},
            .work_units = {work_unit("sender_wu", {"sender"}, pipe.sender_cores())},
            .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, entry_size)}},
        };
        EXPECT_THROW(m2::MakeProgramFromSpec(*mesh_device, spec), std::exception) << "entry_size=" << entry_size;
    }
    Program program = make_sender_program(*mesh_device, pipe, {.entry_size = 256});
    EXPECT_EQ(program.impl().num_prefetcher_pipe_slots(), 1u);
}

TEST_F(PrefetcherPipeFixture, ProgramSpec_RequiresRoleCompleteKernel) {
    if (const auto reason = insufficient_worker_grid_reason(4); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const CoreRangeSet receiver_cores(CoreRange({2, 0}, {3, 0}));
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, 1024);

    // Sender-only program: fine.
    EXPECT_NO_THROW(make_sender_program(*mesh_device, pipe, {.entry_size = 256}));
    // A receiver kernel on one of the two receivers is neither the sender nor the receiver set.
    {
        m2::ProgramSpec spec{
            .name = "partial_receivers",
            .kernels = {receiver_kernel_spec("receiver", {pipe_param}, 4)},
            .work_units = {work_unit("receiver_wu", {"receiver"}, CoreCoord(2, 0))},
            .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, 256)}},
        };
        EXPECT_THROW(m2::MakeProgramFromSpec(*mesh_device, spec), std::exception);
    }
    // Sender in one program, all receivers in another: fine.
    EXPECT_NO_THROW(make_sender_program(*mesh_device, pipe, {.entry_size = 256}));
    EXPECT_NO_THROW(make_receiver_program(*mesh_device, pipe, 256, 4));
}

TEST_F(PrefetcherPipeFixture, ProgramSpec_AssignsDistinctSlotsPerAccessor) {
    if (const auto reason = insufficient_worker_grid_reason(2, 2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe0 = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);
    auto pipe1 = make_pipe(mesh_device.get(), CoreCoord(0, 1), CoreRangeSet(CoreRange({1, 1})), 1024);
    const m2::PrefetcherPipeParamName name0{"p0"};
    const m2::PrefetcherPipeParamName name1{"p1"};

    // Two sender kernels, each binding its own pipe: two slots, id 0 and 1 in kernel order.
    m2::KernelSpec sender0 = make_dm_kernel("sender0", blank_kernel_path);
    bind_pipe(sender0, {name0}, "out");
    m2::KernelSpec sender1 = make_dm_kernel("sender1", blank_kernel_path);
    bind_pipe(sender1, {name1}, "out");
    m2::ProgramSpec spec{
        .name = "two_slots",
        .kernels = {sender0, sender1},
        .work_units =
            {work_unit("sender0_wu", {"sender0"}, pipe0.sender_cores()),
             work_unit("sender1_wu", {"sender1"}, pipe1.sender_cores())},
        .advanced_options =
            {.prefetcher_pipe_parameters = {pipe_parameter(name0, pipe0, 256), pipe_parameter(name1, pipe1, 256)}},
    };
    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);
    EXPECT_EQ(program.impl().num_prefetcher_pipe_slots(), 2u);
    const auto& per_core = program.impl().get_per_core_prefetcher_pipes();
    ASSERT_EQ(per_core.at(CoreCoord(0, 0)).size(), 1u);
    ASSERT_EQ(per_core.at(CoreCoord(0, 1)).size(), 1u);
    EXPECT_EQ(per_core.at(CoreCoord(0, 0))[0].prefetcher_pipe_id, 0u);
    EXPECT_EQ(per_core.at(CoreCoord(0, 1))[0].prefetcher_pipe_id, 1u);

    m2::ProgramRunArgs args;
    args.advanced_options.prefetcher_pipe_args = {
        {name0, m2::PrefetcherPipeArgument{pipe0}}, {name1, m2::PrefetcherPipeArgument{pipe1}}};
    m2::SetProgramRunArgs(program, args);
    EXPECT_EQ(per_core.at(CoreCoord(0, 0))[0].pipe, &pipe0.impl());
    EXPECT_EQ(per_core.at(CoreCoord(0, 1))[0].pipe, &pipe1.impl());

    detail::CompileProgram(mesh_device.get(), program);
    program.impl().finalize_offsets(mesh_device.get());

    const auto& hal = MetalContext::instance().hal();
    const uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    ASSERT_FALSE(program.impl().get_kernel_groups(index).empty());
    EXPECT_NE(
        program.impl().get_kernel_groups(index)[0]->launch_msg.view().kernel_config().prefetcher_pipe_offset(),
        REMOTE_DFB_OFFSET_NONE);
    EXPECT_EQ(
        program.impl().get_kernel_groups(index)[0]->launch_msg.view().kernel_config().cross_node_dfb_offset(),
        REMOTE_DFB_OFFSET_NONE);
}

TEST_F(PrefetcherPipeFixture, ProgramRunArgs_SameObjectMultiplePrograms) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);
    const uint32_t fifo_start = pipe.buffer_address();
    const uint32_t config_addr = pipe.config_address();

    Program program_a = make_sender_program(*mesh_device, pipe, {.entry_size = 256});
    Program program_b = make_sender_program(*mesh_device, pipe, {.entry_size = 256});

    const auto& per_core_a = program_a.impl().get_per_core_prefetcher_pipes().at(CoreCoord(0, 0));
    const auto& per_core_b = program_b.impl().get_per_core_prefetcher_pipes().at(CoreCoord(0, 0));
    EXPECT_EQ(per_core_a[0].config_page_addr, config_addr);
    EXPECT_EQ(per_core_b[0].config_page_addr, config_addr);
    EXPECT_EQ(per_core_a[0].pipe, &pipe.impl());
    EXPECT_EQ(per_core_b[0].pipe, &pipe.impl());
    EXPECT_EQ(pipe.buffer_address(), fifo_start);
}

TEST_F(PrefetcherPipeFixture, ProgramRunArgs_AddressStableAcrossRebuild) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);
    const uint32_t ring_addr = pipe.buffer_address();
    const uint32_t config_addr = pipe.config_address();

    {
        Program program = make_sender_program(*mesh_device, pipe, {.entry_size = 256});
        detail::CompileProgram(mesh_device.get(), program);
        program.impl().finalize_offsets(mesh_device.get());
    }
    {
        Program program = make_sender_program(*mesh_device, pipe, {.entry_size = 256});
        EXPECT_EQ(pipe.buffer_address(), ring_addr);
        EXPECT_EQ(pipe.config_address(), config_addr);
        EXPECT_EQ(program.impl().get_per_core_prefetcher_pipes().at(CoreCoord(0, 0))[0].config_page_addr, config_addr);
    }
}

// ============================================================================
// Cross-program persistence
// ============================================================================

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_CrossProgramPersistence) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);
    const uint32_t ring_addr = pipe.buffer_address();

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, 256, 4), 1u);
    EXPECT_EQ(pipe.buffer_address(), ring_addr);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, 256, 4), 1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_ProducerRelaunchWithOutstandingCredits) {
    // Same-epoch producer relaunch must not barrier on durable outstanding entries:
    // fill half the ring, relaunch sender to fill the rest, then drain once.
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    constexpr uint32_t first_push = 2;
    constexpr uint32_t second_push = 2;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, entry_size, first_push), 1u);
    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, entry_size, second_push), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, entry_size, first_push + second_push), 1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_BackToBackRelaunch) {
    // Two cross-program push→pop cycles on the same PrefetcherPipe.
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, 256, 4), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, 256, 4), 1u);
    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, 256, 4), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, 256, 4), 1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_CrossSubDevicePersistence) {
    if (!is_fast_dispatch()) {
        GTEST_SKIP() << "Sub device managers are unsupported with slow dispatch";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    // Programs may only span one sub-device. Put the sender on SD0 and the receiver on SD1,
    // each program binds only its role cores, and share one PrefetcherPipe across both.
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(1, 0);
    const CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreRangeSet receiver_cores = CoreRangeSet(CoreRange(receiver_core));

    SubDevice sender_sub_device(std::array{sender_cores});
    SubDevice receiver_sub_device(std::array{receiver_cores});
    // local_l1_size=0: PrefetcherPipe ring/config stay on the global allocator so they can
    // cover cores from both sub-devices (same pattern as remote-CB sub-device tests).
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, /*local_l1_size=*/0);
    mesh_device->load_sub_device_manager(sub_device_manager);

    {
        constexpr uint32_t entry_size = 256;
        constexpr uint32_t num_entries = 4;
        auto pipe = make_pipe(mesh_device.get(), sender_core, receiver_cores, entry_size * num_entries);
        const uint32_t ring_addr = pipe.buffer_address();

        EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, entry_size, num_entries), 1u);
        EXPECT_EQ(pipe.buffer_address(), ring_addr);
        EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, entry_size, num_entries), 1u);
    }

    mesh_device->clear_loaded_sub_device_manager();
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_CrossSubDevice_ABC_ReceiverRelaunch) {
    if (!is_fast_dispatch()) {
        GTEST_SKIP() << "Sub device managers are unsupported with slow dispatch";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    // A on SD0 pushes, B on SD1 pops and finishes, then C on SD1 pops a second push from A.
    // Confirms SD1 can relaunch a new consumer program against the same PrefetcherPipe.
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(1, 0);
    const CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreRangeSet receiver_cores = CoreRangeSet(CoreRange(receiver_core));

    SubDevice sender_sub_device(std::array{sender_cores});
    SubDevice receiver_sub_device(std::array{receiver_cores});
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, /*local_l1_size=*/0);
    mesh_device->load_sub_device_manager(sub_device_manager);

    constexpr uint32_t entry_size = 256;
    constexpr uint32_t num_entries = 4;
    auto pipe = make_pipe(mesh_device.get(), sender_core, receiver_cores, entry_size * num_entries);

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, entry_size, num_entries), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, entry_size, num_entries), 1u);
    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, entry_size, num_entries), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, entry_size, num_entries), 1u);

    mesh_device->clear_loaded_sub_device_manager();
}

// ============================================================================
// Sender write primitives
// ============================================================================

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_BasicPushPop_1to1) {
    if (!is_fast_dispatch()) {
        GTEST_SKIP() << "Sub device managers are unsupported with slow dispatch";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    SubDevice sender_sub_device(std::array{CoreRangeSet(CoreRange(sender_core))});
    SubDevice receiver_sub_device(std::array{receiver_cores});
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, /*local_l1_size=*/0);
    mesh_device->load_sub_device_manager(sub_device_manager);

    auto pipe = make_pipe(mesh_device.get(), sender_core, receiver_cores, 1024);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(
            mesh_device, pipe, 256, 4, /*write_primitive=*/2, /*simultaneous_subdevices=*/true),
        1u);

    mesh_device->clear_loaded_sub_device_manager();
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_WriteBroadcast_1to4) {
    if (!is_fast_dispatch()) {
        GTEST_SKIP() << "Sub device managers are unsupported with slow dispatch";
    }
    if (const auto reason = insufficient_worker_grid_reason(5); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));
    SubDevice sender_sub_device(std::array{CoreRangeSet(CoreRange(sender_core))});
    SubDevice receiver_sub_device(std::array{receiver_cores});
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, /*local_l1_size=*/0);
    mesh_device->load_sub_device_manager(sub_device_manager);

    auto pipe = make_pipe(mesh_device.get(), sender_core, receiver_cores, 1024);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(
            mesh_device, pipe, 256, 4, /*write_primitive=*/0, /*simultaneous_subdevices=*/true),
        4u);

    mesh_device->clear_loaded_sub_device_manager();
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_WriteStrided_1to4) {
    if (!is_fast_dispatch()) {
        GTEST_SKIP() << "Sub device managers are unsupported with slow dispatch";
    }
    if (const auto reason = insufficient_worker_grid_reason(5); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));
    SubDevice sender_sub_device(std::array{CoreRangeSet(CoreRange(sender_core))});
    SubDevice receiver_sub_device(std::array{receiver_cores});
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, /*local_l1_size=*/0);
    mesh_device->load_sub_device_manager(sub_device_manager);

    auto pipe = make_pipe(mesh_device.get(), sender_core, receiver_cores, 1024);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(
            mesh_device, pipe, 256, 4, /*write_primitive=*/1, /*simultaneous_subdevices=*/true),
        4u);

    mesh_device->clear_loaded_sub_device_manager();
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_WriteToReceiver_ReceiverContiguous) {
    if (const auto reason = insufficient_worker_grid_reason(5); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {4, 0})), 1024);
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pipe, 256, 4, /*write_primitive=*/2), 4u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RoundRobinPushBackToReceiver) {
    if (const auto reason = insufficient_worker_grid_reason(5); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {4, 0})), 256);
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pipe, 256, 1, /*write_primitive=*/3), 4u);
}

// Quasar 2x1: every sender write_primitive with 2 DM threads on 1S×1R (tid1 idle when R=1).
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDMSender_AllFlows_2P1C) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-DM PrefetcherPipe sender requires Quasar DM clusters";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t num_entries = 4;
    constexpr uint32_t num_sender_threads = 2;
    // 0=broadcast, 1=strided, 2=receiver-contiguous+push_back, 3=Flow C,
    // 4=decoupled broadcast, 5=entry-major per-receiver credit.
    for (const uint32_t write_primitive : {0u, 1u, 2u, 3u, 4u, 5u}) {
        SCOPED_TRACE("write_primitive=" + std::to_string(write_primitive));
        auto pipe =
            make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * num_entries);
        EXPECT_EQ(
            run_persistent_mt_sender_partition_r(
                mesh_device, pipe, entry_size, num_entries, num_sender_threads, write_primitive),
            1u);
    }
}

// 2 sender DMs is partition-R (needs R>=2). This test is lane credits on 1S×1R: the receiver
// kernel runs P=2 threads (arming two credit lanes when its Program binds), while a single
// sender DM stripes pages_sent across lanes. (Sharing one receiver across sender DMs is deferred.)
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDM_2P2R_1S1R) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-DM PrefetcherPipe requires Quasar DM clusters";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    distributed::MeshDevice& device = *mesh_device;
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t num_entries = 4;
    constexpr uint32_t num_credit_lanes = 2;
    const CoreRangeSet receiver_cores = CoreRangeSet(CoreRange({1, 0}, {1, 0}));

    for (const uint32_t write_primitive : {0u, 1u, 2u, 3u, 4u, 5u}) {
        SCOPED_TRACE("write_primitive=" + std::to_string(write_primitive));
        log_info(tt::LogTest, "MultiDM_2P2R_1S1R: start write_primitive={}", write_primitive);
        auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, entry_size * num_entries);

        // Matching P on both ends, but still use two programs + async SD: a shared Program
        // has hung on RTL sim; async overlap is the proven path for lane-credit MultiDM.
        Program receiver_program = make_receiver_program(device, pipe, entry_size, num_entries, num_credit_lanes);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), num_credit_lanes);
        const SenderKernelParams sender_params{
            .entry_size = entry_size, .num_entries = num_entries, .write_primitive = write_primitive};
        Program sender_program = make_sender_program(device, pipe, sender_params);

        persistent_run_overlapping_programs(mesh_device, std::move(sender_program), std::move(receiver_program));
        log_info(tt::LogTest, "MultiDM_2P2R_1S1R: programs finished write_primitive={}", write_primitive);

        EXPECT_TRUE(prefetcher_pipe_test::verify_receiver_ring(
            device, pipe, CoreCoord(1, 0), sender_params.data_pattern(), entry_size, num_entries, 0, 1));
        log_info(tt::LogTest, "MultiDM_2P2R_1S1R: done write_primitive={}", write_primitive);
    }
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDMSender_PartitionR_2P2C) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-DM PrefetcherPipe sender requires Quasar DM clusters";
    }
    if (const auto reason = insufficient_worker_grid_reason(3); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t num_entries = 4;
    constexpr uint32_t num_sender_threads = 2;
    const CoreRangeSet receivers = CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    // Both sender DMs are active (R=2): all write primitives.
    for (const uint32_t write_primitive : {0u, 1u, 2u, 3u, 4u, 5u}) {
        SCOPED_TRACE("write_primitive=" + std::to_string(write_primitive));
        auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), receivers, entry_size * num_entries);
        EXPECT_EQ(
            run_persistent_mt_sender_partition_r(
                mesh_device, pipe, entry_size, num_entries, num_sender_threads, write_primitive),
            2u);
    }
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDMSender_PartitionR_2P4C) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-DM PrefetcherPipe sender requires Quasar DM clusters";
    }
    if (const auto reason = insufficient_worker_grid_reason(5); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t num_entries = 2;
    constexpr uint32_t num_sender_threads = 2;
    const CoreRangeSet receivers = CoreRangeSet(CoreRange({1, 0}, {4, 0}));
    for (const uint32_t write_primitive : {0u, 1u, 2u, 3u, 4u, 5u}) {
        SCOPED_TRACE("write_primitive=" + std::to_string(write_primitive));
        auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), receivers, entry_size * num_entries);
        EXPECT_EQ(
            run_persistent_mt_sender_partition_r(
                mesh_device, pipe, entry_size, num_entries, num_sender_threads, write_primitive),
            4u);
    }
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_PerReceiverCreditInterleaved_RingDepth4) {
    if (const auto reason = insufficient_worker_grid_reason(3); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {2, 0})), 1024);
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pipe, 256, 4, /*write_primitive=*/5), 2u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_CursorSurvivesCreditCounterWrap) {
    // entries_sent is a free-running uint32, so a cursor derived from it as (sent % ring_units)
    // only survives the 2^32 wrap when ring_units divides 2^32. This ring is 384 KiB = 24576
    // units, which does not. The spin below runs the pipe up to the last lap boundary before the
    // wrap; the data phase then writes and verifies a full ring across it.
    if (MetalContext::instance().rtoptions().get_simulator_enabled()) {
        // 2^32 units of credit is half a million NoC round trips. Under half a second on
        // silicon, six and a half minutes under simulation, where it would be the longest
        // test in the binary by two orders of magnitude. The property is architecture-
        // independent, so the hardware SKUs cover it.
        GTEST_SKIP() << "credit-wrap spin is too slow to be worth its runtime under simulation";
    }
    if (const auto reason = insufficient_worker_grid_reason(3); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 128 * 1024;
    constexpr uint32_t num_entries = 3;
    const CoreRangeSet receiver_cores(CoreRange({1, 0}, {2, 0}));
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, entry_size * num_entries);

    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t entry_units = entry_size / l1_alignment;
    const uint32_t ring_units = entry_units * num_entries;
    ASSERT_NE(ring_units & (ring_units - 1), 0u) << "the wrap is only lossy when ring_units is not a power of two";
    // Stop on a lap boundary: the ring is empty and every cursor is back at its base, which is the
    // state the data phase's verification expects. What is left of the counter is then 2^32 mod
    // ring_units, a whole number of entries, so the wrap falls between two verified entries.
    const uint32_t spin_units = static_cast<uint32_t>((0x100000000ull / ring_units) * ring_units);
    const uint32_t units_to_wrap = static_cast<uint32_t>(0x100000000ull - spin_units);
    ASSERT_EQ(units_to_wrap % entry_units, 0u);
    ASSERT_LT(units_to_wrap / entry_units, num_entries) << "the wrap must leave a verified entry behind it";
    spin_pipe_credits(mesh_device, pipe, entry_size, spin_units / entry_units);

    // Entry `units_to_wrap / entry_units` crosses the wrap; a derived cursor would put the entry
    // after it back at the ring base, on top of the first.
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pipe, entry_size, num_entries, /*write_primitive=*/0), 2u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_DecoupledWriteThenCredit) {
    if (const auto reason = insufficient_worker_grid_reason(5); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {4, 0})), 1024);
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pipe, 256, 4, /*write_primitive=*/4), 4u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_StaleCommitRejected) {
    // After set_entry_size updates word[5] (PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE),
    // commit() with a stale iface.fifo_page_size must not overwrite word[4]
    // (PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT). Push one entry (not a full ring) so the
    // good checkpoint is distinguishable from fifo_start and from the poison wr_ptr.
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    distributed::MeshDevice& device = *mesh_device;
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t new_entry_size = 512;
    constexpr uint32_t num_entries = 4;

    const CoreCoord sender_core(0, 0);
    auto pipe = make_pipe(mesh_device.get(), sender_core, CoreRangeSet(CoreRange({1, 0})), entry_size * num_entries);
    const uint32_t poison_wr_ptr = pipe.buffer_address() + 2 * entry_size;
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(2);

    m2::KernelSpec sender =
        make_dm_kernel("sender", "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_stale_commit.cpp");
    bind_pipe(sender, {pipe_param}, "out");
    sender.compile_time_args = {
        {"entry_size", entry_size}, {"new_entry_size", new_entry_size}, {"poison_wr_ptr", poison_wr_ptr}};
    sender.runtime_arg_schema.runtime_arg_names = {"staging_addr"};
    sender.compiler_options.defines = {{"PREFETCHER_PIPE_TEST_HELPERS", "1"}};
    m2::ProgramSpec spec{
        .name = "pipe_stale_commit",
        .kernels = {sender},
        .work_units = {work_unit("sender_wu", {"sender"}, pipe.sender_cores())},
        .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, entry_size)}},
    };
    Program program = m2::MakeProgramFromSpec(device, spec);
    prefetcher_pipe_test::write_sender_l1_staging(
        device, pipe.sender_cores(), pipe, data_pattern, entry_size, /*num_entries=*/1, 1);
    m2::SetProgramRunArgs(program, pipe_run_args(pipe, {sender_staging_args("sender", pipe)}));

    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);

    const uint32_t expected_checkpoint = pipe.buffer_address() + entry_size;
    std::vector<uint32_t> words(2, 0);
    slow_dispatch::ReadFromL1(
        device,
        sender_core,
        pipe.config_address() + PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT * sizeof(uint32_t),
        std::span<uint8_t>(reinterpret_cast<uint8_t*>(words.data()), 2 * sizeof(uint32_t)),
        CoreType::WORKER);

    // PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT (word[4]) kept the good post-push checkpoint; poison from the stale
    // commit was rejected.
    EXPECT_EQ(words[0], expected_checkpoint);
    EXPECT_NE(words[0], poison_wr_ptr);
    EXPECT_NE(words[0], pipe.buffer_address());
    // PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE (word[5]) reflects the successful resize that created the new epoch.
    EXPECT_EQ(words[1], new_entry_size);
}

// ============================================================================
// Relay DFB: host-side relationship and lane programming
// ============================================================================

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_HostRelationshipValidation) {
    if (const auto reason = insufficient_worker_grid_reason(is_quasar() ? 2 : 3); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreRangeSet receiver_cores =
        is_quasar() ? CoreRangeSet(CoreRange({1, 0}, {1, 0})) : CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    auto pipe = make_pipe(mesh_device.get(), sender_core, receiver_cores, 1024);

    {
        Program program =
            m2::MakeProgramFromSpec(*mesh_device, receiver_program_spec(pipe, {.entry_size = 256, .with_relay = true}));
        const uint32_t relay_host_id = program.impl().get_dfb_handle("relay");
        const auto* relay_dfb = program.impl().get_dataflow_buffer(relay_host_id).get();
        ASSERT_NE(relay_dfb, nullptr);
        EXPECT_TRUE(relay_dfb->config.is_relay);
        const auto prefetcher_pipe_id = program.impl().get_prefetcher_pipe_id_for_relay(relay_host_id);
        ASSERT_TRUE(prefetcher_pipe_id.has_value());
        EXPECT_EQ(*prefetcher_pipe_id, 0u);
        const uint8_t expected_slot = static_cast<uint8_t>(relay_dfb->device_slot);
        for (const CoreCoord& core : corerange_to_cores(receiver_cores)) {
            const auto& participant = program.impl().get_per_core_prefetcher_pipes().at(core).at(0);
            EXPECT_EQ(participant.relay_dfb_id, expected_slot);
        }
        // The relay has no address until the pipe binds; then it aliases the ring.
        EXPECT_EQ(relay_dfb->borrowed_addr_, 0u);
        m2::SetProgramRunArgs(program, pipe_run_args(pipe, {compute_result_args(receiver_cores, 0u)}));
        EXPECT_EQ(relay_dfb->borrowed_addr_, pipe.buffer_address());

        // Sender-only programs carry no relay.
        Program sender_program = make_sender_program(*mesh_device, pipe, {.entry_size = 256});
        EXPECT_EQ(
            sender_program.impl().get_per_core_prefetcher_pipes().at(sender_core).at(0).relay_dfb_id,
            std::numeric_limits<uint8_t>::max());

        // genfiles reads DataflowBufferBindingHandle.prefetcher_pipe_id when emitting
        // RelayDFBBindingToken. Mirror MakeDataflowBufferBindingHandles and verify the callback
        // genfiles uses sees the PrefetcherPipe slot (not the 0xFF default).
        DataflowBufferBindingHandleMap handles;
        handles.emplace(
            "relay_dfb",
            DataflowBufferBindingHandle{
                .logical_dfb_id = static_cast<uint16_t>(relay_dfb->device_slot),
                .is_relay = relay_dfb->config.is_relay,
                .prefetcher_pipe_id = *prefetcher_pipe_id});
        if (is_quasar()) {
            EXPECT_EQ(handles.at("relay_dfb").prefetcher_pipe_id, 0u);
            EXPECT_TRUE(handles.at("relay_dfb").is_relay);
            EXPECT_EQ(handles.at("relay_dfb").logical_dfb_id, expected_slot);
        } else {
            auto kernel = std::make_shared<ComputeKernel>(
                program.impl().get_context_id(),
                KernelSource::from_source("void kernel_main() {}"),
                receiver_cores,
                ComputeConfig{},
                /*is_metal2_kernel=*/true,
                handles);
            bool saw_binding = false;
            kernel->process_dataflow_buffer_binding_handles(
                [&](const std::string& name,
                    uint16_t logical_id,
                    bool is_relay,
                    uint8_t prefetcher_pipe_id,
                    const std::optional<LLKMetadata>&) {
                    EXPECT_EQ(name, "relay_dfb");
                    EXPECT_EQ(logical_id, expected_slot);
                    EXPECT_TRUE(is_relay);
                    EXPECT_EQ(prefetcher_pipe_id, 0u);
                    saw_binding = true;
                });
            EXPECT_TRUE(saw_binding);
        }
    }

    {
        // Relay entry size must match the pipe parameter's.
        m2::ProgramSpec spec = receiver_program_spec(pipe, {.entry_size = 256, .with_relay = true});
        spec.dataflow_buffers[0].entry_size = 128;
        spec.dataflow_buffers[0].num_entries = 8;
        EXPECT_THROW(m2::MakeProgramFromSpec(*mesh_device, spec), std::exception);
    }

    {
        // A relay whose producer does not bind the relayed pipe is rejected.
        m2::ProgramSpec spec = receiver_program_spec(pipe, {.entry_size = 256, .with_relay = true});
        spec.kernels[0].advanced_options.prefetcher_pipe_bindings.clear();
        EXPECT_THROW(m2::MakeProgramFromSpec(*mesh_device, spec), std::exception);
    }
}

// Host-only: the receiver kernel's num_threads (with or without a relay) arms the pipe's lane
// count, which dispatch packs into each program's kernel-config slot; over-capacity and
// reprogram-to-different-P are rejected.
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_CreditLanesHostProgramming) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Lane-credit capacity > 1 is Quasar-only";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(1, 0);
    const CoreRangeSet receiver_cores = CoreRangeSet(CoreRange(receiver_core));
    const auto new_pipe = [&] { return make_pipe(mesh_device.get(), sender_core, receiver_cores, /*ring_size=*/1024); };

    {
        // Arm lanes with a 2-thread receiver kernel, no relay.
        auto pipe = new_pipe();
        EXPECT_EQ(pipe.impl().credit_lane_capacity(), PREFETCHER_PIPE_MAX_CREDIT_LANES);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);

        Program receiver_program =
            m2::MakeProgramFromSpec(*mesh_device, receiver_program_spec(pipe, {.num_threads = 2}));
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);  // reserved, not yet bound
        m2::SetProgramRunArgs(receiver_program, pipe_run_args(pipe));
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
        // P is not in the persistent page (word[9] stays reserved); it travels in the program's
        // kernel-config slot, packed above relay_dfb_id, on sender and receiver cores alike.
        Program sender_program = make_sender_program(*mesh_device, pipe, {.entry_size = 256});
        for (const CoreCoord core : {sender_core, receiver_core}) {
            EXPECT_EQ(pipe.impl().config_page(core)[9], 0u);
        }
        const uint32_t recv_word = slot_relay_word(receiver_program, receiver_core);
        EXPECT_EQ(prefetcher_pipe_slot_credit_lanes(recv_word), 2u);
        EXPECT_EQ(prefetcher_pipe_slot_relay_id(recv_word), std::numeric_limits<uint8_t>::max());
        const uint32_t send_word = slot_relay_word(sender_program, sender_core);
        EXPECT_EQ(prefetcher_pipe_slot_credit_lanes(send_word), 2u);
        EXPECT_EQ(prefetcher_pipe_slot_relay_id(send_word), std::numeric_limits<uint8_t>::max());

        // A single-lane pipe's slot is bit-identical to the pre-lane encoding.
        auto pipe_single = new_pipe();
        Program program_single = make_receiver_program(*mesh_device, pipe_single, 256, 4);
        EXPECT_EQ(
            slot_relay_word(program_single, receiver_core), static_cast<uint32_t>(std::numeric_limits<uint8_t>::max()));
    }

    {
        // Lanes armed through a relay: the relay's num_producers is the receiver kernel's thread
        // count. The receiver slot carries the relay's device slot; the sender's does not.
        auto pipe = new_pipe();
        Program receiver_program = m2::MakeProgramFromSpec(
            *mesh_device, receiver_program_spec(pipe, {.num_threads = 2, .with_relay = true, .num_consumers = 2}));
        m2::SetProgramRunArgs(receiver_program, pipe_run_args(pipe, {compute_result_args(receiver_cores, 0u)}));
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
        const uint32_t recv_word = slot_relay_word(receiver_program, receiver_core);
        EXPECT_EQ(prefetcher_pipe_slot_credit_lanes(recv_word), 2u);
        EXPECT_EQ(
            prefetcher_pipe_slot_relay_id(recv_word),
            receiver_program.impl().get_dataflow_buffer(receiver_program.impl().get_dfb_handle("relay"))->device_slot);
        Program sender_program = make_sender_program(*mesh_device, pipe, {.entry_size = 256});
        const uint32_t send_word = slot_relay_word(sender_program, sender_core);
        EXPECT_EQ(prefetcher_pipe_slot_credit_lanes(send_word), 2u);
        EXPECT_EQ(prefetcher_pipe_slot_relay_id(send_word), std::numeric_limits<uint8_t>::max());
    }

    {
        // Lanes are armed once per pipe lifetime: a second Program whose receiver runs a
        // different thread count is rejected at bind and leaves the pipe as it was.
        auto pipe = new_pipe();
        Program armed = make_receiver_program(*mesh_device, pipe, 256, 4, /*num_threads=*/2);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
        Program three = m2::MakeProgramFromSpec(*mesh_device, receiver_program_spec(pipe, {.num_threads = 4}));
        EXPECT_THROW(m2::SetProgramRunArgs(three, pipe_run_args(pipe)), std::exception);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
        Program one = m2::MakeProgramFromSpec(*mesh_device, receiver_program_spec(pipe, {.num_threads = 1}));
        EXPECT_THROW(m2::SetProgramRunArgs(one, pipe_run_args(pipe)), std::exception);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
    }

    {
        // More lanes than the page reserves is rejected when the Program is built.
        auto pipe = new_pipe();
        EXPECT_THROW(
            m2::MakeProgramFromSpec(
                *mesh_device, receiver_program_spec(pipe, {.num_threads = PREFETCHER_PIPE_MAX_CREDIT_LANES + 1})),
            std::exception);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);
    }

    {
        // Lane mode needs an exact entry ring with an entry count divisible by P; a rejected
        // Program must not leave the persistent pipe armed.
        auto pipe = new_pipe();
        // 1024 % 384 != 0: trailing gap would never be credited with striped lanes.
        EXPECT_THROW(
            m2::MakeProgramFromSpec(*mesh_device, receiver_program_spec(pipe, {.entry_size = 384, .num_threads = 2})),
            std::exception);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);
        // 4 entries is not a multiple of 3 lanes.
        EXPECT_THROW(
            m2::MakeProgramFromSpec(*mesh_device, receiver_program_spec(pipe, {.entry_size = 256, .num_threads = 3})),
            std::exception);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);
        EXPECT_NO_THROW(make_receiver_program(*mesh_device, pipe, 256, 4, /*num_threads=*/2));
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
    }

    {
        // Multi-producer ALL relay: the DFB must be serialized lane-interleaved (stride P) so
        // producer h's TC walks the entries pipe lane h receives (h, h+P, ...), not a
        // contiguous per-producer block. A standalone ALL DFB keeps stride 1.
        auto pipe = new_pipe();
        m2::ProgramSpec spec = receiver_program_spec(
            pipe, {.num_threads = 2, .with_relay = true, .num_consumers = 2, .cap = m2::DFBAccessPattern::ALL});
        const m2::DFBSpecName plain_name{"plain"};
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = plain_name,
            .entry_size = 256,
            .num_entries = 4,
            .data_format_metadata = tt::DataFormat::Float16_b,
        });
        spec.kernels[0].dfb_bindings.push_back(m2::ProducerOf(plain_name, "plain"));
        spec.kernels[1].dfb_bindings.push_back(m2::AllConsumerOf(plain_name, "plain"));
        Program program = m2::MakeProgramFromSpec(*mesh_device, spec);
        const auto relay = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("relay"));
        EXPECT_EQ(relay->stride_in_entries, 2u);
        EXPECT_EQ(relay->capacity, 2u);
        const auto plain = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("plain"));
        EXPECT_EQ(plain->stride_in_entries, 1u);
    }
}

// ============================================================================
// Relay DFB: end to end (pipe -> receiver DM -> relay -> TRISC)
// ============================================================================

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_CrossProgram_DMToCompute) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    constexpr uint32_t total_entries = 4;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = total_entries,
                .batch_size = 1,
            }),
        1u);
}

// One Program: sender DM on (0,0), receiver DM + TRISC on (1,0) joined by a relay DFB, with
// more entries than the ring holds so the sender wraps under live backpressure.
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_SenderRelayReceiver_SameProgram_1S1R) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    constexpr uint32_t total_entries = 8;
    const CoreCoord receiver_core(1, 0);
    auto pipe =
        make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange(receiver_core)), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = total_entries,
                .batch_size = 1,
                .same_program = true,
            }),
        1u);
    // After total_entries pushes the ring's slots hold the last ring_depth entries.
    EXPECT_TRUE(prefetcher_pipe_test::verify_receiver_ring(
        *mesh_device,
        pipe,
        receiver_core,
        cross_node_dfb_test::data_pattern_for_write_primitive(0),
        entry_size,
        ring_depth,
        0,
        1,
        total_entries - ring_depth));
}

// Quasar multi-TC relay matrix (PrefetcherPipe credits remain single-DM-owned).
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_Parallel_STRIDED_1P2C) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-consumer PrefetcherPipe relay uses Quasar DFB TC slots";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 4,
                .batch_size = 1,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::STRIDED,
            }),
        1u);
}

// 1 sender Tensix × 1 receiver Tensix: 2 sender DMs (Flow A) + 2 TRISC relay consumers.
// Pipe receiver DM stays single-threaded (relay num_producers=1); tid0 owns wait/pop.
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDMSender_2P_Relay_STRIDED_2C_1S1R) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-DM PrefetcherPipe sender + multi-TC relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 4,
                .batch_size = 1,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::STRIDED,
                .num_sender_threads = 2,
            }),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDMSender_2P_Relay_ALL_2C_1S1R) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-DM PrefetcherPipe sender + multi-TC relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 4,
                .batch_size = 1,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::ALL,
                .num_sender_threads = 2,
            }),
        1u);
}

// Full stack on 1S×1R: 2 sender DMs + 2 pipe-consumer/relay-producer DMs + 2 TRISC
// relay consumers. Pipe consumers own STRIDED entries and publish into the relay.
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDM_2P_RelayProducers2_Consumers2_1S1R) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-producer PrefetcherPipe relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 4,
                .batch_size = 1,
                .num_producers = 2,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::STRIDED,
                .num_sender_threads = 2,
            }),
        1u);
}

// Lane credits: multi-producer pipe consumers batch owned strides (batch_size=2).
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDM_2P_RelayProducers2_Consumers2_Batch2) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-producer PrefetcherPipe relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 8;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 2,
                .num_producers = 2,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::STRIDED,
                .num_sender_threads = 2,
            }),
        1u);
}

// Multi-producer relay into ALL consumers (each TRISC sees every entry). The relay DFB is
// serialized lane-interleaved so producer h's TC follows pipe lane h (h, h+P, ...).
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDM_2P_RelayProducers2_Consumers2_ALL_1S1R) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-producer PrefetcherPipe relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 4,
                .batch_size = 1,
                .num_producers = 2,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::ALL,
                .num_sender_threads = 2,
            }),
        1u);
}

// 2 sender DMs + 2 relay producers + 4 TRISC STRIDED consumers on 1S×1R.
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDM_2P_RelayProducers2_Consumers4_1S1R) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-producer PrefetcherPipe relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 8;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 1,
                .num_producers = 2,
                .num_consumers = 4,
                .cap = m2::DFBAccessPattern::STRIDED,
                .num_sender_threads = 2,
            }),
        1u);
}

// Lane credits at capacity: 2 sender DMs + 4 pipe-consumer/relay-producer DMs + 4 TRISC.
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDM_2P_RelayProducers4_Consumers4_ALL_1S1R) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-producer PrefetcherPipe relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 8;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(pipe.impl().credit_lane_capacity(), PREFETCHER_PIPE_MAX_CREDIT_LANES);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 1,
                .num_producers = 4,
                .num_consumers = 4,
                .cap = m2::DFBAccessPattern::ALL,
                .num_sender_threads = 2,
            }),
        1u);
    EXPECT_EQ(pipe.impl().num_credit_lanes(), 4u);
}

// Lane credits: P=4 pipe consumers batch owned strides (batch_size=2).
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDM_2P_RelayProducers4_Consumers4_Batch2) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-producer PrefetcherPipe relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 8;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 2,
                .num_producers = 4,
                .num_consumers = 4,
                .cap = m2::DFBAccessPattern::STRIDED,
                .num_sender_threads = 2,
            }),
        1u);
}

// Same-program: 2 sender DMs + single pipe-consumer DM + 2 TRISC with delay (credit stall).
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDMSender_2P_Relay_STRIDED_2C_Backpressure) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-DM PrefetcherPipe sender + multi-TC relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 2;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 1,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::STRIDED,
                .trisc_delay_iterations = 50000,
                .same_program = true,
                .num_sender_threads = 2,
            }),
        1u);
}

// Same-program full stack under TRISC delay: 2 sender DMs + 2 relay producers + 2 TRISC.
// ring_depth must be > num_credit_lanes so each lane has >1 slot (depth=2 with P=2
// is a single-slot lane and same-program overwrite races TRISC reads).
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDM_RelayProducers2_Consumers2_Backpressure) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-producer PrefetcherPipe relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 1,
                .num_producers = 2,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::STRIDED,
                .trisc_delay_iterations = 50000,
                .same_program = true,
                .num_sender_threads = 2,
            }),
        1u);
}

// 2 sender DMs + batch_size=2 STRIDED TRISC consumers (num_producers=1).
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiDMSender_2P_Relay_STRIDED_2C_Batch2) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-DM PrefetcherPipe sender + multi-TC relay requires Quasar";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 4,
                .batch_size = 2,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::STRIDED,
                .num_sender_threads = 2,
            }),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_Parallel_STRIDED_1P4C) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-consumer PrefetcherPipe relay uses Quasar DFB TC slots";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 8;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 1,
                .num_consumers = 4,
                .cap = m2::DFBAccessPattern::STRIDED,
            }),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_Parallel_ALL_1P2C) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-consumer PrefetcherPipe relay uses Quasar DFB TC slots";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 4,
                .batch_size = 1,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::ALL,
            }),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_Parallel_ALL_1P4C) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-consumer PrefetcherPipe relay uses Quasar DFB TC slots";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 8;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 1,
                .num_consumers = 4,
                .cap = m2::DFBAccessPattern::ALL,
            }),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_Parallel_STRIDED_1P2C_Backpressure) {
    // Re-enabled after #55938 (wait_front blocks until WAIT_TILES resolves). Previously
    // Neo1's first owned entry often read as 0 under same-program STRIDED + TRISC delay.
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-consumer PrefetcherPipe relay uses Quasar DFB TC slots";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 2;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 1,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::STRIDED,
                .trisc_delay_iterations = 50000,
                .same_program = true,
            }),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_Parallel_ALL_1P2C_Backpressure) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-consumer PrefetcherPipe relay uses Quasar DFB TC slots";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 2;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 1,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::ALL,
                .trisc_delay_iterations = 1000,
                .same_program = true,
            }),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_Parallel_STRIDED_1P2C_Batch2) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Multi-consumer PrefetcherPipe relay uses Quasar DFB TC slots";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 4,
                .batch_size = 2,
                .num_consumers = 2,
                .cap = m2::DFBAccessPattern::STRIDED,
            }),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_Backpressure_NoOverwrite) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 2;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), entry_size * ring_depth);
    EXPECT_EQ(
        run_prefetcher_pipe_relay(
            mesh_device,
            pipe,
            PrefetcherPipeRelayParams{
                .entry_size = entry_size,
                .ring_depth = ring_depth,
                .total_entries = 8,
                .batch_size = 1,
                .trisc_delay_iterations = 1000,
                .same_program = true,
            }),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_CrossProgram_DifferentEntrySize) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    // Full drain: A finishes on E1, then B binds the pipe at E2 with a matching relay.
    auto mesh_device = devices_[0];
    constexpr uint32_t e1 = 256;
    constexpr uint32_t e2 = 512;
    constexpr uint32_t ring_depth_e1 = 4;  // ring bytes = 1024 → 2 entries at E2
    constexpr uint32_t total_entries_e1 = 2;
    auto pipe = make_pipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), e1 * ring_depth_e1);
    EXPECT_EQ(
        run_prefetcher_pipe_relay_cross_program(
            mesh_device, pipe, e1, ring_depth_e1, total_entries_e1, /*batch_size=*/1, e2),
        1u);
}

// ============================================================================
// Coordinated live-peer resize
// ============================================================================

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_CrossSubDevice_CoordinatedLivePeerNonDividingE2) {
    if (!is_fast_dispatch()) {
        GTEST_SKIP() << "Sub device managers are unsupported with slow dispatch";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    // Live-peer E1→E2 prefetch with a non-dividing E2:
    //   A (SD0): push E1 → set_entry_size(E2) without draining → signal → wait go → push E2
    //   B (SD1): bound at E1, pop E1 → set_receiver_entry_size(E2), finish
    //   C (SD1): bound at E2 while A is still alive; consume E2
    //
    // A reaches set_entry_size before B is even enqueued, proving resize itself does
    // not wait for E1 acknowledgements. E2=384 has a 768-byte usable limit in the
    // 1024-byte allocation; the second E2 push credits the 256-byte trailing gap.
    auto mesh_device = devices_[0];
    distributed::MeshDevice& device = *mesh_device;
    constexpr uint32_t e1 = 256;
    constexpr uint32_t e2 = 384;
    constexpr uint32_t ring_depth_e1 = 4;  // 1024 bytes → 768 usable bytes / 2 entries at E2
    constexpr uint32_t total_entries_e1 = 4;
    constexpr uint32_t total_entries_e2 = 2;
    constexpr uint32_t data_pattern = 0;  // multicast counter

    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(1, 0);
    const CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreRangeSet receiver_cores = CoreRangeSet(CoreRange(receiver_core));

    SubDevice sender_sub_device(std::array{sender_cores});
    SubDevice receiver_sub_device(std::array{receiver_cores});
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, /*local_l1_size=*/0);
    mesh_device->load_sub_device_manager(sub_device_manager);

    // Semaphores are allocated top-down before the PrefetcherPipe. The test-only
    // sender staging scratch is placed immediately above the persistent arena.
    auto resized_sem = CreateGlobalSemaphore(*mesh_device, sender_cores, /*initial_value=*/0);
    auto go_sem = CreateGlobalSemaphore(*mesh_device, sender_cores, /*initial_value=*/0);

    auto pipe = make_pipe(mesh_device.get(), sender_core, receiver_cores, e1 * ring_depth_e1);
    distributed::Synchronize(*mesh_device, std::nullopt);

    // --- Program A (SD0): push E1 → resize without drain → signal → wait go → push E2 ---
    m2::KernelSpec sender_kernel = make_dm_kernel(
        "sender", "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_coordinated_resize_sender.cpp");
    bind_pipe(sender_kernel, {pipe_param}, "out");
    sender_kernel.compile_time_args = {
        {"entry_size_e1", e1},
        {"num_entries_e1", total_entries_e1},
        {"entry_size_e2", e2},
        {"num_entries_e2", total_entries_e2}};
    sender_kernel.runtime_arg_schema.runtime_arg_names = {
        "staging_addr", "resized_sem_addr", "go_sem_addr", "credit_base_addr"};
    m2::ProgramSpec spec_a{
        .name = "resize_sender",
        .kernels = {sender_kernel},
        .work_units = {work_unit("sender_wu", {"sender"}, sender_cores)},
        .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, e1)}},
    };
    Program program_a = m2::MakeProgramFromSpec(device, spec_a);
    prefetcher_pipe_test::write_sender_l1_staging(
        device,
        sender_cores,
        pipe,
        data_pattern,
        e1,
        total_entries_e1,
        1,
        /*counter_base=*/0,
        e2,
        total_entries_e2);
    const uint32_t credit_base = pipe.config_address() + pipe.credit_reset_offset();
    m2::SetProgramRunArgs(
        program_a,
        pipe_run_args(
            pipe,
            {m2::ProgramRunArgs::KernelRunArgs{
                .kernel = m2::KernelSpecName{"sender"},
                .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(
                    sender_core,
                    {{"staging_addr", prefetcher_pipe_test::sender_l1_staging_address(pipe)},
                     {"resized_sem_addr", static_cast<uint32_t>(resized_sem.address())},
                     {"go_sem_addr", static_cast<uint32_t>(go_sem.address())},
                     {"credit_base_addr", credit_base}})}}));

    distributed::MeshWorkload workload_a;
    workload_a.add_program(persistent_unit_mesh_device_range(), std::move(program_a));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_a, false);

    // Subsequent FD work waits for SD1 idle (not long-running A on SD0).
    mesh_device->set_sub_device_stall_group({{SubDeviceId{1}}});

    // A must resize before any receiver is enqueued. This would time out if
    // set_entry_size still contained an acked == sent barrier.
    const auto device_id = mesh_device->get_device_ids()[0];
    const auto physical_sender = mesh_device->worker_core_from_logical_core(sender_core);
    bool resized = false;
    for (uint32_t i = 0; i < 10000; ++i) {
        const auto sem_vals = MetalContext::instance().get_cluster().read_core(
            device_id, physical_sender, resized_sem.address(), sizeof(uint32_t));
        if (!sem_vals.empty() && sem_vals[0] == 1u) {
            resized = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto [sender_sent, sender_acked] = prefetcher_pipe_test::read_pipe_credits(device, pipe, sender_core);
    ASSERT_TRUE(resized) << "Timed out waiting for barrier-free set_entry_size(E2); sender credits=" << sender_sent
                         << "/" << sender_acked;
    // Quasar: local pages_sent is updated via cached stores; mid-run host TL1 peeks of the
    // sender slot often still see 0 even after an L2 flush on emu. The receiver slot is
    // incremented by NOC atomics into TL1, so it is the reliable undrained-E1 probe.
    if (is_quasar()) {
        const auto [recv_sent, recv_acked] = prefetcher_pipe_test::read_pipe_credits(device, pipe, receiver_core);
        EXPECT_LT(recv_acked, recv_sent)
            << "E1 unexpectedly drained before its receiver program was enqueued; receiver credits=" << recv_sent << "/"
            << recv_acked << " sender credits=" << sender_sent << "/" << sender_acked;
    } else {
        EXPECT_LT(sender_acked, sender_sent) << "E1 unexpectedly drained before its receiver program was enqueued";
    }

    // --- Program B (SD1): consume E1, then consume resize pad credits at E2 ---
    m2::KernelSpec receiver_b = make_dm_kernel(
        "receiver", "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_coordinated_resize_receiver.cpp");
    bind_pipe(receiver_b, {pipe_param}, "in");
    receiver_b.compile_time_args = {{"num_entries_e1", total_entries_e1}, {"entry_size_e2", e2}};
    m2::ProgramSpec spec_b{
        .name = "resize_receiver",
        .kernels = {receiver_b},
        .work_units = {work_unit("receiver_wu", {"receiver"}, receiver_cores)},
        .advanced_options = {.prefetcher_pipe_parameters = {pipe_parameter(pipe_param, pipe, e1)}},
    };
    Program program_b = m2::MakeProgramFromSpec(device, spec_b);
    m2::SetProgramRunArgs(program_b, pipe_run_args(pipe));
    distributed::MeshWorkload workload_b;
    workload_b.add_program(persistent_unit_mesh_device_range(), std::move(program_b));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_b, false);

    // --- Program C (SD1): same-epoch bind at E2 while A is still alive ---
    Program program_c = make_receiver_program(device, pipe, e2, total_entries_e2);
    distributed::MeshWorkload workload_c;
    workload_c.add_program(persistent_unit_mesh_device_range(), std::move(program_c));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_c, false);

    // Release A to push at E2; C is already bound/running on the peer sub-device.
    MetalContext::instance().get_cluster().write_core(
        device_id, physical_sender, std::vector<uint32_t>{1}, go_sem.address());

    mesh_device->reset_sub_device_stall_group();
    distributed::Finish(mesh_device->mesh_command_queue());

    // E2 entries begin at checkpoint offset 0. Their expected counter values continue
    // after the four E1 entries while the final E2 push also advances over the ring gap.
    EXPECT_TRUE(prefetcher_pipe_test::verify_receiver_ring(
        device,
        pipe,
        receiver_core,
        data_pattern,
        e2,
        total_entries_e2,
        /*receiver_idx=*/0,
        /*num_receivers=*/1,
        /*counter_base=*/total_entries_e1));

    mesh_device->clear_loaded_sub_device_manager();
}

// ============================================================================
// Multi-pipe accessor: one kernel binary, several pipes resolved per node
// ============================================================================

// Two pipes from one space, one accessor per side, two Programs. Sender Program: one DM kernel on
// both sender nodes whose accessor "out" names {a, b}; each node pushes into its own pipe.
// Receiver Program: one DM kernel on both receiver nodes whose accessor "in" names {a, b},
// relaying to TRISC through one DFB that names both pipes. The space guarantees the two pipes
// share a ring address, which is what lets one relay DFB alias both.
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_MultiPipeAccessor_CrossProgram_2S2R) {
    if (const auto reason = insufficient_worker_grid_reason(2, 2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    if (is_fast_dispatch()) {
        GTEST_SKIP() << "Two-program overlap uses asynchronous slow dispatch";
    }
    auto mesh_device = devices_[0];
    distributed::MeshDevice& device = *mesh_device;

    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    constexpr uint32_t ring_size = entry_size * ring_depth;
    constexpr uint32_t total_entries = 8;
    const CoreCoord sender_a(0, 0);
    const CoreCoord sender_b(1, 0);
    const CoreCoord receiver_a(0, 1);
    const CoreCoord receiver_b(1, 1);
    const CoreRangeSet receivers_a = CoreRangeSet(CoreRange(receiver_a));
    const CoreRangeSet receivers_b = CoreRangeSet(CoreRange(receiver_b));
    const CoreRangeSet sender_cores(CoreRange(sender_a, sender_b));
    const CoreRangeSet receiver_cores(CoreRange(receiver_a, receiver_b));

    auto space = m2::CreatePrefetcherPipeSpace(
        *mesh_device,
        m2::PrefetcherPipeSpaceConfig{
            .sender_cores = sender_cores,
            .receiver_domain = receiver_cores,
            .ring_size = ring_size,
            .max_receivers_per_pipe = 1,
        });
    const std::vector<std::pair<CoreCoord, CoreRangeSet>> mappings = {{sender_a, receivers_a}, {sender_b, receivers_b}};
    std::vector<m2::PrefetcherPipe> pipes = space.create_pipes(mappings);
    ASSERT_EQ(pipes.size(), 2u);
    m2::PrefetcherPipe& pipe_a = pipes[0];
    m2::PrefetcherPipe& pipe_b = pipes[1];
    ASSERT_EQ(pipe_a.buffer_address(), pipe_b.buffer_address());
    auto result_buffer = cross_node_dfb_test::make_cross_node_data_buffer(device, receiver_cores, 32, 1);
    const uint32_t result_addr = static_cast<uint32_t>(result_buffer->address());

    const m2::PrefetcherPipeParamName name_a{"a"};
    const m2::PrefetcherPipeParamName name_b{"b"};
    const std::vector<m2::PrefetcherPipeParameter> pipe_params = {
        pipe_parameter(name_a, pipe_a, entry_size),
        pipe_parameter(name_b, pipe_b, entry_size),
    };
    const auto two_pipe_args = [&](std::vector<m2::ProgramRunArgs::KernelRunArgs> kernels) {
        m2::ProgramRunArgs args;
        args.kernel_run_args = std::move(kernels);
        args.advanced_options.prefetcher_pipe_args = {
            {name_a, m2::PrefetcherPipeArgument{pipe_a}}, {name_b, m2::PrefetcherPipeArgument{pipe_b}}};
        return args;
    };

    // Receiver Program first: binding its receivers arms the pipes' lanes before any sender runs.
    RelayConsumerSpecs consumer = make_relay_consumer(
        {name_a, name_b},
        RelayConsumerParams{
            .entry_size = entry_size,
            .ring_depth = ring_depth,
            .pipe_total_entries = total_entries,
            .entries_per_consumer = total_entries,
        });
    m2::ProgramSpec receiver_spec{
        .name = "pipe_2s2r_receivers",
        .kernels = {consumer.receiver, consumer.compute},
        .dataflow_buffers = {consumer.relay},
        .work_units = {work_unit("receiver_wu", {"receiver", "compute"}, receiver_cores)},
        .advanced_options = {.prefetcher_pipe_parameters = pipe_params},
    };
    Program receiver_program = m2::MakeProgramFromSpec(device, receiver_spec);
    EXPECT_EQ(receiver_program.impl().num_prefetcher_pipe_slots(), 1u);
    m2::SetProgramRunArgs(receiver_program, two_pipe_args({compute_result_args(receiver_cores, result_addr)}));
    // Each receiver node's slot record resolved to the pipe present there.
    {
        const auto& per_core = receiver_program.impl().get_per_core_prefetcher_pipes();
        ASSERT_EQ(per_core.at(receiver_a).size(), 1u);
        ASSERT_EQ(per_core.at(receiver_b).size(), 1u);
        EXPECT_EQ(per_core.at(receiver_a)[0].pipe, &pipe_a.impl());
        EXPECT_EQ(per_core.at(receiver_b)[0].pipe, &pipe_b.impl());
    }

    const SenderKernelParams sender_params{
        .entry_size = entry_size, .num_entries = total_entries, .write_primitive = 0};
    m2::ProgramSpec sender_spec{
        .name = "pipe_2s2r_senders",
        .kernels = {sender_kernel_spec("sender", {name_a, name_b}, sender_params)},
        .work_units = {work_unit("sender_wu", {"sender"}, sender_cores)},
        .advanced_options = {.prefetcher_pipe_parameters = pipe_params},
    };
    Program sender_program = m2::MakeProgramFromSpec(device, sender_spec);
    EXPECT_EQ(sender_program.impl().num_prefetcher_pipe_slots(), 1u);

    const uint32_t data_pattern = sender_params.data_pattern();
    prefetcher_pipe_test::write_sender_l1_staging(
        device, pipe_a.sender_cores(), pipe_a, data_pattern, entry_size, total_entries, 1);
    prefetcher_pipe_test::write_sender_l1_staging(
        device, pipe_b.sender_cores(), pipe_b, data_pattern, entry_size, total_entries, 1);
    {
        m2::ProgramRunArgs::KernelRunArgs sender_args{.kernel = m2::KernelSpecName{"sender"}};
        m2::AddRuntimeArgsForNode(
            sender_args.runtime_arg_values,
            sender_a,
            {{"staging_addr", prefetcher_pipe_test::sender_l1_staging_address(pipe_a)}});
        m2::AddRuntimeArgsForNode(
            sender_args.runtime_arg_values,
            sender_b,
            {{"staging_addr", prefetcher_pipe_test::sender_l1_staging_address(pipe_b)}});
        m2::SetProgramRunArgs(sender_program, two_pipe_args({std::move(sender_args)}));
    }

    persistent_run_overlapping_programs(mesh_device, std::move(sender_program), std::move(receiver_program));

    EXPECT_TRUE(prefetcher_pipe_test::verify_receiver_ring(
        device, pipe_a, receiver_a, data_pattern, entry_size, ring_depth, 0, 1, total_entries - ring_depth));
    EXPECT_TRUE(prefetcher_pipe_test::verify_receiver_ring(
        device, pipe_b, receiver_b, data_pattern, entry_size, ring_depth, 0, 1, total_entries - ring_depth));
    EXPECT_EQ(
        verify_relay_results(
            device, receiver_cores, result_addr, 1, m2::DFBAccessPattern::STRIDED, total_entries, total_entries),
        2u);
}

}  // namespace tt::tt_metal
