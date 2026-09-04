// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
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

#include "impl/dataflow_buffer/cross_node_dfb.hpp"
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/dispatch.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/host_api/temp_quasar_api.hpp"
#include "mesh_dispatch_fixture.hpp"
#include "tests/tt_metal/tt_metal/api/cross_node_dfb_test_utils.hpp"
#include "hostdev/remote_dfb_config_layout.h"
#include "hostdev/remote_dfb_constants.h"
#include "tests/tt_metal/tt_metal/api/prefetcher_pipe_test_utils.hpp"

namespace tt::tt_metal {

class PrefetcherPipeFixture : public MeshDispatchFixture {
protected:
    void SetUp() override { MeshDispatchFixture::SetUp(); }

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
};

namespace {

bool is_quasar_arch() { return MetalContext::instance().get_cluster().arch() == tt::ARCH::QUASAR; }

KernelHandle create_dm_kernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::vector<uint32_t>& compile_args,
    const std::map<std::string, std::string>& defines = {},
    uint32_t num_threads_per_cluster = 1) {
    if (is_quasar_arch()) {
        return experimental::quasar::CreateKernel(
            program,
            file_name,
            core_spec,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = num_threads_per_cluster,
                .compile_args = compile_args,
                .defines = defines,
                .is_legacy_kernel = true,
            });
    }
    TT_FATAL(num_threads_per_cluster == 1, "Non-Quasar PrefetcherPipe tests only support 1 DM thread");
    return CreateKernel(
        program,
        file_name,
        core_spec,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_args,
            .defines = defines,
        });
}

KernelHandle create_compute_kernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::vector<uint32_t>& compile_args,
    uint32_t num_threads_per_cluster = 1) {
    if (is_quasar_arch()) {
        return experimental::quasar::CreateKernel(
            program,
            file_name,
            core_spec,
            experimental::quasar::QuasarComputeConfig{
                .num_threads_per_cluster = num_threads_per_cluster,
                .compile_args = compile_args,
            });
    }
    TT_FATAL(num_threads_per_cluster == 1, "Non-Quasar PrefetcherPipe tests only support 1 compute thread");
    return CreateKernel(program, file_name, core_spec, ComputeConfig{.compile_args = compile_args});
}

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

// Overlap producer/consumer across two programs (Metal2 cross-program shape). Prefer a
// single Program when both ends can share one LaunchProgram — more reliable on RTL sim.
// Default SD Finish only tracks the last enqueue's cores; async SD merges the wait set. Fast
// dispatch already queues both programs and Finish waits for everything, and the toggle is
// SD-only (it fatals under FD), so only switch it on for slow dispatch.
void persistent_run_overlapping_programs(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, Program first_program, Program second_program) {
    const bool use_async_slow_dispatch = !MetalContext::instance().rtoptions().get_fast_dispatch();
    if (use_async_slow_dispatch) {
        experimental::DispatchContext::get().enable_asynchronous_slow_dispatch(mesh_device.get());
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
    if (use_async_slow_dispatch) {
        experimental::DispatchContext::get().disable_asynchronous_slow_dispatch(mesh_device.get());
    }
}

// Run `num_pushes` entries of credit through the pipe with no payload, so both endpoints reach
// a counter state that would otherwise take that many entries of real traffic. Producer and
// consumer sit in one program so they run at the same time: the producer can only get a ring
// ahead, so this costs one NoC round trip per lap. Every counter and cursor moves the way it
// does under real traffic, because it is the same credit path that moves it.
void spin_pipe_credits(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_pushes) {
    Program program = CreateProgram();
    EXPECT_EQ(AttachPrefetcherPipe(program, pipe, pipe.all_cores(), entry_size), 0u);
    const auto spin_kernel = [&](const CoreRangeSet& cores, uint32_t is_sender) {
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_credit_spin.cpp",
            cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {0u, num_pushes, 1u, is_sender}});
    };
    spin_kernel(pipe.sender_cores(), 1u);
    spin_kernel(pipe.receiver_cores(), 0u);

    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);
}

uint32_t run_persistent_sender_push(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_entries,
    uint8_t prefetcher_pipe_id) {
    distributed::MeshDevice& device = *mesh_device;
    const CoreRangeSet sender_cores = pipe.sender_cores();
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(2);

    Program program = CreateProgram();
    // Attach only sender cores — PrefetcherPipe is cross-program; this program owns the producer role.
    EXPECT_EQ(AttachPrefetcherPipe(program, pipe, sender_cores, entry_size), prefetcher_pipe_id);
    KernelHandle sender_k = create_dm_kernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_sender.cpp",
        sender_cores,
        {prefetcher_pipe_id, entry_size, num_entries, 2u, data_pattern, 0u});
    prefetcher_pipe_test::write_sender_l1_staging(device, sender_cores, pipe, data_pattern, entry_size, num_entries, 1);
    prefetcher_pipe_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, pipe);
    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);
    return 1u;
}

uint32_t run_persistent_receiver_pop(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_entries,
    uint8_t prefetcher_pipe_id) {
    distributed::MeshDevice& device = *mesh_device;
    const CoreRangeSet receiver_cores = pipe.receiver_cores();
    Program program = CreateProgram();
    // Attach only receiver cores — consumer role in a separate program.
    EXPECT_EQ(AttachPrefetcherPipe(program, pipe, receiver_cores, entry_size), prefetcher_pipe_id);
    create_dm_kernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_receiver.cpp",
        receiver_cores,
        {prefetcher_pipe_id, entry_size, num_entries, 0u});
    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(2);
    return prefetcher_pipe_test::verify_receiver_ring(
               device, pipe, CoreCoord(1, 0), data_pattern, entry_size, num_entries, 0, 1)
               ? 1u
               : 0u;
}

// Cross-program PrefetcherPipe equivalent of CrossNode's run_1toN_program:
// Program A pushes on sender cores; Program B pops on receivers and verifies rings.
uint32_t run_persistent_1toN_cross_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t write_primitive,
    uint8_t prefetcher_pipe_id = 0,
    bool simultaneous_subdevices = false) {
    distributed::MeshDevice& device = *mesh_device;
    const CoreRangeSet sender_cores = pipe.sender_cores();
    const CoreRangeSet receiver_cores = pipe.receiver_cores();
    const auto receivers = corerange_to_cores(receiver_cores);
    const uint32_t num_receivers = static_cast<uint32_t>(receivers.size());
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(write_primitive);

    auto add_sender_kernel = [&](Program& program) -> KernelHandle {
        KernelHandle sender_k = create_dm_kernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_sender.cpp",
            sender_cores,
            {prefetcher_pipe_id, entry_size, num_entries, write_primitive, data_pattern, 0u});
        prefetcher_pipe_test::write_sender_l1_staging(
            device, sender_cores, pipe, data_pattern, entry_size, num_entries, num_receivers);
        prefetcher_pipe_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, pipe);
        return sender_k;
    };
    auto add_receiver_kernels = [&](Program& program) {
        for (uint32_t ri = 0; ri < num_receivers; ++ri) {
            const CoreRangeSet single = CoreRangeSet(CoreRange(receivers[ri]));
            create_dm_kernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_receiver.cpp",
                single,
                {prefetcher_pipe_id, entry_size, num_entries, ri});
        }
    };

    if (simultaneous_subdevices) {
        Program sender_program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(sender_program, pipe, sender_cores, entry_size), prefetcher_pipe_id);
        add_sender_kernel(sender_program);
        Program receiver_program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(receiver_program, pipe, receiver_cores, entry_size), prefetcher_pipe_id);
        add_receiver_kernels(receiver_program);

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
        // Two programs with async SD so producer/consumer overlap when the ring would
        // otherwise fill before the consumer is launched. (Do not merge into one Program
        // when sender/receiver use different num_threads_per_cluster — GO enables disagree.)
        Program sender_program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(sender_program, pipe, sender_cores, entry_size), prefetcher_pipe_id);
        add_sender_kernel(sender_program);
        Program receiver_program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(receiver_program, pipe, receiver_cores, entry_size), prefetcher_pipe_id);
        add_receiver_kernels(receiver_program);
        persistent_run_overlapping_programs(mesh_device, std::move(sender_program), std::move(receiver_program));
    }

    uint32_t pass_count = 0;
    for (uint32_t ri = 0; ri < num_receivers; ++ri) {
        if (prefetcher_pipe_test::verify_receiver_ring(
                device, pipe, receivers[ri], data_pattern, entry_size, num_entries, ri, num_receivers)) {
            ++pass_count;
        }
    }
    return pass_count;
}

// Quasar multi-DM sender (partition-R via get_num_threads): same Flow C kernel as
// single-threaded, launched with num_threads_per_cluster > 1. APIs skip non-owned receivers.
uint32_t run_persistent_mt_sender_partition_r(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PrefetcherPipe& pipe,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_sender_threads,
    uint32_t write_primitive = 3) {
    TT_FATAL(is_quasar_arch(), "Multi-DM PrefetcherPipe sender tests require Quasar");
    TT_FATAL(num_sender_threads >= 2, "Multi-DM sender test requires num_sender_threads >= 2");

    distributed::MeshDevice& device = *mesh_device;
    const CoreRangeSet sender_cores = pipe.sender_cores();
    const CoreRangeSet receiver_cores = pipe.receiver_cores();
    const auto receivers = corerange_to_cores(receiver_cores);
    const uint32_t num_receivers = static_cast<uint32_t>(receivers.size());
    TT_FATAL(num_receivers >= 1, "Multi-DM sender test needs at least one receiver");
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(write_primitive);

    Program sender_program = CreateProgram();
    EXPECT_EQ(AttachPrefetcherPipe(sender_program, pipe, sender_cores, entry_size), 0u);
    KernelHandle sender_k = create_dm_kernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_sender.cpp",
        sender_cores,
        {0u, entry_size, num_entries, write_primitive, data_pattern, 0u},
        {},
        num_sender_threads);
    prefetcher_pipe_test::write_sender_l1_staging(
        device, sender_cores, pipe, data_pattern, entry_size, num_entries, num_receivers);
    prefetcher_pipe_test::set_sender_l1_staging_runtime_args(sender_program, sender_k, sender_cores, pipe);

    Program receiver_program = CreateProgram();
    EXPECT_EQ(AttachPrefetcherPipe(receiver_program, pipe, receiver_cores, entry_size), 0u);
    for (uint32_t ri = 0; ri < num_receivers; ++ri) {
        const CoreRangeSet single = CoreRangeSet(CoreRange(receivers[ri]));
        create_dm_kernel(
            receiver_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_receiver.cpp",
            single,
            {0u, entry_size, num_entries, ri},
            {},
            /*num_threads_per_cluster=*/1);
    }

    // Async SD two-program overlap (sender threads != receiver threads — cannot share one Program).
    persistent_run_overlapping_programs(mesh_device, std::move(sender_program), std::move(receiver_program));

    uint32_t pass_count = 0;
    for (uint32_t ri = 0; ri < num_receivers; ++ri) {
        if (prefetcher_pipe_test::verify_receiver_ring(
                device, pipe, receivers[ri], data_pattern, entry_size, num_entries, ri, num_receivers)) {
            ++pass_count;
        }
    }
    return pass_count;
}

}  // namespace

TEST_F(PrefetcherPipeFixture, CreatePrefetcherPipe_TopologyRejects) {
    auto mesh_device = devices_[0];
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const CoreRangeSet receivers0(CoreRange({1, 0}));
    const CoreRangeSet receivers1(CoreRange({3, 0}));

    {
        EXPECT_NO_THROW(experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receivers0, 1024));
    }
    if (grid.x >= 4) {
        auto pipe0 = experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receivers0, 1024);
        EXPECT_NO_THROW(experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(2, 0), receivers1, 1024));
        (void)pipe0;
    }
    {
        const CoreRangeSet overlap(CoreRange({0, 0}));
        EXPECT_THROW(
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), overlap, 1024), std::exception);
    }
    {
        EXPECT_THROW(
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet{}, 1024),
            std::exception);
    }
}

TEST_F(PrefetcherPipeFixture, CreatePrefetcherPipe_GeometryRejects) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};

    EXPECT_THROW(
        experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 0), std::exception);
    EXPECT_THROW(
        experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 33), std::exception);
    EXPECT_THROW(
        experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024, BufferType::DRAM),
        std::exception);
}

TEST_F(PrefetcherPipeFixture, PersistentArenaSharesAddressesAcrossDisjointCores) {
    if (const auto reason = insufficient_worker_grid_reason(4); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    auto pipe0 =
        experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);
    auto pipe1 =
        experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(2, 0), CoreRangeSet(CoreRange({3, 0})), 1024);

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

    // M=3 independent 1:N pipes (N=2). Each Create call is one logical pipe.
    // Their participating core sets are disjoint, so the persistent arena can
    // place every ring/config at the same per-core L1 addresses.
    auto pipe0 = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {2, 0})), ring_bytes);
    auto pipe1 = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 1), CoreRangeSet(CoreRange({1, 1}, {2, 1})), ring_bytes);
    auto pipe2 = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 2), CoreRangeSet(CoreRange({1, 2}, {2, 2})), ring_bytes);

    EXPECT_EQ(pipe1.buffer_address(), pipe0.buffer_address());
    EXPECT_EQ(pipe2.buffer_address(), pipe0.buffer_address());
    EXPECT_EQ(pipe1.config_address(), pipe0.config_address());
    EXPECT_EQ(pipe2.config_address(), pipe0.config_address());

    EXPECT_EQ(
        run_persistent_1toN_cross_program(mesh_device, pipe0, entry_size, num_entries, /*write_primitive=*/0), 2u);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(mesh_device, pipe1, entry_size, num_entries, /*write_primitive=*/0), 2u);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(mesh_device, pipe2, entry_size, num_entries, /*write_primitive=*/0), 2u);
}

TEST_F(PrefetcherPipeFixture, PersistentArenaSerializesOverlappingCoresAndReusesFreedSpace) {
    if (const auto reason = insufficient_worker_grid_reason(3); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    uint32_t first_ring_address = 0;
    uint32_t first_config_address = 0;
    {
        auto pipe0 = experimental::CreatePrefetcherPipe(
            mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);
        first_ring_address = pipe0.buffer_address();
        first_config_address = pipe0.config_address();

        auto pipe1 = experimental::CreatePrefetcherPipe(
            mesh_device.get(), CoreCoord(2, 0), CoreRangeSet(CoreRange({1, 0})), 1024);
        EXPECT_GE(pipe1.buffer_address(), pipe0.config_address() + pipe0.config_page_size());
    }

    auto replacement =
        experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0})), 1024);
    EXPECT_EQ(replacement.buffer_address(), first_ring_address);
    EXPECT_EQ(replacement.config_address(), first_config_address);
}

TEST_F(PrefetcherPipeFixture, AttachPrefetcherPipe_EntrySizeRejects) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}))};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);

    for (const uint32_t entry_size : {0u, 33u, 1280u}) {
        Program program = CreateProgram();
        EXPECT_THROW(AttachPrefetcherPipe(program, pipe, pipe.all_cores(), entry_size), std::exception);
    }
    Program program = CreateProgram();
    EXPECT_EQ(AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256), 0u);
}

TEST_F(PrefetcherPipeFixture, AttachPrefetcherPipe_RequiresRoleCompleteProgram) {
    if (const auto reason = insufficient_worker_grid_reason(4); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const CoreRangeSet receiver_cores(CoreRange({2, 0}, {3, 0}));
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, 1024);

    {
        Program program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(program, pipe, CoreRangeSet(CoreRange(CoreCoord(0, 0))), 256), 0u);
    }
    {
        Program program = CreateProgram();
        EXPECT_THROW(
            AttachPrefetcherPipe(program, pipe, CoreRangeSet(CoreRange(CoreCoord(2, 0))), 256), std::exception);
    }
    {
        Program sender_program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(sender_program, pipe, pipe.sender_cores(), 256), 0u);
        Program receiver_program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(receiver_program, pipe, pipe.receiver_cores(), 256), 0u);
    }
}

TEST_F(PrefetcherPipeFixture, AttachPrefetcherPipe_AssignsDistinctSlots) {
    if (const auto reason = insufficient_worker_grid_reason(2, 2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const std::pair<CoreCoord, CoreRangeSet> mapping0 = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};
    const std::pair<CoreCoord, CoreRangeSet> mapping1 = {CoreCoord(0, 1), CoreRangeSet(CoreRange({1, 1}, {1, 1}))};

    auto pipe0 = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping0.first, mapping0.second, 1024);
    auto pipe1 = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping1.first, mapping1.second, 1024);

    Program program = CreateProgram();
    create_dm_kernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        CoreRangeSet({CoreRange({0, 0}, {1, 1})}),
        {});

    EXPECT_EQ(AttachPrefetcherPipe(program, pipe0, pipe0.all_cores(), 256), 0u);
    EXPECT_EQ(AttachPrefetcherPipe(program, pipe1, pipe1.all_cores(), 256), 1u);

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

TEST_F(PrefetcherPipeFixture, AttachPrefetcherPipe_SameObjectMultiplePrograms) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);
    const uint32_t fifo_start = pipe.buffer_address();
    const uint32_t config_addr = pipe.config_address();

    Program program_a = CreateProgram();
    Program program_b = CreateProgram();
    AttachPrefetcherPipe(program_a, pipe, pipe.all_cores(), 256);
    AttachPrefetcherPipe(program_b, pipe, pipe.all_cores(), 256);

    const auto& per_core_a = program_a.impl().get_per_core_prefetcher_pipes().at(CoreCoord(0, 0));
    const auto& per_core_b = program_b.impl().get_per_core_prefetcher_pipes().at(CoreCoord(0, 0));
    EXPECT_EQ(per_core_a[0].config_page_addr, config_addr);
    EXPECT_EQ(per_core_b[0].config_page_addr, config_addr);
    EXPECT_EQ(pipe.buffer_address(), fifo_start);
}

TEST_F(PrefetcherPipeFixture, AttachPrefetcherPipe_AddressStableAcrossRebuild) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);
    const uint32_t ring_addr = pipe.buffer_address();
    const uint32_t config_addr = pipe.config_address();

    {
        Program program = CreateProgram();
        AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256);
        detail::CompileProgram(mesh_device.get(), program);
        program.impl().finalize_offsets(mesh_device.get());
    }
    {
        Program program = CreateProgram();
        AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256);
        EXPECT_EQ(pipe.buffer_address(), ring_addr);
        EXPECT_EQ(pipe.config_address(), config_addr);
        EXPECT_EQ(program.impl().get_per_core_prefetcher_pipes().at(CoreCoord(0, 0))[0].config_page_addr, config_addr);
    }
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_CrossProgramPersistence) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);
    const uint32_t ring_addr = pipe.buffer_address();

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, 256, 4, 0u), 1u);
    EXPECT_EQ(pipe.buffer_address(), ring_addr);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, 256, 4, 0u), 1u);
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
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};
    auto pipe =
        experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, entry_size * ring_depth);

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, entry_size, first_push, 0u), 1u);
    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, entry_size, second_push, 0u), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, entry_size, first_push + second_push, 0u), 1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_BackToBackRelaunch) {
    // Two cross-program push→pop cycles on the same PrefetcherPipe.
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, 256, 4, 0u), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, 256, 4, 0u), 1u);
    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, 256, 4, 0u), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, 256, 4, 0u), 1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_CrossSubDevicePersistence) {
    if (!is_fast_dispatch()) {
        GTEST_SKIP() << "Sub device managers are unsupported with slow dispatch";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    // Programs may only span one sub-device. Put the sender on SD0 and the receiver on SD1,
    // Attach each program to only its role cores, and share one PrefetcherPipe across both.
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
        const std::pair<CoreCoord, CoreRangeSet> mapping = {sender_core, receiver_cores};
        auto pipe = experimental::CreatePrefetcherPipe(
            mesh_device.get(), mapping.first, mapping.second, entry_size * num_entries);
        const uint32_t ring_addr = pipe.buffer_address();

        EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, entry_size, num_entries, 0u), 1u);
        EXPECT_EQ(pipe.buffer_address(), ring_addr);
        EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, entry_size, num_entries, 0u), 1u);
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
    const std::pair<CoreCoord, CoreRangeSet> mapping = {sender_core, receiver_cores};
    auto pipe =
        experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, entry_size * num_entries);

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, entry_size, num_entries, 0u), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, entry_size, num_entries, 0u), 1u);
    EXPECT_EQ(run_persistent_sender_push(mesh_device, pipe, entry_size, num_entries, 0u), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pipe, entry_size, num_entries, 0u), 1u);

    mesh_device->clear_loaded_sub_device_manager();
}

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

    const std::pair<CoreCoord, CoreRangeSet> mapping = {sender_core, receiver_cores};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(
            mesh_device,
            pipe,
            256,
            4,
            /*write_primitive=*/2,
            /*prefetcher_pipe_id=*/0,
            /*simultaneous_subdevices=*/true),
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

    const std::pair<CoreCoord, CoreRangeSet> mapping = {sender_core, receiver_cores};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(
            mesh_device,
            pipe,
            256,
            4,
            /*write_primitive=*/0,
            /*prefetcher_pipe_id=*/0,
            /*simultaneous_subdevices=*/true),
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

    const std::pair<CoreCoord, CoreRangeSet> mapping = {sender_core, receiver_cores};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(
            mesh_device,
            pipe,
            256,
            4,
            /*write_primitive=*/1,
            /*prefetcher_pipe_id=*/0,
            /*simultaneous_subdevices=*/true),
        4u);

    mesh_device->clear_loaded_sub_device_manager();
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_WriteToReceiver_ReceiverContiguous) {
    if (const auto reason = insufficient_worker_grid_reason(5); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {4, 0}))};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pipe, 256, 4, /*write_primitive=*/2), 4u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RoundRobinPushBackToReceiver) {
    if (const auto reason = insufficient_worker_grid_reason(5); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {4, 0}))};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 256);
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
        const CoreRangeSet receivers = CoreRangeSet(CoreRange({1, 0}, {1, 0}));
        auto pipe =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receivers, entry_size * num_entries);
        EXPECT_EQ(
            run_persistent_mt_sender_partition_r(
                mesh_device, pipe, entry_size, num_entries, num_sender_threads, write_primitive),
            1u);
    }
}

// 2 sender DMs is partition-R (needs R>=2). This test is lane credits on 1S×1R:
// arm P=2 on the receiver Attach, run 2 receiver DMs, keep a single sender DM that
// stripes pages_sent across lanes. (Sharing one receiver across sender DMs is deferred.)
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
    constexpr uint32_t num_sender_threads = 1;
    const CoreRangeSet sender_cores = CoreRangeSet(CoreRange(CoreCoord(0, 0)));
    const CoreRangeSet receiver_cores = CoreRangeSet(CoreRange({1, 0}, {1, 0}));

    for (const uint32_t write_primitive : {0u, 1u, 2u, 3u, 4u, 5u}) {
        SCOPED_TRACE("write_primitive=" + std::to_string(write_primitive));
        log_info(tt::LogTest, "MultiDM_2P2R_1S1R: start write_primitive={}", write_primitive);
        auto pipe = experimental::CreatePrefetcherPipe(
            mesh_device.get(), CoreCoord(0, 0), receiver_cores, entry_size * num_entries);
        const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(write_primitive);

        // Matching P on both ends, but still use two programs + async SD: a shared Program
        // has hung on RTL sim; async overlap is the proven path for lane-credit MultiDM.
        Program receiver_program = CreateProgram();
        EXPECT_EQ(
            AttachPrefetcherPipe(
                receiver_program,
                pipe,
                receiver_cores,
                entry_size,
                /*num_pipe_consumer_threads=*/num_credit_lanes),
            0u);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), num_credit_lanes);
        create_dm_kernel(
            receiver_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_receiver.cpp",
            receiver_cores,
            {0u, entry_size, num_entries, 0u},
            {},
            num_credit_lanes);

        Program sender_program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(sender_program, pipe, sender_cores, entry_size), 0u);
        KernelHandle sender_k = create_dm_kernel(
            sender_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_sender.cpp",
            sender_cores,
            {0u, entry_size, num_entries, write_primitive, data_pattern, 0u},
            {},
            num_sender_threads);
        prefetcher_pipe_test::write_sender_l1_staging(
            device, sender_cores, pipe, data_pattern, entry_size, num_entries, 1);
        prefetcher_pipe_test::set_sender_l1_staging_runtime_args(sender_program, sender_k, sender_cores, pipe);

        persistent_run_overlapping_programs(mesh_device, std::move(sender_program), std::move(receiver_program));
        log_info(tt::LogTest, "MultiDM_2P2R_1S1R: programs finished write_primitive={}", write_primitive);

        EXPECT_TRUE(prefetcher_pipe_test::verify_receiver_ring(
            device, pipe, CoreCoord(1, 0), data_pattern, entry_size, num_entries, 0, 1));
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
        auto pipe =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receivers, entry_size * num_entries);
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
        auto pipe =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receivers, entry_size * num_entries);
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
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {2, 0}))};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);
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
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 128 * 1024;
    constexpr uint32_t num_entries = 3;
    const CoreRangeSet receiver_cores(CoreRange({1, 0}, {2, 0}));
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), receiver_cores, entry_size * num_entries);

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
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {4, 0}))};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pipe, 256, 4, /*write_primitive=*/4), 4u);
}

TEST_F(PrefetcherPipeFixture, GlobalAndCrossNode_SameProgram_DistinctRegions) {
    if (is_quasar()) {
        GTEST_SKIP() << "PrefetcherPipe Quasar Phase 0 does not support CrossNodeDFB yet";
    }
    auto mesh_device = devices_[0];
    const std::pair<CoreCoord, CoreRangeSet> pipe_mapping = {CoreCoord(2, 0), CoreRangeSet(CoreRange({3, 0}, {3, 0}))};

    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), pipe_mapping.first, pipe_mapping.second, 1024);

    Program program = CreateProgram();
    const CoreRangeSet all_cores =
        CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}, {1, 0}), CoreRange({2, 0}, {3, 0})});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    experimental::CreateCrossNodeDFB(
        program, mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), 256, 4);
    AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256);

    detail::CompileProgram(mesh_device.get(), program);
    program.impl().finalize_offsets(mesh_device.get());

    const auto& hal = MetalContext::instance().hal();
    const uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const auto& kg = program.impl().get_kernel_groups(index)[0]->launch_msg.view().kernel_config();
    EXPECT_NE(kg.cross_node_dfb_offset(), REMOTE_DFB_OFFSET_NONE);
    EXPECT_NE(kg.prefetcher_pipe_offset(), REMOTE_DFB_OFFSET_NONE);
    EXPECT_NE(kg.cross_node_dfb_offset(), kg.prefetcher_pipe_offset());
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_StaleCommitRejected) {
    // After set_entry_size updates word[5] (PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE),
    // commit() with a stale iface.fifo_page_size must not overwrite word[4]
    // (PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT). Push one entry (not a full ring) so the
    // good checkpoint is distinguishable from fifo_start and from the poison wr_ptr.
    auto mesh_device = devices_[0];
    distributed::MeshDevice& device = *mesh_device;
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t new_entry_size = 512;
    constexpr uint32_t num_entries = 4;

    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};
    auto pipe =
        experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, entry_size * num_entries);
    const uint32_t poison_wr_ptr = pipe.buffer_address() + 2 * entry_size;

    const CoreCoord sender_core(0, 0);
    const CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(2);

    Program program = CreateProgram();
    EXPECT_EQ(AttachPrefetcherPipe(program, pipe, sender_cores, entry_size), 0u);
    KernelHandle sender_k = create_dm_kernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_stale_commit.cpp",
        sender_cores,
        {0u, entry_size, new_entry_size, poison_wr_ptr},
        {{"PREFETCHER_PIPE_TEST_HELPERS", "1"}});

    prefetcher_pipe_test::write_sender_l1_staging(
        device, sender_cores, pipe, data_pattern, entry_size, /*num_entries=*/1, 1);
    prefetcher_pipe_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, pipe);

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

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_HostRelationshipValidation) {
    if (const auto reason = insufficient_worker_grid_reason(is_quasar() ? 2 : 3); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores =
        is_quasar() ? CoreRangeSet(CoreRange({1, 0}, {1, 0})) : CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    const std::pair<CoreCoord, CoreRangeSet> mapping = {sender_core, receiver_cores};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, 1024);

    {
        Program program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256), 0u);
        experimental::dfb::DataflowBufferConfig config{.entry_size = 256, .num_entries = 4};
        const uint32_t relay_host_id = experimental::CreatePrefetcherPipeRelayDataflowBuffer(
            program, receiver_cores, config, /*prefetcher_pipe_id=*/0);
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
        EXPECT_EQ(
            program.impl().get_per_core_prefetcher_pipes().at(sender_core).at(0).relay_dfb_id,
            std::numeric_limits<uint8_t>::max());
        EXPECT_EQ(relay_dfb->borrowed_addr_, pipe.buffer_address());

        // Metal 2.0 genfiles reads DataflowBufferBindingHandle.prefetcher_pipe_id when emitting
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
                [&](const std::string& name, uint16_t logical_id, bool is_relay, uint8_t prefetcher_pipe_id) {
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
        Program program = CreateProgram();
        AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256);
        experimental::dfb::DataflowBufferConfig wrong_size{.entry_size = 128, .num_entries = 4};
        EXPECT_THROW(
            experimental::CreatePrefetcherPipeRelayDataflowBuffer(program, receiver_cores, wrong_size, 0),
            std::exception);
    }

    {
        // An entry size the 1024 B ring does not divide: 384 B leaves a 256 B trailing gap, so the
        // relay depth is the 2 whole entries the ring holds. Depth 3 (rounding the gap up into an
        // entry) is what gets rejected.
        Program program = CreateProgram();
        AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 384);
        experimental::dfb::DataflowBufferConfig ceil_depth{.entry_size = 384, .num_entries = 3};
        EXPECT_THROW(
            experimental::CreatePrefetcherPipeRelayDataflowBuffer(program, receiver_cores, ceil_depth, 0),
            std::exception);
        experimental::dfb::DataflowBufferConfig floor_depth{.entry_size = 384, .num_entries = 2};
        const uint32_t relay_host_id =
            experimental::CreatePrefetcherPipeRelayDataflowBuffer(program, receiver_cores, floor_depth, 0);
        const auto* relay_dfb = program.impl().get_dataflow_buffer(relay_host_id).get();
        ASSERT_NE(relay_dfb, nullptr);
        EXPECT_EQ(relay_dfb->borrowed_addr_, pipe.buffer_address());
    }

    {
        Program program = CreateProgram();
        experimental::dfb::DataflowBufferConfig config{.entry_size = 256, .num_entries = 4};
        EXPECT_THROW(
            experimental::CreatePrefetcherPipeRelayDataflowBuffer(
                program, receiver_cores, config, /*prefetcher_pipe_id=*/0),
            std::exception);
    }
}

// Host-only: Attach(..., num_pipe_consumer_threads) and/or relay arm the pipe's lane count,
// which dispatch packs into each program's kernel-config slot; over-capacity and
// reprogram-to-different-P are rejected.
TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_CreditLanesHostProgramming) {
    if (!is_quasar_arch()) {
        GTEST_SKIP() << "Lane-credit capacity > 1 is Quasar-only";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    const CoreRangeSet receiver_cores = CoreRangeSet(CoreRange({1, 0}, {1, 0}));

    {
        // Arm lanes via Attach without a relay.
        auto pipe =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, /*ring_size=*/1024);

        EXPECT_EQ(pipe.impl().credit_lane_capacity(), PREFETCHER_PIPE_MAX_CREDIT_LANES);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);

        Program program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256, /*num_pipe_consumer_threads=*/2), 0u);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
        // P is not in the persistent page (word[9] stays reserved); it travels in the program's
        // kernel-config slot, packed above relay_dfb_id, on sender and receiver cores alike.
        const auto& per_core = program.impl().get_per_core_prefetcher_pipes();
        for (const CoreCoord core : {CoreCoord(0, 0), CoreCoord(1, 0)}) {
            EXPECT_EQ(pipe.impl().config_page(core)[9], 0u);
            const auto payload =
                program_dispatch::build_prefetcher_pipe_config_payload(program.impl(), per_core.at(core));
            const uint32_t relay_word = payload[REMOTE_DFB_REGION_HEADER_WORDS + 2];
            EXPECT_EQ(prefetcher_pipe_slot_credit_lanes(relay_word), 2u);
            EXPECT_EQ(prefetcher_pipe_slot_relay_id(relay_word), std::numeric_limits<uint8_t>::max());
        }
        // A single-lane pipe's slot is bit-identical to the pre-lane encoding.
        auto pipe_single =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, /*ring_size=*/1024);
        Program program_single = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(program_single, pipe_single, pipe_single.all_cores(), 256), 0u);
        const auto single_payload = program_dispatch::build_prefetcher_pipe_config_payload(
            program_single.impl(), program_single.impl().get_per_core_prefetcher_pipes().at(CoreCoord(1, 0)));
        EXPECT_EQ(
            single_payload[REMOTE_DFB_REGION_HEADER_WORDS + 2],
            static_cast<uint32_t>(std::numeric_limits<uint8_t>::max()));

        // Matching relay num_producers is a no-op; mismatch must throw.
        experimental::dfb::DataflowBufferConfig match{
            .entry_size = 256,
            .num_entries = 4,
            .num_producers = 2,
            .pap = experimental::dfb::AccessPattern::STRIDED,
        };
        EXPECT_NO_THROW(experimental::CreatePrefetcherPipeRelayDataflowBuffer(program, receiver_cores, match, 0));
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
    }

    {
        // Lanes armed by a relay *after* the Attach: the slot payload is built from the pipe at
        // dispatch time, so it must carry the relay's P, not the P=1 the participant was added with.
        auto pipe =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, /*ring_size=*/1024);
        Program program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256), 0u);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);
        experimental::dfb::DataflowBufferConfig late{
            .entry_size = 256,
            .num_entries = 4,
            .num_producers = 2,
            .pap = experimental::dfb::AccessPattern::STRIDED,
        };
        const uint32_t relay_id =
            experimental::CreatePrefetcherPipeRelayDataflowBuffer(program, receiver_cores, late, 0);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
        const auto& per_core = program.impl().get_per_core_prefetcher_pipes();
        const auto recv_payload =
            program_dispatch::build_prefetcher_pipe_config_payload(program.impl(), per_core.at(CoreCoord(1, 0)));
        const uint32_t recv_word = recv_payload[REMOTE_DFB_REGION_HEADER_WORDS + 2];
        EXPECT_EQ(prefetcher_pipe_slot_credit_lanes(recv_word), 2u);
        EXPECT_EQ(prefetcher_pipe_slot_relay_id(recv_word), program.impl().get_dataflow_buffer(relay_id)->device_slot);
        const auto send_payload =
            program_dispatch::build_prefetcher_pipe_config_payload(program.impl(), per_core.at(CoreCoord(0, 0)));
        const uint32_t send_word = send_payload[REMOTE_DFB_REGION_HEADER_WORDS + 2];
        EXPECT_EQ(prefetcher_pipe_slot_credit_lanes(send_word), 2u);
        EXPECT_EQ(prefetcher_pipe_slot_relay_id(send_word), std::numeric_limits<uint8_t>::max());
    }

    {
        auto pipe =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, /*ring_size=*/1024);
        Program program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256, /*num_pipe_consumer_threads=*/2), 0u);
        experimental::dfb::DataflowBufferConfig mismatch{
            .entry_size = 256,
            .num_entries = 4,
            .num_producers = 3,
            .pap = experimental::dfb::AccessPattern::STRIDED,
        };
        EXPECT_THROW(
            experimental::CreatePrefetcherPipeRelayDataflowBuffer(program, receiver_cores, mismatch, 0),
            std::exception);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
    }

    {
        auto pipe =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, /*ring_size=*/1024);
        Program program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256), 0u);
        experimental::dfb::DataflowBufferConfig too_many{
            .entry_size = 256,
            .num_entries = 4,
            .num_producers = static_cast<uint8_t>(pipe.impl().credit_lane_capacity() + 1),
            .pap = experimental::dfb::AccessPattern::STRIDED,
        };
        EXPECT_THROW(
            experimental::CreatePrefetcherPipeRelayDataflowBuffer(program, receiver_cores, too_many, 0),
            std::exception);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);
    }

    {
        // Lane mode needs an exact entry ring with an entry count divisible by P; a rejected
        // Attach must not leave the persistent pipe armed.
        auto pipe =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, /*ring_size=*/1024);
        Program program = CreateProgram();
        // 1024 % 384 != 0: trailing gap would never be credited with striped lanes.
        EXPECT_THROW(
            AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 384, /*num_pipe_consumer_threads=*/2),
            std::exception);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);
        // 4 entries is not a multiple of 3 lanes.
        EXPECT_THROW(
            AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256, /*num_pipe_consumer_threads=*/3),
            std::exception);
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 1u);
        EXPECT_NO_THROW(AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256, /*num_pipe_consumer_threads=*/2));
        EXPECT_EQ(pipe.impl().num_credit_lanes(), 2u);
    }

    {
        // Multi-producer ALL relay: the DFB must be serialized lane-interleaved (stride P) so
        // producer h's TC walks the entries pipe lane h receives (h, h+P, ...), not a
        // contiguous per-producer block. A standalone ALL DFB keeps stride 1.
        auto pipe =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, /*ring_size=*/1024);
        Program program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(program, pipe, pipe.all_cores(), 256, /*num_pipe_consumer_threads=*/2), 0u);
        experimental::dfb::DataflowBufferConfig all_multi_producer{
            .entry_size = 256,
            .num_entries = 4,
            .num_producers = 2,
            .pap = experimental::dfb::AccessPattern::STRIDED,
            .num_consumers = 2,
            .cap = experimental::dfb::AccessPattern::ALL,
        };
        const uint32_t relay_id =
            experimental::CreatePrefetcherPipeRelayDataflowBuffer(program, receiver_cores, all_multi_producer, 0);
        EXPECT_EQ(program.impl().get_dataflow_buffer(relay_id)->stride_in_entries, 2u);
        EXPECT_EQ(program.impl().get_dataflow_buffer(relay_id)->capacity, 2u);
        const uint32_t standalone_id =
            experimental::dfb::CreateDataflowBuffer(program, receiver_cores, all_multi_producer);
        EXPECT_EQ(program.impl().get_dataflow_buffer(standalone_id)->stride_in_entries, 1u);
    }

    {
        // Armed lanes must match the receiver DM kernel's thread count; the device guard is a
        // debug-only ASSERT, so the host rejects the mismatch at finalize.
        auto pipe =
            experimental::CreatePrefetcherPipe(mesh_device.get(), CoreCoord(0, 0), receiver_cores, /*ring_size=*/1024);
        Program program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(program, pipe, receiver_cores, 256, /*num_pipe_consumer_threads=*/2), 0u);
        create_dm_kernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            receiver_cores,
            {},
            {},
            /*num_threads_per_cluster=*/1);
        detail::CompileProgram(mesh_device.get(), program);
        EXPECT_THROW(program.impl().finalize_offsets(mesh_device.get()), std::exception);

        // Sender cores are not lane-bound: partition-R uses the kernel's own thread count.
        Program sender_program = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(sender_program, pipe, pipe.sender_cores(), 256), 0u);
        create_dm_kernel(
            sender_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            pipe.sender_cores(),
            {},
            {},
            /*num_threads_per_cluster=*/1);
        detail::CompileProgram(mesh_device.get(), sender_program);
        EXPECT_NO_THROW(sender_program.impl().finalize_offsets(mesh_device.get()));
    }
}

static uint32_t prefetcher_pipe_relay_expected_checksum(uint32_t total_entries) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < total_entries; ++i) {
        checksum += static_cast<uint32_t>(static_cast<uint8_t>(i)) * 0x01010101u;
    }
    return checksum;
}

// PrefetcherPipe relay e2e params. Multi-thread surface is the local relay DFB
// (pap/cap + num_producers/consumers). Pipe consumers are the relay producers:
// num_producers>1 with pap=STRIDED activates PrefetcherPipe lane credits from
// CreatePrefetcherPipeRelayDataflowBuffer so wait_front(n)/pop_front(n) are n owned
// strides (batch_size may be >1). Optional num_sender_threads>1 partitions senders.
struct PrefetcherPipeRelayParams {
    uint32_t entry_size = 256;
    uint32_t ring_depth = 4;
    uint32_t total_entries = 4;
    uint32_t batch_size = 1;
    uint8_t num_producers = 1;
    experimental::dfb::AccessPattern pap = experimental::dfb::AccessPattern::STRIDED;
    uint8_t num_consumers = 1;
    experimental::dfb::AccessPattern cap = experimental::dfb::AccessPattern::STRIDED;
    std::optional<uint32_t> receiver_entry_size_override = std::nullopt;
    uint32_t trisc_delay_iterations = 0;
    // When true: one program with sender + relay receiver + TRISC (backpressure).
    // Supports num_sender_threads>1 (Quasar).
    bool same_program = false;
    // Quasar: >1 = stock sender with num_threads_per_cluster (Flows A–D partition-R).
    uint32_t num_sender_threads = 1;
};

// Prog A (or same-program): sender push. Prog B: receiver DM bind_relay + TRISC consume.
// Returns number of receiver cores whose per-thread results match the expected pattern.
static uint32_t run_prefetcher_pipe_relay(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PrefetcherPipe& pipe,
    const PrefetcherPipeRelayParams& params) {
    TT_FATAL(params.total_entries % params.batch_size == 0, "Relay test total_entries must be divisible by batch_size");
    TT_FATAL(params.ring_depth % params.batch_size == 0, "Relay test ring_depth must be divisible by batch_size");
    TT_FATAL(params.num_producers >= 1, "num_producers must be >= 1");
    TT_FATAL(params.num_consumers >= 1, "num_consumers must be >= 1");
    if (params.cap == experimental::dfb::AccessPattern::STRIDED) {
        TT_FATAL(
            params.total_entries % params.num_consumers == 0,
            "STRIDED: total_entries must be divisible by num_consumers");
        TT_FATAL(
            params.ring_depth % std::max(params.num_producers, params.num_consumers) == 0,
            "STRIDED: ring_depth must be divisible by max(P,C)");
    }
    if (params.pap == experimental::dfb::AccessPattern::STRIDED) {
        TT_FATAL(
            params.ring_depth % std::max(params.num_producers, params.num_consumers) == 0,
            "STRIDED pap: ring_depth must be divisible by max(P,C)");
    }
    TT_FATAL(params.num_sender_threads >= 1, "num_sender_threads must be >= 1");
    if (params.num_producers > 1) {
        TT_FATAL(is_quasar_arch(), "Multi-producer PrefetcherPipe relay requires Quasar");
        TT_FATAL(
            params.pap == experimental::dfb::AccessPattern::STRIDED,
            "Multi-producer PrefetcherPipe relay requires pap=STRIDED");
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
    const CoreCoord sender_core(0, 0);
    const CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreRangeSet receiver_cores = pipe.receiver_cores();
    const uint32_t recv_entry_size = params.receiver_entry_size_override.value_or(params.entry_size);
    const uint32_t recv_num_entries = pipe.ring_size() / recv_entry_size;
    // Floor, not exact division: an entry size the ring does not divide leaves a trailing gap that
    // holds no entry, and the relay borrows only the whole entries. Lane mode (P > 1) does need an
    // exact ring, which validate_lane_geometry enforces at Attach.
    TT_FATAL(
        recv_num_entries == params.ring_depth || params.receiver_entry_size_override.has_value(),
        "ring_depth must match pipe.ring_size/entry_size unless overriding recv entry size");

    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(0);
    const uint32_t recv_total_entries = (params.total_entries * params.entry_size) / recv_entry_size;
    TT_FATAL(
        (params.total_entries * params.entry_size) % recv_entry_size == 0,
        "pushed bytes must be divisible by recv entry size");
    TT_FATAL(recv_total_entries % params.batch_size == 0, "recv_total_entries must be divisible by batch_size");

    const uint32_t entries_per_consumer = (params.cap == experimental::dfb::AccessPattern::ALL)
                                              ? recv_total_entries
                                              : (recv_total_entries / params.num_consumers);
    TT_FATAL(entries_per_consumer % params.batch_size == 0, "entries_per_consumer must be divisible by batch_size");

    // Pipe-side batch for the relay receiver DM. The receiver publishes one relay entry per
    // push, round-robin over its consumer TCs (STRIDED cap with C > P gives each producer
    // C / P TCs). A TRISC batch of b on one TC therefore needs b * (C / P) pipe entries per
    // iteration, or the DM would block in pop_front waiting for consumers that are still
    // waiting for their batch. batch_size == 1 never needs the factor.
    uint32_t pipe_batch_size = params.batch_size;
    if (params.batch_size > 1 && params.cap == experimental::dfb::AccessPattern::STRIDED &&
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

    const uint32_t result_words = static_cast<uint32_t>(params.num_consumers) * 2u;
    const uint32_t result_page_size = std::max(32u, result_words * static_cast<uint32_t>(sizeof(uint32_t)));
    auto result_buffer = cross_node_dfb_test::make_cross_node_data_buffer(device, receiver_cores, result_page_size, 1);

    auto build_relay_and_trisc = [&](Program& program) {
        EXPECT_EQ(
            AttachPrefetcherPipe(
                program, pipe, receiver_cores, recv_entry_size, /*num_pipe_consumer_threads=*/params.num_producers),
            0u);
        experimental::dfb::DataflowBufferConfig relay_config{
            .entry_size = recv_entry_size,
            .num_entries = recv_num_entries,
            .num_producers = params.num_producers,
            .pap = params.pap,
            .num_consumers = params.num_consumers,
            .cap = params.cap,
        };
        const uint32_t relay_host_id =
            experimental::CreatePrefetcherPipeRelayDataflowBuffer(program, receiver_cores, relay_config, 0);
        const uint32_t relay_device_slot = program.impl().get_dataflow_buffer(relay_host_id)->device_slot;

        const KernelHandle receiver_kernel = create_dm_kernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_relay_receiver.cpp",
            receiver_cores,
            {0u, recv_total_entries, pipe_batch_size},
            {},
            /*num_threads_per_cluster=*/params.num_producers);
        const KernelHandle trisc_kernel = create_compute_kernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_relay_trisc.cpp",
            receiver_cores,
            {relay_device_slot, entries_per_consumer, params.batch_size, params.trisc_delay_iterations, 0u},
            /*num_threads_per_cluster=*/params.num_consumers);

        experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program, relay_host_id, receiver_kernel, trisc_kernel);
        SetRuntimeArgs(program, trisc_kernel, receiver_cores, {static_cast<uint32_t>(result_buffer->address())});
        return std::pair<KernelHandle, KernelHandle>{receiver_kernel, trisc_kernel};
    };

    if (params.same_program) {
        Program program = CreateProgram();
        EXPECT_EQ(
            AttachPrefetcherPipe(
                program, pipe, pipe.all_cores(), params.entry_size, /*num_pipe_consumer_threads=*/params.num_producers),
            0u);
        experimental::dfb::DataflowBufferConfig relay_config{
            .entry_size = recv_entry_size,
            .num_entries = recv_num_entries,
            .num_producers = params.num_producers,
            .pap = params.pap,
            .num_consumers = params.num_consumers,
            .cap = params.cap,
        };
        const uint32_t relay_host_id =
            experimental::CreatePrefetcherPipeRelayDataflowBuffer(program, receiver_cores, relay_config, 0);
        const uint32_t relay_device_slot = program.impl().get_dataflow_buffer(relay_host_id)->device_slot;

        const KernelHandle sender_kernel = create_dm_kernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_sender.cpp",
            sender_cores,
            {0u, params.entry_size, params.total_entries, 0u, data_pattern, 0u},
            {},
            /*num_threads_per_cluster=*/params.num_sender_threads);
        const KernelHandle receiver_kernel = create_dm_kernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_relay_receiver.cpp",
            receiver_cores,
            {0u, recv_total_entries, pipe_batch_size},
            {},
            /*num_threads_per_cluster=*/params.num_producers);
        const KernelHandle trisc_kernel = create_compute_kernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_relay_trisc.cpp",
            receiver_cores,
            {relay_device_slot, entries_per_consumer, params.batch_size, params.trisc_delay_iterations, 0u},
            /*num_threads_per_cluster=*/params.num_consumers);

        experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program, relay_host_id, receiver_kernel, trisc_kernel);
        prefetcher_pipe_test::write_sender_l1_staging(
            device, sender_cores, pipe, data_pattern, params.entry_size, params.total_entries, 1);
        prefetcher_pipe_test::set_sender_l1_staging_runtime_args(program, sender_kernel, sender_cores, pipe);
        SetRuntimeArgs(program, trisc_kernel, receiver_cores, {static_cast<uint32_t>(result_buffer->address())});

        distributed::MeshWorkload workload;
        persistent_run_on_mesh_device(mesh_device, std::move(program), workload);
    } else {
        // Metal2 shape: pipe already exists; bind consumer then sender; enqueue sender
        // first is fine because relay registration already wrote active lanes to the pipe.
        Program program_consumer = CreateProgram();
        build_relay_and_trisc(program_consumer);

        Program program_sender = CreateProgram();
        EXPECT_EQ(AttachPrefetcherPipe(program_sender, pipe, sender_cores, params.entry_size), 0u);
        KernelHandle sender_k;
        if (params.num_sender_threads > 1) {
            sender_k = create_dm_kernel(
                program_sender,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_sender.cpp",
                sender_cores,
                {0u, params.entry_size, params.total_entries, /*write_primitive=*/0, data_pattern, /*do_barrier=*/0},
                {},
                params.num_sender_threads);
        } else {
            sender_k = create_dm_kernel(
                program_sender,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_sender.cpp",
                sender_cores,
                {0u, params.entry_size, params.total_entries, /*write_primitive=*/0, data_pattern, /*do_barrier=*/0});
        }
        prefetcher_pipe_test::write_sender_l1_staging(
            device, sender_cores, pipe, data_pattern, params.entry_size, params.total_entries, 1);
        prefetcher_pipe_test::set_sender_l1_staging_runtime_args(program_sender, sender_k, sender_cores, pipe);

        // Consumer bind already armed lanes; overlap sender with relay/TRISC consumer.
        persistent_run_overlapping_programs(mesh_device, std::move(program_sender), std::move(program_consumer));
    }

    const uint32_t expected_checksum = prefetcher_pipe_relay_expected_checksum(recv_total_entries);
    uint32_t pass_count = 0;
    for (const CoreCoord& receiver_core : corerange_to_cores(receiver_cores)) {
        std::vector<uint32_t> result(result_words, 0);
        slow_dispatch::ReadFromL1(
            device,
            receiver_core,
            static_cast<uint32_t>(result_buffer->address()),
            std::span<uint8_t>(reinterpret_cast<uint8_t*>(result.data()), result.size() * sizeof(uint32_t)),
            CoreType::WORKER);

        bool ok = true;
        if (params.cap == experimental::dfb::AccessPattern::ALL) {
            for (uint32_t tid = 0; tid < params.num_consumers; ++tid) {
                if (result[tid * 2 + 0] != entries_per_consumer || result[tid * 2 + 1] != expected_checksum) {
                    ok = false;
                    log_error(
                        tt::LogTest,
                        "PrefetcherPipe relay ALL mismatch on {} tid {}: count {} (expected {}), checksum 0x{:08x} "
                        "(expected 0x{:08x})",
                        receiver_core.str(),
                        tid,
                        result[tid * 2 + 0],
                        entries_per_consumer,
                        result[tid * 2 + 1],
                        expected_checksum);
                }
            }
        } else {
            uint32_t got_entries = 0;
            uint32_t got_checksum = 0;
            for (uint32_t tid = 0; tid < params.num_consumers; ++tid) {
                if (result[tid * 2 + 0] != entries_per_consumer) {
                    ok = false;
                    log_error(
                        tt::LogTest,
                        "PrefetcherPipe relay STRIDED mismatch on {} tid {}: count {} (expected {})",
                        receiver_core.str(),
                        tid,
                        result[tid * 2 + 0],
                        entries_per_consumer);
                }
                got_entries += result[tid * 2 + 0];
                got_checksum += result[tid * 2 + 1];
            }
            if (got_entries != recv_total_entries || got_checksum != expected_checksum) {
                ok = false;
                log_error(
                    tt::LogTest,
                    "PrefetcherPipe relay STRIDED aggregate mismatch on {}: count {} (expected {}), checksum "
                    "0x{:08x} (expected 0x{:08x}); per-tid checksums:",
                    receiver_core.str(),
                    got_entries,
                    recv_total_entries,
                    got_checksum,
                    expected_checksum);
                for (uint32_t tid = 0; tid < params.num_consumers; ++tid) {
                    log_error(
                        tt::LogTest,
                        "  tid {}: count {} checksum 0x{:08x}",
                        tid,
                        result[tid * 2 + 0],
                        result[tid * 2 + 1]);
                }
            }
        }
        if (ok) {
            ++pass_count;
        }
    }
    return pass_count;
}

// Backward-compatible wrapper for existing 1P1C call sites.
static uint32_t run_prefetcher_pipe_relay_cross_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PrefetcherPipe& pipe,
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

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_CrossProgram_DMToCompute) {
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    constexpr uint32_t total_entries = 4;
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};
    auto pipe =
        experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, entry_size * ring_depth);
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::ALL,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(),
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        entry_size * ring_depth,
        BufferType::L1);
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
                .pap = experimental::dfb::AccessPattern::STRIDED,
                .num_consumers = 2,
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(),
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        entry_size * ring_depth,
        BufferType::L1);
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
                .pap = experimental::dfb::AccessPattern::STRIDED,
                .num_consumers = 2,
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(),
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        entry_size * ring_depth,
        BufferType::L1);
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
                .pap = experimental::dfb::AccessPattern::STRIDED,
                .num_consumers = 2,
                .cap = experimental::dfb::AccessPattern::ALL,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(),
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        entry_size * ring_depth,
        BufferType::L1);
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
                .pap = experimental::dfb::AccessPattern::STRIDED,
                .num_consumers = 4,
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(),
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        entry_size * ring_depth,
        BufferType::L1);
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
                .pap = experimental::dfb::AccessPattern::STRIDED,
                .num_consumers = 4,
                .cap = experimental::dfb::AccessPattern::ALL,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(),
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        entry_size * ring_depth,
        BufferType::L1);
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
                .pap = experimental::dfb::AccessPattern::STRIDED,
                .num_consumers = 4,
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(),
        CoreCoord(0, 0),
        CoreRangeSet(CoreRange({1, 0}, {1, 0})),
        entry_size * ring_depth,
        BufferType::L1);
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
                .pap = experimental::dfb::AccessPattern::STRIDED,
                .num_consumers = 2,
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::ALL,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::ALL,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::ALL,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
                .cap = experimental::dfb::AccessPattern::STRIDED,
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
    auto pipe = experimental::CreatePrefetcherPipe(
        mesh_device.get(), CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0})), entry_size * ring_depth);
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
    // Full drain: A finishes on E1, then B Attach/relay with E2.
    auto mesh_device = devices_[0];
    constexpr uint32_t e1 = 256;
    constexpr uint32_t e2 = 512;
    constexpr uint32_t ring_depth_e1 = 4;  // ring bytes = 1024 → 2 entries at E2
    constexpr uint32_t total_entries_e1 = 2;
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};
    auto pipe =
        experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, e1 * ring_depth_e1);
    EXPECT_EQ(
        run_prefetcher_pipe_relay_cross_program(
            mesh_device, pipe, e1, ring_depth_e1, total_entries_e1, /*batch_size=*/1, e2),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_RelayDFB_EntrySizeNotDividingRing) {
    // A relay over an entry size the ring does not divide: 384 B entries in a 1024 B ring leave a
    // 768 B usable limit and a 256 B trailing gap, so the relay borrows the 2 whole entries the
    // ring holds. The second push reaches the usable limit and credits the gap along with its
    // payload, which is what both the receiver and its relay have to skip at the wrap.
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 384;
    constexpr uint32_t ring_bytes = 1024;
    constexpr uint32_t whole_entries = ring_bytes / entry_size;  // 2
    const std::pair<CoreCoord, CoreRangeSet> mapping = {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))};
    auto pipe = experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, ring_bytes);
    EXPECT_EQ(
        run_prefetcher_pipe_relay_cross_program(
            mesh_device,
            pipe,
            entry_size,
            /*ring_depth=*/whole_entries,
            /*total_entries=*/whole_entries,
            /*batch_size=*/1),
        1u);
}

TEST_F(PrefetcherPipeFixture, PrefetcherPipe_CrossSubDevice_CoordinatedLivePeerNonDividingE2) {
    if (!is_fast_dispatch()) {
        GTEST_SKIP() << "Sub device managers are unsupported with slow dispatch";
    }
    if (const auto reason = insufficient_worker_grid_reason(2); !reason.empty()) {
        GTEST_SKIP() << reason;
    }
    // Live-peer E1→E2 prefetch with a non-dividing E2:
    //   A (SD0): push E1 → set_entry_size(E2) without draining → signal → wait go → push E2
    //   B (SD1): Attach(E1), pop E1 → set_receiver_entry_size(E2), finish
    //   C (SD1): Attach(E2) while A is still alive; consume E2
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
    auto resized_sem = CreateGlobalSemaphore(mesh_device.get(), sender_cores, /*initial_value=*/0);
    auto go_sem = CreateGlobalSemaphore(mesh_device.get(), sender_cores, /*initial_value=*/0);

    const std::pair<CoreCoord, CoreRangeSet> mapping = {sender_core, receiver_cores};
    auto pipe =
        experimental::CreatePrefetcherPipe(mesh_device.get(), mapping.first, mapping.second, e1 * ring_depth_e1);
    distributed::Synchronize(*mesh_device, std::nullopt);

    // --- Program A (SD0): push E1 → resize without drain → signal → wait go → push E2 ---
    Program program_a = CreateProgram();
    EXPECT_EQ(AttachPrefetcherPipe(program_a, pipe, sender_cores, e1), 0u);
    const KernelHandle sender_k = create_dm_kernel(
        program_a,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_coordinated_resize_sender.cpp",
        sender_cores,
        {0u, e1, total_entries_e1, e2, total_entries_e2, data_pattern});
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
    SetRuntimeArgs(
        program_a,
        sender_k,
        sender_cores,
        {prefetcher_pipe_test::sender_l1_staging_address(pipe),
         static_cast<uint32_t>(resized_sem.address()),
         static_cast<uint32_t>(go_sem.address()),
         credit_base});

    distributed::MeshWorkload workload_a;
    workload_a.add_program(persistent_unit_mesh_device_range(), std::move(program_a));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_a, false);

    // Subsequent FD work waits for SD1 idle (not long-running A on SD0).
    mesh_device->set_sub_device_stall_group({{SubDeviceId{1}}});

    // A must resize before any receiver is enqueued. This would time out if
    // set_entry_size still contained an acked == sent barrier.
    const auto device_id = mesh_device->get_devices()[0]->id();
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
    Program program_b = CreateProgram();
    EXPECT_EQ(AttachPrefetcherPipe(program_b, pipe, receiver_cores, e1), 0u);
    create_dm_kernel(
        program_b,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_coordinated_resize_receiver.cpp",
        receiver_cores,
        {0u, total_entries_e1, e2});
    distributed::MeshWorkload workload_b;
    workload_b.add_program(persistent_unit_mesh_device_range(), std::move(program_b));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_b, false);

    // --- Program C (SD1): same-epoch Attach E2 while A is still alive ---
    Program program_c = CreateProgram();
    EXPECT_EQ(AttachPrefetcherPipe(program_c, pipe, receiver_cores, e2), 0u);
    create_dm_kernel(
        program_c,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/prefetcher_pipe_receiver.cpp",
        receiver_cores,
        {0u, e2, total_entries_e2, 0u});

    distributed::MeshWorkload workload_c;
    workload_c.add_program(persistent_unit_mesh_device_range(), std::move(program_c));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_c, false);

    // Release A to push at E2; C is already attached/running on the peer sub-device.
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

}  // namespace tt::tt_metal
