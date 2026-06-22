// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/core_coord.hpp>
#include "impl/dataflow_buffer/cross_node_dfb.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <exception>
#include <map>
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
#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/api/cross_node_dfb_test_utils.hpp"

namespace tt::tt_metal {

// ---------------------------------------------------------------------------
// Group 1: Direct mirrors of test_global_circular_buffers.cpp
// ---------------------------------------------------------------------------

TEST_F(MeshDispatchFixture, TensixCreateCrossNodeDFBs) {
    CoreRangeSet cores(CoreRange({1, 1}, {1, 1}));
    CoreRangeSet cores2(CoreRange({1, 1}, {2, 2}));
    CoreRangeSet cores3(CoreRange({3, 3}, {3, 3}));
    auto mesh_device = devices_[0];

    // Valid 1:1 mapping - should not throw.
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), cores}};
        EXPECT_NO_THROW(experimental::CreateCrossNodeDFB(
            mesh_device.get(), mapping, /*entry_size=*/256, /*num_entries=*/4));
    }
    // Sender core appears in its own receiver CoreRangeSet (sender-receiver overlap).
    {
        CoreRangeSet overlap_cores(CoreRange({0, 0}, {0, 0}));
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), overlap_cores}};
        EXPECT_THROW(
            experimental::CreateCrossNodeDFB(
                mesh_device.get(), mapping, 256, 4),
            std::exception);
    }
    // Two senders share a receiver core (receiver sets overlap across senders).
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
            {CoreCoord(0, 0), cores}, {CoreCoord(0, 1), cores2}};
        EXPECT_THROW(
            experimental::CreateCrossNodeDFB(
                mesh_device.get(), mapping, 256, 4),
            std::exception);
    }
}

TEST_F(MeshDispatchFixture, TensixProgramCrossNodeDFBsAPI) {
    CoreCoord sender_core = CoreCoord(0, 0);
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    CoreRangeSet receiver_cores(CoreRange({1, 1}, {2, 2}));
    CoreRangeSet dummy_receiver_cores(CoreRange({3, 3}, {3, 3}));
    auto all_cores = sender_cores.merge(receiver_cores).merge(dummy_receiver_cores);

    auto mesh_device = devices_[0];

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto cross_node_dfb = experimental::CreateCrossNodeDFB(
        mesh_device.get(), mapping, 256, 4);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> dummy_mapping = {
        {CoreCoord(0, 0), dummy_receiver_cores}};
    auto dummy_cross_node_dfb = experimental::CreateCrossNodeDFB(
        mesh_device.get(), dummy_mapping, 256, 4);

    // Valid: attach to the correct receiver cores, no throw.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default});

        EXPECT_NO_THROW(
            experimental::AttachCrossNodeDFB(program, receiver_cores, cross_node_dfb));

        tt::tt_metal::detail::CompileProgram(mesh_device.get(), program);
        program.impl().finalize_offsets(mesh_device.get());

        // CrossNodeDFB config is reserved after finalize. Presence is in launch msg num_cross_node_dfbs;
        // remote_cross_node_dfb_offset may be 0 when CrossNodeDFB is the first kernel-config region.
        const auto& hal = MetalContext::instance().hal();
        uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        EXPECT_FALSE(program.impl().get_per_core_cross_node_dfbs().empty());
        const auto& kernel_groups = program.impl().get_kernel_groups(index);
        ASSERT_FALSE(kernel_groups.empty());
        EXPECT_GT(kernel_groups[0]->launch_msg.view().kernel_config().num_cross_node_dfbs(), static_cast<uint8_t>(0));
    }
    // Throw: attach to cores not in the CrossNodeDFB all_cores.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default});
        EXPECT_THROW(
            experimental::AttachCrossNodeDFB(
                program, dummy_receiver_cores, cross_node_dfb),
            std::exception);
    }
    // UpdateDynamicCrossNodeDFBAddress: valid case - succeeds.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default});
        experimental::AttachCrossNodeDFB(program, receiver_cores, cross_node_dfb);
        tt::tt_metal::detail::CompileProgram(mesh_device.get(), program);
        program.impl().finalize_offsets(mesh_device.get());
        EXPECT_NO_THROW(experimental::UpdateDynamicCrossNodeDFBAddress(program, cross_node_dfb));
    }
    // UpdateDynamicCrossNodeDFBAddress: invalid case - throws when gdfb does not match.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default});
        experimental::AttachCrossNodeDFB(program, receiver_cores, cross_node_dfb);
        tt::tt_metal::detail::CompileProgram(mesh_device.get(), program);
        program.impl().finalize_offsets(mesh_device.get());
        EXPECT_THROW(
            experimental::UpdateDynamicCrossNodeDFBAddress(program, dummy_cross_node_dfb),
            std::exception);
    }
    // No CrossNodeDFBs attached: remote_cross_node_dfb_offset must be 0.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default});
        tt::tt_metal::detail::CompileProgram(mesh_device.get(), program);
        program.impl().finalize_offsets(mesh_device.get());
        const auto& hal = MetalContext::instance().hal();
        uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        EXPECT_EQ(program.impl().get_program_config(index).remote_cross_node_dfb_offset, 0u);
    }
}

// ---------------------------------------------------------------------------
// Group 2: DFB-specific host-API tests (no kernel execution)
// ---------------------------------------------------------------------------

TEST_F(MeshDispatchFixture, TensixCreateCrossNodeDFBs_MultiSender) {
    auto mesh_device = devices_[0];

    // Valid M:N: 2 independent senders, each with a disjoint CoreRangeSet.
    {
        CoreRangeSet recv0(CoreRange({2, 0}, {3, 0}));
        CoreRangeSet recv1(CoreRange({2, 1}, {3, 1}));
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
            {CoreCoord(0, 0), recv0},
            {CoreCoord(1, 0), recv1}};
        EXPECT_NO_THROW(experimental::CreateCrossNodeDFB(
            mesh_device.get(), mapping, 256, 4));
    }
    // Single sender with 2 receivers: creates without error.
    {
        CoreRangeSet recv(CoreRange({1, 0}, {2, 0}));
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), recv}};
        EXPECT_NO_THROW(experimental::CreateCrossNodeDFB(
            mesh_device.get(), mapping, 256, 4));
    }
}

TEST_F(MeshDispatchFixture, TensixProgramCrossNodeDFBsAPI_RelayDFBNames) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {2, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    auto all_cores = sender_cores.merge(receiver_cores);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto gdfb = experimental::CreateCrossNodeDFB(
        mesh_device.get(), mapping, 256, 4);

    // Valid: relay DFB name provided; compiles and finalizes without throw.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default});

        EXPECT_NO_THROW(experimental::AttachCrossNodeDFB(
            program, receiver_cores, gdfb, {"local_dfb_0"}));
        tt::tt_metal::detail::CompileProgram(mesh_device.get(), program);
        EXPECT_NO_THROW(program.impl().finalize_offsets(mesh_device.get()));
    }
    // Verify relay_dfb_names are stored in per_core_cross_node_dfbs_.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default});
        EXPECT_NO_THROW(experimental::AttachCrossNodeDFB(
            program, receiver_cores, gdfb, {"nonexistent_dfb_handle"}));

        detail::ProgramImpl& impl = program.impl();
        auto it = impl.get_per_core_cross_node_dfbs().find(CoreCoord(1, 0));
        ASSERT_NE(it, impl.get_per_core_cross_node_dfbs().end());
        ASSERT_FALSE(it->second.empty());
        EXPECT_EQ(it->second[0].relay_dfb_names[0], "nonexistent_dfb_handle");
    }
    // Verify auto_commit is enabled by default.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default});
        EXPECT_NO_THROW(experimental::AttachCrossNodeDFB(program, receiver_cores, gdfb));

        detail::ProgramImpl& impl = program.impl();
        auto it = impl.get_per_core_cross_node_dfbs().find(CoreCoord(1, 0));
        ASSERT_NE(it, impl.get_per_core_cross_node_dfbs().end());
        ASSERT_FALSE(it->second.empty());
        EXPECT_TRUE(it->second[0].auto_commit);
        EXPECT_EQ(it->second[0].remote_dfb_id, 0u);
    }
    // Verify auto_commit can be disabled explicitly.
    {
        tt_metal::Program program = CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default});
        EXPECT_NO_THROW(experimental::AttachCrossNodeDFB(
            program, receiver_cores, gdfb, {}, /*auto_commit=*/false));

        detail::ProgramImpl& impl = program.impl();
        auto it = impl.get_per_core_cross_node_dfbs().find(CoreCoord(1, 0));
        ASSERT_NE(it, impl.get_per_core_cross_node_dfbs().end());
        ASSERT_FALSE(it->second.empty());
        EXPECT_FALSE(it->second[0].auto_commit);
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
    const char* label,
    uint32_t entry_size,
    uint32_t num_entries,
    const std::vector<uint8_t>& bytes) {
    log_info(
        tt::LogTest,
        "{}: {} entries x {} B ({} B total)",
        label,
        num_entries,
        entry_size,
        bytes.size());
    for (uint32_t i = 0; i < num_entries; ++i) {
        const uint32_t off = i * entry_size;
        if (off >= bytes.size()) {
            log_info(tt::LogTest, "  entry[{}]: (out of range)", i);
            continue;
        }
        const uint8_t first = bytes[off];
        const uint32_t check_len = std::min(entry_size, static_cast<uint32_t>(bytes.size() - off));
        const bool uniform =
            std::all_of(bytes.begin() + off + 1, bytes.begin() + off + check_len, [first](uint8_t b) {
                return b == first;
            });
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
    uint32_t entry_size,
    const std::vector<uint8_t>& expected,
    const std::vector<uint8_t>& received) {
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
        log_info(
            tt::LogTest,
            "  size mismatch: expected {} B, received {} B",
            expected.size(),
            received.size());
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
    bool do_commit = false,
    uint32_t counter_base = 0) {
    IDevice* device = mesh_device.get();
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto gdfb = experimental::CreateCrossNodeDFB(
        device, mapping, entry_size, num_entries);

    const uint32_t num_receivers = static_cast<uint32_t>(corerange_to_cores(receiver_cores).size());
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(write_primitive);
    const uint32_t staging_size = cross_node_dfb_test::sender_staging_size_bytes(
        data_pattern, entry_size, num_entries, num_receivers);
    tt_metal::Program program = CreateProgram();

    const uint8_t remote_dfb_id = experimental::AttachCrossNodeDFB(program, sender_cores, gdfb);
    experimental::AttachCrossNodeDFB(program, receiver_cores, gdfb);

    tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries,
                             write_primitive,
                             static_cast<uint32_t>(do_commit),
                             data_pattern,
                             0u, 0u, 0u, 0u}});

    auto recvs = corerange_to_cores(receiver_cores);
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        CoreRangeSet single = CoreRangeSet(CoreRange(recvs[ri]));
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_receiver.cpp",
            single,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, num_entries,
                                 ri,
                                 static_cast<uint32_t>(do_commit),
                                 0u,
                                 0u, 0u, 0u, 0u}});
    }

    // Runtime args must be set before run_on_mesh_device (which internally calls
    // finalize_offsets).  finalize_offsets sizes the RTA region from the currently-set
    // args; if it runs with 0 args the CrossNodeDFB config ends up at the same offset
    // as the RTA slot and overwrites it during dispatch.
    cross_node_dfb_test::set_sender_l1_staging_runtime_args(
        program, sender_k, sender_cores, gdfb, staging_size);

    const auto sender_staging_bytes = cross_node_dfb_test::build_sender_staging_bytes(
        data_pattern, entry_size, num_entries, num_receivers, counter_base);
    cross_node_dfb_test::write_sender_l1_staging(
        device, sender_cores, gdfb, data_pattern, entry_size, num_entries, num_receivers, counter_base);

    log_info(
        tt::LogTest,
        "run_1toN_program: sender=({},{}), receivers={}, write_primitive={}, data_pattern={}, "
        "entry_size={}, num_entries={}, counter_base={}, dfb_buffer=0x{:x}, config_buffer=0x{:x}",
        sender_core.x,
        sender_core.y,
        num_receivers,
        write_primitive,
        data_pattern,
        entry_size,
        num_entries,
        counter_base,
        gdfb.buffer_address(),
        gdfb.config_address());
    log_info(
        tt::LogTest,
        "sender L1 staging=0x{:x} (size {} B, host-written)",
        cross_node_dfb_test::sender_l1_staging_address(gdfb, staging_size),
        staging_size);
    log_cross_node_dfb_byte_summary("sender staging (host-written L1 payload)", entry_size, num_entries, sender_staging_bytes);

    distributed::MeshWorkload workload;
    run_on_mesh_device(mesh_device, std::move(program), workload);

    uint32_t pass_count = 0;
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        const auto expected = cross_node_dfb_test::expected_receiver_ring_bytes(
            data_pattern, entry_size, num_entries, ri, num_receivers, counter_base);
        const auto received = cross_node_dfb_test::read_receiver_ring_bytes(
            device, gdfb, recvs[ri], static_cast<uint32_t>(expected.size()));
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
    return pass_count;
}

TEST_F(MeshDispatchFixture, TensixBasicPushPop_1to1) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    // write_primitive=0 (multicast) with 1 receiver is a basic 1:1 test.
    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 0);
    EXPECT_EQ(pass, 1u) << "1:1 basic push/pop failed";
}

TEST_F(MeshDispatchFixture, TensixWriteMulticast_1to4) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 0);
    EXPECT_EQ(pass, 4u) << "write_multicast 1:4 failed";
}

TEST_F(MeshDispatchFixture, TensixWriteStrided_1to4) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    // write_primitive=1: sender writes interleaved, each receiver verifies its index pattern.
    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 1);
    EXPECT_EQ(pass, 4u) << "write_strided 1:4 failed";
}

TEST_F(MeshDispatchFixture, TensixWriteToReceiver_ReceiverContiguous) {
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

TEST_F(MeshDispatchFixture, TensixRoundRobinPushBackToReceiver) {
    auto mesh_device = devices_[0];

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 1;

    // write_primitive=3: write_to_receiver + push_back_to_receiver per iteration.
    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 3);
    EXPECT_EQ(pass, 4u) << "write_to_receiver + push_back_to_receiver round-robin failed";
}

TEST_F(MeshDispatchFixture, TensixDecoupledWriteThenCredit) {
    auto mesh_device = devices_[0];

    // Sender issues multiple write_multicast calls before a single push_back.
    // Verifies that credits are not prematurely visible.
    // We use num_entries=1 and write_primitive=0 (multicast) for simplicity;
    // the test is about verifying no hang (credits not visible before writes).
    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    uint32_t pass = run_1toN_program(mesh_device, sender_core, receiver_cores, entry_size, num_entries, 0);
    EXPECT_EQ(pass, 4u) << "Decoupled write-then-credit test failed";
}

TEST_F(MeshDispatchFixture, TensixMultipleSenders_MtoN) {
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

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {sender0, recv0}, {sender1, recv1}};
    auto gdfb = experimental::CreateCrossNodeDFB(
        device, mapping, entry_size, num_entries);

    tt_metal::Program program = CreateProgram();

    const uint8_t remote_dfb_id = experimental::AttachCrossNodeDFB(program, sender_cores, gdfb);
    experimental::AttachCrossNodeDFB(program, receiver_cores, gdfb);

    auto recvs = corerange_to_cores(receiver_cores);
    const uint32_t num_receivers = static_cast<uint32_t>(recvs.size());
    constexpr uint32_t data_pattern =
        static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    const uint32_t staging_size = cross_node_dfb_test::sender_staging_size_bytes(
        data_pattern, entry_size, num_entries, num_receivers);
    tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries,
                             0u, 0u, data_pattern, 0u, 0u, 0u, 0u}});
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        CoreRangeSet single = CoreRangeSet(CoreRange(recvs[ri]));
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_receiver.cpp",
            single,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, num_entries,
                                 ri, 0u, 0u, 0u, 0u, 0u, 0u}});
    }

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(
        program, sender_k, sender_cores, gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        device, sender_cores, gdfb, data_pattern, entry_size, num_entries, num_receivers);

    distributed::MeshWorkload workload;
    run_on_mesh_device(mesh_device, std::move(program), workload);

    uint32_t pass_count = 0;
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        if (cross_node_dfb_test::verify_receiver_ring(
                device, gdfb, recvs[ri], data_pattern, entry_size, num_entries, ri, num_receivers)) {
            pass_count++;
        }
    }
    EXPECT_EQ(pass_count, 4u) << "Not all M:N receivers received expected data";
}

TEST_F(MeshDispatchFixture, TensixCrossProgramPersistence) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 8;
    const uint32_t entries_per_program = 4;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto gdfb = experimental::CreateCrossNodeDFB(
        device, mapping, entry_size, num_entries);

    // build_program returns the program for a single batch of entries.
    auto build_program = [&](uint32_t counter_base) {
        tt_metal::Program program = CreateProgram();
        const uint8_t remote_dfb_id = experimental::AttachCrossNodeDFB(program, sender_cores, gdfb);
        experimental::AttachCrossNodeDFB(program, receiver_cores, gdfb);

        constexpr uint32_t data_pattern =
            static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
        const uint32_t staging_size = cross_node_dfb_test::sender_staging_size_bytes(
            data_pattern, entry_size, entries_per_program, 1);
        tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_sender.cpp",
            sender_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, entries_per_program,
                                 0u,  // write_multicast
                                 1u,  // do_commit
                                 data_pattern,
                                 0u, 0u, 0u, 0u}});
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_receiver.cpp",
            receiver_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, entries_per_program,
                                 0u,
                                 1u,
                                 0u,
                                 0u, 0u, 0u, 0u}});
        cross_node_dfb_test::set_sender_l1_staging_runtime_args(
            program, sender_k, sender_cores, gdfb, staging_size);
        cross_node_dfb_test::write_sender_l1_staging(
            device, sender_cores, gdfb, data_pattern, entry_size, entries_per_program, 1, counter_base);
        return program;
    };

    auto program1 = build_program(0);
    auto program2 = build_program(entries_per_program);

    distributed::MeshWorkload workload1;
    run_on_mesh_device(mesh_device, std::move(program1), workload1);
    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        device, gdfb, receiver_core, static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter),
        entry_size, entries_per_program, 0, 1, 0))
        << "Cross-program persistence: first program data mismatch";

    distributed::MeshWorkload workload2;
    run_on_mesh_device(mesh_device, std::move(program2), workload2);
    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        device, gdfb, receiver_core, static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter),
        entry_size, entries_per_program, 0, 1, entries_per_program))
        << "Cross-program persistence: second program data mismatch";
}

TEST_F(MeshDispatchFixture, TensixAutoCommitPersistence) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 8;
    const uint32_t entries_per_program = 4;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto gdfb = experimental::CreateCrossNodeDFB(
        device, mapping, entry_size, num_entries);

    auto build_program = [&](uint32_t counter_base) {
        tt_metal::Program program = CreateProgram();
        const uint8_t remote_dfb_id = experimental::AttachCrossNodeDFB(program, sender_cores, gdfb);
        experimental::AttachCrossNodeDFB(program, receiver_cores, gdfb);

        constexpr uint32_t data_pattern =
            static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
        const uint32_t staging_size = cross_node_dfb_test::sender_staging_size_bytes(
            data_pattern, entry_size, entries_per_program, 1);
        tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_sender.cpp",
            sender_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, entries_per_program,
                                 0u,  // write_multicast
                                 0u,  // do_commit=0 (auto-commit does it)
                                 data_pattern,
                                 0u, 0u, 0u, 0u}});
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_receiver.cpp",
            receiver_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, entries_per_program,
                                 0u,
                                 0u,
                                 0u,
                                 0u, 0u, 0u, 0u}});
        cross_node_dfb_test::set_sender_l1_staging_runtime_args(
            program, sender_k, sender_cores, gdfb, staging_size);
        cross_node_dfb_test::write_sender_l1_staging(
            device, sender_cores, gdfb, data_pattern, entry_size, entries_per_program, 1, counter_base);
        return program;
    };

    auto program1 = build_program(0);
    auto program2 = build_program(entries_per_program);

    distributed::MeshWorkload workload1;
    run_on_mesh_device(mesh_device, std::move(program1), workload1);
    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        device, gdfb, receiver_core, static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter),
        entry_size, entries_per_program, 0, 1, 0))
        << "Auto-commit persistence: first program data mismatch";

    distributed::MeshWorkload workload2;
    run_on_mesh_device(mesh_device, std::move(program2), workload2);
    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        device, gdfb, receiver_core, static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter),
        entry_size, entries_per_program, 0, 1, entries_per_program))
        << "Auto-commit persistence: second program data mismatch";
}


TEST_F(MeshDispatchFixture, TensixMidFlightResize) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);

    const uint32_t entry_size_initial = 256;
    const uint32_t entry_size_resized = 512;
    const uint32_t num_entries_initial = 2;
    const uint32_t num_entries_after = 2;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto gdfb = experimental::CreateCrossNodeDFB(
        device, mapping, entry_size_initial, num_entries_initial + num_entries_after);

    tt_metal::Program program = CreateProgram();

    const uint8_t remote_dfb_id = experimental::AttachCrossNodeDFB(program, sender_cores, gdfb);
    experimental::AttachCrossNodeDFB(program, receiver_cores, gdfb);

    constexpr uint32_t data_pattern =
        static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    const uint32_t staging_size = cross_node_dfb_test::sender_staging_size_bytes(
        data_pattern,
        entry_size_initial,
        num_entries_initial,
        1,
        entry_size_resized,
        num_entries_after);
    tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size_initial, num_entries_initial,
                             0u, 0u, data_pattern,
                             1u, entry_size_resized, num_entries_after, 0u}});
    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_receiver.cpp",
        receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size_initial, num_entries_initial,
                             0u, 0u, 0u,
                             1u, entry_size_resized, num_entries_after, 0u}});

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(
        program, sender_k, sender_cores, gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        device, sender_cores, gdfb,
        data_pattern, entry_size_initial, num_entries_initial, 1, 0,
        entry_size_resized, num_entries_after);

    distributed::MeshWorkload workload;
    run_on_mesh_device(mesh_device, std::move(program), workload);

    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        device,
        gdfb,
        receiver_core,
        data_pattern,
        entry_size_initial,
        num_entries_initial,
        0,
        1,
        0,
        entry_size_resized,
        num_entries_after))
        << "Mid-flight resize test failed";
}

// TensixRelayDFBAlignment: verifies mid-flight resize correctly propagates fifo_limit
// and fifo_page_size into relay DFBs registered via register_relay_dfbs().
//
// Relay DFB pattern (DM + TRISC split):
//   DM (NCRISC/BRISC):
//     LocalDFB relay(handle);            // pre-allocated, shares CrossNodeDFB FIFO
//     gdfb.register_relay_dfbs(relay);  // align relay; auto-realign on resize
//     gdfb.pop_front(n);                // DM advances rd_ptr + acks sender
//     gdfb.set_receiver_entry_size(sz); // relay fifo_limit + fifo_page_size re-synced
//   TRISC (reads relay, no CrossNodeDFB knowledge):
//     cb_wait_front(relay_id, n);       // polls pages_sent counter
//     cb_pop_front(relay_id, n);        // TRISC advances its own local rd_ptr
//                                       // (DM's pop_front already acked sender)
//
// In this test there is no TRISC kernel; the DM receiver pops entries and host verifies data.
TEST_F(MeshDispatchFixture, TensixRelayDFBAlignment) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);

    const uint32_t entry_size_initial = 256;
    const uint32_t entry_size_resized = 512;
    const uint32_t num_entries = 4;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto gdfb = experimental::CreateCrossNodeDFB(
        device, mapping, entry_size_initial, num_entries);

    tt_metal::Program program = CreateProgram();

    // Host side: attach with relay DFB name for cross-program tracking / metadata setup.
    experimental::AttachCrossNodeDFB(program, receiver_cores, gdfb, {"local_relay_dfb"});
    const uint8_t remote_dfb_id = experimental::AttachCrossNodeDFB(program, sender_cores, gdfb);

    constexpr uint32_t data_pattern =
        static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    const uint32_t staging_size = cross_node_dfb_test::sender_staging_size_bytes(
        data_pattern, entry_size_initial, num_entries, 1, entry_size_resized, num_entries);
    tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size_initial, num_entries,
                             0u, 0u, data_pattern,
                             1u, entry_size_resized, num_entries, 0u}});
    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_receiver.cpp",
        receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size_initial, num_entries,
                             0u, 0u, 0u,
                             1u, entry_size_resized, num_entries,
                             0xFFu}});

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(
        program, sender_k, sender_cores, gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        device, sender_cores, gdfb,
        data_pattern, entry_size_initial, num_entries, 1, 0,
        entry_size_resized, num_entries);

    distributed::MeshWorkload workload;
    run_on_mesh_device(mesh_device, std::move(program), workload);

    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        device,
        gdfb,
        receiver_core,
        data_pattern,
        entry_size_initial,
        num_entries,
        0,
        1,
        0,
        entry_size_resized,
        num_entries))
        << "Relay DFB alignment test failed";
}

// TensixRelayDFBTriscCrossProgramPersistence: DM relay + TRISC consumer across two programs.
// Exercises TRISC-side align_local_cbs_to_cross_node_receiver_dfb() so the relay CB picks up
// the persisted CrossNodeDFB fifo_rd_ptr on program 2 (mirrors GlobalCB persistence behavior).
TEST_F(MeshDispatchFixture, TensixRelayDFBTriscCrossProgramPersistence) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreCoord receiver_core(1, 0);

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 8;
    const uint32_t entries_per_program = 4;
    constexpr uint32_t relay_cb_index = tt::CBIndex::c_0;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto gdfb = experimental::CreateCrossNodeDFB(device, mapping, entry_size, num_entries);

    auto build_program = [&](uint32_t counter_base) {
        tt_metal::Program program = CreateProgram();

        experimental::AttachCrossNodeDFB(program, receiver_cores, gdfb, {"local_relay_dfb"});
        const uint8_t remote_dfb_id = experimental::AttachCrossNodeDFB(program, sender_cores, gdfb);

        const uint32_t ring_size = entry_size * num_entries;
        CircularBufferConfig relay_config =
            CircularBufferConfig(ring_size, {{relay_cb_index, tt::DataFormat::UInt8}})
                .set_page_size(relay_cb_index, entry_size);
        relay_config.set_globally_allocated_address(gdfb.dfb_buffer());
        CreateCircularBuffer(program, receiver_cores, relay_config);

        constexpr uint32_t data_pattern =
            static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
        const uint32_t staging_size = cross_node_dfb_test::sender_staging_size_bytes(
            data_pattern, entry_size, entries_per_program, 1);

        tt::tt_metal::KernelHandle sender_k = tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_sender.cpp",
            sender_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, entries_per_program,
                                 0u, 1u, data_pattern, 0u, 0u, 0u, 0u}});

        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_relay_receiver.cpp",
            receiver_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, entries_per_program, relay_cb_index}});

        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_relay_trisc.cpp",
            receiver_cores,
            tt::tt_metal::ComputeConfig{
                .compile_args = {remote_dfb_id, relay_cb_index, entries_per_program}});

        cross_node_dfb_test::set_sender_l1_staging_runtime_args(
            program, sender_k, sender_cores, gdfb, staging_size);
        cross_node_dfb_test::write_sender_l1_staging(
            device, sender_cores, gdfb, data_pattern, entry_size, entries_per_program, 1, counter_base);
        return program;
    };

    auto program1 = build_program(0);
    auto program2 = build_program(entries_per_program);

    distributed::MeshWorkload workload1;
    run_on_mesh_device(mesh_device, std::move(program1), workload1);
    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        device,
        gdfb,
        receiver_core,
        static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter),
        entry_size,
        entries_per_program,
        0,
        1,
        0))
        << "Relay DFB TRISC persistence: first program data mismatch";

    distributed::MeshWorkload workload2;
    run_on_mesh_device(mesh_device, std::move(program2), workload2);
    EXPECT_TRUE(cross_node_dfb_test::verify_receiver_ring(
        device,
        gdfb,
        receiver_core,
        static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter),
        entry_size,
        entries_per_program,
        0,
        1,
        entries_per_program))
        << "Relay DFB TRISC persistence: second program data mismatch";
}

TEST_F(MeshDispatchFixture, TensixBarrierCompletesAll) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();

    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));

    const uint32_t entry_size = 256;
    const uint32_t num_entries = 4;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto gdfb = experimental::CreateCrossNodeDFB(
        device, mapping, entry_size, num_entries);

    tt_metal::Program program = CreateProgram();

    const uint8_t remote_dfb_id = experimental::AttachCrossNodeDFB(program, sender_cores, gdfb);
    experimental::AttachCrossNodeDFB(program, receiver_cores, gdfb);

    constexpr uint32_t data_pattern =
        static_cast<uint32_t>(cross_node_dfb_test::SenderDataPattern::MulticastCounter);
    const uint32_t staging_size = cross_node_dfb_test::sender_staging_size_bytes(
        data_pattern, entry_size, num_entries, 4);
    tt::tt_metal::KernelHandle send_k = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_sender.cpp",
        sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {remote_dfb_id, entry_size, num_entries,
                             0u, 0u, data_pattern, 0u, 0u, 0u, 1u}});

    auto recvs = corerange_to_cores(receiver_cores);
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        CoreRangeSet single = CoreRangeSet(CoreRange(recvs[ri]));
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/api/kernels/cross_node_dfb_receiver.cpp",
            single,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = {remote_dfb_id, entry_size, num_entries,
                                 ri, 0u, 0u, 0u, 0u, 0u, 0u}});
    }

    cross_node_dfb_test::set_sender_l1_staging_runtime_args(
        program, send_k, sender_cores, gdfb, staging_size);
    cross_node_dfb_test::write_sender_l1_staging(
        device, sender_cores, gdfb, data_pattern, entry_size, num_entries, 4);

    distributed::MeshWorkload workload;
    run_on_mesh_device(mesh_device, std::move(program), workload);

    // barrier() blocks the sender until every receiver has acked all pushed entries.
    // Completion plus host ring verification is sufficient; no device semaphore needed.
    uint32_t recv_pass_count = 0;
    for (uint32_t ri = 0; ri < static_cast<uint32_t>(recvs.size()); ++ri) {
        if (cross_node_dfb_test::verify_receiver_ring(
                device, gdfb, recvs[ri], data_pattern, entry_size, num_entries, ri, 4)) {
            recv_pass_count++;
        }
    }
    EXPECT_EQ(recv_pass_count, 4u) << "Not all receivers received expected data in barrier test";
}

}  // namespace tt::tt_metal
