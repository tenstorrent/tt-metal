// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
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

#include "impl/dataflow_buffer/cross_node_dfb.hpp"
#include "impl/dataflow_buffer/persistent_dfb.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/dispatch.hpp"
#include "impl/context/metal_context.hpp"
#include "mesh_dispatch_fixture.hpp"
#include "tests/tt_metal/tt_metal/api/cross_node_dfb_test_utils.hpp"
#include "hostdev/remote_dfb_config_layout.h"
#include "tests/tt_metal/tt_metal/api/persistent_dfb_test_utils.hpp"

namespace tt::tt_metal {

class PersistentDFBFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        MeshDispatchFixture::SetUp();
        if (this->arch_ == tt::ARCH::QUASAR) {
            GTEST_SKIP() << "PersistentDFB is not supported on Quasar yet";
        }
    }
};

namespace {

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

uint32_t run_persistent_sender_push(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PersistentDFB& pdfb,
    uint32_t entry_size,
    uint32_t num_entries,
    uint8_t persistent_dfb_id) {
    IDevice* device = mesh_device.get();
    const CoreRangeSet sender_cores = pdfb.sender_cores();
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(2);
    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, num_entries, 1);

    Program program = CreateProgram();
    // Attach only sender cores — PersistentDFB is cross-program; this program owns the producer role.
    EXPECT_EQ(AttachPersistentDFB(program, pdfb, sender_cores), persistent_dfb_id);
    KernelHandle sender_k = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_sender.cpp",
        sender_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {persistent_dfb_id, entry_size, num_entries, 2u, data_pattern, 0u}});
    persistent_dfb_test::write_sender_l1_staging(device, sender_cores, pdfb, data_pattern, entry_size, num_entries, 1);
    persistent_dfb_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, pdfb, staging_size);
    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);
    return 1u;
}

uint32_t run_persistent_receiver_pop(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PersistentDFB& pdfb,
    uint32_t entry_size,
    uint32_t num_entries,
    uint8_t persistent_dfb_id) {
    IDevice* device = mesh_device.get();
    const CoreRangeSet receiver_cores = pdfb.receiver_cores();
    Program program = CreateProgram();
    // Attach only receiver cores — consumer role in a separate program.
    EXPECT_EQ(AttachPersistentDFB(program, pdfb, receiver_cores), persistent_dfb_id);
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {persistent_dfb_id, entry_size, num_entries, 0u}});
    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(2);
    return persistent_dfb_test::verify_receiver_ring(
               device, pdfb, CoreCoord(1, 0), data_pattern, entry_size, num_entries, 0, 1)
               ? 1u
               : 0u;
}

// Cross-program Persistent equivalent of CrossNode's run_1toN_program:
// Program A pushes on sender cores; Program B pops on receivers and verifies rings.
uint32_t run_persistent_1toN_cross_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PersistentDFB& pdfb,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t write_primitive,
    uint8_t persistent_dfb_id = 0,
    bool simultaneous_subdevices = false) {
    IDevice* device = mesh_device.get();
    const CoreRangeSet sender_cores = pdfb.sender_cores();
    const CoreRangeSet receiver_cores = pdfb.receiver_cores();
    const auto receivers = corerange_to_cores(receiver_cores);
    const uint32_t num_receivers = static_cast<uint32_t>(receivers.size());
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(write_primitive);
    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, num_entries, num_receivers);

    Program sender_program = CreateProgram();
    EXPECT_EQ(AttachPersistentDFB(sender_program, pdfb, sender_cores), persistent_dfb_id);
    KernelHandle sender_k = CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_sender.cpp",
        sender_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {persistent_dfb_id, entry_size, num_entries, write_primitive, data_pattern, 0u}});
    persistent_dfb_test::write_sender_l1_staging(
        device, sender_cores, pdfb, data_pattern, entry_size, num_entries, num_receivers);
    persistent_dfb_test::set_sender_l1_staging_runtime_args(sender_program, sender_k, sender_cores, pdfb, staging_size);

    Program receiver_program = CreateProgram();
    EXPECT_EQ(AttachPersistentDFB(receiver_program, pdfb, receiver_cores), persistent_dfb_id);
    for (uint32_t ri = 0; ri < num_receivers; ++ri) {
        const CoreRangeSet single = CoreRangeSet(CoreRange(receivers[ri]));
        CreateKernel(
            receiver_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_receiver.cpp",
            single,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {persistent_dfb_id, entry_size, num_entries, ri}});
    }

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
        distributed::MeshWorkload sender_workload;
        persistent_run_on_mesh_device(mesh_device, std::move(sender_program), sender_workload);
        distributed::MeshWorkload receiver_workload;
        persistent_run_on_mesh_device(mesh_device, std::move(receiver_program), receiver_workload);
    }

    uint32_t pass_count = 0;
    for (uint32_t ri = 0; ri < num_receivers; ++ri) {
        if (persistent_dfb_test::verify_receiver_ring(
                device, pdfb, receivers[ri], data_pattern, entry_size, num_entries, ri, num_receivers)) {
            ++pass_count;
        }
    }
    return pass_count;
}

}  // namespace

TEST_F(PersistentDFBFixture, CreatePersistentDFB_TopologyRejects) {
    auto mesh_device = devices_[0];
    CoreRangeSet cores(CoreRange({1, 1}, {1, 1}));
    CoreRangeSet cores2(CoreRange({1, 1}, {2, 2}));

    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), cores}};
        EXPECT_NO_THROW(experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4));
    }
    {
        CoreRangeSet overlap_cores(CoreRange({0, 0}, {0, 0}));
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), overlap_cores}};
        EXPECT_THROW(experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4), std::exception);
    }
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), cores}, {CoreCoord(0, 1), cores2}};
        EXPECT_THROW(experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4), std::exception);
    }
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{CoreCoord(0, 0), CoreRangeSet{}}};
        EXPECT_THROW(experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4), std::exception);
    }
}

TEST_F(PersistentDFBFixture, CreatePersistentDFB_GeometryRejects) {
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};

    EXPECT_THROW(experimental::CreatePersistentDFB(mesh_device.get(), mapping, 0, 4), std::exception);
    EXPECT_THROW(experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 0), std::exception);
    EXPECT_THROW(
        experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4, BufferType::DRAM), std::exception);
}

TEST_F(PersistentDFBFixture, AttachPersistentDFB_AssignsDistinctSlots) {
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping0 = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping1 = {
        {CoreCoord(0, 1), CoreRangeSet(CoreRange({1, 1}, {1, 1}))}};

    auto pdfb0 = experimental::CreatePersistentDFB(mesh_device.get(), mapping0, 256, 4);
    auto pdfb1 = experimental::CreatePersistentDFB(mesh_device.get(), mapping1, 256, 4);

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        CoreRangeSet({CoreRange({0, 0}, {1, 1})}),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_EQ(AttachPersistentDFB(program, pdfb0, pdfb0.all_cores()), 0u);
    EXPECT_EQ(AttachPersistentDFB(program, pdfb1, pdfb1.all_cores()), 1u);

    detail::CompileProgram(mesh_device.get(), program);
    program.impl().finalize_offsets(mesh_device.get());

    const auto& hal = MetalContext::instance().hal();
    const uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    ASSERT_FALSE(program.impl().get_kernel_groups(index).empty());
    EXPECT_NE(
        program.impl().get_kernel_groups(index)[0]->launch_msg.view().kernel_config().persistent_dfb_offset(),
        PERSISTENT_DFB_OFFSET_NONE);
    EXPECT_EQ(
        program.impl().get_kernel_groups(index)[0]->launch_msg.view().kernel_config().cross_node_dfb_offset(),
        CROSS_NODE_DFB_OFFSET_NONE);
}

TEST_F(PersistentDFBFixture, AttachPersistentDFB_SameObjectMultiplePrograms) {
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);
    const uint32_t fifo_start = pdfb.buffer_address();
    const uint32_t config_addr = pdfb.config_address();

    Program program_a = CreateProgram();
    Program program_b = CreateProgram();
    AttachPersistentDFB(program_a, pdfb, pdfb.all_cores());
    AttachPersistentDFB(program_b, pdfb, pdfb.all_cores());

    const auto& per_core_a = program_a.impl().get_per_core_persistent_dfbs().at(CoreCoord(0, 0));
    const auto& per_core_b = program_b.impl().get_per_core_persistent_dfbs().at(CoreCoord(0, 0));
    EXPECT_EQ(per_core_a[0].config_page_addr, config_addr);
    EXPECT_EQ(per_core_b[0].config_page_addr, config_addr);
    EXPECT_EQ(pdfb.buffer_address(), fifo_start);
}

TEST_F(PersistentDFBFixture, AttachPersistentDFB_AddressStableAcrossRebuild) {
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);
    const uint32_t ring_addr = pdfb.buffer_address();
    const uint32_t config_addr = pdfb.config_address();

    {
        Program program = CreateProgram();
        AttachPersistentDFB(program, pdfb, pdfb.all_cores());
        detail::CompileProgram(mesh_device.get(), program);
        program.impl().finalize_offsets(mesh_device.get());
    }
    {
        Program program = CreateProgram();
        AttachPersistentDFB(program, pdfb, pdfb.all_cores());
        EXPECT_EQ(pdfb.buffer_address(), ring_addr);
        EXPECT_EQ(pdfb.config_address(), config_addr);
        EXPECT_EQ(program.impl().get_per_core_persistent_dfbs().at(CoreCoord(0, 0))[0].config_page_addr, config_addr);
    }
}

TEST_F(PersistentDFBFixture, PersistentDFB_CrossProgramPersistence) {
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);
    const uint32_t ring_addr = pdfb.buffer_address();

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pdfb, 256, 4, 0u), 1u);
    EXPECT_EQ(pdfb.buffer_address(), ring_addr);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pdfb, 256, 4, 0u), 1u);
}

TEST_F(PersistentDFBFixture, PersistentDFB_ProducerRelaunchWithOutstandingCredits) {
    // Same-epoch producer relaunch must not barrier on durable outstanding entries:
    // fill half the ring, relaunch sender to fill the rest, then drain once.
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    constexpr uint32_t first_push = 2;
    constexpr uint32_t second_push = 2;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, entry_size, ring_depth);

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pdfb, entry_size, first_push, 0u), 1u);
    EXPECT_EQ(run_persistent_sender_push(mesh_device, pdfb, entry_size, second_push, 0u), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pdfb, entry_size, first_push + second_push, 0u), 1u);
}

TEST_F(PersistentDFBFixture, PersistentDFB_BackToBackRelaunch) {
    // Two cross-program push→pop cycles on the same PersistentDFB.
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pdfb, 256, 4, 0u), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pdfb, 256, 4, 0u), 1u);
    EXPECT_EQ(run_persistent_sender_push(mesh_device, pdfb, 256, 4, 0u), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pdfb, 256, 4, 0u), 1u);
}

TEST_F(PersistentDFBFixture, PersistentDFB_CrossSubDevicePersistence) {
    // Programs may only span one sub-device. Put the sender on SD0 and the receiver on SD1,
    // Attach each program to only its role cores, and share one PersistentDFB across both.
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(1, 0);
    const CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreRangeSet receiver_cores = CoreRangeSet(CoreRange(receiver_core));

    SubDevice sender_sub_device(std::array{sender_cores});
    SubDevice receiver_sub_device(std::array{receiver_cores});
    // local_l1_size=0: PersistentDFB ring/config stay on the global allocator so they can
    // cover cores from both sub-devices (same pattern as remote-CB sub-device tests).
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, /*local_l1_size=*/0);
    mesh_device->load_sub_device_manager(sub_device_manager);

    {
        constexpr uint32_t entry_size = 256;
        constexpr uint32_t num_entries = 4;
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
        auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, entry_size, num_entries);
        const uint32_t ring_addr = pdfb.buffer_address();

        EXPECT_EQ(run_persistent_sender_push(mesh_device, pdfb, entry_size, num_entries, 0u), 1u);
        EXPECT_EQ(pdfb.buffer_address(), ring_addr);
        EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pdfb, entry_size, num_entries, 0u), 1u);
    }

    mesh_device->clear_loaded_sub_device_manager();
}

TEST_F(PersistentDFBFixture, PersistentDFB_CrossSubDevice_ABC_ReceiverRelaunch) {
    // A on SD0 pushes, B on SD1 pops and finishes, then C on SD1 pops a second push from A.
    // Confirms SD1 can relaunch a new consumer program against the same PersistentDFB.
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
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, entry_size, num_entries);

    EXPECT_EQ(run_persistent_sender_push(mesh_device, pdfb, entry_size, num_entries, 0u), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pdfb, entry_size, num_entries, 0u), 1u);
    EXPECT_EQ(run_persistent_sender_push(mesh_device, pdfb, entry_size, num_entries, 0u), 1u);
    EXPECT_EQ(run_persistent_receiver_pop(mesh_device, pdfb, entry_size, num_entries, 0u), 1u);

    mesh_device->clear_loaded_sub_device_manager();
}

TEST_F(PersistentDFBFixture, PersistentDFB_BasicPushPop_1to1) {
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreRangeSet receiver_cores(CoreRange({1, 0}, {1, 0}));
    SubDevice sender_sub_device(std::array{CoreRangeSet(CoreRange(sender_core))});
    SubDevice receiver_sub_device(std::array{receiver_cores});
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, /*local_l1_size=*/0);
    mesh_device->load_sub_device_manager(sub_device_manager);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(
            mesh_device,
            pdfb,
            256,
            4,
            /*write_primitive=*/2,
            /*persistent_dfb_id=*/0,
            /*simultaneous_subdevices=*/true),
        1u);

    mesh_device->clear_loaded_sub_device_manager();
}

TEST_F(PersistentDFBFixture, PersistentDFB_WriteBroadcast_1to4) {
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));
    SubDevice sender_sub_device(std::array{CoreRangeSet(CoreRange(sender_core))});
    SubDevice receiver_sub_device(std::array{receiver_cores});
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, /*local_l1_size=*/0);
    mesh_device->load_sub_device_manager(sub_device_manager);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(
            mesh_device,
            pdfb,
            256,
            4,
            /*write_primitive=*/0,
            /*persistent_dfb_id=*/0,
            /*simultaneous_subdevices=*/true),
        4u);

    mesh_device->clear_loaded_sub_device_manager();
}

TEST_F(PersistentDFBFixture, PersistentDFB_WriteStrided_1to4) {
    auto mesh_device = devices_[0];
    const CoreCoord sender_core(0, 0);
    const CoreRangeSet receiver_cores(CoreRange({1, 0}, {4, 0}));
    SubDevice sender_sub_device(std::array{CoreRangeSet(CoreRange(sender_core))});
    SubDevice receiver_sub_device(std::array{receiver_cores});
    auto sub_device_manager =
        mesh_device->create_sub_device_manager({sender_sub_device, receiver_sub_device}, /*local_l1_size=*/0);
    mesh_device->load_sub_device_manager(sub_device_manager);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);
    EXPECT_EQ(
        run_persistent_1toN_cross_program(
            mesh_device,
            pdfb,
            256,
            4,
            /*write_primitive=*/1,
            /*persistent_dfb_id=*/0,
            /*simultaneous_subdevices=*/true),
        4u);

    mesh_device->clear_loaded_sub_device_manager();
}

TEST_F(PersistentDFBFixture, PersistentDFB_WriteToReceiver_ReceiverContiguous) {
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {4, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pdfb, 256, 4, /*write_primitive=*/2), 4u);
}

TEST_F(PersistentDFBFixture, PersistentDFB_RoundRobinPushBackToReceiver) {
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {4, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 1);
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pdfb, 256, 1, /*write_primitive=*/3), 4u);
}

TEST_F(PersistentDFBFixture, PersistentDFB_PerReceiverCreditInterleaved_RingDepth4) {
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {2, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pdfb, 256, 4, /*write_primitive=*/5), 2u);
}

TEST_F(PersistentDFBFixture, PersistentDFB_DecoupledWriteThenCredit) {
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {4, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);
    EXPECT_EQ(run_persistent_1toN_cross_program(mesh_device, pdfb, 256, 4, /*write_primitive=*/4), 4u);
}

TEST_F(PersistentDFBFixture, GlobalAndCrossNode_SameProgram_DistinctRegions) {
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> cn_mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    std::vector<std::pair<CoreCoord, CoreRangeSet>> pdfb_mapping = {
        {CoreCoord(2, 0), CoreRangeSet(CoreRange({3, 0}, {3, 0}))}};

    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), pdfb_mapping, 256, 4);

    Program program = CreateProgram();
    const CoreRangeSet all_cores =
        CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}, {1, 0}), CoreRange({2, 0}, {3, 0})});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    experimental::CreateCrossNodeDFB(program, mesh_device.get(), cn_mapping, 256, 4);
    AttachPersistentDFB(program, pdfb, pdfb.all_cores());

    detail::CompileProgram(mesh_device.get(), program);
    program.impl().finalize_offsets(mesh_device.get());

    const auto& hal = MetalContext::instance().hal();
    const uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const auto& kg = program.impl().get_kernel_groups(index)[0]->launch_msg.view().kernel_config();
    EXPECT_NE(kg.cross_node_dfb_offset(), CROSS_NODE_DFB_OFFSET_NONE);
    EXPECT_NE(kg.persistent_dfb_offset(), PERSISTENT_DFB_OFFSET_NONE);
    EXPECT_NE(kg.cross_node_dfb_offset(), kg.persistent_dfb_offset());
}

TEST_F(PersistentDFBFixture, PersistentDFB_StaleCommitRejected) {
    // After set_entry_size updates word[5] (PERSISTENT_DFB_CFG_APPLIED_ENTRY_SIZE),
    // commit() with a stale iface.fifo_page_size must not overwrite word[4]
    // (PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT). Push one entry (not a full ring) so the
    // good checkpoint is distinguishable from fifo_start and from the poison wr_ptr.
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t new_entry_size = 512;
    constexpr uint32_t num_entries = 4;

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, entry_size, num_entries);
    const uint32_t poison_wr_ptr = pdfb.buffer_address() + 2 * entry_size;

    const CoreCoord sender_core(0, 0);
    const CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(2);
    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, /*num_entries=*/1, 1);

    Program program = CreateProgram();
    EXPECT_EQ(AttachPersistentDFB(program, pdfb, sender_cores), 0u);
    KernelHandle sender_k = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_stale_commit.cpp",
        sender_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {0u, entry_size, new_entry_size, poison_wr_ptr},
            .defines = {{"PERSISTENT_DFB_TEST_HELPERS", "1"}}});

    persistent_dfb_test::write_sender_l1_staging(
        device, sender_cores, pdfb, data_pattern, entry_size, /*num_entries=*/1, 1);
    persistent_dfb_test::set_sender_l1_staging_runtime_args(program, sender_k, sender_cores, pdfb, staging_size);

    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);

    const uint32_t expected_checkpoint = pdfb.buffer_address() + entry_size;
    std::vector<uint32_t> words(2, 0);
    detail::ReadFromDeviceL1(
        cross_node_dfb_test::local_physical_device(device),
        sender_core,
        pdfb.config_address() + PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT * sizeof(uint32_t),
        std::span<uint8_t>(reinterpret_cast<uint8_t*>(words.data()), 2 * sizeof(uint32_t)),
        CoreType::WORKER);

    // PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT (word[4]) kept the good post-push checkpoint; poison from the stale commit
    // was rejected.
    EXPECT_EQ(words[0], expected_checkpoint);
    EXPECT_NE(words[0], poison_wr_ptr);
    EXPECT_NE(words[0], pdfb.buffer_address());
    // PERSISTENT_DFB_CFG_APPLIED_ENTRY_SIZE (word[5]) reflects the successful resize that created the new epoch.
    EXPECT_EQ(words[1], new_entry_size);
}

TEST_F(PersistentDFBFixture, PersistentDFB_RelayDFB_HostRelationshipValidation) {
    auto mesh_device = devices_[0];
    CoreCoord sender_core(0, 0);
    CoreRangeSet receiver_cores(CoreRange({1, 0}, {2, 0}));
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, 256, 4);

    {
        Program program = CreateProgram();
        EXPECT_EQ(AttachPersistentDFB(program, pdfb, pdfb.all_cores()), 0u);
        experimental::dfb::DataflowBufferConfig config{.entry_size = 256, .num_entries = 4};
        const uint32_t relay_host_id =
            experimental::CreatePersistentRelayDataflowBuffer(program, receiver_cores, config, /*persistent_dfb_id=*/0);
        const auto* relay_dfb = program.impl().get_dataflow_buffer(relay_host_id).get();
        ASSERT_NE(relay_dfb, nullptr);
        EXPECT_TRUE(relay_dfb->config.is_relay);
        const auto persistent_dfb_id = program.impl().get_persistent_dfb_id_for_relay(relay_host_id);
        ASSERT_TRUE(persistent_dfb_id.has_value());
        EXPECT_EQ(*persistent_dfb_id, 0u);
        const uint8_t expected_slot = static_cast<uint8_t>(relay_dfb->device_slot);
        for (const CoreCoord& core : corerange_to_cores(receiver_cores)) {
            const auto& participant = program.impl().get_per_core_persistent_dfbs().at(core).at(0);
            EXPECT_EQ(participant.relay_dfb_id, expected_slot);
        }
        EXPECT_EQ(
            program.impl().get_per_core_persistent_dfbs().at(sender_core).at(0).relay_dfb_id,
            std::numeric_limits<uint8_t>::max());
        EXPECT_EQ(relay_dfb->borrowed_addr_, pdfb.buffer_address());

        // Metal 2.0 genfiles reads DataflowBufferBindingHandle.persistent_dfb_id when emitting
        // RelayDFBBindingToken. Mirror MakeDataflowBufferBindingHandles and verify the callback
        // genfiles uses sees the Persistent slot (not the 0xFF default).
        DataflowBufferBindingHandleMap handles;
        handles.emplace(
            "relay_dfb",
            DataflowBufferBindingHandle{
                .logical_dfb_id = static_cast<uint16_t>(relay_dfb->device_slot),
                .is_relay = relay_dfb->config.is_relay,
                .persistent_dfb_id = *persistent_dfb_id});
        auto kernel = std::make_shared<ComputeKernel>(
            program.impl().get_context_id(),
            KernelSource::from_source("void kernel_main() {}"),
            receiver_cores,
            ComputeConfig{},
            /*is_metal2_kernel=*/true,
            handles);
        bool saw_binding = false;
        kernel->process_dataflow_buffer_binding_handles(
            [&](const std::string& name, uint16_t logical_id, bool is_relay, uint8_t persistent_dfb_id) {
                EXPECT_EQ(name, "relay_dfb");
                EXPECT_EQ(logical_id, expected_slot);
                EXPECT_TRUE(is_relay);
                EXPECT_EQ(persistent_dfb_id, 0u);
                saw_binding = true;
            });
        EXPECT_TRUE(saw_binding);
    }

    {
        Program program = CreateProgram();
        AttachPersistentDFB(program, pdfb, pdfb.all_cores());
        experimental::dfb::DataflowBufferConfig wrong_size{.entry_size = 128, .num_entries = 4};
        EXPECT_THROW(
            experimental::CreatePersistentRelayDataflowBuffer(program, receiver_cores, wrong_size, 0), std::exception);
    }

    {
        Program program = CreateProgram();
        experimental::dfb::DataflowBufferConfig config{.entry_size = 256, .num_entries = 4};
        EXPECT_THROW(
            experimental::CreatePersistentRelayDataflowBuffer(program, receiver_cores, config, /*persistent_dfb_id=*/0),
            std::exception);
    }
}

static uint32_t persistent_relay_expected_checksum(uint32_t total_entries) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < total_entries; ++i) {
        checksum += static_cast<uint32_t>(static_cast<uint8_t>(i)) * 0x01010101u;
    }
    return checksum;
}

// Prog A: sender push. Prog B: receiver DM bind_relay + TRISC consume.
static uint32_t run_persistent_relay_cross_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::PersistentDFB& pdfb,
    uint32_t entry_size,
    uint32_t ring_depth,
    uint32_t total_entries,
    uint32_t batch_size,
    std::optional<uint32_t> receiver_entry_size_override = std::nullopt,
    uint32_t trisc_delay_iterations = 0) {
    TT_FATAL(total_entries % batch_size == 0, "Relay test total_entries must be divisible by batch_size");
    TT_FATAL(ring_depth % batch_size == 0, "Relay test ring_depth must be divisible by batch_size");

    IDevice* device = mesh_device.get();
    const CoreCoord sender_core(0, 0);
    const CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreRangeSet receiver_cores = pdfb.receiver_cores();
    const uint32_t recv_entry_size = receiver_entry_size_override.value_or(entry_size);
    const uint32_t recv_num_entries = pdfb.ring_size() / recv_entry_size;
    TT_FATAL(pdfb.ring_size() % recv_entry_size == 0, "receiver entry size must divide ring");

    // --- Program A: sender push ---
    {
        const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(0);
        const uint32_t staging_size =
            cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, total_entries, 1);
        Program program_a = CreateProgram();
        EXPECT_EQ(AttachPersistentDFB(program_a, pdfb, sender_cores), 0u);
        KernelHandle sender_k = CreateKernel(
            program_a,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_sender.cpp",
            sender_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    0u, entry_size, total_entries, /*write_primitive=*/0, data_pattern, /*do_barrier=*/0}});
        persistent_dfb_test::write_sender_l1_staging(
            device, sender_cores, pdfb, data_pattern, entry_size, total_entries, 1);
        persistent_dfb_test::set_sender_l1_staging_runtime_args(program_a, sender_k, sender_cores, pdfb, staging_size);
        distributed::MeshWorkload workload_a;
        persistent_run_on_mesh_device(mesh_device, std::move(program_a), workload_a);
    }

    // --- Program B: receiver relay + TRISC ---
    constexpr uint32_t result_page_size = 32;
    auto result_buffer = cross_node_dfb_test::make_cross_node_data_buffer(device, receiver_cores, result_page_size, 1);

    Program program_b = CreateProgram();
    EXPECT_EQ(AttachPersistentDFB(program_b, pdfb, receiver_cores, receiver_entry_size_override), 0u);
    experimental::dfb::DataflowBufferConfig relay_config{
        .entry_size = recv_entry_size,
        .num_entries = recv_num_entries,
    };
    const uint32_t relay_host_id =
        experimental::CreatePersistentRelayDataflowBuffer(program_b, receiver_cores, relay_config, 0);
    const uint32_t relay_device_slot = program_b.impl().get_dataflow_buffer(relay_host_id)->device_slot;

    const uint32_t recv_total_entries = (total_entries * entry_size) / recv_entry_size;
    TT_FATAL((total_entries * entry_size) % recv_entry_size == 0, "pushed bytes must be divisible by recv entry size");
    TT_FATAL(recv_total_entries % batch_size == 0, "recv_total_entries must be divisible by batch_size");

    const KernelHandle receiver_kernel = CreateKernel(
        program_b,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_relay_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {0u, recv_total_entries, batch_size}});
    const KernelHandle trisc_kernel = CreateKernel(
        program_b,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_relay_trisc.cpp",
        receiver_cores,
        ComputeConfig{.compile_args = {relay_device_slot, recv_total_entries, batch_size, trisc_delay_iterations, 0u}});

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_b, relay_host_id, receiver_kernel, trisc_kernel);
    SetRuntimeArgs(program_b, trisc_kernel, receiver_cores, {static_cast<uint32_t>(result_buffer->address())});

    distributed::MeshWorkload workload_b;
    persistent_run_on_mesh_device(mesh_device, std::move(program_b), workload_b);

    const uint32_t expected_checksum = persistent_relay_expected_checksum(recv_total_entries);
    uint32_t pass_count = 0;
    IDevice* physical_device = cross_node_dfb_test::local_physical_device(device);
    for (const CoreCoord& receiver_core : corerange_to_cores(receiver_cores)) {
        std::vector<uint32_t> result(2, 0);
        detail::ReadFromDeviceL1(
            physical_device,
            receiver_core,
            static_cast<uint32_t>(result_buffer->address()),
            std::span<uint8_t>(reinterpret_cast<uint8_t*>(result.data()), result.size() * sizeof(uint32_t)),
            CoreType::WORKER);
        if (result[0] == recv_total_entries && result[1] == expected_checksum) {
            ++pass_count;
        } else {
            log_error(
                tt::LogTest,
                "Persistent relay mismatch on {}: count {} (expected {}), checksum 0x{:08x} (expected 0x{:08x})",
                receiver_core.str(),
                result[0],
                recv_total_entries,
                result[1],
                expected_checksum);
        }
    }
    return pass_count;
}

TEST_F(PersistentDFBFixture, PersistentDFB_RelayDFB_CrossProgram_DMToCompute) {
    auto mesh_device = devices_[0];
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 4;
    constexpr uint32_t total_entries = 4;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, entry_size, ring_depth);
    EXPECT_EQ(
        run_persistent_relay_cross_program(mesh_device, pdfb, entry_size, ring_depth, total_entries, /*batch_size=*/1),
        1u);
}

TEST_F(PersistentDFBFixture, PersistentDFB_RelayDFB_Backpressure_NoOverwrite) {
    // Same-program sender + relay receiver + slow TRISC so the ring wraps under backpressure.
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();
    constexpr uint32_t entry_size = 256;
    constexpr uint32_t ring_depth = 2;
    constexpr uint32_t total_entries = 8;
    constexpr uint32_t batch_size = 1;
    constexpr uint32_t trisc_delay = 1000;

    const CoreCoord sender_core(0, 0);
    const CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    const CoreRangeSet receiver_cores = CoreRangeSet(CoreRange({1, 0}, {1, 0}));
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, entry_size, ring_depth);

    const uint32_t data_pattern = cross_node_dfb_test::data_pattern_for_write_primitive(0);
    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, entry_size, total_entries, 1);
    constexpr uint32_t result_page_size = 32;
    auto result_buffer = cross_node_dfb_test::make_cross_node_data_buffer(device, receiver_cores, result_page_size, 1);

    Program program = CreateProgram();
    EXPECT_EQ(AttachPersistentDFB(program, pdfb, pdfb.all_cores()), 0u);
    experimental::dfb::DataflowBufferConfig relay_config{.entry_size = entry_size, .num_entries = ring_depth};
    const uint32_t relay_host_id =
        experimental::CreatePersistentRelayDataflowBuffer(program, receiver_cores, relay_config, 0);
    const uint32_t relay_device_slot = program.impl().get_dataflow_buffer(relay_host_id)->device_slot;

    const KernelHandle sender_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_sender.cpp",
        sender_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {0u, entry_size, total_entries, 0u, data_pattern, 0u}});
    const KernelHandle receiver_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_relay_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {0u, total_entries, batch_size}});
    const KernelHandle trisc_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_relay_trisc.cpp",
        receiver_cores,
        ComputeConfig{.compile_args = {relay_device_slot, total_entries, batch_size, trisc_delay, 0u}});

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, relay_host_id, receiver_kernel, trisc_kernel);
    persistent_dfb_test::write_sender_l1_staging(
        device, sender_cores, pdfb, data_pattern, entry_size, total_entries, 1);
    persistent_dfb_test::set_sender_l1_staging_runtime_args(program, sender_kernel, sender_cores, pdfb, staging_size);
    SetRuntimeArgs(program, trisc_kernel, receiver_cores, {static_cast<uint32_t>(result_buffer->address())});

    distributed::MeshWorkload workload;
    persistent_run_on_mesh_device(mesh_device, std::move(program), workload);

    std::vector<uint32_t> result(2, 0);
    detail::ReadFromDeviceL1(
        cross_node_dfb_test::local_physical_device(device),
        CoreCoord(1, 0),
        static_cast<uint32_t>(result_buffer->address()),
        std::span<uint8_t>(reinterpret_cast<uint8_t*>(result.data()), result.size() * sizeof(uint32_t)),
        CoreType::WORKER);
    EXPECT_EQ(result[0], total_entries);
    EXPECT_EQ(result[1], persistent_relay_expected_checksum(total_entries));
}

TEST_F(PersistentDFBFixture, PersistentDFB_RelayDFB_CrossProgram_DifferentEntrySize) {
    // Full drain: A finishes on E1, then B Attach/relay with E2.
    auto mesh_device = devices_[0];
    constexpr uint32_t e1 = 256;
    constexpr uint32_t e2 = 512;
    constexpr uint32_t ring_depth_e1 = 4;  // ring bytes = 1024 → 2 entries at E2
    constexpr uint32_t total_entries_e1 = 2;
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange({1, 0}, {1, 0}))}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, e1, ring_depth_e1);
    EXPECT_EQ(
        run_persistent_relay_cross_program(
            mesh_device, pdfb, e1, ring_depth_e1, total_entries_e1, /*batch_size=*/1, e2),
        1u);
}

TEST_F(PersistentDFBFixture, PersistentDFB_CrossSubDevice_CoordinatedLivePeerNonDividingE2) {
    // Coordinated live-peer E1→E2 with real drain and a non-dividing E2:
    //   A (SD0): push E1 → barrier → set_entry_size(E2) → signal → wait go → push E2
    //   B (SD1): Attach(E1), pop E1 → set_receiver_entry_size(E2), finish
    //   C (SD1): Attach(E2) while A is still alive; consume E2
    //
    // Four E1 entries make the durable counters wrap the original 1024-byte ring.
    // E2=384 truncates the usable ring to 768 bytes. Without credit rebase, the old
    // 1024-byte credit total reconstructs offset 256 instead of checkpoint offset 0.
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device.get();
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

    // Semaphores are allocated before the PersistentDFB so they sit above it in L1;
    // the sender staging scratch (placed just below the pdfb) must not overlap them.
    auto resized_sem = CreateGlobalSemaphore(mesh_device.get(), sender_cores, /*initial_value=*/0);
    auto go_sem = CreateGlobalSemaphore(mesh_device.get(), sender_cores, /*initial_value=*/0);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {{sender_core, receiver_cores}};
    auto pdfb = experimental::CreatePersistentDFB(mesh_device.get(), mapping, e1, ring_depth_e1);
    distributed::Synchronize(*mesh_device, std::nullopt);

    const uint32_t staging_size =
        cross_node_dfb_test::sender_staging_size_bytes(data_pattern, e1, total_entries_e1, 1, e2, total_entries_e2);

    // --- Program A (SD0): push E1 → barrier → resize → signal → wait go → push E2 ---
    Program program_a = CreateProgram();
    EXPECT_EQ(AttachPersistentDFB(program_a, pdfb, sender_cores), 0u);
    const KernelHandle sender_k = CreateKernel(
        program_a,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_coordinated_resize_sender.cpp",
        sender_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {0u, e1, total_entries_e1, e2, total_entries_e2, data_pattern}});
    persistent_dfb_test::write_sender_l1_staging(
        device,
        sender_cores,
        pdfb,
        data_pattern,
        e1,
        total_entries_e1,
        1,
        /*counter_base=*/0,
        e2,
        total_entries_e2);
    SetRuntimeArgs(
        program_a,
        sender_k,
        sender_cores,
        {persistent_dfb_test::sender_l1_staging_address(pdfb, staging_size),
         static_cast<uint32_t>(resized_sem.address()),
         static_cast<uint32_t>(go_sem.address())});

    distributed::MeshWorkload workload_a;
    workload_a.add_program(persistent_unit_mesh_device_range(), std::move(program_a));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_a, false);

    // Subsequent FD work waits for SD1 idle (not long-running A on SD0).
    mesh_device->set_sub_device_stall_group({{SubDeviceId{1}}});

    // --- Program B (SD1): consume E1, then rebase receiver credits to E2 ---
    Program program_b = CreateProgram();
    EXPECT_EQ(AttachPersistentDFB(program_b, pdfb, receiver_cores), 0u);
    CreateKernel(
        program_b,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_coordinated_resize_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {0u, total_entries_e1, e2}});
    distributed::MeshWorkload workload_b;
    workload_b.add_program(persistent_unit_mesh_device_range(), std::move(program_b));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_b, false);

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
    const uint32_t credit_base = pdfb.config_address() + pdfb.credit_reset_offset();
    const auto [sender_sent, sender_acked] = cross_node_dfb_test::read_credit_pair(device, sender_core, credit_base);
    const auto [receiver_sent, receiver_acked] =
        cross_node_dfb_test::read_credit_pair(device, receiver_core, credit_base);
    ASSERT_TRUE(resized) << "Timed out waiting for A set_entry_size(E2) signal; sender credits=" << sender_sent << "/"
                         << sender_acked << ", receiver credits=" << receiver_sent << "/" << receiver_acked;

    // --- Program C (SD1): same-epoch Attach E2 while A is still alive ---
    Program program_c = CreateProgram();
    EXPECT_EQ(AttachPersistentDFB(program_c, pdfb, receiver_cores, e2), 0u);
    CreateKernel(
        program_c,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/persistent_dfb_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {0u, e2, total_entries_e2, 0u}});

    distributed::MeshWorkload workload_c;
    workload_c.add_program(persistent_unit_mesh_device_range(), std::move(program_c));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload_c, false);

    // Release A to push at E2; C is already attached/running on the peer sub-device.
    MetalContext::instance().get_cluster().write_core(
        device_id, physical_sender, std::vector<uint32_t>{1}, go_sem.address());

    mesh_device->reset_sub_device_stall_group();
    distributed::Finish(mesh_device->mesh_command_queue());

    // E2 entries must begin at checkpoint offset 0. The expected counter values
    // continue after the four E1 entries; a stale 256-byte offset fails this comparison.
    EXPECT_TRUE(persistent_dfb_test::verify_receiver_ring(
        device,
        pdfb,
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
