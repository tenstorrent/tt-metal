// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

distributed::MeshWorkload make_l1_write_workload(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const experimental::NodeCoord& node,
    uint32_t address,
    uint32_t value,
    const std::string& kernel_id) {
    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    const experimental::KernelSpecName DM_KERNEL{kernel_id};
    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = node};
    experimental::ProgramSpec spec{
        .name = std::string("l1_write_") + kernel_id, .kernels = {dm_kernel_spec}, .work_units = {main_wu}};
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = DM_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"address", address}}),
        .common_runtime_arg_values = {{"value", value}},
    }};
    experimental::SetProgramRunArgs(program, params);
    wl.add_program(device_range, std::move(program));
    return wl;
}

void run_cross_cq_handoff(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint8_t producer_cq, uint8_t consumer_cq) {
    IDevice* dev = mesh_device->get_devices()[0];
    const experimental::NodeCoord node{0, 0};

    const uint32_t producer_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t consumer_address = producer_address + sizeof(uint32_t);
    const uint32_t producer_value = 0x2c000000u | producer_cq;
    const uint32_t consumer_value = 0x2c110000u | consumer_cq;

    std::vector<uint32_t> zeros(2, 0);
    tt_metal::detail::WriteToDeviceL1(dev, node, producer_address, zeros);

    distributed::MeshCommandQueue& producer = mesh_device->mesh_command_queue(producer_cq);
    distributed::MeshCommandQueue& consumer = mesh_device->mesh_command_queue(consumer_cq);

    auto producer_wl = make_l1_write_workload(
        mesh_device, node, producer_address, producer_value, "producer_cq" + std::to_string(producer_cq));
    auto consumer_wl = make_l1_write_workload(
        mesh_device, node, consumer_address, consumer_value, "consumer_cq" + std::to_string(consumer_cq));

    // Consumer's workload is ordered after the producer's via the cross-CQ event.
    distributed::EnqueueMeshWorkload(producer, producer_wl, false);
    distributed::MeshEvent handoff_event = producer.enqueue_record_event();
    consumer.enqueue_wait_for_event(handoff_event);
    distributed::EnqueueMeshWorkload(consumer, consumer_wl, false);

    distributed::Finish(producer);
    distributed::Finish(consumer);

    std::vector<uint32_t> producer_out(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, producer_address, sizeof(uint32_t), producer_out);
    ASSERT_EQ(producer_out[0], producer_value);

    std::vector<uint32_t> consumer_out(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, consumer_address, sizeof(uint32_t), consumer_out);
    ASSERT_EQ(consumer_out[0], consumer_value);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, EventSynchronize) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    IDevice* dev = mesh_device->get_devices()[0];
    const experimental::NodeCoord node{0, 0};

    const uint32_t address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t value = 0x12abcd34;

    std::vector<uint32_t> zeros(1, 0);
    tt_metal::detail::WriteToDeviceL1(dev, node, address, zeros);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::MeshEvent event = cq.enqueue_record_event_to_host();

    // Block the host until the CQ has processed past the event point.
    distributed::EventSynchronize(event);

    std::vector<uint32_t> outputs(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address, sizeof(uint32_t), outputs);
    ASSERT_EQ(outputs[0], value);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, EventQuery) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    distributed::MeshEvent event = cq.enqueue_record_event_to_host();

    while (!distributed::EventQuery(event)) {
    };

    ASSERT_TRUE(distributed::EventQuery(event));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, EventBetweenWorkloads) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    IDevice* dev = mesh_device->get_devices()[0];
    const experimental::NodeCoord node{0, 0};

    const uint32_t address_1 = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t address_2 = address_1 + sizeof(uint32_t);
    const uint32_t value_1 = 0xaabb1122;
    const uint32_t value_2 = 0xccdd3344;

    std::vector<uint32_t> zeros(2, 0);
    tt_metal::detail::WriteToDeviceL1(dev, node, address_1, zeros);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    auto wl1 = make_l1_write_workload(mesh_device, node, address_1, value_1, "dm_kernel_1");
    auto wl2 = make_l1_write_workload(mesh_device, node, address_2, value_2, "dm_kernel_2");

    // Enqueue both non-blocking; record event after first to observe ordering.
    distributed::EnqueueMeshWorkload(cq, wl1, false);
    distributed::MeshEvent event_after_wl1 = cq.enqueue_record_event_to_host();
    distributed::EnqueueMeshWorkload(cq, wl2, false);

    // Synchronize on event_after_wl1 and verify wl1's result is present.
    distributed::EventSynchronize(event_after_wl1);

    std::vector<uint32_t> out_1(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address_1, sizeof(uint32_t), out_1);
    ASSERT_EQ(out_1[0], value_1);

    // Drain the remaining wl2.
    distributed::Finish(cq);

    std::vector<uint32_t> out_2(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address_2, sizeof(uint32_t), out_2);
    ASSERT_EQ(out_2[0], value_2);
}

TEST_F(QuasarMultiCQMeshDeviceSingleCardFixture, CrossCQEventHandoffCQ0ToCQ1) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }
    run_cross_cq_handoff(devices_[0], /*producer_cq=*/0, /*consumer_cq=*/1);
}

// Reverse direction: issue_record_event_commands / issue_wait_for_event_commands branch on
// cq_id == 0, so CQ1-produces-CQ0-consumes is a distinct code path worth its own case.
TEST_F(QuasarMultiCQMeshDeviceSingleCardFixture, CrossCQEventHandoffCQ1ToCQ0) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }
    run_cross_cq_handoff(devices_[0], /*producer_cq=*/1, /*consumer_cq=*/0);
}

TEST_F(QuasarMultiCQMeshDeviceSingleCardFixture, RecordEventToHostFromBothCQs) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    IDevice* dev = mesh_device->get_devices()[0];
    const experimental::NodeCoord node{0, 0};

    const uint32_t address_0 = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t address_1 = address_0 + sizeof(uint32_t);
    const uint32_t value_0 = 0x2c2c0000;
    const uint32_t value_1 = 0x2c2c0001;

    std::vector<uint32_t> zeros(2, 0);
    tt_metal::detail::WriteToDeviceL1(dev, node, address_0, zeros);

    distributed::MeshCommandQueue& cq0 = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& cq1 = mesh_device->mesh_command_queue(1);

    auto wl0 = make_l1_write_workload(mesh_device, node, address_0, value_0, "cq0_kernel");
    auto wl1 = make_l1_write_workload(mesh_device, node, address_1, value_1, "cq1_kernel");

    distributed::EnqueueMeshWorkload(cq0, wl0, false);
    distributed::MeshEvent event_0 = cq0.enqueue_record_event_to_host();
    distributed::EventSynchronize(event_0);

    distributed::Finish(cq0);

    distributed::EnqueueMeshWorkload(cq1, wl1, false);
    distributed::MeshEvent event_1 = cq1.enqueue_record_event_to_host();
    distributed::EventSynchronize(event_1);

    std::vector<uint32_t> out_0(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address_0, sizeof(uint32_t), out_0);
    ASSERT_EQ(out_0[0], value_0);

    std::vector<uint32_t> out_1(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address_1, sizeof(uint32_t), out_1);
    ASSERT_EQ(out_1[0], value_1);
}
