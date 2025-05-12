// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/logger.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include "tt-metalium/kernel_types.hpp"
#include <tt-metalium/fabric.hpp>
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/erisc_datamover_builder_helper.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include "ttnn/cpp/ttnn/operations/experimental/reshape/view.hpp"
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tile.hpp>

#include "umd/device/types/arch.h"
#include "umd/device/types/cluster_descriptor_types.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <unordered_set>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

enum TwoInputReaderKernelWriteMode { LOCAL_WRITEBACK, FABRIC_UNICAST, FABRIC_MULTICAST };

static constexpr size_t TEST_WORKERS_SUBDEVICE_INDEX = 0;
static constexpr size_t TEST_EDM_FABRIC_SUBDEVICE_INDEX = 1;

using subdevice_managers_t = std::unordered_map<chip_id_t, SubDeviceManagerId>;
struct SubdeviceInfo {
    std::unordered_map<chip_id_t, SubDeviceManagerId> sub_device_managers;
    std::unordered_map<chip_id_t, SubDeviceId> worker_subdevice_id;
    std::unordered_map<chip_id_t, SubDeviceId> fabric_subdevice_id;
};

using tt::tt_metal::distributed::MeshContainer;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshDeviceView;
using tt::tt_metal::distributed::MeshShape;
using tt::tt_metal::distributed::SystemMesh;

class BaseFabricFixture {
protected:
    tt::ARCH arch_;
    std::size_t num_devices_;
    bool device_open = false;

    // Common constants for both fixtures
    static constexpr size_t TG_NUM_DEVICES = 36;
    static constexpr size_t GALAXY_6U_NUM_DEVICES = 32;

    // Gets the appropriate mesh shape based on device configuration
    MeshShape GetDeterminedMeshShape() const {
        if (num_devices_ == TG_NUM_DEVICES || num_devices_ == GALAXY_6U_NUM_DEVICES) {
            return MeshShape{8, 4};
        } else {
            return MeshShape{2, 4};
        }
    }

    // Validates environment and hardware for tests
    void ValidateEnvironment() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run without TT_METAL_SLOW_DISPATCH_MODE set");
        }

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();

        if (!(arch_ == tt::ARCH::WORMHOLE_B0 && num_devices_ >= 8 &&
              (tt::tt_metal::GetNumPCIeDevices() == 4 || tt::tt_metal::GetNumPCIeDevices() == GALAXY_6U_NUM_DEVICES))) {
            TT_THROW("This suite can only be run on T3000 or TG Wormhole devices");
        }
    }

public:
    BaseFabricFixture() : device_open(false) {}

    BaseFabricFixture(tt::tt_metal::FabricConfig fabric_config) : device_open(false) {
        tt::tt_metal::detail::InitializeFabricConfig(fabric_config);
    }

    virtual ~BaseFabricFixture() { tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::DISABLED); }

    virtual void SetupDevices() = 0;
    virtual void TearDown() = 0;
};

class Fabric1DFixture : public BaseFabricFixture {
public:
    std::shared_ptr<MeshDeviceView> view_;
    std::map<chip_id_t, IDevice*> physical_devices_;

    void SetupDevices() override {
        ValidateEnvironment();

        const MeshShape cluster_shape = GetDeterminedMeshShape();
        const auto& physical_device_ids = SystemMesh::instance().get_mapped_physical_device_ids(cluster_shape);
        physical_devices_ = tt::tt_metal::detail::CreateDevices(physical_device_ids);

        std::vector<IDevice*> devices = {};
        for (auto device_id : physical_device_ids) {
            devices.push_back(physical_devices_.at(device_id));
        }

        MeshContainer<IDevice*> device_container(cluster_shape, devices);
        view_ = std::make_shared<MeshDeviceView>(device_container);
        device_open = true;
    }

    void TearDown() override {
        if (device_open) {
            tt::tt_metal::detail::CloseDevices(physical_devices_);
            device_open = false;
        }
    }

    Fabric1DFixture() : BaseFabricFixture() { this->SetupDevices(); }

    Fabric1DFixture(tt::tt_metal::FabricConfig fabric_config) : BaseFabricFixture(fabric_config) {
        this->SetupDevices();
    }

    ~Fabric1DFixture() override { TearDown(); }
};

class MeshFabric1DFixture : public BaseFabricFixture {
public:
    std::shared_ptr<MeshDevice> mesh_device_;

    void SetupDevices() override {
        ValidateEnvironment();
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(GetDeterminedMeshShape()));
        device_open = true;
    }

    void TearDown() override {
        if (device_open) {
            mesh_device_->close();
            device_open = false;
        }
    }

    MeshFabric1DFixture() : BaseFabricFixture() { this->SetupDevices(); }

    MeshFabric1DFixture(tt::tt_metal::FabricConfig fabric_config) : BaseFabricFixture(fabric_config) {
        this->SetupDevices();
    }

    ~MeshFabric1DFixture() override {
        if (device_open) {
            TearDown();
        }
    }
};

class Fabric1DLineDeviceInitFixture : public Fabric1DFixture {
public:
    Fabric1DLineDeviceInitFixture() : Fabric1DFixture(tt::tt_metal::FabricConfig::FABRIC_1D) {}
};

class Fabric1DRingDeviceInitFixture : public Fabric1DFixture {
public:
    Fabric1DRingDeviceInitFixture() : Fabric1DFixture(tt::tt_metal::FabricConfig::FABRIC_1D_RING) {}
};

class MeshFabric1DLineDeviceInitFixture : public MeshFabric1DFixture {
public:
    MeshFabric1DLineDeviceInitFixture() : MeshFabric1DFixture(tt::tt_metal::FabricConfig::FABRIC_1D) {}
};

class MeshFabric1DRingDeviceInitFixture : public MeshFabric1DFixture {
public:
    MeshFabric1DRingDeviceInitFixture() : MeshFabric1DFixture(tt::tt_metal::FabricConfig::FABRIC_1D_RING) {}
};

struct BankedConfig {
    size_t num_pages;
    size_t size_bytes;
    size_t page_size_bytes;
    BufferType input_buffer_type;
    BufferType output_buffer_type;
    tt::DataFormat l1_data_format;
};

struct KernelXY {
    uint16_t x;
    uint16_t y;

    uint32_t to_uint32() const { return y << 16 | x; }
};

enum Correctness { Correct, Incorrect };

template <typename CONTAINER_T>
Correctness run_output_check(CONTAINER_T const& inputs, CONTAINER_T output_buffer) {
    constexpr bool debug_mode = true;

    log_info(tt::LogTest, "Checking outputs");
    bool pass = true;

    std::size_t num_printed_mismatches = 0;
    for (size_t i = 0; i < inputs.size() && num_printed_mismatches < 64; i++) {
        if (output_buffer[i] != inputs[i]) {
            if (debug_mode) {
                if (pass) {
                    log_error("Output mismatch");
                }
                log_error("[{}]: expected {} got {}", i, inputs[i], output_buffer[i]);
                num_printed_mismatches++;
            }
            pass = false;
        }
    }
    if (num_printed_mismatches > 0) {
        log_error("... (remaining mismatches omitted)");
    }

    log_info(tt::LogTest, "Output check: {}", pass ? "PASS" : "FAIL");
    return pass ? Correctness::Correct : Correctness::Incorrect;
};

static SubdeviceInfo create_subdevices(const std::vector<IDevice*>& devices) {
    SubdeviceInfo subdevice_info;
    std::unordered_map<chip_id_t, SubDeviceManagerId> sub_device_manager_ids;
    for (auto device : devices) {
        const auto& tensix_sub_device =
            tt_metal::SubDevice(std::array{device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0})});
        const auto& eth_sub_device = tt_metal::SubDevice(
            std::array{CoreRangeSet(), device->worker_cores(HalProgrammableCoreType::ACTIVE_ETH, SubDeviceId{0})});
        subdevice_info.sub_device_managers.insert(
            {device->id(), device->create_sub_device_manager({tensix_sub_device, eth_sub_device}, 0)});
        device->load_sub_device_manager(subdevice_info.sub_device_managers.at(device->id()));
        subdevice_info.worker_subdevice_id.insert(
            {device->id(), device->get_sub_device_ids().at(TEST_WORKERS_SUBDEVICE_INDEX)});
        subdevice_info.fabric_subdevice_id.insert(
            {device->id(), device->get_sub_device_ids().at(TEST_EDM_FABRIC_SUBDEVICE_INDEX)});
        device->set_sub_device_stall_group({subdevice_info.worker_subdevice_id.at(device->id())});
    }

    return subdevice_info;
}

static SubdeviceInfo create_worker_subdevices(const std::vector<IDevice*>& devices) {
    SubdeviceInfo subdevice_info;
    std::unordered_map<chip_id_t, SubDeviceManagerId> sub_device_manager_ids;
    for (auto device : devices) {
        const auto& tensix_sub_device =
            tt_metal::SubDevice(std::array{device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0})});
        subdevice_info.sub_device_managers.insert(
            {device->id(), device->create_sub_device_manager({tensix_sub_device}, 0)});
        device->load_sub_device_manager(subdevice_info.sub_device_managers.at(device->id()));
        subdevice_info.worker_subdevice_id.insert(
            {device->id(), device->get_sub_device_ids().at(TEST_WORKERS_SUBDEVICE_INDEX)});
        device->set_sub_device_stall_group({subdevice_info.worker_subdevice_id.at(device->id())});
    }

    return subdevice_info;
}

Correctness run_output_check(
    const std::vector<uint32_t>& all_zeros,
    const std::vector<uint32_t>& inputs,
    std::shared_ptr<Buffer>& output_buffer) {
    constexpr bool debug_mode = true;
    std::vector<uint32_t> readback_data_vec(all_zeros.size(), 0);  // init to 0 data for easier debug

    tt_metal::detail::ReadFromBuffer(output_buffer, readback_data_vec);
    return run_output_check(inputs, readback_data_vec);
};

void run_programs(std::vector<Program>& programs, const std::vector<IDevice*>& devices) {
    EXPECT_EQ(programs.size(), devices.size());
    const size_t num_programs = programs.size();
    try {
        for (size_t i = 0; i < num_programs; i++) {
            tt::tt_metal::detail::CompileProgram(devices.at(i), programs.at(i));
        }
    } catch (std::exception& e) {
        log_error("Failed compile: {}", e.what());
        throw e;
    }

    log_info(tt::LogTest, "Running...");

    std::vector<std::thread> threads;
    threads.reserve(num_programs);
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        for (size_t i = 0; i < num_programs; i++) {
            threads.emplace_back(std::thread([&] { tt_metal::detail::LaunchProgram(devices.at(i), programs.at(i)); }));
        }

        std::ranges::for_each(threads, [](std::thread& t) { t.join(); });
    } else {
        for (size_t i = 0; i < num_programs; i++) {
            tt_metal::EnqueueProgram(devices.at(i)->command_queue(), programs.at(i), false);
        }

        log_debug(tt::LogTest, "Calling Finish");
        for (size_t i = 0; i < num_programs; i++) {
            tt_metal::Finish(devices.at(i)->command_queue());
        }
    }
}

std::tuple<std::shared_ptr<Buffer>, std::vector<uint32_t>> build_input_buffer(
    IDevice* first_device, size_t tensor_size_bytes, const BankedConfig& test_config) {
    auto inputs = std::vector<uint32_t>(tensor_size_bytes / sizeof(uint32_t), 0);
    std::iota(inputs.begin(), inputs.end(), 0);

    // Input buffer
    auto local_input_buffer = CreateBuffer(InterleavedBufferConfig{
        first_device, test_config.size_bytes, test_config.page_size_bytes, test_config.input_buffer_type});
    tt_metal::detail::WriteToBuffer(local_input_buffer, inputs);
    return {local_input_buffer, inputs};
}

static void build_and_enqueue(
    const std::vector<IDevice*>& devices, std::vector<Program>& programs, bool enqueue_only = false) {
    TT_FATAL(
        devices.size() == programs.size(),
        "Number of devices must match number of programs when calling build_and_enqueue in test");
    if (!enqueue_only) {
        for (size_t i = 0; i < devices.size(); i++) {
            tt::tt_metal::detail::CompileProgram(devices[i], programs[i]);
        }
    }
    for (size_t i = 0; i < devices.size(); i++) {
        tt_metal::EnqueueProgram(devices[i]->command_queue(), programs[i], false);
    }
}

struct EthLinkHop {
    CoreCoord hop_src;
    CoreCoord hop_dest;
};

struct ChipConnection {
    std::vector<EthLinkHop> links;
};

struct unicast_send {
    size_t distance;
};
struct mcast_send {
    size_t distance;
    size_t range;
};

using mode_variant_t = std::variant<mcast_send, unicast_send>;

static constexpr size_t PACKET_HEADER_SIZE_BYTES = sizeof(tt::tt_fabric::PacketHeader);
void generate_sender_worker_kernels(
    Program& program,
    IDevice* device,
    const CoreCoord& worker_core,
    const tt::tt_fabric::SenderWorkerAdapterSpec& worker_fabric_connection,
    const mode_variant_t& mode,
    std::size_t edm_buffer_size,
    uint32_t page_plus_header_size,
    uint32_t num_pages_total,
    uint32_t num_pages_per_edm_buffer,
    uint32_t local_worker_fabric_semaphore_id,
    uint32_t local_worker_teardown_semaphore_id,
    uint32_t local_worker_last_message_semaphore_id,
    uint32_t dram_input_buffer_base_addr,
    bool src_is_dram,
    uint32_t dram_output_buffer_base_addr,
    bool dest_is_dram,
    uint32_t worker_buffer_index_semaphore_id,
    // farthest to closest
    const std::vector<tt::tt_fabric::edm_termination_info_t>& edm_termination_infos) {
    const auto& edm_noc_core = CoreCoord(worker_fabric_connection.edm_noc_x, worker_fabric_connection.edm_noc_y);
    std::vector<uint32_t> sender_worker_reader_compile_args{
        src_is_dram,      //
        num_pages_total,  //
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        num_pages_per_edm_buffer};
    std::vector<uint32_t> sender_worker_reader_runtime_args{dram_input_buffer_base_addr};

    log_trace(tt::LogTest, "\tSenderReader CT Args");
    for (const auto& arg : sender_worker_reader_compile_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }
    log_trace(tt::LogTest, "\tSenderReader RT Args");
    for (const auto& arg : sender_worker_reader_runtime_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }

    std::vector<uint32_t> sender_worker_writer_compile_args{
        num_pages_per_edm_buffer,
        num_pages_total,
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        worker_fabric_connection.num_buffers_per_channel,
        dest_is_dram,
        std::holds_alternative<mcast_send>(mode) ? 1 : 0};
    log_trace(tt::LogTest, "worker_fabric_connection.edm_l1_sem_addr: {}", worker_fabric_connection.edm_l1_sem_addr);
    log_trace(tt::LogTest, "worker_buffer_index_semaphore_id: {}", worker_buffer_index_semaphore_id);
    log_trace(tt::LogTest, "last_message_semaphore_address: {}", local_worker_last_message_semaphore_id);
    log_trace(
        tt::LogTest, "Sender communicating with EDM: x={}, y={}", (uint32_t)edm_noc_core.x, (uint32_t)edm_noc_core.y);
    std::vector<uint32_t> sender_worker_writer_runtime_args{
        worker_fabric_connection.edm_buffer_base_addr,
        worker_fabric_connection.edm_l1_sem_addr,
        local_worker_fabric_semaphore_id,
        local_worker_teardown_semaphore_id,
        (uint32_t)edm_noc_core.x,
        (uint32_t)edm_noc_core.y,
        worker_fabric_connection.num_buffers_per_channel,

        worker_fabric_connection.edm_connection_handshake_addr,
        worker_fabric_connection.edm_worker_location_info_addr,
        edm_buffer_size,
        dram_output_buffer_base_addr,
        local_worker_last_message_semaphore_id,
        worker_buffer_index_semaphore_id,
        worker_fabric_connection.persistent_fabric ? 1 : 0,
        worker_fabric_connection.buffer_index_semaphore_id};

    if (std::holds_alternative<mcast_send>(mode)) {
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).distance);
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).range);
    } else {
        sender_worker_writer_runtime_args.push_back(std::get<unicast_send>(mode).distance);
    }

    get_runtime_args_for_edm_termination_infos(edm_termination_infos, sender_worker_writer_runtime_args);

    uint32_t src0_cb_index = CBIndex::c_0;
    log_trace(tt::LogTest, "\tSenderWriter CT Args");
    for (const auto& arg : sender_worker_writer_compile_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }
    log_trace(tt::LogTest, "\tSenderWriter RT Args");
    for (const auto& arg : sender_worker_writer_runtime_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }

    // Just want a dummy DF
    tt::DataFormat df = (page_plus_header_size - PACKET_HEADER_SIZE_BYTES) == 1024   ? tt::DataFormat::Bfp8
                        : (page_plus_header_size - PACKET_HEADER_SIZE_BYTES) == 2048 ? tt::DataFormat::Float16
                                                                                     : tt::DataFormat::Float32;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_plus_header_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, page_plus_header_size);
    CBHandle sender_workers_cb = CreateCircularBuffer(program, worker_core, cb_src0_config);
    auto sender_worker_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/fabric_erisc_datamover_sender_worker_reader.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_worker_reader_compile_args});
    auto sender_worker_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/fabric_erisc_datamover_sender_worker_sender.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = sender_worker_writer_compile_args});
    tt_metal::SetRuntimeArgs(program, sender_worker_reader_kernel, worker_core, sender_worker_reader_runtime_args);
    tt_metal::SetRuntimeArgs(program, sender_worker_writer_kernel, worker_core, sender_worker_writer_runtime_args);
}

bool RunLoopbackTest(
    tt_metal::IDevice* sender_device,
    tt_metal::IDevice* receiver_device,

    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram,
    std::vector<Program>& programs,
    tt::tt_fabric::FabricEriscDatamoverBuilder& chip_0_edm_builder,
    std::optional<SubdeviceInfo>& subdevice_managers,
    bool enable_persistent_fabric) {
    auto& sender_program = programs.at(0);
    std::size_t page_plus_header_size = page_size + sizeof(tt::tt_fabric::PacketHeader);
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    std::vector<CoreCoord> worker_cores = {CoreCoord(0, 0)};

    auto local_worker_fabric_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);
    auto local_worker_teardown_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);
    auto local_worker_last_message_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);
    auto worker_buffer_index_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);

    // Generate inputs
    ////////////////////////////////////////////////////////////////////////////
    //   SETUP THE INPUT CB
    ////////////////////////////////////////////////////////////////////////////

    BankedConfig test_config = BankedConfig{
        .num_pages = num_pages_total,
        .size_bytes = tensor_size_bytes,
        .page_size_bytes = page_size,
        .input_buffer_type = src_is_dram ? BufferType::DRAM : BufferType::L1,
        .output_buffer_type = dest_is_dram ? BufferType::DRAM : BufferType::L1,
        .l1_data_format = tt::DataFormat::Float16_b};

    auto [local_input_buffer, inputs] = build_input_buffer(sender_device, tensor_size_bytes, test_config);

    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    auto local_output_buffer = CreateBuffer(InterleavedBufferConfig{
        sender_device, test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type});

    tt_metal::detail::WriteToBuffer(local_output_buffer, all_zeros);

    auto local_input_buffer_address = local_input_buffer->address();
    auto local_output_buffer_address = local_output_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    // EDM Builder Setup
    ////////////////////////////////////////////////////////////////////////////

    static constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes + PACKET_HEADER_SIZE_BYTES;

    auto chip0_worker_fabric_connection = chip_0_edm_builder.build_connection_to_worker_channel();
    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    const std::size_t pages_per_send =
        (chip0_worker_fabric_connection.buffer_size_bytes - PACKET_HEADER_SIZE_BYTES) / page_size;
    const auto& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    const auto& edm_config = tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size);
    const std::vector<tt::tt_fabric::edm_termination_info_t>& edm_termination_infos =
        enable_persistent_fabric ? std::vector<tt::tt_fabric::edm_termination_info_t>{}
                                 : std::vector<tt::tt_fabric::edm_termination_info_t>{
                                       {1,
                                        sender_device->ethernet_core_from_logical_core(eth_receiver_core).x,
                                        sender_device->ethernet_core_from_logical_core(eth_receiver_core).y,
                                        chip_0_edm_builder.config.termination_signal_address},
                                       {0,
                                        sender_device->ethernet_core_from_logical_core(eth_sender_core).x,
                                        sender_device->ethernet_core_from_logical_core(eth_sender_core).y,
                                        chip_0_edm_builder.config.termination_signal_address}};

    TT_ASSERT(
        (enable_persistent_fabric && edm_termination_infos.size() == 0) ||
        (!enable_persistent_fabric && edm_termination_infos.size() > 0));
    generate_sender_worker_kernels(
        sender_program,
        sender_device,
        worker_core,
        chip0_worker_fabric_connection,
        unicast_send{2},  // 2 hops because we are looping back to ourselves
        edm_buffer_size,
        page_plus_header_size,
        num_pages_total,
        pages_per_send,
        local_worker_fabric_semaphore_id,
        local_worker_teardown_semaphore_id,
        local_worker_last_message_semaphore_id,
        local_input_buffer_address,
        src_is_dram,
        local_output_buffer_address,
        dest_is_dram,
        worker_buffer_index_semaphore_id,
        edm_termination_infos);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<IDevice*> devices = {sender_device};
    if (!enable_persistent_fabric) {
        devices.push_back(receiver_device);
    }
    log_trace(tt::LogTest, "{} programs, {} devices", programs.size(), devices.size());
    run_programs(programs, devices);
    log_info(tt::LogTest, "Reading back outputs");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        pass &= run_output_check(all_zeros, inputs, local_output_buffer) == Correctness::Correct;
    }
    return pass;
}

void generate_multi_input_test_worker_reader_kernel(
    Program& program,
    const std::vector<uint32_t>& cb_indices,
    const std::vector<const Tensor*>& tensors,
    IDevice* device,
    uint32_t page_size,
    const CoreRangeSet& worker_core_range,
    uint32_t num_pages_per_edm_buffer,
    const ttnn::ccl::v2::TensorSlice& in0_command_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& in1_command_tensor_slice,
    ttnn::ccl::cmd::CclCommandCode command_type,
    const DataMovementConfig& datamovement_kernel_config,
    const std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& chip0_worker_forward_fabric_connection,
    const std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& chip0_worker_backward_fabric_connection,
    const std::optional<ttnn::ccl::cmd::CclHostLowLevelCommandSequence>& optional_teardown_sequence,
    const ttnn::ccl::cmd::CclCommandDestArgs& dest_args) {
    bool fabric_enabled = std::holds_alternative<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args) ||
                          std::holds_alternative<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args);
    using namespace ttnn::ccl::cmd::uops;
    using namespace ttnn::ccl::cmd;
    log_trace(
        tt::LogTest,
        "Generating multi input test worker reader kernel for command type: {}",
        static_cast<uint32_t>(command_type));

    TT_FATAL(
        command_type == ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_CB ||
            command_type == ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR,
        "Unsupported tensor IO command type");

    TT_ASSERT(tensors.size() > 0 && tensors.size() <= 2);
    TT_ASSERT(cb_indices.size() == tensors.size());

    auto sender_worker_reader_kernel = ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
        program, cb_indices, tensors, worker_core_range, datamovement_kernel_config);

    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> ccl_command_stream0;
    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> ccl_command_stream1;

    // Add the main tensor slice commands
    if (command_type == ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_CB) {
        log_trace(tt::LogTest, "Adding local noc read");
        if (fabric_enabled) {
            ccl_command_stream0.push_back(
                read_tensor_slice_to_cb_for_eventual_fabric_write(in0_command_tensor_slice, cb_indices.at(0)));
            ccl_command_stream1.push_back(
                read_tensor_slice_to_cb_for_eventual_fabric_write(in1_command_tensor_slice, cb_indices.at(1)));
        } else {
            ccl_command_stream0.push_back(read_tensor_slice_to_cb(in0_command_tensor_slice, cb_indices.at(0)));
            ccl_command_stream1.push_back(read_tensor_slice_to_cb(in1_command_tensor_slice, cb_indices.at(1)));
        }
    } else {
        if (std::holds_alternative<ttnn::ccl::cmd::LocalOnlyCommandDestArgs>(dest_args)) {
            log_trace(tt::LogTest, "Adding local noc write");
            ccl_command_stream0.push_back(local_write_cb_to_tensor_slice(in0_command_tensor_slice, cb_indices.at(0)));
            ccl_command_stream1.push_back(local_write_cb_to_tensor_slice(in1_command_tensor_slice, cb_indices.at(1)));
        } else {
            if (std::holds_alternative<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args)) {
                log_trace(
                    tt::LogTest,
                    "Adding fabric unicast write command. Distance: {}. Forward: {}",
                    std::get<UnicastCommandDestArgs>(dest_args).distance_in_hops,
                    std::get<UnicastCommandDestArgs>(dest_args).is_forward_direction);
                ccl_command_stream0.push_back(fabric_write_cb_to_tensor_slice(
                    in0_command_tensor_slice,
                    cb_indices.at(0),
                    UnicastCommandDestArgs{std::get<UnicastCommandDestArgs>(dest_args)}));
                ccl_command_stream1.push_back(fabric_write_cb_to_tensor_slice(
                    in1_command_tensor_slice,
                    cb_indices.at(1),
                    UnicastCommandDestArgs{std::get<UnicastCommandDestArgs>(dest_args)}));
            } else if (std::holds_alternative<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args)) {
                log_trace(
                    tt::LogTest,
                    "Adding fabric multicast write command. Forward: {}. Backward: {}",
                    std::get<MulticastCommandDestArgs>(dest_args).num_targets_forward_direction,
                    std::get<MulticastCommandDestArgs>(dest_args).num_targets_backward_direction);
                ccl_command_stream0.push_back(fabric_write_cb_to_tensor_slice(
                    in0_command_tensor_slice,
                    cb_indices.at(0),
                    MulticastCommandDestArgs{std::get<MulticastCommandDestArgs>(dest_args)}));
                ccl_command_stream1.push_back(fabric_write_cb_to_tensor_slice(
                    in1_command_tensor_slice,
                    cb_indices.at(1),
                    MulticastCommandDestArgs{std::get<MulticastCommandDestArgs>(dest_args)}));
            } else {
                log_trace(tt::LogTest, "WTF? Should have been caught earlier");
                TT_FATAL(true, "Unsupported dest args type");
            }
        }
    }

    // Now, because we are bringing up/tearing down the fabric per op with this program, we need to queue up the
    // commands to teardown the fabric
    // We need to make sure only one of the command streams is sending out the termination signals, and we
    // need to make sure it only does that after the other command stream is done - so what we do is
    // make the termination command stream wait for a semaphore value (locally) that the other command stream
    // will set after it has finished.
    if (optional_teardown_sequence.has_value()) {
        std::ranges::copy(optional_teardown_sequence.value(), std::back_inserter(ccl_command_stream0));
    }

    ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
        program,
        sender_worker_reader_kernel,
        tensors,
        {page_size, page_size},
        device,
        num_pages_per_edm_buffer,  // TODO: get from fabric
        worker_core_range,
        ccl_command_stream0,
        ccl_command_stream1,
        chip0_worker_forward_fabric_connection,
        chip0_worker_backward_fabric_connection);
}

void generate_multi_input_test_worker_kernels_for_local_tensor_write(
    Program& program,
    IDevice* device,
    Tensor& input_tensor0,
    Tensor& input_tensor1,
    Tensor& output_tensor0,
    Tensor& output_tensor1,
    size_t first_cb_index,
    size_t second_cb_index,
    const CoreCoord& worker_core,
    const uint32_t page_plus_header_size,
    const uint32_t num_pages_per_edm_buffer,
    const ttnn::ccl::v2::TensorSlice& in0_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& in1_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& out0_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& out1_tensor_slice,
    const std::optional<ttnn::ccl::cmd::CclHostLowLevelCommandSequence>& optional_teardown_sequence,
    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& chip0_worker_forward_fabric_connection,
    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& chip0_worker_backward_fabric_connection,
    const ttnn::ccl::cmd::CclCommandDestArgs& dest_args) {
    // Just want a dummy DF
    tt::DataFormat df = (page_plus_header_size - PACKET_HEADER_SIZE_BYTES) == 1024   ? tt::DataFormat::Bfp8
                        : (page_plus_header_size - PACKET_HEADER_SIZE_BYTES) == 2048 ? tt::DataFormat::Float16
                                                                                     : tt::DataFormat::Float32;

    {
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_plus_header_size, {{first_cb_index, df}})
                .set_page_size(first_cb_index, page_plus_header_size);
        CBHandle cb0 = CreateCircularBuffer(program, worker_core, cb_src0_config);
    }
    {
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(
                2 * num_pages_per_edm_buffer * page_plus_header_size, {{second_cb_index, df}})
                .set_page_size(second_cb_index, page_plus_header_size);
        CBHandle cb1 = CreateCircularBuffer(program, worker_core, cb_src1_config);
    }

    generate_multi_input_test_worker_reader_kernel(
        program,
        {first_cb_index, second_cb_index},
        {&input_tensor0, &input_tensor1},
        device,
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        CoreRangeSet({CoreRange(worker_core)}),
        num_pages_per_edm_buffer,
        in0_tensor_slice,
        in1_tensor_slice,
        ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_CB,
        tt_metal::ReaderDataMovementConfig{},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        dest_args);

    generate_multi_input_test_worker_reader_kernel(
        program,
        {first_cb_index, second_cb_index},
        {&output_tensor0, &output_tensor1},
        device,
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        CoreRangeSet({CoreRange(worker_core)}),
        num_pages_per_edm_buffer,
        out0_tensor_slice,
        out1_tensor_slice,
        ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR,
        tt_metal::WriterDataMovementConfig{},
        chip0_worker_forward_fabric_connection,
        chip0_worker_backward_fabric_connection,
        optional_teardown_sequence,
        dest_args);
}

bool RunLocalTestWithMultiInputReaders(
    const std::vector<tt_metal::IDevice*>& devices,
    std::vector<Program>& programs,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& line_fabric,

    Tensor& input_tensor0,
    Tensor& input_tensor1,
    Tensor& output_tensor0,
    Tensor& output_tensor1,
    std::vector<Tensor> input0_tensors,   // Device
    std::vector<Tensor> input1_tensors,   // Device
    std::vector<Tensor> output0_tensors,  // Device
    std::vector<Tensor> output1_tensors,  // Device

    const ttnn::ccl::v2::TensorSlice& in0_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& in1_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& out0_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& out1_tensor_slice,

    const uint32_t page_size,
    TwoInputReaderKernelWriteMode test_mode,
    const ttnn::ccl::cmd::CclCommandDestArgs& dest_args,
    std::optional<SubdeviceInfo>& subdevice_managers,
    bool enable_persistent_fabric) {
    const bool fabric_enabled = test_mode != TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK;
    tt_metal::IDevice* device = devices.at(0);
    for (size_t i = 0; i < devices.size(); i++) {
        log_info(tt::LogTest, "Device[{}] ID: {}", i, devices.at(i)->id());
    }
    auto program_ptrs = std::vector<Program*>();
    program_ptrs.reserve(devices.size());
    std::ranges::transform(programs, std::back_inserter(program_ptrs), [](auto& p) { return &p; });

    size_t output_tensor_dest_device_index = 0;
    if (fabric_enabled) {
        if (std::holds_alternative<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args)) {
            log_info(
                tt::LogTest,
                "Unicast command dest args. Distance in hops: {}",
                std::get<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args).distance_in_hops);
            output_tensor_dest_device_index =
                std::get<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args).distance_in_hops;
            TT_ASSERT(output_tensor_dest_device_index != 0, "Output tensor destination device index must be non-zero");
            TT_ASSERT(test_mode == TwoInputReaderKernelWriteMode::FABRIC_UNICAST);
        } else if (std::holds_alternative<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args)) {
            log_info(
                tt::LogTest,
                "Multicast command dest args. Number of targets forward direction: {}",
                std::get<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args).num_targets_forward_direction);
            output_tensor_dest_device_index =
                std::get<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args).num_targets_forward_direction;
            TT_ASSERT(output_tensor_dest_device_index != 0, "Output tensor destination device index must be non-zero");
            TT_ASSERT(test_mode == TwoInputReaderKernelWriteMode::FABRIC_MULTICAST);
        }
    } else {
        log_info(tt::LogTest, "No fabric enabled");
        TT_ASSERT(
            std::holds_alternative<ttnn::ccl::cmd::DestTypeArgsNull>(dest_args), "Local command dest args expected");
    }

    std::size_t page_plus_header_size = page_size + sizeof(tt::tt_fabric::PacketHeader);

    auto first_cb_index = tt::CB::c_in0;
    auto second_cb_index = tt::CB::c_in1;

    auto output_tensor_dest_device = devices.at(output_tensor_dest_device_index);
    TT_ASSERT(input_tensor0.get_logical_shape()[-2] != 1);

    bool is_fabric_mcast = std::holds_alternative<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args);

    auto input_tensor0_device = input0_tensors.at(0);
    auto input_tensor1_device = input1_tensors.at(0);
    auto output_tensor0_device = output0_tensors.at(output_tensor_dest_device_index);
    auto output_tensor1_device = output1_tensors.at(output_tensor_dest_device_index);

    log_info(tt::LogTest, "input_tensor0_device->address(): {}", input_tensor0_device.buffer()->address());
    log_info(tt::LogTest, "input_tensor1_device->address(): {}", input_tensor1_device.buffer()->address());
    log_info(
        tt::LogTest,
        "output_tensor0_device->address(): {} on device {}",
        output_tensor0_device.buffer()->address(),
        output_tensor_dest_device->id());
    log_info(
        tt::LogTest,
        "output_tensor1_device->address(): {} on device {}",
        output_tensor1_device.buffer()->address(),
        output_tensor_dest_device->id());

    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    const auto& worker_core = CoreCoord(0, 0);

    const size_t num_pages_per_edm_buffer = 2;

    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> chip0_worker_forward_fabric_connection =
        fabric_enabled ? line_fabric->uniquely_connect_worker(devices[0], ttnn::ccl::EdmLineFabricOpInterface::FORWARD)
                       : std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>{std::nullopt};

    // always at start of line for now
    std::optional<std::vector<tt::tt_fabric::edm_termination_info_t>> edm_termination_infos =
        (!fabric_enabled || enable_persistent_fabric)
            ? std::optional<std::vector<tt::tt_fabric::edm_termination_info_t>>{std::nullopt}
            : line_fabric->generate_ordered_termination_info_farthest_to_nearest();
    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> chip0_worker_backward_fabric_connection = std::nullopt;

    std::optional<ttnn::ccl::SyncModeSpec> sync_details;
    std::optional<CoreCoord> teardown_worker_core;
    std::optional<ttnn::ccl::cmd::CclHostLowLevelCommandSequence> teardown_command_stream;
    if (fabric_enabled && !enable_persistent_fabric) {
        teardown_worker_core = worker_core;

        sync_details = ttnn::ccl::SyncModeSpec{};
        sync_details->core = teardown_worker_core.value();
        sync_details->add_signal(tt::tt_metal::CreateSemaphore(programs.at(0), teardown_worker_core.value(), 0), 1);
        teardown_command_stream = {ttnn::ccl::cmd::uops::local_core_semaphore_inc(sync_details->sem_ids.at(0), 1)};
        TT_FATAL(edm_termination_infos.has_value(), "EDM termination infos must be set if fabric is enabled");
        ttnn::ccl::cmd::CclHostLowLevelCommandSequence teardown_commands;

        teardown_commands = ttnn::ccl::worker_detail::build_ccl_cmd_proc_teardown_commands(
            programs.at(0),
            device,
            nullptr,  // forward device - in this test, we have a single source doing all teardown
            devices.size(),
            0,
            edm_termination_infos.value(),
            sync_details.value(),
            line_fabric.value());
        std::ranges::copy(teardown_commands, std::back_inserter(teardown_command_stream.value()));
    }

    generate_multi_input_test_worker_kernels_for_local_tensor_write(
        programs.at(0),
        device,
        input_tensor0_device,
        input_tensor1_device,
        output_tensor0_device,
        output_tensor1_device,
        first_cb_index,
        second_cb_index,
        worker_core,
        page_plus_header_size,
        num_pages_per_edm_buffer,
        in0_tensor_slice,
        in1_tensor_slice,
        out0_tensor_slice,
        out1_tensor_slice,
        teardown_command_stream,
        chip0_worker_forward_fabric_connection,
        chip0_worker_backward_fabric_connection,
        dest_args);

    if (!enable_persistent_fabric) {
        log_info(tt::LogTest, "Building EDM kernels");
        line_fabric->build_kernels();
    }

    log_info(tt::LogTest, "persistent_fabric: {}", enable_persistent_fabric);
    log_info(tt::LogTest, "subdevice_managers.has_value(): {}", subdevice_managers.has_value());
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    run_programs(programs, enable_persistent_fabric ? std::vector<IDevice*>{devices[0]} : devices);
    log_info(tt::LogTest, "Finished");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        log_info(tt::LogTest, "Reading back outputs");
        auto output0_cpu = output_tensor0_device.cpu(true, ttnn::DefaultQueueId);
        auto output1_cpu = output_tensor1_device.cpu(true, ttnn::DefaultQueueId);

        auto in0_tensor_copyback_cpu = input_tensor0_device.cpu(true, ttnn::DefaultQueueId);
        auto in1_tensor_copyback_cpu = input_tensor1_device.cpu(true, ttnn::DefaultQueueId);

        auto in0_tensor_copyback = tt::tt_metal::host_buffer::get_as<uint32_t>(in0_tensor_copyback_cpu);
        auto in1_tensor_copyback = tt::tt_metal::host_buffer::get_as<uint32_t>(in1_tensor_copyback_cpu);

        auto in0_tensor_data = tt::tt_metal::host_buffer::get_as<uint32_t>(input_tensor0);
        auto in1_tensor_data = tt::tt_metal::host_buffer::get_as<uint32_t>(input_tensor1);
        auto out0_tensor_data = tt::tt_metal::host_buffer::get_as<uint32_t>(output0_cpu);
        auto out1_tensor_data = tt::tt_metal::host_buffer::get_as<uint32_t>(output1_cpu);

        bool input0_copyback_check_passed =
            run_output_check(in0_tensor_data, in0_tensor_copyback) == Correctness::Correct;
        bool input1_copyback_check_passed =
            run_output_check(in1_tensor_data, in1_tensor_copyback) == Correctness::Correct;
        TT_FATAL(input0_copyback_check_passed, "Input 0 copyback check failed");
        TT_FATAL(input1_copyback_check_passed, "Input 1 copyback check failed");

        log_info(tt::LogTest, "Comparing outputs");
        pass &= run_output_check(in0_tensor_data, out0_tensor_data) == Correctness::Correct;
        if (pass) {
            log_info(tt::LogTest, "Output check passed for output 0");
        } else {
            log_error(tt::LogTest, "Output check failed for output 0");
        }
        pass &= run_output_check(in1_tensor_data, out1_tensor_data) == Correctness::Correct;
        if (pass) {
            log_info(tt::LogTest, "Output check passed for output 1");
        } else {
            log_error(tt::LogTest, "Output check failed for output 1");
        }
    }

    return pass;
}

bool RunLineFabricTest(
    std::vector<tt_metal::IDevice*> devices,
    std::vector<Program>& programs,

    const size_t mcast_first_chip,
    const size_t mcast_last_chip,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram,

    std::optional<SubdeviceInfo>& subdevice_managers,
    ttnn::ccl::EdmLineFabricOpInterface& line_fabric,
    bool enable_persistent_fabric) {
    std::size_t page_plus_header_size = page_size + sizeof(tt::tt_fabric::PacketHeader);
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    static constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes + PACKET_HEADER_SIZE_BYTES;
    const size_t local_chip_id = 0;
    const size_t remote_chip_id = 1;
    auto program_ptrs = std::vector<Program*>(devices.size());
    std::transform(programs.begin(), programs.end(), program_ptrs.begin(), [](auto& program) { return &program; });

    std::vector<CoreCoord> worker_cores = {CoreCoord(0, 0)};

    // Generate inputs
    ////////////////////////////////////////////////////////////////////////////
    //   SETUP THE INPUT CB
    ////////////////////////////////////////////////////////////////////////////
    BankedConfig test_config = BankedConfig{
        .num_pages = num_pages_total,
        .size_bytes = tensor_size_bytes,
        .page_size_bytes = page_size,
        .input_buffer_type = src_is_dram ? BufferType::DRAM : BufferType::L1,
        .output_buffer_type = dest_is_dram ? BufferType::DRAM : BufferType::L1,
        .l1_data_format = tt::DataFormat::Float16_b};

    // Input buffer
    auto [local_input_buffer, inputs] = build_input_buffer(devices[0], tensor_size_bytes, test_config);
    auto local_input_buffer_address = local_input_buffer->address();

    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    // output buffers
    TT_ASSERT(
        enable_persistent_fabric || mcast_first_chip <= mcast_last_chip,
        "mcast_first_chip must be less than or equal to mcast_last_chip");
    TT_ASSERT(
        enable_persistent_fabric || mcast_last_chip < devices.size(),
        "mcast_last_chip must be less than the number of devices");
    std::vector<std::shared_ptr<Buffer>> output_buffers;
    output_buffers.reserve(devices.size());
    for (size_t i = 0; i < devices.size(); i++) {
        if (i == 0) {
            output_buffers.push_back(CreateBuffer(InterleavedBufferConfig{
                devices.at(i), test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type}));
        } else {
            output_buffers.push_back(CreateBuffer(
                InterleavedBufferConfig{
                    devices.at(i), test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type},
                output_buffers[0]->address()));
        }
        tt_metal::detail::WriteToBuffer(output_buffers.back(), all_zeros);
    }
    auto local_output_buffer_address = output_buffers[0]->address();
    bool all_same_addr = std::ranges::all_of(output_buffers, [local_output_buffer_address](const auto& buffer) {
        return buffer->address() == local_output_buffer_address;
    });
    TT_ASSERT(all_same_addr, "All output buffers must have the same address");

    ////////////////////////////////////////////////////////////////////////////
    //   Setup Semaphores and Builders
    ////////////////////////////////////////////////////////////////////////////

    auto local_worker_fabric_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    auto local_worker_teardown_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    auto local_worker_last_message_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    auto worker_buffer_index_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    const auto& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    const auto edm_termination_infos = enable_persistent_fabric
                                           ? std::vector<tt::tt_fabric::edm_termination_info_t>{}
                                           : line_fabric.generate_ordered_termination_info_farthest_to_nearest();

    auto chip0_worker_fabric_connection =
        line_fabric.uniquely_connect_worker(devices[0], ttnn::ccl::EdmLineFabricOpInterface::FORWARD);

    const std::size_t pages_per_send =
        (chip0_worker_fabric_connection.buffer_size_bytes - PACKET_HEADER_SIZE_BYTES) / page_size;
    generate_sender_worker_kernels(
        programs[0],
        devices[0],
        worker_core,
        chip0_worker_fabric_connection,
        mcast_send{mcast_first_chip, mcast_last_chip - mcast_first_chip + 1},
        edm_buffer_size,
        page_plus_header_size,
        num_pages_total,
        pages_per_send,
        local_worker_fabric_semaphore_id,
        local_worker_teardown_semaphore_id,
        local_worker_last_message_semaphore_id,
        local_input_buffer_address,
        src_is_dram,
        local_output_buffer_address,
        dest_is_dram,
        worker_buffer_index_semaphore_id,
        edm_termination_infos);

    ////////////////////////////////////////////////////////////////////////////
    // Build EDM Kernels
    ////////////////////////////////////////////////////////////////////////////
    if (!enable_persistent_fabric) {
        line_fabric.build_kernels();
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    run_programs(programs, devices);
    log_info(tt::LogTest, "Reading back outputs");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        // Check all output buffers. Make sure only the buffers in the mcast range are
        // non-zero. All other buffers outside the range should be zero filled
        TT_ASSERT(
            !std::all_of(inputs.begin(), inputs.end(), [](uint32_t x) { return x == 0; }),
            "Input buffer expected to not be all 0");
        for (size_t i = 0; i < output_buffers.size(); i++) {
            bool compare_with_input = (mcast_first_chip <= i && i <= mcast_last_chip);
            auto& golden_tensor = compare_with_input ? inputs : all_zeros;
            pass &= run_output_check(all_zeros, golden_tensor, output_buffers.at(i)) == Correctness::Correct;
        }
    }

    return pass;
}

void persistent_fabric_teardown_sequence(
    const std::vector<IDevice*>& devices,
    std::optional<SubdeviceInfo>& subdevice_managers,
    ttnn::ccl::EdmLineFabricOpInterface& line_fabric,
    tt::tt_fabric::TerminationSignal termination_mode = tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE) {
    log_info("Tearing down fabric");

    // Wait for workers to finish
    auto d0_worker_subdevice = devices[0]->get_sub_device_ids()[TEST_WORKERS_SUBDEVICE_INDEX];
    tt_metal::Finish(devices[0]->command_queue(), {subdevice_managers->worker_subdevice_id.at(devices[0]->id())});

    // Teardown the fabric
    line_fabric.teardown_from_host(termination_mode);

    // wait for fabric teardown to finish
    std::ranges::for_each(devices, [&](IDevice* d) {
        tt_metal::Finish(d->command_queue(), {subdevice_managers->fabric_subdevice_id.at(d->id())});
    });
}

void setup_test_with_persistent_fabric(
    const std::vector<IDevice*>& devices,
    std::vector<Program>& programs,
    std::optional<SubdeviceInfo>& subdevice_managers,
    std::optional<std::vector<Program>>& fabric_programs,
    std::vector<Program*>& fabric_program_ptrs,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& line_fabric,
    bool enable_persistent_fabric,
    std::optional<size_t> num_links = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    size_t switch_interval = 0,
    bool loopback_on_last_device = false,
    bool is_galaxy = false) {
    if (enable_persistent_fabric) {
        log_info(tt::LogTest, "Enabling persistent fabric");
        fabric_programs = std::vector<Program>(devices.size());
        subdevice_managers = create_subdevices(devices);
        std::transform(
            fabric_programs->begin(), fabric_programs->end(), std::back_inserter(fabric_program_ptrs), [](auto& p) {
                return &p;
            });
    } else {
        std::transform(
            programs.begin(), programs.end(), std::back_inserter(fabric_program_ptrs), [](auto& p) { return &p; });
    }

    line_fabric = ttnn::ccl::EdmLineFabricOpInterface(
        devices, fabric_program_ptrs, enable_persistent_fabric, num_links.value_or(1), false, topology, is_galaxy);
    line_fabric->set_firmware_context_switch_interval(switch_interval);
    if (loopback_on_last_device) {
        for (auto& edm_builder : line_fabric->edm_builders_backward_direction.at(devices.back()->id())) {
            log_trace(
                tt::LogTest,
                "Implementing loopback on device {} by connecting 1D fabric endpoint to itself at x={}, y={}",
                devices.back()->id(),
                edm_builder.my_noc_x,
                edm_builder.my_noc_y);
            edm_builder.connect_to_downstream_edm(edm_builder);
        }
    }

    if (enable_persistent_fabric) {
        TT_FATAL(fabric_programs.has_value(), "Fabric programs must be set if fabric is enabled");
        TT_FATAL(devices.size() == fabric_programs->size(), "Number of devices must match number of programs");

        log_info(tt::LogTest, "Building EDM kernels");
        line_fabric->build_kernels();
        build_and_enqueue(devices, *fabric_programs);
    }
}

// RESUME HERE AND IMPLEMENT MCAST TEST
int TestLineFabricEntrypoint(
    const size_t mcast_first_chip,
    const size_t mcast_last_chip,
    const uint32_t page_size,
    const uint32_t num_pages_total,
    const bool src_is_dram,
    const bool dest_is_dram,
    bool enable_persistent_fabric) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info("This test can only be run on T3000 devices");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return 0;
    }

    Fabric1DFixture test_fixture;
    auto view = *(test_fixture.view_);

    // build a line of devices
    std::vector<IDevice*> devices = {
        view.get_device(MeshCoordinate(0, 0)),
        view.get_device(MeshCoordinate(0, 1)),
        view.get_device(MeshCoordinate(0, 2)),
        view.get_device(MeshCoordinate(0, 3))};
    std::vector<Program> programs(enable_persistent_fabric ? 1 : devices.size());
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> line_fabric;
    setup_test_with_persistent_fabric(
        devices,
        programs,
        subdevice_managers,
        fabric_programs,
        fabric_program_ptrs,
        line_fabric,
        enable_persistent_fabric);

    auto launch_workers = [&](std::vector<Program>& _programs) -> bool {
        bool success = false;
        try {
            success = RunLineFabricTest(
                enable_persistent_fabric ? std::vector<IDevice*>{devices[0]} : devices,
                _programs,
                // fabric_hops,

                mcast_first_chip,
                mcast_last_chip,

                page_size,
                num_pages_total,
                src_is_dram,
                dest_is_dram,

                subdevice_managers,
                line_fabric.value(),
                enable_persistent_fabric);

        } catch (std::exception& e) {
            log_error("Caught exception: {}", e.what());
            test_fixture.TearDown();
            return false;
        }
        return success;
    };
    bool success = launch_workers(programs);

    if (enable_persistent_fabric) {
        std::vector<Program> second_run_programs(1);
        success = launch_workers(second_run_programs);
        persistent_fabric_teardown_sequence(
            devices, subdevice_managers, line_fabric.value(), tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    }

    test_fixture.TearDown();

    return success ? 0 : -1;
}

int TestLoopbackEntrypoint(
    const uint32_t page_size,
    const uint32_t num_pages_total,
    const bool src_is_dram,
    const bool dest_is_dram,
    bool enable_persistent_fabric) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info("This test can only be run on T3000 devices");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return 0;
    }

    Fabric1DFixture test_fixture;
    auto view = *(test_fixture.view_);

    const auto& device_0 = view.get_device(MeshCoordinate(0, 0));
    const auto& device_1 = view.get_device(MeshCoordinate(0, 1));

    const auto& active_eth_cores = device_0->get_active_ethernet_cores(true);
    auto eth_sender_core_iter = active_eth_cores.begin();
    auto eth_sender_core_iter_end = active_eth_cores.end();
    chip_id_t device_id = std::numeric_limits<chip_id_t>::max();
    tt_xy_pair eth_receiver_core;
    bool initialized = false;
    tt_xy_pair eth_sender_core;
    do {
        TT_FATAL(eth_sender_core_iter != eth_sender_core_iter_end, "Error");
        std::tie(device_id, eth_receiver_core) = device_0->get_connected_ethernet_core(*eth_sender_core_iter);
        eth_sender_core = *eth_sender_core_iter;
        eth_sender_core_iter++;
    } while (device_id != device_1->id());
    TT_ASSERT(device_id == device_1->id());
    // const auto& device_1 = test_fixture.mesh_device_->get_device(device_id);

    std::vector<Program> programs(enable_persistent_fabric ? 1 : 2);
    std::optional<std::vector<Program>> fabric_programs;
    auto& sender_program = programs.at(0);
    if (enable_persistent_fabric) {
        log_info(tt::LogTest, "Enabling persistent fabric");
        fabric_programs = std::vector<Program>(2);
        subdevice_managers = create_subdevices({device_0, device_1});
    }

    auto& fabric_sender_program = enable_persistent_fabric ? fabric_programs->at(0) : sender_program;
    auto& fabric_receiver_program = enable_persistent_fabric ? fabric_programs->at(1) : programs.at(1);
    IDevice* sender_device = device_0;
    IDevice* receiver_device = device_1;

    static constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes + PACKET_HEADER_SIZE_BYTES;
    const chip_id_t local_chip_id = 0;
    const chip_id_t remote_chip_id = 1;
    const auto& edm_config = tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size);
    auto chip_0_edm_builder = tt::tt_fabric::FabricEriscDatamoverBuilder::build(
        sender_device,
        fabric_sender_program,
        eth_sender_core,
        local_chip_id,
        remote_chip_id,
        edm_config,
        enable_persistent_fabric);
    chip_0_edm_builder.set_firmware_context_switch_interval(0);
    auto chip_1_edm_builder = tt::tt_fabric::FabricEriscDatamoverBuilder::build(
        receiver_device,
        fabric_receiver_program,
        eth_receiver_core,
        remote_chip_id,
        local_chip_id,
        edm_config,
        enable_persistent_fabric);
    chip_1_edm_builder.set_firmware_context_switch_interval(0);
    // Create the loopback connection on the second device
    chip_1_edm_builder.connect_to_downstream_edm(chip_1_edm_builder);
    auto local_edm_kernel = ttnn::ccl::generate_edm_kernel(
        fabric_sender_program, sender_device, chip_0_edm_builder, eth_sender_core, NOC::NOC_0);
    auto remote_edm_kernel = ttnn::ccl::generate_edm_kernel(
        fabric_receiver_program, receiver_device, chip_1_edm_builder, eth_receiver_core, NOC::NOC_0);

    if (enable_persistent_fabric) {
        tt::tt_metal::detail::CompileProgram(sender_device, fabric_sender_program);
        tt::tt_metal::detail::CompileProgram(receiver_device, fabric_receiver_program);
        tt_metal::EnqueueProgram(sender_device->command_queue(), fabric_sender_program, false);
        tt_metal::EnqueueProgram(receiver_device->command_queue(), fabric_receiver_program, false);
    }
    log_trace(tt::LogTest, "{} programs ", programs.size());
    bool success = false;
    try {
        success = RunLoopbackTest(
            device_0,
            device_1,

            eth_sender_core,
            eth_receiver_core,

            page_size,
            num_pages_total,
            src_is_dram,
            dest_is_dram,
            programs,
            chip_0_edm_builder,
            subdevice_managers,
            enable_persistent_fabric);
    } catch (std::exception& e) {
        log_error("Caught exception: {}", e.what());
        test_fixture.TearDown();
        return -1;
    }

    if (enable_persistent_fabric) {
        // Run the test twice with a single fabric invocation

        std::vector<Program> second_programs(1);
        try {
            success = RunLoopbackTest(
                device_0,
                device_1,

                eth_sender_core,
                eth_receiver_core,

                page_size,
                num_pages_total,
                src_is_dram,
                dest_is_dram,
                second_programs,
                chip_0_edm_builder,
                subdevice_managers,
                enable_persistent_fabric);
        } catch (std::exception& e) {
            log_error("Caught exception: {}", e.what());
            test_fixture.TearDown();
            return -1;
        }
        // Wait for worker programs to finish

        auto d0_worker_subdevice = device_0->get_sub_device_ids()[TEST_WORKERS_SUBDEVICE_INDEX];
        auto d1_worker_subdevice = device_1->get_sub_device_ids()[TEST_WORKERS_SUBDEVICE_INDEX];
        auto d0_fabric_subdevice = device_0->get_sub_device_ids()[TEST_EDM_FABRIC_SUBDEVICE_INDEX];
        auto d1_fabric_subdevice = device_1->get_sub_device_ids()[TEST_EDM_FABRIC_SUBDEVICE_INDEX];
        // Teardown the fabric
        tt_metal::Finish(sender_device->command_queue(), {d0_worker_subdevice});
        // tt_metal::Finish(receiver_device->command_queue(), {d1_worker_subdevice});

        // Notify fabric of teardown
        chip_1_edm_builder.teardown_from_host(receiver_device);
        chip_0_edm_builder.teardown_from_host(sender_device);

        // wait for fabric finish
        tt_metal::Finish(sender_device->command_queue(), {d0_fabric_subdevice});
        tt_metal::Finish(receiver_device->command_queue(), {d1_fabric_subdevice});
    }

    test_fixture.TearDown();

    return success ? 0 : -1;
}

inline bool TestMultiInputReaderKernel(
    size_t fabric_num_devices,
    Tensor& input_tensor0,
    const MemoryConfig& input_tensor0_mem_config,
    Tensor& input_tensor1,
    const MemoryConfig& input_tensor1_mem_config,
    Tensor& output_tensor0,
    const MemoryConfig& output_tensor0_mem_config,
    Tensor& output_tensor1,
    const MemoryConfig& output_tensor1_mem_config,

    const ttnn::ccl::v2::TensorSlice& in0_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& in1_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& out0_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& out1_tensor_slice,

    const uint32_t page_size,

    TwoInputReaderKernelWriteMode test_mode,
    const ttnn::ccl::cmd::CclCommandDestArgs& dest_args,
    bool enable_persistent_fabric) {
    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info("This test can only be run on T3000 devices");
        return true;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return true;
    }
    Fabric1DFixture test_fixture;

    TT_FATAL(
        !enable_persistent_fabric || test_mode != TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK,
        "Test configuration issue. Set local writeback mode with persistent fabric");

    auto view = *(test_fixture.view_);

    std::vector<IDevice*> devices;
    devices.reserve(fabric_num_devices);
    for (size_t i = 0; i < fabric_num_devices; i++) {
        devices.push_back(view.get_device(MeshCoordinate(0, i)));
    }

    std::vector<Program> programs(enable_persistent_fabric ? 1 : devices.size());
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> line_fabric;
    setup_test_with_persistent_fabric(
        devices,
        programs,
        subdevice_managers,
        fabric_programs,
        fabric_program_ptrs,
        line_fabric,
        enable_persistent_fabric);

    std::vector<Tensor> input0_tensors_device;
    std::vector<Tensor> input1_tensors_device;
    std::vector<Tensor> output0_tensors_device;
    std::vector<Tensor> output1_tensors_device;

    // All this garbage is to make sure the test sets up buffer addresses correctly so we can safely
    // multicast to a consistent destination address
    for (size_t i = 0; i < devices.size(); i++) {
        input0_tensors_device.push_back(
            input_tensor0.to_device(devices.at(i), input_tensor0_mem_config, ttnn::DefaultQueueId));
        input1_tensors_device.push_back(
            input_tensor1.to_device(devices.at(i), input_tensor1_mem_config, ttnn::DefaultQueueId));
        output0_tensors_device.push_back(
            output_tensor0.to_device(devices.at(i), output_tensor0_mem_config, ttnn::DefaultQueueId));
        output1_tensors_device.push_back(
            output_tensor1.to_device(devices.at(i), output_tensor1_mem_config, ttnn::DefaultQueueId));
    }
    TT_FATAL(
        !enable_persistent_fabric || subdevice_managers.has_value(),
        "Subdevice managers must be set if fabric is enabled");
    auto launch_ccl_command_interpreter_workers = [&](std::vector<Program>& _programs) {
        return RunLocalTestWithMultiInputReaders(
            devices,
            _programs,
            line_fabric,

            input_tensor0,
            input_tensor1,
            output_tensor0,
            output_tensor1,

            input0_tensors_device,
            input1_tensors_device,
            output0_tensors_device,
            output1_tensors_device,

            in0_tensor_slice,
            in1_tensor_slice,
            out0_tensor_slice,
            out1_tensor_slice,

            page_size,
            test_mode,
            dest_args,
            subdevice_managers,
            enable_persistent_fabric);
    };

    auto pass = launch_ccl_command_interpreter_workers(programs);
    if (enable_persistent_fabric) {
        std::vector<Program> second_run_programs(1);
        // It looks suspicious that we are dropping the first result but there are two reasons we do this
        // 1) We really only care that we can run back to back safely
        // 2) The first run will end up racing with host and copy-back because there is no
        //    receiver on the destination that can signal to us when we are done. We need to add this
        //    to the test to make it more robust but that is future work
        pass = launch_ccl_command_interpreter_workers(second_run_programs);
        pass = true;

        // Due to race between host and device some packets are in flight by the time host sends shutdown signals so
        // some get shutdown in between any packets in the pipeline. This can only be fixed by having a "drainer" op to
        // make sure it receives all writes before exiting
        persistent_fabric_teardown_sequence(
            devices, subdevice_managers, line_fabric.value(), tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);

        log_info(tt::LogTest, "Finished");
        for (auto d : devices) {
            tt_metal::Synchronize(d, *ttnn::DefaultQueueId);
        }
    }
    return pass;
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
////               LOCAL CHIP TENSOR READ?WRITE (2 INPUT)
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

ttnn::ccl::Shape4D<uint32_t> shape_to_shape_in_tiles(const ttnn::Shape& shape) {
    auto logical_shape = shape;
    logical_shape[-2] /= tt::constants::TILE_HEIGHT;
    logical_shape[-1] /= tt::constants::TILE_WIDTH;
    EXPECT_TRUE(logical_shape.size() == 4);
    ttnn::ccl::Shape4D<uint32_t> shape_in_tiles = {
        logical_shape[0], logical_shape[1], logical_shape[2], logical_shape[3]};
    return shape_in_tiles;
}

bool RunMultiInputReaderTestPropagateFullTensorIn(
    const ttnn::Shape& tensor_shape,
    const Layout& layout,
    const MemoryConfig& in0_memory_config,
    const MemoryConfig& in1_memory_config,
    const MemoryConfig& out0_memory_config,
    const MemoryConfig& out1_memory_config,
    TwoInputReaderKernelWriteMode test_writeback_mode) {
    auto num_elems = std::reduce(tensor_shape.cbegin(), tensor_shape.cend(), 1, std::multiplies<uint32_t>());
    Tensor input_tensor0 =
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::UINT32), tensor_shape).to_layout(layout);
    Tensor input_tensor1 =
        ttnn::experimental::view(ttnn::arange(num_elems, 2 * num_elems, 1, DataType::UINT32), tensor_shape)
            .to_layout(layout);
    Tensor output_tensor0 = ttnn::experimental::view(ttnn::ones(tensor_shape, DataType::UINT32, layout), tensor_shape);
    Tensor output_tensor1 = ttnn::experimental::view(ttnn::ones(tensor_shape, DataType::UINT32, layout), tensor_shape);

    size_t page_size = tile_size(DataFormat::RawUInt32);

    ttnn::ccl::Shape4D<uint32_t> tensor_shape_in_pages = shape_to_shape_in_tiles(tensor_shape);
    ttnn::ccl::Shape4D<uint32_t> tensor_slice_shape_in_pages = tensor_shape_in_pages;
    ttnn::ccl::Shape4D<uint32_t> tensor_slice_offset = {0, 0, 0, 0};
    ttnn::ccl::Shape4D<uint32_t> worker_slice_shape = tensor_shape_in_pages;
    ttnn::ccl::Shape4D<uint32_t> worker_slice_offset = {0, 0, 0, 0};

    ttnn::ccl::v2::TensorSlice tensor_slice{
        tensor_shape_in_pages,
        tensor_slice_shape_in_pages,
        tensor_slice_offset,
        worker_slice_shape,
        worker_slice_offset};

    const auto in0_tensor_slice = tensor_slice;
    const auto in1_tensor_slice = tensor_slice;
    const auto out0_tensor_slice = tensor_slice;
    const auto out1_tensor_slice = tensor_slice;

    auto pass = TestMultiInputReaderKernel(
        1,
        input_tensor0,
        in0_memory_config,
        input_tensor1,
        in1_memory_config,
        output_tensor0,
        out0_memory_config,
        output_tensor1,
        out1_memory_config,

        in0_tensor_slice,
        in1_tensor_slice,
        out0_tensor_slice,
        out1_tensor_slice,

        page_size,
        test_writeback_mode,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs{},
        false);

    return pass;
}

// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////
// ////               FABRIC MCAST TENSOR WRITE (2 INPUT)
// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////

void RunFabricMcastFullTensorPropagateTest(
    const ttnn::Shape& tensor_shape, size_t distance_dest_device, size_t num_devices, bool enable_persistent_fabric) {
    const Layout layout = Layout::TILE;
    const MemoryConfig in0_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const MemoryConfig in1_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const MemoryConfig out0_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const MemoryConfig out1_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);

    auto num_elems = std::reduce(tensor_shape.cbegin(), tensor_shape.cend(), 1, std::multiplies<uint32_t>());
    Tensor input_tensor1 =
        ttnn::experimental::view(ttnn::arange(num_elems, 2 * num_elems, 1, DataType::UINT32), tensor_shape)
            .to_layout(layout);
    Tensor input_tensor0 =
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::UINT32), tensor_shape).to_layout(layout);
    Tensor output_tensor1 = ttnn::experimental::view(ttnn::ones(tensor_shape, DataType::UINT32, layout), tensor_shape);
    Tensor output_tensor0 = ttnn::experimental::view(ttnn::ones(tensor_shape, DataType::UINT32, layout), tensor_shape);
    ASSERT_EQ(input_tensor0.get_logical_shape(), tensor_shape);
    ASSERT_EQ(input_tensor1.get_logical_shape(), tensor_shape);
    ASSERT_EQ(output_tensor0.get_logical_shape(), tensor_shape);
    ASSERT_EQ(output_tensor1.get_logical_shape(), tensor_shape);

    size_t page_size = tile_size(DataFormat::RawUInt32);

    ttnn::ccl::Shape4D<uint32_t> tensor_shape_in_pages = shape_to_shape_in_tiles(tensor_shape);
    ttnn::ccl::Shape4D<uint32_t> tensor_slice_shape_in_pages = tensor_shape_in_pages;
    ttnn::ccl::Shape4D<uint32_t> tensor_slice_offset = {0, 0, 0, 0};
    ttnn::ccl::Shape4D<uint32_t> worker_slice_shape = tensor_shape_in_pages;
    ttnn::ccl::Shape4D<uint32_t> worker_slice_offset = {0, 0, 0, 0};

    ttnn::ccl::v2::TensorSlice tensor_slice{
        tensor_shape_in_pages,
        tensor_slice_shape_in_pages,
        tensor_slice_offset,
        worker_slice_shape,
        worker_slice_offset};

    const auto in0_tensor_slice = tensor_slice;
    const auto in1_tensor_slice = tensor_slice;
    const auto out0_tensor_slice = tensor_slice;
    const auto out1_tensor_slice = tensor_slice;

    ttnn::ccl::cmd::CclCommandDestArgs dest_args = ttnn::ccl::cmd::MulticastCommandDestArgs{distance_dest_device, 0};
    auto pass = TestMultiInputReaderKernel(
        num_devices,
        input_tensor0,
        in0_memory_config,
        input_tensor1,
        in1_memory_config,
        output_tensor0,
        out0_memory_config,
        output_tensor1,
        out1_memory_config,

        in0_tensor_slice,
        in1_tensor_slice,
        out0_tensor_slice,
        out1_tensor_slice,

        page_size,
        TwoInputReaderKernelWriteMode::FABRIC_MULTICAST,
        dest_args,
        enable_persistent_fabric);

    ASSERT_TRUE(pass);
}

bool RunPipelinedWorkersTest(

    ttnn::Shape tensor_shape,
    const size_t split_dim,

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    const size_t num_stages,
    std::vector<size_t> num_workers_per_stage,
    const size_t slices_per_stage,
    const tt::DataFormat data_format,
    const size_t page_size_bytes,
    const size_t cb_packet_size_in_pages,
    const size_t num_packets_per_cb,
    auto layout,

    std::vector<std::vector<size_t>> worker_chunk_read_order,
    std::vector<MemoryConfig> mem_configs) {
    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info("This test can only be run on T3000 devices");
        return true;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return true;
    }

    const auto cb_index = tt::CB::c_in0;

    auto programs = std::vector<Program>(1);
    Program& program = programs[0];

    Fabric1DFixture test_fixture;
    auto view = *(test_fixture.view_);

    IDevice* device = view.get_device(MeshCoordinate(0, 0));

    // General setup is as follows:
    // Worker 1 reads input tensor as a sequence of slices - it forwards to an output tensor and after each slice, it
    // writes a semaphore increment to some known semaphore address on the destination worker so the destination worker
    // knows it's safe to read that slice.
    // HOWEVER. the reader will be programmed to read the chunks in a different order than they were written, this way
    // we can identify synchronization related bugs (e.g. if sender semaphore increments before writes flush)

    TT_FATAL(num_workers_per_stage.size() == num_stages, "Must have a read order for each stage");
    TT_FATAL(worker_chunk_read_order.size() == num_stages, "Must have a read order for each stage");
    for (size_t i = 0; i < num_stages; ++i) {
        TT_FATAL(worker_chunk_read_order[i].size() == slices_per_stage, "Must have a read order for each slice");
    }

    // Validate the test setup
    TT_FATAL(num_stages > 1, "Must have at least 2 stages");
    TT_FATAL(num_stages < 8, "Must have at most 8 stages");
    for (size_t i = 0; i < num_stages; ++i) {
        TT_FATAL(num_workers_per_stage[i] > 0, "Must have at least 1 worker per stage");
        TT_FATAL(num_workers_per_stage[i] < 8, "Must have at most 8 workers per stage");
    }

    std::vector<TensorSpec> tensor_specs;
    tensor_specs.reserve(num_stages + 1);
    for (size_t i = 0; i < num_stages + 1; ++i) {
        tensor_specs.push_back(TensorSpec(
            tensor_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), mem_configs[i])));
    }

    // Allocate the tensors - pull to function
    const size_t num_tensors = num_stages + 1;
    std::vector<Tensor> host_tensors;
    std::vector<Tensor> device_tensors;
    host_tensors.reserve(num_tensors);
    device_tensors.reserve(num_tensors);
    auto num_elems = std::reduce(tensor_shape.cbegin(), tensor_shape.cend(), 1, std::multiplies<uint32_t>());
    host_tensors.push_back(
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::UINT32), tensor_shape).to_layout(layout));
    for (size_t i = 1; i < num_tensors; ++i) {
        host_tensors.push_back(
            ttnn::experimental::view(ttnn::ones(tensor_shape, DataType::UINT32, layout), tensor_shape));
    }
    TT_FATAL(mem_configs.size() == num_tensors, "Must have a memory config for each tensor");
    for (size_t i = 0; i < num_tensors; i++) {
        device_tensors.push_back(host_tensors[i].to_device(device, mem_configs[i]));
        log_info("Tensor[{}] allocated starting at address {}", i, device_tensors[i].buffer()->address());
    }
    TT_ASSERT(device_tensors.size() == num_tensors);
    TT_ASSERT(device_tensors.size() == host_tensors.size());

    // MAIN STUFF

    // Initial setup like worker core assignment, chunk read order, etc.

    std::vector<CoreRangeSet> pipeline_stage_worker_cores = {};
    for (size_t i = 0; i < num_stages; ++i) {
        pipeline_stage_worker_cores.push_back(
            CoreRangeSet(CoreRange(CoreCoord(0, i), CoreCoord(num_workers_per_stage[i] - 1, i))));
    }
    CoreRangeSet all_workers_cores = CoreRangeSet();
    for (size_t i = 0; i < num_stages; ++i) {
    }

    // Create circular buffers
    for (size_t stage = 0; stage < num_stages; stage++) {
        const size_t cb_packet_size_in_pages = 4;
        const size_t num_packets_per_cb = 4;
        tt_metal::CircularBufferConfig cb_config =
            tt_metal::CircularBufferConfig(
                cb_packet_size_in_pages * num_packets_per_cb * page_size_bytes, {{cb_index, data_format}})
                .set_page_size(cb_index, page_size_bytes);
        CBHandle sender_workers_cb = CreateCircularBuffer(program, pipeline_stage_worker_cores[stage], cb_config);
    }

    // Generate the reader semaphores
    std::vector<std::vector<uint32_t>> input_tensor_semaphores;
    input_tensor_semaphores.reserve(num_stages);
    for (size_t stage = 0; stage < num_stages; stage++) {
        input_tensor_semaphores.push_back({});
        for (size_t j = 0; j < slices_per_stage; j++) {
            input_tensor_semaphores[stage].push_back(CreateSemaphore(program, pipeline_stage_worker_cores[stage], 0));
        }
    }

    constexpr size_t num_command_streams = 1;
    std::vector<KernelHandle> reader_kernels;
    std::vector<KernelHandle> writer_kernels;
    // Create the kernel handles for each pipeline stage
    for (size_t stage = 0; stage < num_stages; stage++) {
        auto reader_kernel = ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {tt::CB::c_in0},
            {&device_tensors[stage]},
            pipeline_stage_worker_cores[stage],
            tt_metal::ReaderDataMovementConfig{},
            num_command_streams);
        reader_kernels.push_back(reader_kernel);
        auto writer_kernel = ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {tt::CB::c_in0},
            {&device_tensors[stage + 1]},
            pipeline_stage_worker_cores[stage],
            tt_metal::WriterDataMovementConfig{},
            num_command_streams);
        writer_kernels.push_back(writer_kernel);
    }

    // Generate the tensor slices for each tensor/worker
    std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> tensor_slices;
    tensor_slices.reserve(num_stages + 1);
    for (size_t t = 0; t < num_tensors; t++) {
        tensor_slices.push_back(
            ttnn::ccl::cmd::builder::generate_tensor_slices(slices_per_stage, device_tensors[t], split_dim));
    }
    std::vector<std::vector<std::vector<ttnn::ccl::v2::TensorSlice>>> per_stage_worker_reader_tensor_slices;
    std::vector<std::vector<std::vector<ttnn::ccl::v2::TensorSlice>>> per_stage_worker_writer_tensor_slices;
    per_stage_worker_reader_tensor_slices.reserve(num_tensors);
    per_stage_worker_writer_tensor_slices.reserve(num_tensors);
    for (size_t stage = 0; stage < num_stages; stage++) {
        per_stage_worker_reader_tensor_slices.push_back(
            ttnn::ccl::cmd::builder::split_tensor_slices_across_workers_page_aligned(
                num_workers_per_stage[stage], tensor_slices[stage]));
        // We could compute this once and reuse it but I am generating it twice so I can have size mismatches
        per_stage_worker_writer_tensor_slices.push_back(
            ttnn::ccl::cmd::builder::split_tensor_slices_across_workers_page_aligned(
                num_workers_per_stage[stage], tensor_slices[stage + 1]));
        TT_FATAL(
            per_stage_worker_reader_tensor_slices.back().size() == num_workers_per_stage[stage],
            "Mismatch in tensor slices. Got {} but expected {}",
            per_stage_worker_reader_tensor_slices.back().size(),
            num_workers_per_stage[stage]);
        TT_FATAL(
            per_stage_worker_writer_tensor_slices.back().size() == num_workers_per_stage[stage],
            "Mismatch in tensor slices. Got {} but expected {}",
            per_stage_worker_writer_tensor_slices.back().size(),
            num_workers_per_stage[stage]);
    }

    // Build the command stream for each stage/worker
    // Seminc example
    // - local_core_semaphore_inc(second_command_stream_done_semaphore_id, 1);
    // semwait example
    // - local_semaphore_wait(second_command_stream_done_semaphore_id, 1)
    // read tensor slice to cb example
    // - read_tensor_slice_to_cb(in0_command_tensor_slice, cb_indices.at(0))
    // write tensor slice to cb example
    // - build_write_tensor_slice_to_cb(out0_command_tensor_slice, cb_indices.at(0))
    TT_FATAL(per_stage_worker_reader_tensor_slices.size() == num_stages, "Mismatch in tensor slices");
    for (size_t stage = 0; stage < num_stages; stage++) {
        bool last_stage = stage == num_stages - 1;
        bool first_stage = stage == 0;

        const auto worker_cores = corerange_to_cores(pipeline_stage_worker_cores[stage]);
        TT_FATAL(worker_cores.size() == num_workers_per_stage[stage], "Mismatch in worker cores");
        std::optional<std::vector<CoreCoord>> next_worker_cores =
            !last_stage ? corerange_to_cores(pipeline_stage_worker_cores[stage + 1])
                        : std::optional<std::vector<CoreCoord>>(std::nullopt);

        TT_FATAL(
            per_stage_worker_reader_tensor_slices[stage].size() == num_workers_per_stage[stage],
            "Mismatch in tensor slices");
        TT_FATAL(
            per_stage_worker_writer_tensor_slices[stage].size() == num_workers_per_stage[stage],
            "Mismatch in tensor slices");
        for (size_t worker = 0; worker < num_workers_per_stage[stage]; worker++) {
            std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> reader_cmd_stream;
            std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> writer_cmd_stream;
            TT_FATAL(
                per_stage_worker_reader_tensor_slices[stage][worker].size() == slices_per_stage,
                "Mismatch in tensor slices");
            TT_FATAL(
                per_stage_worker_writer_tensor_slices[stage][worker].size() == slices_per_stage,
                "Mismatch in tensor slices");
            for (size_t slice_logical = 0; slice_logical < slices_per_stage; slice_logical++) {
                const auto slice_actual = worker_chunk_read_order[stage][slice_logical];
                // reader
                if (!first_stage) {
                    reader_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_semaphore_wait(
                        input_tensor_semaphores[stage][slice_actual], num_workers_per_stage[stage - 1]));
                }
                reader_cmd_stream.push_back(ttnn::ccl::cmd::uops::read_tensor_slice_to_cb(
                    per_stage_worker_reader_tensor_slices[stage][worker][slice_actual], cb_index));
                log_info(tt::LogTest, "Worker {} reading/writing slice {}", worker, slice_actual);

                // writer
                writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_write_cb_to_tensor_slice(
                    per_stage_worker_writer_tensor_slices[stage][worker][slice_actual], cb_index));
                if (not last_stage) {
                    for (auto next_worker_xy : next_worker_cores.value()) {
                        log_info(
                            tt::LogTest,
                            "Stage {} Worker {} noc seminc to core (logical) x={},y={}",
                            stage,
                            worker,
                            next_worker_xy.x,
                            next_worker_xy.y);
                        writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_chip_noc_semaphore_inc(
                            device->worker_core_from_logical_core(next_worker_xy).x,
                            device->worker_core_from_logical_core(next_worker_xy).y,
                            input_tensor_semaphores[stage + 1][slice_actual],
                            1));
                    }
                }
            }
            ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
                program,
                reader_kernels[stage],
                {&device_tensors[stage]},
                {page_size_bytes},
                device,
                0,  // link = 0, don't care, since we aren't specifying connections
                cb_packet_size_in_pages,
                {worker_cores.at(worker)},
                reader_cmd_stream,
                std::nullopt,
                std::nullopt,
                std::nullopt);
            ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
                program,
                writer_kernels[stage],
                {&device_tensors[stage + 1]},
                {page_size_bytes},
                device,
                0,  // link = 0, don't care, since we aren't specifying connections
                cb_packet_size_in_pages,
                {worker_cores.at(worker)},
                writer_cmd_stream,
                std::nullopt,
                std::nullopt,
                std::nullopt);
        }
    }

    run_programs(programs, {device});

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        log_info(tt::LogTest, "Reading back outputs");
        auto input_cpu = device_tensors[0].cpu();
        auto final_out_cpu = device_tensors.back().cpu();

        auto in_tensor_copyback = tt::tt_metal::host_buffer::get_as<uint32_t>(input_cpu);
        auto out_tensor_copyback = tt::tt_metal::host_buffer::get_as<uint32_t>(final_out_cpu);

        auto in_tensor_data = tt::tt_metal::host_buffer::get_as<uint32_t>(host_tensors[0]);

        bool input_copyback_check_passed = run_output_check(in_tensor_data, in_tensor_copyback) == Correctness::Correct;
        TT_FATAL(input_copyback_check_passed, "Input 0 copyback check failed");

        log_info(tt::LogTest, "Comparing outputs");

        pass &= run_output_check(in_tensor_data, out_tensor_copyback) == Correctness::Correct;
        if (pass) {
            log_info(tt::LogTest, "Output check passed for output 0");
        } else {
            log_error(tt::LogTest, "Output check failed for output 0");
        }
    }

    return pass;
}

#include "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include <tt-metalium/bfloat16.hpp>

static void wait_for_worker_program_completion(
    const std::vector<IDevice*>& devices, const std::optional<SubdeviceInfo>& subdevice_managers = std::nullopt) {
    if (subdevice_managers) {
        std::ranges::for_each(devices, [&](IDevice* d) {
            tt_metal::Finish(d->command_queue(), {subdevice_managers->worker_subdevice_id.at(d->id())});
        });
    } else {
        std::ranges::for_each(devices, [&](IDevice* d) { tt_metal::Finish(d->command_queue(), {}); });
    }
}

#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
void run_all_gather_with_persistent_fabric(const size_t dim, const size_t num_links, ttnn::Shape const& input_shape) {
    log_info(tt::LogTest, "entering test");
    constexpr auto layout = Layout::TILE;
    // DEVICES setuip
    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    constexpr size_t test_expected_num_devices = 4;
    if (tt::tt_metal::GetNumAvailableDevices() < test_expected_num_devices) {
        log_info("This test can only be run on T3000 devices");
        return;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return;
    }
    // Initialize MeshDevice with 1D Fabric
    MeshFabric1DFixture test_fixture(tt::tt_metal::FabricConfig::FABRIC_1D);
    auto view = test_fixture.mesh_device_->get_view();

    // build a line of devices
    std::vector<IDevice*> devices = {
        view.get_device(MeshCoordinate(0, 0)),
        view.get_device(MeshCoordinate(0, 1)),
        view.get_device(MeshCoordinate(0, 2)),
        view.get_device(MeshCoordinate(0, 3))};
    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    // INPUT TENSOR setup
    log_info(tt::LogTest, "setting up input tensors");
    size_t page_size = tile_size(DataFormat::Float16);
    std::vector<Tensor> device_input_tensors;
    for (size_t i = 0; i < num_devices; i++) {
        auto t = ttnn::experimental::view(ttnn::arange(0, num_elems, 1), input_shape).to_layout(layout);

        device_input_tensors.push_back(t);
    }
    // Need to make it a mesh tensor for use with the op
    const Tensor input_mesh_tensor = ttnn::distributed::aggregate_as_tensor(device_input_tensors, AllGatherTensor{})
                                         .to_device(test_fixture.mesh_device_.get());
    std::optional<SubdeviceInfo> subdevice_managers = create_worker_subdevices(devices);

    log_info(tt::LogTest, "launching op");

    GlobalSemaphore multi_device_global_semaphore = ttnn::global_semaphore::create_global_semaphore(
        test_fixture.mesh_device_.get(),
        test_fixture.mesh_device_.get()->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                            // initial value
        tt::tt_metal::BufferType::L1  // buffer type
    );

    auto output_tensor = ttnn::operations::experimental::ccl::all_gather_async(
        input_mesh_tensor,
        dim,
        multi_device_global_semaphore,
        num_links,
        operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        ttnn::ccl::Topology::Linear,
        SubDeviceId(0));

    // wait for op completion
    wait_for_worker_program_completion(devices, subdevice_managers);
    log_info(tt::LogTest, "Finished");
}

void run_ring_all_gather_with_persistent_fabric(
    const size_t dim, const size_t num_links, const ttnn::Shape& input_shape) {
    log_info(tt::LogTest, "entering test");
    constexpr auto layout = Layout::TILE;
    // DEVICES setuip
    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    constexpr size_t test_expected_num_devices = 8;
    if (tt::tt_metal::GetNumAvailableDevices() < test_expected_num_devices) {
        log_info("This test can only be run on T3000 devices");
        return;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return;
    }
    // Initialize MeshDevice with 1D Fabric
    MeshFabric1DFixture test_fixture(tt::tt_metal::FabricConfig::FABRIC_1D_RING);
    test_fixture.mesh_device_->reshape(MeshShape(1, 8));
    auto view = test_fixture.mesh_device_->get_view();

    // build a line of devices
    std::vector<IDevice*> devices = test_fixture.mesh_device_->get_devices();
    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    // INPUT TENSOR setup
    log_info(tt::LogTest, "setting up input tensors");
    size_t page_size = tile_size(DataFormat::Float16);
    std::vector<Tensor> device_input_tensors;
    for (size_t i = 0; i < num_devices; i++) {
        auto t = ttnn::experimental::view(ttnn::arange(0, num_elems, 1), input_shape).to_layout(layout);

        device_input_tensors.push_back(t);
    }
    // Need to make it a mesh tensor for use with the op
    const Tensor input_mesh_tensor = ttnn::distributed::aggregate_as_tensor(device_input_tensors, AllGatherTensor{})
                                         .to_device(test_fixture.mesh_device_.get());

    std::optional<SubdeviceInfo> subdevice_managers = create_worker_subdevices(devices);
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring;

    log_info(tt::LogTest, "launching op");

    GlobalSemaphore multi_device_global_semaphore = ttnn::global_semaphore::create_global_semaphore(
        test_fixture.mesh_device_.get(),
        devices[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                            // initial value
        tt::tt_metal::BufferType::L1  // buffer type
    );

    auto output_tensor = ttnn::operations::experimental::ccl::all_gather_async(
        input_mesh_tensor,
        dim,
        multi_device_global_semaphore,
        num_links,
        operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        topology,
        SubDeviceId(0));

    // wait for op completion
    wait_for_worker_program_completion(devices, subdevice_managers);
}

enum class FabricTestMode {
    Linear,
    HalfRing,
    FullRing,
    SaturateChipToChipRing,
    RingAsLinear,
};

struct WriteThroughputStabilityTestWithPersistentFabricParams {
    size_t line_size = 4;
    size_t num_devices_with_workers = 0;
    size_t num_links = 0;
    size_t first_link_offset = 0;
    size_t num_op_invocations = 0;
    bool line_sync = true;
    size_t num_fabric_rows = 0;
    size_t num_fabric_cols = 0;
    FabricTestMode fabric_mode = FabricTestMode::Linear;

    // True if you only want the workers on the end to send
    bool disable_sends_for_interior_workers = false;

    bool disable_end_workers_in_backward_direction = false;
    bool senders_are_unidirectional = false;
};

std::vector<CoreCoord> compute_top_row_ethernet_cores(
    IDevice* device,
    bool has_fwd_connection,
    bool has_bwd_connection,
    IDevice* forward_device,
    IDevice* backward_device) {
    std::vector<CoreCoord> reordered_ethernet_cores;
    if (has_fwd_connection) {
        for (auto core : device->get_ethernet_sockets(forward_device->id())) {
            auto core_virtual = device->virtual_core_from_logical_core(core, CoreType::ETH);
            reordered_ethernet_cores.push_back(core_virtual);
        }
        std::sort(reordered_ethernet_cores.begin(), reordered_ethernet_cores.end(), [](auto& a, auto& b) {
            return a.x < b.x;
        });
    } else if (has_bwd_connection) {
        for (auto core : device->get_ethernet_sockets(backward_device->id())) {
            auto core_virtual = device->virtual_core_from_logical_core(core, CoreType::ETH);
            reordered_ethernet_cores.push_back(core_virtual);
        }
        std::sort(reordered_ethernet_cores.begin(), reordered_ethernet_cores.end(), [](auto& a, auto& b) {
            return a.x < b.x;
        });
    }
    for (auto& eth_core : reordered_ethernet_cores) {
        eth_core.y = 16;
    }
    return reordered_ethernet_cores;
}

CoreCoord wh_glx_physical_worker_core_from_logical_core(CoreCoord logical_core) {
    auto physical_x = logical_core.x <= 3 ? logical_core.x + 1 : logical_core.x + 2;
    auto physical_y = logical_core.y <= 4 ? logical_core.y + 1 : logical_core.y + 2;
    return CoreCoord(physical_x, physical_y);
}

CoreRangeSet get_optimal_worker_core_placement(
    IDevice* device, std::vector<CoreCoord> ethernet_cores_virtual, uint32_t num_links, size_t num_skipped_links) {
    std::vector<CoreCoord> sender_worker_cores;
    std::vector<CoreCoord> sender_worker_cores_physical;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Get all logical cores in the worker grid
    std::vector<CoreCoord> compute_cores_logical;
    for (int i = 0; i < num_cores_x; ++i) {
        for (int j = 0; j < num_cores_y; ++j) {
            compute_cores_logical.push_back(CoreCoord(i, j));
        }
    }

    for (uint32_t link = num_skipped_links; link < num_links; link++) {
        auto core_virtual = ethernet_cores_virtual[link];
        CoreCoord eth_core_physical;
        eth_core_physical.x = core_virtual.x >= 22 ? (core_virtual.x - 16) : (core_virtual.x - 17);
        eth_core_physical.y = (core_virtual.y - 16) * 6;
        // shift down the worker core
        auto worker_core_physical = CoreCoord(eth_core_physical.x, eth_core_physical.y + 1);
        sender_worker_cores_physical.push_back(worker_core_physical);
    }

    // Convert to physical worker coordinates to logical.
    for (int i = 0; i < sender_worker_cores_physical.size(); ++i) {
        for (int j = 0; j < compute_cores_logical.size(); ++j) {
            auto core = wh_glx_physical_worker_core_from_logical_core(compute_cores_logical[j]);
            if (sender_worker_cores_physical[i] == core) {
                sender_worker_cores.push_back(compute_cores_logical[j]);
            }
        }
    }

    std::set<CoreRange> sender_worker_cores_set;
    for (int i = 0; i < sender_worker_cores.size(); ++i) {
        sender_worker_cores_set.insert(CoreRange(sender_worker_cores[i]));
    }
    CoreRangeSet sender_worker_corerangeset = CoreRangeSet(sender_worker_cores_set);

    return sender_worker_corerangeset;
}

struct Fabric1DPacketSendTestSpec {
    tt::tt_fabric::ChipSendType chip_send_type = tt::tt_fabric::CHIP_UNICAST;
    tt::tt_fabric::NocSendType noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
    size_t num_messages = 0;
    size_t packet_payload_size_bytes = 0;
    bool flush = true;
};

static std::vector<IDevice*> generate_default_line_fabric_under_test(
    bool use_galaxy, bool use_tg, size_t line_size, ttnn::ccl::Topology topology, const MeshDeviceView& view) {
    std::vector<IDevice*> devices_;
    if (use_galaxy) {
        if (line_size <= 4) {
            if (use_tg) {
                if (topology == ttnn::ccl::Topology::Ring) {
                    devices_ = {
                        view.get_device(MeshCoordinate(0, 0)),
                        view.get_device(MeshCoordinate(1, 0)),
                        view.get_device(MeshCoordinate(1, 1)),
                        view.get_device(MeshCoordinate(0, 1))};
                } else {
                    devices_ = {
                        view.get_device(MeshCoordinate(0, 0)),
                        view.get_device(MeshCoordinate(1, 0)),
                        view.get_device(MeshCoordinate(2, 0)),
                        view.get_device(MeshCoordinate(3, 0))};
                }
            } else {
                devices_ = {
                    view.get_device(MeshCoordinate(0, 0)),
                    view.get_device(MeshCoordinate(0, 1)),
                    view.get_device(MeshCoordinate(0, 2)),
                    view.get_device(MeshCoordinate(0, 3))};
            }
        } else {
            if (topology == ttnn::ccl::Topology::Ring && use_tg) {
                devices_ = {
                    view.get_device(MeshCoordinate(0, 0)),
                    view.get_device(MeshCoordinate(1, 0)),
                    view.get_device(MeshCoordinate(2, 0)),
                    view.get_device(MeshCoordinate(3, 0)),
                    view.get_device(MeshCoordinate(3, 1)),
                    view.get_device(MeshCoordinate(2, 1)),
                    view.get_device(MeshCoordinate(1, 1)),
                    view.get_device(MeshCoordinate(0, 1))};
            } else {
                devices_ = {
                    view.get_device(MeshCoordinate(0, 0)),
                    view.get_device(MeshCoordinate(1, 0)),
                    view.get_device(MeshCoordinate(2, 0)),
                    view.get_device(MeshCoordinate(3, 0)),
                    view.get_device(MeshCoordinate(4, 0)),
                    view.get_device(MeshCoordinate(5, 0)),
                    view.get_device(MeshCoordinate(6, 0)),
                    view.get_device(MeshCoordinate(7, 0))};
            }
        }
    } else {
        // Choosing pcie devices so that more links are supported. More links == more (likelihood of) congestion.
        if (line_size <= 4) {
            devices_ = {
                view.get_device(MeshCoordinate(0, 1)),
                view.get_device(MeshCoordinate(0, 2)),
                view.get_device(MeshCoordinate(1, 2)),
                view.get_device(MeshCoordinate(1, 1))};
        } else {
            devices_ = {
                view.get_device(MeshCoordinate(0, 0)),
                view.get_device(MeshCoordinate(0, 1)),
                view.get_device(MeshCoordinate(0, 2)),
                view.get_device(MeshCoordinate(0, 3)),
                view.get_device(MeshCoordinate(1, 3)),
                view.get_device(MeshCoordinate(1, 2)),
                view.get_device(MeshCoordinate(1, 1)),
                view.get_device(MeshCoordinate(1, 0))};
        }
    }

    return devices_;
}

static std::vector<std::vector<IDevice*>> generate_line_fabrics_under_test(
    const WriteThroughputStabilityTestWithPersistentFabricParams& params,
    bool use_galaxy,
    bool use_tg,
    size_t line_size,
    ttnn::ccl::Topology topology,
    const MeshDeviceView& view) {
    bool use_default_device_selection = params.num_fabric_rows == 0 && params.num_fabric_cols == 0;
    std::vector<std::vector<IDevice*>> fabrics_under_test;
    if (use_default_device_selection) {
        fabrics_under_test.push_back(
            generate_default_line_fabric_under_test(use_galaxy, use_tg, line_size, topology, view));
    } else {
        fabrics_under_test.reserve(params.num_fabric_rows + params.num_fabric_cols);
        TT_FATAL(
            params.num_fabric_rows <= view.num_rows(),
            "num_rows_requested must be less than or equal to the number of rows in the mesh");
        TT_FATAL(
            params.num_fabric_cols <= view.num_cols(),
            "num_cols_requested must be less than or equal to the number of cols in the mesh");
        for (size_t i = 0; i < params.num_fabric_rows; i++) {
            fabrics_under_test.push_back(view.get_devices_on_row(i));
        }
        for (size_t i = 0; i < params.num_fabric_cols; i++) {
            fabrics_under_test.push_back(view.get_devices_on_column(i));
        }
    }

    return fabrics_under_test;
}

template <typename FABRIC_DEVICE_FIXTURE = Fabric1DFixture>
void Run1DFabricPacketSendTest(
    const std::vector<Fabric1DPacketSendTestSpec>& test_specs,
    const WriteThroughputStabilityTestWithPersistentFabricParams& params = {},
    size_t fabric_context_switch_interval =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_firmware_context_switch_interval) {
    constexpr bool use_device_init_fabric = std::is_same_v<FABRIC_DEVICE_FIXTURE, Fabric1DLineDeviceInitFixture> ||
                                            std::is_same_v<FABRIC_DEVICE_FIXTURE, Fabric1DRingDeviceInitFixture>;
    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    TT_FATAL(
        !params.disable_sends_for_interior_workers || params.fabric_mode == FabricTestMode::Linear ||
            params.fabric_mode == FabricTestMode::RingAsLinear,
        "This test can only be run with disable_sends_for_interior_workers set to true or fabric_mode set to Linear");
    TT_FATAL(
        !params.disable_end_workers_in_backward_direction || params.fabric_mode == FabricTestMode::Linear ||
            params.fabric_mode == FabricTestMode::RingAsLinear,
        "This test can only be run with disable_end_workers_in_backward_direction set to true or fabric_mode set to "
        "Linear");
    bool use_galaxy = num_devices == 32;
    bool use_tg = use_galaxy && tt::tt_metal::GetNumPCIeDevices() == 4;
    bool is_6u_galaxy = use_galaxy && tt::tt_metal::GetNumPCIeDevices() == 32;
    if (num_devices < 4) {
        log_info("This test can only be run on T3000 devices");
        return;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return;
    }

    size_t line_size = params.line_size;
    size_t num_devices_with_workers = params.num_devices_with_workers;
    if (num_devices_with_workers == 0) {
        num_devices_with_workers = line_size;
    }
    using namespace ttnn::ccl;
    TT_FATAL(num_devices_with_workers <= line_size, "num_devices_with_workers must be less than or equal to line_size");
    TT_FATAL(
        !(params.num_fabric_rows > 0 && params.num_fabric_cols > 0),
        "Only one of num_fabric_rows and num_fabric_cols may be greater than 0. Test support for both axes live at the "
        "same time is not yet supported");
    TT_FATAL(
        use_device_init_fabric ^ (params.num_fabric_rows == 0 && params.num_fabric_cols == 0),
        "Device init fabric is only supported in this test when launching with multiple fabric rows and/or columns");

    ttnn::ccl::Topology topology;
    FabricTestMode fabric_mode = params.fabric_mode;
    switch (fabric_mode) {
        case FabricTestMode::Linear: topology = ttnn::ccl::Topology::Linear; break;
        case FabricTestMode::SaturateChipToChipRing:
            TT_FATAL(line_size == 4, "SaturateChipToChipRing only supports line_size 4");
        case FabricTestMode::HalfRing:
        case FabricTestMode::FullRing:
        case FabricTestMode::RingAsLinear: topology = ttnn::ccl::Topology::Ring; break;
    }

    auto worker_core_logical = [](size_t link) { return CoreCoord(link, 0); };

    // static constexpr size_t source_l1_buffer_address = 1000000;
    static constexpr uint32_t packet_header_cb_index = tt::CB::c_in0;
    static constexpr uint32_t source_payload_cb_index = tt::CB::c_in1;
    static constexpr size_t packet_header_cb_size_in_headers = 5;
    static constexpr bool enable_persistent_fabric_mode = true;
    auto max_packet_payload_size_bytes =
        std::max_element(test_specs.begin(), test_specs.end(), [](const auto& a, const auto& b) {
            return a.packet_payload_size_bytes < b.packet_payload_size_bytes;
        })->packet_payload_size_bytes;
    size_t dest_buffer_size = max_packet_payload_size_bytes * 4;
    static constexpr tt::DataFormat cb_df = tt::DataFormat::Bfp8;

    log_info("Device open and fabric init");
    // MeshFabric1DLineDeviceInitFixture test_fixture;
    FABRIC_DEVICE_FIXTURE test_fixture;
    log_info("\tDone");
    auto view = *(test_fixture.view_);

    auto fabrics_under_test_devices =
        generate_line_fabrics_under_test(params, use_galaxy, use_tg, line_size, topology, view);

    // Persistent Fabric Setup
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle = std::nullopt;
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs = std::nullopt;
    size_t packet_header_size_bytes = 0;
    if (!use_device_init_fabric) {
        std::vector<Program> dummy_worker_programs;
        std::vector<Program*> fabric_program_ptrs;
        TT_FATAL(
            fabrics_under_test_devices.size() == 1, "Expected 1 fabric under test when device init fabric is not used");
        setup_test_with_persistent_fabric(
            fabrics_under_test_devices[0],
            dummy_worker_programs,
            subdevice_managers,
            fabric_programs,
            fabric_program_ptrs,
            fabric_handle,
            enable_persistent_fabric_mode,
            params.num_links,
            topology,
            fabric_context_switch_interval,
            false,
            is_6u_galaxy);
        packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    } else {
        // TODO: get packet header size from control plane after it adds APIs to present this information
        packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    }
    TT_FATAL(packet_header_size_bytes != 0, "Error in initializing local variable `packet_header_size_bytes`");

    // Other boiler plate setup
    std::vector<std::vector<CoreCoord>> worker_cores_vec_per_device;
    std::vector<CoreCoord> dest_core_coord;
    dest_core_coord.reserve(params.num_links);
    for (size_t l = 0; l < params.num_links; l++) {
        dest_core_coord[l] = CoreCoord(0, l + 1);
    }
    auto sync_core_coord = CoreCoord(0, 0);

    ttnn::SmallVector<std::shared_ptr<Buffer>> device_dest_buffers;
    std::vector<IDevice*> devices_ = {};
    device_dest_buffers.reserve(line_size);
    for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices.size(); fabric_index++) {
        auto& devices = fabrics_under_test_devices[fabric_index];
        for (auto* d : devices) {
            if (std::find(devices_.begin(), devices_.end(), d) == devices_.end()) {
                devices_.push_back(d);
            }
            auto local_input_buffer =
                CreateBuffer(InterleavedBufferConfig{d, dest_buffer_size, dest_buffer_size, BufferType::L1});
            device_dest_buffers.push_back(local_input_buffer);
        }
    }

    size_t dest_bank_addr = device_dest_buffers[0]->address();
    TT_FATAL(
        std::all_of(
            device_dest_buffers.begin(),
            device_dest_buffers.end(),
            [dest_bank_addr](const auto& buffer) { return buffer->address() == dest_bank_addr; }),
        "Test setup error: all destination buffers must have the same bank address across devices");

    std::vector<tt::tt_metal::DeviceAddr> global_semaphore_addrs;
    global_semaphore_addrs.reserve(line_size + 1);
    std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> global_semaphore_handles;
    for (size_t i = 0; i < line_size * 4; i++) {
        auto global_semaphores = ttnn::global_semaphore::create_global_semaphore_with_same_address(
            devices_,
            fabrics_under_test_devices[0][0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
            0,                             // initial value
            tt::tt_metal::BufferType::L1,  // buffer type,
            1000                           // attempts
        );
        global_semaphore_handles.push_back(global_semaphores);
        auto global_semaphore_addr =
            ttnn::global_semaphore::get_global_semaphore_address(global_semaphores.global_semaphores.at(0));
        global_semaphore_addrs.push_back(global_semaphore_addr);
    }

    std::vector<std::vector<IDevice*>> fabric_under_test_worker_devices(fabrics_under_test_devices.size());
    std::vector<std::vector<Program>> programs_per_fabric(fabrics_under_test_devices.size());
    for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices.size(); fabric_index++) {
        auto& devices = fabrics_under_test_devices[fabric_index];
        auto& worker_devices = fabric_under_test_worker_devices[fabric_index];
        for (size_t i = 0; i < num_devices_with_workers; i++) {
            worker_devices.push_back(devices[i]);
        }
        // Worker program setup
        programs_per_fabric[fabric_index] = std::vector<Program>(num_devices_with_workers);
        TT_FATAL(
            programs_per_fabric[fabric_index].size() == worker_devices.size(),
            "Test misconfiguration. Mismatch in line size and devices. Expected line size of {} but got {} devices "
            "instead.",
            line_size,
            worker_devices.size());
    }
    std::vector<std::vector<KernelHandle>> worker_kernel_ids_per_fabric(fabrics_under_test_devices.size());
    std::vector<std::vector<size_t>> per_fabric_per_device_global_sem_addr_rt_arg(fabrics_under_test_devices.size());
    for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices.size(); fabric_index++) {
        auto& devices = fabrics_under_test_devices[fabric_index];
        auto& programs = programs_per_fabric[fabric_index];
        auto& per_device_global_sem_addr_rt_arg = per_fabric_per_device_global_sem_addr_rt_arg[fabric_index];
        auto& worker_kernel_ids = worker_kernel_ids_per_fabric[fabric_index];
        for (size_t i = 0; i < num_devices_with_workers; i++) {
            const size_t line_index = i;
            auto& program = programs[i];
            auto* device = devices[i];

            IDevice* backward_device;
            IDevice* forward_device;
            bool has_forward_connection;
            bool has_backward_connection;
            bool unicast_forward;
            size_t num_fwd_hops;
            size_t num_bwd_hops;
            size_t sync_num_fwd_hops;
            size_t sync_num_bwd_hops;
            size_t sync_count_per_link;
            if (topology == ttnn::ccl::Topology::Ring && fabric_mode != FabricTestMode::RingAsLinear) {
                backward_device = i == 0 ? devices.back() : devices[i - 1];
                forward_device = i == line_size - 1 ? devices.front() : devices[i + 1];

                // Initialize the fabric handle for worker connection
                has_forward_connection = true;
                has_backward_connection = true;
                unicast_forward = true;
                // Have the sync for ring always use the same algorithm as HalfRing
                sync_num_fwd_hops = tt::div_up(line_size - 1, 2);
                sync_num_bwd_hops = line_size - 1 - sync_num_fwd_hops;
                if (i % 2 == 0) {
                    std::swap(sync_num_fwd_hops, sync_num_bwd_hops);
                }
                if (fabric_mode == FabricTestMode::HalfRing) {
                    num_fwd_hops = tt::div_up(line_size - 1, 2);
                    num_bwd_hops = line_size - 1 - num_fwd_hops;
                    if (i % 2 == 0) {
                        std::swap(num_fwd_hops, num_bwd_hops);
                    }
                    sync_num_fwd_hops = num_fwd_hops;
                    sync_num_bwd_hops = num_bwd_hops;
                    // We will get 1 inc per remote chip + 1 local
                    sync_count_per_link = num_devices_with_workers;
                } else if (fabric_mode == FabricTestMode::FullRing) {
                    num_fwd_hops = line_size - 1;
                    num_bwd_hops = line_size - 1;
                    sync_num_fwd_hops = num_fwd_hops;
                    sync_num_bwd_hops = num_bwd_hops;
                    // We will get 2 inc per remote chip + 1 local
                    sync_count_per_link = 2 * (num_devices_with_workers - 1) + 1;
                } else if (fabric_mode == FabricTestMode::SaturateChipToChipRing) {
                    // We want to saturate the middle links between chip 1 and 2 in a 4 chip ring with the dateline
                    // between the first and last chip Mcast 2 hops from chip 1 F and chip 2 B, which is S0 -> R0 Mcast
                    // 3 hops from chip 0 F and chip 3 B, which is S1 -> R0 Mcast 4 hops from Chip 3 F and chip 0 B,
                    // which is S2 -> R1
                    if (line_index == line_size - 1) {
                        num_fwd_hops = line_size - 1;
                    } else {
                        num_fwd_hops = line_size - 2 - line_index;
                    }
                    if (line_index == 0) {
                        num_bwd_hops = line_size - 1;
                    } else {
                        num_bwd_hops = line_index - 1;
                    }
                    // The above calculations calculates the number of hops to land on the dest chip
                    // Extend by one so we mcast through them
                    if (num_fwd_hops != 0) {
                        num_fwd_hops++;
                    }
                    if (num_bwd_hops != 0) {
                        num_bwd_hops++;
                    }
                    // Flush all the way around the ring
                    sync_num_fwd_hops = line_size;
                    sync_num_bwd_hops = line_size;
                    // We will get 2 inc for all chips + 1 local
                    sync_count_per_link = 2 * num_devices_with_workers + 1;
                } else {
                    TT_THROW("Invalid fabric mode");
                }
                if (num_fwd_hops >= num_bwd_hops) {
                    unicast_forward = true;
                } else {
                    unicast_forward = false;
                }
            } else {
                backward_device = i == 0 ? nullptr : devices[i - 1];
                forward_device = i == line_size - 1 ? nullptr : devices[i + 1];

                // Initialize the fabric handle for worker connection
                bool start_of_line = line_index == 0;
                bool end_of_line = line_index == line_size - 1;
                has_forward_connection = !end_of_line;
                has_backward_connection = !start_of_line;
                unicast_forward = line_index < (line_size / 2);
                num_fwd_hops = line_size - line_index - 1;
                num_bwd_hops = line_index;
                sync_num_fwd_hops = num_fwd_hops;
                sync_num_bwd_hops = num_bwd_hops;

                // Do this AFTER sync_num_fwd_hops and sync_num_bwd_hops are set
                // otherwise sync hops will be misconfigured - you'll get a hang because
                // setup/teardown will be done incorrectly

                if (params.senders_are_unidirectional) {
                    if (unicast_forward) {
                        num_bwd_hops = 0;
                    } else {
                        num_fwd_hops = 0;
                    }
                }
                // We will get 1 inc per remote chip + 1 local
                sync_count_per_link = num_devices_with_workers;
            }

            // compute worker based on ethernet cores
            CoreRangeSet worker_cores = {};
            if (use_tg and topology == ttnn::ccl::Topology::Linear) {
                std::vector<CoreCoord> ethernet_cores_virtual = compute_top_row_ethernet_cores(
                    device, has_forward_connection, has_backward_connection, forward_device, backward_device);
                worker_cores = get_optimal_worker_core_placement(
                    device, ethernet_cores_virtual, params.num_links, params.first_link_offset);
            } else {
                worker_cores =
                    CoreRangeSet(CoreRange(CoreCoord(params.first_link_offset, 0), CoreCoord(params.num_links - 1, 0)));
            }
            auto worker_cores_vec = corerange_to_cores(worker_cores, std::nullopt, false);
            worker_cores_vec_per_device.push_back(worker_cores_vec);

            // sync core
            const size_t sync_core_noc_x = device->worker_core_from_logical_core(worker_cores_vec[0]).x;
            const size_t sync_core_noc_y = device->worker_core_from_logical_core(worker_cores_vec[0]).y;

            std::optional<ttnn::ccl::EdmLineFabricOpInterface> local_device_fabric_handle = std::nullopt;
            if (!use_device_init_fabric) {
                local_device_fabric_handle =
                    ttnn::ccl::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
                        device,
                        forward_device,
                        backward_device,
                        &program,
                        enable_persistent_fabric_mode,
                        params.num_links,
                        topology);
            }

            // reserve CB
            tt_metal::CircularBufferConfig cb_src0_config =
                tt_metal::CircularBufferConfig(
                    packet_header_cb_size_in_headers * packet_header_size_bytes, {{packet_header_cb_index, cb_df}})
                    .set_page_size(packet_header_cb_index, packet_header_size_bytes);
            CBHandle sender_workers_cb = CreateCircularBuffer(program, worker_cores, cb_src0_config);

            tt_metal::CircularBufferConfig cb_src1_config =
                tt_metal::CircularBufferConfig(max_packet_payload_size_bytes, {{source_payload_cb_index, cb_df}})
                    .set_page_size(source_payload_cb_index, max_packet_payload_size_bytes);
            CBHandle sender_workers_payload_cb = CreateCircularBuffer(program, worker_cores, cb_src1_config);

            std::vector<uint32_t> worker_ct_args = {params.line_sync, params.line_sync};

            auto worker_kernel_id = tt_metal::CreateKernel(
                program,
                "tests/ttnn/unit_tests/gtests/ccl/kernels/edm_fabric_writer.cpp",
                worker_cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::NOC_0,
                    .compile_args = worker_ct_args});
            worker_kernel_ids.push_back(worker_kernel_id);

            auto build_connection_args = [use_device_init_fabric, &local_device_fabric_handle, device, &program](
                                             CoreCoord& worker_core,
                                             size_t link,
                                             bool is_connected_in_direction,
                                             IDevice* connected_device,
                                             ttnn::ccl::EdmLineFabricOpInterface::Direction direction,
                                             std::vector<uint32_t>& rt_args_out) {
                rt_args_out.push_back(is_connected_in_direction);
                if (is_connected_in_direction) {
                    if (use_device_init_fabric) {
                        tt::tt_fabric::append_fabric_connection_rt_args(
                            device->id(), connected_device->id(), link, program, {worker_core}, rt_args_out);
                    } else {
                        const auto connection = local_device_fabric_handle->uniquely_connect_worker(device, direction);
                        const auto new_rt_args = ttnn::ccl::worker_detail::generate_edm_connection_rt_args(
                            connection, program, {worker_core});
                        log_info(
                            tt::LogTest,
                            "On device: {}, connecting to EDM fabric in {} direction. EDM noc_x: {}, noc_y: {}",
                            device->id(),
                            direction,
                            connection.edm_noc_x,
                            connection.edm_noc_y);
                        std::copy(new_rt_args.begin(), new_rt_args.end(), std::back_inserter(rt_args_out));
                    }
                }
            };

            for (size_t l = 0; l < params.num_links; l++) {
                auto worker_core = worker_cores_vec[l];
                const size_t dest_noc_x = device->worker_core_from_logical_core(dest_core_coord[l]).x;
                const size_t dest_noc_y = device->worker_core_from_logical_core(dest_core_coord[l]).y;

                // RT ARGS
                bool disable_sends_for_worker =
                    (params.disable_sends_for_interior_workers && (i != 0) && (i != line_size - 1)) ||
                    (params.disable_end_workers_in_backward_direction && (i == line_size - 1));
                // Get forward and backward destination coordinates
                const size_t dest_noc_x_fwd = device->worker_core_from_logical_core(dest_core_coord[l]).x;
                const size_t dest_noc_y_fwd = device->worker_core_from_logical_core(dest_core_coord[l]).y;
                const size_t dest_noc_x_bwd = device->worker_core_from_logical_core(dest_core_coord[l]).x;
                const size_t dest_noc_y_bwd = device->worker_core_from_logical_core(dest_core_coord[l]).y;

                // New format for send types
                std::vector<size_t> send_types;
                std::vector<size_t> chip_send_types;
                std::vector<size_t> send_counts_per_type;
                std::vector<size_t> num_fwd_hops_per_type;
                std::vector<size_t> num_bwd_hops_per_type;
                std::vector<size_t> send_type_payload_sizes;
                std::vector<bool> flush_send;
                if (!disable_sends_for_worker) {
                    for (const auto& test_spec : test_specs) {
                        send_types.push_back(static_cast<size_t>(test_spec.noc_send_type));
                        chip_send_types.push_back(static_cast<size_t>(test_spec.chip_send_type));
                        send_counts_per_type.push_back(test_spec.num_messages);
                        num_fwd_hops_per_type.push_back(num_fwd_hops);
                        num_bwd_hops_per_type.push_back(num_bwd_hops);
                        send_type_payload_sizes.push_back(test_spec.packet_payload_size_bytes);
                        flush_send.push_back(test_spec.flush);
                    }
                }

                size_t num_send_types = disable_sends_for_worker ? 0 : test_specs.size();
                std::vector<uint32_t> rt_args = {
                    dest_bank_addr,
                    dest_noc_x_fwd,
                    dest_noc_y_fwd,
                    dest_noc_x_bwd,
                    dest_noc_y_bwd,
                    num_send_types,
                };

                // Reserve space for all arrays upfront
                rt_args.reserve(
                    rt_args.size() + num_send_types * 6 +  // 6 arrays of size num_send_types
                    3 +                                    // CB indices
                    (has_forward_connection ? 10 : 1) +    // Forward connection args
                    (has_backward_connection ? 10 : 1) +   // Backward connection args
                    (params.line_sync ? 6 : 0));           // Line sync args

                // Add send types arrays using std::copy
                std::copy(send_types.begin(), send_types.end(), std::back_inserter(rt_args));
                std::copy(chip_send_types.begin(), chip_send_types.end(), std::back_inserter(rt_args));
                std::copy(send_counts_per_type.begin(), send_counts_per_type.end(), std::back_inserter(rt_args));
                std::copy(num_fwd_hops_per_type.begin(), num_fwd_hops_per_type.end(), std::back_inserter(rt_args));
                std::copy(num_bwd_hops_per_type.begin(), num_bwd_hops_per_type.end(), std::back_inserter(rt_args));
                std::copy(send_type_payload_sizes.begin(), send_type_payload_sizes.end(), std::back_inserter(rt_args));
                std::copy(flush_send.begin(), flush_send.end(), std::back_inserter(rt_args));

                rt_args.push_back(source_payload_cb_index);
                rt_args.push_back(packet_header_cb_index);
                rt_args.push_back(packet_header_cb_size_in_headers);

                build_connection_args(
                    worker_core,
                    l,
                    has_forward_connection,
                    forward_device,
                    ttnn::ccl::EdmLineFabricOpInterface::FORWARD,
                    rt_args);
                build_connection_args(
                    worker_core,
                    l,
                    has_backward_connection,
                    backward_device,
                    ttnn::ccl::EdmLineFabricOpInterface::BACKWARD,
                    rt_args);

                if (params.line_sync) {
                    rt_args.push_back(sync_core_noc_x);
                    rt_args.push_back(sync_core_noc_y);
                    if (l == 0) {
                        per_device_global_sem_addr_rt_arg.push_back(rt_args.size());
                    }
                    TT_FATAL(global_semaphore_addrs.at(0) != -1, "Invalid test setup. Global semaphore address is -1");
                    rt_args.push_back(global_semaphore_addrs.at(0));
                    rt_args.push_back(params.num_links * sync_count_per_link);
                    rt_args.push_back(sync_num_fwd_hops);
                    rt_args.push_back(sync_num_bwd_hops);
                }

                tt_metal::SetRuntimeArgs(program, worker_kernel_id, worker_core, rt_args);
            }
        }
    }

    for (size_t i = 0; i < params.num_op_invocations; i++) {
        log_info(tt::LogTest, "Iteration: {}", i);
        if (i != 0 && params.line_sync) {
            for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices.size(); fabric_index++) {
                auto& programs = programs_per_fabric[fabric_index];
                auto& worker_kernel_ids = worker_kernel_ids_per_fabric[fabric_index];
                auto& per_device_global_sem_addr_rt_arg = per_fabric_per_device_global_sem_addr_rt_arg[fabric_index];
                for (size_t k = 0; k < worker_kernel_ids.size(); k++) {
                    auto& worker_rt_args_by_core = GetRuntimeArgs(programs[k], worker_kernel_ids[k]);
                    auto global_sem_addr_rt_arg_idx = per_device_global_sem_addr_rt_arg[k];
                    for (size_t l = 0; l < params.num_links; l++) {
                        auto& worker_rt_args = worker_rt_args_by_core[worker_cores_vec_per_device[k][l].x]
                                                                     [worker_cores_vec_per_device[k][l].y];
                        worker_rt_args.at(global_sem_addr_rt_arg_idx) =
                            global_semaphore_addrs[i % global_semaphore_addrs.size()];
                    }
                }
            }
        }

        for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices.size(); fabric_index++) {
            auto& worker_devices = fabric_under_test_worker_devices[fabric_index];
            auto& programs = programs_per_fabric[fabric_index];
            build_and_enqueue(worker_devices, programs, i != 0);
        }

        log_info(tt::LogTest, "Waiting for Op finish on all devices");
        for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices.size(); fabric_index++) {
            auto& worker_devices = fabric_under_test_worker_devices[fabric_index];
            wait_for_worker_program_completion(worker_devices, subdevice_managers);
        }
        log_info(tt::LogTest, "Main op done");
    }

    if (!use_device_init_fabric) {
        auto& devices = fabrics_under_test_devices[0];
        TT_FATAL(fabric_programs->size() == devices.size(), "Expected fabric programs size to be same as devices size");
        log_info(tt::LogTest, "Fabric teardown");
        persistent_fabric_teardown_sequence(
            devices, subdevice_managers, fabric_handle.value(), tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE);
    }

    log_info(tt::LogTest, "Waiting for teardown completion");
    for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices.size(); fabric_index++) {
        auto& devices = fabrics_under_test_devices[fabric_index];
        for (IDevice* d : devices) {
            tt_metal::Synchronize(d, *ttnn::DefaultQueueId);
        }
    }

    for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices.size(); fabric_index++) {
        auto& devices = fabric_under_test_worker_devices[fabric_index];
        for (IDevice* d : devices) {
            tt_metal::detail::DumpDeviceProfileResults(d);
        }
    }
    log_info(tt::LogTest, "Finished");
}

void RunWriteThroughputStabilityTestWithPersistentFabric(
    size_t num_mcasts,
    size_t num_unicasts,
    size_t num_links,
    size_t num_op_invocations,
    const WriteThroughputStabilityTestWithPersistentFabricParams& params = {},
    size_t packet_payload_size_bytes = tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes) {
    std::vector<Fabric1DPacketSendTestSpec> test_specs;
    if (num_mcasts > 0) {
        test_specs.push_back(
            {.chip_send_type = tt::tt_fabric::ChipSendType::CHIP_MULTICAST,
             .noc_send_type = tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE,
             .num_messages = num_mcasts,
             .packet_payload_size_bytes = packet_payload_size_bytes});
    }
    if (num_unicasts > 0) {
        test_specs.push_back(
            {.chip_send_type = tt::tt_fabric::ChipSendType::CHIP_UNICAST,
             .noc_send_type = tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE,
             .num_messages = num_unicasts,
             .packet_payload_size_bytes = packet_payload_size_bytes});
    }
    auto params_copy = params;
    params_copy.num_links = num_links;
    params_copy.num_op_invocations = num_op_invocations;
    Run1DFabricPacketSendTest(test_specs, params_copy, 0);
}

void RunRingDeadlockStabilityTestWithPersistentFabric(
    size_t num_mcasts,
    size_t num_links,
    size_t num_devices,
    size_t num_op_invocations,
    bool has_forward_connection,
    bool has_backward_connection,
    size_t packet_payload_size_bytes = tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes) {
    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
    switch (cluster_type) {
        case ClusterType::T3K:
            if (num_devices != 8) {
                log_info(tt::LogTest, "This test can only be run 8 chips on T3000 devices");
                return;
            }
            break;
        case ClusterType::GALAXY:
            if (num_devices != 4 && num_devices != 8) {
                log_info(tt::LogTest, "This test can only be run on 4 or 8 chips on Galaxy devices");
                return;
            }
            break;
        default: log_info(tt::LogTest, "This test can only be run on T3000 or Galaxy devices"); return;
    }

    using namespace ttnn::ccl;
    auto topology = ttnn::ccl::Topology::Ring;
    constexpr size_t num_unicasts = 0;
    size_t line_size = num_devices;
    size_t num_devices_with_workers = line_size;
    constexpr bool line_sync = false;

    auto worker_core_logical = [](size_t link) { return CoreCoord(link, 0); };

    // static constexpr size_t source_l1_buffer_address = 1000000;
    static constexpr uint32_t packet_header_cb_index = tt::CB::c_in0;
    static constexpr uint32_t source_payload_cb_index = tt::CB::c_in1;
    static constexpr size_t packet_header_cb_size_in_headers = 5;
    static constexpr bool enable_persistent_fabric_mode = true;
    size_t dest_buffer_size = packet_payload_size_bytes * 4;
    static constexpr tt::DataFormat cb_df = tt::DataFormat::Bfp8;

    Fabric1DRingDeviceInitFixture test_fixture;
    auto view = *(test_fixture.view_);

    std::vector<IDevice*> devices_;

    if (cluster_type == ClusterType::GALAXY) {
        if (num_devices == 4) {
            devices_ = {
                view.get_device(MeshCoordinate(0, 0)),
                view.get_device(MeshCoordinate(0, 1)),
                view.get_device(MeshCoordinate(0, 2)),
                view.get_device(MeshCoordinate(0, 3))};
        } else if (num_devices == 8) {
            devices_ = {
                view.get_device(MeshCoordinate(0, 0)),
                view.get_device(MeshCoordinate(1, 0)),
                view.get_device(MeshCoordinate(2, 0)),
                view.get_device(MeshCoordinate(3, 0)),
                view.get_device(MeshCoordinate(4, 0)),
                view.get_device(MeshCoordinate(5, 0)),
                view.get_device(MeshCoordinate(6, 0)),
                view.get_device(MeshCoordinate(7, 0))};
        }
    } else {
        devices_ = {
            view.get_device(MeshCoordinate(0, 0)),
            view.get_device(MeshCoordinate(0, 1)),
            view.get_device(MeshCoordinate(0, 2)),
            view.get_device(MeshCoordinate(0, 3)),
            view.get_device(MeshCoordinate(1, 3)),
            view.get_device(MeshCoordinate(1, 2)),
            view.get_device(MeshCoordinate(1, 1)),
            view.get_device(MeshCoordinate(1, 0))};
    }

    std::vector<IDevice*> devices;
    devices.reserve(line_size);
    for (size_t i = 0; i < line_size; i++) {
        devices.push_back(devices_[i]);
    }
    // build the mesh device

    // Persistent Fabric Setup
    std::vector<Program> dummy_worker_programs;

    // Other boiler plate setup
    CoreRangeSet worker_cores = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_links - 1, 0)));
    auto worker_cores_vec = corerange_to_cores(worker_cores, std::nullopt, false);
    std::vector<CoreCoord> dest_core_coord;
    dest_core_coord.reserve(num_links);
    for (size_t l = 0; l < num_links; l++) {
        dest_core_coord[l] = CoreCoord(0, l + 1);
    }
    auto sync_core_coord = CoreCoord(0, 0);

    ttnn::SmallVector<std::shared_ptr<Buffer>> device_dest_buffers;
    device_dest_buffers.reserve(line_size);
    for (auto* d : devices) {
        auto local_input_buffer =
            CreateBuffer(InterleavedBufferConfig{d, dest_buffer_size, dest_buffer_size, BufferType::L1});
        device_dest_buffers.push_back(local_input_buffer);
    }

    size_t dest_bank_addr = device_dest_buffers[0]->address();
    TT_FATAL(
        std::all_of(
            device_dest_buffers.begin(),
            device_dest_buffers.end(),
            [dest_bank_addr](const auto& buffer) { return buffer->address() == dest_bank_addr; }),
        "Test setup error: all destination buffers must have the same bank address across devices");

    std::vector<tt::tt_metal::DeviceAddr> global_semaphore_addrs;
    global_semaphore_addrs.reserve(line_size + 1);
    std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> global_semaphore_handles;
    for (size_t i = 0; i < line_size * 4; i++) {
        auto global_semaphores = ttnn::global_semaphore::create_global_semaphore_with_same_address(
            devices_,
            devices_[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
            0,                             // initial value
            tt::tt_metal::BufferType::L1,  // buffer type
            1000                           // attempts
        );
        global_semaphore_handles.push_back(global_semaphores);
        auto global_semaphore_addr =
            ttnn::global_semaphore::get_global_semaphore_address(global_semaphores.global_semaphores.at(0));
        global_semaphore_addrs.push_back(global_semaphore_addr);
    }

    std::vector<IDevice*> worker_devices;
    for (size_t i = 0; i < num_devices_with_workers; i++) {
        worker_devices.push_back(devices[i]);
    }
    // Worker program setup
    std::vector<Program> programs(num_devices_with_workers);
    TT_FATAL(
        programs.size() == worker_devices.size(),
        "Test misconfiguration. Mismatch in line size and devices. Expected line size of {} but got {} devices "
        "instead.",
        line_size,
        worker_devices.size());
    std::vector<KernelHandle> worker_kernel_ids;
    std::vector<size_t> per_device_global_sem_addr_rt_arg;
    uint32_t num_dirs = (uint32_t)has_forward_connection + (uint32_t)has_backward_connection;
    for (size_t i = 0; i < num_devices_with_workers; i++) {
        const size_t line_index = i;
        auto& program = programs[i];
        auto* device = devices[i];
        const size_t sync_core_noc_x = device->worker_core_from_logical_core(sync_core_coord).x;
        const size_t sync_core_noc_y = device->worker_core_from_logical_core(sync_core_coord).y;

        IDevice* backward_device;
        IDevice* forward_device;
        size_t mcast_fwd_hops;
        size_t mcast_bwd_hops;

        backward_device = i == 0 ? devices.back() : devices[i - 1];
        forward_device = i == line_size - 1 ? devices.front() : devices[i + 1];

        // Initialize the fabric handle for worker connection
        mcast_fwd_hops = has_forward_connection ? line_size - 1 : 0;
        mcast_bwd_hops = has_backward_connection ? line_size - 1 : 0;

        // reserve CB
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                packet_header_cb_size_in_headers * sizeof(tt::tt_fabric::PacketHeader),
                {{packet_header_cb_index, cb_df}})
                .set_page_size(packet_header_cb_index, sizeof(tt::tt_fabric::PacketHeader));
        CBHandle sender_workers_cb = CreateCircularBuffer(program, worker_cores, cb_src0_config);

        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(packet_payload_size_bytes, {{source_payload_cb_index, cb_df}})
                .set_page_size(source_payload_cb_index, packet_payload_size_bytes);
        CBHandle sender_workers_payload_cb = CreateCircularBuffer(program, worker_cores, cb_src1_config);

        std::vector<uint32_t> worker_ct_args = {line_sync, line_sync};

        auto worker_kernel_id = tt_metal::CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/ccl/kernels/edm_fabric_writer.cpp",
            worker_cores,
            tt_metal::WriterDataMovementConfig(worker_ct_args));
        worker_kernel_ids.push_back(worker_kernel_id);
        for (size_t l = 0; l < num_links; l++) {
            auto worker_core = worker_cores_vec[l];
            const size_t dest_noc_x = device->worker_core_from_logical_core(dest_core_coord[l]).x;
            const size_t dest_noc_y = device->worker_core_from_logical_core(dest_core_coord[l]).y;
            auto build_connection_args =
                [device, &program, &worker_core, l](
                    bool is_connected_in_direction, IDevice* connected_device, std::vector<uint32_t>& rt_args_out) {
                    rt_args_out.push_back(is_connected_in_direction);
                    if (is_connected_in_direction) {
                        tt::tt_fabric::append_fabric_connection_rt_args(
                            device->id(), connected_device->id(), l, program, {worker_core}, rt_args_out);
                    }
                };

            // Define the send type parameters
            // There is no atomic in
            bool flush_writes_before_atomic_inc = false;
            const size_t num_send_types = 1;
            std::vector<uint32_t> send_types = {static_cast<uint32_t>(tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE)};
            std::vector<uint32_t> chip_send_types = {static_cast<uint32_t>(tt::tt_fabric::CHIP_MULTICAST)};
            std::vector<uint32_t> send_counts_per_type = {static_cast<uint32_t>(num_mcasts)};
            std::vector<uint32_t> num_fwd_hops_per_type = {
                static_cast<uint32_t>(has_forward_connection ? mcast_fwd_hops : 0)};
            std::vector<uint32_t> num_bwd_hops_per_type = {
                static_cast<uint32_t>(has_backward_connection ? mcast_bwd_hops : 0)};
            std::vector<uint32_t> send_type_payload_sizes = {static_cast<uint32_t>(packet_payload_size_bytes)};
            std::vector<uint32_t> flush_send = {static_cast<uint32_t>(flush_writes_before_atomic_inc)};

            // Initialize the base runtime args
            // RT ARGS
            std::vector<uint32_t> rt_args = {
                dest_bank_addr,
                dest_noc_x,
                dest_noc_y,
                dest_noc_x,
                dest_noc_y,

                // Number of send types
                num_send_types};

            // Reserve space for all the arrays we'll add
            rt_args.reserve(
                rt_args.size() + num_send_types * 6 +  // 6 arrays of size num_send_types
                3 +                                    // CB indices
                (has_forward_connection ? 10 : 1) +    // Forward connection args
                (has_backward_connection ? 10 : 1) +   // Backward connection args
                (line_sync ? 6 : 0));                  // Line sync args

            // Copy in all the send type arrays
            std::copy(send_types.begin(), send_types.end(), std::back_inserter(rt_args));
            std::copy(chip_send_types.begin(), chip_send_types.end(), std::back_inserter(rt_args));
            std::copy(send_counts_per_type.begin(), send_counts_per_type.end(), std::back_inserter(rt_args));
            std::copy(num_fwd_hops_per_type.begin(), num_fwd_hops_per_type.end(), std::back_inserter(rt_args));
            std::copy(num_bwd_hops_per_type.begin(), num_bwd_hops_per_type.end(), std::back_inserter(rt_args));
            std::copy(send_type_payload_sizes.begin(), send_type_payload_sizes.end(), std::back_inserter(rt_args));
            std::copy(flush_send.begin(), flush_send.end(), std::back_inserter(rt_args));

            // Add CB indices
            rt_args.push_back(source_payload_cb_index);
            rt_args.push_back(packet_header_cb_index);
            rt_args.push_back(packet_header_cb_size_in_headers);

            // Add fabric connection args
            build_connection_args(has_forward_connection, forward_device, rt_args);
            build_connection_args(has_backward_connection, backward_device, rt_args);

            if (line_sync) {
                rt_args.push_back(sync_core_noc_x);
                rt_args.push_back(sync_core_noc_y);
                if (l == 0) {
                    per_device_global_sem_addr_rt_arg.push_back(rt_args.size());
                }
                TT_FATAL(global_semaphore_addrs.at(0) != -1, "Invalid test setup. Global semaphore address is -1");
                rt_args.push_back(global_semaphore_addrs.at(0));
                // Receives one ack per direction per chip (not including self), plus 1 for self
                rt_args.push_back(num_links * (num_dirs * (num_devices_with_workers - 1) + 1));
                rt_args.push_back(mcast_fwd_hops);
                rt_args.push_back(mcast_bwd_hops);
            }

            tt_metal::SetRuntimeArgs(program, worker_kernel_id, worker_core, rt_args);
        }
    }

    for (size_t i = 0; i < num_op_invocations; i++) {
        log_info(tt::LogTest, "Iteration: {}", i);
        if (i != 0 && line_sync) {
            for (size_t k = 0; k < worker_kernel_ids.size(); k++) {
                auto& worker_rt_args_by_core = GetRuntimeArgs(programs[k], worker_kernel_ids[k]);
                auto global_sem_addr_rt_arg_idx = per_device_global_sem_addr_rt_arg[k];
                for (size_t l = 0; l < num_links; l++) {
                    auto& worker_rt_args = worker_rt_args_by_core[worker_cores_vec[l].x][worker_cores_vec[l].y];
                    worker_rt_args.at(global_sem_addr_rt_arg_idx) =
                        global_semaphore_addrs[i % global_semaphore_addrs.size()];
                }
            }
        }

        build_and_enqueue(worker_devices, programs, i != 0);

        log_info(tt::LogTest, "Waiting for Op finish on all devices");
        for (IDevice* d : devices) {
            tt_metal::Synchronize(d, *ttnn::DefaultQueueId);
        }
        log_info(tt::LogTest, "Main op done");
    }

    log_info(tt::LogTest, "Finished");
}
