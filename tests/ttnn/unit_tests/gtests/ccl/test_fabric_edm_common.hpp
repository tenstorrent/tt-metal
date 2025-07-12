// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include "tt-metalium/kernel_types.hpp"
#include <tt-metalium/fabric.hpp>
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/common/executor.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/erisc_datamover_builder_helper.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "ttnn/operations/experimental/reshape/view.hpp"
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

        if (!(num_devices_ >= 8 &&
              (tt::tt_metal::GetNumPCIeDevices() == 4 || tt::tt_metal::GetNumPCIeDevices() == GALAXY_6U_NUM_DEVICES))) {
            TT_THROW("This suite can only be run on T3000 or TG Wormhole devices");
        }
    }

public:
    BaseFabricFixture() : device_open(false) {}

    BaseFabricFixture(
        tt::tt_fabric::FabricConfig fabric_config,
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) :
        device_open(false) {
        tt::tt_fabric::SetFabricConfig(fabric_config, reliability_mode);
    }

    virtual ~BaseFabricFixture() { tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED); }

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
        devices.reserve(physical_device_ids.size());
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

    Fabric1DFixture(
        tt::tt_fabric::FabricConfig fabric_config,
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) :
        BaseFabricFixture(fabric_config, reliability_mode) {
        this->SetupDevices();
    }

    ~Fabric1DFixture() override { TearDown(); }
};

class Fabric1DDeviceInitFixture {
public:
    tt::ARCH arch_;
    std::size_t num_devices_;
    bool device_open = false;

    // Common constants for both fixtures
    static constexpr size_t TG_NUM_DEVICES = 36;
    static constexpr size_t GALAXY_6U_NUM_DEVICES = 32;

    std::shared_ptr<MeshDevice> mesh_device_;

    // Gets the appropriate mesh shape based on device configuration
    MeshShape GetDeterminedMeshShape() const { return SystemMesh::instance().shape(); }

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

    void SetupDevices() {
        ValidateEnvironment();

        const MeshShape cluster_shape = GetDeterminedMeshShape();
        const auto& physical_device_ids = SystemMesh::instance().get_mapped_physical_device_ids(cluster_shape);

        mesh_device_ = MeshDevice::create(MeshDeviceConfig(cluster_shape));
        device_open = true;
    }

    void TearDown() {
        if (device_open) {
            mesh_device_->close();
            device_open = false;
        }
    }

    Fabric1DDeviceInitFixture() : device_open(false) { this->SetupDevices(); }

    Fabric1DDeviceInitFixture(
        tt::tt_fabric::FabricConfig fabric_config,
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) :
        device_open(false) {
        tt::tt_fabric::SetFabricConfig(fabric_config, reliability_mode);
        this->SetupDevices();
    }

    ~Fabric1DDeviceInitFixture() {
        TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
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

    MeshFabric1DFixture(
        tt::tt_fabric::FabricConfig fabric_config,
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) :
        BaseFabricFixture(fabric_config, reliability_mode) {
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
    Fabric1DLineDeviceInitFixture() : Fabric1DFixture(tt::tt_fabric::FabricConfig::FABRIC_1D) {}
};

class Fabric1DRingDeviceInitFixture : public Fabric1DFixture {
public:
    Fabric1DRingDeviceInitFixture() : Fabric1DFixture(tt::tt_fabric::FabricConfig::FABRIC_1D_RING) {}
};
class Fabric1DRingStrictDeviceInitFixture : public Fabric1DDeviceInitFixture {
public:
    Fabric1DRingStrictDeviceInitFixture() :
        Fabric1DDeviceInitFixture(
            tt::tt_fabric::FabricConfig::FABRIC_1D_RING,
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) {}
};
class Fabric1DRingRelaxedDeviceInitFixture : public Fabric1DDeviceInitFixture {
public:
    Fabric1DRingRelaxedDeviceInitFixture() :
        Fabric1DDeviceInitFixture(
            tt::tt_fabric::FabricConfig::FABRIC_1D_RING,
            tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE) {}
};

class MeshFabric1DLineDeviceInitFixture : public MeshFabric1DFixture {
public:
    MeshFabric1DLineDeviceInitFixture() : MeshFabric1DFixture(tt::tt_fabric::FabricConfig::FABRIC_1D) {}
};

class MeshFabric1DRingDeviceInitFixture : public MeshFabric1DFixture {
public:
    MeshFabric1DRingDeviceInitFixture() : MeshFabric1DFixture(tt::tt_fabric::FabricConfig::FABRIC_1D_RING) {}
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
                    log_error(tt::LogTest, "Output mismatch");
                }
                log_error(tt::LogTest, "[{}]: expected {} got {}", i, inputs[i], output_buffer[i]);
                num_printed_mismatches++;
            }
            pass = false;
        }
    }
    if (num_printed_mismatches > 0) {
        log_error(tt::LogTest, "... (remaining mismatches omitted)");
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
        log_error(tt::LogTest, "Failed compile: {}", e.what());
        throw e;
    }

    log_trace(tt::LogTest, "Running...");

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

template <typename ProgramContainer>
static void build_and_enqueue(
    const std::vector<IDevice*>& devices, ProgramContainer& programs, bool enqueue_only = false) {
    static_assert(
        std::is_same_v<ProgramContainer, std::vector<Program*>> ||
            std::is_same_v<ProgramContainer, std::vector<Program>>,
        "programs must be a vector of Program* or Program");
    TT_FATAL(
        devices.size() == programs.size(),
        "Number of devices must match number of programs when calling build_and_enqueue in test");

    // Parallel compile and enqueue as a single atomic operation per device
    std::vector<std::shared_future<void>> futures;
    futures.reserve(devices.size());

    for (size_t i = 0; i < devices.size(); i++) {
        futures.emplace_back(tt::tt_metal::detail::async([&devices, &programs, i, enqueue_only]() {
            if constexpr (std::is_same_v<ProgramContainer, std::vector<Program*>>) {
                if (!enqueue_only) {
                    tt::tt_metal::detail::CompileProgram(devices[i], *programs[i]);
                }
                tt_metal::EnqueueProgram(devices[i]->command_queue(), *programs[i], false);
            } else {
                if (!enqueue_only) {
                    tt::tt_metal::detail::CompileProgram(devices[i], programs[i]);
                }
                tt_metal::EnqueueProgram(devices[i]->command_queue(), programs[i], false);
            }
        }));
    }

    // Wait for all compile and enqueue operations to complete
    for (const auto& future : futures) {
        future.get();
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
    std::size_t edm_buffer_size_no_header,
    uint32_t page_size,
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
    uint32_t packet_header_buffer_cb_id,
    bool scatter_write) {
    const auto& edm_noc_core = CoreCoord(worker_fabric_connection.edm_noc_x, worker_fabric_connection.edm_noc_y);
    std::vector<uint32_t> sender_worker_reader_compile_args{
        src_is_dram,      //
        num_pages_total,  //
        page_size,
        num_pages_per_edm_buffer,
        scatter_write};
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
        page_size,
        worker_fabric_connection.num_buffers_per_channel,
        dest_is_dram,
        std::holds_alternative<mcast_send>(mode) ? 1 : 0,
        scatter_write};
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
        edm_buffer_size_no_header,
        dram_output_buffer_base_addr,
        local_worker_last_message_semaphore_id,
        worker_buffer_index_semaphore_id,
        worker_fabric_connection.buffer_index_semaphore_id,
        packet_header_buffer_cb_id};

    if (std::holds_alternative<mcast_send>(mode)) {
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).distance);
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).range);
    } else {
        sender_worker_writer_runtime_args.push_back(std::get<unicast_send>(mode).distance);
    }

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
    tt::DataFormat df = page_size == 1024   ? tt::DataFormat::Bfp8
                        : page_size == 2048 ? tt::DataFormat::Float16
                                            : tt::DataFormat::Float32;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, page_size);
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
    bool scatter_write) {
    auto& sender_program = programs.at(0);
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

    uint32_t packet_header_buffer_cb_id = tt::CBIndex::c_1;
    // allocate a circular buffer of size 8k
    constexpr size_t packet_header_buffer_size = 8192;
    tt_metal::CircularBufferConfig packet_header_buffer_config =
        tt_metal::CircularBufferConfig(
            packet_header_buffer_size, {{packet_header_buffer_cb_id, tt::DataFormat::Float16}})
            .set_page_size(packet_header_buffer_cb_id, page_size);
    auto packet_header_buffer =
        tt_metal::CreateCircularBuffer(sender_program, worker_cores.at(0), packet_header_buffer_config);
    tt_metal::detail::WriteToBuffer(local_output_buffer, all_zeros);

    auto local_input_buffer_address = local_input_buffer->address();
    auto local_output_buffer_address = local_output_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    // EDM Builder Setup
    ////////////////////////////////////////////////////////////////////////////

    const std::size_t edm_buffer_size_no_header =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes;

    auto chip0_worker_fabric_connection = chip_0_edm_builder.build_connection_to_worker_channel();
    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    const std::size_t pages_per_send = chip0_worker_fabric_connection.buffer_size_bytes / page_size;
    const auto& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    generate_sender_worker_kernels(
        sender_program,
        sender_device,
        worker_core,
        chip0_worker_fabric_connection,
        unicast_send{2},  // 2 hops because we are looping back to ourselves
        edm_buffer_size_no_header,
        page_size,
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
        packet_header_buffer_cb_id,
        scatter_write);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<IDevice*> devices = {sender_device};
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
    const uint32_t page_size,
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
    tt::DataFormat df = page_size == 1024   ? tt::DataFormat::Bfp8
                        : page_size == 2048 ? tt::DataFormat::Float16
                                            : tt::DataFormat::Float32;

    {
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_size, {{first_cb_index, df}})
                .set_page_size(first_cb_index, page_size);
        CBHandle cb0 = CreateCircularBuffer(program, worker_core, cb_src0_config);
    }
    {
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_size, {{second_cb_index, df}})
                .set_page_size(second_cb_index, page_size);
        CBHandle cb1 = CreateCircularBuffer(program, worker_core, cb_src1_config);
    }

    generate_multi_input_test_worker_reader_kernel(
        program,
        {first_cb_index, second_cb_index},
        {&input_tensor0, &input_tensor1},
        device,
        page_size,
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
        page_size,
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
    std::optional<SubdeviceInfo>& subdevice_managers) {
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

    auto first_cb_index = tt::CB::c_in0;
    auto second_cb_index = tt::CB::c_in1;

    auto output_tensor_dest_device = devices.at(output_tensor_dest_device_index);
    TT_ASSERT(input_tensor0.logical_shape()[-2] != 1);

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

    auto chip0_worker_forward_fabric_connection =
        fabric_enabled ? line_fabric->uniquely_connect_worker(devices[0], ttnn::ccl::EdmLineFabricOpInterface::FORWARD)
                       : std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>{std::nullopt};

    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> chip0_worker_backward_fabric_connection = std::nullopt;

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
        page_size,
        num_pages_per_edm_buffer,
        in0_tensor_slice,
        in1_tensor_slice,
        out0_tensor_slice,
        out1_tensor_slice,
        std::nullopt,
        chip0_worker_forward_fabric_connection,
        chip0_worker_backward_fabric_connection,
        dest_args);

    log_info(tt::LogTest, "subdevice_managers.has_value(): {}", subdevice_managers.has_value());
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    run_programs(programs, std::vector<IDevice*>{devices[0]});
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
    bool scatter_write) {
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    const std::size_t edm_buffer_size_no_header =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes;
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

    uint32_t packet_header_buffer_cb_id = tt::CBIndex::c_1;
    // allocate a circular buffer of size 8k
    constexpr size_t packet_header_buffer_size = 8192;
    tt_metal::CircularBufferConfig packet_header_buffer_config =
        tt_metal::CircularBufferConfig(
            packet_header_buffer_size, {{packet_header_buffer_cb_id, tt::DataFormat::Float16}})
            .set_page_size(packet_header_buffer_cb_id, page_size);
    auto packet_header_buffer =
        tt_metal::CreateCircularBuffer(programs[0], worker_cores.at(0), packet_header_buffer_config);

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

    auto chip0_worker_fabric_connection =
        line_fabric.uniquely_connect_worker(devices[0], ttnn::ccl::EdmLineFabricOpInterface::FORWARD);

    const std::size_t pages_per_send = chip0_worker_fabric_connection.buffer_size_bytes / page_size;
    generate_sender_worker_kernels(
        programs[0],
        devices[0],
        worker_core,
        chip0_worker_fabric_connection,
        mcast_send{mcast_first_chip, mcast_last_chip - mcast_first_chip + 1},
        edm_buffer_size_no_header,
        page_size,
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
        packet_header_buffer_cb_id,
        scatter_write);

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
    tt::tt_fabric::TerminationSignal termination_mode = tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE) {
    log_info(tt::LogTest, "Tearing down fabric");

    // Wait for workers to finish
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
    std::optional<SubdeviceInfo>& subdevice_managers,
    std::optional<std::vector<Program>>& fabric_programs,
    std::vector<Program*>& fabric_program_ptrs,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& line_fabric,
    std::optional<size_t> num_links = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    size_t switch_interval = 0,
    bool loopback_on_last_device = false,
    bool is_galaxy = false,
    const tt::tt_fabric::FabricRouterBufferConfig& edm_buffer_config = tt::tt_fabric::FabricRouterBufferConfig{}) {
    log_info(tt::LogTest, "Enabling persistent fabric");
    fabric_programs = std::vector<Program>(devices.size());
    subdevice_managers = create_subdevices(devices);
    std::transform(
        fabric_programs->begin(), fabric_programs->end(), std::back_inserter(fabric_program_ptrs), [](auto& p) {
            return &p;
        });

    line_fabric = ttnn::ccl::EdmLineFabricOpInterface(
        devices, fabric_program_ptrs, num_links.value_or(1), false, topology, is_galaxy, edm_buffer_config);
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

    TT_FATAL(fabric_programs.has_value(), "Fabric programs must be set if fabric is enabled");
    TT_FATAL(devices.size() == fabric_programs->size(), "Number of devices must match number of programs");

    log_info(tt::LogTest, "Building EDM kernels");
    line_fabric->build_kernels();
    build_and_enqueue(devices, *fabric_programs);
}

// RESUME HERE AND IMPLEMENT MCAST TEST
int TestLineFabricEntrypoint(
    const size_t mcast_first_chip,
    const size_t mcast_last_chip,
    const uint32_t page_size,
    const uint32_t num_pages_total,
    const bool src_is_dram,
    const bool dest_is_dram,
    bool scatter_write = false) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops

    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
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
    std::vector<Program> programs(1);
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> line_fabric;
    setup_test_with_persistent_fabric(devices, subdevice_managers, fabric_programs, fabric_program_ptrs, line_fabric);

    auto launch_workers = [&](std::vector<Program>& _programs) -> bool {
        bool success = false;
        try {
            success = RunLineFabricTest(
                std::vector<IDevice*>{devices[0]},
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
                scatter_write);

        } catch (std::exception& e) {
            log_error(tt::LogTest, "Caught exception: {}", e.what());
            test_fixture.TearDown();
            return false;
        }
        return success;
    };
    bool success = launch_workers(programs);

    std::vector<Program> second_run_programs(1);
    success = launch_workers(second_run_programs);
    persistent_fabric_teardown_sequence(
        devices, subdevice_managers, line_fabric.value(), tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);

    test_fixture.TearDown();

    return success ? 0 : -1;
}

int TestLoopbackEntrypoint(
    const uint32_t page_size,
    const uint32_t num_pages_total,
    const bool src_is_dram,
    const bool dest_is_dram,
    bool scatter_write = false) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;

    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
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
    tt_xy_pair eth_sender_core;
    do {
        TT_FATAL(eth_sender_core_iter != eth_sender_core_iter_end, "Error");
        std::tie(device_id, eth_receiver_core) = device_0->get_connected_ethernet_core(*eth_sender_core_iter);
        eth_sender_core = *eth_sender_core_iter;
        eth_sender_core_iter++;
    } while (device_id != device_1->id());
    TT_ASSERT(device_id == device_1->id());
    // const auto& device_1 = test_fixture.mesh_device_->get_device(device_id);

    std::vector<Program> programs(1);
    std::optional<std::vector<Program>> fabric_programs;
    auto& sender_program = programs.at(0);

    log_info(tt::LogTest, "Enabling persistent fabric");
    fabric_programs = std::vector<Program>(2);
    subdevice_managers = create_subdevices({device_0, device_1});

    auto& fabric_sender_program = fabric_programs->at(0);
    auto& fabric_receiver_program = fabric_programs->at(1);
    IDevice* sender_device = device_0;
    IDevice* receiver_device = device_1;

    const std::size_t edm_buffer_size = tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes;
    const chip_id_t local_chip_id = 0;
    const chip_id_t remote_chip_id = 1;
    // Note this is a fabric level test so we can keep the use of packet header here (we aren't using the
    // fabric API yet because that requires us to use device init fabric, which doesn't implement the loopback
    // that we take advantage of here)
    const auto& edm_config = tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size + sizeof(PACKET_HEADER_TYPE));
    auto chip_0_edm_builder = tt::tt_fabric::FabricEriscDatamoverBuilder::build(
        sender_device, fabric_sender_program, eth_sender_core, local_chip_id, remote_chip_id, edm_config);
    chip_0_edm_builder.set_firmware_context_switch_interval(0);
    auto chip_1_edm_builder = tt::tt_fabric::FabricEriscDatamoverBuilder::build(
        receiver_device, fabric_receiver_program, eth_receiver_core, remote_chip_id, local_chip_id, edm_config);
    chip_1_edm_builder.set_firmware_context_switch_interval(0);
    // Create the loopback connection on the second device
    chip_1_edm_builder.connect_to_downstream_edm(chip_1_edm_builder);
    auto local_edm_kernel = ttnn::ccl::generate_edm_kernel(
        fabric_sender_program,
        sender_device,
        chip_0_edm_builder,
        eth_sender_core,
        DataMovementProcessor::RISCV_0,
        NOC::NOC_0);
    auto remote_edm_kernel = ttnn::ccl::generate_edm_kernel(
        fabric_receiver_program,
        receiver_device,
        chip_1_edm_builder,
        eth_receiver_core,
        DataMovementProcessor::RISCV_0,
        NOC::NOC_0);

    tt::tt_metal::detail::CompileProgram(sender_device, fabric_sender_program);
    tt::tt_metal::detail::CompileProgram(receiver_device, fabric_receiver_program);
    tt_metal::EnqueueProgram(sender_device->command_queue(), fabric_sender_program, false);
    tt_metal::EnqueueProgram(receiver_device->command_queue(), fabric_receiver_program, false);
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
            scatter_write);
    } catch (std::exception& e) {
        log_error(tt::LogTest, "Caught exception: {}", e.what());
        test_fixture.TearDown();
        return -1;
    }

    {
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
                scatter_write);
        } catch (std::exception& e) {
            log_error(tt::LogTest, "Caught exception: {}", e.what());
            test_fixture.TearDown();
            return -1;
        }
        // Wait for worker programs to finish

        auto d0_worker_subdevice = device_0->get_sub_device_ids()[TEST_WORKERS_SUBDEVICE_INDEX];
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
    const ttnn::ccl::cmd::CclCommandDestArgs& dest_args) {
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
        return true;
    }
    Fabric1DFixture test_fixture;

    auto view = *(test_fixture.view_);

    std::vector<IDevice*> devices;
    devices.reserve(fabric_num_devices);
    for (size_t i = 0; i < fabric_num_devices; i++) {
        devices.push_back(view.get_device(MeshCoordinate(0, i)));
    }

    std::vector<Program> programs(1);
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> line_fabric;
    if (test_mode != TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK) {
        setup_test_with_persistent_fabric(
            devices, subdevice_managers, fabric_programs, fabric_program_ptrs, line_fabric);
        TT_FATAL(subdevice_managers.has_value(), "Subdevice managers must be set if fabric is enabled");
    }

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
            subdevice_managers);
    };

    auto pass = launch_ccl_command_interpreter_workers(programs);

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
    if (test_mode != TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK) {
        persistent_fabric_teardown_sequence(
            devices, subdevice_managers, line_fabric.value(), tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    }

    log_info(tt::LogTest, "Finished");
    for (auto d : devices) {
        tt_metal::Synchronize(d, *ttnn::DefaultQueueId);
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
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs{});

    return pass;
}

// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////
// ////               FABRIC MCAST TENSOR WRITE (2 INPUT)
// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////

void RunFabricMcastFullTensorPropagateTest(
    const ttnn::Shape& tensor_shape, size_t distance_dest_device, size_t num_devices) {
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
    ASSERT_EQ(input_tensor0.logical_shape(), tensor_shape);
    ASSERT_EQ(input_tensor1.logical_shape(), tensor_shape);
    ASSERT_EQ(output_tensor0.logical_shape(), tensor_shape);
    ASSERT_EQ(output_tensor1.logical_shape(), tensor_shape);

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
        dest_args);

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
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
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
        log_info(tt::LogTest, "Tensor[{}] allocated starting at address {}", i, device_tensors[i].buffer()->address());
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

#include "ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
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

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
void run_all_gather_with_persistent_fabric(const size_t dim, const size_t num_links, ttnn::Shape const& input_shape) {
    log_info(tt::LogTest, "entering test");
    constexpr auto layout = Layout::TILE;
    // DEVICES setuip
    constexpr size_t test_expected_num_devices = 4;
    if (tt::tt_metal::GetNumAvailableDevices() < test_expected_num_devices) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
        return;
    }

    // Initialize MeshDevice with 1D Fabric
    MeshFabric1DFixture test_fixture(tt::tt_fabric::FabricConfig::FABRIC_1D);
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

    // Replicate the tensor across (1, num_devices) submesh.
    const Tensor input_mesh_tensor = ttnn::distributed::distribute_tensor(
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::BFLOAT16), input_shape).to_layout(layout),
        *ttnn::distributed::create_mesh_mapper(
            *test_fixture.mesh_device_,
            ttnn::distributed::MeshMapperConfig{
                .placements =
                    {ttnn::distributed::MeshMapperConfig::Replicate{},
                     ttnn::distributed::MeshMapperConfig::Replicate{}},
                .mesh_shape_override = MeshShape{1, num_devices}}),
        *test_fixture.mesh_device_);
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
        {multi_device_global_semaphore},
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
    constexpr size_t test_expected_num_devices = 8;
    if (tt::tt_metal::GetNumAvailableDevices() < test_expected_num_devices) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
        return;
    }

    // Initialize MeshDevice with 1D Fabric
    MeshFabric1DFixture test_fixture(tt::tt_fabric::FabricConfig::FABRIC_1D_RING);
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

    // Replicate the tensor across (1, num_devices) submesh.
    const Tensor input_mesh_tensor = ttnn::distributed::distribute_tensor(
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::BFLOAT16), input_shape).to_layout(layout),
        *ttnn::distributed::create_mesh_mapper(
            *test_fixture.mesh_device_,
            ttnn::distributed::MeshMapperConfig{
                .placements =
                    {ttnn::distributed::MeshMapperConfig::Replicate{},
                     ttnn::distributed::MeshMapperConfig::Replicate{}},
                .mesh_shape_override = MeshShape{1, num_devices}}),
        *test_fixture.mesh_device_);

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
        {multi_device_global_semaphore},
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

struct Fabric1DWorkerConfig {
    IDevice* backward_device = nullptr;
    IDevice* forward_device = nullptr;
    bool has_forward_connection = false;
    bool has_backward_connection = false;
    bool unicast_forward = false;
    size_t num_fwd_hops = 0;
    size_t num_bwd_hops = 0;
    size_t sync_num_fwd_hops = 0;
    size_t sync_num_bwd_hops = 0;
    size_t sync_count_per_link = 0;
};

std::vector<CoreCoord> compute_top_row_ethernet_cores(IDevice* device, const Fabric1DWorkerConfig& worker_config) {
    std::vector<CoreCoord> reordered_ethernet_cores;
    if (worker_config.has_forward_connection) {
        for (auto core : device->get_ethernet_sockets(worker_config.forward_device->id())) {
            auto core_virtual = device->virtual_core_from_logical_core(core, CoreType::ETH);
            reordered_ethernet_cores.push_back(core_virtual);
        }
        std::sort(reordered_ethernet_cores.begin(), reordered_ethernet_cores.end(), [](auto& a, auto& b) {
            return a.x < b.x;
        });
    } else if (worker_config.has_backward_connection) {
        for (auto core : device->get_ethernet_sockets(worker_config.backward_device->id())) {
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
    bool use_galaxy,
    bool use_tg,
    size_t line_size,
    ttnn::ccl::Topology topology,
    const MeshDeviceView& view,
    size_t num_fabric_rows,
    size_t num_fabric_cols) {
    bool use_default_device_selection = num_fabric_rows == 0 && num_fabric_cols == 0;
    std::vector<std::vector<IDevice*>> fabrics_under_test;
    if (use_default_device_selection) {
        fabrics_under_test.push_back(
            generate_default_line_fabric_under_test(use_galaxy, use_tg, line_size, topology, view));
    } else {
        fabrics_under_test.reserve(num_fabric_rows + num_fabric_cols);
        TT_FATAL(
            num_fabric_rows <= view.num_rows(),
            "num_rows_requested must be less than or equal to the number of rows in the mesh. Requested: {}, "
            "Available: {}",
            num_fabric_rows,
            view.num_rows());
        TT_FATAL(
            num_fabric_cols <= view.num_cols(),
            "num_cols_requested must be less than or equal to the number of cols in the mesh. Requested: {}, "
            "Available: {}",
            num_fabric_cols,
            view.num_cols());
        for (size_t i = 0; i < num_fabric_rows; i++) {
            fabrics_under_test.push_back(view.get_devices_on_row(i));
        }
        for (size_t i = 0; i < num_fabric_cols; i++) {
            fabrics_under_test.push_back(view.get_devices_on_column(i));
        }
    }

    return fabrics_under_test;
}

template <typename FABRIC_DEVICE_FIXTURE>
void create_fabric_fixture(std::unique_ptr<Fabric1DFixture>& test_fixture, bool use_galaxy) {
    auto fixture_recreate_needed = []() -> bool {
        auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
        return (
            // prev not Fabric1D, now Fabric1D
            (fabric_config != tt::tt_fabric::FabricConfig::DISABLED &&
             std::is_same_v<FABRIC_DEVICE_FIXTURE, Fabric1DFixture>) ||
            // prev not Fabric1DLine, now Fabric1DLine
            (fabric_config != tt::tt_fabric::FabricConfig::FABRIC_1D &&
             std::is_same_v<FABRIC_DEVICE_FIXTURE, Fabric1DLineDeviceInitFixture>) ||
            // prev not Fabric1DRing, now Fabric1DRing
            (fabric_config != tt::tt_fabric::FabricConfig::FABRIC_1D_RING &&
             std::is_same_v<FABRIC_DEVICE_FIXTURE, Fabric1DRingDeviceInitFixture>));
    }();

    if (test_fixture == nullptr) {
        test_fixture = std::make_unique<FABRIC_DEVICE_FIXTURE>();
    } else {
        // NOTE: Currently (device init fabric || galaxy) is always recreate fabrix fixture
        if (fixture_recreate_needed || use_galaxy) {
            test_fixture.reset();
            test_fixture = std::make_unique<FABRIC_DEVICE_FIXTURE>();
        }
    }
}

Fabric1DWorkerConfig get_fabric_1d_worker_config(
    size_t device_index,
    const std::vector<IDevice*>& devices,
    ttnn::ccl::Topology topology,
    FabricTestMode fabric_mode,
    size_t line_size,
    size_t line_index,
    bool senders_are_unidirectional,
    size_t num_devices_with_workers) {
    Fabric1DWorkerConfig config;
    if (topology == ttnn::ccl::Topology::Ring && fabric_mode != FabricTestMode::RingAsLinear) {
        config.backward_device = device_index == 0 ? devices.back() : devices[device_index - 1];
        config.forward_device = device_index == line_size - 1 ? devices.front() : devices[device_index + 1];

        // Initialize the fabric handle for worker connection
        config.has_forward_connection = true;
        config.has_backward_connection = true;
        config.unicast_forward = true;
        // Have the sync for ring always use the same algorithm as HalfRing
        config.sync_num_fwd_hops = tt::div_up(line_size - 1, 2);
        config.sync_num_bwd_hops = line_size - 1 - config.sync_num_fwd_hops;
        if (device_index % 2 == 0) {
            std::swap(config.sync_num_fwd_hops, config.sync_num_bwd_hops);
        }
        if (fabric_mode == FabricTestMode::HalfRing) {
            config.num_fwd_hops = tt::div_up(line_size - 1, 2);
            config.num_bwd_hops = line_size - 1 - config.num_fwd_hops;
            if (device_index % 2 == 0) {
                std::swap(config.num_fwd_hops, config.num_bwd_hops);
            }
            config.sync_num_fwd_hops = config.num_fwd_hops;
            config.sync_num_bwd_hops = config.num_bwd_hops;
            // We will get 1 inc per remote chip + 1 local
            config.sync_count_per_link = num_devices_with_workers;
        } else if (fabric_mode == FabricTestMode::FullRing) {
            config.num_fwd_hops = line_size - 1;
            config.num_bwd_hops = line_size - 1;
            config.sync_num_fwd_hops = config.num_fwd_hops;
            config.sync_num_bwd_hops = config.num_bwd_hops;
            // We will get 2 inc per remote chip + 1 local
            config.sync_count_per_link = 2 * (num_devices_with_workers - 1) + 1;
        } else if (fabric_mode == FabricTestMode::SaturateChipToChipRing) {
            // We want to saturate the middle links between chip 1 and 2 in a 4 chip ring with the dateline
            // between the first and last chip Mcast 2 hops from chip 1 F and chip 2 B, which is S0 -> R0
            // Mcast 3 hops from chip 0 F and chip 3 B, which is S1 -> R0 Mcast 4 hops from Chip 3 F and
            // chip 0 B, which is S2 -> R1
            if (line_index == line_size - 1) {
                config.num_fwd_hops = line_size - 1;
            } else {
                config.num_fwd_hops = line_size - 2 - line_index;
            }
            if (line_index == 0) {
                config.num_bwd_hops = line_size - 1;
            } else {
                config.num_bwd_hops = line_index - 1;
            }
            // The above calculations calculates the number of hops to land on the dest chip
            // Extend by one so we mcast through them
            if (config.num_fwd_hops != 0) {
                config.num_fwd_hops++;
            }
            if (config.num_bwd_hops != 0) {
                config.num_bwd_hops++;
            }
            // Flush all the way around the ring
            config.sync_num_fwd_hops = line_size;
            config.sync_num_bwd_hops = line_size;
            // We will get 2 inc for all chips + 1 local
            config.sync_count_per_link = 2 * num_devices_with_workers + 1;
        } else {
            TT_THROW("Invalid fabric mode");
        }
        if (config.num_fwd_hops >= config.num_bwd_hops) {
            config.unicast_forward = true;
        } else {
            config.unicast_forward = false;
        }
    } else {
        config.backward_device = device_index == 0 ? nullptr : devices[device_index - 1];
        config.forward_device = device_index == line_size - 1 ? nullptr : devices[device_index + 1];

        // Initialize the fabric handle for worker connection
        bool start_of_line = line_index == 0;
        bool end_of_line = line_index == line_size - 1;
        config.has_forward_connection = !end_of_line;
        config.has_backward_connection = !start_of_line;
        config.unicast_forward = line_index < (line_size / 2);
        config.num_fwd_hops = line_size - line_index - 1;
        config.num_bwd_hops = line_index;
        config.sync_num_fwd_hops = config.num_fwd_hops;
        config.sync_num_bwd_hops = config.num_bwd_hops;

        // Do this AFTER sync_num_fwd_hops and sync_num_bwd_hops are set
        // otherwise sync hops will be misconfigured - you'll get a hang because
        // setup/teardown will be done incorrectly

        if (senders_are_unidirectional) {
            if (config.unicast_forward) {
                config.num_bwd_hops = 0;
            } else {
                config.num_fwd_hops = 0;
            }
        }
        // We will get 1 inc per remote chip + 1 local
        config.sync_count_per_link = num_devices_with_workers;
    }
    return config;
}

tt::tt_fabric::FabricRouterBufferConfig get_edm_buffer_config_wormhole(
    tt::ClusterType cluster_type, FabricTestMode fabric_mode, size_t line_size) {
    tt::tt_fabric::FabricRouterBufferConfig buffer_config{};
    switch (cluster_type) {
        case tt::ClusterType::TG:
        case tt::ClusterType::GALAXY:
            // long axis, more buffering.
            if (line_size >= tt::tt_fabric::FabricEriscDatamoverConfig::MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD) {
                if (fabric_mode == FabricTestMode::HalfRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, true, true};
                } else if (fabric_mode == FabricTestMode::FullRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{false, true, true, false, true};
                } else if (fabric_mode == FabricTestMode::SaturateChipToChipRing) {
                    // SaturateChipToChipRing cannot use the buffering optimization since it writes back to itself.
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, false, false, false};
                } else if (fabric_mode == FabricTestMode::RingAsLinear) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, true, false};
                }
            } else {
                if (fabric_mode == FabricTestMode::HalfRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, true, false};
                } else if (fabric_mode == FabricTestMode::FullRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, false, true};
                } else if (fabric_mode == FabricTestMode::SaturateChipToChipRing) {
                    // SaturateChipToChipRing cannot use the buffering optimization since it writes back to itself.
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, false, false, false};
                } else if (fabric_mode == FabricTestMode::RingAsLinear) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, true, false};
                }
            }
            break;
        case tt::ClusterType::T3K:
            // Need more tunning on T3K, for now keep the long/short axis the same.
            if (line_size >= tt::tt_fabric::FabricEriscDatamoverConfig::MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD) {
                if (fabric_mode == FabricTestMode::HalfRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, false, true, true};
                } else if (fabric_mode == FabricTestMode::FullRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{false, true, true, false, true};
                } else if (fabric_mode == FabricTestMode::SaturateChipToChipRing) {
                    // SaturateChipToChipRing cannot use the buffering optimization since it writes back to itself.
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, false, false, false};
                } else if (fabric_mode == FabricTestMode::RingAsLinear) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, true, false};
                }
            } else {
                if (fabric_mode == FabricTestMode::HalfRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, false, true, true};
                } else if (fabric_mode == FabricTestMode::FullRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{false, true, true, false, true};
                } else if (fabric_mode == FabricTestMode::SaturateChipToChipRing) {
                    // SaturateChipToChipRing cannot use the buffering optimization since it writes back to itself.
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, false, false, false};
                } else if (fabric_mode == FabricTestMode::RingAsLinear) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, true, false};
                }
            }
            break;
        default: break;
    }
    return buffer_config;
}

tt::tt_fabric::FabricRouterBufferConfig get_edm_buffer_config_blackhole(
    tt::ClusterType cluster_type, FabricTestMode fabric_mode, size_t line_size) {
    tt::tt_fabric::FabricRouterBufferConfig buffer_config{};
    switch (cluster_type) {
        // For now just copy the galaxy config to BH since we don't have any data points.
        case tt::ClusterType::P150_X4:
            if (line_size >= tt::tt_fabric::FabricEriscDatamoverConfig::MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD) {
                if (fabric_mode == FabricTestMode::HalfRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, true, false};
                } else if (fabric_mode == FabricTestMode::FullRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{false, true, true, false, true};
                } else if (fabric_mode == FabricTestMode::SaturateChipToChipRing) {
                    // SaturateChipToChipRing cannot use the buffering optimization since it writes back to itself.
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, false, false, false};
                } else if (fabric_mode == FabricTestMode::RingAsLinear) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, true, false};
                }
            } else {
                if (fabric_mode == FabricTestMode::HalfRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, true, false};
                } else if (fabric_mode == FabricTestMode::FullRing) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{false, true, true, false, true};
                } else if (fabric_mode == FabricTestMode::SaturateChipToChipRing) {
                    // SaturateChipToChipRing cannot use the buffering optimization since it writes back to itself.
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, false, false, false};
                } else if (fabric_mode == FabricTestMode::RingAsLinear) {
                    buffer_config = tt::tt_fabric::FabricRouterBufferConfig{true, true, true, true, false};
                }
            }
            break;
        // the other BH cluster type P150_X2 only has 2 devices, not suitable for ring.
        default: break;
    }
    return buffer_config;
}

tt::tt_fabric::FabricRouterBufferConfig get_edm_buffer_config_helper(FabricTestMode fabric_mode, size_t line_size) {
    const auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
    const auto arch = tt::tt_metal::MetalContext::instance().hal().get_arch();
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: return get_edm_buffer_config_wormhole(cluster_type, fabric_mode, line_size);
        case tt::ARCH::BLACKHOLE: return get_edm_buffer_config_blackhole(cluster_type, fabric_mode, line_size);
        default: return tt::tt_fabric::FabricRouterBufferConfig{};
    }
}

template <typename FABRIC_DEVICE_FIXTURE = Fabric1DFixture>
void Run1DFabricPacketSendTest(
    std::unique_ptr<Fabric1DFixture>& test_fixture,
    const std::vector<Fabric1DPacketSendTestSpec>& test_specs,
    const WriteThroughputStabilityTestWithPersistentFabricParams& params = {},
    size_t fabric_context_switch_interval =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_firmware_context_switch_interval) {
    constexpr bool use_device_init_fabric = std::is_same_v<FABRIC_DEVICE_FIXTURE, Fabric1DLineDeviceInitFixture> ||
                                            std::is_same_v<FABRIC_DEVICE_FIXTURE, Fabric1DRingDeviceInitFixture>;
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
    bool use_t3k = num_devices == 8;
    bool use_galaxy = num_devices == 32;
    bool use_tg = use_galaxy && tt::tt_metal::GetNumPCIeDevices() == 4;
    bool is_6u_galaxy = use_galaxy && tt::tt_metal::GetNumPCIeDevices() == 32;
    if (num_devices < 4) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
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
    if (use_device_init_fabric && params.num_fabric_rows == 0 && params.num_fabric_cols == 0) {
        TT_FATAL(use_t3k, "Using the full mesh as one ring topoplogy is only supported for T3K");
    }

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

    const auto edm_buffer_config = get_edm_buffer_config_helper(fabric_mode, line_size);

    // static constexpr size_t source_l1_buffer_address = 1000000;
    static constexpr uint32_t packet_header_cb_index = tt::CB::c_in0;
    static constexpr uint32_t source_payload_cb_index = tt::CB::c_in1;
    static constexpr size_t packet_header_cb_size_in_headers = 5;
    auto max_packet_payload_size_bytes =
        std::max_element(test_specs.begin(), test_specs.end(), [](const auto& a, const auto& b) {
            return a.packet_payload_size_bytes < b.packet_payload_size_bytes;
        })->packet_payload_size_bytes;
    size_t dest_buffer_size = max_packet_payload_size_bytes * 4;
    static constexpr tt::DataFormat cb_df = tt::DataFormat::Bfp8;

    log_info(tt::LogTest, "Device open and fabric init");
    // MeshFabric1DLineDeviceInitFixture test_fixture;
    create_fabric_fixture<FABRIC_DEVICE_FIXTURE>(test_fixture, use_galaxy);
    log_info(tt::LogTest, "\tDone");
    auto view = *(test_fixture->view_);

    auto fabrics_under_test_devices = generate_line_fabrics_under_test(
        use_galaxy, use_tg, line_size, topology, view, params.num_fabric_rows, params.num_fabric_cols);

    // Persistent Fabric Setup
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle = std::nullopt;
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs = std::nullopt;
    if (!use_device_init_fabric) {
        std::vector<Program*> fabric_program_ptrs;
        TT_FATAL(
            fabrics_under_test_devices.size() == 1, "Expected 1 fabric under test when device init fabric is not used");
        setup_test_with_persistent_fabric(
            fabrics_under_test_devices[0],
            subdevice_managers,
            fabric_programs,
            fabric_program_ptrs,
            fabric_handle,
            params.num_links,
            topology,
            fabric_context_switch_interval,
            false,
            is_6u_galaxy,
            edm_buffer_config);
    }

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

            auto worker_config = get_fabric_1d_worker_config(
                i,
                devices,
                topology,
                fabric_mode,
                line_size,
                line_index,
                params.senders_are_unidirectional,
                num_devices_with_workers);

            // compute worker based on ethernet cores
            CoreRangeSet worker_cores = {};
            if (use_tg and topology == ttnn::ccl::Topology::Linear) {
                std::vector<CoreCoord> ethernet_cores_virtual = compute_top_row_ethernet_cores(device, worker_config);
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
                        worker_config.forward_device,
                        worker_config.backward_device,
                        &program,
                        params.num_links,
                        topology);
            }

            // reserve CB
            constexpr size_t packet_header_buffer_size = 8192;
            tt_metal::CircularBufferConfig cb_src0_config =
                tt_metal::CircularBufferConfig(packet_header_buffer_size, {{packet_header_cb_index, cb_df}})
                    .set_page_size(packet_header_cb_index, packet_header_buffer_size);
            CBHandle sender_workers_cb = CreateCircularBuffer(program, worker_cores, cb_src0_config);

            tt_metal::CircularBufferConfig cb_src1_config =
                tt_metal::CircularBufferConfig(max_packet_payload_size_bytes, {{source_payload_cb_index, cb_df}})
                    .set_page_size(source_payload_cb_index, max_packet_payload_size_bytes);
            CBHandle sender_workers_payload_cb = CreateCircularBuffer(program, worker_cores, cb_src1_config);

            std::vector<uint32_t> worker_ct_args = {params.line_sync, params.line_sync};

            TT_FATAL(
                std::any_of(
                    worker_cores_vec.begin(),
                    worker_cores_vec.end(),
                    [&sync_core_coord](const CoreCoord& core) {
                        return core.x == sync_core_coord.x && core.y == sync_core_coord.y;
                    }),
                "Atleast one worker core must be mapped onto sync core: x={}, y={}",
                sync_core_coord.x,
                sync_core_coord.y);
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
                                             // not updated to CCL line direction because this is a metal/fabric level
                                             // test
                                             ttnn::ccl::EdmLineFabricOpInterface::Direction direction,
                                             std::vector<uint32_t>& rt_args_out) {
                rt_args_out.push_back(is_connected_in_direction);
                if (is_connected_in_direction) {
                    if (use_device_init_fabric) {
                        const auto& device_fabric_node_id =
                            tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id());
                        const auto& connected_device_fabric_node_id =
                            tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(connected_device->id());
                        tt::tt_fabric::append_fabric_connection_rt_args(
                            device_fabric_node_id,
                            connected_device_fabric_node_id,
                            link,
                            program,
                            {worker_core},
                            rt_args_out);
                    } else {
                        const auto connection = local_device_fabric_handle->uniquely_connect_worker(device, direction);
                        const auto new_rt_args = ttnn::ccl::worker_detail::generate_edm_connection_rt_args(
                            connection, device->id(), program, {worker_core});
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
                        num_fwd_hops_per_type.push_back(worker_config.num_fwd_hops);
                        num_bwd_hops_per_type.push_back(worker_config.num_bwd_hops);
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
                    rt_args.size() + num_send_types * 6 +               // 6 arrays of size num_send_types
                    3 +                                                 // CB indices
                    (worker_config.has_forward_connection ? 10 : 1) +   // Forward connection args
                    (worker_config.has_backward_connection ? 10 : 1) +  // Backward connection args
                    (params.line_sync ? 6 : 0));                        // Line sync args

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
                    worker_config.has_forward_connection,
                    worker_config.forward_device,
                    ttnn::ccl::EdmLineFabricOpInterface::FORWARD,
                    rt_args);
                build_connection_args(
                    worker_core,
                    l,
                    worker_config.has_backward_connection,
                    worker_config.backward_device,
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
                    rt_args.push_back(params.num_links * worker_config.sync_count_per_link);
                    rt_args.push_back(worker_config.sync_num_fwd_hops);
                    rt_args.push_back(worker_config.sync_num_bwd_hops);
                }

                tt_metal::SetRuntimeArgs(program, worker_kernel_id, worker_core, rt_args);
            }
        }
    }

    for (size_t i = 0; i < params.num_op_invocations; i++) {
        log_trace(tt::LogTest, "Iteration: {}", i);
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

        log_trace(tt::LogTest, "Waiting for Op finish on all devices");
        for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices.size(); fabric_index++) {
            auto& worker_devices = fabric_under_test_worker_devices[fabric_index];
            wait_for_worker_program_completion(worker_devices, subdevice_managers);
        }
        log_trace(tt::LogTest, "Main op done");
    }

    if (!use_device_init_fabric) {
        auto& devices = fabrics_under_test_devices[0];
        TT_FATAL(fabric_programs->size() == devices.size(), "Expected fabric programs size to be same as devices size");
        log_trace(tt::LogTest, "Fabric teardown");
        persistent_fabric_teardown_sequence(
            devices,
            subdevice_managers,
            fabric_handle.value(),
            tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
        for (auto& device : devices) {
            device->clear_loaded_sub_device_manager();
        }
    }

    log_trace(tt::LogTest, "Waiting for teardown completion");
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
    log_trace(tt::LogTest, "Finished");
}

struct FullMeshTestParams {
    constexpr static size_t MAX_NUM_AXES = 2;
    std::array<size_t, MAX_NUM_AXES> line_size = {4, 0};
    std::array<size_t, MAX_NUM_AXES> num_devices_with_workers = {0, 0};
    std::array<size_t, MAX_NUM_AXES> num_links = {0, 0};
    std::array<size_t, MAX_NUM_AXES> first_link_offset = {0, 0};
    size_t num_op_invocations = 0;
    bool line_sync = true;
    size_t num_fabric_rows = 0;
    size_t num_fabric_cols = 0;
    std::array<FabricTestMode, MAX_NUM_AXES> fabric_mode = {FabricTestMode::Linear, FabricTestMode::Linear};

    // True if you only want the workers on the end to send
    std::array<bool, MAX_NUM_AXES> disable_sends_for_interior_workers = {false, false};

    std::array<bool, MAX_NUM_AXES> disable_end_workers_in_backward_direction = {false, false};
    std::array<bool, MAX_NUM_AXES> senders_are_unidirectional = {false, false};
};

void validate_fabric_packet_send_test_params(const FullMeshTestParams& full_mesh_params) {
    for (size_t axis = 0; axis < FullMeshTestParams::MAX_NUM_AXES; axis++) {
        TT_FATAL(
            !full_mesh_params.disable_sends_for_interior_workers[axis] ||
                full_mesh_params.fabric_mode[axis] == FabricTestMode::Linear ||
                full_mesh_params.fabric_mode[axis] == FabricTestMode::RingAsLinear,
            "This test can only be run with disable_sends_for_interior_workers set to true or fabric_mode set to "
            "Linear");
        TT_FATAL(
            !full_mesh_params.disable_end_workers_in_backward_direction[axis] ||
                full_mesh_params.fabric_mode[axis] == FabricTestMode::Linear ||
                full_mesh_params.fabric_mode[axis] == FabricTestMode::RingAsLinear,
            "This test can only be run with disable_end_workers_in_backward_direction set to true or fabric_mode set "
            "to "
            "Linear");
        TT_FATAL(
            full_mesh_params.num_devices_with_workers[axis] <= full_mesh_params.line_size[axis],
            "num_devices_with_workers must be less than or equal to line_size");
        TT_FATAL(full_mesh_params.num_links[axis] > 0, "num_links must be greater than 0");
        TT_FATAL(
            full_mesh_params.num_links[axis] <= full_mesh_params.line_size[axis],
            "num_links must be less than or equal to line_size");
        TT_FATAL(
            full_mesh_params.first_link_offset[axis] < full_mesh_params.num_links[axis],
            "first_link_offset must be less than num_links");
        if (full_mesh_params.fabric_mode[axis] == FabricTestMode::SaturateChipToChipRing) {
            TT_FATAL(full_mesh_params.line_size[axis] == 4, "SaturateChipToChipRing only supports line_size 4");
        }
    }
}

void validate_fabric_packet_send_test_params(
    const std::variant<WriteThroughputStabilityTestWithPersistentFabricParams, FullMeshTestParams>& params) {
    if (std::holds_alternative<WriteThroughputStabilityTestWithPersistentFabricParams>(params)) {
        TT_THROW("Not commonized yet");
    } else {
        const auto& full_mesh_params = std::get<FullMeshTestParams>(params);
        validate_fabric_packet_send_test_params(full_mesh_params);
    }
}

ttnn::ccl::Topology get_topology(FabricTestMode fabric_mode) {
    switch (fabric_mode) {
        case FabricTestMode::Linear: return ttnn::ccl::Topology::Linear;
        case FabricTestMode::SaturateChipToChipRing:
        case FabricTestMode::HalfRing:
        case FabricTestMode::FullRing:
        case FabricTestMode::RingAsLinear: return ttnn::ccl::Topology::Ring;
        default: TT_THROW("Invalid fabric mode");
    }
    return ttnn::ccl::Topology::Linear;
}

template <typename T>
using per_axis_array_t = std::array<T, FullMeshTestParams::MAX_NUM_AXES>;

static void validate_sync_core_is_on_a_worker(
    const CoreCoord& sync_core_coord,
    const std::unordered_map<size_t, std::vector<CoreCoord>>& worker_cores_per_device) {
    for (const auto& [device, worker_cores_vec] : worker_cores_per_device) {
        bool sync_core_found = false;
        for (const auto& core : worker_cores_vec) {
            if (core.x == sync_core_coord.x && core.y == sync_core_coord.y) {
                sync_core_found = true;
                break;
            }
        }
        TT_FATAL(
            sync_core_found,
            "Atleast one worker core must be mapped onto sync core: x={}, y={}, device={}",
            sync_core_coord.x,
            sync_core_coord.y,
            device);
    }
}

void launch_kernels_and_wait_for_completion(
    const FullMeshTestParams& params,
    const per_axis_array_t<std::vector<std::vector<IDevice*>>>& fabrics_under_test_devices_per_axis,
    const per_axis_array_t<std::vector<std::vector<KernelHandle>>>& worker_kernel_ids_per_fabric,
    const per_axis_array_t<std::vector<std::vector<size_t>>>& per_fabric_per_device_global_sem_addr_rt_arg,
    std::unordered_map<IDevice*, Program>& device_programs,
    const per_axis_array_t<std::vector<std::vector<CoreCoord>>>& worker_cores_vec_per_axis_per_device,
    const per_axis_array_t<std::vector<tt::tt_metal::DeviceAddr>>& global_semaphore_addrs_per_axis) {
    for (size_t i = 0; i < params.num_op_invocations; i++) {
        log_trace(tt::LogTest, "Iteration: {}", i);
        if (i != 0 && params.line_sync) {
            for (size_t axis = 0; axis < FullMeshTestParams::MAX_NUM_AXES; axis++) {
                for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices_per_axis[axis].size();
                     fabric_index++) {
                    auto& worker_kernel_ids = worker_kernel_ids_per_fabric[axis][fabric_index];
                    auto& per_device_global_sem_addr_rt_arg =
                        per_fabric_per_device_global_sem_addr_rt_arg[axis][fabric_index];
                    for (size_t k = 0; k < worker_kernel_ids.size(); k++) {
                        auto& devices = fabrics_under_test_devices_per_axis[axis][fabric_index];
                        auto& program = device_programs.at(devices.at(k));
                        auto& worker_rt_args_by_core = GetRuntimeArgs(program, worker_kernel_ids[k]);
                        auto global_sem_addr_rt_arg_idx = per_device_global_sem_addr_rt_arg[k];
                        for (size_t l = 0; l < params.num_links[axis]; l++) {
                            auto& worker_rt_args =
                                worker_rt_args_by_core[worker_cores_vec_per_axis_per_device[axis][k][l].x]
                                                      [worker_cores_vec_per_axis_per_device[axis][k][l].y];
                            worker_rt_args.at(global_sem_addr_rt_arg_idx) =
                                global_semaphore_addrs_per_axis[axis][i % global_semaphore_addrs_per_axis[axis].size()];
                        }
                    }
                }
            }
        }

        // Both axes share programs, because we want to run them together, so we only launch once
        std::vector<Program*> program_ptrs;
        std::vector<IDevice*> worker_devices;
        {
            std::set<IDevice*> all_worker_devices_set;
            for (size_t axis = 0; axis < FullMeshTestParams::MAX_NUM_AXES; axis++) {
                for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices_per_axis[axis].size();
                     fabric_index++) {
                    auto& fabric_worker_devices = fabrics_under_test_devices_per_axis[axis][fabric_index];
                    for (auto* device : fabric_worker_devices) {
                        all_worker_devices_set.insert(device);
                    }
                }
            }
            program_ptrs.reserve(all_worker_devices_set.size());
            std::transform(
                all_worker_devices_set.begin(),
                all_worker_devices_set.end(),
                std::back_inserter(program_ptrs),
                [&device_programs](IDevice* d) { return &device_programs.at(d); });
            std::copy(all_worker_devices_set.begin(), all_worker_devices_set.end(), std::back_inserter(worker_devices));
            build_and_enqueue(worker_devices, program_ptrs, i != 0);
        }

        for (size_t axis = 0; axis < FullMeshTestParams::MAX_NUM_AXES; axis++) {
            for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices_per_axis[axis].size();
                 fabric_index++) {
                auto& worker_devices = fabrics_under_test_devices_per_axis[axis][fabric_index];
                std::stringstream ss;
                for (auto* device : worker_devices) {
                    ss << device->id() << " ";
                }
                wait_for_worker_program_completion(worker_devices, std::nullopt);
            }
        }
    }

    TT_FATAL(device_programs.size() > 0, "No devices found");
    for (const auto& [device, program] : device_programs) {
        tt_metal::Synchronize(device, *ttnn::DefaultQueueId);
    }
    for (const auto& [device, program] : device_programs) {
        tt_metal::detail::DumpDeviceProfileResults(device);
    }
}

std::tuple<size_t, per_axis_array_t<std::vector<std::vector<Fabric1DWorkerConfig>>>>
generate_1D_fabric_on_full_mesh_worker_configs(
    const FullMeshTestParams& params,
    const per_axis_array_t<std::vector<std::vector<IDevice*>>>& fabrics_under_test_devices_per_axis,
    const per_axis_array_t<ttnn::ccl::Topology>& topologies,
    const std::array<FabricTestMode, FullMeshTestParams::MAX_NUM_AXES>& fabric_modes) {
    per_axis_array_t<std::vector<std::vector<Fabric1DWorkerConfig>>> worker_configs_per_axis_per_fabric_per_device;
    size_t sync_count = 0;

    for (size_t axis = 0; axis < FullMeshTestParams::MAX_NUM_AXES; axis++) {
        auto line_size = params.line_size[axis];
        auto num_devices_with_workers = params.num_devices_with_workers[axis];
        if (num_devices_with_workers == 0) {
            num_devices_with_workers = line_size;
        }
        auto senders_are_unidirectional = params.senders_are_unidirectional[axis];

        worker_configs_per_axis_per_fabric_per_device[axis].resize(fabrics_under_test_devices_per_axis[axis].size());
        std::optional<size_t> sync_count_per_axis = std::nullopt;
        for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices_per_axis[axis].size(); fabric_index++) {
            worker_configs_per_axis_per_fabric_per_device[axis][fabric_index].resize(num_devices_with_workers);
            auto& devices = fabrics_under_test_devices_per_axis[axis][fabric_index];
            for (size_t i = 0; i < num_devices_with_workers; i++) {
                const size_t line_index = i;

                auto worker_config = get_fabric_1d_worker_config(
                    i,
                    devices,
                    topologies[axis],
                    fabric_modes[axis],
                    params.line_size[axis],
                    line_index,
                    senders_are_unidirectional,
                    num_devices_with_workers);
                worker_configs_per_axis_per_fabric_per_device[axis][fabric_index][i] = worker_config;
                if (!sync_count_per_axis) {
                    sync_count_per_axis = worker_config.sync_count_per_link;
                } else {
                    TT_FATAL(
                        sync_count_per_axis.value() == worker_config.sync_count_per_link,
                        "Sync count per axis must be the same for all fabrics");
                }
            }
        }
        TT_FATAL(sync_count_per_axis, "Sync count per axis must be set");
        sync_count += params.num_links[axis] * sync_count_per_axis.value();
    }

    return std::make_tuple(sync_count, worker_configs_per_axis_per_fabric_per_device);
}

void generate_1d_fabric_on_full_mesh_worker_rt_args(
    const FullMeshTestParams& params,
    const per_axis_array_t<std::vector<std::vector<Fabric1DWorkerConfig>>>&
        worker_configs_per_axis_per_fabric_per_device,
    const per_axis_array_t<std::vector<std::vector<CoreCoord>>>& worker_cores_vec_per_axis_per_device,
    const per_axis_array_t<std::vector<tt::tt_metal::DeviceAddr>>& global_semaphore_addrs_per_axis,
    const std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore>& global_semaphore_handles_per_axis,
    size_t axis,
    size_t i,
    size_t line_size,
    const Fabric1DPacketSendTestSpec& test_specs,
    const Fabric1DWorkerConfig& worker_config,
    const KernelHandle& worker_kernel_id,
    const std::vector<CoreCoord>& worker_cores_vec,
    IDevice* device,
    size_t num_messages,
    size_t dest_bank_addr,
    const std::vector<CoreCoord>& dest_core_coord,
    size_t source_payload_cb_index,
    size_t packet_header_cb_index,
    size_t packet_header_cb_size_in_headers,
    const CoreCoord& sync_noc_core_coord,
    size_t sync_count,
    Program& program,
    std::vector<size_t>& per_device_global_sem_addr_rt_arg) {
    auto build_connection_args = [device, &program](
                                     CoreCoord& worker_core,
                                     size_t link,
                                     bool is_connected_in_direction,
                                     IDevice* connected_device,
                                     ttnn::ccl::EdmLineFabricOpInterface::Direction direction,
                                     std::vector<uint32_t>& rt_args_out) {
        rt_args_out.push_back(is_connected_in_direction);
        if (is_connected_in_direction) {
            const auto& device_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id());
            const auto& connected_device_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(connected_device->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                device_fabric_node_id, connected_device_fabric_node_id, link, program, {worker_core}, rt_args_out);
        }
    };

    for (size_t l = 0; l < params.num_links[axis]; l++) {
        auto worker_core = worker_cores_vec[l];

        // RT ARGS
        bool disable_sends_for_worker =
            (params.disable_sends_for_interior_workers[axis] && (i != 0) && (i != line_size - 1)) ||
            (params.disable_end_workers_in_backward_direction[axis] && (i == line_size - 1));

        // New format for send types
        std::vector<uint32_t> send_types;
        std::vector<uint32_t> chip_send_types;
        std::vector<uint32_t> send_counts_per_type;
        std::vector<uint32_t> num_fwd_hops_per_type;
        std::vector<uint32_t> num_bwd_hops_per_type;
        std::vector<uint32_t> send_type_payload_sizes;
        std::vector<bool> flush_send;
        if (!disable_sends_for_worker) {
            send_types.push_back(static_cast<size_t>(test_specs.noc_send_type));
            chip_send_types.push_back(static_cast<size_t>(test_specs.chip_send_type));
            send_counts_per_type.push_back(num_messages);
            num_fwd_hops_per_type.push_back(worker_config.num_fwd_hops);
            num_bwd_hops_per_type.push_back(worker_config.num_bwd_hops);
            send_type_payload_sizes.push_back(test_specs.packet_payload_size_bytes);
            flush_send.push_back(test_specs.flush);
        }

        // Get forward and backward destination coordinates
        const size_t dest_noc_x_fwd = device->worker_core_from_logical_core(dest_core_coord[l]).x;
        const size_t dest_noc_y_fwd = device->worker_core_from_logical_core(dest_core_coord[l]).y;
        const size_t dest_noc_x_bwd = device->worker_core_from_logical_core(dest_core_coord[l]).x;
        const size_t dest_noc_y_bwd = device->worker_core_from_logical_core(dest_core_coord[l]).y;
        size_t num_send_types = !disable_sends_for_worker;
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
            rt_args.size() + num_send_types * 6 +               // 6 arrays of size num_send_types
            3 +                                                 // CB indices
            (worker_config.has_forward_connection ? 10 : 1) +   // Forward connection args
            (worker_config.has_backward_connection ? 10 : 1) +  // Backward connection args
            (params.line_sync ? 6 : 0));                        // Line sync args

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
            worker_config.has_forward_connection,
            worker_config.forward_device,
            ttnn::ccl::EdmLineFabricOpInterface::FORWARD,
            rt_args);
        build_connection_args(
            worker_core,
            l,
            worker_config.has_backward_connection,
            worker_config.backward_device,
            ttnn::ccl::EdmLineFabricOpInterface::BACKWARD,
            rt_args);

        if (params.line_sync) {
            rt_args.push_back(sync_noc_core_coord.x);
            rt_args.push_back(sync_noc_core_coord.y);
            if (l == 0) {
                per_device_global_sem_addr_rt_arg.push_back(rt_args.size());
            }
            TT_FATAL(
                global_semaphore_addrs_per_axis[0].at(0) != -1, "Invalid test setup. Global semaphore address is -1");
            rt_args.push_back(global_semaphore_addrs_per_axis[0].at(0));
            rt_args.push_back(sync_count);
            rt_args.push_back(worker_config.sync_num_fwd_hops);
            rt_args.push_back(worker_config.sync_num_bwd_hops);
        }

        tt_metal::SetRuntimeArgs(program, worker_kernel_id, worker_core, rt_args);
    }
}

std::vector<CoreCoord> setup_worker_core_coords(
    const FullMeshTestParams& params,
    size_t axis,
    IDevice* device,
    per_axis_array_t<CoreRangeSet>& worker_cores_per_axis,
    std::unordered_map<size_t, std::vector<CoreCoord>>& worker_cores_per_device,
    per_axis_array_t<std::vector<std::vector<CoreCoord>>>& worker_cores_vec_per_axis_per_device) {
    worker_cores_per_axis[axis] = {};
    constexpr size_t OFFSET_PER_AXIS = 4;
    worker_cores_per_axis[axis] = CoreRangeSet(CoreRange(
        CoreCoord(params.first_link_offset[axis], axis * OFFSET_PER_AXIS),
        CoreCoord(params.num_links[axis] - 1, axis * OFFSET_PER_AXIS)));
    auto worker_cores_vec = corerange_to_cores(worker_cores_per_axis[axis], std::nullopt, false);
    std::for_each(worker_cores_vec.begin(), worker_cores_vec.end(), [&](const CoreCoord& core) {
        worker_cores_per_device[device->id()].push_back(core);
    });
    worker_cores_vec_per_axis_per_device[axis].push_back(worker_cores_vec);

    return worker_cores_vec;
}

template <typename FABRIC_DEVICE_FIXTURE = Fabric1DFixture>
void Run1DFullMeshFabricPacketSendTest(
    std::unique_ptr<Fabric1DFixture>& test_fixture_,
    const Fabric1DPacketSendTestSpec& test_specs,
    const FullMeshTestParams& params = {},
    size_t fabric_context_switch_interval =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_firmware_context_switch_interval) {
    log_info(tt::LogTest, "Running 1D Full Mesh Fabric Packet Send Test");
    using namespace ttnn::ccl;
    static constexpr size_t packet_header_cb_size_in_headers = 5;
    static constexpr tt::DataFormat cb_df = tt::DataFormat::Bfp8;
    constexpr size_t MAX_NUM_AXES = FullMeshTestParams::MAX_NUM_AXES;

    auto max_packet_payload_size_bytes = test_specs.packet_payload_size_bytes;

    auto num_devices = tt::tt_metal::GetNumAvailableDevices();

    validate_fabric_packet_send_test_params(params);

    per_axis_array_t<ttnn::ccl::Topology> topologies;
    std::array<FabricTestMode, MAX_NUM_AXES> fabric_modes = params.fabric_mode;
    for (size_t axis = 0; axis < MAX_NUM_AXES; axis++) {
        topologies[axis] = get_topology(fabric_modes[axis]);
    }

    auto worker_core_logical = [](size_t link, size_t axis) { return CoreCoord(link, 4 * axis); };

    // static constexpr size_t source_l1_buffer_address = 1000000;
    size_t dest_buffer_size = max_packet_payload_size_bytes * 4;

    const bool use_galaxy = num_devices == 32;
    const bool use_tg = use_galaxy && tt::tt_metal::GetNumPCIeDevices() == 4;
    const bool is_6u_galaxy = use_galaxy && tt::tt_metal::GetNumPCIeDevices() == 32;

    create_fabric_fixture<FABRIC_DEVICE_FIXTURE>(test_fixture_, use_galaxy);
    auto view = *(test_fixture_->view_);
    // FABRIC_DEVICE_FIXTURE test_fixture;
    // auto view = *(test_fixture.view_);

    per_axis_array_t<std::vector<std::vector<IDevice*>>> fabrics_under_test_devices_per_axis;
    for (size_t axis = 0; axis < MAX_NUM_AXES; axis++) {
        size_t num_fabric_rows = axis == 0 ? params.num_fabric_rows : 0;
        size_t num_fabric_cols = axis == 1 ? params.num_fabric_cols : 0;
        fabrics_under_test_devices_per_axis[axis] = generate_line_fabrics_under_test(
            use_galaxy, use_tg, params.line_size[axis], topologies[axis], view, num_fabric_rows, num_fabric_cols);
    }

    size_t packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    TT_FATAL(packet_header_size_bytes != 0, "Error in initializing local variable `packet_header_size_bytes`");

    // Big boiler plate setup loop
    CoreCoord sync_core_coord = worker_core_logical(0, 0);
    per_axis_array_t<std::vector<std::vector<CoreCoord>>> worker_cores_vec_per_axis_per_device;
    std::unordered_map<size_t, std::vector<CoreCoord>> worker_cores_per_device;
    ttnn::SmallVector<std::shared_ptr<Buffer>> device_dest_buffers;
    std::unordered_map<IDevice*, Program> device_programs;
    per_axis_array_t<std::vector<CoreCoord>> dest_core_coord_per_axis;
    per_axis_array_t<std::vector<tt::tt_metal::DeviceAddr>> global_semaphore_addrs_per_axis;
    std::vector<ttnn::global_semaphore::MultiDeviceGlobalSemaphore> global_semaphore_handles_per_axis;
    per_axis_array_t<std::vector<std::vector<KernelHandle>>> worker_kernel_ids_per_fabric;
    per_axis_array_t<std::vector<std::vector<size_t>>> per_fabric_per_device_global_sem_addr_rt_arg;
    device_dest_buffers.reserve(params.line_size[0]);
    // Initialization logic for the above datastructures.
    for (size_t axis = 0; axis < MAX_NUM_AXES; axis++) {
        dest_core_coord_per_axis[axis].reserve(params.num_links[axis]);
        for (size_t l = 0; l < params.num_links[axis]; l++) {
            dest_core_coord_per_axis[axis][l] = CoreCoord(axis * 4, l + 1);
        }

        // Don't need to allocate unique per axis because each axis sends to different workers
        for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices_per_axis[axis].size(); fabric_index++) {
            auto& devices = fabrics_under_test_devices_per_axis[axis][fabric_index];
            for (auto* d : devices) {
                if (device_programs.find(d) == device_programs.end()) {
                    device_programs.insert({d, Program()});
                }
                auto local_input_buffer =
                    CreateBuffer(InterleavedBufferConfig{d, dest_buffer_size, dest_buffer_size, BufferType::L1});
                device_dest_buffers.push_back(local_input_buffer);
            }
        }

        global_semaphore_addrs_per_axis[axis].reserve(params.line_size[axis] + 1);
        auto global_semaphores = ttnn::global_semaphore::create_global_semaphore_with_same_address(
            test_fixture_->view_.get()->get_devices(),
            fabrics_under_test_devices_per_axis[axis][0][0]->worker_cores(
                HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
            0,                             // initial value
            tt::tt_metal::BufferType::L1,  // buffer type,
            1000                           // attempts
        );
        global_semaphore_handles_per_axis.push_back(global_semaphores);
        auto global_semaphore_addr =
            ttnn::global_semaphore::get_global_semaphore_address(global_semaphores.global_semaphores.at(0));
        global_semaphore_addrs_per_axis[axis].push_back(global_semaphore_addr);

        worker_kernel_ids_per_fabric[axis].resize(fabrics_under_test_devices_per_axis[axis].size());
        per_fabric_per_device_global_sem_addr_rt_arg[axis].resize(fabrics_under_test_devices_per_axis[axis].size());
    }

    size_t dest_bank_addr = device_dest_buffers[0]->address();
    per_axis_array_t<CoreRangeSet> worker_cores_per_axis;
    const size_t max_line_size = *std::max_element(params.line_size.begin(), params.line_size.end());

    per_axis_array_t<std::vector<std::vector<Fabric1DWorkerConfig>>> worker_configs_per_axis_per_fabric_per_device;
    size_t sync_count;
    std::tie(sync_count, worker_configs_per_axis_per_fabric_per_device) =
        generate_1D_fabric_on_full_mesh_worker_configs(
            params, fabrics_under_test_devices_per_axis, topologies, fabric_modes);

    for (size_t axis = 0; axis < MAX_NUM_AXES; axis++) {
        const uint32_t packet_header_cb_index = axis == 0 ? tt::CB::c_in0 : tt::CB::c_in2;
        const uint32_t source_payload_cb_index = axis == 0 ? tt::CB::c_in1 : tt::CB::c_in3;
        auto line_size = params.line_size[axis];
        auto& dest_core_coord = dest_core_coord_per_axis[axis];
        auto num_devices_with_workers = params.num_devices_with_workers[axis];
        if (num_devices_with_workers == 0) {
            num_devices_with_workers = line_size;
        }
        size_t num_messages = (test_specs.num_messages * line_size) / max_line_size;

        for (size_t fabric_index = 0; fabric_index < fabrics_under_test_devices_per_axis[axis].size(); fabric_index++) {
            auto& devices = fabrics_under_test_devices_per_axis[axis][fabric_index];

            auto& per_device_global_sem_addr_rt_arg = per_fabric_per_device_global_sem_addr_rt_arg[axis][fabric_index];
            auto& worker_kernel_ids = worker_kernel_ids_per_fabric[axis][fabric_index];
            for (size_t i = 0; i < num_devices_with_workers; i++) {
                auto* device = devices[i];
                auto& program = device_programs.at(device);

                auto worker_cores_vec = setup_worker_core_coords(
                    params,
                    axis,
                    device,
                    worker_cores_per_axis,
                    worker_cores_per_device,
                    worker_cores_vec_per_axis_per_device);

                // reserve CB
                tt_metal::CircularBufferConfig cb_src0_config =
                    tt_metal::CircularBufferConfig(
                        packet_header_cb_size_in_headers * packet_header_size_bytes, {{packet_header_cb_index, cb_df}})
                        .set_page_size(packet_header_cb_index, packet_header_size_bytes);
                CBHandle sender_workers_cb = CreateCircularBuffer(program, worker_cores_per_axis[axis], cb_src0_config);

                tt_metal::CircularBufferConfig cb_src1_config =
                    tt_metal::CircularBufferConfig(max_packet_payload_size_bytes, {{source_payload_cb_index, cb_df}})
                        .set_page_size(source_payload_cb_index, max_packet_payload_size_bytes);
                CBHandle sender_workers_payload_cb =
                    CreateCircularBuffer(program, worker_cores_per_axis[axis], cb_src1_config);

                auto worker_kernel_id = tt_metal::CreateKernel(
                    program,
                    "tests/ttnn/unit_tests/gtests/ccl/kernels/edm_fabric_writer.cpp",
                    worker_cores_per_axis[axis],
                    tt_metal::DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt_metal::NOC::NOC_0,
                        .compile_args = {params.line_sync, params.line_sync}});
                worker_kernel_ids.push_back(worker_kernel_id);

                auto& worker_config = worker_configs_per_axis_per_fabric_per_device.at(axis).at(fabric_index).at(i);
                generate_1d_fabric_on_full_mesh_worker_rt_args(
                    params,
                    worker_configs_per_axis_per_fabric_per_device,
                    worker_cores_vec_per_axis_per_device,
                    global_semaphore_addrs_per_axis,
                    global_semaphore_handles_per_axis,
                    axis,
                    i,
                    line_size,
                    test_specs,
                    worker_config,
                    worker_kernel_id,
                    worker_cores_vec,
                    device,
                    num_messages,
                    dest_bank_addr,
                    dest_core_coord,
                    source_payload_cb_index,
                    packet_header_cb_index,
                    packet_header_cb_size_in_headers,
                    device->worker_core_from_logical_core(sync_core_coord),
                    sync_count,
                    program,
                    per_device_global_sem_addr_rt_arg);
            }
        }
    }

    validate_sync_core_is_on_a_worker(sync_core_coord, worker_cores_per_device);

    launch_kernels_and_wait_for_completion(
        params,
        fabrics_under_test_devices_per_axis,
        worker_kernel_ids_per_fabric,
        per_fabric_per_device_global_sem_addr_rt_arg,
        device_programs,
        worker_cores_vec_per_axis_per_device,
        global_semaphore_addrs_per_axis);
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
    std::unique_ptr<Fabric1DFixture> test_fixture = nullptr;
    Run1DFabricPacketSendTest(test_fixture, test_specs, params_copy, 0);
}

size_t get_number_of_links_for_ring_deadlock_stability_test(
    const MeshDevice& mesh_device,
    ClusterType cluster_type,
    std::optional<size_t> num_links_opt,
    size_t num_devices,
    std::optional<size_t> row_or_col) {
    size_t num_links = num_links_opt.value_or(0);
    if (num_links == 0) {
        if (cluster_type == ClusterType::GALAXY) {
            for (size_t i = 0; i < mesh_device.shape()[0]; i++) {
                size_t cluster_axis = 1;
                auto nl =
                    tt::tt_fabric::experimental::get_number_of_available_routing_planes(mesh_device, cluster_axis, i);
                log_debug(tt::LogTest, "Number of links for Galaxy cluster_axis 0, row {}: {}", i, nl);
            }
            for (size_t i = 0; i < mesh_device.shape()[1]; i++) {
                size_t cluster_axis = 0;
                auto nl =
                    tt::tt_fabric::experimental::get_number_of_available_routing_planes(mesh_device, cluster_axis, i);
                log_debug(tt::LogTest, "Number of links for Galaxy cluster_axis 1, row {}: {}", i, nl);
            }

            bool is_long_edge = num_devices == 8;
            size_t cluster_axis = !is_long_edge ? 1 : 0;
            num_links = tt::tt_fabric::experimental::get_number_of_available_routing_planes(
                mesh_device, cluster_axis, row_or_col.value());
        } else {
            TT_THROW("This test can only be run on 4 or 8 chips on Galaxy devices");
        }
    }

    return num_links;
}

std::vector<IDevice*> get_devices_for_ring_deadlock_stability_test(
    const MeshDeviceView& view, ClusterType cluster_type, size_t num_devices, std::optional<size_t> row_or_col) {
    std::vector<IDevice*> devices_;
    log_debug(tt::LogTest, "Getting devices for ring deadlock stability test. Row or col: {}", row_or_col.value_or(0));

    if (cluster_type == ClusterType::GALAXY) {
        if (num_devices == 4) {
            log_debug(
                tt::LogTest,
                "Getting 4 devices for ring deadlock stability test. Row or col: {}",
                row_or_col.value_or(0));
            devices_ = {
                view.get_device(MeshCoordinate(row_or_col.value_or(0), 0)),
                view.get_device(MeshCoordinate(row_or_col.value_or(0), 1)),
                view.get_device(MeshCoordinate(row_or_col.value_or(0), 2)),
                view.get_device(MeshCoordinate(row_or_col.value_or(0), 3))};
        } else if (num_devices == 8) {
            log_debug(
                tt::LogTest,
                "Getting 8 devices for ring deadlock stability test. Row or col: {}",
                row_or_col.value_or(0));
            devices_ = {
                view.get_device(MeshCoordinate(0, row_or_col.value_or(0))),
                view.get_device(MeshCoordinate(1, row_or_col.value_or(0))),
                view.get_device(MeshCoordinate(2, row_or_col.value_or(0))),
                view.get_device(MeshCoordinate(3, row_or_col.value_or(0))),
                view.get_device(MeshCoordinate(4, row_or_col.value_or(0))),
                view.get_device(MeshCoordinate(5, row_or_col.value_or(0))),
                view.get_device(MeshCoordinate(6, row_or_col.value_or(0))),
                view.get_device(MeshCoordinate(7, row_or_col.value_or(0)))};
        }
    } else {
        log_debug(
            tt::LogTest, "Getting 8 devices for ring deadlock stability test. Row or col: {}", row_or_col.value_or(0));
        TT_FATAL(!row_or_col.has_value(), "Row or col must be provided for T3000 devices");
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
    return devices_;
}

template <typename DeviceInitFixture = Fabric1DRingStrictDeviceInitFixture>
void RunRingDeadlockStabilityTestWithPersistentFabric(
    size_t num_mcasts,
    std::optional<size_t> num_links_opt,
    size_t num_devices,
    size_t num_op_invocations,
    bool has_forward_connection,
    bool has_backward_connection,
    size_t packet_payload_size_bytes = tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes,
    std::optional<size_t> row_or_col = std::nullopt) {
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
    switch (cluster_type) {
        case ClusterType::T3K:
            if (num_devices != 8) {
                log_debug(tt::LogTest, "This test can only be run 8 chips on T3000 devices");
                return;
            }
            break;
        case ClusterType::GALAXY:
            if (num_devices != 4 && num_devices != 8) {
                log_debug(tt::LogTest, "This test can only be run on 4 or 8 chips on Galaxy devices");
                return;
            }
            break;
        default: log_debug(tt::LogTest, "This test can only be run on T3000 or Galaxy devices"); return;
    }

    using namespace ttnn::ccl;
    size_t line_size = num_devices;
    size_t num_devices_with_workers = line_size;
    constexpr bool line_sync = false;

    auto worker_core_logical = [](size_t link) { return CoreCoord(link, 0); };

    // static constexpr size_t source_l1_buffer_address = 1000000;
    static constexpr uint32_t packet_header_cb_index = tt::CB::c_in0;
    static constexpr uint32_t source_payload_cb_index = tt::CB::c_in1;
    static constexpr size_t packet_header_cb_size_in_headers = 5;
    size_t dest_buffer_size = packet_payload_size_bytes * 4;
    static constexpr tt::DataFormat cb_df = tt::DataFormat::Bfp8;

    DeviceInitFixture test_fixture;
    auto view = test_fixture.mesh_device_->get_view();

    std::vector<IDevice*> devices_ =
        get_devices_for_ring_deadlock_stability_test(view, cluster_type, num_devices, row_or_col);
    size_t num_links = get_number_of_links_for_ring_deadlock_stability_test(
        *test_fixture.mesh_device_.get(), cluster_type, num_links_opt, num_devices, row_or_col);

    std::vector<IDevice*> devices;
    devices.reserve(line_size);
    for (size_t i = 0; i < line_size; i++) {
        devices.push_back(devices_[i]);
    }

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
        TT_FATAL(d != nullptr, "Device is nullptr");
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
        auto global_semaphores = ttnn::global_semaphore::create_global_semaphore(
            devices_,
            devices_[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
            0,                              // initial value
            tt::tt_metal::BufferType::L1);  //,  // buffer type
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
    log_debug(tt::LogTest, "Initializing worker programs");
    for (size_t i = 0; i < num_devices_with_workers; i++) {
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
        constexpr size_t packet_header_buffer_size = 8192;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(packet_header_buffer_size, {{packet_header_cb_index, cb_df}})
                .set_page_size(packet_header_cb_index, packet_header_buffer_size);
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
                        const auto& device_fabric_node_id =
                            tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id());
                        const auto& connected_device_fabric_node_id =
                            tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(connected_device->id());
                        tt::tt_fabric::append_fabric_connection_rt_args(
                            device_fabric_node_id,
                            connected_device_fabric_node_id,
                            l,
                            program,
                            {worker_core},
                            rt_args_out);
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
        log_trace(tt::LogTest, "Iteration: {}", i);
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

        log_trace(tt::LogTest, "Waiting for Op finish on all devices");
        for (IDevice* d : devices) {
            tt_metal::Synchronize(d, *ttnn::DefaultQueueId);
        }
        log_trace(tt::LogTest, "Main op done");
    }

    log_trace(tt::LogTest, "Finished");
}
