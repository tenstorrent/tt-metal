// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/kernel.hpp>
#include "erisc_datamover_builder.hpp"
#include "tt-metalium/kernel_types.hpp"
#include <tt-metalium/fabric.hpp>
#include "erisc_datamover_builder_helper.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/common/executor.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "tt_metal/fabric/erisc_datamover_builder_helper.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_stl/small_vector.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

#include "umd/device/types/arch.h"
#include "umd/device/types/cluster_descriptor_types.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <limits>

#include <tt_metal/fabric/ccl/ccl_common.hpp>

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
        } else if (num_devices_ >= 8) {
            return MeshShape{2, 4};
        } else if (num_devices_ == 2) {
            return MeshShape{1, 2};
        } else {
            TT_THROW("Invalid number of devices: {}", num_devices_);
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

        if (!(num_devices_ >= 2 ||
              (tt::tt_metal::GetNumPCIeDevices() == 4 || tt::tt_metal::GetNumPCIeDevices() == GALAXY_6U_NUM_DEVICES))) {
            TT_THROW("This suite can only be run on 2+ device systems");
        }
    }

public:
    BaseFabricFixture() : device_open(false) {}

    BaseFabricFixture(tt::tt_metal::FabricConfig fabric_config, tt::tt_metal::FabricReliabilityMode reliability_mode = tt::tt_metal::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) : device_open(false) {
        tt::tt_metal::detail::SetFabricConfig(fabric_config, reliability_mode);
    }

    virtual ~BaseFabricFixture() { tt::tt_metal::detail::SetFabricConfig(tt::tt_metal::FabricConfig::DISABLED); }

    virtual void SetupDevices() = 0;
    virtual void TearDown() = 0;
};

class Fabric1DFixture : public BaseFabricFixture {
public:
    std::shared_ptr<MeshDevice> mesh_device_;

    void SetupDevices() override {
        ValidateEnvironment();
        const MeshShape cluster_shape = GetDeterminedMeshShape();
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(cluster_shape));
        device_open = true;
    }

    void TearDown() override {
        if (device_open) {
            tt::tt_metal::CloseDevice(mesh_device_.get());
            device_open = false;
        }
    }

    Fabric1DFixture() : BaseFabricFixture() { this->SetupDevices(); }

    Fabric1DFixture(
        tt::tt_metal::FabricConfig fabric_config,
        tt::tt_metal::FabricReliabilityMode reliability_mode =
            tt::tt_metal::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) :
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
    MeshShape GetDeterminedMeshShape() const { return SystemMesh::instance().get_shape(); }

    // Validates environment and hardware for tests
    void ValidateEnvironment() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run without TT_METAL_SLOW_DISPATCH_MODE set");
        }

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();

        if (num_devices_ < 2) {
            TT_THROW("This suite can only be run on 2+ device systems");
        }

        // if (!(arch_ == tt::ARCH::WORMHOLE_B0 && num_devices_ >= 8 &&
        //       (tt::tt_metal::GetNumPCIeDevices() == 4 || tt::tt_metal::GetNumPCIeDevices() ==
        //       GALAXY_6U_NUM_DEVICES))) {
        //     TT_THROW("This suite can only be run on T3000 or TG Wormhole devices");
        // }
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
        tt::tt_metal::FabricConfig fabric_config,
        tt::tt_metal::FabricReliabilityMode reliability_mode =
            tt::tt_metal::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) :
        device_open(false) {
        tt::tt_metal::detail::SetFabricConfig(fabric_config, reliability_mode);
        this->SetupDevices();
    }

    ~Fabric1DDeviceInitFixture() {
        TearDown();
        tt::tt_metal::detail::SetFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
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
        tt::tt_metal::FabricConfig fabric_config,
        tt::tt_metal::FabricReliabilityMode reliability_mode =
            tt::tt_metal::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) :
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
    Fabric1DLineDeviceInitFixture() : Fabric1DFixture(tt::tt_metal::FabricConfig::FABRIC_1D) {}
};

class Fabric1DRingDeviceInitFixture : public Fabric1DFixture {
public:
    Fabric1DRingDeviceInitFixture() : Fabric1DFixture(tt::tt_metal::FabricConfig::FABRIC_1D_RING) {}
};
class Fabric1DRingStrictDeviceInitFixture : public Fabric1DDeviceInitFixture {
public:
    Fabric1DRingStrictDeviceInitFixture() :
        Fabric1DDeviceInitFixture(
            tt::tt_metal::FabricConfig::FABRIC_1D_RING,
            tt::tt_metal::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) {}
};
class Fabric1DRingRelaxedDeviceInitFixture : public Fabric1DDeviceInitFixture {
public:
    Fabric1DRingRelaxedDeviceInitFixture() :
        Fabric1DDeviceInitFixture(
            tt::tt_metal::FabricConfig::FABRIC_1D_RING,
            tt::tt_metal::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE) {}
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
        const auto& tensix_sub_device = tt::tt_metal::SubDevice(std::array{
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::SubDeviceId{0})});
        const auto& eth_sub_device = tt_metal::SubDevice(std::array{
            CoreRangeSet(),
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::SubDeviceId{0})});
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


static Correctness run_output_check(
    const std::vector<uint32_t>& all_zeros,
    const std::vector<uint32_t>& inputs,
    std::shared_ptr<Buffer>& output_buffer) {
    constexpr bool debug_mode = true;
    std::vector<uint32_t> readback_data_vec(all_zeros.size(), 0);  // init to 0 data for easier debug

    tt_metal::detail::ReadFromBuffer(output_buffer, readback_data_vec);
    return run_output_check(inputs, readback_data_vec);
};

static void run_programs(std::vector<Program>& programs, const std::vector<IDevice*>& devices) {
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

static std::tuple<std::shared_ptr<Buffer>, std::vector<uint32_t>> build_input_buffer(
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
static void generate_fabric_test_kernels(
    Program& sender_program,
    IDevice* sender_device,
    const CoreCoord& worker_core,
    const uint32_t fabric_connection_link_index,
    const mode_variant_t& mode,
    uint32_t page_size,
    uint32_t num_pages_total,
    uint32_t num_pages_per_edm_buffer,
    // uint32_t local_worker_fabric_semaphore_id,
    // uint32_t local_worker_teardown_semaphore_id,
    // uint32_t local_worker_last_message_semaphore_id,
    uint32_t dram_input_buffer_base_addr,
    bool src_is_dram,
    uint32_t dram_output_buffer_base_addr,
    bool dest_is_dram,
    // uint32_t worker_buffer_index_semaphore_id,
    uint32_t packet_header_buffer_cb_id,
    Program& receiver_program,
    IDevice* receiver_device,
    const CoreCoord& receiver_worker_core,
    bool scatter_write) {
    using tt::tt_fabric::FabricNodeId;
    using tt::tt_fabric::MeshId;
    // Create global semaphore and receiver kernel if needed
    uint32_t receiver_noc_x = 0;
    uint32_t receiver_noc_y = 0;
    TT_FATAL(receiver_device != nullptr, "receiver_device must not be null");

    // Create global semaphore on receiver device
    auto global_semaphore = tt::tt_metal::CreateGlobalSemaphore(
        receiver_device,
        receiver_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::SubDeviceId{0}),
        0,  // initial value
        tt::tt_metal::BufferType::L1);
    uint32_t global_completion_semaphore_addr = global_semaphore.address();

    // Calculate receiver NOC coordinates
    auto receiver_noc_coord = receiver_device->worker_core_from_logical_core(receiver_worker_core);
    receiver_noc_x = receiver_noc_coord.x;
    receiver_noc_y = receiver_noc_coord.y;

    // const auto& edm_noc_core = CoreCoord(worker_fabric_connection.edm_noc_x, worker_fabric_connection.edm_noc_y);
    std::vector<uint32_t> sender_worker_reader_compile_args{
        src_is_dram,      //
        num_pages_total,  //
        page_size,
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
        num_pages_total, page_size, dest_is_dram, std::holds_alternative<mcast_send>(mode) ? 1 : 0, scatter_write};
    std::vector<uint32_t> sender_worker_writer_runtime_args{
        dram_output_buffer_base_addr, global_completion_semaphore_addr, packet_header_buffer_cb_id};

    tt::tt_fabric::append_fabric_connection_rt_args(
        FabricNodeId(MeshId{0}, 0),
        FabricNodeId(MeshId{0}, 1),
        fabric_connection_link_index,
        sender_program,
        worker_core,
        sender_worker_writer_runtime_args);

    if (std::holds_alternative<mcast_send>(mode)) {
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).distance);
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).range);
    } else {
        sender_worker_writer_runtime_args.push_back(std::get<unicast_send>(mode).distance);
    }

    // Add receiver NOC coordinates
    sender_worker_writer_runtime_args.push_back(receiver_noc_x);
    sender_worker_writer_runtime_args.push_back(receiver_noc_y);

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
    CBHandle sender_workers_cb = CreateCircularBuffer(sender_program, worker_core, cb_src0_config);
    auto sender_worker_reader_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_erisc_datamover_sender_worker_reader.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_worker_reader_compile_args});
    auto sender_worker_writer_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_erisc_datamover_sender_worker_sender.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = sender_worker_writer_compile_args});
    tt_metal::SetRuntimeArgs(
        sender_program, sender_worker_reader_kernel, worker_core, sender_worker_reader_runtime_args);
    tt_metal::SetRuntimeArgs(
        sender_program, sender_worker_writer_kernel, worker_core, sender_worker_writer_runtime_args);

    // Create receiver kernel if needed (for both unicast and multicast)

    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_erisc_datamover_receiver_worker_signal_wait.cpp",
        receiver_worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    std::vector<uint32_t> receiver_rt_args = {
        global_completion_semaphore_addr,  // global semaphore address
        1                                  // expected number of signals
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_worker_core, receiver_rt_args);
}

static bool RunLoopbackTest(
    tt_metal::IDevice* sender_device,
    tt_metal::IDevice* receiver_device,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram,
    std::vector<Program>& programs,
    bool scatter_write) {
    auto& sender_program = programs.at(0);
    auto& receiver_program = programs.at(1);
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    std::vector<CoreCoord> worker_cores = {CoreCoord(0, 0)};

    // Receiver kernel core on different device
    CoreCoord receiver_worker_core = {0, 1};  // Different core from sender

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
    // Output buffer is now on the receiver device
    auto output_buffer = CreateBuffer(InterleavedBufferConfig{
        receiver_device, test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type});

    uint32_t packet_header_buffer_cb_id = tt::CBIndex::c_1;
    // allocate a circular buffer of size 8k
    constexpr size_t packet_header_buffer_size = 8192;
    tt_metal::CircularBufferConfig packet_header_buffer_config =
        tt_metal::CircularBufferConfig(
            packet_header_buffer_size, {{packet_header_buffer_cb_id, tt::DataFormat::Float16}})
            .set_page_size(packet_header_buffer_cb_id, page_size);
    auto packet_header_buffer =
        tt_metal::CreateCircularBuffer(sender_program, worker_cores.at(0), packet_header_buffer_config);
    tt_metal::detail::WriteToBuffer(output_buffer, all_zeros);

    auto local_input_buffer_address = local_input_buffer->address();
    auto output_buffer_address = output_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    const std::size_t pages_per_send = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / page_size;
    const auto& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    constexpr size_t fabric_connection_routing_plane_index = 0;
    generate_fabric_test_kernels(
        sender_program,
        sender_device,
        worker_core,
        fabric_connection_routing_plane_index,
        unicast_send{1},  // 1 hops because we are sending to the next chip where receiver lives
        page_size,
        num_pages_total,
        pages_per_send,
        local_input_buffer_address,
        src_is_dram,
        output_buffer_address,
        dest_is_dram,
        packet_header_buffer_cb_id,
        receiver_program,
        receiver_device,
        receiver_worker_core,
        scatter_write);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<IDevice*> devices = {sender_device, receiver_device};
    log_trace(tt::LogTest, "{} programs, {} devices", programs.size(), devices.size());
    run_programs(programs, devices);
    log_info(tt::LogTest, "Reading back outputs");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        pass &= run_output_check(all_zeros, inputs, output_buffer) == Correctness::Correct;
    }
    return pass;
}

static bool RunLineFabricTest(
    std::vector<tt_metal::IDevice*> devices,
    std::vector<Program>& programs,

    const size_t mcast_first_chip,
    const size_t mcast_last_chip,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram,
    bool scatter_write) {
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    const std::size_t edm_buffer_size_no_header = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
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
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    const auto& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    const std::size_t pages_per_send = edm_buffer_size_no_header / page_size;

    // For multicast, create a receiver program on the furthest device
    auto furthest_device = devices.back();
    programs.push_back(Program());
    auto& receiver_program = programs.back();
    CoreCoord receiver_worker_core = {0, 1};  // Different core from sender

    constexpr size_t fabric_connection_routing_plane_index = 0;
    generate_fabric_test_kernels(
        programs[0],
        devices[0],
        worker_core,
        fabric_connection_routing_plane_index,
        mcast_send{mcast_first_chip, mcast_last_chip - mcast_first_chip + 1},
        page_size,
        num_pages_total,
        pages_per_send,
        local_input_buffer_address,
        src_is_dram,
        local_output_buffer_address,
        dest_is_dram,
        packet_header_buffer_cb_id,
        receiver_program,
        furthest_device,
        receiver_worker_core,
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

static void persistent_fabric_teardown_sequence(
    const std::vector<IDevice*>& devices,
    std::optional<SubdeviceInfo>& subdevice_managers,
    tt::tt_fabric::EdmLineFabricOpInterface& line_fabric,
    tt::tt_fabric::TerminationSignal termination_mode = tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE) {
    log_info(tt::LogTest, "Tearing down fabric");

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

static void setup_test_with_persistent_fabric(
    const std::vector<IDevice*>& devices,
    std::optional<SubdeviceInfo>& subdevice_managers,
    std::optional<std::vector<Program>>& fabric_programs,
    std::vector<Program*>& fabric_program_ptrs,
    std::optional<tt::tt_fabric::EdmLineFabricOpInterface>& line_fabric,
    std::optional<size_t> num_links = std::nullopt,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    size_t switch_interval = 0,
    bool loopback_on_last_device = false,
    bool is_galaxy = false,
    const tt::tt_fabric::FabricRouterBufferConfig& edm_buffer_config = tt::tt_fabric::FabricRouterBufferConfig{}) {
    fabric_programs = std::vector<Program>(devices.size());
    subdevice_managers = create_subdevices(devices);
    std::transform(
        fabric_programs->begin(), fabric_programs->end(), std::back_inserter(fabric_program_ptrs), [](auto& p) {
            return &p;
        });

    line_fabric = tt::tt_fabric::EdmLineFabricOpInterface(
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
static int TestLineFabricEntrypoint(
    const size_t mcast_first_chip,
    const size_t mcast_last_chip,
    const uint32_t page_size,
    const uint32_t num_pages_total,
    const bool src_is_dram,
    const bool dest_is_dram,
    const bool scatter_write = false) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops

    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
        return 0;
    }

    auto test_fixture = Fabric1DDeviceInitFixture(tt::tt_metal::FabricConfig::FABRIC_1D);
    auto &mesh_device = *(test_fixture.mesh_device_);

    // build a line of devices
    std::vector<IDevice*> devices = {
        mesh_device.get_device(MeshCoordinate(0, 0)),
        mesh_device.get_device(MeshCoordinate(0, 1)),
        mesh_device.get_device(MeshCoordinate(0, 2)),
        mesh_device.get_device(MeshCoordinate(0, 3))};

    std::vector<Program> programs(1);

    auto launch_workers = [&](std::vector<Program>& _programs) -> bool {
        bool success = false;
        try {
            success = RunLineFabricTest(
                std::vector<IDevice*>{devices[0]},
                _programs,
                mcast_first_chip,
                mcast_last_chip,
                page_size,
                num_pages_total,
                src_is_dram,
                dest_is_dram,
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

    test_fixture.TearDown();

    return success ? 0 : -1;
}

static int TestLoopbackEntrypoint(
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
    if (num_devices < 2) {
        log_info(tt::LogTest, "This test can only be run on 2+ chip systems");
        return 0;
    }

    auto test_fixture = Fabric1DDeviceInitFixture(tt::tt_metal::FabricConfig::FABRIC_1D);
    auto &mesh_device = *(test_fixture.mesh_device_);

    const auto& device_0 = mesh_device.get_device(MeshCoordinate(0, 0));
    const auto& device_1 = mesh_device.get_device(MeshCoordinate(0, 1));
    // const auto& device_1 = test_fixture.mesh_device_->get_device(device_id);

    std::vector<Program> programs(2);

    IDevice* sender_device = device_0;
    IDevice* receiver_device = device_1;

    log_trace(tt::LogTest, "{} programs ", programs.size());
    bool success = false;
    try {
        success = RunLoopbackTest(
            device_0,
            device_1,

            page_size,
            num_pages_total,
            src_is_dram,
            dest_is_dram,
            programs,
            scatter_write);
    } catch (std::exception& e) {
        log_error(tt::LogTest, "Caught exception: {}", e.what());
        test_fixture.TearDown();
        return -1;
    }

    {
        // Run the test twice with a single fabric invocation

        std::vector<Program> worker_programs(2);
        try {
            success = RunLoopbackTest(
                device_0,
                device_1,

                page_size,
                num_pages_total,
                src_is_dram,
                dest_is_dram,
                worker_programs,
                scatter_write);
        } catch (std::exception& e) {
            log_error(tt::LogTest, "Caught exception: {}", e.what());
            test_fixture.TearDown();
            return -1;
        }
        // Wait for worker programs to finish

        // Teardown the fabric
        tt_metal::Finish(sender_device->command_queue());
        // tt_metal::Finish(receiver_device->command_queue(), {d1_worker_subdevice});
    }

    test_fixture.TearDown();

    return success ? 0 : -1;
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

static std::vector<CoreCoord> compute_top_row_ethernet_cores(IDevice* device, const Fabric1DWorkerConfig& worker_config) {
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

static CoreCoord wh_glx_physical_worker_core_from_logical_core(CoreCoord logical_core) {
    auto physical_x = logical_core.x <= 3 ? logical_core.x + 1 : logical_core.x + 2;
    auto physical_y = logical_core.y <= 4 ? logical_core.y + 1 : logical_core.y + 2;
    return CoreCoord(physical_x, physical_y);
}

static CoreRangeSet get_optimal_worker_core_placement(
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
    bool use_galaxy, bool use_tg, size_t line_size, tt::tt_fabric::Topology topology, const MeshDeviceView& view) {
    std::vector<IDevice*> devices_;
    if (use_galaxy) {
        if (line_size <= 4) {
            if (use_tg) {
                if (topology == tt::tt_fabric::Topology::Ring) {
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
            if (topology == tt::tt_fabric::Topology::Ring && use_tg) {
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
    tt::tt_fabric::Topology topology,
    const MeshDevice& mesh_device,
    size_t num_fabric_rows,
    size_t num_fabric_cols) {
    bool use_default_device_selection = num_fabric_rows == 0 && num_fabric_cols == 0;
    std::vector<std::vector<IDevice*>> fabrics_under_test;
    if (use_default_device_selection) {
        fabrics_under_test.push_back(
            generate_default_line_fabric_under_test(use_galaxy, use_tg, line_size, topology, mesh_device.get_view()));
    } else {
        fabrics_under_test.reserve(num_fabric_rows + num_fabric_cols);
        TT_FATAL(
            num_fabric_rows <= mesh_device.num_rows(),
            "num_rows_requested must be less than or equal to the number of rows in the mesh. Requested: {}, "
            "Available: {}",
            num_fabric_rows,
            mesh_device.num_rows());
        TT_FATAL(
            num_fabric_cols <= mesh_device.num_cols(),
            "num_cols_requested must be less than or equal to the number of cols in the mesh. Requested: {}, "
            "Available: {}",
            num_fabric_cols,
            mesh_device.num_cols());
        for (size_t i = 0; i < num_fabric_rows; i++) {
            fabrics_under_test.push_back(mesh_device.get_view().get_devices_on_row(i));
        }
        for (size_t i = 0; i < num_fabric_cols; i++) {
            fabrics_under_test.push_back(mesh_device.get_view().get_devices_on_column(i));
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
            (fabric_config != tt::tt_metal::FabricConfig::DISABLED &&
             std::is_same_v<FABRIC_DEVICE_FIXTURE, Fabric1DFixture>) ||
            // prev not Fabric1DLine, now Fabric1DLine
            (fabric_config != tt::tt_metal::FabricConfig::FABRIC_1D &&
             std::is_same_v<FABRIC_DEVICE_FIXTURE, Fabric1DLineDeviceInitFixture>) ||
            // prev not Fabric1DRing, now Fabric1DRing
            (fabric_config != tt::tt_metal::FabricConfig::FABRIC_1D_RING &&
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

static Fabric1DWorkerConfig get_fabric_1d_worker_config(
    size_t device_index,
    const std::vector<IDevice*>& devices,
    tt::tt_fabric::Topology topology,
    FabricTestMode fabric_mode,
    size_t line_size,
    size_t line_index,
    bool senders_are_unidirectional,
    size_t num_devices_with_workers) {
    Fabric1DWorkerConfig config;
    if (topology == tt::tt_fabric::Topology::Ring && fabric_mode != FabricTestMode::RingAsLinear) {
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

static tt::tt_fabric::FabricRouterBufferConfig get_edm_buffer_config_wormhole(
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

static tt::tt_fabric::FabricRouterBufferConfig get_edm_buffer_config_blackhole(
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

static tt::tt_fabric::FabricRouterBufferConfig get_edm_buffer_config_helper(FabricTestMode fabric_mode, size_t line_size) {
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

    TT_FATAL(num_devices_with_workers <= line_size, "num_devices_with_workers must be less than or equal to line_size");
    TT_FATAL(
        !(params.num_fabric_rows > 0 && params.num_fabric_cols > 0),
        "Only one of num_fabric_rows and num_fabric_cols may be greater than 0. Test support for both axes live at the "
        "same time is not yet supported");
    if (use_device_init_fabric && params.num_fabric_rows == 0 && params.num_fabric_cols == 0) {
        TT_FATAL(use_t3k, "Using the full mesh as one ring topoplogy is only supported for T3K");
    }

    tt::tt_fabric::Topology topology;
    FabricTestMode fabric_mode = params.fabric_mode;
    switch (fabric_mode) {
        case FabricTestMode::Linear: topology = tt::tt_fabric::Topology::Linear; break;
        case FabricTestMode::SaturateChipToChipRing:
            TT_FATAL(line_size == 4, "SaturateChipToChipRing only supports line_size 4");
        case FabricTestMode::HalfRing:
        case FabricTestMode::FullRing:
        case FabricTestMode::RingAsLinear: topology = tt::tt_fabric::Topology::Ring; break;
    }

    const auto edm_buffer_config = get_edm_buffer_config_helper(fabric_mode, line_size);

    auto worker_core_logical = [](size_t link) { return CoreCoord(link, 0); };

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
    auto &mesh_device = *(test_fixture->mesh_device_);

    auto fabrics_under_test_devices = generate_line_fabrics_under_test(
        use_galaxy, use_tg, line_size, topology, mesh_device, params.num_fabric_rows, params.num_fabric_cols);

    // Persistent Fabric Setup
    std::optional<tt::tt_fabric::EdmLineFabricOpInterface> fabric_handle = std::nullopt;
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

    tt::stl::SmallVector<std::shared_ptr<Buffer>> device_dest_buffers;
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
    std::vector<tt::tt_metal::GlobalSemaphore> global_semaphore_handles;
    for (size_t i = 0; i < line_size * 4; i++) {
        auto global_semaphore = tt::tt_metal::CreateGlobalSemaphore(
            &mesh_device,
            devices_[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
            0,                              // initial value
            tt::tt_metal::BufferType::L1   // buffer type
        );
        global_semaphore_handles.push_back(global_semaphore);
        auto global_semaphore_addr = global_semaphore.address();
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
            if (use_tg and topology == tt::tt_fabric::Topology::Linear) {
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

            std::optional<tt::tt_fabric::EdmLineFabricOpInterface> local_device_fabric_handle = std::nullopt;
            if (!use_device_init_fabric) {
                local_device_fabric_handle =
                    tt::tt_fabric::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
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
                "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/edm_fabric_writer.cpp",
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
                                             tt::tt_fabric::EdmLineFabricOpInterface::Direction direction,
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
                        {
                            auto worker_flow_control_semaphore_id = CreateSemaphore(program, {worker_core}, 0);
                            auto worker_teardown_semaphore_id = CreateSemaphore(program, {worker_core}, 0);
                            auto worker_buffer_index_semaphore_id = CreateSemaphore(program, {worker_core}, 0);
                            append_worker_to_fabric_edm_sender_rt_args(
                                connection,
                                worker_flow_control_semaphore_id,
                                worker_teardown_semaphore_id,
                                worker_buffer_index_semaphore_id,
                                rt_args_out);
                            log_info(
                                tt::LogTest,
                                "On device: {}, connecting to EDM fabric in {} direction. EDM noc_x: {}, noc_y: {}",
                                device->id(),
                                direction,
                                connection.edm_noc_x,
                                connection.edm_noc_y);
                        }
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
                    tt::tt_fabric::EdmLineFabricOpInterface::FORWARD,
                    rt_args);
                build_connection_args(
                    worker_core,
                    l,
                    worker_config.has_backward_connection,
                    worker_config.backward_device,
                    tt::tt_fabric::EdmLineFabricOpInterface::BACKWARD,
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
            tt_metal::Synchronize(d);
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

static void validate_fabric_packet_send_test_params(const FullMeshTestParams& full_mesh_params) {
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

static void validate_fabric_packet_send_test_params(
    const std::variant<WriteThroughputStabilityTestWithPersistentFabricParams, FullMeshTestParams>& params) {
    if (std::holds_alternative<WriteThroughputStabilityTestWithPersistentFabricParams>(params)) {
        const auto& write_throughput_params = std::get<WriteThroughputStabilityTestWithPersistentFabricParams>(params);
        TT_THROW("Not commonized yet");
    } else {
        const auto& full_mesh_params = std::get<FullMeshTestParams>(params);
        validate_fabric_packet_send_test_params(full_mesh_params);
    }
}

static tt::tt_fabric::Topology get_topology(FabricTestMode fabric_mode) {
    switch (fabric_mode) {
        case FabricTestMode::Linear: return tt::tt_fabric::Topology::Linear;
        case FabricTestMode::SaturateChipToChipRing:
        case FabricTestMode::HalfRing:
        case FabricTestMode::FullRing:
        case FabricTestMode::RingAsLinear: return tt::tt_fabric::Topology::Ring;
        default: TT_THROW("Invalid fabric mode");
    }
    return tt::tt_fabric::Topology::Linear;
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

static void launch_kernels_and_wait_for_completion(
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
        tt_metal::Synchronize(device);
    }
    for (const auto& [device, program] : device_programs) {
        tt_metal::detail::DumpDeviceProfileResults(device);
    }
}

static std::tuple<size_t, per_axis_array_t<std::vector<std::vector<Fabric1DWorkerConfig>>>>
generate_1D_fabric_on_full_mesh_worker_configs(
    const FullMeshTestParams& params,
    const per_axis_array_t<std::vector<std::vector<IDevice*>>>& fabrics_under_test_devices_per_axis,
    const per_axis_array_t<tt::tt_fabric::Topology>& topologies,
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

static void generate_1d_fabric_on_full_mesh_worker_rt_args(
    const FullMeshTestParams& params,
    const per_axis_array_t<std::vector<std::vector<Fabric1DWorkerConfig>>>&
        worker_configs_per_axis_per_fabric_per_device,
    const per_axis_array_t<std::vector<std::vector<CoreCoord>>>& worker_cores_vec_per_axis_per_device,
    const per_axis_array_t<std::vector<tt::tt_metal::DeviceAddr>>& global_semaphore_addrs_per_axis,
    const std::vector<tt::tt_metal::GlobalSemaphore>& global_semaphore_handles_per_axis,
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
                                     tt::tt_fabric::EdmLineFabricOpInterface::Direction direction,
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
            tt::tt_fabric::EdmLineFabricOpInterface::FORWARD,
            rt_args);
        build_connection_args(
            worker_core,
            l,
            worker_config.has_backward_connection,
            worker_config.backward_device,
            tt::tt_fabric::EdmLineFabricOpInterface::BACKWARD,
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

static std::vector<CoreCoord> setup_worker_core_coords(
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

    static constexpr size_t packet_header_cb_size_in_headers = 5;
    static constexpr bool enable_persistent_fabric_mode = true;
    static constexpr tt::DataFormat cb_df = tt::DataFormat::Bfp8;
    constexpr size_t MAX_NUM_AXES = FullMeshTestParams::MAX_NUM_AXES;

    auto max_packet_payload_size_bytes = test_specs.packet_payload_size_bytes;

    auto num_devices = tt::tt_metal::GetNumAvailableDevices();

    validate_fabric_packet_send_test_params(params);

    per_axis_array_t<tt::tt_fabric::Topology> topologies;
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
    auto &mesh_device = *(test_fixture_->mesh_device_);
    // FABRIC_DEVICE_FIXTURE test_fixture;
    // auto view = *(test_fixture.view_);

    per_axis_array_t<std::vector<std::vector<IDevice*>>> fabrics_under_test_devices_per_axis;
    for (size_t axis = 0; axis < MAX_NUM_AXES; axis++) {
        size_t num_fabric_rows = axis == 0 ? params.num_fabric_rows : 0;
        size_t num_fabric_cols = axis == 1 ? params.num_fabric_cols : 0;
        fabrics_under_test_devices_per_axis[axis] = generate_line_fabrics_under_test(
            use_galaxy, use_tg, params.line_size[axis], topologies[axis], mesh_device, num_fabric_rows, num_fabric_cols);
    }

    size_t packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    TT_FATAL(packet_header_size_bytes != 0, "Error in initializing local variable `packet_header_size_bytes`");

    // Big boiler plate setup loop
    CoreCoord sync_core_coord = worker_core_logical(0, 0);
    per_axis_array_t<std::vector<std::vector<CoreCoord>>> worker_cores_vec_per_axis_per_device;
    std::unordered_map<size_t, std::vector<CoreCoord>> worker_cores_per_device;
    tt::stl::SmallVector<std::shared_ptr<Buffer>> device_dest_buffers;
    std::unordered_map<IDevice*, Program> device_programs;
    per_axis_array_t<std::vector<CoreCoord>> dest_core_coord_per_axis;
    per_axis_array_t<std::vector<tt::tt_metal::DeviceAddr>> global_semaphore_addrs_per_axis;
    std::vector<tt::tt_metal::GlobalSemaphore> global_semaphore_handles_per_axis;
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
        auto global_semaphore = tt::tt_metal::CreateGlobalSemaphore(
            &mesh_device,
            fabrics_under_test_devices_per_axis[axis][0][0]->worker_cores(
                HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
            0,                             // initial value
            tt::tt_metal::BufferType::L1   // buffer type
        );
        global_semaphore_handles_per_axis.push_back(global_semaphore);
        auto global_semaphore_addr = global_semaphore.address();
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
        auto fabric_mode = fabric_modes[axis];
        auto topology = topologies[axis];
        auto& dest_core_coord = dest_core_coord_per_axis[axis];
        auto num_devices_with_workers = params.num_devices_with_workers[axis];
        if (num_devices_with_workers == 0) {
            num_devices_with_workers = line_size;
        }
        auto num_links = params.num_links[axis];
        auto first_link_offset = params.first_link_offset[axis];
        auto num_op_invocations = params.num_op_invocations;
        auto senders_are_unidirectional = params.senders_are_unidirectional[axis];
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
                    "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/edm_fabric_writer.cpp",
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

static void RunWriteThroughputStabilityTestWithPersistentFabric(
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

static size_t get_number_of_links_for_ring_deadlock_stability_test(
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

static std::vector<IDevice*> get_devices_for_ring_deadlock_stability_test(
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
    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

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

    auto topology = tt::tt_fabric::Topology::Ring;
    constexpr size_t num_unicasts = 0;
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
    auto mesh_device = test_fixture.mesh_device_;

    std::vector<IDevice*> devices_ =
        get_devices_for_ring_deadlock_stability_test(mesh_device.get()->get_view(), cluster_type, num_devices, row_or_col);
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

    tt::stl::SmallVector<std::shared_ptr<Buffer>> device_dest_buffers;
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
    std::vector<tt::tt_metal::GlobalSemaphore> global_semaphore_handles;
    for (size_t i = 0; i < line_size * 4; i++) {
        auto global_semaphore = tt::tt_metal::CreateGlobalSemaphore(
            mesh_device.get(),
            devices_[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
            0,                              // initial value
            tt::tt_metal::BufferType::L1);  //,  // buffer type
        global_semaphore_handles.push_back(global_semaphore);
        auto global_semaphore_addr = global_semaphore.address();
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
            "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/edm_fabric_writer.cpp",
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
            tt_metal::Synchronize(d);
        }
        log_trace(tt::LogTest, "Main op done");
    }

    log_trace(tt::LogTest, "Finished");
}
