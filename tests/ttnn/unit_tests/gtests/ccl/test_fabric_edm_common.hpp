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
#include <tt-metalium/fabric_edm_packet_header.hpp>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "tt_metal/fabric/ccl/ccl_common.hpp"
#include "tt_metal/fabric/erisc_datamover_builder_helper.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
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
    tt::ARCH arch_{tt::ARCH::Invalid};
    std::size_t num_devices_{};
    bool device_open = false;

    // Common constants for both fixtures
    static constexpr size_t TG_NUM_DEVICES = 36;
    static constexpr size_t GALAXY_6U_NUM_DEVICES = 32;

    // Gets the appropriate mesh shape based on device configuration
    MeshShape GetDeterminedMeshShape() const {
        if (num_devices_ == TG_NUM_DEVICES || num_devices_ == GALAXY_6U_NUM_DEVICES) {
            return MeshShape{8, 4};
        } else if (num_devices_ == 4) {
            return MeshShape{1, 4};
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

        switch (arch_) {
            case tt::ARCH::WORMHOLE_B0:
                if (!(num_devices_ >= 8 && (tt::tt_metal::GetNumPCIeDevices() == 4 ||
                                            tt::tt_metal::GetNumPCIeDevices() == GALAXY_6U_NUM_DEVICES))) {
                    TT_THROW("This suite can only be run on T3000 or TG Wormhole devices");
                }
                break;

            case tt::ARCH::BLACKHOLE:
                if (num_devices_ != 4) {
                    TT_THROW("This suite can only be run on LLMBox");
                }
                break;

            default: TT_THROW("Only Wormhole or Blackhole devices are supported in this test suite");
        };
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

        auto mapped_devices = SystemMesh::instance().get_mapped_devices(cluster_shape);

        const std::vector<int> physical_device_ids = extract_locals(mapped_devices.device_ids);
        TT_FATAL(physical_device_ids.size() == cluster_shape.mesh_size(), "Some of the devices are remote");
        physical_devices_ = tt::tt_metal::detail::CreateDevices(physical_device_ids);

        std::vector<IDevice*> devices;
        devices.reserve(physical_device_ids.size());
        for (auto device_id : physical_device_ids) {
            devices.push_back(physical_devices_.at(device_id));
        }

        view_ = std::make_shared<MeshDeviceView>(cluster_shape, devices, mapped_devices.fabric_node_ids);
        device_open = true;
    }

    void TearDown() override {
        if (device_open) {
            tt::tt_metal::detail::CloseDevices(physical_devices_);
            device_open = false;
        }
    }

    Fabric1DFixture() : BaseFabricFixture() { this->SetupDevices(); }

    Fabric1DFixture(tt::tt_fabric::FabricConfig fabric_config) : BaseFabricFixture(fabric_config) {
        this->SetupDevices();
    }

    ~Fabric1DFixture() override { TearDown(); }
};

class Fabric1DDeviceInitFixture {
public:
    tt::ARCH arch_{tt::ARCH::Invalid};
    std::size_t num_devices_{};
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

    MeshFabric1DFixture(tt::tt_fabric::FabricConfig fabric_config) : BaseFabricFixture(fabric_config) {
        this->SetupDevices();
    }

    ~MeshFabric1DFixture() override {
        if (device_open) {
            TearDown();
        }
    }
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
        device->set_sub_device_stall_group({{subdevice_info.worker_subdevice_id.at(device->id())}});
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
static void build_and_enqueue(
    const std::vector<IDevice*>& devices, std::vector<Program*>& program_ptrs, bool enqueue_only = false) {
    TT_FATAL(
        devices.size() == program_ptrs.size(),
        "Number of devices must match number of programs when calling build_and_enqueue in test");
    if (!enqueue_only) {
        for (size_t i = 0; i < devices.size(); i++) {
            tt::tt_metal::detail::CompileProgram(devices[i], *program_ptrs[i]);
        }
    }
    for (size_t i = 0; i < devices.size(); i++) {
        tt_metal::EnqueueProgram(devices[i]->command_queue(), *program_ptrs[i], false);
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
        CreateCircularBuffer(program, worker_core, cb_src0_config);
    }
    {
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_size, {{second_cb_index, df}})
                .set_page_size(second_cb_index, page_size);
        CreateCircularBuffer(program, worker_core, cb_src1_config);
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
    std::optional<tt::tt_fabric::EdmLineFabricOpInterface>& line_fabric,

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
        fabric_enabled
            ? line_fabric->uniquely_connect_worker(devices[0], tt::tt_fabric::EdmLineFabricOpInterface::FORWARD)
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
    TT_FATAL(test_mode == TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK, "Only LOCAL_WRITEBACK is supported");
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
        return true;
    }
    MeshFabric1DFixture test_fixture;
    auto mesh_device = test_fixture.mesh_device_;

    std::vector<IDevice*> devices;
    devices.reserve(fabric_num_devices);
    for (size_t i = 0; i < fabric_num_devices; i++) {
        devices.push_back(mesh_device->get_device(MeshCoordinate(0, i)));
    }

    std::vector<std::shared_ptr<distributed::MeshDevice>> mesh_devices;
    mesh_devices.reserve(fabric_num_devices);
    for (size_t i = 0; i < fabric_num_devices; i++) {
        mesh_devices.push_back(mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, i)));
    }

    std::vector<Program> programs(1);
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<tt::tt_fabric::EdmLineFabricOpInterface> line_fabric;

    std::vector<Tensor> input0_tensors_device;
    std::vector<Tensor> input1_tensors_device;
    std::vector<Tensor> output0_tensors_device;
    std::vector<Tensor> output1_tensors_device;

    // All this garbage is to make sure the test sets up buffer addresses correctly so we can safely
    // multicast to a consistent destination address
    for (size_t i = 0; i < devices.size(); i++) {
        input0_tensors_device.push_back(
            input_tensor0.to_device(mesh_devices.at(i).get(), input_tensor0_mem_config, ttnn::DefaultQueueId));
        input1_tensors_device.push_back(
            input_tensor1.to_device(mesh_devices.at(i).get(), input_tensor1_mem_config, ttnn::DefaultQueueId));
        output0_tensors_device.push_back(
            output_tensor0.to_device(mesh_devices.at(i).get(), output_tensor0_mem_config, ttnn::DefaultQueueId));
        output1_tensors_device.push_back(
            output_tensor1.to_device(mesh_devices.at(i).get(), output_tensor1_mem_config, ttnn::DefaultQueueId));
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
    TT_FATAL(
        test_writeback_mode == TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK, "Only LOCAL_WRITEBACK is supported");
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

    MeshFabric1DFixture test_fixture;
    auto full_mesh_device = test_fixture.mesh_device_;

    IDevice* device = full_mesh_device->get_device(MeshCoordinate(0, 0));
    std::shared_ptr<distributed::MeshDevice> mesh_device =
        full_mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));

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
        device_tensors.push_back(host_tensors[i].to_device(mesh_device.get(), mem_configs[i]));
        log_info(tt::LogTest, "Tensor[{}] allocated starting at address {}", i, device_tensors[i].buffer()->address());
    }
    TT_ASSERT(device_tensors.size() == num_tensors);
    TT_ASSERT(device_tensors.size() == host_tensors.size());

    // MAIN STUFF

    // Initial setup like worker core assignment, chunk read order, etc.

    std::vector<CoreRangeSet> pipeline_stage_worker_cores = {};
    pipeline_stage_worker_cores.reserve(num_stages);
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
        CreateCircularBuffer(program, pipeline_stage_worker_cores[stage], cb_config);
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
            tt_metal::Finish(d->command_queue(), {{subdevice_managers->worker_subdevice_id.at(d->id())}});
        });
    } else {
        std::ranges::for_each(devices, [&](IDevice* d) { tt_metal::Finish(d->command_queue(), {}); });
    }
}

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
void run_all_gather_with_persistent_fabric(const size_t dim, const size_t num_links, const ttnn::Shape& input_shape) {
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
