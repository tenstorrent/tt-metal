
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/logger.hpp"
#include "sub_device/sub_device_types.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/erisc_datamover_builder.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/cpp/ttnn/operations/experimental/reshape/reshape.hpp"

#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/distributed/mesh_device_view.hpp"

#include "tt_metal/impl/tile/tile.hpp"

#include "umd/device/types/arch.h"
#include "umd/device/types/cluster_descriptor_types.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <unordered_set>

using namespace tt;
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

using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshDeviceView;
using tt::tt_metal::distributed::MeshShape;
class T3000TestDevice {
public:
    T3000TestDevice() : device_open(false) {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run without TT_METAL_SLOW_DISPATCH_MODE set");
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::WORMHOLE_B0 and num_devices_ == 8 and tt::tt_metal::GetNumPCIeDevices() == 4) {
            mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{2, 4}));

            std::vector<chip_id_t> ids(num_devices_, 0);
            std::iota(ids.begin(), ids.end(), 0);

        } else {
            TT_THROW("This suite can only be run on T3000 Wormhole devices");
        }
        device_open = true;
    }
    ~T3000TestDevice() {
        if (device_open) {
            TearDown();
        }
    }

    void TearDown() {
        device_open = false;
        mesh_device_->close_devices();
    }

    tt::ARCH arch_;
    size_t num_devices_;
    std::shared_ptr<MeshDevice> mesh_device_;

private:
    bool device_open;
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

struct EthLinkBuilder {
    ttnn::ccl::FabricEriscDatamoverBuilder sender_edm_builder;
    ttnn::ccl::FabricEriscDatamoverBuilder receiver_edm_builder;
    tt_xy_pair sender_core;
    tt_xy_pair receiver_core;
};

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

static SubdeviceInfo create_subdevices(std::vector<Device*> const& devices) {
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
    }

    return subdevice_info;
}

Correctness run_output_check(
    std::vector<uint32_t> const& all_zeros,
    std::vector<uint32_t> const& inputs,
    std::shared_ptr<Buffer>& output_buffer) {
    constexpr bool debug_mode = true;
    std::vector<uint32_t> readback_data_vec(all_zeros.size(), 0);  // init to 0 data for easier debug

    tt_metal::detail::ReadFromBuffer(output_buffer, readback_data_vec);
    return run_output_check(inputs, readback_data_vec);
};

void run_programs(
    std::vector<Program>& programs,
    std::vector<Device*> const& devices,
    std::optional<std::unordered_map<chip_id_t, SubDeviceId>> const& sub_device_ids = std::nullopt) {
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
            if (sub_device_ids.has_value()) {
                tt_metal::Finish(devices.at(i)->command_queue(), {sub_device_ids.value().at(devices.at(i)->id())});
            } else {
                tt_metal::Finish(devices.at(i)->command_queue());
            }
        }
    }
}

std::tuple<std::shared_ptr<Buffer>, std::vector<uint32_t>> build_input_buffer(
    Device* first_device, size_t tensor_size_bytes, BankedConfig const& test_config) {
    auto inputs = std::vector<uint32_t>(tensor_size_bytes / sizeof(uint32_t), 0);
    std::iota(inputs.begin(), inputs.end(), 0);

    // Input buffer
    auto local_input_buffer = CreateBuffer(InterleavedBufferConfig{
        first_device, test_config.size_bytes, test_config.page_size_bytes, test_config.input_buffer_type});
    tt_metal::detail::WriteToBuffer(local_input_buffer, inputs);
    return {local_input_buffer, inputs};
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

static constexpr size_t PACKET_HEADER_SIZE_BYTES = sizeof(tt::fabric::PacketHeader);
void generate_sender_worker_kernels(
    Program& program,
    Device* device,
    CoreCoord const& worker_core,
    ttnn::ccl::SenderWorkerAdapterSpec const& worker_fabric_connection,
    mode_variant_t const& mode,
    std::size_t edm_buffer_size,
    uint32_t page_plus_header_size,
    uint32_t num_pages_total,
    uint32_t num_pages_per_edm_buffer,
    uint32_t local_worker_fabric_semaphore_id,
    uint32_t local_worker_last_message_semaphore_id,
    uint32_t dram_input_buffer_base_addr,
    bool src_is_dram,
    uint32_t dram_output_buffer_base_addr,
    bool dest_is_dram,
    uint32_t worker_buffer_index_semaphore_id,
    // farthest to closest
    std::vector<ttnn::ccl::edm_termination_info_t> const& edm_termination_infos) {
    auto const& edm_noc_core = CoreCoord(worker_fabric_connection.edm_noc_x, worker_fabric_connection.edm_noc_y);
    std::vector<uint32_t> sender_worker_reader_compile_args{
        src_is_dram,      //
        num_pages_total,  //
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        num_pages_per_edm_buffer};
    std::vector<uint32_t> sender_worker_reader_runtime_args{dram_input_buffer_base_addr};

    log_trace(tt::LogTest, "\tSenderReader CT Args");
    for (auto const& arg : sender_worker_reader_compile_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }
    log_trace(tt::LogTest, "\tSenderReader RT Args");
    for (auto const& arg : sender_worker_reader_runtime_args) {
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
    for (auto const& arg : sender_worker_writer_compile_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }
    log_trace(tt::LogTest, "\tSenderWriter RT Args");
    for (auto const& arg : sender_worker_writer_runtime_args) {
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
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,

    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram,
    std::vector<Program>& programs,
    ttnn::ccl::FabricEriscDatamoverBuilder& chip_0_edm_builder,
    std::optional<SubdeviceInfo>& subdevice_managers,
    bool enable_persistent_fabric) {
    auto& sender_program = programs.at(0);
    std::size_t page_plus_header_size = page_size + sizeof(tt::fabric::PacketHeader);
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    std::vector<CoreCoord> worker_cores = {CoreCoord(0, 0)};

    auto local_worker_fabric_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);
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

    static constexpr std::size_t edm_buffer_size = 4096 + PACKET_HEADER_SIZE_BYTES;

    auto chip0_worker_fabric_connection = chip_0_edm_builder.build_connection_to_worker_channel();
    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    const std::size_t pages_per_send =
        (chip0_worker_fabric_connection.buffer_size_bytes - PACKET_HEADER_SIZE_BYTES) / page_size;
    auto const& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    std::vector<ttnn::ccl::edm_termination_info_t> const& edm_termination_infos =
        enable_persistent_fabric ? std::vector<ttnn::ccl::edm_termination_info_t>{}
                                 : std::vector<ttnn::ccl::edm_termination_info_t>{
                                       {1,
                                        sender_device->ethernet_core_from_logical_core(eth_receiver_core).x,
                                        sender_device->ethernet_core_from_logical_core(eth_receiver_core).y,
                                        ttnn::ccl::FabricEriscDatamoverConfig::termination_signal_address},
                                       {0,
                                        sender_device->ethernet_core_from_logical_core(eth_sender_core).x,
                                        sender_device->ethernet_core_from_logical_core(eth_sender_core).y,
                                        ttnn::ccl::FabricEriscDatamoverConfig::termination_signal_address}};

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
    std::vector<Device*> devices = {sender_device};
    if (!enable_persistent_fabric) {
        devices.push_back(receiver_device);
    }
    log_trace(tt::LogTest, "{} programs, {} devices", programs.size(), devices.size());
    run_programs(
        programs,
        devices,
        subdevice_managers.has_value() ? subdevice_managers.value().worker_subdevice_id
                                       : std::optional<std::unordered_map<chip_id_t, SubDeviceId>>{std::nullopt});
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
    std::vector<uint32_t> const& cb_indices,
    std::vector<Tensor const*> const& tensors,
    Device* device,
    uint32_t page_size,
    CoreRangeSet const& worker_core_range,
    uint32_t num_pages_per_edm_buffer,
    ttnn::ccl::v2::TensorSlice const& in0_command_tensor_slice,
    ttnn::ccl::v2::TensorSlice const& in1_command_tensor_slice,
    ttnn::ccl::cmd::CclCommandCode command_type,
    DataMovementConfig const& datamovement_kernel_config,
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> const& chip0_worker_forward_fabric_connection,
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> const& chip0_worker_backward_fabric_connection,
    std::optional<ttnn::ccl::cmd::CclHostLowLevelCommandSequence> const& optional_teardown_sequence,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args) {
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
    Device* device,
    Tensor& input_tensor0,
    Tensor& input_tensor1,
    Tensor& output_tensor0,
    Tensor& output_tensor1,
    size_t first_cb_index,
    size_t second_cb_index,
    CoreCoord const& worker_core,
    const uint32_t page_plus_header_size,
    const uint32_t num_pages_per_edm_buffer,
    ttnn::ccl::v2::TensorSlice const& in0_tensor_slice,
    ttnn::ccl::v2::TensorSlice const& in1_tensor_slice,
    ttnn::ccl::v2::TensorSlice const& out0_tensor_slice,
    ttnn::ccl::v2::TensorSlice const& out1_tensor_slice,
    std::optional<ttnn::ccl::cmd::CclHostLowLevelCommandSequence> const& optional_teardown_sequence,
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec>& chip0_worker_forward_fabric_connection,
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec>& chip0_worker_backward_fabric_connection,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args) {
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
    std::vector<tt_metal::Device*> const& devices,
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

    ttnn::ccl::v2::TensorSlice const& in0_tensor_slice,
    ttnn::ccl::v2::TensorSlice const& in1_tensor_slice,
    ttnn::ccl::v2::TensorSlice const& out0_tensor_slice,
    ttnn::ccl::v2::TensorSlice const& out1_tensor_slice,

    const uint32_t page_size,
    TwoInputReaderKernelWriteMode test_mode,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args,
    std::optional<SubdeviceInfo>& subdevice_managers,
    bool enable_persistent_fabric) {
    const bool fabric_enabled = test_mode != TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK;
    tt_metal::Device* device = devices.at(0);
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

    std::size_t page_plus_header_size = page_size + sizeof(tt::fabric::PacketHeader);

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
    auto const& worker_core = CoreCoord(0, 0);

    const size_t num_pages_per_edm_buffer = 2;

    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> chip0_worker_forward_fabric_connection =
        fabric_enabled ? line_fabric->uniquely_connect_worker(devices[0], ttnn::ccl::EdmLineFabricOpInterface::FORWARD)
                       : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>{std::nullopt};

    // always at start of line for now
    std::optional<std::vector<ttnn::ccl::edm_termination_info_t>> edm_termination_infos =
        (!fabric_enabled || enable_persistent_fabric)
            ? std::optional<std::vector<ttnn::ccl::edm_termination_info_t>>{std::nullopt}
            : line_fabric->generate_ordered_termination_info_farthest_to_nearest();
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> chip0_worker_backward_fabric_connection = std::nullopt;

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
    run_programs(
        programs,
        enable_persistent_fabric ? std::vector<Device*>{devices[0]} : devices,
        subdevice_managers.has_value() ? subdevice_managers.value().worker_subdevice_id
                                       : std::optional<std::unordered_map<chip_id_t, SubDeviceId>>{std::nullopt}

    );
    log_info(tt::LogTest, "Finished");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        log_info(tt::LogTest, "Reading back outputs");
        std::vector<SubDeviceId> out_tensor_worker_subdevice_id =
            subdevice_managers.has_value() ? std::vector<SubDeviceId>{subdevice_managers->worker_subdevice_id.at(
                                                 devices.at(output_tensor_dest_device_index)->id())}
                                           : std::vector<SubDeviceId>{};
        auto output0_cpu = output_tensor0_device.cpu(true, ttnn::DefaultQueueId, out_tensor_worker_subdevice_id);
        auto output1_cpu = output_tensor1_device.cpu(true, ttnn::DefaultQueueId, out_tensor_worker_subdevice_id);

        auto in_tensor_worker_subdevice_id =
            subdevice_managers.has_value()
                ? std::vector<SubDeviceId>{subdevice_managers->worker_subdevice_id.at(devices.at(0)->id())}
                : std::vector<SubDeviceId>{};
        auto in0_tensor_copyback_cpu =
            input_tensor0_device.cpu(true, ttnn::DefaultQueueId, in_tensor_worker_subdevice_id);
        auto in1_tensor_copyback_cpu =
            input_tensor1_device.cpu(true, ttnn::DefaultQueueId, in_tensor_worker_subdevice_id);

        auto in0_tensor_copyback = tt::tt_metal::owned_buffer::get_as<uint32_t>(in0_tensor_copyback_cpu);
        auto in1_tensor_copyback = tt::tt_metal::owned_buffer::get_as<uint32_t>(in1_tensor_copyback_cpu);

        auto in0_tensor_data = tt::tt_metal::owned_buffer::get_as<uint32_t>(input_tensor0);
        auto in1_tensor_data = tt::tt_metal::owned_buffer::get_as<uint32_t>(input_tensor1);
        auto out0_tensor_data = tt::tt_metal::owned_buffer::get_as<uint32_t>(output0_cpu);
        auto out1_tensor_data = tt::tt_metal::owned_buffer::get_as<uint32_t>(output1_cpu);

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
    std::vector<tt_metal::Device*> devices,
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
    std::size_t page_plus_header_size = page_size + sizeof(tt::fabric::PacketHeader);
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    static constexpr std::size_t edm_buffer_size = 4096 + PACKET_HEADER_SIZE_BYTES;
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
    bool all_same_addr = std::ranges::all_of(output_buffers, [local_output_buffer_address](auto const& buffer) {
        return buffer->address() == local_output_buffer_address;
    });
    TT_ASSERT(all_same_addr, "All output buffers must have the same address");

    ////////////////////////////////////////////////////////////////////////////
    //   Setup Semaphores and Builders
    ////////////////////////////////////////////////////////////////////////////

    auto local_worker_fabric_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    auto local_worker_last_message_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    auto worker_buffer_index_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    auto const& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    const auto edm_termination_infos = enable_persistent_fabric
                                           ? std::vector<ttnn::ccl::edm_termination_info_t>{}
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

    run_programs(
        programs,
        devices,
        subdevice_managers.has_value() ? subdevice_managers.value().worker_subdevice_id
                                       : std::optional<std::unordered_map<chip_id_t, SubDeviceId>>{std::nullopt});
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
    std::vector<Device*> const& devices,
    std::optional<SubdeviceInfo>& subdevice_managers,
    ttnn::ccl::EdmLineFabricOpInterface& line_fabric,
    tt::fabric::TerminationSignal termination_mode = tt::fabric::TerminationSignal::GRACEFULLY_TERMINATE) {
    log_info("Tearing down fabric");

    // Wait for workers to finish
    auto d0_worker_subdevice = devices[0]->get_sub_device_ids()[TEST_WORKERS_SUBDEVICE_INDEX];
    tt_metal::Finish(devices[0]->command_queue(), {subdevice_managers->worker_subdevice_id.at(devices[0]->id())});

    // Teardown the fabric
    line_fabric.teardown_from_host(termination_mode);

    // wait for fabric teardown to finish
    std::ranges::for_each(devices, [&](Device* d) {
        tt_metal::Finish(d->command_queue(), {subdevice_managers->fabric_subdevice_id.at(d->id())});
    });
}

void setup_test_with_persistent_fabric(
    std::vector<Device*> const& devices,
    std::vector<Program>& programs,
    std::optional<SubdeviceInfo>& subdevice_managers,
    std::optional<std::vector<Program>>& fabric_programs,
    std::vector<Program*>& fabric_program_ptrs,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& line_fabric,
    bool enable_persistent_fabric,
    std::optional<size_t> num_links = std::nullopt) {
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
        devices, fabric_program_ptrs, enable_persistent_fabric, num_links.value_or(1));

    if (enable_persistent_fabric) {
        TT_FATAL(fabric_programs.has_value(), "Fabric programs must be set if fabric is enabled");
        TT_FATAL(devices.size() == fabric_programs->size(), "Number of devices must match number of programs");

        log_info(tt::LogTest, "Building EDM kernels");
        line_fabric->build_kernels();
        for (size_t i = 0; i < devices.size(); i++) {
            tt::tt_metal::detail::CompileProgram(devices[i], fabric_programs->at(i));
        }
        for (size_t i = 0; i < devices.size(); i++) {
            tt_metal::EnqueueProgram(devices[i]->command_queue(), fabric_programs->at(i), false);
        }
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

    T3000TestDevice test_fixture;
    auto view = test_fixture.mesh_device_->get_view();

    // build a line of devices
    std::vector<Device*> devices = {
        view.get_device(0, 0), view.get_device(0, 1), view.get_device(0, 2), view.get_device(0, 3)};
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
                enable_persistent_fabric ? std::vector<Device*>{devices[0]} : devices,
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
            devices, subdevice_managers, line_fabric.value(), tt::fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
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

    T3000TestDevice test_fixture;
    auto view = test_fixture.mesh_device_->get_view();

    const auto& device_0 = view.get_device(0, 0);
    const auto& device_1 = view.get_device(0, 1);

    auto const& active_eth_cores = device_0->get_active_ethernet_cores(true);
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
    Device* sender_device = device_0;
    Device* receiver_device = device_1;

    static constexpr std::size_t edm_buffer_size = 4096 + PACKET_HEADER_SIZE_BYTES;
    const chip_id_t local_chip_id = 0;
    const chip_id_t remote_chip_id = 1;
    auto const& edm_config = ttnn::ccl::FabricEriscDatamoverConfig(edm_buffer_size, 1, 2);
    auto chip_0_edm_builder = ttnn::ccl::FabricEriscDatamoverBuilder::build(
        sender_device,
        fabric_sender_program,
        eth_sender_core,
        local_chip_id,
        remote_chip_id,
        edm_config,
        enable_persistent_fabric);
    auto chip_1_edm_builder = ttnn::ccl::FabricEriscDatamoverBuilder::build(
        receiver_device,
        fabric_receiver_program,
        eth_receiver_core,
        remote_chip_id,
        local_chip_id,
        edm_config,
        enable_persistent_fabric);
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

bool TestMultiInputReaderKernel(
    size_t fabric_num_devices,
    Tensor& input_tensor0,
    MemoryConfig const& input_tensor0_mem_config,
    Tensor& input_tensor1,
    MemoryConfig const& input_tensor1_mem_config,
    Tensor& output_tensor0,
    MemoryConfig const& output_tensor0_mem_config,
    Tensor& output_tensor1,
    MemoryConfig const& output_tensor1_mem_config,

    ttnn::ccl::v2::TensorSlice const& in0_tensor_slice,
    ttnn::ccl::v2::TensorSlice const& in1_tensor_slice,
    ttnn::ccl::v2::TensorSlice const& out0_tensor_slice,
    ttnn::ccl::v2::TensorSlice const& out1_tensor_slice,

    const uint32_t page_size,

    TwoInputReaderKernelWriteMode test_mode,
    ttnn::ccl::cmd::CclCommandDestArgs const& dest_args,
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
    T3000TestDevice test_fixture;

    TT_FATAL(
        !enable_persistent_fabric || test_mode != TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK,
        "Test configuration issue. Set local writeback mode with persistent fabric");

    auto view = test_fixture.mesh_device_->get_view();

    std::vector<Device*> devices;
    devices.reserve(fabric_num_devices);
    for (size_t i = 0; i < fabric_num_devices; i++) {
        devices.push_back(view.get_device(0, i));
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
        std::vector<SubDeviceId> subdevice_target =
            subdevice_managers.has_value()
                ? std::vector<SubDeviceId>{subdevice_managers->worker_subdevice_id.at(devices[i]->id())}
                : std::vector<SubDeviceId>{};
        input0_tensors_device.push_back(
            input_tensor0.to(devices.at(i), input_tensor0_mem_config, ttnn::DefaultQueueId, subdevice_target));
        input1_tensors_device.push_back(
            input_tensor1.to(devices.at(i), input_tensor1_mem_config, ttnn::DefaultQueueId, subdevice_target));
        output0_tensors_device.push_back(
            output_tensor0.to(devices.at(i), output_tensor0_mem_config, ttnn::DefaultQueueId, subdevice_target));
        output1_tensors_device.push_back(
            output_tensor1.to(devices.at(i), output_tensor1_mem_config, ttnn::DefaultQueueId, subdevice_target));
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
            devices, subdevice_managers, line_fabric.value(), tt::fabric::TerminationSignal::IMMEDIATELY_TERMINATE);

        log_info(tt::LogTest, "Finished");
        for (auto d : devices) {
            tt_metal::Synchronize(d, ttnn::DefaultQueueId);
        }
    }
    return pass;
}

////////////////////////////////////////////////////////////////////
///  MESSAGE COUNT TERMINATION MODE
////////////////////////////////////////////////////////////////////

TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_SingleMessage) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_2_messages) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 2;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}
// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_10_messages) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender and receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_20_messages) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 20;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}

TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}

// -------------------------
// Persistent Fabric
// -------------------------

TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_SingleMessage_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_2_messages_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 2;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}
// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_10_messages_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender and receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_20_messages_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 20;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}

TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}

////////////////////////////////

TEST(WorkerFabricEdmDatapath, LineFabricMcast_SingleMessage_SingleSource) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram, false);

    ASSERT_EQ(result, 0);
}

// Non-functional on harvested parts. Needs testing on unharvested parts.
TEST(WorkerFabricEdmDatapath, LineFabricMcast_ManyMessages_SingleSource) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram, false);

    ASSERT_EQ(result, 0);
}

TEST(WorkerFabricEdmDatapath, LineFabricMcast_SingleMessage_SingleSource_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram, true);

    ASSERT_EQ(result, 0);
}

// Non-functional on harvested parts. Needs testing on unharvested parts.
TEST(WorkerFabricEdmDatapath, LineFabricMcast_ManyMessages_SingleSource_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram, true);

    ASSERT_EQ(result, 0);
}

#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
////               LOCAL CHIP TENSOR READ?WRITE (2 INPUT)
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

ttnn::ccl::Shape4D<uint32_t> shape_to_shape_in_tiles(ttnn::Shape const& shape) {
    auto logical_shape = shape.logical_shape();
    logical_shape[-2] /= tt::constants::TILE_HEIGHT;
    logical_shape[-1] /= tt::constants::TILE_WIDTH;
    EXPECT_TRUE(logical_shape.size() == 4);
    ttnn::ccl::Shape4D<uint32_t> shape_in_tiles = {
        logical_shape[0], logical_shape[1], logical_shape[2], logical_shape[3]};
    return shape_in_tiles;
}

bool RunMultiInputReaderTestPropagateFullTensorIn(
    ttnn::Shape const& tensor_shape,
    Layout const& layout,
    MemoryConfig const& in0_memory_config,
    MemoryConfig const& in1_memory_config,
    MemoryConfig const& out0_memory_config,
    MemoryConfig const& out1_memory_config,
    TwoInputReaderKernelWriteMode test_writeback_mode) {
    auto logical_shape = tensor_shape.logical_shape();
    auto num_elems = std::reduce(logical_shape.cbegin(), logical_shape.cend(), 1, std::multiplies<uint32_t>());
    Tensor input_tensor0 =
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::UINT32), tensor_shape).to(layout);
    Tensor input_tensor1 =
        ttnn::experimental::view(ttnn::arange(num_elems, 2 * num_elems, 1, DataType::UINT32), tensor_shape).to(layout);
    Tensor output_tensor0 = ttnn::experimental::view(ttnn::ones(tensor_shape, DataType::UINT32, layout), tensor_shape);
    Tensor output_tensor1 = ttnn::experimental::view(ttnn::ones(tensor_shape, DataType::UINT32, layout), tensor_shape);
    input_tensor0.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), in0_memory_config)));
    input_tensor1.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), in1_memory_config)));
    output_tensor0.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), out0_memory_config)));
    output_tensor1.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), out1_memory_config)));

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

    auto const in0_tensor_slice = tensor_slice;
    auto const in1_tensor_slice = tensor_slice;
    auto const out0_tensor_slice = tensor_slice;
    auto const out1_tensor_slice = tensor_slice;

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

TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_SinglePageTile) {
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        ttnn::Shape{1, 1, 32, 32},
        Layout::TILE,
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0) {
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        ttnn::Shape{1, 1, 32, 64},
        Layout::TILE,
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded) {
    ttnn::Shape tensor_shape = {1, 1, 32, 64};
    auto logical_shape = tensor_shape.logical_shape();
    auto mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}}},
            {logical_shape[0] * logical_shape[1] * logical_shape[2], logical_shape[3]},
            ShardOrientation::ROW_MAJOR,
            false,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config,
        mem_config,
        mem_config,
        mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded1) {
    ttnn::Shape tensor_shape = {1, 1, 32, 128};
    auto logical_shape = tensor_shape.logical_shape();
    auto mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}}},
            {logical_shape[0] * logical_shape[1] * logical_shape[2], logical_shape[3]},
            ShardOrientation::ROW_MAJOR,
            false,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config,
        mem_config,
        mem_config,
        mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded2) {
    ttnn::Shape tensor_shape = {1, 1, 32, 128};
    auto logical_shape = tensor_shape.logical_shape();
    auto mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}}},
            {logical_shape[0] * logical_shape[1] * logical_shape[2], logical_shape[3] / 4},
            ShardOrientation::ROW_MAJOR,
            false,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config,
        mem_config,
        mem_config,
        mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded3) {
    ttnn::Shape tensor_shape = {1, 1, 32, 8192};
    auto logical_shape = tensor_shape.logical_shape();
    size_t ncores_x = 8;
    size_t ncores_y = 4;
    auto mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{ncores_x - 1, ncores_y - 1}}}},
            {logical_shape[0] * logical_shape[1] * logical_shape[2], logical_shape[3] / (ncores_x * ncores_y)},
            ShardOrientation::ROW_MAJOR,
            false,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config,
        mem_config,
        mem_config,
        mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded4) {
    ttnn::Shape tensor_shape = {1, 1, 32, 1024};
    auto logical_shape = tensor_shape.logical_shape();
    size_t ncores_x = 8;
    size_t ncores_y = 4;
    auto mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{ncores_x - 1, ncores_y - 1}}}},
            {logical_shape[0] * logical_shape[1] * logical_shape[2], logical_shape[3] / (ncores_x * ncores_y)},
            ShardOrientation::ROW_MAJOR,
            false,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config,
        mem_config,
        mem_config,
        mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded_WithReshard0) {
    ttnn::Shape tensor_shape = {1, 1, 32, 128};
    auto logical_shape = tensor_shape.logical_shape();
    Layout const layout = Layout::TILE;
    auto input_mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}}},
            {logical_shape[0] * logical_shape[1] * logical_shape[2], logical_shape[3]},
            ShardOrientation::ROW_MAJOR,
            false,
            ShardMode::LOGICAL));
    auto output_mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}}},
            {logical_shape[0] * logical_shape[1] * logical_shape[2], logical_shape[3] / 4},
            ShardOrientation::ROW_MAJOR,
            false,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        input_mem_config,
        input_mem_config,
        output_mem_config,
        output_mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded_WithReshard0_UniquePerStream) {
    ttnn::Shape tensor_shape = {1, 1, 32, 128};
    auto logical_shape = tensor_shape.logical_shape();
    Layout const layout = Layout::TILE;
    size_t in_shard_grid_x = 1;
    size_t in_shard_grid_y = 1;
    size_t out_shard_grid_x = 4;
    size_t out_shard_grid_y = 1;
    auto mem_config0 = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{
                std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{in_shard_grid_x - 1, in_shard_grid_y - 1}}}},
            {logical_shape[0] * logical_shape[1] * logical_shape[2],
             logical_shape[3] / (in_shard_grid_x * in_shard_grid_y)},
            ShardOrientation::ROW_MAJOR,
            false,
            ShardMode::LOGICAL));
    auto mem_config1 = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{
                std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{out_shard_grid_x - 1, out_shard_grid_y - 1}}}},
            {logical_shape[0] * logical_shape[1] * logical_shape[2],
             logical_shape[3] / (out_shard_grid_x * out_shard_grid_y)},
            ShardOrientation::ROW_MAJOR,
            false,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config0,
        mem_config1,
        mem_config1,
        mem_config0,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

// Copying even slightly large tensors exposes issues in underlying tensor code
// that isn't under test here
TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage1) {
    ttnn::Shape tensor_shape = {1, 1, 256, 256};
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

// TODO: update the test infra to be able to properly compare tensors if we are only
// doing a slice of the larger tensor

// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////
// ////               FABRIC UNICAST TENSOR WRITE (2 INPUT)
// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////

TEST(WorkerCclCommandProcessingKernelFabricUnicastMode, MultiInputReader_SinglePageTile_OneHop) {
    ttnn::Shape tensor_shape = {1, 1, 32, 32};
    constexpr size_t distance_dest_device = 1;
    constexpr size_t num_devices = 4;
    auto logical_shape = tensor_shape.logical_shape();
    Layout const layout = Layout::TILE;
    MemoryConfig const in0_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const in1_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const out0_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const out1_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);

    auto num_elems = std::reduce(logical_shape.cbegin(), logical_shape.cend(), 1, std::multiplies<uint32_t>());
    Tensor input_tensor0 =
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::UINT32), tensor_shape).to(layout);
    Tensor input_tensor1 =
        ttnn::experimental::view(ttnn::arange(num_elems, 2 * num_elems, 1, DataType::UINT32), tensor_shape).to(layout);
    Tensor output_tensor0 =
        ttnn::experimental::view(ttnn::ones(tensor_shape.value, DataType::UINT32, layout), tensor_shape);
    Tensor output_tensor1 =
        ttnn::experimental::view(ttnn::ones(tensor_shape.value, DataType::UINT32, layout), tensor_shape);

    input_tensor0.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), in0_memory_config)));
    input_tensor1.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), in1_memory_config)));
    output_tensor0.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), out0_memory_config)));
    output_tensor1.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), out1_memory_config)));

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

    auto const in0_tensor_slice = tensor_slice;
    auto const in1_tensor_slice = tensor_slice;
    auto const out0_tensor_slice = tensor_slice;
    auto const out1_tensor_slice = tensor_slice;

    ttnn::ccl::cmd::CclCommandDestArgs dest_args = ttnn::ccl::cmd::UnicastCommandDestArgs{distance_dest_device, true};
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
        TwoInputReaderKernelWriteMode::FABRIC_UNICAST,
        dest_args,
        false);

    ASSERT_TRUE(pass);
}

TEST(WorkerCclCommandProcessingKernelFabricUnicastMode, MultiInputReader_SinglePageTile_OneHop_PersistentFabric) {
    ttnn::Shape tensor_shape = {1, 1, 32, 32};
    constexpr size_t distance_dest_device = 1;
    constexpr size_t num_devices = 4;
    auto logical_shape = tensor_shape.logical_shape();
    Layout const layout = Layout::TILE;
    MemoryConfig const in0_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const in1_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const out0_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const out1_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);

    auto num_elems = std::reduce(logical_shape.cbegin(), logical_shape.cend(), 1, std::multiplies<uint32_t>());
    Tensor input_tensor0 =
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::UINT32), tensor_shape).to(layout);
    Tensor input_tensor1 =
        ttnn::experimental::view(ttnn::arange(num_elems, 2 * num_elems, 1, DataType::UINT32), tensor_shape).to(layout);
    Tensor output_tensor0 =
        ttnn::experimental::view(ttnn::ones(tensor_shape.value, DataType::UINT32, layout), tensor_shape);
    Tensor output_tensor1 =
        ttnn::experimental::view(ttnn::ones(tensor_shape.value, DataType::UINT32, layout), tensor_shape);

    input_tensor0.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), in0_memory_config)));
    input_tensor1.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), in1_memory_config)));
    output_tensor0.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), out0_memory_config)));
    output_tensor1.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), out1_memory_config)));

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

    auto const in0_tensor_slice = tensor_slice;
    auto const in1_tensor_slice = tensor_slice;
    auto const out0_tensor_slice = tensor_slice;
    auto const out1_tensor_slice = tensor_slice;

    ttnn::ccl::cmd::CclCommandDestArgs dest_args = ttnn::ccl::cmd::UnicastCommandDestArgs{distance_dest_device, true};
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
        TwoInputReaderKernelWriteMode::FABRIC_UNICAST,
        dest_args,
        true);

    ASSERT_TRUE(pass);
}

// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////
// ////               FABRIC MCAST TENSOR WRITE (2 INPUT)
// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////

void RunFabricMcastFullTensorPropagateTest(
    ttnn::Shape const& tensor_shape, size_t distance_dest_device, size_t num_devices, bool enable_persistent_fabric) {
    auto logical_shape = tensor_shape.logical_shape();
    Layout const layout = Layout::TILE;
    MemoryConfig const in0_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const in1_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const out0_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const out1_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);

    auto num_elems = std::reduce(logical_shape.cbegin(), logical_shape.cend(), 1, std::multiplies<uint32_t>());
    Tensor input_tensor1 =
        ttnn::experimental::view(ttnn::arange(num_elems, 2 * num_elems, 1, DataType::UINT32), tensor_shape).to(layout);
    Tensor input_tensor0 =
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::UINT32), tensor_shape).to(layout);
    Tensor output_tensor1 =
        ttnn::experimental::view(ttnn::ones(tensor_shape.value, DataType::UINT32, layout), tensor_shape);
    Tensor output_tensor0 =
        ttnn::experimental::view(ttnn::ones(tensor_shape.value, DataType::UINT32, layout), tensor_shape);
    input_tensor0.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), in0_memory_config)));
    input_tensor1.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), in1_memory_config)));
    output_tensor0.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), out0_memory_config)));
    output_tensor1.set_tensor_spec(TensorSpec(
        logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), out1_memory_config)));
    ASSERT_EQ(input_tensor0.get_logical_shape(), tensor_shape.logical_shape());
    ASSERT_EQ(input_tensor1.get_logical_shape(), tensor_shape.logical_shape());
    ASSERT_EQ(output_tensor0.get_logical_shape(), tensor_shape.logical_shape());
    ASSERT_EQ(output_tensor1.get_logical_shape(), tensor_shape.logical_shape());

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

    auto const in0_tensor_slice = tensor_slice;
    auto const in1_tensor_slice = tensor_slice;
    auto const out0_tensor_slice = tensor_slice;
    auto const out1_tensor_slice = tensor_slice;

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

TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_SinglePageTile_SingleHop) {
    ttnn::Shape tensor_shape = {1, 1, 32, 32};
    constexpr size_t distance_dest_device = 1;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, false);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_SinglePageTile_TwoHop) {
    ttnn::Shape tensor_shape = {1, 1, 32, 32};
    constexpr size_t distance_dest_device = 2;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, false);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_SinglePageTile_ThreeHop) {
    ttnn::Shape tensor_shape = {1, 1, 32, 32};
    constexpr size_t distance_dest_device = 3;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, false);
}

TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_4PageTile_SingleHop) {
    ttnn::Shape tensor_shape = {1, 1, 32, 128};
    constexpr size_t distance_dest_device = 1;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, false);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, DMultiInputReader_4PageTile_TwoHop) {
    ttnn::Shape tensor_shape = {1, 1, 128, 32};
    constexpr size_t distance_dest_device = 2;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, false);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_4PageTile_ThreeHop) {
    ttnn::Shape tensor_shape = {1, 1, 64, 64};
    constexpr size_t distance_dest_device = 3;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, false);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_lotsPageTile_ThreeHop) {
    ttnn::Shape tensor_shape = {1, 1, 64, 16384};
    constexpr size_t distance_dest_device = 3;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, false);
}

TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_SinglePageTile_SingleHop_PersistentFabric) {
    ttnn::Shape tensor_shape = {1, 1, 32, 32};
    constexpr size_t distance_dest_device = 1;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}

TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_SinglePageTile_TwoHop_PersistentFabric) {
    ttnn::Shape tensor_shape = {1, 1, 32, 32};
    constexpr size_t distance_dest_device = 2;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_SinglePageTile_ThreeHop_PersistentFabric) {
    ttnn::Shape tensor_shape = {1, 1, 32, 32};
    constexpr size_t distance_dest_device = 3;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}

TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_4PageTile_SingleHop_PersistentFabric) {
    ttnn::Shape tensor_shape = {1, 1, 32, 128};
    constexpr size_t distance_dest_device = 1;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, DMultiInputReader_4PageTile_TwoHop_PersistentFabric) {
    ttnn::Shape tensor_shape = {1, 1, 128, 32};
    constexpr size_t distance_dest_device = 2;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_4PageTile_ThreeHop_PersistentFabric) {
    ttnn::Shape tensor_shape = {1, 1, 64, 64};
    constexpr size_t distance_dest_device = 3;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_lotsPageTile_ThreeHop_PersistentFabric) {
    ttnn::Shape tensor_shape = {1, 1, 64, 16384};
    constexpr size_t distance_dest_device = 3;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
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

    auto logical_shape = tensor_shape.logical_shape();
    auto const cb_index = tt::CB::c_in0;

    auto programs = std::vector<Program>(1);
    Program& program = programs[0];

    T3000TestDevice test_fixture;
    auto view = test_fixture.mesh_device_->get_view();

    Device* device = view.get_device(0, 0);
    ;

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
            logical_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), mem_configs[i])));
    }

    // Allocate the tensors - pull to function
    const size_t num_tensors = num_stages + 1;
    std::vector<Tensor> host_tensors;
    std::vector<Tensor> device_tensors;
    host_tensors.reserve(num_tensors);
    device_tensors.reserve(num_tensors);
    auto num_elems = std::reduce(logical_shape.cbegin(), logical_shape.cend(), 1, std::multiplies<uint32_t>());
    host_tensors.push_back(
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::UINT32), tensor_shape).to(layout));
    for (size_t i = 1; i < num_tensors; ++i) {
        host_tensors.push_back(
            ttnn::experimental::view(ttnn::ones(tensor_shape.value, DataType::UINT32, layout), tensor_shape));
    }
    TT_FATAL(mem_configs.size() == num_tensors, "Must have a memory config for each tensor");
    for (size_t i = 0; i < num_tensors; i++) {
        host_tensors[i].set_tensor_spec(tensor_specs[i]);
        device_tensors.push_back(host_tensors[i].to(device, mem_configs[i]));
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

        auto in_tensor_copyback = tt::tt_metal::owned_buffer::get_as<uint32_t>(input_cpu);
        auto out_tensor_copyback = tt::tt_metal::owned_buffer::get_as<uint32_t>(final_out_cpu);

        auto in_tensor_data = tt::tt_metal::owned_buffer::get_as<uint32_t>(host_tensors[0]);

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

TEST(WorkerCclCommandProcessingKernels, ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly0) {
    ttnn::Shape tensor_shape = {1, 1, 64, 16384};
    auto logical_shape = tensor_shape.logical_shape();
    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const size_t slices_per_stage = 4;
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<size_t> num_workers_per_stage = {1, 1, 1, 1};

    std::vector<std::vector<size_t>> worker_chunk_read_order = {
        {0, 1, 2, 3},  // first input
        {3, 2, 1, 0},  // read in reverse order
        {2, 0, 3, 1},  // read in non-sequential order
        {1, 2, 3, 0}   // read in non-sequential order
    };
    std::vector<MemoryConfig> mem_configs{
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)};

    auto pass = RunPipelinedWorkersTest(

        tensor_shape,
        split_dim,

        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
        num_stages,
        num_workers_per_stage,
        slices_per_stage,
        data_format,
        page_size_bytes,
        cb_packet_size_in_pages,
        num_packets_per_cb,
        layout,

        worker_chunk_read_order,
        mem_configs);

    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernels, ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly1) {
    ttnn::Shape tensor_shape = {1, 1, 64, 128};
    auto logical_shape = tensor_shape.logical_shape();
    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const size_t slices_per_stage = 4;
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<size_t> num_workers_per_stage = {1, 1, 1, 1};

    std::vector<std::vector<size_t>> worker_chunk_read_order = {
        {0, 1, 2, 3},  // first input
        {3, 2, 1, 0},  // read in reverse order
        {2, 0, 3, 1},  // read in non-sequential order
        {1, 2, 3, 0}   // read in non-sequential order
    };
    std::vector<MemoryConfig> mem_configs{
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)};

    auto pass = RunPipelinedWorkersTest(

        tensor_shape,
        split_dim,

        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
        num_stages,
        num_workers_per_stage,
        slices_per_stage,
        data_format,
        page_size_bytes,
        cb_packet_size_in_pages,
        num_packets_per_cb,
        layout,

        worker_chunk_read_order,
        mem_configs);

    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernels, ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly2) {
    ttnn::Shape tensor_shape = {1, 1, 64, 8192};
    auto logical_shape = tensor_shape.logical_shape();
    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const size_t slices_per_stage = 2;
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<size_t> num_workers_per_stage = {1, 1, 1, 1};

    std::vector<std::vector<size_t>> worker_chunk_read_order = {
        {0, 1},  // first input
        {1, 0},  // read in reverse order
        {1, 0},  // read in non-sequential order
        {0, 1}   // read in non-sequential order
    };
    std::vector<MemoryConfig> mem_configs{
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)};

    auto pass = RunPipelinedWorkersTest(

        tensor_shape,
        split_dim,

        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
        num_stages,
        num_workers_per_stage,
        slices_per_stage,
        data_format,
        page_size_bytes,
        cb_packet_size_in_pages,
        num_packets_per_cb,
        layout,

        worker_chunk_read_order,
        mem_configs);

    ASSERT_TRUE(pass);
}

// Hits issues with input tensor copy-back
TEST(
    WorkerCclCommandProcessingKernels,
    DISABLED_ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly_SmallSweep) {
    std::vector<ttnn::Shape> tensor_shapes = {
        {1, 1, 64, 8192}, {1, 4, 64, 768}, {4, 1, 64, 768}, {4, 4, 64, 768}, {1, 1, 64, 768}, {5, 3, 64, 768}};

    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const std::vector<size_t> slices_per_stage_sweep = {2, 3, 4};
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<std::vector<size_t>> num_workers_per_stage_sweep = {
        {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};

    std::vector<std::vector<std::vector<size_t>>> worker_chunk_read_order = {
        {{}},
        {
            {0},
            {0},
            {0},
            {0},
        },
        {
            {0, 1},
            {1, 0},
            {1, 0},
            {0, 1},
        },
        {
            {2, 0, 1},
            {1, 0, 2},
            {0, 1, 2},
            {2, 1, 0},
        },
        {
            {0, 1, 2, 3},  // first input
            {3, 2, 1, 0},  // read in reverse order
            {2, 0, 3, 1},  // read in non-sequential order
            {1, 2, 3, 0}   // read in non-sequential order
        }};
    std::vector<std::vector<MemoryConfig>> mem_configs_sweep = {
        {
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        },
        {MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1)},
        {MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)},
        {MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)},
    };

    for (auto& tensor_shape : tensor_shapes) {
        for (auto& num_workers_per_stage : num_workers_per_stage_sweep) {
            for (size_t slices_per_stage : slices_per_stage_sweep) {
                for (auto& mem_configs : mem_configs_sweep) {
                    log_info(
                        tt::LogTest,
                        "tensor shape {} and workers stage {} slices_per_stage {}",
                        tensor_shape,
                        num_workers_per_stage,
                        slices_per_stage);
                    auto pass = RunPipelinedWorkersTest(

                        tensor_shape,
                        split_dim,

                        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be
                        // configurable)
                        num_stages,
                        num_workers_per_stage,
                        slices_per_stage,
                        data_format,
                        page_size_bytes,
                        cb_packet_size_in_pages,
                        num_packets_per_cb,
                        layout,

                        worker_chunk_read_order[slices_per_stage],
                        mem_configs);

                    ASSERT_TRUE(pass);
                }
            }
        }
    }
}

#include "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include "tt_metal/common/bfloat16.hpp"
TEST(CclAsyncOp, ReduceScatterSmall_PersistentFabric) {
    const size_t dim = 3;
    const size_t num_links = 1;
    constexpr auto layout = Layout::TILE;
    // DEVICES setup
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
    T3000TestDevice test_fixture;
    auto view = test_fixture.mesh_device_->get_view();

    // build a line of devices
    std::vector<Device*> devices = {
        view.get_device(0, 1), view.get_device(1, 1), view.get_device(1, 2), view.get_device(0, 2)};
    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);
    const ttnn::Shape input_shape = ttnn::Shape{1, 1, 32, 32 * num_devices};
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    auto const logical_shape = input_shape.logical_shape();
    const auto num_elems = logical_shape.volume();

    // INPUT TENSOR setup
    size_t page_size = tile_size(DataFormat::Float16);
    std::vector<Tensor> device_input_tensors;
    for (size_t i = 0; i < num_devices; i++) {
        // host_input_tensors.push_back(ttnn::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f) ,
        // {logical_shape[0],logical_shape[1],logical_shape[2],logical_shape[3]}, layout).to(devices[i]));
        auto t = ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::BFLOAT16), input_shape).to(layout);
        t.set_tensor_spec(TensorSpec(
            logical_shape, TensorLayout(DataType::BFLOAT16, PageConfig(layout, tt_metal::Tile()), in_memory_config)));

        device_input_tensors.push_back(t.to(devices[i]));
    }
    // Need to make it a mesh tensor for use with the op
    const Tensor input_mesh_tensor = ttnn::distributed::aggregate_as_tensor(device_input_tensors, AllGatherTensor{});

    // FABRIC setup
    const bool enable_persistent_fabric = true;

    std::vector<Program> dummy_worker_programs;
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle;
    setup_test_with_persistent_fabric(
        devices,
        dummy_worker_programs,
        subdevice_managers,
        fabric_programs,
        fabric_program_ptrs,
        fabric_handle,
        enable_persistent_fabric,
        num_links);

    auto output_tensor = ttnn::operations::experimental::ccl::reduce_scatter(
        input_mesh_tensor,
        dim,
        ttnn::operations::reduction::ReduceType::Sum,
        operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        ttnn::ccl::Topology::Linear,
        num_links,
        subdevice_managers->worker_subdevice_id.at(devices[0]->id()),
        true,
        fabric_handle);

    // wait for op completion
    log_info(tt::LogTest, "Waiting for Op finish");
    std::ranges::for_each(devices, [&](Device* d) {
        tt_metal::Finish(d->command_queue(), {subdevice_managers->worker_subdevice_id.at(d->id())});
    });
    log_info(tt::LogTest, "Main op done");

    log_info(tt::LogTest, "Fabric teardown");
    persistent_fabric_teardown_sequence(
        devices, subdevice_managers, fabric_handle.value(), tt::fabric::TerminationSignal::GRACEFULLY_TERMINATE);

    log_info(tt::LogTest, "Waiting for teardown completion");
    for (auto d : devices) {
        tt_metal::Synchronize(d, ttnn::DefaultQueueId);
    }
    log_info(tt::LogTest, "Finished");
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
    T3000TestDevice test_fixture;
    auto view = test_fixture.mesh_device_->get_view();

    // build a line of devices
    std::vector<Device*> devices = {
        view.get_device(0, 0), view.get_device(0, 1), view.get_device(0, 2), view.get_device(0, 3)};
    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    auto const logical_shape = input_shape.logical_shape();
    const auto num_elems = logical_shape.volume();

    // INPUT TENSOR setup
    log_info(tt::LogTest, "setting up input tensors");
    size_t page_size = tile_size(DataFormat::Float16);
    std::vector<Tensor> device_input_tensors;
    for (size_t i = 0; i < num_devices; i++) {
        auto t = ttnn::experimental::view(ttnn::arange(0, num_elems, 1), input_shape).to(layout);
        t.set_tensor_spec(TensorSpec(
            logical_shape, TensorLayout(DataType::BFLOAT16, PageConfig(layout, tt_metal::Tile()), in_memory_config)));

        device_input_tensors.push_back(t.to(devices[i]));
    }
    // Need to make it a mesh tensor for use with the op
    const Tensor input_mesh_tensor = ttnn::distributed::aggregate_as_tensor(device_input_tensors, AllGatherTensor{});

    // FABRIC setup
    const bool enable_persistent_fabric = true;

    std::vector<Program> dummy_worker_programs;
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle;
    setup_test_with_persistent_fabric(
        devices,
        dummy_worker_programs,
        subdevice_managers,
        fabric_programs,
        fabric_program_ptrs,
        fabric_handle,
        enable_persistent_fabric,
        num_links);
    log_info(tt::LogTest, "Lauching op");

    auto output_tensor = ttnn::operations::experimental::ccl::all_gather_async(
        input_mesh_tensor,
        dim,
        num_links,
        operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        ttnn::ccl::Topology::Linear,
        SubDeviceId(0),
        true);

    // wait for op completion
    log_info(tt::LogTest, "Waiting for Op finish");
    std::ranges::for_each(devices, [&](Device* d) {
        tt_metal::Finish(d->command_queue(), {subdevice_managers->worker_subdevice_id.at(d->id())});
    });
    log_info(tt::LogTest, "Main op done");

    log_info(tt::LogTest, "Fabric teardown");
    persistent_fabric_teardown_sequence(
        devices, subdevice_managers, fabric_handle.value(), tt::fabric::TerminationSignal::IMMEDIATELY_TERMINATE);

    log_info(tt::LogTest, "Waiting for teardown completion");
    for (auto d : devices) {
        tt_metal::Synchronize(d, ttnn::DefaultQueueId);
    }
    log_info(tt::LogTest, "Finished");
}

TEST(CclAsyncOp, AllGather_PersistentFabric_Dim3_Links1_Shape1_1_32_128) {
    run_all_gather_with_persistent_fabric(3, 1, ttnn::Shape{1, 1, 32, 128});
}
TEST(CclAsyncOp, AllGather_PersistentFabric_Dim3_Links1_Shape1_1_32_8192) {
    run_all_gather_with_persistent_fabric(3, 1, ttnn::Shape{1, 1, 32, 8192});
}
// Mesh device setup seems to not provide the correct configuration for multi-link? To be investigated
TEST(CclAsyncOp, DISABLED_AllGather_PersistentFabric_Dim3_Links2_Shape1_1_32_128) {
    run_all_gather_with_persistent_fabric(3, 2, ttnn::Shape{1, 1, 32, 128});
}
// Mesh device setup seems to not provide the correct configuration for multi-link? To be investigated
TEST(CclAsyncOp, DISABLED_AllGather_PersistentFabric_Dim3_Links2_Shape1_1_32_8192) {
    run_all_gather_with_persistent_fabric(3, 2, ttnn::Shape{1, 1, 32, 8192});
}
