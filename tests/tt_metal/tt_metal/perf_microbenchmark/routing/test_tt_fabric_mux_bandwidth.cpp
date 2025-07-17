// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <string>
#include <optional>
#include <chrono>
#include <stdint.h>
#include <vector>
#include <map>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "test_common.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/profiler/profiler_paths.hpp"

const std::string mux_kernel_src = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
const std::string drainer_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_mux_ubench_drainer.cpp";
const std::string worker_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_mux_ubench_sender.cpp";

const uint32_t test_results_size_bytes = 128;
const uint32_t noc_address_padding_bytes = 16;
const uint32_t packet_header_buffer_size_bytes = 1024;

struct TestParams {
    size_t num_full_size_channels = 0;
    size_t num_header_only_channels = 0;
    uint8_t num_buffers_full_size_channel = 0;
    uint8_t num_buffers_header_only_channel = 0;
    uint32_t num_packets = 0;
    uint32_t packet_payload_size_bytes = 0;
    size_t num_full_size_channel_iters = 0;
    size_t num_iters_between_teardown_checks = 0;
    size_t buffer_size_bytes_full_size_channel = 0;
    size_t buffer_size_bytes_header_only_channel = 0;
};

struct MuxTestConfig {
    tt::tt_fabric::FabricMuxConfig* mux_kernel_config = nullptr;
    CoreCoord mux_logical_core;
    CoreCoord mux_virtual_core;
};

struct DrainerTestConfig {
    // we can re-use the mux infra to get the address layout for the drainer kernel as well
    tt::tt_fabric::FabricMuxConfig* drainer_kernel_config = nullptr;
    CoreCoord drainer_logical_core;
    CoreCoord drainer_virtual_core;
};

struct WorkerMemoryMap {
    uint32_t test_results_address = 0;
    uint32_t workers_sync_address = 0;
    uint32_t local_mux_status_address = 0;
    uint32_t local_flow_control_address = 0;
    uint32_t local_teardown_address = 0;
    uint32_t local_buffer_index_address = 0;
    uint32_t packet_header_buffer_address = 0;
    uint32_t target_address = 0;
    uint32_t payload_buffer_address = 0;
};

struct WorkerTestConfig {
    WorkerMemoryMap* memory_map = nullptr;
    CoreCoord worker_logical_core;
    uint8_t worker_id = 0;
    tt::tt_fabric::FabricMuxChannelType channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    uint32_t num_buffers = 0;
    uint32_t buffer_size_bytes = 0;
    uint32_t receiver_noc_xy_encoding = 0;
    std::optional<uint32_t> mcast_encoding = std::nullopt;
    std::optional<uint32_t> num_mcast_dests = std::nullopt;
};

WorkerMemoryMap create_worker_memory_map(const uint32_t base_l1_address) {
    WorkerMemoryMap worker_memory_map;

    worker_memory_map.test_results_address = base_l1_address;
    worker_memory_map.workers_sync_address = worker_memory_map.test_results_address + test_results_size_bytes;
    worker_memory_map.local_mux_status_address = worker_memory_map.workers_sync_address + noc_address_padding_bytes;
    worker_memory_map.local_flow_control_address =
        worker_memory_map.local_mux_status_address + noc_address_padding_bytes;
    worker_memory_map.local_teardown_address = worker_memory_map.local_flow_control_address + noc_address_padding_bytes;
    worker_memory_map.local_buffer_index_address = worker_memory_map.local_teardown_address + noc_address_padding_bytes;
    worker_memory_map.packet_header_buffer_address =
        worker_memory_map.local_buffer_index_address + noc_address_padding_bytes;
    worker_memory_map.target_address = worker_memory_map.packet_header_buffer_address + packet_header_buffer_size_bytes;
    worker_memory_map.payload_buffer_address = worker_memory_map.target_address;

    return worker_memory_map;
}

void create_kernel(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program_handle,
    const std::string& kernel_src,
    const CoreCoord& logical_core,
    const std::vector<uint32_t>& ct_args,
    const std::vector<uint32_t>& rt_args,
    const std::vector<std::pair<size_t, size_t>>& addresses_to_clear) {
    auto kernel_handle = tt::tt_metal::CreateKernel(
        program_handle,
        kernel_src,
        {logical_core},
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
    tt::tt_metal::SetRuntimeArgs(program_handle, kernel_handle, logical_core, rt_args);

    for (const auto& [start_address, num_bytes] : addresses_to_clear) {
        std::vector<uint32_t> zero_vec((num_bytes / sizeof(uint32_t)), 0);
        tt::tt_metal::detail::WriteToDeviceL1(device, logical_core, start_address, zero_vec);
    }
}

void create_mux_kernel(
    const TestParams& test_params,
    const MuxTestConfig& mux_test_config,
    const DrainerTestConfig& drainer_test_config,
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program_handle) {
    auto mux_kernel_config = mux_test_config.mux_kernel_config;
    auto drainer_kernel_config = drainer_test_config.drainer_kernel_config;
    auto mux_logical_core = mux_test_config.mux_logical_core;

    // getting mux ct args like this will result in compilation error if fabric is not enabled
    // and may not work properly since we need drainer's status address and num buffers, but by
    // default mux config works with edm's status address and buffers
    // std::vector<uint32_t> mux_ct_args = mux_kernel_config->get_fabric_mux_compile_time_args();

    auto default_channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    size_t mux_status_address = mux_kernel_config->get_status_address();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    std::vector<uint32_t> mux_ct_args = {
        test_params.num_full_size_channels,
        test_params.num_buffers_full_size_channel,
        test_params.buffer_size_bytes_full_size_channel,
        test_params.num_header_only_channels,
        test_params.num_buffers_header_only_channel,
        mux_status_address,
        mux_kernel_config->get_termination_signal_address(),
        mux_kernel_config->get_connection_info_address(default_channel_type, 0),
        mux_kernel_config->get_connection_handshake_address(default_channel_type, 0),
        mux_kernel_config->get_flow_control_address(default_channel_type, 0),
        mux_kernel_config->get_channel_base_address(default_channel_type, 0),
        mux_status_address + noc_address_padding_bytes,  // risky, could change if mux address map is updated
        drainer_kernel_config->get_status_address(),
        drainer_kernel_config->get_num_buffers(default_channel_type),
        test_params.num_full_size_channel_iters,
        test_params.num_iters_between_teardown_checks,
        hal.get_programmable_core_type_index(tt::tt_metal::HalProgrammableCoreType::TENSIX)};

    // semaphores needed to build connection with drainer core using the build_from_args API
    auto worker_flow_control_semaphore_id = tt::tt_metal::CreateSemaphore(program_handle, mux_logical_core, 0);
    auto worker_teardown_semaphore_id = tt::tt_metal::CreateSemaphore(program_handle, mux_logical_core, 0);
    auto worker_buffer_index_semaphore_id = tt::tt_metal::CreateSemaphore(program_handle, mux_logical_core, 0);

    auto memory_regions_to_clear = mux_kernel_config->get_memory_regions_to_clear();
    std::vector<uint32_t> memory_regions_to_clear_args;
    memory_regions_to_clear_args.reserve(memory_regions_to_clear.size() * 2 + 1);
    memory_regions_to_clear_args.push_back(static_cast<uint32_t>(memory_regions_to_clear.size()));
    for (const auto& [address, size] : memory_regions_to_clear) {
        memory_regions_to_clear_args.push_back(static_cast<uint32_t>(address));
        memory_regions_to_clear_args.push_back(static_cast<uint32_t>(size));
    }

    // mux to drainer will always be a full size channel connection
    auto drainer_channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    tt::tt_fabric::SenderWorkerAdapterSpec sender_worker_adapter_spec{
        .edm_noc_x = drainer_test_config.drainer_virtual_core.x,
        .edm_noc_y = drainer_test_config.drainer_virtual_core.y,
        .edm_buffer_base_addr = drainer_kernel_config->get_channel_base_address(drainer_channel_type, 0),
        .num_buffers_per_channel = drainer_kernel_config->get_num_buffers(drainer_channel_type),
        .edm_l1_sem_addr = drainer_kernel_config->get_flow_control_address(drainer_channel_type, 0),
        .edm_connection_handshake_addr =
            drainer_kernel_config->get_connection_handshake_address(drainer_channel_type, 0),
        .edm_worker_location_info_addr = drainer_kernel_config->get_connection_info_address(drainer_channel_type, 0),
        .buffer_size_bytes = drainer_kernel_config->get_buffer_size_bytes(drainer_channel_type),
        .buffer_index_semaphore_id = drainer_kernel_config->get_buffer_index_address(drainer_channel_type, 0),
        .edm_direction = tt::tt_fabric::eth_chan_directions::EAST, /* ignored, direction */
    };
    std::vector<uint32_t> mux_fabric_connection_rt_args;
    tt::tt_fabric::append_worker_to_fabric_edm_sender_rt_args(
        sender_worker_adapter_spec,
        device->id(),
        {mux_logical_core},
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        mux_fabric_connection_rt_args);

    std::vector<uint32_t> mux_rt_args;
    mux_rt_args.reserve(memory_regions_to_clear_args.size() + mux_fabric_connection_rt_args.size());
    mux_rt_args.insert(mux_rt_args.end(), memory_regions_to_clear_args.begin(), memory_regions_to_clear_args.end());
    mux_rt_args.insert(mux_rt_args.end(), mux_fabric_connection_rt_args.begin(), mux_fabric_connection_rt_args.end());

    std::vector<std::pair<size_t, size_t>> addresses_to_clear = {};
    create_kernel(
        device, program_handle, mux_kernel_src, mux_logical_core, mux_ct_args, mux_rt_args, addresses_to_clear);
}

void create_drainer_kernel(
    const DrainerTestConfig& drainer_test_config,
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program_handle) {
    auto drainer_kernel_config = drainer_test_config.drainer_kernel_config;
    auto drainer_logical_core = drainer_test_config.drainer_logical_core;
    auto drainer_channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;

    std::vector<uint32_t> drainer_ct_args = {
        drainer_kernel_config->get_num_buffers(drainer_channel_type),
        drainer_kernel_config->get_buffer_size_bytes(drainer_channel_type),
        drainer_kernel_config->get_status_address(),
        drainer_kernel_config->get_termination_signal_address(),
        drainer_kernel_config->get_connection_info_address(drainer_channel_type, 0),
        drainer_kernel_config->get_connection_handshake_address(drainer_channel_type, 0),
        drainer_kernel_config->get_flow_control_address(drainer_channel_type, 0),
        drainer_kernel_config->get_channel_base_address(drainer_channel_type, 0)};

    auto memory_regions_to_clear = drainer_kernel_config->get_memory_regions_to_clear();
    std::vector<uint32_t> memory_regions_to_clear_args;
    memory_regions_to_clear_args.reserve(memory_regions_to_clear.size() * 2 + 1);
    memory_regions_to_clear_args.push_back(static_cast<uint32_t>(memory_regions_to_clear.size()));
    for (const auto& [address, size] : memory_regions_to_clear) {
        memory_regions_to_clear_args.push_back(static_cast<uint32_t>(address));
        memory_regions_to_clear_args.push_back(static_cast<uint32_t>(size));
    }

    std::vector<uint32_t> drainer_rt_args = memory_regions_to_clear_args;

    std::vector<std::pair<size_t, size_t>> addresses_to_clear = {};
    create_kernel(
        device,
        program_handle,
        drainer_kernel_src,
        drainer_logical_core,
        drainer_ct_args,
        drainer_rt_args,
        addresses_to_clear);
}

void create_worker_kernel(
    const TestParams& test_params,
    const WorkerTestConfig& worker_test_config,
    const MuxTestConfig& mux_test_config,
    const DrainerTestConfig& drainer_test_config,
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program_handle) {
    auto mux_kernel_config = mux_test_config.mux_kernel_config;
    auto channel_type = worker_test_config.channel_type;
    auto worker_id = worker_test_config.worker_id;

    std::vector<uint32_t> worker_ct_args = {
        mux_test_config.mux_virtual_core.x,
        mux_test_config.mux_virtual_core.y,
        worker_test_config.num_buffers,
        worker_test_config.buffer_size_bytes,
        mux_kernel_config->get_channel_base_address(channel_type, worker_id),
        mux_kernel_config->get_connection_info_address(channel_type, worker_id),
        mux_kernel_config->get_connection_handshake_address(channel_type, worker_id),
        mux_kernel_config->get_flow_control_address(channel_type, worker_id),
        mux_kernel_config->get_buffer_index_address(channel_type, worker_id),
        mux_kernel_config->get_status_address(),
        worker_test_config.mcast_encoding.has_value(),
        channel_type == tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config->get_channel_credits_stream_id(channel_type, worker_id)};

    auto worker_memory_map = worker_test_config.memory_map;
    std::vector<uint32_t> worker_rt_args = {
        test_params.num_packets,
        test_params.packet_payload_size_bytes,
        test_results_size_bytes,
        worker_memory_map->test_results_address,
        worker_memory_map->workers_sync_address,
        worker_memory_map->local_mux_status_address,
        worker_memory_map->local_flow_control_address,
        worker_memory_map->local_teardown_address,
        worker_memory_map->local_buffer_index_address,
        worker_memory_map->target_address,
        worker_memory_map->packet_header_buffer_address,
        worker_memory_map->payload_buffer_address,
        worker_test_config.receiver_noc_xy_encoding};

    if (worker_test_config.mcast_encoding.has_value()) {
        std::vector<uint32_t> additional_rt_args = {
            drainer_test_config.drainer_virtual_core.x,
            drainer_test_config.drainer_virtual_core.y,
            drainer_test_config.drainer_kernel_config->get_status_address(),
            worker_test_config.mcast_encoding.value(),
            worker_test_config.num_mcast_dests.value()};
        worker_rt_args.insert(worker_rt_args.end(), additional_rt_args.begin(), additional_rt_args.end());
    }

    std::vector<std::pair<size_t, size_t>> addresses_to_clear = {
        std::make_pair(worker_memory_map->workers_sync_address, noc_address_padding_bytes),
        std::make_pair(worker_memory_map->local_flow_control_address, noc_address_padding_bytes),
        std::make_pair(worker_memory_map->local_teardown_address, noc_address_padding_bytes),
        std::make_pair(worker_memory_map->local_buffer_index_address, noc_address_padding_bytes)};

    create_kernel(
        device,
        program_handle,
        worker_kernel_src,
        worker_test_config.worker_logical_core,
        worker_ct_args,
        worker_rt_args,
        addresses_to_clear);
}

int main(int argc, char** argv) {
    const std::string default_log_file_path =
        std::string(std::getenv("TT_METAL_HOME")) + "/generated/fabric_mux_bandwidth_temp.txt";
    const std::string default_test_name = "default_mux_ubench";
    const size_t default_num_full_size_channels = 8;
    const size_t default_num_header_only_channels = 0;
    const size_t default_num_buffers_full_size_channel = 8;
    const size_t default_num_buffers_header_only_channel = 8;
    const uint32_t default_num_packets = 10000;
    const uint32_t default_packet_payload_size_bytes = 4096;
    const size_t default_num_full_size_channel_iters = 1;
    const size_t default_num_iters_between_teardown_checks = 32;

    std::vector<std::string> input_args(argv, argv + argc);
    TestParams test_params;

    std::string log_file_path = test_args::get_command_option(input_args, "--log_file", default_log_file_path);
    std::string test_name = test_args::get_command_option(input_args, "--test_name", default_test_name);
    test_params.num_full_size_channels =
        test_args::get_command_option_uint32(input_args, "--num_full_size_channels", default_num_full_size_channels);
    test_params.num_header_only_channels = test_args::get_command_option_uint32(
        input_args, "--num_header_only_channels", default_num_header_only_channels);
    test_params.num_buffers_full_size_channel = test_args::get_command_option_uint32(
        input_args, "--num_buffers_full_size_channel", default_num_buffers_full_size_channel);
    test_params.num_buffers_header_only_channel = test_args::get_command_option_uint32(
        input_args, "--num_buffers_header_only_channel", default_num_buffers_header_only_channel);
    test_params.num_packets = test_args::get_command_option_uint32(input_args, "--num_packets", default_num_packets);
    test_params.packet_payload_size_bytes = test_args::get_command_option_uint32(
        input_args, "--packet_payload_size_bytes", default_packet_payload_size_bytes);
    test_params.num_full_size_channel_iters = test_args::get_command_option_uint32(
        input_args, "--num_full_size_channel_iters", default_num_full_size_channel_iters);
    test_params.num_iters_between_teardown_checks = test_args::get_command_option_uint32(
        input_args, "--num_iters_between_teardown_checks", default_num_iters_between_teardown_checks);

    // set to default only for compilation sake
    if (test_params.num_buffers_full_size_channel == 0) {
        test_params.num_buffers_full_size_channel = default_num_buffers_full_size_channel;
    }
    if (test_params.num_buffers_header_only_channel == 0) {
        test_params.num_buffers_header_only_channel = default_num_buffers_header_only_channel;
    }

    test_params.buffer_size_bytes_full_size_channel =
        sizeof(tt::tt_fabric::PacketHeader) + test_params.packet_payload_size_bytes;
    test_params.buffer_size_bytes_header_only_channel = sizeof(tt::tt_fabric::PacketHeader);

    tt::tt_fabric::SetFabricConfig(
        tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    std::vector<chip_id_t> all_device_ids;
    for (unsigned int id = 0; id < num_devices; id++) {
        all_device_ids.push_back(id);
    }
    std::map<chip_id_t, tt::tt_metal::IDevice*> devices = tt::tt_metal::detail::CreateDevices(all_device_ids);

    // for now, just use one device for running benchmarks
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_id = control_plane.get_user_physical_mesh_ids()[0];
    chip_id_t logical_chip_id = 0;
    auto physical_chip_id =
        control_plane.get_physical_chip_id_from_fabric_node_id(tt::tt_fabric::FabricNodeId(mesh_id, logical_chip_id));
    tt::tt_metal::IDevice* device = devices.at(physical_chip_id);
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    std::vector<CoreCoord> worker_logical_cores;
    auto grid_size = device->compute_with_storage_grid_size();
    for (auto i = 0; i < grid_size.x; i++) {
        for (auto j = 0; j < grid_size.y; j++) {
            worker_logical_cores.push_back(CoreCoord({i, j}));
        }
    }

    // core assignment, reserved (0,0) -> (0, grid_size.y) for mux/drainer etc
    CoreCoord mux_logical_core = worker_logical_cores[0];
    CoreCoord drainer_logical_core = worker_logical_cores[1];

    auto worker_cores_offset = grid_size.y;
    auto core_range_virtual_start = device->worker_core_from_logical_core(worker_logical_cores[worker_cores_offset]);
    auto core_range_virtual_end = device->worker_core_from_logical_core(worker_logical_cores.back());
    uint32_t mcast_encoding = tt::tt_metal::MetalContext::instance().hal().noc_multicast_encoding(
        core_range_virtual_start.x, core_range_virtual_start.y, core_range_virtual_end.x, core_range_virtual_end.y);
    uint32_t num_mcast_dests = worker_logical_cores.size() - grid_size.y;

    const uint32_t l1_unreserved_base_address =
        device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        test_params.num_full_size_channels,
        test_params.num_header_only_channels,
        test_params.num_buffers_full_size_channel,
        test_params.num_buffers_header_only_channel,
        test_params.buffer_size_bytes_full_size_channel,
        l1_unreserved_base_address);
    MuxTestConfig mux_test_config = {
        .mux_kernel_config = &mux_kernel_config,
        .mux_logical_core = mux_logical_core,
        .mux_virtual_core = device->worker_core_from_logical_core(mux_logical_core),
    };

    auto drainer_kernel_config = tt::tt_fabric::FabricMuxConfig(
        1,                                          /* num_full_size_channels */
        0,                                          /* num_header_only_channels */
        16,                                         /* num_buffers_full_size_channel */
        8,                                          /* num_buffers_header_only_channel */
        sizeof(tt::tt_fabric::PacketHeader) + 4096, /* buffer_size_bytes_full_size_channel (4K packet) */
        l1_unreserved_base_address);
    DrainerTestConfig drainer_test_config = {
        .drainer_kernel_config = &drainer_kernel_config,
        .drainer_logical_core = drainer_logical_core,
        .drainer_virtual_core = device->worker_core_from_logical_core(drainer_logical_core),
    };

    auto worker_memory_map = create_worker_memory_map(l1_unreserved_base_address);

    create_mux_kernel(test_params, mux_test_config, drainer_test_config, device, program);

    create_drainer_kernel(drainer_test_config, device, program);

    // keep the receiver noc xy encoding same for all workers, wont matter since we are not committing any
    // packets into receiver's L1
    CoreCoord default_receiver_virtual_core = device->worker_core_from_logical_core(worker_logical_cores.back());
    uint32_t default_receiver_noc_xy_encoding = tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(
        default_receiver_virtual_core.x, default_receiver_virtual_core.y);

    auto full_size_channel_worker_offset = worker_cores_offset;
    for (auto i = 0; i < test_params.num_full_size_channels; i++) {
        CoreCoord logical_core = worker_logical_cores[full_size_channel_worker_offset + i];
        WorkerTestConfig worker_test_config = {
            .memory_map = &worker_memory_map,
            .worker_logical_core = logical_core,
            .worker_id = i,
            .channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
            .num_buffers = test_params.num_buffers_full_size_channel,
            .buffer_size_bytes = test_params.buffer_size_bytes_full_size_channel,
            .receiver_noc_xy_encoding = default_receiver_noc_xy_encoding,
            .mcast_encoding = i == 0 ? std::make_optional(mcast_encoding) : std::nullopt,
            .num_mcast_dests = i == 0 ? std::make_optional(num_mcast_dests) : std::nullopt,
        };
        create_worker_kernel(test_params, worker_test_config, mux_test_config, drainer_test_config, device, program);
    }

    auto header_only_channel_worker_offset = worker_cores_offset + test_params.num_full_size_channels;
    for (auto i = 0; i < test_params.num_header_only_channels; i++) {
        CoreCoord logical_core = worker_logical_cores[header_only_channel_worker_offset + i];
        WorkerTestConfig worker_test_config = {
            .memory_map = &worker_memory_map,
            .worker_logical_core = logical_core,
            .worker_id = i,
            .channel_type = tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL,
            .num_buffers = test_params.num_buffers_header_only_channel,
            .buffer_size_bytes = test_params.buffer_size_bytes_header_only_channel,
            .receiver_noc_xy_encoding = default_receiver_noc_xy_encoding,
            .mcast_encoding = std::nullopt,
            .num_mcast_dests = std::nullopt,
        };
        create_worker_kernel(test_params, worker_test_config, mux_test_config, drainer_test_config, device, program);
    }

    log_info(tt::LogTest, "Launching programs");
    tt::tt_metal::CommandQueue& cq = device->command_queue();
    tt::tt_metal::EnqueueProgram(cq, program, false);

    log_info(tt::LogTest, "Waiting for workers to complete");
    size_t num_active_workers = test_params.num_full_size_channels + test_params.num_header_only_channels;
    for (size_t i = worker_cores_offset; i < worker_cores_offset + num_active_workers; i++) {
        CoreCoord core = worker_logical_cores[i];
        std::vector<uint32_t> worker_status(1, 0);
        while ((worker_status[0] & 0xFFFF) == 0) {
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, core, worker_memory_map.test_results_address, 4, worker_status);
        }
    }

    log_info(tt::LogTest, "Workers done, terminating mux kernel");
    std::vector<uint32_t> termiation_signal(1, tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    tt::tt_metal::detail::WriteToDeviceL1(
        device, mux_logical_core, mux_kernel_config.get_termination_signal_address(), termiation_signal);

    log_info(tt::LogTest, "Waiting for mux kernel to terminate");
    // need to wait before terminating driner core otherwise the mux kernel will hang while closing connection
    std::vector<uint32_t> mux_status(1, 0);
    while (mux_status[0] != tt::tt_fabric::EDMStatus::TERMINATED) {
        tt::tt_metal::detail::ReadFromDeviceL1(
            device, mux_logical_core, mux_kernel_config.get_status_address(), 4, mux_status);
    }

    log_info(tt::LogTest, "Terminating drainer kernel");
    tt::tt_metal::detail::WriteToDeviceL1(
        device, drainer_logical_core, drainer_kernel_config.get_termination_signal_address(), termiation_signal);

    log_info(tt::LogTest, "Waiting for programs");
    tt::tt_metal::Finish(cq);

    tt::tt_metal::detail::CloseDevices(devices);
    tt::tt_fabric::SetFabricConfig(
        tt::tt_fabric::FabricConfig::DISABLED, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

    log_info(tt::LogTest, "Collecting results");

    uint64_t max_elapsed_cycles = 0;
    for (size_t i = worker_cores_offset; i < worker_cores_offset + num_active_workers; i++) {
        CoreCoord core = worker_logical_cores[i];
        std::vector<uint32_t> worker_status;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device, core, worker_memory_map.test_results_address, test_results_size_bytes, worker_status);
        uint64_t worker_cycles =
            (((uint64_t)worker_status[TT_FABRIC_CYCLES_INDEX + 1]) << 32) | worker_status[TT_FABRIC_CYCLES_INDEX];
        max_elapsed_cycles = std::max(max_elapsed_cycles, worker_cycles);
    }

    size_t total_bytes_sent =
        (test_params.num_full_size_channels * test_params.buffer_size_bytes_full_size_channel +
         test_params.num_header_only_channels * test_params.buffer_size_bytes_header_only_channel) *
        test_params.num_packets;
    double total_bw = (double)total_bytes_sent / max_elapsed_cycles;
    log_info(tt::LogTest, "Total bandwidth (B/c): {}", total_bw);

    // dump results
    std::filesystem::path log_path = log_file_path;
    std::ofstream log_file;
    log_file.open(log_path, std::ios_base::out);
    log_file << total_bw;
    log_file.close();

    return 0;
}
