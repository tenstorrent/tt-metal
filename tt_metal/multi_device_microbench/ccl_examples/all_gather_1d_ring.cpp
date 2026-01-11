// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Basically, this is clone of the microbenchmark program `run_unicast_once.cpp`
// Check it for more details.

#include <functional>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "common/tt_backend_api_types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/allocator/allocator.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "system_mesh.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "multi_device_microbench/common.hpp"

#include <cstdint>
#include <vector>
#include <chrono>
#include <utility>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

constexpr const char* KERNEL_DIR = "tt_metal/multi_device_microbench/ccl_examples/kernels/";

void append_fabric_mux_connection_ct_args(
    const uint32_t num_workers_per_direction,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& worker_ct_args) {
    worker_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));  // fabric_mux_num_buffers_per_channel
    worker_ct_args.push_back(
        mux_kernel_config.get_buffer_size_bytes(channel_type));        // fabric_mux_channel_buffer_size_bytes
    worker_ct_args.push_back(mux_kernel_config.get_status_address());  // fabric_mux_status_address
    worker_ct_args.push_back(
        mux_kernel_config.get_termination_signal_address());  // fabric_mux_termination_signal_address
    worker_ct_args.push_back(num_workers_per_direction);      // num_mux_clients
}

void append_fabric_mux_connection_rt_args(
    const bool mux_connection_valid,
    const bool is_termination_master,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const CoreCoord& mux_virtual_core,
    const uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    std::vector<uint32_t>& worker_rt_args) {
    worker_rt_args.push_back(mux_connection_valid);   // mux_connection_valid
    worker_rt_args.push_back(is_termination_master);  // is_termination_master
    worker_rt_args.push_back(mux_virtual_core.x);     // fabric_mux_x
    worker_rt_args.push_back(mux_virtual_core.y);     // fabric_mux_y
    worker_rt_args.push_back(
        mux_kernel_config.get_channel_base_address(channel_type, worker_id));  // fabric_mux_channel_base_address
    worker_rt_args.push_back(
        mux_kernel_config.get_connection_info_address(channel_type, worker_id));  // fabric_mux_connection_info_address
    worker_rt_args.push_back(mux_kernel_config.get_connection_handshake_address(
        channel_type, worker_id));  // fabric_mux_connection_handshake_address
    worker_rt_args.push_back(
        mux_kernel_config.get_flow_control_address(channel_type, worker_id));  // fabric_mux_flow_control_address
    worker_rt_args.push_back(
        mux_kernel_config.get_buffer_index_address(channel_type, worker_id));  // fabric_mux_buffer_index_address
    worker_rt_args.push_back(
        mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id));  // fabric_mux_channel_id
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // termination_sync_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_fabric_mux_status_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_flow_control_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_teardown_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_buffer_index_address
    worker_rt_args.push_back(termination_master_virtual_core.x);                    // termination_master_noc_x
    worker_rt_args.push_back(termination_master_virtual_core.y);                    // termination_master_noc_y
}

// TODO : implement for galaxy later.
int main() {
    // Initialize mesh descriptor
    MeshDescriptor mesh_desc{};

    // ------------ Setup Fabric Config ------------
    // Ensure that fabric config is set before creating MeshDevice.
    // Extract core mesh_desc
    auto cluster_type = MetalContext::instance().get_cluster().get_cluster_type();
    bool is_n300_or_t3k_cluster = cluster_type == ClusterType::T3K or cluster_type == ClusterType::N300;
    auto core_type =
        (mesh_desc.num_cqs >= 2 and is_n300_or_t3k_cluster) ? DispatchCoreType::ETH : DispatchCoreType::WORKER;

    if (is_n300_or_t3k_cluster) {
        // TODO : Should it be FABRIC_1D?
        tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::FABRIC_1D);  // no physical ring connection on N300/T3K
    } else {
        tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::FABRIC_1D_RING);
    }

    constexpr uint32_t num_links = 1;
    constexpr uint32_t num_directions_per_link = 2;      // bidirectional links
    constexpr uint32_t num_mux_cores_per_direction = 1;  // dedicated mux core per direction
    constexpr uint32_t num_workers_per_direction = 2;

    // ------------ Setup MeshDevice ------------

    // Validate requested mesh shape against system mesh
    const auto& system_mesh_shape = distributed::SystemMesh::instance().shape();
    distributed::MeshShape requested_shape(1, 8);

    // Option 1: Check if requested shape fits in system mesh
    TT_FATAL(
        system_mesh_shape.mesh_size() >= requested_shape.mesh_size(),
        "System mesh size is smaller than requested mesh size");

    // Create a mesh device
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(requested_shape),
        mesh_desc.l1_small_size,
        mesh_desc.trace_region_size,
        mesh_desc.num_cqs,
        core_type,
        {},  // l1_bank_remap
        mesh_desc.worker_l1_size);
    const auto& mesh_view = mesh_device->get_view();
    auto& mesh_cq = mesh_device->mesh_command_queue();
    const auto num_devices = mesh_view.shape().mesh_size();

    // Get control plance instance
    const auto& control_plane = MetalContext::instance().get_control_plane();

    // Get fabric context
    const auto& fabric_context = control_plane.get_fabric_context();

    // print all fabric packet spec info
    log_info(tt::LogTest, "fabric packet header size bytes : {}", fabric_context.get_fabric_packet_header_size_bytes());
    log_info(tt::LogTest, "fabric max payload size bytes : {}", fabric_context.get_fabric_max_payload_size_bytes());
    // channel buffer size == max payload size + packet header size
    log_info(
        tt::LogTest, "fabric channel buffer size bytes : {}", fabric_context.get_fabric_channel_buffer_size_bytes());

    // ------------ Setup Mesh Buffer ------------
    constexpr uint32_t page_size = sizeof(uint16_t) * tt::constants::TILE_HW;
    constexpr uint32_t num_pages_per_worker = 8;
    const uint32_t num_input_pages = num_pages_per_worker * num_workers_per_direction * num_links;
    const uint32_t num_output_pages = num_input_pages * num_devices;
    const uint32_t input_buffer_size = num_input_pages * page_size;
    const uint32_t output_buffer_size = num_output_pages * page_size;

    const uint32_t num_input_host_words = (input_buffer_size) / sizeof(uint32_t);
    const uint32_t num_output_host_words = (output_buffer_size) / sizeof(uint32_t);
    log_info(tt::LogTest, "num_input_pages : {}", num_input_pages);
    log_info(tt::LogTest, "num_output_pages : {}", num_output_pages);
    log_info(tt::LogTest, "num_input_host_words: {}", num_input_host_words);
    log_info(tt::LogTest, "num_output_host_words: {}", num_output_host_words);

    // buffer config
    distributed::DeviceLocalBufferConfig dram_local_config{
        .page_size = page_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig input_buffer_glob_config{.size = input_buffer_size};
    distributed::ReplicatedBufferConfig output_buffer_glob_config{.size = output_buffer_size};

    // buffer creations
    auto input_buf = distributed::MeshBuffer::create(input_buffer_glob_config, dram_local_config, mesh_device.get());
    auto output_buf = distributed::MeshBuffer::create(output_buffer_glob_config, dram_local_config, mesh_device.get());

    std::vector<uint32_t> output_expected_words;
    output_expected_words.reserve(num_output_host_words);
    std::vector<uint32_t> output_initial_words(num_output_host_words, 0xdeadbeef);

    std::vector<std::vector<uint32_t>> all_input_words_per_device(num_devices);
    uint32_t input_data_seed = 0;
    for (uint32_t i = 0; i < num_devices; i++) {
        all_input_words_per_device[i] = make_src_data<uint32_t>(num_input_host_words, input_data_seed);
        input_data_seed += num_input_host_words;
        output_expected_words.insert(
            output_expected_words.end(), all_input_words_per_device[i].begin(), all_input_words_per_device[i].end());
    }

    // ------------ Global Semaphore ------------
    const auto grid_size = mesh_device->compute_with_storage_grid_size();
    CoreRange all_core_range = CoreRange(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1));
    static GlobalSemaphore out_ready_forward_semaphore =
        CreateGlobalSemaphore(mesh_device.get(), all_core_range, /*init_val*/ 0);
    static GlobalSemaphore out_ready_backward_semaphore =
        CreateGlobalSemaphore(mesh_device.get(), all_core_range, /*init_val*/ 0);
    static GlobalSemaphore barrier_semaphore = CreateGlobalSemaphore(mesh_device.get(), all_core_range, /*init_val*/ 0);

    // ------------ Setup workload & programs ------------
    distributed::MeshWorkload mesh_workload;
    const auto& mesh_shape = mesh_view.shape();
    const uint32_t ring_size = num_devices;
    const uint32_t distance_between_chips = 1;

    const uint32_t num_cols = mesh_view.num_cols();

    // fix target devices in ring topology excluding me.
    constexpr uint32_t num_devices_right = 3;  // number of forwarding when direction is from left to right
    constexpr uint32_t num_devices_left = 4;   // number of forwarding when direction is from right to left

    // forward direction : left to right
    // backward direction : right to left
    for (const auto& mesh_coord : distributed::MeshCoordinateRange(mesh_shape)) {
        tt::tt_metal::Program program{};

        uint32_t cur_device_linear_idx = mesh_coord.to_linear_index(mesh_shape);
        const auto my_node_id = mesh_view.get_fabric_node_id(mesh_coord);
        log_info(tt::LogTest, "Device linear idx: {}", cur_device_linear_idx);

        // Get a neighbor in forward direction
        int forward_neighbor = (cur_device_linear_idx + distance_between_chips) % num_devices;
        distributed::MeshCoordinate forward_coord = distributed::MeshCoordinate(
            static_cast<uint32_t>(forward_neighbor / num_cols), static_cast<uint32_t>(forward_neighbor % num_cols));

        // Get a neighbor in backward direction
        int backward_neighbor = (cur_device_linear_idx - distance_between_chips + num_devices) % num_devices;
        distributed::MeshCoordinate backward_coord = distributed::MeshCoordinate(
            static_cast<uint32_t>(backward_neighbor / num_cols), static_cast<uint32_t>(backward_neighbor % num_cols));

        log_info(
            tt::LogTest,
            "Device coord: {}, forward_coord: {}, backward_coord: {}",
            mesh_coord,
            forward_coord,
            backward_coord);

        // write buffer data
        distributed::WriteShard(
            mesh_cq, input_buf, all_input_words_per_device[cur_device_linear_idx], mesh_coord, /*blocking=*/true);
        distributed::WriteShard(mesh_cq, output_buf, output_initial_words, mesh_coord, /*blocking=*/true);

        // prepare fabric routing args
        uint32_t mesh_id = mesh_view.get_fabric_node_id(mesh_coord).mesh_id.get();
        // unicast args
        std::array<uint32_t, 2> forward_unicast_args = {mesh_id, distance_between_chips};
        std::array<uint32_t, 2> backward_unicast_args = {mesh_id, distance_between_chips};

        // multicast args
        std::array<uint32_t, 6> forward_barrier_mcast_args = {
            distance_between_chips, ring_size - 1, /*pad 4 args*/ 0, 0, 0, 0};
        std::array<uint32_t, 6> backward_barrier_mcast_args = {
            distance_between_chips, ring_size - 1, /*pad 4 args*/ 0, 0, 0, 0};

        // Aggregate cores
        //  use default sub_device_id initialized in mesh device creation.
        const auto worker_core_range_set =
            mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, /*sub_device_id*/ SubDeviceId{0});
        const uint32_t num_cores_per_link =
            (num_mux_cores_per_direction + num_workers_per_direction) * num_directions_per_link * num_links;

        CoreRangeSet all_core_range;
        auto add_core = [&](const CoreCoord& coord) {
            all_core_range = all_core_range.merge(CoreRangeSet(CoreRange(coord, coord)));
            return all_core_range.num_cores() >= num_cores_per_link;
        };
        for (const auto& core_range : worker_core_range_set.ranges()) {
            bool done = false;
            for (size_t y = core_range.start_coord.y; y <= core_range.end_coord.y && !done; y++) {
                for (size_t x = core_range.start_coord.x; x <= core_range.end_coord.x && !done; x++) {
                    done = add_core(CoreCoord(x, y));
                }
            }
            if (done) {
                break;
            }
        }
        auto all_cores = corerange_to_cores(all_core_range, std::nullopt, true);

        // Core ranges
        std::vector<CoreRange> sender_worker_core_ranges;
        std::vector<CoreRange> mux_core_ranges;
        std::vector<CoreRange> termination_master_core_ranges;
        std::set<CoreRange> sender_forward_core_ranges;
        std::set<CoreRange> sender_backward_core_ranges;

        uint32_t core_id = 0;
        for (uint32_t link = 0; link < num_links; link++) {
            for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
                const auto& mux_core = all_cores[core_id++];
                mux_core_ranges.emplace_back(mux_core);

                for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                    const auto& worker_core = all_cores[core_id++];

                    if (worker == 0) {
                        termination_master_core_ranges.emplace_back(worker_core);
                    }

                    if (dir) {
                        sender_forward_core_ranges.emplace(worker_core);
                    } else {
                        sender_backward_core_ranges.emplace(worker_core);
                    }
                    sender_worker_core_ranges.emplace_back(worker_core);
                }
            }
        }
        CoreRangeSet sender_worker_core_range_set =
            CoreRangeSet(sender_worker_core_ranges);  // forward + backward workers
        CoreRangeSet mux_core_range_set = CoreRangeSet(mux_core_ranges);

        // L1 Scratch CB Creation
        const uint32_t packet_size_bytes = fabric_context.get_fabric_channel_buffer_size_bytes();

        // fabric_unicast_noc_scatter_write currently supports 4 distinct noc addresses
        uint32_t max_target_noc_addresses_per_packet = 4;
        uint32_t num_pages_per_packet = packet_size_bytes / page_size;
        uint32_t num_pages_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
        log_info(tt::LogOp, "num_pages_to_write_per_packet: {}", num_pages_to_write_per_packet);

        // CBs for transferring data between sender_reader and sender_writer
        uint32_t sender_cb_index = tt::CB::c_in0;
        uint32_t cb_num_pages = 3 * num_pages_to_write_per_packet;  // triple buffering
        tt::tt_metal::CircularBufferConfig cb_sender_config =
            tt::tt_metal::CircularBufferConfig(cb_num_pages * page_size, {{sender_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(sender_cb_index, page_size);
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_sender_config);

        // Get base address for mux
        const uint32_t l1_unreserved_base_address =
            mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        const uint32_t mux_base_l1_address = l1_unreserved_base_address;

        // Configure FabricMux
        // channel organization : [ ... full_size_channels ... | ... header_only_channels ... ]
        const uint32_t num_full_size_channels = num_workers_per_direction;
        const uint32_t num_header_only_channels = 0;
        const uint32_t num_buffers_full_size_channels = 0;
        const uint32_t num_buffers_header_only_channels = 0;
        const uint32_t buffer_size_bytes_full_size_channel = packet_size_bytes;
        const auto mux_kernel_config = tt_fabric::FabricMuxConfig(
            num_full_size_channels,
            num_header_only_channels,
            num_buffers_full_size_channels,
            num_buffers_header_only_channels,
            buffer_size_bytes_full_size_channel,
            mux_base_l1_address);

        // Set reader compile args and create kernel
        std::vector<uint32_t> sender_reader_compile_args = {
            ring_size,                      // ring_size
            cur_device_linear_idx,          // logical_chip_id
            sender_cb_index,                // cb_forward_id
            num_pages_to_write_per_packet,  // num_pages_to_write_per_packet
            page_size,                      // page_size
            num_devices_right,              // num_slices_forward_direction
            num_devices_left,               // num_slices_backward_direction
        };
        TensorAccessorArgs(*input_buf).append_to(sender_reader_compile_args);
        TensorAccessorArgs(*output_buf).append_to(sender_reader_compile_args);
        const auto reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            std::string(KERNEL_DIR) + "reader_all_gather_1d_ring.cpp",
            sender_worker_core_range_set,
            tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, {}));

        // Set writer compile args and create kernel
        const auto channel_type = tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
        std::vector<uint32_t> sender_writer_compile_args = {
            ring_size,                      // ring_size
            cur_device_linear_idx,          // logical_chip_id
            sender_cb_index,                // cb_output_id
            num_pages_to_write_per_packet,  // num_pages_to_write_per_packet
            page_size,                      // page_size
            num_devices_right,              // num_targets_forward_direction
            num_devices_left,               // num_targets_backward_direction
        };
        append_fabric_mux_connection_ct_args(
            num_workers_per_direction, channel_type, mux_kernel_config, sender_writer_compile_args);

        sender_writer_compile_args.insert(
            sender_writer_compile_args.end(), forward_unicast_args.begin(), forward_unicast_args.end());
        sender_writer_compile_args.insert(
            sender_writer_compile_args.end(), forward_barrier_mcast_args.begin(), forward_barrier_mcast_args.end());
        sender_writer_compile_args.insert(
            sender_writer_compile_args.end(), backward_unicast_args.begin(), backward_unicast_args.end());
        sender_writer_compile_args.insert(
            sender_writer_compile_args.end(), backward_barrier_mcast_args.begin(), backward_barrier_mcast_args.end());
        tt::tt_metal::TensorAccessorArgs(*output_buf).append_to(sender_writer_compile_args);
        const auto writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            std::string(KERNEL_DIR) + "writer_all_gather_1d_ring.cpp",
            sender_worker_core_range_set,
            tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, {}));

        // Create mux kernel
        const auto mux_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
            mux_core_range_set,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

        auto worker_core_iter = sender_worker_core_range_set.ranges().cbegin();
        auto mux_core_iter = mux_core_range_set.ranges().cbegin();
        auto termination_master_core_iter = termination_master_core_ranges.cbegin();

        for (uint32_t link = 0; link < num_links; link++) {
            for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
                bool is_forward = (dir == 0);

                auto mux_logical_core = *((mux_core_iter++)->begin());
                CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);

                std::vector<uint32_t> mux_rt_args = {};
                if (is_forward) {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord);
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        my_node_id, dst_node_id, link, program, {mux_logical_core});
                    log_info(
                        tt::LogOp,
                        "[forward] my_linear_id : {}, src_chip_id: {}, dst_chip_id: {}",
                        cur_device_linear_idx,
                        my_node_id.chip_id,
                        dst_node_id.chip_id);
                } else {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord);
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        my_node_id, dst_node_id, link, program, {mux_logical_core});
                    log_info(
                        tt::LogOp,
                        "[backward] my_linear_id : {}, src_chip_id: {}, dst_chip_id: {}",
                        cur_device_linear_idx,
                        my_node_id.chip_id,
                        dst_node_id.chip_id);
                }
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);

                auto termination_master_logical_core = *((termination_master_core_iter++)->begin());
                for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                    auto core = *((worker_core_iter++)->begin());
                    CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
                    CoreCoord mapped_core = all_cores
                        [(link * num_cores_per_link) +
                         ((1 - dir) * (num_mux_cores_per_direction + num_workers_per_direction)) +
                         num_mux_cores_per_direction + worker];
                    CoreCoord mapped_core_coord = mesh_device->worker_core_from_logical_core(mapped_core);

                    uint32_t global_worker_id = (link * num_workers_per_direction) + worker;

                    uint32_t input_page_id_start = global_worker_id * num_pages_per_worker;
                    uint32_t input_page_id_end = (global_worker_id + 1) * num_pages_per_worker;
                    log_info(
                        tt::LogTest,
                        "  Worker {}, core {}, mapped_core {}, dir {}, input_page_id_start {}, input_page_id_end {}",
                        global_worker_id,
                        core,
                        mapped_core,
                        dir,
                        input_page_id_start,
                        input_page_id_end);

                    uint32_t pages_per_sync = std::max(
                        (input_page_id_end - input_page_id_start) / num_pages_to_write_per_packet, (uint32_t)1);
                    std::vector<uint32_t> reader_rt_args = {
                        static_cast<uint32_t>(input_buf->address()),
                        static_cast<uint32_t>(output_buf->address()),
                        is_forward ? static_cast<uint32_t>(out_ready_forward_semaphore.address())
                                   : static_cast<uint32_t>(out_ready_backward_semaphore.address()),
                        dir,
                        input_page_id_start,
                        input_page_id_end,
                        pages_per_sync,
                        num_input_pages,
                        num_pages_per_worker * num_workers_per_direction,  // pages per direction
                    };
                    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

                    CoreCoord termination_master_virtual_core =
                        mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                    std::vector<uint32_t> writer_rt_args = {
                        static_cast<uint32_t>(output_buf->address()),
                        static_cast<uint32_t>(virtual_core.x),
                        static_cast<uint32_t>(virtual_core.y),
                        is_forward ? static_cast<uint32_t>(out_ready_forward_semaphore.address())
                                   : static_cast<uint32_t>(out_ready_backward_semaphore.address()),
                        static_cast<uint32_t>(barrier_semaphore.address()),
                        static_cast<uint32_t>(mapped_core_coord.x),
                        static_cast<uint32_t>(mapped_core_coord.y),
                        dir,
                        input_page_id_start,
                        input_page_id_end,
                        pages_per_sync,
                        num_input_pages,
                        num_pages_per_worker * num_workers_per_direction,  // pages per direction
                    };
                    append_fabric_mux_connection_rt_args(
                        true,
                        worker == 0,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_virtual_core,
                        worker,
                        core,
                        mux_kernel_config,
                        program,
                        termination_master_virtual_core,
                        writer_rt_args);
                    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);
                }
            }
        }

        // Add program to workload
        mesh_workload.add_program(distributed::MeshCoordinateRange{mesh_coord}, std::move(program));
    }

    // ------------ Run Workload ------------
    distributed::EnqueueMeshWorkload(mesh_cq, mesh_workload, /*blocking=*/false);
    distributed::Finish(mesh_cq);

    // ------------ Verify Results ------------
    std::vector<std::vector<uint32_t>> output_data_per_device(
        num_devices, std::vector<uint32_t>(num_output_host_words, 0));

    for (const auto& mesh_coord : distributed::MeshCoordinateRange(mesh_shape)) {
        uint32_t dev_id = mesh_coord.to_linear_index(mesh_shape);
        distributed::ReadShard(
            mesh_cq,
            output_data_per_device[dev_id],
            output_buf,
            mesh_coord,
            /*blocking=*/true);
    }

    // Compare results
    bool is_any_failed = false;
    for (uint32_t dev_id = 0; dev_id < num_devices; dev_id++) {
        const auto& output_data = output_data_per_device[dev_id];
        uint32_t mismatch_count = 0;
        for (uint32_t word = 0; word < num_output_host_words; word++) {
            uint32_t expected = output_expected_words[word];
            uint32_t actual = output_data[word];
            if (expected != actual) {
                if (mismatch_count == 0) {
                    log_critical(
                        tt::LogTest,
                        "Device {} first mismatch at word {}: expected 0x{:08x}, actual 0x{:08x}",
                        dev_id,
                        word,
                        expected,
                        actual);
                }
                mismatch_count++;
            }
        }
        if (mismatch_count > 0) {
            log_critical(
                tt::LogTest, "Device {} total mismatches: {}/{} words", dev_id, mismatch_count, num_output_host_words);
            is_any_failed = true;
        }
    }
    if (is_any_failed) {
        log_critical(tt::LogTest, "All-gather verification FAILED");
    } else {
        log_info(tt::LogTest, "All-gather verification PASSED for all {} devices", num_devices);
    }

    // Teardown mesh device
    mesh_device->close();
    mesh_device.reset();
}
