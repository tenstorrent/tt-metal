// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "host_api.hpp"
#include "tests/tt_metal/tt_fabric/common/test_fabric_edm_common.hpp"
#include <cstdint>
#include <cstddef>
#include <optional>

// The writer that sends the packets that will have latency measured
struct LatencyPacketTestWriterSpec {
    size_t num_bursts;
    size_t burst_size_num_messages;
};
struct DatapathBusyDataWriterSpec {
    size_t message_size_bytes;
    bool mcast;
    size_t write_distance;
};

struct WriterSpec {
    std::variant<LatencyPacketTestWriterSpec, DatapathBusyDataWriterSpec> spec;
    CoreCoord worker_core_logical;
    size_t message_size_bytes;
};

using LatencyTestWriterSpecs = std::vector<std::optional<WriterSpec>>;

template <typename MESH_DEVICE_OR_VIEW_T>
static std::vector<IDevice*> get_test_devices_impl(
    const MESH_DEVICE_OR_VIEW_T& mesh_device_or_view, size_t line_size, bool is_6u) {
    std::vector<IDevice*> devices_;
    if (is_6u) {
        // on 6u galaxy systems, we can form a 2D torus so we can just use a full row or column
        devices_.reserve(line_size);
        size_t r = 0;
        size_t c = 0;
        size_t* loop_var = nullptr;
        if (line_size == 4) {
            loop_var = &c;
        } else if (line_size == 8) {
            loop_var = &r;
        } else {
            TT_THROW(
                "Invalid line size for 6u system. Supported line sizes are 4 and 8 but {} was specified.", line_size);
        }
        for (; *loop_var < line_size; (*loop_var)++) {
            devices_.push_back(mesh_device_or_view.get_device(MeshCoordinate(r, c)));
        }
    } else {
        if (line_size == 2) {
            devices_ = {mesh_device_or_view.get_device(MeshCoordinate(0, 0)), mesh_device_or_view.get_device(MeshCoordinate(0, 1))};
        } else if (line_size == 4) {
            devices_ = {
                mesh_device_or_view.get_device(MeshCoordinate(0, 0)),
                mesh_device_or_view.get_device(MeshCoordinate(0, 1)),
                mesh_device_or_view.get_device(MeshCoordinate(0, 2)),
                mesh_device_or_view.get_device(MeshCoordinate(0, 3))};
        } else {
            devices_ = {
                mesh_device_or_view.get_device(MeshCoordinate(0, 0)),
                mesh_device_or_view.get_device(MeshCoordinate(0, 1)),
                mesh_device_or_view.get_device(MeshCoordinate(0, 2)),
                mesh_device_or_view.get_device(MeshCoordinate(0, 3)),
                mesh_device_or_view.get_device(MeshCoordinate(1, 3)),
                mesh_device_or_view.get_device(MeshCoordinate(1, 2)),
                mesh_device_or_view.get_device(MeshCoordinate(1, 1)),
                mesh_device_or_view.get_device(MeshCoordinate(1, 0))};
        }
    }
    return devices_;
}
template <typename TEST_FIXTURE_T>
static std::vector<IDevice*> get_test_devices(const TEST_FIXTURE_T& test_fixture, size_t line_size, bool is_6u) {
    return get_test_devices_impl(*test_fixture.mesh_device_, line_size, is_6u);
}

template <typename TEST_FIXTURE_T>
static DeviceAddr get_new_global_semaphore_address(const TEST_FIXTURE_T& test_fixture) {
    auto global_semaphore = tt::tt_metal::CreateGlobalSemaphore(
        test_fixture.mesh_device_.get(),
        test_fixture.mesh_device_.get()->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                            // initial value
        tt::tt_metal::BufferType::L1  // buffer type
    );
    return global_semaphore.address();
}

static bool is_seminc_only_mode(const LatencyTestWriterSpecs& writer_specs) {
    for (const auto& spec : writer_specs) {
        if (spec.has_value() && std::holds_alternative<LatencyPacketTestWriterSpec>(spec->spec)) {
            return spec->message_size_bytes == 0;
        }
    }
    TT_FATAL(false, "No packet sender found, invalid test configuration");
    return false;
}

static std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> allocate_mesh_buffer(
    tt::tt_metal::distributed::MeshDevice* mesh_device, size_t size_bytes) {
    tt::tt_metal::distributed::ReplicatedBufferConfig global_buffer_config = {.size = size_bytes * 1000};
    tt::tt_metal::distributed::DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = size_bytes, .buffer_type = BufferType::L1};
    return tt::tt_metal::distributed::MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device);
};

static auto get_largest_write_size(const LatencyTestWriterSpecs& writer_specs) {
    size_t largest_write_size = 16;
    for (const auto& spec : writer_specs) {
        if (spec.has_value()) {
            if (std::holds_alternative<LatencyPacketTestWriterSpec>(spec->spec)) {
                largest_write_size = std::max(largest_write_size, spec->message_size_bytes);
            } else if (std::holds_alternative<DatapathBusyDataWriterSpec>(spec->spec)) {
                largest_write_size = std::max(largest_write_size, spec->message_size_bytes);
            }
        }
    }
    return largest_write_size;
}

template <typename DEVICE_FIXTURE_T>
inline void RunPersistent1dFabricLatencyTest(
    // Args for the measured writer
    LatencyTestWriterSpecs writer_specs,
    size_t line_size,
    bool enable_fused_payload_with_sync,
    tt::tt_fabric::Topology topology) {
    const bool is_ring = topology == tt::tt_fabric::Topology::Ring;

    size_t num_links = 1;

    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    bool is_6u = num_devices == 32 && tt::tt_metal::GetNumPCIeDevices() == num_devices;
    if (num_devices < 4 && !is_6u) {
        log_info(tt::LogTest, "This test can only be run on T3000 or 6u systems");
        return;
    }

    TT_FATAL(writer_specs.size() < line_size, "num_devices_with_workers must be less than or equal to num_links");

    DEVICE_FIXTURE_T test_fixture;

    std::vector<IDevice*> devices = get_test_devices<DEVICE_FIXTURE_T>(test_fixture, line_size, is_6u);
    std::vector<IDevice*> devices_with_workers;
    devices.reserve(line_size);
    for (size_t i = 0; i < line_size; i++) {
        if (writer_specs.size() > i && writer_specs.at(i).has_value()) {
            devices_with_workers.push_back(devices.at(i));
        }
    }

    if (!is_ring) {
        // add a program slot for the ack responder
        devices_with_workers.push_back(devices.at(line_size - 1));
    }

    auto mesh_coords = {
        MeshCoordinate(0, 0),
        MeshCoordinate(0, 1),
        MeshCoordinate(0, 2),
        MeshCoordinate(0, 3),
        MeshCoordinate(1, 3),
        MeshCoordinate(1, 2),
        MeshCoordinate(1, 1),
        MeshCoordinate(1, 0)};
    for (const auto& coord : mesh_coords) {
        auto d = test_fixture.mesh_device_->get_device(coord);
        log_info(
            tt::LogTest,
            "coord: [{}, {}] -> device->id(): {}, fabric_node_id: {}",
            coord.coords()[0],
            coord.coords()[1],
            d->id(),
            tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(d->id()));
    }

    // Find latency writer specs and location
    std::optional<size_t> latency_writer_index_opt;
    for (size_t i = 0; i < writer_specs.size(); i++) {
        if (writer_specs[i].has_value() && std::holds_alternative<LatencyPacketTestWriterSpec>(writer_specs[i]->spec)) {
            latency_writer_index_opt = i;
            break;
        }
    }
    TT_FATAL(latency_writer_index_opt.has_value(), "Latency writer not found");
    size_t latency_writer_index = latency_writer_index_opt.value();

    tt::tt_metal::DeviceAddr ping_message_received_semaphore_address = get_new_global_semaphore_address(test_fixture);
    tt::tt_metal::DeviceAddr congestion_writers_ready_semaphore_address =
        get_new_global_semaphore_address(test_fixture);
    tt::tt_metal::DeviceAddr worker_done_sem_addr = get_new_global_semaphore_address(test_fixture);

    // Temporary until we move this to be under tt_metal and migrate to device init fabric
    // OR packet header management is removed from user space, whichever comes first
    constexpr size_t packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    static constexpr uint32_t packet_header_cb_index = tt::CB::c_in0;
    static constexpr size_t packet_header_cb_size_in_headers = 4;
    std::vector<size_t> dest_buffer_addresses(writer_specs.size(), 0);

    auto largest_write_size_bytes = get_largest_write_size(writer_specs);
    TT_FATAL(largest_write_size_bytes > 0, "Largest write size is 0, invalid test configuration");

    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> latency_writer_buffer =
        allocate_mesh_buffer(test_fixture.mesh_device_.get(), largest_write_size_bytes);
    auto latency_writer_buffer_address = latency_writer_buffer->address();

    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> largest_write_buffer =
        allocate_mesh_buffer(test_fixture.mesh_device_.get(), largest_write_size_bytes);
    auto congestion_writer_buffer_address = largest_write_buffer->address();
    TT_FATAL(
        latency_writer_buffer_address != congestion_writer_buffer_address,
        "Latency and congestion writer buffers must have different addresses");
    TT_FATAL(
        latency_writer_buffer_address != ping_message_received_semaphore_address,
        "Latency and ping message received semaphore must have different addresses");
    TT_FATAL(
        latency_writer_buffer_address != congestion_writers_ready_semaphore_address,
        "Latency and congestion writers ready semaphore must have different addresses");
    TT_FATAL(
        latency_writer_buffer_address != worker_done_sem_addr,
        "Latency and worker done semaphore must have different addresses");

    static constexpr tt::DataFormat cb_df = tt::DataFormat::Bfp8;
    // Get the inner 4 device ring on a WH T3K device so that we can use both links for all devices

    // Other boiler plate setup
    CoreRangeSet worker_cores = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_links - 1, 0)));
    auto worker_cores_vec = corerange_to_cores(worker_cores, std::nullopt, false);

    size_t num_congestion_writers = 0;
    for (const auto& spec : writer_specs) {
        if (spec.has_value() && std::holds_alternative<DatapathBusyDataWriterSpec>(spec->spec)) {
            num_congestion_writers++;
        }
    }

    std::vector<Program> programs(devices_with_workers.size());

    auto get_upstream_congestion_writer = [](const LatencyTestWriterSpecs& writer_specs) -> std::optional<WriterSpec> {
        std::optional<WriterSpec> upstream_congestion_writer = std::nullopt;
        for (size_t i = 0; i < writer_specs.size(); i++) {
            if (std::holds_alternative<DatapathBusyDataWriterSpec>(writer_specs.at(i)->spec)) {
                upstream_congestion_writer = writer_specs.at(i);
                break;
            } else if (std::holds_alternative<LatencyPacketTestWriterSpec>(writer_specs.at(i)->spec)) {
                break;
            }
        }
        return upstream_congestion_writer;
    };

    std::vector<KernelHandle> worker_kernel_ids;
    std::vector<size_t> per_device_global_sem_addr_rt_arg;
    size_t program_device_index = 0;

    auto build_connection_args = [is_ring](
                                     IDevice* device,
                                     IDevice* forward_device,
                                     IDevice* backward_device,
                                     Program& program,
                                     CoreCoord worker_core_logical,
                                     bool is_connected_in_direction,
                                     tt::tt_fabric::EdmLineFabricOpInterface::Direction direction,
                                     std::vector<uint32_t>& rt_args_out) {
        rt_args_out.push_back(is_connected_in_direction);

        if (is_connected_in_direction) {
            const auto device_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id());
            chip_id_t connected_chip_id = direction == tt::tt_fabric::EdmLineFabricOpInterface::FORWARD
                                              ? forward_device->id()
                                              : backward_device->id();
            const auto connected_device_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(connected_chip_id);
            tt::tt_fabric::append_fabric_connection_rt_args(
                device_fabric_node_id, connected_device_fabric_node_id, 0, program, {worker_core_logical}, rt_args_out);
        }
    };

    bool sem_inc_only = is_seminc_only_mode(writer_specs);
    std::optional<size_t> sender_buffer_address;
    for (size_t i = 0; i < writer_specs.size(); i++) {
        if (!writer_specs.at(i).has_value()) {
            continue;
        }
        const size_t line_index = i;
        auto& program = programs.at(program_device_index);
        auto* device = devices_with_workers.at(program_device_index);
        const CoreCoord& worker_core_logical = writer_specs.at(i)->worker_core_logical;
        const size_t dest_noc_x = device->worker_core_from_logical_core(worker_core_logical).x;
        const size_t dest_noc_y = device->worker_core_from_logical_core(worker_core_logical).y;

        bool is_latency_packet_sender = std::holds_alternative<LatencyPacketTestWriterSpec>(writer_specs[i]->spec);

        IDevice* backward_device = i == 0 ? is_ring ? devices.at(line_size - 1) : nullptr : devices.at(i - 1);
        IDevice* forward_device = i == line_size - 1 ? is_ring ? devices.at(0) : nullptr : devices.at(i + 1);

        // Initialize the fabric handle for worker connection
        bool start_of_line = line_index == 0;
        bool end_of_line = line_index == line_size - 1;
        bool has_forward_connection = is_ring || !end_of_line;
        bool has_backward_connection = is_ring || !start_of_line;


        std::vector<uint32_t> worker_ct_args = {};
        std::string kernel_path =
            is_latency_packet_sender
                ? "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/1D_fabric_loopback_latency_test_writer.cpp"
                : "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/1D_fabric_latency_datapath_congestion_writer.cpp";
        if (is_latency_packet_sender) {
            bool payloads_are_mcast = false;
            worker_ct_args = {!sem_inc_only && enable_fused_payload_with_sync, payloads_are_mcast, sem_inc_only};
        } else {
            const auto& datapath_spec = std::get<DatapathBusyDataWriterSpec>(writer_specs[i]->spec);
            worker_ct_args.push_back(datapath_spec.mcast);
        }
        auto worker_kernel_id = tt_metal::CreateKernel(
            program, kernel_path, worker_cores, tt_metal::WriterDataMovementConfig(worker_ct_args));
        worker_kernel_ids.push_back(worker_kernel_id);

        // RT ARGS
        std::vector<uint32_t> rt_args = {};
        size_t distance_to_responder = is_ring ? line_size : (line_size - 1 - line_index);
        if (is_latency_packet_sender) {
            size_t dest_bank_addr = latency_writer_buffer_address;
            sender_buffer_address = dest_bank_addr;
            std::vector<size_t> downstream_writer_semaphore_addresses;
            std::vector<size_t> downstream_writer_noc_x_list;
            std::vector<size_t> downstream_writer_noc_y_list;
            std::vector<size_t> downstream_writer_hop_distance_list;
            for (size_t i = 0; i < writer_specs.size(); i++) {
                if (!writer_specs.at(i).has_value()) {
                    continue;
                }
                const auto& ws = writer_specs.at(i);
                if (std::holds_alternative<LatencyPacketTestWriterSpec>(ws->spec)) {
                } else if (std::holds_alternative<DatapathBusyDataWriterSpec>(ws->spec)) {
                    const auto& datapath_spec = std::get<DatapathBusyDataWriterSpec>(ws->spec);
                    const auto downstream_worker_core_noc =
                        device->worker_core_from_logical_core(ws->worker_core_logical);
                    downstream_writer_semaphore_addresses.push_back(worker_done_sem_addr);
                    downstream_writer_noc_x_list.push_back(downstream_worker_core_noc.x);
                    downstream_writer_noc_y_list.push_back(downstream_worker_core_noc.y);
                    downstream_writer_hop_distance_list.push_back(datapath_spec.write_distance);
                } else {
                    TT_THROW("Invalid writer spec");
                }
            }

            const auto& packet_spec = std::get<LatencyPacketTestWriterSpec>(writer_specs[i]->spec);
            rt_args = {
                dest_bank_addr,
                ping_message_received_semaphore_address,
                writer_specs.at(i)->message_size_bytes,
                packet_spec.burst_size_num_messages,
                packet_spec.num_bursts,
                distance_to_responder,
                congestion_writers_ready_semaphore_address,
                num_congestion_writers};
            const auto& upstream_congestion_writer = get_upstream_congestion_writer(writer_specs);
            rt_args.push_back(upstream_congestion_writer.has_value());
            if (upstream_congestion_writer.has_value()) {
                const auto upstream_worker_core_noc =
                    device->worker_core_from_logical_core(upstream_congestion_writer->worker_core_logical);
                rt_args.push_back(worker_done_sem_addr);
                rt_args.push_back(upstream_worker_core_noc.x);
                rt_args.push_back(upstream_worker_core_noc.y);
            }
            rt_args.push_back(downstream_writer_semaphore_addresses.size());
            std::for_each(
                downstream_writer_semaphore_addresses.begin(),
                downstream_writer_semaphore_addresses.end(),
                [&rt_args](size_t sem_addr) { rt_args.push_back(sem_addr); });
            std::for_each(
                downstream_writer_noc_x_list.begin(), downstream_writer_noc_x_list.end(), [&rt_args](size_t noc_x) {
                    rt_args.push_back(noc_x);
                });
            std::for_each(
                downstream_writer_noc_y_list.begin(), downstream_writer_noc_y_list.end(), [&rt_args](size_t noc_y) {
                    rt_args.push_back(noc_y);
                });
            std::for_each(
                downstream_writer_hop_distance_list.begin(),
                downstream_writer_hop_distance_list.end(),
                [&rt_args](size_t hop_distance) { rt_args.push_back(hop_distance); });

        } else {
            size_t dest_bank_addr = congestion_writer_buffer_address;
            const auto& datapath_spec = std::get<DatapathBusyDataWriterSpec>(writer_specs[i]->spec);
            rt_args = {
                dest_bank_addr,
                writer_specs.at(i)->message_size_bytes,
                dest_noc_x,
                dest_noc_y,
                datapath_spec.write_distance,
                worker_done_sem_addr};

            const bool is_downstream = i > latency_writer_index;
            const auto latency_writer_core = devices[latency_writer_index]->worker_core_from_logical_core(
                writer_specs[latency_writer_index]->worker_core_logical);

            rt_args.push_back(is_downstream ? 1 : 0);
            rt_args.push_back(latency_writer_core.x);
            rt_args.push_back(latency_writer_core.y);
            rt_args.push_back(congestion_writers_ready_semaphore_address);
            rt_args.push_back(std::abs(static_cast<int>(i) - static_cast<int>(latency_writer_index)));
        }

        build_connection_args(
            device,
            forward_device,
            backward_device,
            program,
            worker_core_logical,
            has_forward_connection,
            tt::tt_fabric::EdmLineFabricOpInterface::FORWARD,
            rt_args);
        build_connection_args(
            device,
            forward_device,
            backward_device,
            program,
            worker_core_logical,
            has_backward_connection,
            tt::tt_fabric::EdmLineFabricOpInterface::BACKWARD,
            rt_args);
        tt_metal::SetRuntimeArgs(program, worker_kernel_id, worker_core_logical, rt_args);

        program_device_index++;
    }

    // Add the ack writer kernel
    // create the kernel for
    // "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/1D_fabric_latency_test_ack_writer.cpp"
    if (!is_ring) {
        auto my_device = devices[line_size - 1];
        auto backward_device = devices[line_size - 2];
        size_t num_hops_upstream_to_writer = line_size - 1 - latency_writer_index;
        auto& ack_writer_program = programs.back();
        auto& latency_writer_spec = writer_specs.at(latency_writer_index);
        const auto& packet_spec = std::get<LatencyPacketTestWriterSpec>(latency_writer_spec->spec);
        // reserve CB
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                packet_header_cb_size_in_headers * packet_header_size_bytes, {{packet_header_cb_index, cb_df}})
                .set_page_size(packet_header_cb_index, packet_header_size_bytes);
        CreateCircularBuffer(ack_writer_program, worker_cores, cb_src0_config);
        auto ct_args = std::vector<uint32_t>{enable_fused_payload_with_sync, sem_inc_only};
        auto rt_args = std::vector<uint32_t>{
            latency_writer_buffer_address,
            ping_message_received_semaphore_address,
            latency_writer_spec->message_size_bytes,
            packet_spec.burst_size_num_messages,
            packet_spec.num_bursts,
            packet_header_cb_index,
            packet_header_cb_size_in_headers,
            num_hops_upstream_to_writer};

        build_connection_args(
            my_device,
            nullptr,
            backward_device,
            ack_writer_program,
            latency_writer_spec->worker_core_logical,
            true,
            tt::tt_fabric::EdmLineFabricOpInterface::BACKWARD,
            rt_args);
        auto ack_writer_kernel = tt_metal::CreateKernel(
            ack_writer_program,
            "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/1D_fabric_latency_test_ack_writer.cpp",
            worker_cores,
            tt_metal::WriterDataMovementConfig(ct_args));
        tt_metal::SetRuntimeArgs(ack_writer_program, ack_writer_kernel, worker_cores, rt_args);
    }

    for (auto d : devices_with_workers) {
        log_info(tt::LogTest, "launch on Device {}", d->id());
    }
    build_and_enqueue(devices_with_workers, programs);

    log_info(tt::LogTest, "Waiting for Op finish on all devices");
    wait_for_worker_program_completion(devices_with_workers);

    log_info(tt::LogTest, "Finished");
}

int main(int argc, char** argv) {
    std::size_t arg_idx = 1;
    // the line length, not total length of fabric after including loopback
    std::size_t line_size = std::stoi(argv[arg_idx++]);
    std::size_t latency_measurement_worker_line_index = std::stoi(argv[arg_idx++]);
    std::size_t latency_ping_message_size_bytes = std::stoi(argv[arg_idx++]);
    std::size_t latency_ping_burst_size = std::stoi(argv[arg_idx++]);
    std::size_t latency_ping_burst_count = std::stoi(argv[arg_idx++]);
    TT_FATAL(
        latency_ping_burst_size == 1,
        "Latency ping burst size must be 1. Support for accurately measuring latency with burst size > 1 is not "
        "implemented");

    bool add_upstream_fabric_congestion_writers = std::stoi(argv[arg_idx++]) != 0;
    std::size_t num_downstream_fabric_congestion_writers = std::stoi(argv[arg_idx++]);
    std::size_t congestion_writers_message_size = std::stoi(argv[arg_idx++]);
    bool congestion_writers_use_mcast = std::stoi(argv[arg_idx++]) != 0;
    bool enable_fused_payload_with_sync = std::stoi(argv[arg_idx++]) != 0;
    std::string topology_str = argv[arg_idx++];
    TT_FATAL(arg_idx == argc, "Read past end of args or didn't read all args");

    uint32_t test_expected_num_devices = 8;
    size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
    bool is_6u = num_devices == 32 && tt::tt_metal::GetNumPCIeDevices() == num_devices;
    if (num_devices < test_expected_num_devices) {
        log_warning(tt::LogTest, "This test can only be run on T3000 devices");
        return 1;
    }

    const bool is_ring = topology_str == "ring";
    auto compute_loopback_distance_to_start_of_line = [line_size, is_ring](std::size_t line_index) {
        return is_ring ? line_size - 1 : ((line_size - 1) * 2) - line_index;
    };

    LatencyTestWriterSpecs writer_specs(line_size - 1, std::nullopt);
    writer_specs.at(latency_measurement_worker_line_index) = WriterSpec{
        .spec =
            LatencyPacketTestWriterSpec{
                .num_bursts = latency_ping_burst_count,
                .burst_size_num_messages = latency_ping_burst_size,
            },
        .worker_core_logical = CoreCoord(0, 0),
        .message_size_bytes = latency_ping_message_size_bytes};

    if (add_upstream_fabric_congestion_writers) {
        TT_FATAL(
            latency_measurement_worker_line_index != 0,
            "Tried adding upstream congestion writer but the latency measurement packet router was added to line index "
            "0. If there is an upstream congestion writer, then the latency test writer cannot be at line index 0.");
        TT_FATAL(congestion_writers_message_size != 0, "upstream congestion writer message size must be non-zero");
        size_t upstream_worker_line_index = latency_measurement_worker_line_index - 1;
        writer_specs.at(upstream_worker_line_index) = WriterSpec{
            .spec =
                DatapathBusyDataWriterSpec{
                    .message_size_bytes = congestion_writers_message_size,
                    .mcast = congestion_writers_use_mcast,
                    .write_distance = compute_loopback_distance_to_start_of_line(upstream_worker_line_index),
                },
            .worker_core_logical = CoreCoord(0, 0),
            .message_size_bytes = congestion_writers_message_size};
    }

    TT_FATAL(
        num_downstream_fabric_congestion_writers + latency_measurement_worker_line_index < line_size,
        "Tried adding {} downstream congestion writers but there is not enough space left in the line."
        "the latency packet writer is at index {} and line_size is {}. Therefore, the largest number of downstream "
        "writers for this configuration is {}.",
        num_downstream_fabric_congestion_writers,
        latency_measurement_worker_line_index,
        line_size,
        line_size - 1 - latency_measurement_worker_line_index);
    for (size_t i = 0; i < num_downstream_fabric_congestion_writers; i++) {
        TT_FATAL(congestion_writers_message_size != 0, "downstream congestion writer message size must be non-zero");
        size_t downstream_worker_line_index = latency_measurement_worker_line_index + 1 + i;
        size_t distance =
            is_ring ? line_size - 1 : downstream_worker_line_index - latency_measurement_worker_line_index;
        writer_specs.at(downstream_worker_line_index) = WriterSpec{
            .spec =
                DatapathBusyDataWriterSpec{
                    .message_size_bytes = congestion_writers_message_size,
                    .mcast = congestion_writers_use_mcast,
                    .write_distance = distance,
                },
            .worker_core_logical = CoreCoord(0, 0),
            .message_size_bytes = congestion_writers_message_size};
    }

    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;
    if (topology_str == "linear") {
        topology = tt::tt_fabric::Topology::Linear;
    } else if (topology_str == "ring") {
        topology = tt::tt_fabric::Topology::Ring;
    } else if (topology_str == "mesh") {
        topology = tt::tt_fabric::Topology::Mesh;
        TT_THROW("Topology \"mesh\" is currently unsupported.");
    } else {
        TT_THROW("Invalid topology: {}", topology_str);
    }

    if (is_ring) {
        if (is_6u) {
            RunPersistent1dFabricLatencyTest<Fabric1DRingDeviceInitFixture>(
                writer_specs, line_size, enable_fused_payload_with_sync, topology);
        } else {
            RunPersistent1dFabricLatencyTest<Fabric1DRingDeviceInitFixture>(
                writer_specs, line_size, enable_fused_payload_with_sync, topology);
        }
    } else {
        RunPersistent1dFabricLatencyTest<Fabric1DLineDeviceInitFixture>(
            writer_specs, line_size, enable_fused_payload_with_sync, topology);
    }
}
