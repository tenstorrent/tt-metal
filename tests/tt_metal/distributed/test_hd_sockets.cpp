// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/work_split.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include "gmock/gmock.h"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "tt_metal/distributed/mesh_socket_serialization.hpp"
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <cstring>
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"

namespace tt::tt_metal::distributed {

bool is_device_coord_mmio_mapped(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, const MeshCoordinate& device_coord);

namespace {

constexpr uint32_t kTargetTrayId = 1;
constexpr uint32_t kTargetAsicLocation = 6;
constexpr std::size_t kPagesPerIteration = 1;
constexpr std::size_t kL1DataBudgetBytes = 1400000;
constexpr double kCyclesPerUs = 1350.0;
constexpr uint32_t kWarmupIters = 5;

const std::vector<std::size_t> kTotalDataSizes = {
    16 * 1024,            // 16KB
    32 * 1024,            // 32KB
    512 * 1024,           // 512KB
    1024 * 1024,          // 1MB
    16 * 1024 * 1024,     // 16MB
    512UL * 1024 * 1024,  // 512MB
    1024UL * 1024 * 1024  // 1GB
};

const std::vector<std::size_t> kPageSizes = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};

const std::vector<std::size_t> kD2HThroughputFifoSizes = {
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
    256 * 1024 * 1024,
    512 * 1024 * 1024,
};

const std::vector<std::size_t> kD2HLatencyFifoSizes = {1024, 4096, 16384, 65536, 512UL * 1024 * 1024};
const std::vector<std::size_t> kH2DThroughputFifoSizes = {
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024};
const std::vector<std::size_t> kH2DLatencyFifoSizes = {1024, 4096, 16384, 65536, 262144, 524288};

std::size_t compute_iteration_data_size_bytes(std::size_t page_size) {
    std::size_t pages = std::min<std::size_t>(kPagesPerIteration, kL1DataBudgetBytes / page_size);
    return page_size * std::max<std::size_t>(pages, 1);
}

const char* h2d_mode_name(H2DMode mode) { return (mode == H2DMode::HOST_PUSH) ? "HOST_PUSH" : "DEVICE_PULL"; }

PhysicalSystemDescriptor make_physical_system_descriptor() {
    return PhysicalSystemDescriptor(
        MetalContext::instance().get_cluster().get_driver(),
        MetalContext::instance().get_distributed_context_ptr(),
        &MetalContext::instance().hal(),
        MetalContext::instance().rtoptions(),
        true);
}

// Create an L1 mesh buffer sharded to a single logical core.
std::shared_ptr<MeshBuffer> make_l1_mesh_buffer(MeshDevice* mesh_device, const CoreCoord& core, uint32_t size) {
    auto shard_params = ShardSpecBuffer(CoreRangeSet(core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const DeviceLocalBufferConfig local_config{
        .page_size = size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    return MeshBuffer::create(ReplicatedBufferConfig{.size = size}, local_config, mesh_device);
}

// Enqueue a single-core program on the device (non-blocking).
void enqueue_on_core(MeshDevice& device, const MeshCoreCoord& core, Program program) {
    auto workload = MeshWorkload();
    workload.add_program(MeshCoordinateRange(core.device_coord), std::move(program));
    EnqueueMeshWorkload(device.mesh_command_queue(), workload, false);
}

// Read a single uint64 from device L1.
uint64_t read_l1_uint64(const MeshDevice& device, const MeshCoreCoord& core, uint64_t addr) {
    uint64_t val = 0;
    MetalContext::instance().get_cluster().read_core(
        &val,
        sizeof(uint64_t),
        tt_cxy_pair(device.get_device(core.device_coord)->id(), device.worker_core_from_logical_core(core.core_coord)),
        addr);
    return val;
}

// Read an array of uint64 from device L1 into a pre-sized vector.
void read_l1_uint64s(const MeshDevice& device, const MeshCoreCoord& core, uint64_t addr, std::vector<uint64_t>& out) {
    MetalContext::instance().get_cluster().read_core(
        out.data(),
        out.size() * sizeof(uint64_t),
        tt_cxy_pair(device.get_device(core.device_coord)->id(), device.worker_core_from_logical_core(core.core_coord)),
        addr);
}

struct ChipInfo {
    MeshCoordinate coord;
    uint32_t tray_id;
    uint32_t asic_location;
};

// Collect every MMIO-mapped chip with its tray/ASIC metadata.
std::vector<ChipInfo> enumerate_mmio_chips(const std::shared_ptr<MeshDevice>& mesh_device) {
    auto phys_desc = make_physical_system_descriptor();
    const auto& cp = MetalContext::instance().get_control_plane();
    std::vector<ChipInfo> chips;
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        if (!is_device_coord_mmio_mapped(mesh_device, coord)) {
            continue;
        }
        auto asic_id = cp.get_asic_id_from_fabric_node_id(mesh_device->get_fabric_node_id(coord));
        auto desc = phys_desc.get_asic_descriptors()[asic_id];
        chips.push_back({coord, *desc.tray_id, *desc.asic_location});
    }
    return chips;
}

std::optional<MeshCoordinate> find_target_mmio_coord(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t target_tray_id,
    uint32_t target_asic_location) {
    auto physical_system_descriptor = make_physical_system_descriptor();
    const auto& control_plane = MetalContext::instance().get_control_plane();
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        if (!is_device_coord_mmio_mapped(mesh_device, coord)) {
            continue;
        }
        auto fabric_node_id = mesh_device->get_fabric_node_id(coord);
        auto asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        auto asic_desc = physical_system_descriptor.get_asic_descriptors()[asic_id];
        if (*asic_desc.tray_id == target_tray_id && *asic_desc.asic_location == target_asic_location) {
            return coord;
        }
    }

    return std::nullopt;
}

MeshCoreCoord get_target_benchmark_worker_core(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device) {
    auto coord_opt = find_target_mmio_coord(mesh_device, kTargetTrayId, kTargetAsicLocation);
    TT_FATAL(coord_opt.has_value(), "No MMIO-mapped target device found for benchmark");
    return {coord_opt.value(), CoreCoord(0, 0)};
}

struct LatencySummary {
    double avg_us;
    double min_us;
    double max_us;
    double p50_us;
    double p99_us;
    double avg_cycles;
    uint64_t min_cycles;
    uint64_t max_cycles;
};

LatencySummary summarize_latency_cycles(const std::vector<uint64_t>& cycles) {
    TT_FATAL(!cycles.empty(), "Expected non-empty cycle measurements");

    auto sorted_cycles = cycles;
    std::sort(sorted_cycles.begin(), sorted_cycles.end());
    const uint64_t min_c = sorted_cycles.front();
    const uint64_t max_c = sorted_cycles.back();
    const uint64_t p50_c = sorted_cycles[sorted_cycles.size() / 2];
    const uint64_t p99_c = sorted_cycles[(sorted_cycles.size() * 99) / 100];

    double avg_c = 0.0;
    for (auto c : cycles) {
        avg_c += static_cast<double>(c);
    }
    avg_c /= static_cast<double>(cycles.size());

    return {
        .avg_us = avg_c / kCyclesPerUs,
        .min_us = static_cast<double>(min_c) / kCyclesPerUs,
        .max_us = static_cast<double>(max_c) / kCyclesPerUs,
        .p50_us = static_cast<double>(p50_c) / kCyclesPerUs,
        .p99_us = static_cast<double>(p99_c) / kCyclesPerUs,
        .avg_cycles = avg_c,
        .min_cycles = min_c,
        .max_cycles = max_c,
    };
}

void emit_latency_csv_row(
    std::size_t page_size,
    std::size_t socket_fifo_size,
    uint32_t num_iterations,
    const LatencySummary& stats,
    const MeshCoordinate& device_coord) {
    std::cout << page_size << "," << socket_fifo_size << "," << num_iterations << "," << stats.avg_us << ","
              << stats.min_us << "," << stats.max_us << "," << stats.p50_us << "," << stats.p99_us << ","
              << stats.avg_cycles << "," << stats.min_cycles << "," << stats.max_cycles << "," << device_coord
              << std::endl;
}

void emit_latency_csv_row(
    std::size_t page_size,
    std::size_t socket_fifo_size,
    const std::string& mode_str,
    uint32_t num_iterations,
    const LatencySummary& stats,
    const MeshCoordinate& device_coord) {
    std::cout << page_size << "," << socket_fifo_size << "," << mode_str << "," << num_iterations << "," << stats.avg_us
              << "," << stats.min_us << "," << stats.max_us << "," << stats.p50_us << "," << stats.p99_us << ","
              << stats.avg_cycles << "," << stats.min_cycles << "," << stats.max_cycles << "," << device_coord
              << std::endl;
}

void emit_d2h_throughput_csv_row(
    std::size_t page_size,
    std::size_t socket_fifo_size,
    std::size_t total_data,
    std::size_t data_size,
    std::size_t pages_per_iter,
    uint32_t num_iterations,
    uint64_t total_pages,
    double avg_per_page_us,
    double avg_per_page_cycles,
    const MeshCoordinate& device_coord) {
    const double throughput_gbps = static_cast<double>(page_size) / (avg_per_page_us * 1e3);
    std::cout << page_size << "," << socket_fifo_size << "," << total_data << "," << data_size << "," << pages_per_iter
              << "," << num_iterations << "," << total_pages << "," << avg_per_page_us << "," << avg_per_page_cycles
              << "," << throughput_gbps << "," << device_coord << std::endl;
}

void emit_h2d_throughput_csv_row(
    std::size_t page_size,
    std::size_t socket_fifo_size,
    H2DMode mode,
    std::size_t total_data,
    std::size_t data_size,
    std::size_t pages_per_iter,
    uint32_t num_iterations,
    uint64_t total_pages,
    double avg_per_page_us,
    double avg_per_page_cycles,
    const MeshCoordinate& device_coord) {
    const double throughput_gbps = static_cast<double>(page_size) / (avg_per_page_us * 1e3);
    std::cout << page_size << "," << socket_fifo_size << "," << h2d_mode_name(mode) << "," << total_data << ","
              << data_size << "," << pages_per_iter << "," << num_iterations << "," << total_pages << ","
              << avg_per_page_us << "," << avg_per_page_cycles << "," << throughput_gbps << "," << device_coord
              << std::endl;
}

}  // namespace

void test_h2d_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    H2DMode h2d_mode,
    uint32_t num_iterations = 10,
    const MeshCoreCoord& recv_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)}) {
    auto input_socket = H2DSocket(mesh_device, recv_core, BufferType::L1, socket_fifo_size, h2d_mode);
    input_socket.set_page_size(page_size);

    TT_FATAL(data_size % page_size == 0, "Data size must be a multiple of page size");

    auto recv_data_buffer = make_l1_mesh_buffer(mesh_device.get(), recv_core.core_coord, data_size);
    auto recv_program = CreateProgram();
    CreateKernel(
        recv_program,
        h2d_mode == H2DMode::DEVICE_PULL ? "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_receiver.cpp"
                                         : "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(input_socket.get_config_buffer_address()),
                static_cast<uint32_t>(recv_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(num_iterations),
            }});

    enqueue_on_core(*mesh_device, recv_core, std::move(recv_program));

    uint32_t num_writes = data_size / page_size;
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));

    auto recv_core_virtual = mesh_device->worker_core_from_logical_core(recv_core.core_coord);
    uint32_t page_size_words = page_size / sizeof(uint32_t);
    auto start_time = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++) {
        for (uint32_t j = 0; j < num_writes; j++) {
            input_socket.write(src_vec.data() + (j * page_size_words), 1);
        }
        input_socket.barrier();
        std::vector<uint32_t> recv_data_readback(data_size / sizeof(uint32_t));
        MetalContext::instance().get_cluster().read_core(
            recv_data_readback.data(),
            data_size,
            tt_cxy_pair(mesh_device->get_device(recv_core.device_coord)->id(), recv_core_virtual),
            recv_data_buffer->address());
        EXPECT_EQ(src_vec, recv_data_readback);
    }
    input_socket.barrier();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "Write time: " << duration.count() << "ns for: " << recv_core.device_coord << " "
              << num_writes * num_iterations << " writes" << std::endl;
}

void test_d2h_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)}) {
    auto output_socket = D2HSocket(mesh_device, sender_core, socket_fifo_size);
    output_socket.set_page_size(page_size);

    TT_FATAL(data_size % page_size == 0, "Data size must be a multiple of page size");

    uint32_t num_txns = data_size / page_size;

    auto sender_data_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, data_size);
    // The kernel is invoked with num_iterations=1 and writes a single uint64_t timestamp.
    auto measurement_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, sizeof(uint64_t));

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_sender.cpp",
        sender_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(output_socket.get_config_buffer_address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(1),  // 1 kernel iteration: host reads exactly num_txns pages
            }});

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);
    WriteShard(mesh_device->mesh_command_queue(), sender_data_buffer, src_vec, sender_core.device_coord);
    enqueue_on_core(*mesh_device, sender_core, std::move(send_program));

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_txns; i++) {
        output_socket.read(dst_vec.data() + (i * page_size_words), 1);
    }
    output_socket.barrier();
    EXPECT_EQ(src_vec, dst_vec);

    // Ensure the device kernel has flushed the measurement write to L1 before reading it.
    Finish(mesh_device->mesh_command_queue());

    auto physical_system_descriptor = make_physical_system_descriptor();
    auto fabric_node_id = mesh_device->get_fabric_node_id(sender_core.device_coord);
    auto asic_id = MetalContext::instance().get_control_plane().get_asic_id_from_fabric_node_id(fabric_node_id);
    auto asic_desc = physical_system_descriptor.get_asic_descriptors()[asic_id];

    uint64_t total_cycles = read_l1_uint64(*mesh_device, sender_core, measurement_buffer->address());
    double avg_cycles_per_transaction = static_cast<double>(total_cycles) / num_txns;
    double avg_latency_us = avg_cycles_per_transaction / kCyclesPerUs;

    std::cout << "Average D2H Round-Trip Latency: " << avg_latency_us << " us"
              << " (cycles: " << avg_cycles_per_transaction << ")"
              << " for: " << sender_core.device_coord << " Tray ID: " << *(asic_desc.tray_id)
              << " ASIC Location: " << *(asic_desc.asic_location) << std::endl;
}

// Forward declaration for benchmark helper
std::pair<double, double> benchmark_d2h_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core);

void test_hd_socket_loopback(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    H2DMode h2d_mode,
    uint32_t num_iterations = 10,
    const MeshCoreCoord& socket_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)}) {
    auto input_socket = H2DSocket(mesh_device, socket_core, BufferType::L1, socket_fifo_size, h2d_mode);
    auto output_socket = D2HSocket(mesh_device, socket_core, socket_fifo_size);

    input_socket.set_page_size(page_size);
    output_socket.set_page_size(page_size);

    TT_FATAL(data_size % page_size == 0, "Data size must be a multiple of page size");

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
        socket_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(input_socket.get_config_buffer_address()),
                static_cast<uint32_t>(output_socket.get_config_buffer_address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(num_iterations),
                h2d_mode == H2DMode::DEVICE_PULL,
            }});

    uint32_t num_txns = data_size / page_size;
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(data_size / sizeof(uint32_t));

    enqueue_on_core(*mesh_device, socket_core, std::move(send_program));

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_iterations; i++) {
        std::iota(src_vec.begin(), src_vec.end(), i);
        for (uint32_t j = 0; j < num_txns; j++) {
            input_socket.write(src_vec.data() + (j * page_size_words), 1);
            output_socket.read(dst_vec.data() + (j * page_size_words), 1);
        }
    }
    input_socket.barrier();
    output_socket.barrier();
    EXPECT_EQ(src_vec, dst_vec);
}

void test_hd_socket_multithreaded_loopback(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    H2DMode h2d_mode,
    uint32_t num_iterations = 10,
    const MeshCoreCoord& socket_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)}) {
    auto input_socket = H2DSocket(mesh_device, socket_core, BufferType::L1, socket_fifo_size, h2d_mode);
    auto output_socket = D2HSocket(mesh_device, socket_core, socket_fifo_size);

    TT_FATAL(data_size % page_size == 0, "Data size must be a multiple of page size");

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
        socket_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(input_socket.get_config_buffer_address()),
                static_cast<uint32_t>(output_socket.get_config_buffer_address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(num_iterations),
                h2d_mode == H2DMode::DEVICE_PULL,
            }});

    uint32_t num_txns = data_size / page_size;
    std::vector<uint32_t> src_vec(data_size * num_iterations / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(data_size * num_iterations / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    enqueue_on_core(*mesh_device, socket_core, std::move(send_program));

    input_socket.set_page_size(page_size);
    output_socket.set_page_size(page_size);

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    uint32_t data_size_words = data_size / sizeof(uint32_t);

    std::thread write_thread([&]() {
        for (uint32_t i = 0; i < num_iterations; i++) {
            for (uint32_t j = 0; j < num_txns; j++) {
                input_socket.write(src_vec.data() + (i * data_size_words) + (j * page_size_words), 1);
            }
        }
    });

    std::thread read_thread([&]() {
        for (uint32_t i = 0; i < num_iterations; i++) {
            for (uint32_t j = 0; j < num_txns; j++) {
                output_socket.read(dst_vec.data() + (i * data_size_words) + (j * page_size_words), 1);
            }
        }
    });
    // Barrier with a timeout in the main thread ensure that the read/write threads are not hung.
    input_socket.barrier(10000);
    output_socket.barrier(10000);

    write_thread.join();
    read_thread.join();

    EXPECT_EQ(src_vec, dst_vec);
}

bool is_device_coord_mmio_mapped(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, const MeshCoordinate& device_coord) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto device_id = mesh_device->get_device(device_coord)->id();
    return cluster.get_associated_mmio_device(device_id) == device_id;
}

using HDSocketFixture = MeshDevice1x2Fixture;
TEST_F(HDSocketFixture, H2DSocket) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        for (const auto& recv_coord : MeshCoordinateRange(mesh_device_->shape())) {
            if (!is_device_coord_mmio_mapped(mesh_device_, recv_coord)) {
                continue;
            }
            test_h2d_socket(mesh_device_, 1024, 64, 4096, h2d_mode, 500, MeshCoreCoord(recv_coord, CoreCoord(0, 0)));
        }
    }
}

TEST_F(HDSocketFixture, D2HSocket) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    for (const auto& sender_coord : MeshCoordinateRange(mesh_device_->shape())) {
        if (!is_device_coord_mmio_mapped(mesh_device_, sender_coord)) {
            continue;
        }
        test_d2h_socket(mesh_device_, 1024, 64, 4096, 500, MeshCoreCoord(sender_coord, CoreCoord(0, 0)));
    }
}

TEST_F(HDSocketFixture, D2HSocketThroughputBenchmark) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    MeshCoreCoord sender_core = get_target_benchmark_worker_core(mesh_device_);
    const auto& sender_coord = sender_core.device_coord;
    std::cout << "# sender target tray=" << kTargetTrayId << " asic_location=" << kTargetAsicLocation
              << " mesh_coord=" << sender_coord << std::endl;

    std::cout << "page_size,socket_fifo_size,total_data,data_size,pages_per_iter,"
              << "num_iterations,total_pages,avg_per_page_us,avg_per_page_cycles,"
              << "throughput_gbps,device_coord" << std::endl;

    for (auto fifo_size : kD2HThroughputFifoSizes) {
        for (auto page_size : kPageSizes) {
            if (page_size > fifo_size) {
                continue;
            }
            std::size_t data_size = compute_iteration_data_size_bytes(page_size);
            std::size_t pages_per_iter = data_size / page_size;

            for (auto total_data : kTotalDataSizes) {
                uint32_t num_iterations = total_data / data_size;
                if (num_iterations == 0) {
                    continue;
                }
                uint64_t total_pages = static_cast<uint64_t>(pages_per_iter) * num_iterations;

                auto [us, cycles] =
                    benchmark_d2h_socket(mesh_device_, fifo_size, page_size, data_size, num_iterations, sender_core);

                emit_d2h_throughput_csv_row(
                    page_size,
                    fifo_size,
                    total_data,
                    data_size,
                    pages_per_iter,
                    num_iterations,
                    total_pages,
                    us,
                    cycles,
                    sender_coord);
                std::cout.flush();
            }
        }
    }
}

TEST_F(HDSocketFixture, D2HSocketMultiChipMaxThroughputBenchmark) {
    // Iterates over every MMIO-mapped chip on the system and measures D2H throughput
    // at 64KB page size (max-throughput configuration) across a range of FIFO sizes
    // that quadruple each step: 1MB → 4MB → 16MB → 64MB → 256MB.
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    constexpr std::size_t kBenchPageSize = 65536;               // 64KB – max throughput
    constexpr std::size_t kBenchTotalData = 1024UL * 1024 * 1024;  // 1GB
    const std::vector<std::size_t> kBenchFifoSizes = {
        1UL * 1024 * 1024,    //   1MB
        4UL * 1024 * 1024,    //   4MB
        16UL * 1024 * 1024,   //  16MB
        64UL * 1024 * 1024,   //  64MB
        256UL * 1024 * 1024,  // 256MB
    };

    auto mmio_chips = enumerate_mmio_chips(mesh_device_);

    std::size_t data_size = compute_iteration_data_size_bytes(kBenchPageSize);
    std::size_t pages_per_iter = data_size / kBenchPageSize;
    uint32_t num_iterations = kBenchTotalData / data_size;
    uint64_t total_pages = static_cast<uint64_t>(pages_per_iter) * num_iterations;

    std::cout << "# D2HSocketMultiChipMaxThroughputBenchmark" << std::endl;
    std::cout << "# page_size=64KB  total_data=1GB  chips=" << mmio_chips.size() << std::endl;
    std::cout << std::endl;
    std::cout << "tray_id,asic_location,mesh_coord,socket_fifo_size,total_data,data_size,"
              << "pages_per_iter,num_iterations,total_pages,avg_per_page_us,avg_per_page_cycles,"
              << "throughput_gbps" << std::endl;

    for (const auto& chip : mmio_chips) {
        std::cout << "# chip: tray_id=" << chip.tray_id << " asic_location=" << chip.asic_location
                  << " mesh_coord=" << chip.coord << std::endl;

        MeshCoreCoord sender_core{chip.coord, CoreCoord(0, 0)};

        for (auto fifo_size : kBenchFifoSizes) {
            auto [us, cycles] =
                benchmark_d2h_socket(mesh_device_, fifo_size, kBenchPageSize, data_size, num_iterations, sender_core);

            const double throughput_gbps = static_cast<double>(kBenchPageSize) / (us * 1e3);
            std::cout << chip.tray_id << "," << chip.asic_location << "," << chip.coord << ","
                      << fifo_size << "," << kBenchTotalData << "," << data_size << ","
                      << pages_per_iter << "," << num_iterations << "," << total_pages << ","
                      << us << "," << cycles << "," << throughput_gbps << std::endl;
            std::cout.flush();
        }
    }
}

std::pair<double, double> benchmark_h2d_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    uint32_t num_iterations,
    H2DMode h2d_mode,
    const MeshCoreCoord& recv_core);

TEST_F(HDSocketFixture, H2DSocketMultiChipMaxThroughputBenchmark) {
    // Iterates over every MMIO-mapped chip and measures H2D throughput using DEVICE_PULL
    // at 256KB page size (max-throughput configuration for DEVICE_PULL) across a range
    // of FIFO sizes: 256KB → 512KB → 768KB → 1MB.
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    constexpr std::size_t kBenchPageSize = 262144;                 // 256KB – max throughput for DEVICE_PULL
    constexpr std::size_t kBenchTotalData = 1024UL * 1024 * 1024;  // 1GB
    const std::vector<std::size_t> kBenchFifoSizes = {
        256 * 1024,   // 256KB (depth 1)
        512 * 1024,   // 512KB (depth 2)
        768 * 1024,   // 768KB (depth 3)
        1024 * 1024,  //   1MB (depth 4)
    };

    auto mmio_chips = enumerate_mmio_chips(mesh_device_);

    std::size_t data_size = compute_iteration_data_size_bytes(kBenchPageSize);
    std::size_t pages_per_iter = data_size / kBenchPageSize;
    uint32_t num_iterations = kBenchTotalData / data_size;
    uint64_t total_pages = static_cast<uint64_t>(pages_per_iter) * num_iterations;

    std::cout << "# H2DSocketMultiChipMaxThroughputBenchmark" << std::endl;
    std::cout << "# page_size=256KB  mode=DEVICE_PULL  total_data=1GB  chips=" << mmio_chips.size() << std::endl;
    std::cout << std::endl;
    std::cout << "tray_id,asic_location,mesh_coord,socket_fifo_size,total_data,data_size,"
              << "pages_per_iter,num_iterations,total_pages,avg_per_page_us,avg_per_page_cycles,"
              << "throughput_gbps" << std::endl;

    for (const auto& chip : mmio_chips) {
        std::cout << "# chip: tray_id=" << chip.tray_id << " asic_location=" << chip.asic_location
                  << " mesh_coord=" << chip.coord << std::endl;

        MeshCoreCoord recv_core{chip.coord, CoreCoord(0, 0)};

        for (auto fifo_size : kBenchFifoSizes) {
            auto [us, cycles] = benchmark_h2d_socket(
                mesh_device_, fifo_size, kBenchPageSize, data_size, num_iterations, H2DMode::DEVICE_PULL, recv_core);

            const double throughput_gbps = static_cast<double>(kBenchPageSize) / (us * 1e3);
            std::cout << chip.tray_id << "," << chip.asic_location << "," << chip.coord << "," << fifo_size << ","
                      << kBenchTotalData << "," << data_size << "," << pages_per_iter << "," << num_iterations << ","
                      << total_pages << "," << us << "," << cycles << "," << throughput_gbps << std::endl;
            std::cout.flush();
        }
    }
}

// Returns per-page latency in microseconds and cycles.
std::pair<double, double> benchmark_d2h_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core) {
    auto output_socket = D2HSocket(mesh_device, sender_core, socket_fifo_size);
    output_socket.set_page_size(page_size);

    uint32_t num_txns = data_size / page_size;

    auto sender_data_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, data_size);
    auto measurement_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, sizeof(uint64_t));

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_sender.cpp",
        sender_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(output_socket.get_config_buffer_address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(num_iterations),
            }});

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);
    WriteShard(mesh_device->mesh_command_queue(), sender_data_buffer, src_vec, sender_core.device_coord, true);
    enqueue_on_core(*mesh_device, sender_core, std::move(send_program));

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_iterations; i++) {
        for (uint32_t j = 0; j < num_txns; j++) {
            output_socket.read(dst_vec.data() + (j * page_size_words), 1);
        }
    }
    output_socket.barrier();
    // Ensure the launched program has fully completed and flushed measurement writes
    // before reading measurement_buffer from host.
    Finish(mesh_device->mesh_command_queue());

    uint64_t total_cycles = read_l1_uint64(*mesh_device, sender_core, measurement_buffer->address());
    uint64_t total_transactions = num_txns * num_iterations;
    double avg_cycles_per_transaction = static_cast<double>(total_cycles) / total_transactions;
    double avg_latency_us = avg_cycles_per_transaction / kCyclesPerUs;

    return {avg_latency_us, avg_cycles_per_transaction};
}

// D2H Latency: per-iteration round-trip measurement using pcie_socket_data_ping kernel.
// Returns a vector of per-iteration cycle counts.
std::vector<uint64_t> benchmark_d2h_latency(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core) {
    auto output_socket = D2HSocket(mesh_device, sender_core, socket_fifo_size);
    output_socket.set_page_size(page_size);

    // Data buffer: single page in L1 (kernel always sends from same address)
    auto sender_data_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, page_size);

    // Measurement buffer: one uint64_t per iteration
    uint32_t measurement_buffer_size = num_iterations * sizeof(uint64_t);
    auto measurement_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, measurement_buffer_size);

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_data_ping.cpp",
        sender_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(output_socket.get_config_buffer_address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(page_size),  // data_size = page_size (unused by ping kernel)
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(num_iterations),
            }});

    enqueue_on_core(*mesh_device, sender_core, std::move(send_program));

    // Host side: read warmup + timed pages (must match kernel's WARMUP_ITERS)
    uint32_t page_size_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> dst_vec(page_size_words);
    for (uint32_t w = 0; w < kWarmupIters; w++) {
        output_socket.read(dst_vec.data(), 1);
    }
    for (uint32_t i = 0; i < num_iterations; i++) {
        output_socket.read(dst_vec.data(), 1);
    }
    output_socket.barrier();
    // Ensure the device kernel has flushed all per-iteration timestamps to L1.
    Finish(mesh_device->mesh_command_queue());

    std::vector<uint64_t> cycles(num_iterations);
    read_l1_uint64s(*mesh_device, sender_core, measurement_buffer->address(), cycles);

    return cycles;
}

TEST_F(HDSocketFixture, D2HSocketLatencyBenchmark) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    uint32_t num_iterations = 100;

    MeshCoreCoord sender_core = get_target_benchmark_worker_core(mesh_device_);
    const auto& sender_coord = sender_core.device_coord;

    std::cout << "page_size,socket_fifo_size,num_iterations,"
              << "avg_us,min_us,max_us,p50_us,p99_us,"
              << "avg_cycles,min_cycles,max_cycles,device_coord" << std::endl;

    for (auto fifo_size : kD2HLatencyFifoSizes) {
        for (auto page_size : kPageSizes) {
            if (page_size > fifo_size) {
                continue;
            }

            auto cycles = benchmark_d2h_latency(mesh_device_, fifo_size, page_size, num_iterations, sender_core);

            auto stats = summarize_latency_cycles(cycles);
            emit_latency_csv_row(page_size, fifo_size, num_iterations, stats, sender_coord);
            std::cout.flush();
        }
    }
}

TEST_F(HDSocketFixture, H2DSocketThroughputBenchmark) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    MeshCoreCoord recv_core = get_target_benchmark_worker_core(mesh_device_);
    const auto& recv_coord = recv_core.device_coord;

    std::cout << "page_size,socket_fifo_size,h2d_mode,total_data,data_size,pages_per_iter,"
              << "num_iterations,total_pages,avg_per_page_us,avg_per_page_cycles,"
              << "throughput_gbps,device_coord" << std::endl;

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        for (auto fifo_size : kH2DThroughputFifoSizes) {
            for (auto page_size : kPageSizes) {
                if (page_size > fifo_size) {
                    continue;
                }
                std::size_t data_size = compute_iteration_data_size_bytes(page_size);
                std::size_t pages_per_iter = data_size / page_size;

                for (auto total_data : kTotalDataSizes) {
                    uint32_t num_iterations = total_data / data_size;
                    if (num_iterations == 0) {
                        continue;
                    }
                    uint64_t total_pages = static_cast<uint64_t>(pages_per_iter) * num_iterations;

                    auto [us, cycles] = benchmark_h2d_socket(
                        mesh_device_, fifo_size, page_size, data_size, num_iterations, h2d_mode, recv_core);

                    emit_h2d_throughput_csv_row(
                        page_size,
                        fifo_size,
                        h2d_mode,
                        total_data,
                        data_size,
                        pages_per_iter,
                        num_iterations,
                        total_pages,
                        us,
                        cycles,
                        recv_coord);
                    std::cout.flush();
                }
            }
        }
    }
}

// D2H Ping: pure signaling round-trip, no data DMA.
std::vector<uint64_t> benchmark_d2h_ping(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core) {
    auto output_socket = D2HSocket(mesh_device, sender_core, socket_fifo_size);
    output_socket.set_page_size(page_size);

    auto sender_data_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, page_size);
    uint32_t measurement_buffer_size = num_iterations * sizeof(uint64_t);
    auto measurement_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, measurement_buffer_size);

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_ping.cpp",
        sender_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(output_socket.get_config_buffer_address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(num_iterations),
            }});

    enqueue_on_core(*mesh_device, sender_core, std::move(send_program));

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> dst_vec(page_size_words);
    for (uint32_t i = 0; i < kWarmupIters; i++) {
        output_socket.read(dst_vec.data(), 1);
    }
    for (uint32_t i = 0; i < num_iterations; i++) {
        output_socket.read(dst_vec.data(), 1);
    }
    output_socket.barrier();
    // Ensure the device kernel has flushed all per-iteration timestamps to L1.
    Finish(mesh_device->mesh_command_queue());

    std::vector<uint64_t> cycles(num_iterations);
    read_l1_uint64s(*mesh_device, sender_core, measurement_buffer->address(), cycles);

    return cycles;
}

TEST_F(HDSocketFixture, D2HSocketPingBenchmark) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    std::vector<std::size_t> page_sizes = {64};
    std::vector<std::size_t> fifo_sizes = {4096};
    uint32_t num_iterations = 100;

    MeshCoreCoord sender_core = get_target_benchmark_worker_core(mesh_device_);
    const auto& sender_coord = sender_core.device_coord;

    std::cout << "page_size,socket_fifo_size,num_iterations,"
              << "avg_us,min_us,max_us,p50_us,p99_us,"
              << "avg_cycles,min_cycles,max_cycles,device_coord" << std::endl;

    bool dumped_iterations = false;
    for (auto fifo_size : fifo_sizes) {
        for (auto page_size : page_sizes) {
            if (page_size > fifo_size) {
                continue;
            }

            auto cycles = benchmark_d2h_ping(mesh_device_, fifo_size, page_size, num_iterations, sender_core);

            // Dump per-iteration data for the first config
            if (!dumped_iterations) {
                std::ofstream iter_file("d2h_ping_iterations.csv");
                iter_file << "iteration,latency_us,cycles" << std::endl;
                for (uint32_t i = 0; i < num_iterations; i++) {
                    double lat_us = static_cast<double>(cycles[i]) / kCyclesPerUs;
                    iter_file << i << "," << lat_us << "," << cycles[i] << std::endl;
                }
                iter_file.close();
                dumped_iterations = true;
            }

            auto stats = summarize_latency_cycles(cycles);
            emit_latency_csv_row(page_size, fifo_size, num_iterations, stats, sender_coord);
            std::cout.flush();
        }
    }
}

std::vector<uint64_t> benchmark_h2d_ping(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    H2DMode h2d_mode,
    const MeshCoreCoord& recv_core) {
    auto input_socket = H2DSocket(mesh_device, recv_core, BufferType::L1, socket_fifo_size, h2d_mode);
    input_socket.set_page_size(page_size);

    uint32_t measurement_buffer_size = num_iterations * sizeof(uint64_t);
    auto measurement_buffer = make_l1_mesh_buffer(mesh_device.get(), recv_core.core_coord, measurement_buffer_size);

    auto recv_program = CreateProgram();
    CreateKernel(
        recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/h2d_socket_ping.cpp",
        recv_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(input_socket.get_config_buffer_address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(num_iterations),
            }});

    enqueue_on_core(*mesh_device, recv_core, std::move(recv_program));

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> src_vec(page_size_words);
    for (uint32_t w = 0; w < kWarmupIters; w++) {
        input_socket.write(src_vec.data(), 1);
        input_socket.barrier();
    }
    for (uint32_t i = 0; i < num_iterations; i++) {
        input_socket.write(src_vec.data(), 1);
        input_socket.barrier();
    }
    // Ensure the device kernel has flushed all per-iteration timestamps to L1.
    Finish(mesh_device->mesh_command_queue());

    std::vector<uint64_t> cycles(num_iterations);
    read_l1_uint64s(*mesh_device, recv_core, measurement_buffer->address(), cycles);

    return cycles;
}

TEST_F(HDSocketFixture, H2DSocketPingBenchmark) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    std::size_t page_size = 64;
    std::size_t fifo_size = 4096;
    uint32_t num_iterations = 100;

    MeshCoreCoord recv_core = get_target_benchmark_worker_core(mesh_device_);
    const auto& recv_coord = recv_core.device_coord;

    std::vector<H2DMode> modes = {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL};

    std::cout << "page_size,socket_fifo_size,h2d_mode,num_iterations,"
              << "avg_us,min_us,max_us,p50_us,p99_us,"
              << "avg_cycles,min_cycles,max_cycles,device_coord" << std::endl;

    for (auto h2d_mode : modes) {
        auto cycles = benchmark_h2d_ping(mesh_device_, fifo_size, page_size, num_iterations, h2d_mode, recv_core);

        // Dump per-iteration data
        std::string iter_filename = std::string("h2d_ping_iterations_") + h2d_mode_name(h2d_mode) + ".csv";
        std::ofstream iter_file(iter_filename);
        iter_file << "iteration,latency_us,cycles" << std::endl;
        for (uint32_t i = 0; i < num_iterations; i++) {
            double lat_us = static_cast<double>(cycles[i]) / kCyclesPerUs;
            iter_file << i << "," << lat_us << "," << cycles[i] << std::endl;
        }
        iter_file.close();

        auto stats = summarize_latency_cycles(cycles);
        emit_latency_csv_row(
            page_size, fifo_size, std::string(h2d_mode_name(h2d_mode)), num_iterations, stats, recv_coord);
        std::cout.flush();
    }
}

std::pair<double, double> benchmark_h2d_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    uint32_t num_iterations,
    H2DMode h2d_mode,
    const MeshCoreCoord& recv_core) {
    auto input_socket = H2DSocket(mesh_device, recv_core, BufferType::L1, socket_fifo_size, h2d_mode);
    input_socket.set_page_size(page_size);

    uint32_t num_txns = data_size / page_size;

    auto recv_data_buffer = make_l1_mesh_buffer(mesh_device.get(), recv_core.core_coord, data_size);
    auto measurement_buffer = make_l1_mesh_buffer(mesh_device.get(), recv_core.core_coord, sizeof(uint64_t));

    const char* kernel_path = (h2d_mode == H2DMode::DEVICE_PULL)
                                  ? "tests/tt_metal/tt_metal/test_kernels/misc/socket/h2d_throughput_device_pull.cpp"
                                  : "tests/tt_metal/tt_metal/test_kernels/misc/socket/h2d_throughput_host_push.cpp";

    auto recv_program = CreateProgram();
    CreateKernel(
        recv_program,
        kernel_path,
        recv_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(input_socket.get_config_buffer_address()),
                static_cast<uint32_t>(recv_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(num_iterations),
            }});

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    enqueue_on_core(*mesh_device, recv_core, std::move(recv_program));
    uint32_t page_size_words = page_size / sizeof(uint32_t);

    for (uint32_t i = 0; i < num_iterations; i++) {
        for (uint32_t j = 0; j < num_txns; j++) {
            input_socket.write(src_vec.data() + (j * page_size_words), 1);
        }
    }
    input_socket.barrier();
    // Ensure the device kernel has fully completed and flushed the measurement
    // write before reading measurement_buffer from host.
    Finish(mesh_device->mesh_command_queue());

    uint64_t total_cycles = read_l1_uint64(*mesh_device, recv_core, measurement_buffer->address());
    uint64_t total_pages = static_cast<uint64_t>(num_txns) * num_iterations;
    double avg_per_page_cycles = static_cast<double>(total_cycles) / total_pages;
    double avg_per_page_us = avg_per_page_cycles / kCyclesPerUs;

    return {avg_per_page_us, avg_per_page_cycles};
}

std::vector<uint64_t> benchmark_h2d_latency(
    const std::shared_ptr<MeshDevice>& mesh_device,
    std::size_t fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    H2DMode h2d_mode,
    const MeshCoreCoord& recv_core) {
    auto input_socket = H2DSocket(mesh_device, recv_core, BufferType::L1, fifo_size, h2d_mode);
    input_socket.set_page_size(page_size);

    // Recv data buffer: single page in L1 (kernel overwrites each iteration)
    auto recv_data_buffer = make_l1_mesh_buffer(mesh_device.get(), recv_core.core_coord, page_size);

    // Measurement buffer: one uint64_t per iteration
    uint32_t measurement_buffer_size = num_iterations * sizeof(uint64_t);
    auto measurement_buffer = make_l1_mesh_buffer(mesh_device.get(), recv_core.core_coord, measurement_buffer_size);

    const char* kernel_path =
        (h2d_mode == H2DMode::DEVICE_PULL)
            ? "tests/tt_metal/tt_metal/test_kernels/misc/socket/h2d_socket_data_ping_device_pull.cpp"
            : "tests/tt_metal/tt_metal/test_kernels/misc/socket/h2d_socket_data_ping_host_push.cpp";
    auto recv_program = CreateProgram();
    CreateKernel(
        recv_program,
        kernel_path,
        recv_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(input_socket.get_config_buffer_address()),
                static_cast<uint32_t>(recv_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(num_iterations),
            }});

    enqueue_on_core(*mesh_device, recv_core, std::move(recv_program));

    // Host side: write + barrier per page, matching kernel's warmup + timed iterations
    uint32_t page_size_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> src_vec(page_size_words);
    for (uint32_t w = 0; w < kWarmupIters; w++) {
        input_socket.write(src_vec.data(), 1);
        input_socket.barrier();
    }
    for (uint32_t i = 0; i < num_iterations; i++) {
        input_socket.write(src_vec.data(), 1);
        input_socket.barrier();
    }
    // Ensure the device kernel has flushed all per-iteration timestamps to L1.
    Finish(mesh_device->mesh_command_queue());

    std::vector<uint64_t> cycles(num_iterations);
    read_l1_uint64s(*mesh_device, recv_core, measurement_buffer->address(), cycles);

    return cycles;
}

TEST_F(HDSocketFixture, H2DSocketLatencyBenchmark) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    uint32_t num_iterations = 100;

    MeshCoreCoord recv_core = get_target_benchmark_worker_core(mesh_device_);
    const auto& recv_coord = recv_core.device_coord;

    std::cout << "page_size,socket_fifo_size,h2d_mode,num_iterations,"
              << "avg_us,min_us,max_us,p50_us,p99_us,"
              << "avg_cycles,min_cycles,max_cycles,device_coord" << std::endl;

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        std::string mode_str = h2d_mode_name(h2d_mode);
        for (auto fifo_size : kH2DLatencyFifoSizes) {
            for (auto page_size : kPageSizes) {
                if (page_size > fifo_size) {
                    continue;
                }

                auto cycles =
                    benchmark_h2d_latency(mesh_device_, fifo_size, page_size, num_iterations, h2d_mode, recv_core);

                auto stats = summarize_latency_cycles(cycles);
                emit_latency_csv_row(page_size, fifo_size, mode_str, num_iterations, stats, recv_coord);
                std::cout.flush();
            }
        }
    }
}

TEST_F(HDSocketFixture, H2DSocketLoopback) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    for (auto h2d_mode : {H2DMode::DEVICE_PULL, H2DMode::HOST_PUSH}) {
        for (const auto& socket_coord : MeshCoordinateRange(mesh_device_->shape())) {
            if (!is_device_coord_mmio_mapped(mesh_device_, socket_coord)) {
                continue;
            }
            // No wrap
            test_hd_socket_loopback(
                mesh_device_, 1024, 64, 1024, h2d_mode, 50, MeshCoreCoord(socket_coord, CoreCoord(0, 0)));
            // Even wrap
            test_hd_socket_loopback(
                mesh_device_, 1024, 64, 32768, h2d_mode, 50, MeshCoreCoord(socket_coord, CoreCoord(1, 1)));
            // Uneven wrap
            test_hd_socket_loopback(
                mesh_device_, 4096, 1088, 78336, h2d_mode, 50, MeshCoreCoord(socket_coord, CoreCoord(0, 1)));
            // Uneven wrap with multiple pages on host allocated.
            // On most hosts, page size is 4K, so this should lead to 5 pages being allocated on the host.
            test_hd_socket_loopback(
                mesh_device_, 16512, 1088, 156672, h2d_mode, 50, MeshCoreCoord(socket_coord, CoreCoord(0, 1)));
        }
    }
}

TEST_F(HDSocketFixture, H2DSocketLoopbackMultiThreadedStress) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    for (auto h2d_mode : {H2DMode::DEVICE_PULL, H2DMode::HOST_PUSH}) {
        for (const auto& socket_coord : MeshCoordinateRange(mesh_device_->shape())) {
            if (!is_device_coord_mmio_mapped(mesh_device_, socket_coord)) {
                continue;
            }
            // No wrap
            test_hd_socket_multithreaded_loopback(
                mesh_device_, 1024, 64, 1024, h2d_mode, 100, MeshCoreCoord(socket_coord, CoreCoord(0, 0)));
            // Even wrap
            test_hd_socket_multithreaded_loopback(
                mesh_device_, 1024, 64, 32768, h2d_mode, 100, MeshCoreCoord(socket_coord, CoreCoord(1, 1)));
            // Uneven wrap
            test_hd_socket_multithreaded_loopback(
                mesh_device_, 4096, 1088, 78336, h2d_mode, 100, MeshCoreCoord(socket_coord, CoreCoord(0, 1)));
            // Uneven wrap with multiple pages on host allocated.
            // On most hosts, page size is 4K, so this should lead to 5 pages being allocated on the host.
            test_hd_socket_multithreaded_loopback(
                mesh_device_, 16512, 1088, 156672, h2d_mode, 100, MeshCoreCoord(socket_coord, CoreCoord(0, 1)));
        }
    }
}

}  // namespace tt::tt_metal::distributed
