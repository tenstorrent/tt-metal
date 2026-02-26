// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hd_socket_test_utils.hpp"

#include <benchmark/benchmark.h>

#include <numeric>
#include <sstream>

namespace tt::tt_metal::distributed {

namespace {

constexpr uint32_t kTargetTrayId = 1;
constexpr uint32_t kTargetAsicLocation = 1;

std::optional<MeshCoordinate> find_target_mmio_coord(
    const std::shared_ptr<MeshDevice>& mesh_device, uint32_t tray_id, uint32_t asic_location) {
    auto phys_desc = make_physical_system_descriptor();
    const auto& cp = MetalContext::instance().get_control_plane();
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        if (!is_device_coord_mmio_mapped(mesh_device, coord)) {
            continue;
        }
        auto asic_id = cp.get_asic_id_from_fabric_node_id(mesh_device->get_fabric_node_id(coord));
        auto desc = phys_desc.get_asic_descriptors()[asic_id];
        if (*desc.tray_id == tray_id && *desc.asic_location == asic_location) {
            return coord;
        }
    }
    return std::nullopt;
}

MeshCoreCoord get_target_worker_core(const std::shared_ptr<MeshDevice>& mesh_device) {
    auto coord = find_target_mmio_coord(mesh_device, kTargetTrayId, kTargetAsicLocation);
    TT_FATAL(coord.has_value(), "Target chip not found (tray={} asic={})", kTargetTrayId, kTargetAsicLocation);
    return MeshCoreCoord{coord.value(), CoreCoord(0, 0)};
}

// Returns average per-page latency in (microseconds, cycles).
std::pair<double, double> run_d2h_throughput(
    const std::shared_ptr<MeshDevice>& mesh_device,
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
    execute_program_on_device(*mesh_device, sender_core.device_coord, std::move(send_program));

    for (uint32_t i = 0; i < num_iterations; i++) {
        for (uint32_t j = 0; j < num_txns; j++) {
            output_socket.read(dst_vec.data() + (j * (page_size / sizeof(uint32_t))), 1);
        }
    }
    output_socket.barrier();
    Finish(mesh_device->mesh_command_queue());

    uint64_t total_cycles = read_l1_uint64(*mesh_device, sender_core, measurement_buffer->address());
    double avg_cycles = static_cast<double>(total_cycles) / (num_txns * num_iterations);
    double avg_us = avg_cycles / get_cycles_per_us(*mesh_device);
    return {avg_us, avg_cycles};
}

// Args: [0]=page_size  [1]=fifo_size  [2]=total_data
static void BM_D2HSocketThroughput(benchmark::State& state) {
    // MeshDevice is created once and shared across all benchmark configurations.
    static auto mesh_device = MeshDevice::create(
        MeshDeviceConfig(MeshShape{1, 2}),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_cqs=*/1,
        DispatchCoreType::WORKER);
    static MeshCoreCoord sender_core = get_target_worker_core(mesh_device);

    auto page_size = static_cast<std::size_t>(state.range(0));
    auto fifo_size = static_cast<std::size_t>(state.range(1));
    auto total_data = static_cast<std::size_t>(state.range(2));

    if (!experimental::GetMemoryPinningParameters(*mesh_device).can_map_to_noc) {
        state.SkipWithMessage("Mapping host memory to NOC is not supported on this system");
        return;
    }

    if (page_size > fifo_size) {
        state.SkipWithMessage("page_size > fifo_size");
        return;
    }

    uint32_t num_iterations = static_cast<uint32_t>(total_data / page_size);

    double avg_per_page_us = 0.0;
    double avg_per_page_cycles = 0.0;
    for ([[maybe_unused]] auto _ : state) {
        auto [us, cycles] =
            run_d2h_throughput(mesh_device, fifo_size, page_size, page_size, num_iterations, sender_core);
        avg_per_page_us = us;
        avg_per_page_cycles = cycles;
    }

    double throughput_gbps = static_cast<double>(page_size) / (avg_per_page_us * 1e3);
    state.counters["data_size"] = static_cast<double>(page_size);  // data_size == page_size (1 page/iter)
    state.counters["pages_per_iter"] = 1;
    state.counters["num_iterations"] = num_iterations;
    state.counters["total_pages"] = num_iterations;
    state.counters["avg_per_page_us"] = avg_per_page_us;
    state.counters["avg_per_page_cycles"] = avg_per_page_cycles;
    state.counters["throughput_gbps"] = throughput_gbps;
    // device_coord goes in the label column of the CSV.
    std::ostringstream coord_ss;
    coord_ss << sender_core.device_coord;
    state.SetLabel(coord_ss.str());

    // SetBytesProcessed drives the bytes_per_second column (cross-check for throughput_gbps).
    state.SetBytesProcessed(static_cast<int64_t>(total_data) * state.iterations());
}

}  // namespace

}  // namespace tt::tt_metal::distributed

using namespace tt::tt_metal::distributed;

BENCHMARK(BM_D2HSocketThroughput)
    ->ArgsProduct({
        {32768, 65536, 131072, 262144},           // page_size
        {128LL << 20, 256LL << 20, 512LL << 20},  // fifo_size
        {512LL << 20, 1024LL << 20},              // total_data
    })
    ->ArgNames({"page_size", "fifo_size", "total_data"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);
