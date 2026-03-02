// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hd_socket_test_utils.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <enchantum/enchantum.hpp>
#include <fstream>
#include <numeric>
#include <sstream>

namespace tt::tt_metal::distributed {

namespace {

struct TargetChip {
    uint32_t tray_id;
    uint32_t asic_location;
};

// Gen 4 ×8 PCIe link (16 GB/s peak)
// Chips with asic location 6 are high-bandwidth.
constexpr TargetChip kHighBwChip = {1, 6};

// Gen 4 x1 PCIe link (2 GB/s peak)
// Chips that are not ASIC 6 are low-bandwidth.
constexpr TargetChip kLowBwChip = {1, 1};

constexpr uint32_t kWarmupIters = 5;
constexpr uint32_t kLatencyIters = 100;

// Sweep parameters for single-chip throughput benchmarks.
const std::vector<int64_t> kThroughputPageSizes = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
const std::vector<int64_t> kThroughputTotalData = {1LL << 20, 16LL << 20, 512LL << 20, 1024LL << 20};

// D2H FIFOs live in host memory (hugepages), so they can be large.
const std::vector<int64_t> kD2HThroughputFifoSizes = {
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    128LL << 10,
    256LL << 10,
    512LL << 10,
    1LL << 20,
    128LL << 20,
    256LL << 20,
    512LL << 20,
};

// H2D FIFOs live in L1, so they are capped at 1MB.
const std::vector<int64_t> kH2DThroughputFifoSizes = {
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    128LL << 10,
    256LL << 10,
    512LL << 10,
    1LL << 20,
};

// Sweep parameters for single-chip latency benchmarks.
const std::vector<int64_t> kLatencyPageSizes = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};

// D2H latency FIFOs can be large (host memory).
const std::vector<int64_t> kD2HLatencyFifoSizes = {1024, 4096, 16384, 65536, 262144, 524288, 512LL << 20};

// H2D latency FIFOs are capped at 1MB (L1).
const std::vector<int64_t> kH2DLatencyFifoSizes = {1024, 4096, 16384, 65536, 262144, 524288, 1LL << 20};

// Shared sweep parameters for ping benchmarks (D2H and H2D).
const std::vector<int64_t> kPingPageSizes = {64};
const std::vector<int64_t> kPingFifoSizes = {4096};

// Shared sweep parameters for D2H multi-chip throughput benchmarks.
const std::vector<int64_t> kMultiChipD2HFifoSizes = {1LL << 20, 4LL << 20, 16LL << 20, 64LL << 20, 256LL << 20};
const std::vector<int64_t> kMultiChipH2DFifoSizes = {1LL << 18, 1LL << 19, 1LL << 20};
const std::vector<int64_t> kMultiChipPageSizes = {262144};
const std::vector<int64_t> kMultiChipTotalData = {1LL << 30};

struct DeviceFixture {
    std::shared_ptr<MeshDevice> mesh_device;

    DeviceFixture() :
        mesh_device(MeshDevice::create(
            MeshDeviceConfig(std::nullopt),
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            /*num_command_queues=*/1,
            DispatchCoreType::WORKER)) {}
};

DeviceFixture& get_device_fixture() {
    static DeviceFixture fixture;
    return fixture;
}

std::optional<MeshCoordinate> find_target_mmio_coord(
    const std::shared_ptr<MeshDevice>& mesh_device, uint32_t tray_id, uint32_t asic_location) {
    auto phys_desc = make_physical_system_descriptor();
    const auto& cp = MetalContext::instance().get_control_plane();
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        if (!is_device_coord_mmio_mapped(mesh_device, coord)) {
            continue;
        }
        auto asic_id = cp.get_asic_id_from_fabric_node_id(mesh_device->get_fabric_node_id(coord));
        const auto& desc = phys_desc.get_asic_descriptors()[asic_id];
        if (*desc.tray_id == tray_id && *desc.asic_location == asic_location) {
            return coord;
        }
    }
    return std::nullopt;
}

std::optional<MeshCoreCoord> find_benchmark_core(
    const std::shared_ptr<MeshDevice>& mesh_device, uint32_t tray_id, uint32_t asic_location) {
    auto coord = find_target_mmio_coord(mesh_device, tray_id, asic_location);
    if (!coord.has_value()) {
        return std::nullopt;
    }
    return MeshCoreCoord{coord.value(), CoreCoord(0, 0)};
}

struct ChipInfo {
    MeshCoordinate coord;
    uint32_t tray_id;
    uint32_t asic_location;
};

std::vector<ChipInfo> enumerate_mmio_chips(const std::shared_ptr<MeshDevice>& mesh_device) {
    auto phys_desc = make_physical_system_descriptor();
    const auto& cp = MetalContext::instance().get_control_plane();
    std::vector<ChipInfo> chips;
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        if (!is_device_coord_mmio_mapped(mesh_device, coord)) {
            continue;
        }
        auto asic_id = cp.get_asic_id_from_fabric_node_id(mesh_device->get_fabric_node_id(coord));
        const auto& desc = phys_desc.get_asic_descriptors()[asic_id];
        chips.push_back({coord, *desc.tray_id, *desc.asic_location});
    }
    return chips;
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

LatencySummary summarize_latency_cycles(const std::vector<uint64_t>& cycles, double cycles_per_us) {
    TT_FATAL(!cycles.empty(), "Expected non-empty cycle measurements");

    auto sorted = cycles;
    std::sort(sorted.begin(), sorted.end());
    double avg_c = 0.0;
    for (auto c : cycles) {
        avg_c += static_cast<double>(c);
    }
    avg_c /= static_cast<double>(cycles.size());

    auto to_us = [&](double c) { return c / cycles_per_us; };
    return {
        .avg_us = to_us(avg_c),
        .min_us = to_us(static_cast<double>(sorted.front())),
        .max_us = to_us(static_cast<double>(sorted.back())),
        .p50_us = to_us(static_cast<double>(sorted[sorted.size() / 2])),
        .p99_us = to_us(static_cast<double>(sorted[(sorted.size() * 99) / 100])),
        .avg_cycles = avg_c,
        .min_cycles = sorted.front(),
        .max_cycles = sorted.back(),
    };
}

LatencySummary summarize_latency_us(const std::vector<double>& us_values, double cycles_per_us) {
    TT_FATAL(!us_values.empty(), "Expected non-empty latency measurements");

    auto sorted = us_values;
    std::sort(sorted.begin(), sorted.end());
    double avg_us = 0.0;
    for (auto v : us_values) {
        avg_us += v;
    }
    avg_us /= static_cast<double>(us_values.size());

    return {
        .avg_us = avg_us,
        .min_us = sorted.front(),
        .max_us = sorted.back(),
        .p50_us = sorted[sorted.size() / 2],
        .p99_us = sorted[(sorted.size() * 99) / 100],
        .avg_cycles = avg_us * cycles_per_us,
        .min_cycles = static_cast<uint64_t>(sorted.front() * cycles_per_us),
        .max_cycles = static_cast<uint64_t>(sorted.back() * cycles_per_us),
    };
}

// Returns {per_page_us, per_page_cycles}.
std::pair<double, double> run_d2h_throughput(
    const std::shared_ptr<MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core) {
    auto output_socket = D2HSocket(mesh_device, sender_core, socket_fifo_size);
    output_socket.set_page_size(page_size);

    auto sender_data_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, page_size);
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
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(num_iterations),
            }});

    std::vector<uint32_t> src_vec(page_size / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(page_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);
    WriteShard(mesh_device->mesh_command_queue(), sender_data_buffer, src_vec, sender_core.device_coord, true);
    execute_program_on_device(*mesh_device, sender_core.device_coord, std::move(send_program));

    for (uint32_t i = 0; i < num_iterations; i++) {
        output_socket.read(dst_vec.data(), 1);
    }
    output_socket.barrier();
    Finish(mesh_device->mesh_command_queue());

    uint64_t total_cycles = read_l1_uint64(*mesh_device, sender_core, measurement_buffer->address());
    double per_page_cycles = static_cast<double>(total_cycles) / num_iterations;
    double per_page_us = per_page_cycles / get_cycles_per_us(*mesh_device);
    return {per_page_us, per_page_cycles};
}

std::vector<uint64_t> run_d2h_latency_cycles(
    const std::shared_ptr<MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core) {
    auto output_socket = D2HSocket(mesh_device, sender_core, socket_fifo_size);
    output_socket.set_page_size(page_size);

    auto sender_data_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, page_size);
    std::size_t measurement_buffer_size = num_iterations * sizeof(uint64_t);
    auto measurement_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, measurement_buffer_size);

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/d2h_socket_loopback_latency.cpp",
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

    execute_program_on_device(*mesh_device, sender_core.device_coord, std::move(send_program));

    std::size_t page_size_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> dst_vec(page_size_words);
    for (uint32_t w = 0; w < kWarmupIters; w++) {
        output_socket.read(dst_vec.data(), 1);
    }
    for (uint32_t i = 0; i < num_iterations; i++) {
        output_socket.read(dst_vec.data(), 1);
    }
    output_socket.barrier();
    Finish(mesh_device->mesh_command_queue());

    std::vector<uint64_t> cycles(num_iterations);
    read_l1_uint64s(*mesh_device, sender_core, measurement_buffer->address(), cycles);
    return cycles;
}

std::vector<uint64_t> run_d2h_ping_cycles(
    const std::shared_ptr<MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core) {
    auto output_socket = D2HSocket(mesh_device, sender_core, socket_fifo_size);
    output_socket.set_page_size(page_size);

    auto sender_data_buffer = make_l1_mesh_buffer(mesh_device.get(), sender_core.core_coord, page_size);
    std::size_t measurement_buffer_size = num_iterations * sizeof(uint64_t);
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

    execute_program_on_device(*mesh_device, sender_core.device_coord, std::move(send_program));

    std::size_t page_size_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> dst_vec(page_size_words);
    for (uint32_t w = 0; w < kWarmupIters; w++) {
        output_socket.read(dst_vec.data(), 1);
    }
    for (uint32_t i = 0; i < num_iterations; i++) {
        output_socket.read(dst_vec.data(), 1);
    }
    output_socket.barrier();
    Finish(mesh_device->mesh_command_queue());

    std::vector<uint64_t> cycles(num_iterations);
    read_l1_uint64s(*mesh_device, sender_core, measurement_buffer->address(), cycles);
    return cycles;
}

// Returns {per_page_us, per_page_cycles} via host-side wall-clock.
std::pair<double, double> run_h2d_throughput(
    const std::shared_ptr<MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    H2DMode h2d_mode,
    const MeshCoreCoord& recv_core) {
    auto input_socket = H2DSocket(mesh_device, recv_core, BufferType::L1, socket_fifo_size, h2d_mode);
    input_socket.set_page_size(page_size);

    auto recv_data_buffer = make_l1_mesh_buffer(mesh_device.get(), recv_core.core_coord, page_size);
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
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(num_iterations),
            }});

    std::vector<uint32_t> src_vec(page_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    execute_program_on_device(*mesh_device, recv_core.device_coord, std::move(recv_program));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++) {
        input_socket.write(src_vec.data(), 1);
    }
    input_socket.barrier();
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_page_us = elapsed_us / num_iterations;
    double per_page_cycles = per_page_us * get_cycles_per_us(*mesh_device);
    return {per_page_us, per_page_cycles};
}

// Per-iteration host-side latencies in microseconds.
std::vector<double> run_h2d_latency_us(
    const std::shared_ptr<MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    H2DMode h2d_mode,
    const MeshCoreCoord& recv_core) {
    auto input_socket = H2DSocket(mesh_device, recv_core, BufferType::L1, socket_fifo_size, h2d_mode);
    input_socket.set_page_size(page_size);

    auto recv_data_buffer = make_l1_mesh_buffer(mesh_device.get(), recv_core.core_coord, page_size);
    std::size_t measurement_buffer_size = num_iterations * sizeof(uint64_t);
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

    execute_program_on_device(*mesh_device, recv_core.device_coord, std::move(recv_program));

    std::size_t page_size_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> src_vec(page_size_words);

    for (uint32_t w = 0; w < kWarmupIters; w++) {
        input_socket.write(src_vec.data(), 1);
        input_socket.barrier();
    }

    std::vector<double> latencies_us(num_iterations);
    for (uint32_t i = 0; i < num_iterations; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        input_socket.write(src_vec.data(), 1);
        input_socket.barrier();
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    return latencies_us;
}

// H2D pure ping (no data DMA). Returns per-iteration host-side latencies in microseconds.
std::vector<double> run_h2d_ping_us(
    const std::shared_ptr<MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    H2DMode h2d_mode,
    const MeshCoreCoord& recv_core) {
    auto input_socket = H2DSocket(mesh_device, recv_core, BufferType::L1, socket_fifo_size, h2d_mode);
    input_socket.set_page_size(page_size);

    std::size_t measurement_buffer_size = num_iterations * sizeof(uint64_t);
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

    execute_program_on_device(*mesh_device, recv_core.device_coord, std::move(recv_program));

    std::size_t page_size_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> src_vec(page_size_words);
    for (uint32_t w = 0; w < kWarmupIters; w++) {
        input_socket.write(src_vec.data(), 1);
        input_socket.barrier();
    }
    std::vector<double> latencies_us(num_iterations);
    for (uint32_t i = 0; i < num_iterations; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        input_socket.write(src_vec.data(), 1);
        input_socket.barrier();
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    return latencies_us;
}

void set_latency_counters(benchmark::State& state, const LatencySummary& s, uint32_t num_iterations) {
    state.counters["num_iterations"] = num_iterations;
    state.counters["avg_us"] = s.avg_us;
    state.counters["min_us"] = s.min_us;
    state.counters["max_us"] = s.max_us;
    state.counters["p50_us"] = s.p50_us;
    state.counters["p99_us"] = s.p99_us;
    state.counters["avg_cycles"] = s.avg_cycles;
    state.counters["min_cycles"] = static_cast<double>(s.min_cycles);
    state.counters["max_cycles"] = static_cast<double>(s.max_cycles);
}

// Pre-register counter names so the CSV header always includes them,
// even when the first reported benchmark is skipped.
void init_throughput_counters(benchmark::State& state) {
    state.counters["data_size"] = 0;
    state.counters["num_iterations"] = 0;
    state.counters["per_page_us"] = 0;
    state.counters["per_page_cycles"] = 0;
    state.counters["throughput_gbps"] = 0;
}

void init_latency_counters(benchmark::State& state) {
    state.counters["num_iterations"] = 0;
    state.counters["avg_us"] = 0;
    state.counters["min_us"] = 0;
    state.counters["max_us"] = 0;
    state.counters["p50_us"] = 0;
    state.counters["p99_us"] = 0;
    state.counters["avg_cycles"] = 0;
    state.counters["min_cycles"] = 0;
    state.counters["max_cycles"] = 0;
}

void init_multichip_counters(benchmark::State& state) {
    state.counters["tray_id"] = 0;
    state.counters["asic_location"] = 0;
    state.counters["data_size"] = 0;
    state.counters["num_iterations"] = 0;
    state.counters["total_data"] = 0;
    state.counters["per_page_us"] = 0;
    state.counters["per_page_cycles"] = 0;
    state.counters["throughput_gbps"] = 0;
}

// Returns true if the benchmark should proceed. Skips and returns false on failure.
bool check_preconditions(
    benchmark::State& state, MeshDevice& mesh_device, std::size_t page_size, std::size_t fifo_size) {
    if (!experimental::GetMemoryPinningParameters(mesh_device).can_map_to_noc) {
        state.SkipWithMessage("Mapping host memory to NOC is not supported on this system");
        return false;
    }
    if (page_size > fifo_size) {
        state.SkipWithMessage("page_size > fifo_size");
        return false;
    }
    return true;
}

void BM_D2HSocketThroughput(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    init_throughput_counters(state);
    auto& fx = get_device_fixture();

    auto tray_id = static_cast<uint32_t>(state.range(0));
    auto asic_location = static_cast<uint32_t>(state.range(1));
    auto page_size = static_cast<std::size_t>(state.range(2));
    auto fifo_size = static_cast<std::size_t>(state.range(3));
    auto total_data = static_cast<std::size_t>(state.range(4));

    if (!check_preconditions(state, *fx.mesh_device, page_size, fifo_size)) {
        return;
    }
    auto target_core = find_benchmark_core(fx.mesh_device, tray_id, asic_location);
    if (!target_core) {
        state.SkipWithMessage("Target chip not found");
        return;
    }

    uint32_t num_iterations = static_cast<uint32_t>(total_data / page_size);

    for (auto _ : state) {
        auto [per_page_us, per_page_cycles] =
            run_d2h_throughput(fx.mesh_device, fifo_size, page_size, num_iterations, *target_core);

        double throughput_gbps = static_cast<double>(page_size) / (per_page_us * 1e3);
        state.counters["data_size"] = static_cast<double>(page_size);
        state.counters["num_iterations"] = num_iterations;
        state.counters["per_page_us"] = per_page_us;
        state.counters["per_page_cycles"] = per_page_cycles;
        state.counters["throughput_gbps"] = throughput_gbps;

        std::ostringstream coord_ss;
        coord_ss << target_core->device_coord;
        state.SetLabel(coord_ss.str());
    }
}

void BM_D2HSocketLatency(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    init_latency_counters(state);
    auto& fx = get_device_fixture();

    auto tray_id = static_cast<uint32_t>(state.range(0));
    auto asic_location = static_cast<uint32_t>(state.range(1));
    auto page_size = static_cast<std::size_t>(state.range(2));
    auto fifo_size = static_cast<std::size_t>(state.range(3));

    if (!check_preconditions(state, *fx.mesh_device, page_size, fifo_size)) {
        return;
    }
    auto target_core = find_benchmark_core(fx.mesh_device, tray_id, asic_location);
    if (!target_core) {
        state.SkipWithMessage("Target chip not found");
        return;
    }

    for (auto _ : state) {
        auto cycles = run_d2h_latency_cycles(fx.mesh_device, fifo_size, page_size, kLatencyIters, *target_core);
        auto stats = summarize_latency_cycles(cycles, get_cycles_per_us(*fx.mesh_device));

        set_latency_counters(state, stats, kLatencyIters);

        std::ostringstream coord_ss;
        coord_ss << target_core->device_coord;
        state.SetLabel(coord_ss.str());
    }
}

void BM_D2HSocketPing(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    init_latency_counters(state);
    auto& fx = get_device_fixture();

    auto tray_id = static_cast<uint32_t>(state.range(0));
    auto asic_location = static_cast<uint32_t>(state.range(1));
    auto page_size = static_cast<std::size_t>(state.range(2));
    auto fifo_size = static_cast<std::size_t>(state.range(3));

    if (!check_preconditions(state, *fx.mesh_device, page_size, fifo_size)) {
        return;
    }
    auto target_core = find_benchmark_core(fx.mesh_device, tray_id, asic_location);
    if (!target_core) {
        state.SkipWithMessage("Target chip not found");
        return;
    }

    for (auto _ : state) {
        auto cycles = run_d2h_ping_cycles(fx.mesh_device, fifo_size, page_size, kLatencyIters, *target_core);

        double cycles_per_us = get_cycles_per_us(*fx.mesh_device);
        auto stats = summarize_latency_cycles(cycles, cycles_per_us);
        set_latency_counters(state, stats, kLatencyIters);

        std::ostringstream coord_ss;
        coord_ss << target_core->device_coord;
        state.SetLabel(coord_ss.str());

        std::string csv_path =
            std::string("d2h_ping_iterations_") + std::to_string(page_size) + "_" + std::to_string(fifo_size) + ".csv";
        std::ofstream f(csv_path);
        if (f.is_open()) {
            f << "iteration,latency_us,cycles\n";
            for (uint32_t i = 0; i < kLatencyIters; i++) {
                f << i << "," << (static_cast<double>(cycles[i]) / cycles_per_us) << "," << cycles[i] << "\n";
            }
        }
    }
}

void BM_D2HSocketMultiChipThroughput(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    init_multichip_counters(state);
    auto& fx = get_device_fixture();

    auto chip_index = static_cast<std::size_t>(state.range(0));
    auto fifo_size = static_cast<std::size_t>(state.range(1));
    auto page_size = static_cast<std::size_t>(state.range(2));
    auto total_data = static_cast<std::size_t>(state.range(3));

    if (!check_preconditions(state, *fx.mesh_device, page_size, fifo_size)) {
        return;
    }

    static auto mmio_chips = enumerate_mmio_chips(fx.mesh_device);
    if (chip_index >= mmio_chips.size()) {
        state.SkipWithMessage("chip_index out of range");
        return;
    }

    uint32_t num_iterations = static_cast<uint32_t>(total_data / page_size);
    const auto& chip = mmio_chips[chip_index];
    MeshCoreCoord sender_core{chip.coord, CoreCoord(0, 0)};

    for (auto _ : state) {
        auto [per_page_us, per_page_cycles] =
            run_d2h_throughput(fx.mesh_device, fifo_size, page_size, num_iterations, sender_core);

        double throughput_gbps = static_cast<double>(page_size) / (per_page_us * 1e3);
        state.counters["tray_id"] = chip.tray_id;
        state.counters["asic_location"] = chip.asic_location;
        state.counters["data_size"] = static_cast<double>(page_size);
        state.counters["num_iterations"] = num_iterations;
        state.counters["total_data"] = static_cast<double>(total_data);
        state.counters["per_page_us"] = per_page_us;
        state.counters["per_page_cycles"] = per_page_cycles;
        state.counters["throughput_gbps"] = throughput_gbps;

        std::ostringstream coord_ss;
        coord_ss << chip.coord;
        state.SetLabel(coord_ss.str());
    }
}

void BM_H2DSocketThroughput(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    init_throughput_counters(state);
    state.counters["h2d_mode"] = 0;
    auto& fx = get_device_fixture();

    auto tray_id = static_cast<uint32_t>(state.range(0));
    auto asic_location = static_cast<uint32_t>(state.range(1));
    auto page_size = static_cast<std::size_t>(state.range(2));
    auto fifo_size = static_cast<std::size_t>(state.range(3));
    auto total_data = static_cast<std::size_t>(state.range(4));
    auto mode_index = static_cast<int>(state.range(5));
    auto h2d_mode = static_cast<H2DMode>(mode_index);

    if (!check_preconditions(state, *fx.mesh_device, page_size, fifo_size)) {
        return;
    }
    auto target_core = find_benchmark_core(fx.mesh_device, tray_id, asic_location);
    if (!target_core) {
        state.SkipWithMessage("Target chip not found");
        return;
    }

    uint32_t num_iterations = static_cast<uint32_t>(total_data / page_size);

    for (auto _ : state) {
        auto [per_page_us, per_page_cycles] =
            run_h2d_throughput(fx.mesh_device, fifo_size, page_size, num_iterations, h2d_mode, *target_core);

        double throughput_gbps = static_cast<double>(page_size) / (per_page_us * 1e3);
        state.counters["h2d_mode"] = mode_index;
        state.counters["data_size"] = static_cast<double>(page_size);
        state.counters["num_iterations"] = num_iterations;
        state.counters["per_page_us"] = per_page_us;
        state.counters["per_page_cycles"] = per_page_cycles;
        state.counters["throughput_gbps"] = throughput_gbps;

        std::ostringstream label_ss;
        label_ss << enchantum::to_string(h2d_mode) << " " << target_core->device_coord;
        state.SetLabel(label_ss.str());
    }
}

void BM_H2DSocketLatency(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    init_latency_counters(state);
    state.counters["h2d_mode"] = 0;
    auto& fx = get_device_fixture();

    auto tray_id = static_cast<uint32_t>(state.range(0));
    auto asic_location = static_cast<uint32_t>(state.range(1));
    auto page_size = static_cast<std::size_t>(state.range(2));
    auto fifo_size = static_cast<std::size_t>(state.range(3));
    auto mode_index = static_cast<int>(state.range(4));
    auto h2d_mode = static_cast<H2DMode>(mode_index);

    if (!check_preconditions(state, *fx.mesh_device, page_size, fifo_size)) {
        return;
    }
    auto target_core = find_benchmark_core(fx.mesh_device, tray_id, asic_location);
    if (!target_core) {
        state.SkipWithMessage("Target chip not found");
        return;
    }

    for (auto _ : state) {
        auto latencies_us =
            run_h2d_latency_us(fx.mesh_device, fifo_size, page_size, kLatencyIters, h2d_mode, *target_core);
        auto stats = summarize_latency_us(latencies_us, get_cycles_per_us(*fx.mesh_device));

        state.counters["h2d_mode"] = mode_index;
        set_latency_counters(state, stats, kLatencyIters);

        std::ostringstream label_ss;
        label_ss << enchantum::to_string(h2d_mode) << " " << target_core->device_coord;
        state.SetLabel(label_ss.str());
    }
}

void BM_H2DSocketPing(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    init_latency_counters(state);
    state.counters["h2d_mode"] = 0;
    auto& fx = get_device_fixture();

    auto tray_id = static_cast<uint32_t>(state.range(0));
    auto asic_location = static_cast<uint32_t>(state.range(1));
    auto page_size = static_cast<std::size_t>(state.range(2));
    auto fifo_size = static_cast<std::size_t>(state.range(3));
    auto mode_index = static_cast<int>(state.range(4));
    auto h2d_mode = static_cast<H2DMode>(mode_index);

    if (!check_preconditions(state, *fx.mesh_device, page_size, fifo_size)) {
        return;
    }
    auto target_core = find_benchmark_core(fx.mesh_device, tray_id, asic_location);
    if (!target_core) {
        state.SkipWithMessage("Target chip not found");
        return;
    }

    for (auto _ : state) {
        auto latencies_us =
            run_h2d_ping_us(fx.mesh_device, fifo_size, page_size, kLatencyIters, h2d_mode, *target_core);
        double cycles_per_us = get_cycles_per_us(*fx.mesh_device);
        auto stats = summarize_latency_us(latencies_us, cycles_per_us);
        state.counters["h2d_mode"] = mode_index;
        set_latency_counters(state, stats, kLatencyIters);

        std::ostringstream label_ss;
        label_ss << enchantum::to_string(h2d_mode) << " " << target_core->device_coord;
        state.SetLabel(label_ss.str());

        std::string csv_path =
            std::string("h2d_ping_iterations_") + std::string(enchantum::to_string(h2d_mode)) + ".csv";
        std::ofstream f(csv_path);
        if (f.is_open()) {
            f << "iteration,latency_us,cycles\n";
            for (uint32_t i = 0; i < kLatencyIters; i++) {
                f << i << "," << latencies_us[i] << "," << (latencies_us[i] * cycles_per_us) << "\n";
            }
        }
    }
}

void BM_H2DSocketMultiChipThroughput(benchmark::State& state) {
    TT_FATAL(state.max_iterations == 1, "Only single-iteration benchmarks are supported");
    init_multichip_counters(state);
    auto& fx = get_device_fixture();

    auto chip_index = static_cast<std::size_t>(state.range(0));
    auto fifo_size = static_cast<std::size_t>(state.range(1));
    auto page_size = static_cast<std::size_t>(state.range(2));
    auto total_data = static_cast<std::size_t>(state.range(3));

    if (!check_preconditions(state, *fx.mesh_device, page_size, fifo_size)) {
        return;
    }

    static auto mmio_chips = enumerate_mmio_chips(fx.mesh_device);
    if (chip_index >= mmio_chips.size()) {
        state.SkipWithMessage("chip_index out of range");
        return;
    }

    uint32_t num_iterations = static_cast<uint32_t>(total_data / page_size);
    const auto& chip = mmio_chips[chip_index];
    MeshCoreCoord recv_core{chip.coord, CoreCoord(0, 0)};

    for (auto _ : state) {
        auto [per_page_us, per_page_cycles] =
            run_h2d_throughput(fx.mesh_device, fifo_size, page_size, num_iterations, H2DMode::DEVICE_PULL, recv_core);

        double throughput_gbps = static_cast<double>(page_size) / (per_page_us * 1e3);
        state.counters["tray_id"] = chip.tray_id;
        state.counters["asic_location"] = chip.asic_location;
        state.counters["data_size"] = static_cast<double>(page_size);
        state.counters["num_iterations"] = num_iterations;
        state.counters["total_data"] = static_cast<double>(total_data);
        state.counters["per_page_us"] = per_page_us;
        state.counters["per_page_cycles"] = per_page_cycles;
        state.counters["throughput_gbps"] = throughput_gbps;

        std::ostringstream coord_ss;
        coord_ss << chip.coord;
        state.SetLabel(coord_ss.str());
    }
}

}  // namespace

}  // namespace tt::tt_metal::distributed

using namespace tt::tt_metal::distributed;

// ── High-bandwidth chip ──────────────────────────────────────────

BENCHMARK(BM_D2HSocketThroughput)
    ->ArgsProduct({
        {kHighBwChip.tray_id},        // tray_id
        {kHighBwChip.asic_location},  // asic_location
        kThroughputPageSizes,         // page_size
        kD2HThroughputFifoSizes,      // fifo_size
        kThroughputTotalData,         // total_data
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size", "total_data"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_D2HSocketLatency)
    ->ArgsProduct({
        {kHighBwChip.tray_id},        // tray_id
        {kHighBwChip.asic_location},  // asic_location
        kLatencyPageSizes,            // page_size
        kD2HLatencyFifoSizes,         // fifo_size
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_D2HSocketPing)
    ->ArgsProduct({
        {kHighBwChip.tray_id},        // tray_id
        {kHighBwChip.asic_location},  // asic_location
        kPingPageSizes,               // page_size
        kPingFifoSizes,               // fifo_size
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_H2DSocketThroughput)
    ->ArgsProduct({
        {kHighBwChip.tray_id},                                                                   // tray_id
        {kHighBwChip.asic_location},                                                             // asic_location
        kThroughputPageSizes,                                                                    // page_size
        kH2DThroughputFifoSizes,                                                                 // fifo_size
        kThroughputTotalData,                                                                    // total_data
        {static_cast<int64_t>(H2DMode::HOST_PUSH), static_cast<int64_t>(H2DMode::DEVICE_PULL)},  // mode_index
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size", "total_data", "mode_index"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_H2DSocketLatency)
    ->ArgsProduct({
        {kHighBwChip.tray_id},                                                                   // tray_id
        {kHighBwChip.asic_location},                                                             // asic_location
        kLatencyPageSizes,                                                                       // page_size
        kH2DLatencyFifoSizes,                                                                    // fifo_size
        {static_cast<int64_t>(H2DMode::HOST_PUSH), static_cast<int64_t>(H2DMode::DEVICE_PULL)},  // mode_index
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size", "mode_index"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_H2DSocketPing)
    ->ArgsProduct({
        {kHighBwChip.tray_id},                                                                   // tray_id
        {kHighBwChip.asic_location},                                                             // asic_location
        kPingPageSizes,                                                                          // page_size
        kPingFifoSizes,                                                                          // fifo_size
        {static_cast<int64_t>(H2DMode::HOST_PUSH), static_cast<int64_t>(H2DMode::DEVICE_PULL)},  // mode_index
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size", "mode_index"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

// ── Low-bandwidth chip ───────────────────────────────────

BENCHMARK(BM_D2HSocketThroughput)
    ->ArgsProduct({
        {kLowBwChip.tray_id},        // tray_id
        {kLowBwChip.asic_location},  // asic_location
        kThroughputPageSizes,        // page_size
        kD2HThroughputFifoSizes,     // fifo_size
        kThroughputTotalData,        // total_data
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size", "total_data"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_D2HSocketLatency)
    ->ArgsProduct({
        {kLowBwChip.tray_id},        // tray_id
        {kLowBwChip.asic_location},  // asic_location
        kLatencyPageSizes,           // page_size
        kD2HLatencyFifoSizes,        // fifo_size
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_D2HSocketPing)
    ->ArgsProduct({
        {kLowBwChip.tray_id},        // tray_id
        {kLowBwChip.asic_location},  // asic_location
        kPingPageSizes,              // page_size
        kPingFifoSizes,              // fifo_size
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_H2DSocketThroughput)
    ->ArgsProduct({
        {kLowBwChip.tray_id},                                                                    // tray_id
        {kLowBwChip.asic_location},                                                              // asic_location
        kThroughputPageSizes,                                                                    // page_size
        kH2DThroughputFifoSizes,                                                                 // fifo_size
        kThroughputTotalData,                                                                    // total_data
        {static_cast<int64_t>(H2DMode::HOST_PUSH), static_cast<int64_t>(H2DMode::DEVICE_PULL)},  // mode_index
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size", "total_data", "mode_index"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_H2DSocketLatency)
    ->ArgsProduct({
        {kLowBwChip.tray_id},                                                                    // tray_id
        {kLowBwChip.asic_location},                                                              // asic_location
        kLatencyPageSizes,                                                                       // page_size
        kH2DLatencyFifoSizes,                                                                    // fifo_size
        {static_cast<int64_t>(H2DMode::HOST_PUSH), static_cast<int64_t>(H2DMode::DEVICE_PULL)},  // mode_index
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size", "mode_index"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_H2DSocketPing)
    ->ArgsProduct({
        {kLowBwChip.tray_id},                                                                    // tray_id
        {kLowBwChip.asic_location},                                                              // asic_location
        kPingPageSizes,                                                                          // page_size
        kPingFifoSizes,                                                                          // fifo_size
        {static_cast<int64_t>(H2DMode::HOST_PUSH), static_cast<int64_t>(H2DMode::DEVICE_PULL)},  // mode_index
    })
    ->ArgNames({"tray_id", "asic_location", "page_size", "fifo_size", "mode_index"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

// ── Multi-chip sweep (all 32 chips) ─────────────────────────────────────────

BENCHMARK(BM_D2HSocketMultiChipThroughput)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, 31, 1),  // chip_index
        kMultiChipD2HFifoSizes,                 // fifo_size
        kMultiChipPageSizes,                    // page_size
        kMultiChipTotalData,                    // total_data
    })
    ->ArgNames({"chip_index", "fifo_size", "page_size", "total_data"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

BENCHMARK(BM_H2DSocketMultiChipThroughput)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, 31, 1),  // chip_index
        kMultiChipH2DFifoSizes,                 // fifo_size
        kMultiChipPageSizes,                    // page_size
        kMultiChipTotalData,                    // total_data
    })
    ->ArgNames({"chip_index", "fifo_size", "page_size", "total_data"})
    ->UseRealTime()
    ->Iterations(1)
    ->Unit(benchmark::kSecond);
