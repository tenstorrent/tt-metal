// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hd_socket_test_utils.hpp"

#include <chrono>
#include <numeric>

namespace tt::tt_metal::distributed {

namespace {

// Single-chip benchmarks target Tray 1, ASIC Location 6: the high-bandwidth chip on that tray,
// which has a PCIe Gen 4 ×8 link to the host root complex (~16 GB/s theoretical peak).
// All other ASICs on the tray have significantly lower host-facing bandwidth (see §1.1 of
// tech_reports/TT-Distributed/HDSocketsModel.md).  Any of the 4 high-bandwidth chips
// (one per tray, always ASIC Location 6) would give equivalent results; Tray 1 is used as a
// fixed reference for reproducibility.
constexpr uint32_t kTargetTrayId = 1;
constexpr uint32_t kTargetAsicLocation = 6;
constexpr uint32_t kWarmupIters = 5;

const std::vector<std::size_t> kTotalDataSizes = {
    512UL * 1024 * 1024,  // 512MB
    1024UL * 1024 * 1024  // 1GB
};

const std::vector<std::size_t> kPageSizes = {32768, 65536, 131072, 262144};

const std::vector<std::size_t> kD2HThroughputFifoSizes = {
    128 * 1024 * 1024,
    256 * 1024 * 1024,
    512 * 1024 * 1024,
};

const std::vector<std::size_t> kD2HLatencyFifoSizes = {1024, 4096, 16384, 65536, 512UL * 1024 * 1024};
const std::vector<std::size_t> kH2DThroughputFifoSizes = {
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024};
const std::vector<std::size_t> kH2DLatencyFifoSizes = {1024, 4096, 16384, 65536, 262144, 524288};

const char* h2d_mode_name(H2DMode mode) { return (mode == H2DMode::HOST_PUSH) ? "HOST_PUSH" : "DEVICE_PULL"; }

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
    const std::shared_ptr<MeshDevice>& mesh_device, uint32_t target_tray_id, uint32_t target_asic_location) {
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

MeshCoreCoord get_target_benchmark_worker_core(const std::shared_ptr<MeshDevice>& mesh_device) {
    auto preferred = find_target_mmio_coord(mesh_device, kTargetTrayId, kTargetAsicLocation);
    if (preferred.has_value()) {
        return MeshCoreCoord{preferred.value(), CoreCoord(0, 0)};
    }
    TT_FATAL(
        false, "Desired chip not found for benchmark (target tray={} asic={})", kTargetTrayId, kTargetAsicLocation);
    __builtin_unreachable();
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
        .avg_us = avg_c / cycles_per_us,
        .min_us = static_cast<double>(min_c) / cycles_per_us,
        .max_us = static_cast<double>(max_c) / cycles_per_us,
        .p50_us = static_cast<double>(p50_c) / cycles_per_us,
        .p99_us = static_cast<double>(p99_c) / cycles_per_us,
        .avg_cycles = avg_c,
        .min_cycles = min_c,
        .max_cycles = max_c,
    };
}

LatencySummary summarize_latency_us(const std::vector<double>& us_values, double cycles_per_us) {
    TT_FATAL(!us_values.empty(), "Expected non-empty latency measurements");

    auto sorted = us_values;
    std::sort(sorted.begin(), sorted.end());
    const double min_us = sorted.front();
    const double max_us = sorted.back();
    const double p50_us = sorted[sorted.size() / 2];
    const double p99_us = sorted[(sorted.size() * 99) / 100];

    double avg_us = 0.0;
    for (auto v : us_values) {
        avg_us += v;
    }
    avg_us /= static_cast<double>(us_values.size());

    return {
        .avg_us = avg_us,
        .min_us = min_us,
        .max_us = max_us,
        .p50_us = p50_us,
        .p99_us = p99_us,
        .avg_cycles = avg_us * cycles_per_us,
        .min_cycles = static_cast<uint64_t>(min_us * cycles_per_us),
        .max_cycles = static_cast<uint64_t>(max_us * cycles_per_us),
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

// Benchmark tests use a distinct fixture alias so that the GTest class names don't collide
// with the identically-named tests in test_hd_sockets.cpp when both TUs are linked together.
using HDSocketBenchFixture = HDSocketFixture;

// Forward declarations
std::pair<double, double> benchmark_d2h_throughput(
    const std::shared_ptr<MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core);

std::pair<double, double> benchmark_h2d_throughput(
    const std::shared_ptr<MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    uint32_t num_iterations,
    H2DMode h2d_mode,
    const MeshCoreCoord& recv_core);

TEST_F(HDSocketBenchFixture, D2HSocketThroughputBenchmark) {
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
            for (auto total_data : kTotalDataSizes) {
                uint32_t num_iterations = total_data / page_size;
                if (num_iterations == 0) {
                    continue;
                }

                auto [us, cycles] = benchmark_d2h_throughput(
                    mesh_device_, fifo_size, page_size, page_size, num_iterations, sender_core);

                emit_d2h_throughput_csv_row(
                    page_size,
                    fifo_size,
                    total_data,
                    page_size,
                    /*pages_per_iter=*/1,
                    num_iterations,
                    /*total_pages=*/num_iterations,
                    us,
                    cycles,
                    sender_coord);
                std::cout.flush();
            }
        }
    }
}

TEST_F(HDSocketBenchFixture, D2HSocketMultiChipMaxThroughputBenchmark) {
    // Iterates over every MMIO-mapped chip on the system and measures D2H throughput
    // at 64KB page size (max-throughput configuration) across a range of FIFO sizes
    // that quadruple each step: 1MB → 4MB → 16MB → 64MB → 256MB.
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    constexpr std::size_t kBenchPageSize = 65536;                  // 64KB – max throughput
    constexpr std::size_t kBenchTotalData = 1024UL * 1024 * 1024;  // 1GB
    const std::vector<std::size_t> kBenchFifoSizes = {
        1UL * 1024 * 1024,    //   1MB
        4UL * 1024 * 1024,    //   4MB
        16UL * 1024 * 1024,   //  16MB
        64UL * 1024 * 1024,   //  64MB
        256UL * 1024 * 1024,  // 256MB
    };

    auto mmio_chips = enumerate_mmio_chips(mesh_device_);

    uint32_t num_iterations = kBenchTotalData / kBenchPageSize;

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
            auto [us, cycles] = benchmark_d2h_throughput(
                mesh_device_, fifo_size, kBenchPageSize, kBenchPageSize, num_iterations, sender_core);

            const double throughput_gbps = static_cast<double>(kBenchPageSize) / (us * 1e3);
            std::cout << chip.tray_id << "," << chip.asic_location << "," << chip.coord << "," << fifo_size << ","
                      << kBenchTotalData << "," << kBenchPageSize << "," << 1 << "," << num_iterations << ","
                      << num_iterations << "," << us << "," << cycles << "," << throughput_gbps << std::endl;
            std::cout.flush();
        }
    }
}

TEST_F(HDSocketBenchFixture, H2DSocketMultiChipMaxThroughputBenchmark) {
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

    uint32_t num_iterations = kBenchTotalData / kBenchPageSize;

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
            auto [us, cycles] = benchmark_h2d_throughput(
                mesh_device_,
                fifo_size,
                kBenchPageSize,
                kBenchPageSize,
                num_iterations,
                H2DMode::DEVICE_PULL,
                recv_core);

            const double throughput_gbps = static_cast<double>(kBenchPageSize) / (us * 1e3);
            std::cout << chip.tray_id << "," << chip.asic_location << "," << chip.coord << "," << fifo_size << ","
                      << kBenchTotalData << "," << kBenchPageSize << "," << 1 << "," << num_iterations << ","
                      << num_iterations << "," << us << "," << cycles << "," << throughput_gbps << std::endl;
            std::cout.flush();
        }
    }
}

// Returns average per-page throughput in microseconds and cycles.
std::pair<double, double> benchmark_d2h_throughput(
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
    double avg_latency_us = avg_cycles_per_transaction / get_cycles_per_us(*mesh_device);

    return {avg_latency_us, avg_cycles_per_transaction};
}

namespace {
// D2H Latency: per-iteration round-trip measurement using d2h_socket_loopback_latency kernel.
// Returns a vector of per-iteration cycle counts.
std::vector<uint64_t> benchmark_d2h_latency(
    const std::shared_ptr<MeshDevice>& mesh_device,
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
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/d2h_socket_loopback_latency.cpp",
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

    execute_program_on_device(*mesh_device, sender_core.device_coord, std::move(send_program));

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
}  // namespace

TEST_F(HDSocketBenchFixture, D2HSocketLatencyBenchmark) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    uint32_t num_iterations = 100;

    MeshCoreCoord sender_core = get_target_benchmark_worker_core(mesh_device_);
    const auto& sender_coord = sender_core.device_coord;
    const double cycles_per_us = get_cycles_per_us(*mesh_device_);

    std::cout << "page_size,socket_fifo_size,num_iterations,"
              << "avg_us,min_us,max_us,p50_us,p99_us,"
              << "avg_cycles,min_cycles,max_cycles,device_coord" << std::endl;

    for (auto fifo_size : kD2HLatencyFifoSizes) {
        for (auto page_size : kPageSizes) {
            if (page_size > fifo_size) {
                continue;
            }

            auto cycles = benchmark_d2h_latency(mesh_device_, fifo_size, page_size, num_iterations, sender_core);

            auto stats = summarize_latency_cycles(cycles, cycles_per_us);
            emit_latency_csv_row(page_size, fifo_size, num_iterations, stats, sender_coord);
            std::cout.flush();
        }
    }
}

TEST_F(HDSocketBenchFixture, H2DSocketThroughputBenchmark) {
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
                for (auto total_data : kTotalDataSizes) {
                    uint32_t num_iterations = total_data / page_size;
                    if (num_iterations == 0) {
                        continue;
                    }

                    auto [us, cycles] = benchmark_h2d_throughput(
                        mesh_device_, fifo_size, page_size, page_size, num_iterations, h2d_mode, recv_core);

                    emit_h2d_throughput_csv_row(
                        page_size,
                        fifo_size,
                        h2d_mode,
                        total_data,
                        page_size,
                        /*pages_per_iter=*/1,
                        num_iterations,
                        /*total_pages=*/num_iterations,
                        us,
                        cycles,
                        recv_coord);
                    std::cout.flush();
                }
            }
        }
    }
}

namespace {
// D2H Ping: pure signaling round-trip, no data DMA.
std::vector<uint64_t> benchmark_d2h_ping(
    const std::shared_ptr<MeshDevice>& mesh_device,
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

    execute_program_on_device(*mesh_device, sender_core.device_coord, std::move(send_program));

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
}  // namespace

TEST_F(HDSocketBenchFixture, D2HSocketPingBenchmark) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    std::vector<std::size_t> page_sizes = {64};
    std::vector<std::size_t> fifo_sizes = {4096};
    uint32_t num_iterations = 100;

    MeshCoreCoord sender_core = get_target_benchmark_worker_core(mesh_device_);
    const auto& sender_coord = sender_core.device_coord;
    const double cycles_per_us = get_cycles_per_us(*mesh_device_);

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
                    double lat_us = static_cast<double>(cycles[i]) / cycles_per_us;
                    iter_file << i << "," << lat_us << "," << cycles[i] << std::endl;
                }
                iter_file.close();
                dumped_iterations = true;
            }

            auto stats = summarize_latency_cycles(cycles, cycles_per_us);
            emit_latency_csv_row(page_size, fifo_size, num_iterations, stats, sender_coord);
            std::cout.flush();
        }
    }
}

namespace {
std::vector<uint64_t> benchmark_h2d_ping(
    const std::shared_ptr<MeshDevice>& mesh_device,
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

    execute_program_on_device(*mesh_device, recv_core.device_coord, std::move(recv_program));

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
}  // namespace

TEST_F(HDSocketBenchFixture, H2DSocketPingBenchmark) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    std::size_t page_size = 64;
    std::size_t fifo_size = 4096;
    uint32_t num_iterations = 100;

    MeshCoreCoord recv_core = get_target_benchmark_worker_core(mesh_device_);
    const auto& recv_coord = recv_core.device_coord;
    const double cycles_per_us = get_cycles_per_us(*mesh_device_);

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
            double lat_us = static_cast<double>(cycles[i]) / cycles_per_us;
            iter_file << i << "," << lat_us << "," << cycles[i] << std::endl;
        }
        iter_file.close();

        auto stats = summarize_latency_cycles(cycles, cycles_per_us);
        emit_latency_csv_row(
            page_size, fifo_size, std::string(h2d_mode_name(h2d_mode)), num_iterations, stats, recv_coord);
        std::cout.flush();
    }
}

std::pair<double, double> benchmark_h2d_throughput(
    const std::shared_ptr<MeshDevice>& mesh_device,
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
    // Kernel still writes a device-side timestamp to this buffer; we don't read it back
    // since we measure throughput from the host side.
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

    execute_program_on_device(*mesh_device, recv_core.device_coord, std::move(recv_program));
    uint32_t page_size_words = page_size / sizeof(uint32_t);

    // Measure host-side: start → all writes issued → socket barrier (device has acked all data).
    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++) {
        for (uint32_t j = 0; j < num_txns; j++) {
            input_socket.write(src_vec.data() + (j * page_size_words), 1);
        }
    }
    input_socket.barrier();
    auto t1 = std::chrono::high_resolution_clock::now();

    uint64_t total_pages = static_cast<uint64_t>(num_txns) * num_iterations;
    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double avg_per_page_us = elapsed_us / total_pages;
    double avg_per_page_cycles = avg_per_page_us * get_cycles_per_us(*mesh_device);

    return {avg_per_page_us, avg_per_page_cycles};
}

namespace {
std::vector<double> benchmark_h2d_latency(
    const std::shared_ptr<MeshDevice>& mesh_device,
    std::size_t fifo_size,
    std::size_t page_size,
    uint32_t num_iterations,
    H2DMode h2d_mode,
    const MeshCoreCoord& recv_core) {
    auto input_socket = H2DSocket(mesh_device, recv_core, BufferType::L1, fifo_size, h2d_mode);
    input_socket.set_page_size(page_size);

    auto recv_data_buffer = make_l1_mesh_buffer(mesh_device.get(), recv_core.core_coord, page_size);
    // Measurement buffer reserved for ABI compatibility with the kernel; latency measured on host.
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

    execute_program_on_device(*mesh_device, recv_core.device_coord, std::move(recv_program));

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    std::vector<uint32_t> src_vec(page_size_words);

    for (uint32_t w = 0; w < kWarmupIters; w++) {
        input_socket.write(src_vec.data(), 1);
        input_socket.barrier();
    }

    // Time each write + barrier pair on the host.
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
}  // namespace

TEST_F(HDSocketBenchFixture, H2DSocketLatencyBenchmark) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    uint32_t num_iterations = 100;

    MeshCoreCoord recv_core = get_target_benchmark_worker_core(mesh_device_);
    const auto& recv_coord = recv_core.device_coord;
    const double cycles_per_us = get_cycles_per_us(*mesh_device_);

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

                auto latencies_us =
                    benchmark_h2d_latency(mesh_device_, fifo_size, page_size, num_iterations, h2d_mode, recv_core);

                auto stats = summarize_latency_us(latencies_us, cycles_per_us);
                emit_latency_csv_row(page_size, fifo_size, mode_str, num_iterations, stats, recv_coord);
                std::cout.flush();
            }
        }
    }
}

}  // namespace tt::tt_metal::distributed
