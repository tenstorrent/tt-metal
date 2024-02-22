// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/common/metal_soc_descriptor.h"

constexpr uint32_t DEFAULT_ITERATIONS = 1000;
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 2;
constexpr uint32_t DEFAULT_PAGE_SIZE = 2048;
constexpr uint32_t DEFAULT_BATCH_SIZE_K = 512;

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Test read bw and latency from host/dram/l1
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

uint32_t iterations_g = DEFAULT_ITERATIONS;
uint32_t warmup_iterations_g = DEFAULT_WARMUP_ITERATIONS;
CoreRange worker_g = {{0, 0}, {0, 0}};;
CoreCoord src_worker_g = {0, 0};
uint32_t page_size_g;
uint32_t page_count_g;
uint32_t source_mem_g;
uint32_t dram_channel_g;
bool latency_g;
bool lazy_g;
bool time_just_finish_g;
bool read_one_packet_g;
bool page_size_as_runtime_arg_g; // useful particularly on GS multi-dram tests (multiply)

void init(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -w: warm-up iterations before starting timer (default {}), ", DEFAULT_WARMUP_ITERATIONS);
        log_info(LogTest, "  -i: iterations (default {})", DEFAULT_ITERATIONS);
        log_info(LogTest, "  -bs: batch size in K of data to xfer in one iteration (default {}K)", DEFAULT_BATCH_SIZE_K);
        log_info(LogTest, "  -p: page size (default {})", DEFAULT_PAGE_SIZE);
        log_info(LogTest, "  -m: source mem, 0:PCIe, 1:DRAM, 2:L1, 3:ALL_DRAMs, 4:HOST_READ, 5:HOST_WRITE (default 0:PCIe)");
        log_info(LogTest, "  -l: measure latency (default is bandwidth)");
        log_info(LogTest, "  -rx: X of core to issue read (default {})", 1);
        log_info(LogTest, "  -ry: Y of core to issue read (default {})", 0);
        log_info(LogTest, "  -c: when reading from dram, DRAM channel (default 0)");
        log_info(LogTest, "  -sx: when reading from L1, X of core to read from (default {})", 0);
        log_info(LogTest, "  -sy: when reading from L1, Y of core to read (default {})", 0);
        log_info(LogTest, "  -f: time just the finish call (use w/ lazy mode) (default disabled)");
        log_info(LogTest, "  -o: use read_one_packet API.  restrices page size to 8K max (default {})", 0);
        log_info(LogTest, "  -z: enable dispatch lazy mode (default disabled)");
        log_info(LogTest, "  -psrta: pass page size as a runtime argument (default compile time define)");
        exit(0);
    }

    uint32_t core_x = test_args::get_command_option_uint32(input_args, "-rx", 1);
    uint32_t core_y = test_args::get_command_option_uint32(input_args, "-ry", 0);
    warmup_iterations_g = test_args::get_command_option_uint32(input_args, "-w", DEFAULT_WARMUP_ITERATIONS);
    iterations_g = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    lazy_g = test_args::has_command_option(input_args, "-z");
    time_just_finish_g = test_args::has_command_option(input_args, "-f");
    source_mem_g = test_args::get_command_option_uint32(input_args, "-m", 0);
    dram_channel_g = test_args::get_command_option_uint32(input_args, "-c", 0);
    uint32_t src_core_x = test_args::get_command_option_uint32(input_args, "-sx", 0);
    uint32_t src_core_y = test_args::get_command_option_uint32(input_args, "-sy", 0);
    uint32_t size_bytes = test_args::get_command_option_uint32(input_args, "-bs", DEFAULT_BATCH_SIZE_K) * 1024;
    latency_g = test_args::has_command_option(input_args, "-l");
    page_size_g = test_args::get_command_option_uint32(input_args, "-p", DEFAULT_PAGE_SIZE);
    page_size_as_runtime_arg_g = test_args::has_command_option(input_args, "-psrta");
    read_one_packet_g = test_args::has_command_option(input_args, "-o");
    if (read_one_packet_g && page_size_g > 8192) {
        log_info(LogTest, "Page size must be <= 8K for read_one_packet\n");
        exit(-1);
    }
    page_count_g = size_bytes / page_size_g;

    worker_g = CoreRange({core_x, core_y}, {core_x, core_y});
    src_worker_g = {src_core_x, src_core_y};
}

int main(int argc, char **argv) {
    init(argc, argv);

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        CommandQueue& cq = device->command_queue();

        tt_metal::Program program = tt_metal::CreateProgram();

        string src_mem;
        uint32_t noc_addr_x, noc_addr_y;
        uint64_t noc_mem_addr = 0;
        uint32_t dram_banked = 0;
        const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device->id());
        switch (source_mem_g) {
        case 0:
        default:
            {
                src_mem = "FROM_PCIE";
                vector<CoreCoord> pcie_cores = soc_d.get_pcie_cores();
                TT_ASSERT(pcie_cores.size() > 0);
                noc_addr_x = pcie_cores[0].x;
                noc_addr_y = pcie_cores[0].y;
                noc_mem_addr = tt::Cluster::instance().get_pcie_base_addr_from_device(device->id());
            }
            break;
        case 1:
            {
                src_mem = "FROM_DRAM";
                vector<CoreCoord> dram_cores = soc_d.get_dram_cores();
                TT_ASSERT(dram_cores.size() > dram_channel_g);
                noc_addr_x = dram_cores[dram_channel_g].x;
                noc_addr_y = dram_cores[dram_channel_g].y;
            }
            break;
        case 2:
            {
                src_mem = "FROM_L1";
                CoreCoord w = device->physical_core_from_logical_core(src_worker_g, CoreType::WORKER);
                noc_addr_x = w.x;
                noc_addr_y = w.y;
            }
            break;
        case 3:
            {
                src_mem = "FROM_ALL_DRAMS";
                dram_banked = 1;
                noc_addr_x = -1; // unused
                noc_addr_y = -1; // unused
                noc_mem_addr = 0;
            }
            break;
        case 4:
            {
                src_mem = "FROM_L1_TO_HOST";
                log_info(LogTest, "Host bw test overriding page_count to 1");
                CoreCoord w = device->physical_core_from_logical_core(src_worker_g, CoreType::WORKER);
                page_count_g = 1;
                noc_addr_x = w.x;
                noc_addr_y = w.y;
            }
            break;
        case 5:
            {
                src_mem = "FROM_HOST_TO_L1";
                log_info(LogTest, "Host bw test overriding page_count to 1");
                CoreCoord w = device->physical_core_from_logical_core(src_worker_g, CoreType::WORKER);
                page_count_g = 1;
                noc_addr_x = w.x;
                noc_addr_y = w.y;
            }
            break;
        }

        std::map<string, string> defines = {
            {"ITERATIONS", std::to_string(iterations_g)},
            {"PAGE_COUNT", std::to_string(page_count_g)},
            {"LATENCY", std::to_string(latency_g)},
            {"NOC_ADDR_X", std::to_string(noc_addr_x)},
            {"NOC_ADDR_Y", std::to_string(noc_addr_y)},
            {"NOC_MEM_ADDR", std::to_string(noc_mem_addr)},
            {"READ_ONE_PACKET", std::to_string(read_one_packet_g)},
            {"DRAM_BANKED", std::to_string(dram_banked)}
        };
        if (!page_size_as_runtime_arg_g) {
            defines.insert(pair<string, string>("PAGE_SIZE", std::to_string(page_size_g)));
        }

        tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(page_size_g * page_count_g, {{0, tt::DataFormat::Float32}})
            .set_page_size(0, page_size_g);
        auto cb = tt_metal::CreateCircularBuffer(program, worker_g, cb_config);

        auto dm0 = tt_metal::CreateKernel(
                                          program,
                                          "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/bw_and_latency.cpp",
                                          worker_g,
                                          tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .defines = defines});
        if (page_size_as_runtime_arg_g) {
            vector<uint32_t> args;
            args.push_back(page_size_g);
            tt_metal::SetRuntimeArgs(program, dm0, worker_g.start, args);
        }

        CoreCoord w = device->physical_core_from_logical_core(worker_g.start, CoreType::WORKER);
        log_info(LogTest, "Master core: {}", w.str());
        if (source_mem_g == 3) {
            log_info(LogTest, "Reading: {}", src_mem);
        } else if (source_mem_g == 4) {
            log_info(LogTest, "Reading: {} - core ({}, {})", src_mem, w.x, w.y);
        } else if (source_mem_g == 5) {
            log_info(LogTest, "Writing: {} - core ({}, {})", src_mem, w.x, w.y);
        } else {
            log_info(LogTest, "Reading: {} - core ({}, {})", src_mem, noc_addr_x, noc_addr_y);
        }
        if (source_mem_g != 4) {
            log_info(LogTest, "Using API: {}", read_one_packet_g ? "noc_async_read_one_packet" : "noc_async_read");
            log_info(LogTest, "Lazy: {}", lazy_g);
            log_info(LogTest, "Page size ({}): {}", page_size_as_runtime_arg_g ? "runtime arg" : "compile time define", page_size_g);
            log_info(LogTest, "Size per iteration: {}", page_count_g * page_size_g);
        }
        log_info(LogTest, "Iterations: {}", iterations_g);

        std::chrono::duration<double> elapsed_seconds;
        if (source_mem_g < 4) {
            // Cache stuff
            for (int i = 0; i < warmup_iterations_g; i++) {
                EnqueueProgram(cq, program, false);
            }
            Finish(cq);

            if (lazy_g) {
                tt_metal::detail::SetLazyCommandQueueMode(true);
            }

            auto start = std::chrono::system_clock::now();
            EnqueueProgram(cq, program, false);
            if (time_just_finish_g) {
                start = std::chrono::system_clock::now();
            }
            Finish(cq);
            auto end = std::chrono::system_clock::now();
            elapsed_seconds = (end-start);
        } else {
            vector<std::uint32_t> vec;
            vec.resize(page_size_g / sizeof(uint32_t));

            for (int i = 0; i < warmup_iterations_g; i++) {
                if (source_mem_g == 4) {
                    tt::Cluster::instance().read_core(vec, sizeof(uint32_t), tt_cxy_pair(device->id(), w), L1_UNRESERVED_BASE);
                } else {
                    tt::Cluster::instance().write_core(vec.data(), vec.size() * sizeof(uint32_t), tt_cxy_pair(device->id(), w), L1_UNRESERVED_BASE, vec.size() == 1);
                }
            }

            auto start = std::chrono::system_clock::now();
            for (int i = 0; i < iterations_g; i++) {
                if (source_mem_g == 4) {
                    tt::Cluster::instance().read_core(vec, page_size_g, tt_cxy_pair(device->id(), w), L1_UNRESERVED_BASE);
                } else {
                    tt::Cluster::instance().write_core(vec.data(), vec.size() * sizeof(uint32_t), tt_cxy_pair(device->id(), w), L1_UNRESERVED_BASE, vec.size() == 1);
                }
            }
            auto end = std::chrono::system_clock::now();
            elapsed_seconds = (end-start);
        }

        log_info(LogTest, "Ran in {}us", std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds).count());
        if (latency_g) {
            log_info(LogTest, "Latency: {} us",
                (float)std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds).count() / (page_count_g * iterations_g));
        } else {
            float bw = (float)page_count_g * (float)page_size_g * (float)iterations_g / (elapsed_seconds.count() * 1024.0 * 1024.0 * 1024.0);
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << bw;
            log_info(LogTest, "BW: {} GB/s", ss.str());
        }

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }
}
