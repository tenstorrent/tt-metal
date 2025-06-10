// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <emmintrin.h>
#include <fmt/base.h>
#include <stdlib.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/event.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_common.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/command_queue_common.hpp"
#include "umd/device/tt_core_coordinates.h"
#include "umd/device/tt_xy_pair.h"
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

constexpr uint32_t DEFAULT_ITERATIONS = 1000;
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 2;
constexpr uint32_t DEFAULT_PAGE_SIZE = 2048;
constexpr uint32_t DEFAULT_BATCH_SIZE_K = 512;

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Test read/write bw and latency from host/dram/l1
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;

uint32_t iterations_g = DEFAULT_ITERATIONS;
uint32_t warmup_iterations_g = DEFAULT_WARMUP_ITERATIONS;
CoreRange worker_g = {{0, 0}, {0, 0}};
CoreCoord src_worker_g = {0, 0};
CoreRange mcast_src_workers_g = {{0, 0}, {0, 0}};
uint32_t page_size_g;
uint32_t page_count_g;
uint32_t source_mem_g;
uint32_t dram_channel_g;
bool latency_g;
bool time_just_finish_g;
bool read_one_packet_g;
bool page_size_as_runtime_arg_g;  // useful particularly on GS multi-dram tests (multiply)
bool hammer_write_reg_g = false;
bool hammer_pcie_g = false;
bool hammer_pcie_type_g = false;
bool test_write = false;
bool linked = false;
bool dump_profiler_results = false;
uint32_t nop_count_g = 0;

void init(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") || test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -w: warm-up iterations before starting timer (default {}), ", DEFAULT_WARMUP_ITERATIONS);
        log_info(LogTest, "  -i: iterations (default {})", DEFAULT_ITERATIONS);
        log_info(
            LogTest, "  -bs: batch size in K of data to xfer in one iteration (default {}K)", DEFAULT_BATCH_SIZE_K);
        log_info(LogTest, "  -p: page size (default {})", DEFAULT_PAGE_SIZE);
        log_info(
            LogTest,
            "  -m: source mem, 0:PCIe, 1:DRAM, 2:L1, 3:ALL_DRAMs, 4:HOST_READ, 5:HOST_WRITE, 6:MULTICAST_WRITE "
            "(default 0:PCIe)");
        log_info(LogTest, "  -l: measure latency (default is bandwidth)");
        log_info(LogTest, "  -rx: X of core to issue read or write (default {})", 1);
        log_info(LogTest, "  -ry: Y of core to issue read or write (default {})", 0);
        log_info(
            LogTest,
            "  -sx: when reading from L1, X of core to read from. when issuing a multicast write, X of start core to "
            "write to. (default {})",
            0);
        log_info(
            LogTest,
            "  -sy: when reading from L1, Y of core to read from. when issuing a multicast write, Y of start core to "
            "write to. (default {})",
            0);
        log_info(LogTest, "  -tx: when issuing a multicast write, X of end core to write to (default {})", 0);
        log_info(LogTest, "  -ty: when issuing a multicast write, Y of end core to write to (default {})", 0);
        log_info(LogTest, "  -wr: issue unicast write instead of read (default false)");
        log_info(LogTest, "  -c: when reading from dram, DRAM channel (default 0)");
        log_info(LogTest, "  -f: time just the finish call (default disabled)");
        log_info(LogTest, "  -o: use read_one_packet API.  restricts page size to 8K max (default {})", 0);
        log_info(LogTest, "-link: link mcast transactions");
        log_info(LogTest, " -hr: hammer write_reg while executing (for PCIe test)");
        log_info(LogTest, " -hp: hammer hugepage PCIe memory while executing (for PCIe test)");
        log_info(LogTest, " -hpt:hammer hugepage PCIe hammer type: 0:32bit writes 1:128bit non-temporal writes");
        log_info(LogTest, "  -psrta: pass page size as a runtime argument (default compile time define)");
        log_info(LogTest, " -nop: time loop of <n> nops");
        log_info(LogTest, "-profdump: dump profiler results before closing device");
        exit(0);
    }

    uint32_t core_x = test_args::get_command_option_uint32(input_args, "-rx", 1);
    uint32_t core_y = test_args::get_command_option_uint32(input_args, "-ry", 0);
    warmup_iterations_g = test_args::get_command_option_uint32(input_args, "-w", DEFAULT_WARMUP_ITERATIONS);
    iterations_g = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    hammer_write_reg_g = test_args::has_command_option(input_args, "-hr");
    hammer_pcie_g = test_args::has_command_option(input_args, "-hp");
    hammer_pcie_type_g = test_args::get_command_option_uint32(input_args, "-hpt", 0);
    time_just_finish_g = test_args::has_command_option(input_args, "-f");
    source_mem_g = test_args::get_command_option_uint32(input_args, "-m", 0);
    uint32_t src_core_x = test_args::get_command_option_uint32(input_args, "-sx", 0);
    uint32_t src_core_y = test_args::get_command_option_uint32(input_args, "-sy", 0);
    uint32_t mcast_end_core_x = test_args::get_command_option_uint32(input_args, "-tx", 0);
    uint32_t mcast_end_core_y = test_args::get_command_option_uint32(input_args, "-ty", 0);
    dram_channel_g = test_args::get_command_option_uint32(input_args, "-c", 0);
    uint32_t size_bytes = test_args::get_command_option_uint32(input_args, "-bs", DEFAULT_BATCH_SIZE_K) * 1024;
    latency_g = test_args::has_command_option(input_args, "-l");
    page_size_g = test_args::get_command_option_uint32(input_args, "-p", DEFAULT_PAGE_SIZE);
    page_size_as_runtime_arg_g = test_args::has_command_option(input_args, "-psrta");
    read_one_packet_g = test_args::has_command_option(input_args, "-o");
    nop_count_g = test_args::get_command_option_uint32(input_args, "-nop", 0);

    if (read_one_packet_g && page_size_g > 8192) {
        log_info(LogTest, "Page size must be <= 8K for read_one_packet\n");
        exit(-1);
    }
    page_count_g = size_bytes / page_size_g;

    test_write = test_args::has_command_option(input_args, "-wr");
    if (test_write && (source_mem_g != 2 && source_mem_g != 6)) {
        log_info(LogTest, "Writing only tested w/ L1 destination\n");
        exit(-1);
    }

    linked = test_args::has_command_option(input_args, "-link");

    dump_profiler_results = test_args::has_command_option(input_args, "-profdump");

    worker_g = CoreRange({core_x, core_y}, {core_x, core_y});
    src_worker_g = {src_core_x, src_core_y};

    if (source_mem_g == 6) {
        if (mcast_end_core_x < src_core_x || mcast_end_core_y < src_core_y) {
            log_info(LogTest, "X of end core must be >= X of start core, Y of end core must be >= Y of start core");
            exit(-1);
        }

        mcast_src_workers_g = CoreRange({src_core_x, src_core_y}, {mcast_end_core_x, mcast_end_core_y});

        if (mcast_src_workers_g.intersects(worker_g)) {
            log_info(
                LogTest,
                "Multicast destination rectangle and core that issues the multicast cannot overlap - Multicast "
                "destination rectangle: {} Master core: {}",
                mcast_src_workers_g.str(),
                worker_g.start_coord.str());
            exit(-1);
        }
    }
}

#define CACHE_LINE_SIZE 64
void nt_memcpy(uint8_t* __restrict dst, const uint8_t* __restrict src, size_t n) {
    size_t num_lines = n / CACHE_LINE_SIZE;

    size_t i;
    for (i = 0; i < num_lines; i++) {
        size_t j;
        for (j = 0; j < CACHE_LINE_SIZE / sizeof(__m128i); j++) {
            __m128i blk = _mm_loadu_si128((const __m128i*)src);
            /* non-temporal store */
            _mm_stream_si128((__m128i*)dst, blk);
            src += sizeof(__m128i);
            dst += sizeof(__m128i);
        }
        n -= CACHE_LINE_SIZE;
    }

    if (num_lines > 0) {
        tt_driver_atomics::sfence();
    }
}

int main(int argc, char** argv) {
    init(argc, argv);

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        CommandQueue& cq = device->command_queue();

        tt_metal::Program program = tt_metal::CreateProgram();

        string src_mem;
        uint32_t noc_addr_x, noc_addr_y;
        uint64_t noc_mem_addr = 0;
        uint32_t dram_banked = 0;
        uint32_t issue_mcast = 0;
        uint32_t num_mcast_dests = mcast_src_workers_g.size();
        uint32_t mcast_noc_addr_end_x = 0;
        uint32_t mcast_noc_addr_end_y = 0;

        chip_id_t mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device->id());
        void* host_pcie_base =
            (void*)tt::tt_metal::MetalContext::instance().get_cluster().host_dma_address(0, mmio_device_id, channel);
        uint64_t dev_pcie_base =
            tt::tt_metal::MetalContext::instance().get_cluster().get_pcie_base_addr_from_device(device->id());
        uint64_t pcie_offset = 1024 * 1024 * 50;  // beyond where FD will write...maybe

        const metal_SocDescriptor& soc_d =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
        switch (source_mem_g) {
            case 0:
            default: {
                src_mem = "FROM_PCIE";
                vector<tt::umd::CoreCoord> pcie_cores = soc_d.get_cores(CoreType::PCIE, soc_d.get_umd_coord_system());
                TT_ASSERT(pcie_cores.size() > 0);
                noc_addr_x = pcie_cores[0].x;
                noc_addr_y = pcie_cores[0].y;
                noc_mem_addr = dev_pcie_base + pcie_offset;
            } break;
            case 1: {
                src_mem = "FROM_DRAM";
                vector<tt::umd::CoreCoord> dram_cores = soc_d.get_cores(CoreType::DRAM, soc_d.get_umd_coord_system());
                TT_ASSERT(dram_cores.size() > dram_channel_g);
                noc_addr_x = dram_cores[dram_channel_g].x;
                noc_addr_y = dram_cores[dram_channel_g].y;
            } break;
            case 2: {
                src_mem = test_write ? "TO_L1" : "FROM_L1";
                CoreCoord w = device->worker_core_from_logical_core(src_worker_g);
                noc_addr_x = w.x;
                noc_addr_y = w.y;
            } break;
            case 3: {
                src_mem = "FROM_ALL_DRAMS";
                dram_banked = 1;
                noc_addr_x = -1;  // unused
                noc_addr_y = -1;  // unused
                noc_mem_addr = 0;
            } break;
            case 4: {
                src_mem = "FROM_L1_TO_HOST";
                log_info(LogTest, "Host bw test overriding page_count to 1");
                CoreCoord w = device->worker_core_from_logical_core(src_worker_g);
                page_count_g = 1;
                noc_addr_x = w.x;
                noc_addr_y = w.y;
            } break;
            case 5: {
                src_mem = "FROM_HOST_TO_L1";
                log_info(LogTest, "Host bw test overriding page_count to 1");
                CoreCoord w = device->worker_core_from_logical_core(src_worker_g);
                page_count_g = 1;
                noc_addr_x = w.x;
                noc_addr_y = w.y;
            } break;
            case 6: {
                src_mem = "FROM_L1_TO_MCAST";
                issue_mcast = 1;
                CoreCoord start = device->worker_core_from_logical_core(mcast_src_workers_g.start_coord);
                CoreCoord end = device->worker_core_from_logical_core(mcast_src_workers_g.end_coord);
                noc_addr_x = start.x;
                noc_addr_y = start.y;
                mcast_noc_addr_end_x = end.x;
                mcast_noc_addr_end_y = end.y;
                test_write = true;
            } break;
        }

        std::map<string, string> defines = {
            {"ITERATIONS", std::to_string(iterations_g)},
            {"PAGE_COUNT", std::to_string(page_count_g)},
            {"LATENCY", std::to_string(latency_g)},
            {"NOC_ADDR_X", std::to_string(noc_addr_x)},
            {"NOC_ADDR_Y", std::to_string(noc_addr_y)},
            {"NOC_MEM_ADDR", std::to_string(noc_mem_addr)},
            {"READ_ONE_PACKET", std::to_string(read_one_packet_g)},
            {"DRAM_BANKED", std::to_string(dram_banked)},
            {"ISSUE_MCAST", std::to_string(issue_mcast)},
            {"WRITE", std::to_string(test_write)},
            {"LINKED", std::to_string(linked)},
            {"NUM_MCAST_DESTS", std::to_string(num_mcast_dests)},
            {"MCAST_NOC_END_ADDR_X", std::to_string(mcast_noc_addr_end_x)},
            {"MCAST_NOC_END_ADDR_Y", std::to_string(mcast_noc_addr_end_y)},
            {"NOP_COUNT", std::to_string(nop_count_g)},
        };
        if (!page_size_as_runtime_arg_g) {
            defines.insert(std::pair<string, string>("PAGE_SIZE", std::to_string(page_size_g)));
        }

        tt_metal::CircularBufferConfig cb_config =
            tt_metal::CircularBufferConfig(page_size_g * page_count_g, {{0, tt::DataFormat::Float32}})
                .set_page_size(0, page_size_g);
        auto cb = tt_metal::CreateCircularBuffer(program, worker_g, cb_config);

        auto dm0 = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/bw_and_latency.cpp",
            worker_g,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .defines = defines});
        if (page_size_as_runtime_arg_g) {
            tt_metal::SetRuntimeArgs(program, dm0, worker_g.start_coord, {page_size_g});
        }

        std::shared_ptr<Event> sync_event = std::make_shared<Event>();

        CoreCoord w = device->worker_core_from_logical_core(worker_g.start_coord);
        log_info(LogTest, "Master core: {}", w.str());
        string direction = test_write ? "Writing" : "Reading";
        if (source_mem_g == 3) {
            log_info(LogTest, "{}: {}", direction, src_mem);
        } else if (source_mem_g == 4) {
            log_info(LogTest, "{}: {} - core ({}, {})", direction, src_mem, w.x, w.y);
        } else if (source_mem_g == 5) {
            log_info(LogTest, "{}: {} - core ({}, {})", test_write, src_mem, w.x, w.y);
        } else if (source_mem_g == 6) {
            log_info(
                LogTest,
                "direction: {} - core grid [({}, {}) - ({}, {})]",
                direction,
                src_mem,
                noc_addr_x,
                noc_addr_y,
                mcast_noc_addr_end_x,
                mcast_noc_addr_end_y);
        } else {
            log_info(LogTest, "{}: {} - core ({}, {})", direction, src_mem, noc_addr_x, noc_addr_y);
        }
        if (source_mem_g < 4 || source_mem_g == 6) {
            std::string api;
            string read_write = test_write ? "write" : "read";
            if (issue_mcast) {
                api = "noc_async_" + read_write + "_multicast";
            } else if (read_one_packet_g) {
                api = "noc_async_" + read_write + "_one_packet";
            } else {
                api = "noc_async_" + read_write;
            }
            log_info(LogTest, "Using API: {}", api);
            log_info(
                LogTest,
                "Page size ({}): {}",
                page_size_as_runtime_arg_g ? "runtime arg" : "compile time define",
                page_size_g);
            log_info(LogTest, "Size per iteration: {}", page_count_g * page_size_g);
        }
        log_info(LogTest, "Iterations: {}", iterations_g);
        if (hammer_pcie_g) {
            log_warning(LogTest, "WARNING: Hardcoded PCIe addresses may not be safe w/ FD, check above if hung");
        }

        vector<uint32_t> blank(page_size_g / sizeof(uint32_t));
        std::chrono::duration<double> elapsed_seconds;
        if (source_mem_g < 4 || source_mem_g == 6) {
            // Cache stuff
            for (int i = 0; i < warmup_iterations_g; i++) {
                EnqueueProgram(cq, program, false);
            }
            Finish(cq);

            auto start = std::chrono::system_clock::now();
            EnqueueProgram(cq, program, false);
            if (time_just_finish_g) {
                start = std::chrono::system_clock::now();
            }
            if (hammer_write_reg_g || hammer_pcie_g) {
                EnqueueRecordEvent(cq, sync_event);

                bool done = false;
                uint32_t addr = 0xfafafafa;
                uint32_t offset = 0;
                uint32_t page = 0;
                uint32_t* pcie_base = (uint32_t*)host_pcie_base + pcie_offset / sizeof(uint32_t);
                uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
                while (!done) {
                    if (hammer_write_reg_g) {
                        tt::tt_metal::MetalContext::instance().get_cluster().write_reg(
                            &addr, tt_cxy_pair(device->id(), w), l1_unreserved_base);
                    }
                    if (hammer_pcie_g) {
                        if (page == page_count_g) {
                            page = 0;
                            offset = 0;
                        }

                        if (hammer_pcie_type_g == 0) {
                            for (int i = 0; i < page_size_g / sizeof(uint32_t); i++) {
                                pcie_base[offset++] = 0;
                            }
                        } else {
                            uint32_t* pcie_addr = ((uint32_t*)pcie_base) + offset;
                            nt_memcpy((uint8_t*)pcie_addr, (uint8_t*)&blank[0], page_size_g);
                        }
                        page++;
                    }
                    if (EventQuery(sync_event)) {
                        done = true;
                    }
                }
            }

            Finish(cq);
            auto end = std::chrono::system_clock::now();
            elapsed_seconds = (end - start);
        } else if (source_mem_g == 4 || source_mem_g == 5) {
            vector<std::uint32_t> vec;
            vec.resize(page_size_g / sizeof(uint32_t));

            uint32_t dispatch_l1_unreserved_base =
                MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
                    CommandQueueDeviceAddrType::UNRESERVED);
            for (int i = 0; i < warmup_iterations_g; i++) {
                if (source_mem_g == 4) {
                    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                        vec, sizeof(uint32_t), tt_cxy_pair(device->id(), w), dispatch_l1_unreserved_base);
                } else {
                    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                        vec.data(),
                        vec.size() * sizeof(uint32_t),
                        tt_cxy_pair(device->id(), w),
                        dispatch_l1_unreserved_base);
                }
            }

            auto start = std::chrono::system_clock::now();
            for (int i = 0; i < iterations_g; i++) {
                if (source_mem_g == 4) {
                    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                        vec, page_size_g, tt_cxy_pair(device->id(), w), dispatch_l1_unreserved_base);
                } else {
                    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                        vec.data(),
                        vec.size() * sizeof(uint32_t),
                        tt_cxy_pair(device->id(), w),
                        dispatch_l1_unreserved_base);
                }
            }
            auto end = std::chrono::system_clock::now();
            elapsed_seconds = (end - start);
        }

        log_info(
            LogTest, "Ran in {}us", std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds).count());
        if (latency_g) {
            log_info(
                LogTest,
                "Latency: {} us",
                (float)std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds).count() /
                    (page_count_g * iterations_g));
        } else {
            float bw = (float)page_count_g * (float)page_size_g * (float)iterations_g /
                       (elapsed_seconds.count() * 1000.0 * 1000.0 * 1000.0);
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << bw;
            log_info(LogTest, "BW: {} GB/s", ss.str());
        }

        if (dump_profiler_results) {
            tt_metal::detail::DumpDeviceProfileResults(device);
        }

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(tt::LogTest, "{}", e.what());
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }
}
