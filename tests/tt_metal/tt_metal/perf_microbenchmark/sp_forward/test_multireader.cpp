// SPDX-License-Identifier: Apache-2.0
// MULTI-READER: P readers per DRAM bank (8*P total), each reading a UNIQUE contiguous 1/P slice of its
// bank's in1 shard at the peak pattern (reader==consumer, no forwarding). Goal: keep aggregate read BW
// high (450+) with MORE reader cores, so those cores double as compute workers for Mt>1.
// Levers: --preaders P, --noc-mode {noc0|noc1|split|dual}, --place {adjacent|spread}.
//   split = alternate readers within a bank between NOC0 (near NOC0-optimal core) and NOC1 (near
//           NOC1-optimal core) -> hits different DRAM subchannel endpoints, opposite routing directions.
//   dual  = P readers PER CORE using both RISCs/NoCs (BRISC/NOC0 + NCRISC/NOC1) from one physical core.
//
// Usage: TT_METAL_DEVICE_PROFILER=1 ./test_multireader --preaders 2 --noc-mode split --place adjacent
#include <cstdint>
#include <exception>
#include <string>
#include <vector>
#include <set>
#include <array>
#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>
#include "test_common.hpp"

using namespace tt;

int main(int argc, char** argv) {
    bool pass = true;
    uint32_t Kt = 192, N_band = 18, P = 2, block_tiles = 16, num_tests = 6;
    std::string noc_mode = "split", place = "adjacent";
    std::vector<std::string> a(argv, argv + argc);
    std::tie(Kt, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--kt", Kt);
    std::tie(N_band, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nband", N_band);
    std::tie(P, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--preaders", P);
    std::tie(block_tiles, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--block", block_tiles);
    std::tie(num_tests, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--num-tests", num_tests);
    std::tie(noc_mode, a) = test_args::get_command_option_and_remaining_args(a, "--noc-mode", noc_mode);
    std::tie(place, a) = test_args::get_command_option_and_remaining_args(a, "--place", place);

    try {
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        uint32_t tb = tt::tile_size(tt::DataFormat::Float16_b);
        auto grid = device->compute_with_storage_grid_size();
        uint32_t bank_tiles = Kt * N_band;
        bool dual = (noc_mode == "dual");
        // slices per bank = P (dual: P slices but 2 per core -> readers still P, on P/... ). For dual, each
        // core hosts 2 readers (NOC0+NOC1) each reading its own slice, so #cores/bank = ceil(P/2)... keep
        // it simple: dual uses exactly 2 readers/core, P must be even, #cores/bank = P/2.
        TT_FATAL(bank_tiles % (P * block_tiles) == 0, "bank_tiles {} % (P*block) {}", bank_tiles, P * block_tiles);
        if (dual) {
            TT_FATAL(P % 2 == 0, "dual mode needs even P");
        }
        uint32_t slice_tiles = bank_tiles / P;
        uint32_t num_blocks = slice_tiles / block_tiles;
        uint32_t block_bytes = block_tiles * tb;
        uint32_t page_bytes = std::min<uint32_t>(block_bytes, 16384);
        TT_FATAL(block_bytes % page_bytes == 0, "block%page");
        uint32_t num_pages = block_bytes / page_bytes;
        uint64_t total_bytes = (uint64_t)bank_tiles * tb * 8;

        auto opt0 = device->get_optimal_dram_bank_to_logical_worker_assignment(tt_metal::NOC::NOC_0);
        auto opt1 = device->get_optimal_dram_bank_to_logical_worker_assignment(tt_metal::NOC::NOC_1);
        for (uint32_t b = 0; b < 8; ++b) {
            auto p0 = device->worker_core_from_logical_core(opt0[b]);
            auto p1 = device->worker_core_from_logical_core(opt1[b]);
            log_info(
                LogTest,
                "bank {} : NOC0-opt logical ({},{}) phys ({},{}) | NOC1-opt logical ({},{}) phys ({},{})",
                b,
                opt0[b].x,
                opt0[b].y,
                p0.x,
                p0.y,
                opt1[b].x,
                opt1[b].y,
                p1.x,
                p1.y);
        }

        std::set<CoreCoord> used;
        auto find_near = [&](CoreCoord t) -> CoreCoord {
            for (int d = 0; d < (int)(grid.x + grid.y); ++d) {
                for (int dx = -d; dx <= d; ++dx) {
                    int dy = d - std::abs(dx);
                    for (int sy : {dy, -dy}) {
                        int x = (int)t.x + dx, y = (int)t.y + sy;
                        if (x >= 0 && x < (int)grid.x && y >= 0 && y < (int)grid.y) {
                            CoreCoord c(x, y);
                            if (!used.count(c)) {
                                used.insert(c);
                                return c;
                            }
                        }
                    }
                }
            }
            TT_FATAL(false, "no free core");
            return t;
        };

        // reader records: core, bank, vc, base_off, noc(0/1)
        struct R {
            CoreCoord core;
            uint32_t bank, vc, base_off, noc;
        };
        std::vector<R> readers;
        for (uint32_t b = 0; b < 8; ++b) {
            for (uint32_t p = 0; p < P; ++p) {
                uint32_t noc;
                if (noc_mode == "noc0") {
                    noc = 0;
                } else if (noc_mode == "noc1") {
                    noc = 1;
                } else {
                    noc = p & 1;  // split or dual: alternate
                }
                uint32_t base_off = p * slice_tiles * tb;
                uint32_t vc = (b + p) & 0x3;
                if (dual) {
                    // 2 readers share a core (p even = NOC0, p odd = NOC1 on the same core)
                    if (p % 2 == 0) {
                        CoreCoord c = find_near(opt0[b]);
                        readers.push_back({c, b, vc, base_off, 0});
                        // pair the next (odd) slice onto the same core, NOC1
                        uint32_t base_off2 = (p + 1) * slice_tiles * tb;
                        readers.push_back({c, b, (uint32_t)((b + p + 1) & 0x3), base_off2, 1});
                    }
                    // odd p handled above
                    if (p % 2 == 1) {
                        continue;
                    }
                } else {
                    CoreCoord center = noc ? opt1[b] : opt0[b];
                    CoreCoord c = find_near(center);
                    readers.push_back({c, b, vc, base_off, noc});
                }
            }
        }

        std::set<CoreRange> noc0_set, noc1_set, all_set;
        for (auto& r : readers) {
            all_set.insert(CoreRange(r.core));
            if (r.noc == 0) {
                noc0_set.insert(CoreRange(r.core));
            } else {
                noc1_set.insert(CoreRange(r.core));
            }
        }
        CoreRangeSet all_cores(all_set);
        log_info(
            LogTest,
            "P {} noc-mode {} place {} | slice_tiles {} num_blocks {} block {} pages {} | "
            "readers {} (noc0 {} noc1 {}) total {:.1f}MB",
            P,
            noc_mode,
            place,
            slice_tiles,
            num_blocks,
            block_tiles,
            num_pages,
            readers.size(),
            noc0_set.size(),
            noc1_set.size(),
            total_bytes / 1048576.0);

        tt_metal::Program program = tt_metal::Program();
        // scratch CB per reader core (3 blocks). For dual, a core has 2 readers using cb0 and cb1.
        tt_metal::CircularBufferConfig cb0(3 * block_bytes, {{0, tt::DataFormat::Float16_b}});
        cb0.set_page_size(0, tb);
        tt_metal::CreateCircularBuffer(program, all_cores, cb0);
        if (dual) {
            tt_metal::CircularBufferConfig cb1(3 * block_bytes, {{1, tt::DataFormat::Float16_b}});
            cb1.set_page_size(1, tb);
            tt_metal::CreateCircularBuffer(program, all_cores, cb1);
        }

        const char* kpath = "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/reader_mr.cpp";
        tt_metal::KernelHandle k0 = 0, k1 = 0;
        bool have0 = !noc0_set.empty(), have1 = !noc1_set.empty();
        if (have0) {
            k0 = tt_metal::CreateKernel(
                program,
                kpath,
                CoreRangeSet(noc0_set),
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = {num_blocks, num_pages, page_bytes, 0u}});
        }
        if (have1) {
            k1 = tt_metal::CreateKernel(
                program,
                kpath,
                CoreRangeSet(noc1_set),
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = {num_blocks, num_pages, page_bytes, dual ? 1u : 0u}});
        }
        tt_metal::distributed::DeviceLocalBufferConfig lc{.page_size = tb, .buffer_type = tt_metal::BufferType::DRAM};
        tt_metal::distributed::ReplicatedBufferConfig gc{.size = total_bytes};
        auto buf = tt_metal::distributed::MeshBuffer::create(gc, lc, device.get());
        for (auto& r : readers) {
            auto kh = (r.noc == 0) ? k0 : k1;
            tt_metal::SetRuntimeArgs(program, kh, r.core, {(uint32_t)buf->address(), r.bank, r.vc, r.base_off});
        }

        auto wl = tt_metal::distributed::MeshWorkload();
        wl.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        for (uint32_t i = 0; i < num_tests; ++i) {
            tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), wl, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
        }
        tt_metal::ReadMeshDeviceProfilerResults(*device);
        log_info(LogTest, "DONE total_bytes {}", total_bytes);
        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    log_info(LogTest, "{}", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
