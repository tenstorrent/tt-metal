// SPDX-License-Identifier: Apache-2.0
// WORKER-PULL: 8 bank-adjacent readers stream their in1 shard from DRAM into a D-deep L1 ring (BRISC/NOC0)
// and do nothing else but signal readiness. Each reader's K workers NoC-read (pull) the shared in1 straight
// out of the reader's L1 (NCRISC/NOC1) and credit-return freed slots. This keeps the L1 read-OUT off the
// reader (charged to the workers), unlike the mcast-forward where the reader both read DRAM and sourced the
// send (which starved to ~18 GB/s/core). Decisive: does the reader read stay ~peak as K grows 1..8, and
// what unique-in1 delivery BW do the workers achieve?
//
// Usage: TT_METAL_DEVICE_PROFILER=1 ./test_worker_pull --k 4 --block 16 --dring 8 --wd 4
#include <cstdint>
#include <exception>
#include <string>
#include <vector>
#include <set>
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
    uint32_t Kt = 192, N_band = 18, K = 4, block_tiles = 16, D = 8, WD = 4, num_tests = 6;
    std::vector<std::string> a(argv, argv + argc);
    std::tie(Kt, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--kt", Kt);
    std::tie(N_band, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nband", N_band);
    std::tie(K, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--k", K);
    std::tie(block_tiles, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--block", block_tiles);
    std::tie(D, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--dring", D);
    std::tie(WD, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--wd", WD);
    std::tie(num_tests, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--num-tests", num_tests);

    try {
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        uint32_t tb = tt::tile_size(tt::DataFormat::Float16_b);
        auto grid = device->compute_with_storage_grid_size();
        uint32_t bank_tiles = Kt * N_band;
        TT_FATAL(bank_tiles % block_tiles == 0, "bank_tiles {} % block {}", bank_tiles, block_tiles);
        uint32_t num_blocks = bank_tiles / block_tiles;
        uint32_t block_bytes = block_tiles * tb;
        uint32_t page_bytes = std::min<uint32_t>(block_bytes, 16384);
        TT_FATAL(block_bytes % page_bytes == 0, "block_bytes {} % page {}", block_bytes, page_bytes);
        uint32_t num_pages = block_bytes / page_bytes;
        TT_FATAL(D >= 3 && D >= WD, "D {} must be >=3 and >=WD {}", D, WD);
        uint64_t bank_bytes = (uint64_t)bank_tiles * tb;
        uint64_t total_bytes = bank_bytes * 8;

        // 8 bank-adjacent readers
        auto opt = device->get_optimal_dram_bank_to_logical_worker_assignment(tt_metal::NOC::NOC_0);
        std::vector<CoreCoord> readers(opt.begin(), opt.begin() + 8);
        std::set<CoreCoord> used(readers.begin(), readers.end());

        // K workers per reader in a single logical column near it (contiguous phys rect, reader outside).
        std::vector<std::vector<CoreCoord>> workers(8);
        std::vector<std::array<uint32_t, 4>> rects(8);  // x0,y0,x1,y1 phys (min..max)
        std::vector<std::array<uint32_t, 2>> reader_phys(8);
        for (uint32_t b = 0; b < 8; ++b) {
            int rx = readers[b].x;
            std::vector<int> cols;
            for (int d = 0; d < (int)grid.x; ++d) {
                if (rx - d >= 0) {
                    cols.push_back(rx - d);
                }
                if (d && rx + d < (int)grid.x) {
                    cols.push_back(rx + d);
                }
            }
            bool placed = false;
            for (int cx : cols) {
                for (int r0 = 0; r0 + (int)K <= (int)grid.y && !placed; ++r0) {
                    bool ok = true;
                    for (uint32_t r = 0; r < K; ++r) {
                        CoreCoord c(cx, r0 + r);
                        if (used.count(c) || c == readers[b]) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) {
                        continue;
                    }
                    for (uint32_t r = 0; r < K; ++r) {
                        CoreCoord c(cx, r0 + r);
                        workers[b].push_back(c);
                        used.insert(c);
                    }
                    auto p0 = device->worker_core_from_logical_core(CoreCoord(cx, r0));
                    auto p1 = device->worker_core_from_logical_core(CoreCoord(cx, r0 + K - 1));
                    rects[b] = {std::min(p0.x, p1.x), std::min(p0.y, p1.y), std::max(p0.x, p1.x), std::max(p0.y, p1.y)};
                    placed = true;
                }
                if (placed) {
                    break;
                }
            }
            TT_FATAL(placed, "could not place {} workers for reader {}", K, b);
            auto rp = device->worker_core_from_logical_core(readers[b]);
            reader_phys[b] = {rp.x, rp.y};
        }

        tt_metal::distributed::DeviceLocalBufferConfig lc{.page_size = tb, .buffer_type = tt_metal::BufferType::DRAM};
        tt_metal::distributed::ReplicatedBufferConfig gc{.size = total_bytes};
        auto buf = tt_metal::distributed::MeshBuffer::create(gc, lc, device.get());

        std::set<CoreRange> reader_set, worker_set, cb_set;
        for (auto& c : readers) {
            reader_set.insert(CoreRange(c));
            cb_set.insert(CoreRange(c));
        }
        for (auto& wv : workers) {
            for (auto& c : wv) {
                worker_set.insert(CoreRange(c));
                cb_set.insert(CoreRange(c));
            }
        }
        CoreRangeSet reader_cores(reader_set), worker_cores(worker_set), cb_cores(cb_set);

        log_info(
            LogTest,
            "WORKER-PULL Kt {} N_band {} K {} block {} D {} WD {} | num_blocks {} total {:.1f}MB",
            Kt,
            N_band,
            K,
            block_tiles,
            D,
            WD,
            num_blocks,
            total_bytes / 1048576.0);

        tt_metal::Program program = tt_metal::Program();
        // cb0 = reader ring (shared layout, D blocks); cb1 = worker local scratch (WD blocks)
        tt_metal::CircularBufferConfig cb0(D * block_bytes, {{0, tt::DataFormat::Float16_b}});
        cb0.set_page_size(0, tb);
        tt_metal::CreateCircularBuffer(program, cb_cores, cb0);
        tt_metal::CircularBufferConfig cb1(WD * block_bytes, {{1, tt::DataFormat::Float16_b}});
        cb1.set_page_size(1, tb);
        tt_metal::CreateCircularBuffer(program, cb_cores, cb1);

        uint32_t valid_sem = tt_metal::CreateSemaphore(program, cb_cores, 0);
        uint32_t free_sem = tt_metal::CreateSemaphore(program, cb_cores, 0);
        uint32_t produced_sem = tt_metal::CreateSemaphore(program, cb_cores, 0);

        auto reader_k = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/reader_pull.cpp",
            reader_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = {
                    num_blocks, num_pages, page_bytes, block_tiles, 0u, D, K, valid_sem, free_sem, produced_sem}});
        auto worker_k = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/worker_pull.cpp",
            worker_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {num_blocks, block_tiles, block_bytes, D, WD, 0u, 1u, valid_sem, free_sem}});

        for (uint32_t b = 0; b < 8; ++b) {
            uint32_t vc = b & 0x3;
            // reader valid mcast on NOC0 -> normal corner order (min,max)
            tt_metal::SetRuntimeArgs(
                program,
                reader_k,
                readers[b],
                {(uint32_t)buf->address(), b, vc, rects[b][0], rects[b][1], rects[b][2], rects[b][3]});
            for (auto& w : workers[b]) {
                tt_metal::SetRuntimeArgs(program, worker_k, w, {reader_phys[b][0], reader_phys[b][1]});
            }
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
