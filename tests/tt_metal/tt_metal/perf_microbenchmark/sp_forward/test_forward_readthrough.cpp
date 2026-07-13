// SPDX-License-Identifier: Apache-2.0
// EXP2: full forward read-through. 8 bank-adjacent readers each read their in1 shard (Kt x N_band tiles)
// from their DRAM channel at the peak contiguous pattern on BRISC/NOC0, AND multicast every block to K
// worker cores on NCRISC/NOC1 (workers receive-only). Decisive: does aggregate read BW (BRISC kernel
// time, total=54MB) stay ~494 or funnel because the mcast can't keep up (CB backpressure)?
//
// Usage: TT_METAL_DEVICE_PROFILER=1 ./test_forward_readthrough --k 2 --block 8 --cbd 8 --md 2
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
    uint32_t Kt = 192, N_band = 18, K = 2, block_tiles = 8, cbd = 8, md = 2, num_tests = 6, decouple = 0;
    std::vector<std::string> a(argv, argv + argc);
    std::tie(decouple, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--decouple", decouple);
    std::tie(Kt, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--kt", Kt);
    std::tie(N_band, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nband", N_band);
    std::tie(K, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--k", K);
    std::tie(block_tiles, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--block", block_tiles);
    std::tie(cbd, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--cbd", cbd);
    std::tie(md, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--md", md);
    std::tie(num_tests, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--num-tests", num_tests);

    try {
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        uint32_t tb = tt::tile_size(tt::DataFormat::Float16_b);  // 2048
        auto grid = device->compute_with_storage_grid_size();
        uint32_t bank_tiles = Kt * N_band;
        TT_FATAL(bank_tiles % block_tiles == 0, "bank_tiles {} % block {}", bank_tiles, block_tiles);
        uint32_t num_blocks = bank_tiles / block_tiles;
        uint32_t block_bytes = block_tiles * tb;
        uint32_t page_bytes = std::min<uint32_t>(block_bytes, 16384);
        TT_FATAL(block_bytes % page_bytes == 0, "block_bytes {} % page {}", block_bytes, page_bytes);
        uint32_t num_pages = block_bytes / page_bytes;
        uint64_t bank_bytes = (uint64_t)bank_tiles * tb;
        uint64_t total_bytes = bank_bytes * 8;
        TT_FATAL(cbd >= 3, "cbd must be >= 3 blocks");

        // 8 bank-adjacent reader cores
        auto opt = device->get_optimal_dram_bank_to_logical_worker_assignment(tt_metal::NOC::NOC_0);
        std::vector<CoreCoord> readers(opt.begin(), opt.begin() + 8);
        std::set<CoreCoord> used(readers.begin(), readers.end());

        // For each reader, K workers in a single logical column (contiguous phys rectangle), source
        // (reader) outside it. Scan columns near the reader for K consecutive free rows.
        std::vector<std::vector<CoreCoord>> workers(8);
        std::vector<std::array<uint32_t, 4>> rects(8);  // x0,y0,x1,y1 phys
        std::vector<int> col_order;
        auto add_col = [&](int c) {
            if (c >= 0 && c < (int)grid.x) {
                col_order.push_back(c);
            }
        };
        for (uint32_t b = 0; b < 8; ++b) {
            int rx = readers[b].x, ry = readers[b].y;
            col_order.clear();
            for (int d = 0; d < (int)grid.x; ++d) {
                add_col(rx - d);
                if (d) {
                    add_col(rx + d);
                }
            }
            bool placed = false;
            for (int cx : col_order) {
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
            TT_FATAL(placed, "could not place {} workers for reader {} at ({},{})", K, b, rx, ry);
            log_info(
                LogTest,
                "reader {} ({},{}) bank {} vc {} -> {} workers col {} rect phys ({},{})-({},{})",
                b,
                rx,
                ry,
                b,
                b & 0x3,
                K,
                workers[b][0].x,
                rects[b][0],
                rects[b][1],
                rects[b][2],
                rects[b][3]);
        }

        // DRAM buffer: one bank shard * 8 (page=tile). Each reader reads its bank contiguously.
        tt_metal::distributed::DeviceLocalBufferConfig lc{.page_size = tb, .buffer_type = tt_metal::BufferType::DRAM};
        tt_metal::distributed::ReplicatedBufferConfig gc{.size = total_bytes};
        auto buf = tt_metal::distributed::MeshBuffer::create(gc, lc, device.get());

        // core sets
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
            "Kt {} N_band {} K {} block {} (bytes {}) pages {} cbd {} md {} | num_blocks {} total {:.1f}MB",
            Kt,
            N_band,
            K,
            block_tiles,
            block_bytes,
            num_pages,
            cbd,
            md,
            num_blocks,
            total_bytes / 1048576.0);

        tt_metal::Program program = tt_metal::Program();
        // cb0 (reader->mcast) depth cbd blocks; cb1 (mcast dst) 1 block. Allocate BOTH on the union so
        // cb1's L1 offset matches between reader (source) and workers.
        tt_metal::CircularBufferConfig cb0(cbd * block_bytes, {{0, tt::DataFormat::Float16_b}});
        cb0.set_page_size(0, tb);
        tt_metal::CreateCircularBuffer(program, cb_cores, cb0);
        tt_metal::CircularBufferConfig cb1(block_bytes, {{1, tt::DataFormat::Float16_b}});
        cb1.set_page_size(1, tb);
        tt_metal::CreateCircularBuffer(program, cb_cores, cb1);

        auto reader_k = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/reader_fwd.cpp",
            reader_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = {num_blocks, num_pages, page_bytes, block_tiles, 0u, decouple ? 0u : 1u}});
        auto mcast_k = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/mcast_fwd.cpp",
            reader_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {num_blocks, block_tiles, block_bytes, 0u, 1u, md, decouple ? 0u : 1u}});
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/noop.cpp",
            worker_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        for (uint32_t b = 0; b < 8; ++b) {
            uint32_t vc = b & 0x3;
            tt_metal::SetRuntimeArgs(program, reader_k, readers[b], {(uint32_t)buf->address(), b, vc});
            // mcast on NOC1 -> pass corners SWAPPED (start=max, end=min) so DYNAMIC_NOC flip yields a
            // valid NOC1-frame rectangle (multi-core mcast hangs otherwise).
            tt_metal::SetRuntimeArgs(
                program, mcast_k, readers[b], {K, rects[b][2], rects[b][3], rects[b][0], rects[b][1]});
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
