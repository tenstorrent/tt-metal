// SPDX-License-Identifier: Apache-2.0
// in0 CONCURRENCY: split-NOC multi-reader in1 (8P readers at ~500 GB/s) PLUS L loader cores reading in0
// (interleaved, K-block streamed) and optionally broadcasting each K-block to a compute rectangle. Answers:
//  (1) can 1-2 cores read in0 interleaved fast enough concurrent with in1 at the DRAM cap, without denting
//      in1? (mode=read)  (2) idea1 vs idea2: does the broadcast NoC matter? (mode=mc0 vs mc1)
// Metrics: in1 aggregate BW (readers, no-filter over reader RISCs) vs the no-in0 baseline; in0 loader time.
//
// Usage: TT_METAL_DEVICE_PROFILER=1 ./test_in0_concurrent --preaders 8 --loaders 1 --mt 1 --mode mc1
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
    uint32_t Kt = 192, N_band = 18, P = 8, L = 1, Mt = 1, Kb = 8, block_tiles = 16, num_tests = 6;
    std::string mode = "mc1";  // none | read | mc0 | mc1
    std::vector<std::string> a(argv, argv + argc);
    std::tie(Kt, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--kt", Kt);
    std::tie(N_band, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nband", N_band);
    std::tie(P, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--preaders", P);
    std::tie(L, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--loaders", L);
    std::tie(Mt, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--mt", Mt);
    std::tie(Kb, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--kb", Kb);
    std::tie(block_tiles, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--block", block_tiles);
    std::tie(num_tests, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--num-tests", num_tests);
    std::tie(mode, a) = test_args::get_command_option_and_remaining_args(a, "--mode", mode);

    try {
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        uint32_t tb = tt::tile_size(tt::DataFormat::Float16_b);
        auto grid = device->compute_with_storage_grid_size();
        // ---- in1 (split multireader) ----
        uint32_t bank_tiles = Kt * N_band;
        TT_FATAL(bank_tiles % (P * block_tiles) == 0, "bank%(P*block)");
        uint32_t slice_tiles = bank_tiles / P, num_blocks = slice_tiles / block_tiles;
        uint32_t block_bytes = block_tiles * tb;
        uint32_t page_bytes = std::min<uint32_t>(block_bytes, 16384);
        uint32_t num_pages = block_bytes / page_bytes;
        uint64_t in1_bytes = (uint64_t)bank_tiles * tb * 8;
        // ---- in0 ----
        bool do_mc = (mode == "mc0" || mode == "mc1" || mode == "mc1c");
        bool mc_noc1 = (mode == "mc1" || mode == "mc1c");
        bool contig = (mode == "mc1c");  // DRAM-sharded contiguous in0 read (vs interleaved)
        TT_FATAL(Kt % Kb == 0, "Kt%Kb");
        uint32_t nkb = Kt / Kb;
        TT_FATAL(nkb % L == 0, "nkb%L");
        uint32_t tiles_per_kb = Mt * Kb, kb_per_loader = nkb / L;
        uint64_t in0_tiles = (uint64_t)Mt * Kt;
        uint64_t in0_bytes = in0_tiles * tb;

        auto opt0 = device->get_optimal_dram_bank_to_logical_worker_assignment(tt_metal::NOC::NOC_0);
        auto opt1 = device->get_optimal_dram_bank_to_logical_worker_assignment(tt_metal::NOC::NOC_1);
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

        struct R {
            CoreCoord core;
            uint32_t bank, vc, base_off, noc;
        };
        std::vector<R> readers;
        for (uint32_t b = 0; b < 8; ++b) {
            for (uint32_t p = 0; p < P; ++p) {
                uint32_t noc = p & 1;  // split
                CoreCoord c = find_near(noc ? opt1[b] : opt0[b]);
                readers.push_back({c, b, (uint32_t)((b + p) & 0x3), p * slice_tiles * tb, noc});
            }
        }
        // loaders: place in the top-right corner region (cols grid.x-1 .. ), outside the mcast band (cols 0..6)
        std::vector<CoreCoord> loaders;
        for (uint32_t i = 0; i < L; ++i) {
            loaders.push_back(find_near(CoreCoord(grid.x - 1, grid.y - 1 - i)));
        }
        for (auto& c : loaders) {
            auto lp = device->worker_core_from_logical_core(c);
            log_info(LogTest, "LOADER phys ({},{})", lp.x, lp.y);
        }

        // mcast band = logical cols 0..6, all rows (contiguous valid phys rect; loaders are in cols 7..10)
        uint32_t band_x1 = std::min<uint32_t>(6, grid.x - 1);
        auto bp0 = device->worker_core_from_logical_core(CoreCoord(0, 0));
        auto bp1 = device->worker_core_from_logical_core(CoreCoord(band_x1, grid.y - 1));
        uint32_t rx0 = std::min(bp0.x, bp1.x), rx1 = std::max(bp0.x, bp1.x);
        uint32_t ry0 = std::min(bp0.y, bp1.y), ry1 = std::max(bp0.y, bp1.y);
        uint32_t ndest = (band_x1 + 1) * grid.y;

        std::set<CoreRange> noc0_set, noc1_set, reader_all;
        for (auto& r : readers) {
            reader_all.insert(CoreRange(r.core));
            (r.noc ? noc1_set : noc0_set).insert(CoreRange(r.core));
        }
        std::set<CoreRange> loader_set;
        for (auto& c : loaders) {
            loader_set.insert(CoreRange(c));
        }

        log_info(
            LogTest,
            "in1: P={} split readers={} ({:.1f}MB) | in0: mode={} L={} Mt={} Kb={} nkb={} in0={:.3f}MB "
            "band phys ({},{})-({},{}) ndest={}",
            P,
            readers.size(),
            in1_bytes / 1048576.0,
            mode,
            L,
            Mt,
            Kb,
            nkb,
            in0_bytes / 1048576.0,
            rx0,
            ry0,
            rx1,
            ry1,
            ndest);

        tt_metal::Program program = tt_metal::Program();
        // cb_dst (in0 receive) on ALL grid cores FIRST -> uniform L1 offset everywhere.
        std::set<CoreRange> all_grid;
        for (uint32_t y = 0; y < grid.y; ++y) {
            for (uint32_t x = 0; x < grid.x; ++x) {
                all_grid.insert(CoreRange(CoreCoord(x, y)));
            }
        }
        CoreRangeSet all_cores(all_grid);
        if (do_mc) {
            tt_metal::CircularBufferConfig cbd(tiles_per_kb * tb, {{2, tt::DataFormat::Float16_b}});
            cbd.set_page_size(2, tb);
            tt_metal::CreateCircularBuffer(program, all_cores, cbd);
        }
        // in1 scratch cb0 on reader cores
        tt_metal::CircularBufferConfig cb0(3 * block_bytes, {{0, tt::DataFormat::Float16_b}});
        cb0.set_page_size(0, tb);
        tt_metal::CreateCircularBuffer(program, CoreRangeSet(reader_all), cb0);
        // in0 loader read scratch cb1 (depth 4 blocks => read/mcast overlap)
        if (mode != "none") {
            tt_metal::CircularBufferConfig cb1(4 * tiles_per_kb * tb, {{1, tt::DataFormat::Float16_b}});
            cb1.set_page_size(1, tb);
            tt_metal::CreateCircularBuffer(program, CoreRangeSet(loader_set), cb1);
        }

        // DRAM buffers
        tt_metal::distributed::DeviceLocalBufferConfig lc{.page_size = tb, .buffer_type = tt_metal::BufferType::DRAM};
        auto in1_buf = tt_metal::distributed::MeshBuffer::create(
            tt_metal::distributed::ReplicatedBufferConfig{.size = in1_bytes}, lc, device.get());
        std::shared_ptr<tt_metal::distributed::MeshBuffer> in0_buf;
        if (mode != "none") {
            uint64_t in0_alloc = contig ? 8 * in0_bytes : in0_bytes;  // contig: each bank holds a full shard
            in0_buf = tt_metal::distributed::MeshBuffer::create(
                tt_metal::distributed::ReplicatedBufferConfig{.size = in0_alloc}, lc, device.get());
        }

        // in1 reader kernels (split)
        const char* rk = "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/reader_mr.cpp";
        tt_metal::KernelHandle k0 = 0, k1 = 0;
        if (!noc0_set.empty()) {
            k0 = tt_metal::CreateKernel(
                program,
                rk,
                CoreRangeSet(noc0_set),
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = {num_blocks, num_pages, page_bytes, 0u}});
        }
        if (!noc1_set.empty()) {
            k1 = tt_metal::CreateKernel(
                program,
                rk,
                CoreRangeSet(noc1_set),
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = {num_blocks, num_pages, page_bytes, 0u}});
        }
        for (auto& r : readers) {
            tt_metal::SetRuntimeArgs(
                program, r.noc ? k1 : k0, r.core, {(uint32_t)in1_buf->address(), r.bank, r.vc, r.base_off});
        }

        // in0 loader kernels: READ stage (BRISC/NOC0) always; MCAST stage (NCRISC/chosen NoC) if do_mc.
        if (mode != "none") {
            auto rdk = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/loader_read.cpp",
                CoreRangeSet(loader_set),
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = {
                        kb_per_loader, tiles_per_kb, tb, 1u /*cb1*/, 8u /*banks*/, do_mc ? 1u : 0u, contig ? 1u : 0u}});
            for (uint32_t i = 0; i < L; ++i) {
                // interleaved: base_tile = shard offset in tile-id space. contig: read from bank i, offset 0.
                uint32_t base_tile = contig ? 0u : i * kb_per_loader * tiles_per_kb;
                tt_metal::SetRuntimeArgs(
                    program, rdk, loaders[i], {(uint32_t)(in0_buf ? in0_buf->address() : 0), base_tile, i % 8u});
            }
            if (do_mc) {
                auto noc = mc_noc1 ? tt_metal::NOC::RISCV_1_default : tt_metal::NOC::RISCV_0_default;
                auto mck = tt_metal::CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/loader_mcast.cpp",
                    CoreRangeSet(loader_set),
                    tt_metal::DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = noc,
                        .compile_args = {kb_per_loader, tiles_per_kb, tb, 1u /*cb1*/, 2u /*cb_dst*/, 2u /*md*/}});
                uint32_t sx0 = rx0, sy0 = ry0, sx1 = rx1, sy1 = ry1;
                if (mc_noc1) {
                    sx0 = rx1;
                    sy0 = ry1;
                    sx1 = rx0;
                    sy1 = ry0;
                }  // NOC1 corner swap
                for (uint32_t i = 0; i < L; ++i) {
                    tt_metal::SetRuntimeArgs(program, mck, loaders[i], {ndest, sx0, sy0, sx1, sy1});
                }
            }
        }

        auto wl = tt_metal::distributed::MeshWorkload();
        wl.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        for (uint32_t i = 0; i < num_tests; ++i) {
            tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), wl, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
        }
        tt_metal::ReadMeshDeviceProfilerResults(*device);
        log_info(LogTest, "DONE in1_bytes {} in0_bytes {}", in1_bytes, in0_bytes);
        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    log_info(LogTest, "{}", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
