// SPDX-License-Identifier: Apache-2.0
// Read-only BW probe for bank-sharded reads. P readers per DRAM bank (bank-adjacent placement),
// each reads its share of the bank in contiguous (K-slice) or strided (N-sub) mode, sweepable depth.
// Emulates an in1 shard of Kt x N_band tiles per bank. Reports kernel-time BW (max BRISC across cores).
//
// Usage: TT_METAL_DEVICE_PROFILER=1 ./test_bank_read --kt 192 --nband 18 --preaders 2
//        --mode strided|contig --depth 8 [--rtiles 8]
#include <cstdint>
#include <exception>
#include <string>
#include <vector>
#include <set>

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
    uint32_t Kt = 192, N_band = 18, preaders = 2, depth = 8, rtiles = 8, num_tests = 6;
    std::string mode = "strided";
    std::vector<std::string> a(argv, argv + argc);
    std::tie(Kt, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--kt", Kt);
    std::tie(N_band, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nband", N_band);
    std::tie(preaders, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--preaders", preaders);
    std::tie(depth, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--depth", depth);
    std::tie(rtiles, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--rtiles", rtiles);
    std::tie(num_tests, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--num-tests", num_tests);
    std::tie(mode, a) = test_args::get_command_option_and_remaining_args(a, "--mode", mode);

    try {
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        uint32_t tb = tt::tile_size(tt::DataFormat::Float16_b);  // 2048
        auto grid = device->compute_with_storage_grid_size();
        uint32_t P = preaders, C = 8 * P;
        TT_FATAL(N_band % P == 0, "N_band {} must divide P {}", N_band, P);
        TT_FATAL((Kt * N_band) % P == 0, "Kt*N_band must divide P");
        uint32_t ns = N_band / P;

        // per-reader read plan
        uint32_t read_bytes, stride_bytes, nblocks;
        if (mode == "contig") {
            // K-slice: contiguous (Kt*N_band/P) tiles, read in rtiles-tile chunks
            uint32_t per_reader_tiles = Kt * N_band / P;
            TT_FATAL(per_reader_tiles % rtiles == 0, "per_reader_tiles {} % rtiles {}", per_reader_tiles, rtiles);
            read_bytes = rtiles * tb;
            stride_bytes = read_bytes;
            nblocks = per_reader_tiles / rtiles;
        } else {  // strided N-sub: ns tiles/row, stride N_band, Kt rows
            read_bytes = ns * tb;
            stride_bytes = N_band * tb;
            nblocks = Kt;
        }
        uint64_t per_reader_bytes = (uint64_t)nblocks * read_bytes;
        uint64_t total_bytes = per_reader_bytes * C;
        uint64_t bank_bytes = (uint64_t)Kt * N_band * tb;  // one bank's shard

        log_info(
            LogTest,
            "mode {} Kt {} N_band {} P {} (C {}) ns {} | read_bytes {} stride {} nblocks {} depth {} | "
            "per_reader {:.2f}MB total {:.1f}MB",
            mode,
            Kt,
            N_band,
            P,
            C,
            ns,
            read_bytes,
            stride_bytes,
            nblocks,
            depth,
            per_reader_bytes / 1048576.0,
            total_bytes / 1048576.0);

        // DRAM buffer sized to one bank's shard * 8 banks (interleaved page=tile). Each reader reads
        // within its bank; base_off keeps it inside [0, bank_bytes).
        tt_metal::distributed::DeviceLocalBufferConfig lc{.page_size = tb, .buffer_type = tt_metal::BufferType::DRAM};
        tt_metal::distributed::ReplicatedBufferConfig gc{.size = bank_bytes * 8};
        auto buf = tt_metal::distributed::MeshBuffer::create(gc, lc, device.get());

        // bank-adjacent placement: P cores clustered near each bank's optimal core
        auto opt = device->get_optimal_dram_bank_to_logical_worker_assignment(tt_metal::NOC::NOC_0);
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
        std::vector<CoreCoord> cores;
        std::vector<uint32_t> cbank, cbase;
        std::set<CoreRange> cs;
        for (uint32_t b = 0; b < 8; ++b) {
            for (uint32_t p = 0; p < P; ++p) {
                CoreCoord c = find_near(opt[b]);
                cores.push_back(c);
                cbank.push_back(b);
                cbase.push_back(mode == "contig" ? p * (Kt * N_band / P) * tb : p * ns * tb);
                cs.insert(CoreRange(c, c));
            }
        }
        CoreRangeSet all_cores(cs);

        tt_metal::Program program = tt_metal::Program();
        uint32_t scratch = 2 * depth * read_bytes;
        tt_metal::CircularBufferConfig cbc(scratch, {{0, tt::DataFormat::Float16_b}});
        cbc.set_page_size(0, read_bytes);
        tt_metal::CreateCircularBuffer(program, all_cores, cbc);
        auto k = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp_bankread/kernels/bank_reader.cpp",
            all_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = {read_bytes, stride_bytes, nblocks, depth}});
        for (uint32_t i = 0; i < C; ++i) {
            tt_metal::SetRuntimeArgs(program, k, cores[i], {(uint32_t)buf->address(), cbase[i], cbank[i]});
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
