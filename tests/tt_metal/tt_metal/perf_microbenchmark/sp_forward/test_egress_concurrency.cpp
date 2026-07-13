// SPDX-License-Identifier: Apache-2.0
// Isolate NOC1 egress under concurrency, with NO DRAM read. nsrc source cores each drive a logical stream
// to K receivers (mcast) or 1 receiver (unicast) on NCRISC/NOC1. Each source uses its own logical column
// (x=s): receivers rows 0..K-1, source row K -> contiguous valid phys rectangle, source outside it.
// Aggregate egress = nsrc*tiles*tile_bytes / max-core kernel time. Compares to the single-source ceiling
// (Exp1) to separate NOC1-fabric contention from the reader/L1 coupling seen in the read-through test.
//
// Usage: TT_METAL_DEVICE_PROFILER=1 ./test_egress_concurrency --nsrc 8 --k 2 --chunk 8 --depth 8 --mode mcast
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
    uint32_t nsrc = 8, K = 2, chunk = 8, depth = 8, tiles = 16384, num_tests = 6;
    std::string mode = "mcast";
    std::vector<std::string> a(argv, argv + argc);
    std::tie(nsrc, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--nsrc", nsrc);
    std::tie(K, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--k", K);
    std::tie(chunk, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--chunk", chunk);
    std::tie(depth, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--depth", depth);
    std::tie(tiles, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--tiles", tiles);
    std::tie(num_tests, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--num-tests", num_tests);
    std::tie(mode, a) = test_args::get_command_option_and_remaining_args(a, "--mode", mode);

    try {
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        uint32_t tb = tt::tile_size(tt::DataFormat::Float16_b);
        auto grid = device->compute_with_storage_grid_size();
        bool uni = (mode == "unicast");
        uint32_t Kr = uni ? 1 : K;  // receivers per source
        TT_FATAL(nsrc <= grid.x, "nsrc {} > grid.x {}", nsrc, grid.x);
        TT_FATAL(Kr + 1 <= grid.y, "Kr+1 {} > grid.y {}", Kr + 1, grid.y);

        std::vector<CoreCoord> srcs;
        std::vector<std::vector<CoreCoord>> recv(nsrc);
        std::set<CoreRange> src_set, recv_set, union_set;
        for (uint32_t s = 0; s < nsrc; ++s) {
            CoreCoord sc(s, Kr);  // source at row Kr, just below the receiver column
            srcs.push_back(sc);
            src_set.insert(CoreRange(sc));
            union_set.insert(CoreRange(sc));
            for (uint32_t r = 0; r < Kr; ++r) {
                CoreCoord rc(s, r);
                recv[s].push_back(rc);
                recv_set.insert(CoreRange(rc));
                union_set.insert(CoreRange(rc));
            }
        }
        CoreRangeSet src_cores(src_set), recv_cores(recv_set), union_cores(union_set);
        log_info(LogTest, "mode {} nsrc {} Kr {} chunk {} depth {} tiles {}", mode, nsrc, Kr, chunk, depth, tiles);

        tt_metal::Program program = tt_metal::Program();
        tt_metal::CircularBufferConfig cbs(chunk * tb, {{0, tt::DataFormat::Float16_b}});
        cbs.set_page_size(0, tb);
        tt_metal::CreateCircularBuffer(program, src_cores, cbs);
        tt_metal::CircularBufferConfig cbd(chunk * tb, {{1, tt::DataFormat::Float16_b}});
        cbd.set_page_size(1, tb);
        tt_metal::CreateCircularBuffer(program, union_cores, cbd);

        const char* kpath = uni ? "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/unicast_egress.cpp"
                                : "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/mcast_egress.cpp";
        auto k = tt_metal::CreateKernel(
            program,
            kpath,
            src_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {tb, 0u, 1u}});
        for (uint32_t s = 0; s < nsrc; ++s) {
            if (uni) {
                auto p = device->worker_core_from_logical_core(recv[s][0]);
                tt_metal::SetRuntimeArgs(program, k, srcs[s], {p.x, p.y, tiles, chunk, depth});
            } else {
                auto p0 = device->worker_core_from_logical_core(recv[s][0]);
                auto p1 = device->worker_core_from_logical_core(recv[s][Kr - 1]);
                uint32_t x0 = std::min(p0.x, p1.x), x1 = std::max(p0.x, p1.x);
                uint32_t y0 = std::min(p0.y, p1.y), y1 = std::max(p0.y, p1.y);
                // NOC1: pass swapped corners (start=max,end=min)
                tt_metal::SetRuntimeArgs(program, k, srcs[s], {Kr, x1, y1, x0, y0, tiles, chunk, depth});
            }
        }
        if (recv_cores.num_cores() > 0) {
            tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/noop.cpp",
                recv_cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
        }

        auto wl = tt_metal::distributed::MeshWorkload();
        wl.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        for (uint32_t i = 0; i < num_tests; ++i) {
            tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), wl, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
        }
        tt_metal::ReadMeshDeviceProfilerResults(*device);
        uint64_t total_bytes = (uint64_t)nsrc * tiles * tb;
        log_info(LogTest, "DONE total_bytes {}", total_bytes);
        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    log_info(LogTest, "{}", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
