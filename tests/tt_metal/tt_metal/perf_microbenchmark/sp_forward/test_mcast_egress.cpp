// SPDX-License-Identifier: Apache-2.0
// EXP1: per-core L1->L1 multicast egress ceiling. One source core multicasts a logical stream to K
// receiver cores (contiguous valid worker rectangle) on NCRISC/NOC1. Measures NCRISC kernel-time.
// Egress BW = total_tiles*tile_bytes / time (counted once). Question: can one core sustain >= ~62 GB/s?
//
// Usage: TT_METAL_DEVICE_PROFILER=1 ./test_mcast_egress --k 4 --chunk 8 --depth 4 --tiles 16384
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
    uint32_t K = 4, chunk = 8, depth = 4, total_tiles = 16384, num_tests = 6;
    std::vector<std::string> a(argv, argv + argc);
    std::tie(K, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--k", K);
    std::tie(chunk, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--chunk", chunk);
    std::tie(depth, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--depth", depth);
    std::tie(total_tiles, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--tiles", total_tiles);
    std::tie(num_tests, a) = test_args::get_command_option_uint32_and_remaining_args(a, "--num-tests", num_tests);

    try {
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        uint32_t tb = tt::tile_size(tt::DataFormat::Float16_b);  // 2048
        auto grid = device->compute_with_storage_grid_size();
        TT_FATAL(K >= 1 && K <= grid.y - 1, "K {} must be in [1, {}]", K, grid.y - 1);

        // Receivers: vertical column at logical x=0, rows 0..K-1 -> contiguous phys rectangle.
        // Source at logical (0, K): same column, just below rect -> outside it, valid worker.
        std::vector<CoreCoord> recv;
        std::set<CoreRange> recv_set;
        for (uint32_t r = 0; r < K; ++r) {
            recv.push_back(CoreCoord(0, r));
            recv_set.insert(CoreRange(CoreCoord(0, r)));
        }
        CoreCoord src_logical(0, K);
        CoreRangeSet recv_cores(recv_set);
        CoreRangeSet src_set(CoreRange(src_logical, src_logical));
        std::set<CoreRange> union_set = recv_set;
        union_set.insert(CoreRange(src_logical, src_logical));
        CoreRangeSet union_cores(union_set);

        // physical mcast rectangle corners
        auto p0 = device->worker_core_from_logical_core(CoreCoord(0, 0));
        auto p1 = device->worker_core_from_logical_core(CoreCoord(0, K - 1));
        uint32_t x0 = std::min(p0.x, p1.x), x1 = std::max(p0.x, p1.x);
        uint32_t y0 = std::min(p0.y, p1.y), y1 = std::max(p0.y, p1.y);
        log_info(
            LogTest,
            "K {} chunk {} depth {} total_tiles {} | mcast rect phys ({},{})-({},{})",
            K,
            chunk,
            depth,
            total_tiles,
            x0,
            y0,
            x1,
            y1);

        tt_metal::Program program = tt_metal::Program();
        // cb_src (data to send) on source only; cb_dst on union so L1 offset matches everywhere.
        uint32_t buf_tiles = chunk;  // single chunk scratch is enough (fire-and-forget)
        tt_metal::CircularBufferConfig cbs(buf_tiles * tb, {{0, tt::DataFormat::Float16_b}});
        cbs.set_page_size(0, tb);
        tt_metal::CreateCircularBuffer(program, src_set, cbs);
        tt_metal::CircularBufferConfig cbd(buf_tiles * tb, {{1, tt::DataFormat::Float16_b}});
        cbd.set_page_size(1, tb);
        tt_metal::CreateCircularBuffer(program, union_cores, cbd);

        auto mcast = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/mcast_egress.cpp",
            src_set,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {tb, 0u, 1u}});
        // mcast runs on NOC1 -> pass corners SWAPPED (start=max, end=min); DYNAMIC_NOC flips them into
        // valid ascending order in the NOC1 frame (single-core rects work either way, multi-core hang if not).
        tt_metal::SetRuntimeArgs(program, mcast, src_logical, {K, x1, y1, x0, y0, total_tiles, chunk, depth});

        // receiver noop kernels (so cores are in the program)
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp_forward/kernels/noop.cpp",
            recv_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        auto wl = tt_metal::distributed::MeshWorkload();
        wl.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        for (uint32_t i = 0; i < num_tests; ++i) {
            tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), wl, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
        }
        tt_metal::ReadMeshDeviceProfilerResults(*device);
        uint64_t total_bytes = (uint64_t)total_tiles * tb;
        log_info(LogTest, "DONE total_bytes {}", total_bytes);
        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    log_info(LogTest, "{}", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
