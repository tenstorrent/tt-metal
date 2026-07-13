// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SP5: measure DRAM read BW for the Regime-B pattern -- many compute cores each read a
// CONTIGUOUS range of tile ids from a DRAM-INTERLEAVED bf16 tensor (reader==consumer,
// no forwarding, no compute). Reports KERNEL-TIME BW via the device profiler; wall-clock
// is unreliable (see SP1). Sweep --depth to find the interleaved saturation ceiling.
//
// Usage:
//   TT_METAL_DEVICE_PROFILER=1 ./test_interleaved_read --input-size <bytes> --depth <tiles>
//       --num-cores <n> --num-tests <n>

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <vector>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>
#include "test_common.hpp"

using namespace tt;

int main(int argc, char** argv) {
    bool pass = true;
    uint64_t input_size = 256ull * 1024 * 1024;
    uint32_t depth = 64;
    uint32_t num_tests = 6;
    uint32_t req_cores = 110;
    bool dual_risc = false;
    uint32_t gx = 0, gy = 0;                           // rectangular compute grid (overrides --num-cores when both > 0)
    uint32_t wr_tiles = 0;                             // SP4: per-core output-write tiles
    uint32_t mc_tiles = 0, mc_reps = 1, mc_chunk = 8;  // SP3: multicast small-operand tiles/reps/chunk

    std::vector<std::string> input_args(argv, argv + argc);
    std::tie(input_size, input_args) =
        test_args::get_command_option_uint64_and_remaining_args(input_args, "--input-size", input_size);
    std::tie(depth, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--depth", depth);
    std::tie(num_tests, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", num_tests);
    std::tie(req_cores, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-cores", req_cores);
    std::tie(dual_risc, input_args) = test_args::has_command_option_and_remaining_args(input_args, "--dual-risc");
    std::tie(gx, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--gx", gx);
    std::tie(gy, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--gy", gy);
    std::tie(wr_tiles, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--wr-tiles", wr_tiles);
    std::tie(mc_tiles, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--mc-tiles", mc_tiles);
    std::tie(mc_reps, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--mc-reps", mc_reps);
    std::tie(mc_chunk, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--mc-chunk", mc_chunk);

    try {
        int device_id = 0;
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

        tt::DataFormat tile_format = tt::DataFormat::Float16_b;
        uint32_t tile_bytes = tt::tile_size(tile_format);  // bf16 tile = 2048 B

        auto grid = device->compute_with_storage_grid_size();
        uint32_t max_cores = grid.x * grid.y;
        bool rect = (gx > 0 && gy > 0);
        uint32_t num_cores = rect ? gx * gy : std::min(req_cores, max_cores);

        // tiles, padded so every core gets an equal contiguous range
        uint64_t num_tiles = (input_size + tile_bytes - 1) / tile_bytes;
        uint32_t tiles_per_core = (num_tiles + num_cores - 1) / num_cores;
        num_tiles = static_cast<uint64_t>(tiles_per_core) * num_cores;
        uint64_t buf_bytes = num_tiles * tile_bytes;

        // build the list of compute (reader) cores
        std::vector<CoreCoord> cores;
        std::set<CoreRange> core_set;
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord c = rect ? CoreCoord(i % gx, i / gx) : CoreCoord(i % grid.x, i / grid.x);
            cores.push_back(c);
            core_set.insert(CoreRange(c));
        }
        CoreRangeSet all_cores(core_set);

        // For mcast, the source core sits just outside the compute rect. CBs shared with the mcast
        // (reader cb0, mcast-dst cb2) must be allocated on source+compute with an IDENTICAL layout so
        // the dst L1 offset matches across cores. Build a union set including the source.
        CoreCoord mc_src_logical = (gx < grid.x) ? CoreCoord(gx, 0) : CoreCoord(0, gy);
        std::set<CoreRange> cb_set = core_set;
        if (mc_tiles > 0) {
            cb_set.insert(CoreRange(mc_src_logical, mc_src_logical));
        }
        CoreRangeSet cb_cores(cb_set);

        log_info(
            LogTest,
            "input {:.1f} MB, {} tiles, {} cores, {} tiles/core ({:.2f} MB/core), depth {}",
            buf_bytes / 1048576.0,
            num_tiles,
            num_cores,
            tiles_per_core,
            tiles_per_core * tile_bytes / 1048576.0,
            depth);

        // interleaved DRAM buffer, page = tile
        tt_metal::distributed::DeviceLocalBufferConfig local{
            .page_size = tile_bytes, .buffer_type = tt_metal::BufferType::DRAM};
        tt_metal::distributed::ReplicatedBufferConfig global{.size = buf_bytes};
        auto input_buffer = tt_metal::distributed::MeshBuffer::create(global, local, device.get());
        std::vector<uint32_t> input_vec(buf_bytes / sizeof(uint32_t), 0x3c003c00);  // ~1.0 bf16
        tt_metal::distributed::EnqueueWriteMeshBuffer(device->mesh_command_queue(), input_buffer, input_vec, false);
        tt_metal::distributed::Finish(device->mesh_command_queue());

        // program
        tt_metal::Program program = tt_metal::Program();
        const char* kpath =
            "tests/tt_metal/tt_metal/perf_microbenchmark/sp5_interleaved_read/kernels/reader_interleaved.cpp";
        uint32_t cb_tiles = 2 * depth;  // double-buffered scratch, per RISC

        auto make_ct = [&](uint32_t cb_id) {
            std::vector<uint32_t> ct = {depth, tile_bytes, cb_id};
            tt_metal::TensorAccessorArgs(*input_buffer->get_reference_buffer()).append_to(ct);
            return ct;
        };

        // RISC0 / NOC0 (allocate on cb_cores so cb0/cb2 offsets match on the mcast source too)
        tt_metal::CircularBufferConfig cb0(cb_tiles * tile_bytes, {{0, tile_format}});
        cb0.set_page_size(0, tile_bytes);
        tt_metal::CreateCircularBuffer(program, cb_cores, cb0);
        auto reader0 = tt_metal::CreateKernel(
            program,
            kpath,
            all_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = make_ct(0)});

        tt_metal::KernelHandle reader1 = 0;
        if (dual_risc) {
            tt_metal::CircularBufferConfig cb1(cb_tiles * tile_bytes, {{1, tile_format}});
            cb1.set_page_size(1, tile_bytes);
            tt_metal::CreateCircularBuffer(program, all_cores, cb1);
            reader1 = tt_metal::CreateKernel(
                program,
                kpath,
                all_cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = make_ct(1)});
        }

        uint32_t half = dual_risc ? tiles_per_core / 2 : tiles_per_core;
        for (uint32_t i = 0; i < num_cores; ++i) {
            uint32_t start_tile = i * tiles_per_core;
            const std::array<uint32_t, 3> rt0 = {(uint32_t)input_buffer->address(), start_tile, half};
            tt_metal::SetRuntimeArgs(program, reader0, cores[i], rt0);
            if (dual_risc) {
                const std::array<uint32_t, 3> rt1 = {
                    (uint32_t)input_buffer->address(), start_tile + half, tiles_per_core - half};
                tt_metal::SetRuntimeArgs(program, reader1, cores[i], rt1);
            }
        }

        // SP4: interleaved output write on NCRISC (all compute cores), concurrent with the read.
        if (wr_tiles > 0) {
            uint64_t out_tiles = (uint64_t)num_cores * wr_tiles;
            tt_metal::distributed::DeviceLocalBufferConfig olocal{
                .page_size = tile_bytes, .buffer_type = tt_metal::BufferType::DRAM};
            tt_metal::distributed::ReplicatedBufferConfig oglobal{.size = out_tiles * tile_bytes};
            auto out_buffer = tt_metal::distributed::MeshBuffer::create(oglobal, olocal, device.get());
            // scratch src CB (cb 1)
            tt_metal::CircularBufferConfig cbw(depth * tile_bytes, {{1, tile_format}});
            cbw.set_page_size(1, tile_bytes);
            tt_metal::CreateCircularBuffer(program, all_cores, cbw);
            std::vector<uint32_t> wct = {wr_tiles, tile_bytes, depth, 1u};
            tt_metal::TensorAccessorArgs(*out_buffer->get_reference_buffer()).append_to(wct);
            auto writer = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/sp5_interleaved_read/kernels/writer_interleaved.cpp",
                all_cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = wct});
            for (uint32_t i = 0; i < num_cores; ++i) {
                const std::array<uint32_t, 2> rt = {(uint32_t)out_buffer->address(), i * wr_tiles};
                tt_metal::SetRuntimeArgs(program, writer, cores[i], rt);
            }
            log_info(LogTest, "SP4 write: {} tiles/core, out {:.2f} MB", wr_tiles, out_tiles * tile_bytes / 1048576.0);
        }

        // SP3: multicast the small operand from a source core (outside the compute rect) to all
        // compute cores' L1, on NCRISC, concurrent with the read.
        if (mc_tiles > 0) {
            CoreCoord src_logical = mc_src_logical;
            CoreRangeSet src_set(CoreRange(src_logical, src_logical));
            // mcast-dst CB (cb 2) on the union (source+compute) so its L1 offset is identical
            // everywhere; source's src scratch CB (cb 3) on source only.
            tt_metal::CircularBufferConfig cbd(mc_chunk * tile_bytes, {{2, tile_format}});
            cbd.set_page_size(2, tile_bytes);
            tt_metal::CreateCircularBuffer(program, cb_cores, cbd);
            tt_metal::CircularBufferConfig cbs(mc_chunk * tile_bytes, {{3, tile_format}});
            cbs.set_page_size(3, tile_bytes);
            tt_metal::CreateCircularBuffer(program, src_set, cbs);
            auto mcast = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/sp5_interleaved_read/kernels/mcast_src.cpp",
                src_set,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = {mc_tiles, tile_bytes, mc_reps, mc_chunk, 3u, 2u}});
            auto c0 = device->worker_core_from_logical_core(CoreCoord(0, 0));
            auto c1 = device->worker_core_from_logical_core(CoreCoord(gx - 1, gy - 1));
            uint32_t x0 = std::min(c0.x, c1.x), x1 = std::max(c0.x, c1.x);
            uint32_t y0 = std::min(c0.y, c1.y), y1 = std::max(c0.y, c1.y);
            const std::array<uint32_t, 5> rt = {num_cores, x0, y0, x1, y1};
            tt_metal::SetRuntimeArgs(program, mcast, src_logical, rt);
            log_info(
                LogTest,
                "SP3 mcast: {} tiles x{} reps to rect ({},{})-({},{}) ndest {}",
                mc_tiles,
                mc_reps,
                x0,
                y0,
                x1,
                y1,
                num_cores);
        }

        auto workload = tt_metal::distributed::MeshWorkload();
        workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));

        for (uint32_t i = 0; i < num_tests; ++i) {
            tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
        }
        tt_metal::ReadMeshDeviceProfilerResults(*device);

        log_info(LogTest, "DONE total_bytes {}", buf_bytes);
        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    log_info(LogTest, "{}", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
