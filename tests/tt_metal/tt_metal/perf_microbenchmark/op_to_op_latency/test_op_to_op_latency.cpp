// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Op-to-op latency benchmark.
//
// Background: HW team wants to measure the time between when one op's math
// finishes and the next op's math starts (op-to-op latency). That gap is
// firmware tear-down + dispatch + barrier overhead, and is what they need to
// shrink (e.g. by virtualising init registers so program N+1 can preload
// while N is still finishing, or by relaxing end-of-op barriers).
//
// This benchmark does the textbook reader -> CB -> compute -> CB -> writer
// pipeline on every Tensix core, with interleaved DRAM in/out, then runs the
// same MeshWorkload back-to-back N times in either Fast-Dispatch or Trace
// mode. The compute kernel wraps its per-tile body in
// DeviceZoneScopedMainN("MATH"), so a profiler-enabled build dumps
// MATH_begin/MATH_end timestamps per TRISC per program-run per core into
// generated/profiler/.logs/profile_log_device.csv.
//
// To get op-to-op latency:
//   op_to_op_latency_per_core = MATH_begin[program N+1] - MATH_end[program N]
//
// CLI:
//   --num-pages-per-core N  pages of interleaved DRAM per core (default 2)
//   --compute-nops N        TTI_NOPs in the per-tile body (default 0;
//                           tune up so the math zone is comfortably > 10us)
//   --num-programs N        back-to-back enqueues per measurement (default 8)
//   --use-trace             capture once + replay (default: FD mode)
//   --use-device-profiler   call ReadMeshDeviceProfilerResults after Finish
//                           (requires --enable-profiler build + env var
//                            TT_METAL_DEVICE_PROFILER=1)
//   --no-warmup             skip the warmup enqueue
//   --trace-region-size N   trace buffer size, bytes (default 1 MiB)
//   --device-id N           device under test (default 0)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/work_split.hpp>
#include <hostdevcommon/common_values.hpp>
#include "impl/context/metal_context.hpp"

#include "test_common.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Defaults:
//   num_pages_per_core = 2 matches Almeet's "each core gets 2 pages" suggestion
//   num_nops_per_tile  = 0 keeps the math zone short for now; Step 4 will tune
//                        this up to ~10us once we have profiler markers
//                        in place to actually measure the kernel runtime.
//   num_programs       = 8 enqueues back-to-back per measurement run; gives
//                        seven (N-1) op-to-op gaps to look at later.
//   warmup             = true runs one untimed enqueue first to absorb
//                        dispatcher / kernel-setup transients.
//   use_trace          = false keeps step-2 FD behaviour as default. Pass
//                        --use-trace to capture-once / replay-once.
//   trace_region_size  = 1 MiB. Plenty for tens of small back-to-back
//                        programs; tunable via --trace-region-size if
//                        --num-programs is bumped a lot.
struct BenchmarkConfig {
    uint32_t num_pages_per_core = 2;
    uint32_t num_nops_per_tile = 0;
    uint32_t num_programs = 8;
    uint32_t device_id = 0;
    bool warmup = true;
    bool use_trace = false;
    bool use_device_profiler = false;
    size_t trace_region_size = 1ull << 20;  // 1 MiB
};

BenchmarkConfig parse_args(const std::vector<std::string>& args) {
    BenchmarkConfig cfg;
    cfg.num_pages_per_core = test_args::get_command_option_uint32(args, "--num-pages-per-core", cfg.num_pages_per_core);
    cfg.num_nops_per_tile = test_args::get_command_option_uint32(args, "--compute-nops", cfg.num_nops_per_tile);
    cfg.num_programs = test_args::get_command_option_uint32(args, "--num-programs", cfg.num_programs);
    cfg.device_id = test_args::get_command_option_uint32(args, "--device-id", cfg.device_id);
    cfg.trace_region_size =
        test_args::get_command_option_uint32(args, "--trace-region-size", static_cast<uint32_t>(cfg.trace_region_size));
    if (test_args::has_command_option(args, "--no-warmup")) {
        cfg.warmup = false;
    }
    if (test_args::has_command_option(args, "--use-trace")) {
        cfg.use_trace = true;
    }
    if (test_args::has_command_option(args, "--use-device-profiler")) {
        cfg.use_device_profiler = true;
    }
    return cfg;
}

// Build the reader -> compute -> writer program covering every Tensix core,
// with the per-core tile slice assigned via runtime args.
Program build_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const BenchmarkConfig& cfg,
    const std::shared_ptr<distributed::MeshBuffer>& input_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& output_buffer,
    uint32_t total_num_tiles,
    uint32_t single_tile_size_bytes) {
    Program program = CreateProgram();

    const auto grid = mesh_device->compute_with_storage_grid_size();
    constexpr bool kRowMajor = true;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid, total_num_tiles, kRowMajor);

    log_info(
        LogTest,
        "Grid {}x{}, {} cores active, total_num_tiles={} ({} tiles/core in group_1, {} in group_2)",
        grid.x,
        grid.y,
        num_cores,
        total_num_tiles,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2);

    // CBs: c_0 is reader -> compute, c_16 is compute -> writer. Sized at 2
    // tiles each (double-buffered) so the reader/compute/writer can overlap.
    constexpr uint32_t kInputCbId = tt::CBIndex::c_0;
    constexpr uint32_t kOutputCbId = tt::CBIndex::c_16;
    constexpr uint32_t kCbDepthInTiles = 2;
    constexpr auto kDataFormat = tt::DataFormat::Float16_b;

    CircularBufferConfig cb_in_config =
        CircularBufferConfig(kCbDepthInTiles * single_tile_size_bytes, {{kInputCbId, kDataFormat}})
            .set_page_size(kInputCbId, single_tile_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_in_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(kCbDepthInTiles * single_tile_size_bytes, {{kOutputCbId, kDataFormat}})
            .set_page_size(kOutputCbId, single_tile_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_out_config);

    // Reader: NOC1 / RISCV1, pulls from interleaved DRAM via TensorAccessor.
    std::vector<uint32_t> reader_compile_time_args = {kInputCbId};
    TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/kernels/reader_interleaved.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    // Writer: NOC0 / RISCV0, pushes to interleaved DRAM.
    std::vector<uint32_t> writer_compile_time_args = {kOutputCbId};
    TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);
    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/kernels/writer_interleaved.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compute: copy_tile + tunable NOP spin.
    std::vector<uint32_t> compute_compile_time_args = {kInputCbId, kOutputCbId, cfg.num_nops_per_tile};
    auto compute_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/kernels/compute_copy_with_nops.cpp",
        all_cores,
        ComputeConfig{.compile_args = compute_compile_time_args});

    // Per-core runtime args: each core gets [start_tile_id, start_tile_id + n_tiles).
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};
    uint32_t start_tile_id = 0;
    for (const auto& [group, tiles_per_core] : work_groups) {
        if (tiles_per_core == 0) {
            continue;
        }
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                SetRuntimeArgs(program, reader_kernel, core, {input_buffer->address(), tiles_per_core, start_tile_id});
                SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address(), tiles_per_core, start_tile_id});
                SetRuntimeArgs(program, compute_kernel, core, {tiles_per_core});
                start_tile_id += tiles_per_core;
            }
        }
    }
    TT_FATAL(
        start_tile_id == total_num_tiles,
        "Tile distribution mismatch: assigned {} but expected {}",
        start_tile_id,
        total_num_tiles);

    return program;
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const std::vector<std::string> args(argv, argv + argc);
        const BenchmarkConfig cfg = parse_args(args);

        log_info(
            LogTest,
            "op_to_op_latency: device_id={}, pages_per_core={}, compute_nops={}, num_programs={}, warmup={}, "
            "use_trace={}, use_device_profiler={}, trace_region_size={}",
            cfg.device_id,
            cfg.num_pages_per_core,
            cfg.num_nops_per_tile,
            cfg.num_programs,
            cfg.warmup,
            cfg.use_trace,
            cfg.use_device_profiler,
            cfg.trace_region_size);
        TT_FATAL(cfg.num_programs >= 1, "--num-programs must be >= 1");

        // If the user asked for a CSV dump, make sure the runtime profiler is
        // actually enabled (--enable-profiler build + TT_METAL_DEVICE_PROFILER=1
        // env var). Without this check, the CSV would just be empty and the
        // failure mode would be silent. Pattern mirrors test_dram_offchip.cpp.
        if (cfg.use_device_profiler) {
            const bool profiler_runtime_enabled =
                tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_enabled();
            TT_FATAL(
                profiler_runtime_enabled,
                "--use-device-profiler requires the device profiler to be enabled at runtime. "
                "Build with `./build_metal.sh --enable-profiler` and set TT_METAL_DEVICE_PROFILER=1 "
                "(e.g. `TT_METAL_DEVICE_PROFILER=1 ./test_op_to_op_latency --use-device-profiler ...`).");
        }

        // trace_region_size = 0 would disable trace capture entirely (legacy
        // path) -- in trace mode we always want a non-zero value. The default
        // is 1 MiB which covers tens of small programs comfortably.
        const size_t trace_region_size = cfg.use_trace ? cfg.trace_region_size : 0;
        auto mesh_device =
            distributed::MeshDevice::create_unit_mesh(cfg.device_id, DEFAULT_L1_SMALL_SIZE, trace_region_size);

        const auto grid = mesh_device->compute_with_storage_grid_size();
        const uint32_t num_cores = grid.x * grid.y;
        const uint32_t total_num_tiles = num_cores * cfg.num_pages_per_core;

        constexpr auto kDataFormat = tt::DataFormat::Float16_b;
        const uint32_t single_tile_size_bytes = tile_size(kDataFormat);
        const uint32_t buffer_size_bytes = total_num_tiles * single_tile_size_bytes;

        // Interleaved DRAM buffer: page_size == one tile, so pages are spread
        // across DRAM banks round-robin. This is what makes the read/write
        // pattern look like a real op.
        distributed::DeviceLocalBufferConfig dram_local_config{
            .page_size = single_tile_size_bytes,
            .buffer_type = BufferType::DRAM,
        };
        distributed::ReplicatedBufferConfig dram_buf_config{.size = buffer_size_bytes};

        auto input_buffer = distributed::MeshBuffer::create(dram_buf_config, dram_local_config, mesh_device.get());
        auto output_buffer = distributed::MeshBuffer::create(dram_buf_config, dram_local_config, mesh_device.get());

        // Random bf16 input data, packed two values per uint32_t.
        const auto seed = static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<uint32_t> input_data = create_random_vector_of_bfloat16(buffer_size_bytes, /*rand_max=*/100, seed);

        auto& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data, /*blocking=*/false);

        Program program =
            build_program(mesh_device, cfg, input_buffer, output_buffer, total_num_tiles, single_tile_size_bytes);

        distributed::MeshWorkload workload;
        const distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        // Warmup (untimed): one eager enqueue absorbs the JIT-load /
        // dispatcher prefetch cost so they do not pollute either the FD timing
        // loop or the trace-capture step.
        if (cfg.warmup) {
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
            distributed::Finish(cq);
        }

        constexpr uint8_t kCqId = 0;
        long long elapsed_us = 0;
        const char* mode_label = nullptr;

        if (cfg.use_trace) {
            // Step 3: capture one trace containing all N back-to-back enqueues,
            // then time a single replay + Finish. Capture itself is untimed --
            // it's a one-time setup cost that is amortised when the same
            // trace is replayed many times in real workloads.
            //
            // Note: BeginTraceCapture is the only trace API exposed as a free
            // distributed:: function; end / replay / release are MeshDevice
            // member methods.
            const distributed::MeshTraceId tid = distributed::BeginTraceCapture(mesh_device.get(), kCqId);
            for (uint32_t i = 0; i < cfg.num_programs; ++i) {
                distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
            }
            mesh_device->end_mesh_trace(kCqId, tid);
            distributed::Finish(cq);  // make sure capture is fully committed before timing

            const auto t_begin = std::chrono::steady_clock::now();
            mesh_device->replay_mesh_trace(kCqId, tid, /*blocking=*/false);
            distributed::Finish(cq);
            const auto t_end = std::chrono::steady_clock::now();
            elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();

            mesh_device->release_mesh_trace(tid);
            mode_label = "Trace replay";
        } else {
            // Step 2: enqueue the same MeshWorkload back-to-back num_programs
            // times under one Finish.
            const auto t_begin = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < cfg.num_programs; ++i) {
                distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
            }
            distributed::Finish(cq);
            const auto t_end = std::chrono::steady_clock::now();
            elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
            mode_label = "FD back-to-back";
        }

        const double avg_us_per_program = static_cast<double>(elapsed_us) / cfg.num_programs;
        log_info(
            LogTest,
            "{}: {} programs in {} us (avg {:.2f} us/program)",
            mode_label,
            cfg.num_programs,
            elapsed_us,
            avg_us_per_program);

        // Step 4: flush device profiler buffers to
        // generated/profiler/.logs/profile_log_device.csv. Filter the rows
        // for zone "MATH" and compute MATH_begin[N+1] - MATH_end[N] per core
        // to get the on-chip op-to-op latency. tools/tracy/process_device_log.py
        // can post-process the CSV.
        if (cfg.use_device_profiler) {
            tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh_device);
            log_info(
                LogTest,
                "Device profiler results dumped. Inspect "
                "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv "
                "and look for the 'MATH' zone to compute op-to-op latency.");
        }

        std::vector<uint32_t> output_data;
        distributed::EnqueueReadMeshBuffer(cq, output_data, output_buffer, /*blocking=*/true);

        if (output_data.size() != input_data.size()) {
            log_error(
                LogTest, "Output size mismatch: got {} elems, expected {}", output_data.size(), input_data.size());
            pass = false;
        } else {
            const bool match = std::equal(input_data.begin(), input_data.end(), output_data.begin());
            if (!match) {
                size_t mismatch_count = 0;
                size_t first_mismatch = 0;
                for (size_t i = 0; i < input_data.size(); ++i) {
                    if (input_data[i] != output_data[i]) {
                        if (mismatch_count == 0) {
                            first_mismatch = i;
                        }
                        ++mismatch_count;
                    }
                }
                log_error(
                    LogTest,
                    "Output mismatch: {}/{} elements differ; first mismatch at index {} "
                    "(in=0x{:08x}, out=0x{:08x})",
                    mismatch_count,
                    input_data.size(),
                    first_mismatch,
                    input_data[first_mismatch],
                    output_data[first_mismatch]);
                pass = false;
            }
        }

        mesh_device->close();
    } catch (const std::exception& e) {
        log_error(LogTest, "Caught exception: {}", e.what());
        pass = false;
    }

    if (pass) {
        log_info(LogTest, "op_to_op_latency: PASSED");
        return 0;
    }
    log_error(LogTest, "op_to_op_latency: FAILED");
    return 1;
}
