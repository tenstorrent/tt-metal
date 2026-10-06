// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

struct RunReport {
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    uint32_t active_cores = 0;
    std::string tiles_per_core;
    double elapsed_s = 0.0;
    double tflops = 0.0;
    double per_iter_ms = 0.0;
    std::string start_time;
    std::string end_time;
};

struct GridCandidate {
    uint32_t x = 0;
    uint32_t y = 0;
};

static std::string format_time(std::chrono::system_clock::time_point tp) {
    auto t = std::chrono::system_clock::to_time_t(tp);
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()) %
        std::chrono::seconds(1);

    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif

    char buffer[64];
    std::snprintf(
        buffer,
        sizeof(buffer),
        "%04d-%02d-%02d %02d:%02d:%02d.%06lld",
        tm_buf.tm_year + 1900,
        tm_buf.tm_mon + 1,
        tm_buf.tm_mday,
        tm_buf.tm_hour,
        tm_buf.tm_min,
        tm_buf.tm_sec,
        static_cast<long long>(us.count()));

    return std::string(buffer);
}

// Lowest grid width swept. Below this the core count is too small for the split to be
// meaningful, and the first interval's idle baseline is one-sided anyway.
static constexpr uint32_t kMinGridX = 3;

// The sweep walks the near-square band of the core array: for each width x, it runs the three
// heights x-1, x and x+1, clipped to the device. That gives a dense ascending core-count series
// while also pairing grids of equal core count but different aspect ratio (3x4 vs 4x3, 6x7 vs
// 7x6, 8x7 vs 7x8), which is what separates a core-count effect from a NoC-shape one.
//
// Generated rather than listed so the band continues to whatever the device provides instead of
// stopping where the smaller generation did. On a Wormhole n300 (8x7) this reproduces the
// original fifteen-grid list exactly; on Blackhole's 11x10 it carries the same three-per-width
// pattern up to the full array, where a hard-coded list previously jumped along the x,x-1
// diagonal alone (7x7 -> 8x7 -> 9x8 -> 10x9 -> 11x10) and skipped every grid beside it.
static std::vector<GridCandidate> build_test_grids(uint32_t max_x, uint32_t max_y) {
    std::vector<GridCandidate> result;

    for (uint32_t x = kMinGridX; x <= max_x; ++x) {
        for (uint32_t y = x - 1; y <= x + 1; ++y) {
            if (y >= 2 && y <= max_y) {
                result.push_back({x, y});
            }
        }
    }

    const GridCandidate max_grid{max_x, max_y};
    const bool max_grid_already_present = std::any_of(
        result.begin(),
        result.end(),
        [&](const GridCandidate& g) {
            return g.x == max_grid.x && g.y == max_grid.y;
        });

    if (!max_grid_already_present) {
        result.push_back(max_grid);
    }

    return result;
}

// Power-experiment configuration. Each kernel can be switched to "idle" mode, in which it still
// performs its full circular-buffer handshake (so the other kernels never deadlock) but skips its
// real NoC transfer / FPU work. This isolates which part of the pipeline actually costs power,
// versus merely keeping a core alive and cycling its CBs.
struct PowerExperimentConfig {
    bool disable_reader = false;
    bool disable_compute = false;
    bool disable_writer = false;
    uint32_t write_amplification_pct = 0;
    // Which per-tile instruction the compute kernel runs. The reader and writer are identical
    // for every value, so this varies the math unit's work against fixed data movement -- see
    // kernels/compute/compute.cpp.
    std::string op = "matmul";
};

static bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::string(v) == "1";
}

// LONG_MATMUL_OP value -> the JIT define the compute kernel branches on.
static const std::map<std::string, std::string> kComputeOps = {
    {"matmul", "LONG_MATMUL_OP_MATMUL"},
    {"add", "LONG_MATMUL_OP_ADD"},
    {"silu", "LONG_MATMUL_OP_SILU"},
    {"exp", "LONG_MATMUL_OP_EXP"},
    {"sigmoid", "LONG_MATMUL_OP_SIGMOID"},
    {"gelu", "LONG_MATMUL_OP_GELU"},
    {"recip", "LONG_MATMUL_OP_RECIP"},
};

static std::string resolve_compute_op() {
    const char* v = std::getenv("LONG_MATMUL_OP");
    if (v == nullptr || *v == '\0') {
        return "matmul";
    }
    std::string op(v);
    if (kComputeOps.find(op) == kComputeOps.end()) {
        std::string valid;
        for (const auto& [name, _] : kComputeOps) {
            valid += (valid.empty() ? "" : ", ") + name;
        }
        TT_THROW("LONG_MATMUL_OP must be one of [{}], got '{}'", valid, op);
    }
    return op;
}

static PowerExperimentConfig resolve_power_experiment() {
    PowerExperimentConfig cfg;
    const std::string op = resolve_compute_op();

    // POWER_CASE, when set, overrides all four individual flags with one of the canonical
    // scenarios. Cases 1-5 all hold the writer at 100% amplification so that turning exactly one
    // of reader / compute / writer off gives a clean single-variable comparison against case 1.
    if (const char* pc = std::getenv("POWER_CASE"); pc != nullptr && *pc != '\0') {
        const int c = std::stoi(pc);
        switch (c) {
            case 0: cfg = {false, false, false, 0};    break;  // regular (baseline)
            case 1: cfg = {false, false, false, 100};  break;  // writer_amp
            case 2: cfg = {false, true,  false, 100};  break;  // compute_idle
            case 3: cfg = {true,  true,  false, 100};  break;  // reader_compute_idle
            case 4: cfg = {true,  false, false, 100};  break;  // reader_idle2
            // Case 5 is the writer's own idle case, symmetric with cases 2 and 4. Case 0 was the
            // stand-in for it before this existed, but case 0 only turns write amplification
            // off, which measures the marginal cost of the extra writes rather than the
            // writer's full contribution.
            case 5: cfg = {false, false, true,  100};  break;  // writer_idle
            default:
                TT_THROW("POWER_CASE must be in [0, 5], got {}", c);
        }
        cfg.op = op;
        fmt::print(
            "POWER_CASE={} -- reader={} compute={} writer={} write_amplification_pct={} op={}\n",
            c,
            cfg.disable_reader ? "idle" : "real",
            cfg.disable_compute ? "idle" : "real",
            cfg.disable_writer ? "idle" : "real",
            cfg.write_amplification_pct,
            cfg.op);
        return cfg;
    }

    // No POWER_CASE: fall back to the individual flags for finer manual control.
    cfg.disable_reader = env_flag("LONG_MATMUL_DISABLE_READER");
    cfg.disable_compute = env_flag("LONG_MATMUL_DISABLE_COMPUTE");
    cfg.disable_writer = env_flag("LONG_MATMUL_DISABLE_WRITER");
    if (const char* amp = std::getenv("LONG_MATMUL_WRITE_AMPLIFICATION_PCT");
        amp != nullptr && *amp != '\0') {
        cfg.write_amplification_pct = static_cast<uint32_t>(std::stoul(amp));
    }
    cfg.op = op;
    return cfg;
}

int main(int argc, char* argv[]) {
    uint32_t M = 256;
    uint32_t N = 256;
    uint32_t K = 512;
    uint32_t num_iterations = 100000;
    uint32_t fixed_tiles_per_core = 0;  // 0 = split total work (default), >0 = each core does exactly this many tiles

    if (argc >= 2) { M = std::stoul(argv[1]); }
    if (argc >= 3) { N = std::stoul(argv[2]); }
    if (argc >= 4) { K = std::stoul(argv[3]); }
    if (argc >= 5) { num_iterations = std::stoul(argv[4]); }
    if (argc >= 6) { fixed_tiles_per_core = std::stoul(argv[5]); }

    TT_FATAL(M % TILE_HEIGHT == 0, "M ({}) must be divisible by TILE_HEIGHT ({})", M, TILE_HEIGHT);
    TT_FATAL(N % TILE_WIDTH == 0, "N ({}) must be divisible by TILE_WIDTH ({})", N, TILE_WIDTH);
    TT_FATAL(K % TILE_WIDTH == 0, "K ({}) must be divisible by TILE_WIDTH ({})", K, TILE_WIDTH);

    const uint32_t Mt = M / TILE_HEIGHT;
    const uint32_t Kt = K / TILE_WIDTH;
    const uint32_t Nt = N / TILE_WIDTH;
    const uint32_t total_output_tiles = Mt * Nt;

    const double flops_per_iter =
        2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double total_flops = flops_per_iter * static_cast<double>(num_iterations);

    fmt::print("=== Long Matmul Workload ===\n");
    fmt::print("Matrix: M={} N={} K={} (tiles: {}x{}x{})\n", M, N, K, Mt, Nt, Kt);
    fmt::print("Output tiles: {}  |  Iterations: {}\n", total_output_tiles, num_iterations);
    fmt::print("Math fidelity: HiFi4  |  Data format: Float16_b\n");
    if (fixed_tiles_per_core > 0) {
        fmt::print("Mode: FIXED per-core ({} tiles/core) — total work scales with core count\n", fixed_tiles_per_core);
    } else {
        fmt::print("Mode: SPLIT total work — tiles divided equally across cores\n");
    }
    fmt::print("Expected FLOPs per full run: {:.2e}\n\n", total_flops);

    const PowerExperimentConfig power_cfg = resolve_power_experiment();

    // Output blocking. Each core accumulates a BLOCK_M x BLOCK_N patch of output tiles in the
    // destination registers at once, so one column slice of A and one row slice of B feed
    // BLOCK_M*BLOCK_N multiplies. DRAM tile reads per multiply fall from 2 to
    // (BLOCK_M + BLOCK_N) / (BLOCK_M * BLOCK_N). 1x1 is the original tile-at-a-time behaviour.
    uint32_t block_m = 1, block_n = 1;
    if (const char* v = std::getenv("LONG_MATMUL_BLOCK_M"); v != nullptr && *v != '\0') {
        block_m = static_cast<uint32_t>(std::stoul(v));
    }
    if (const char* v = std::getenv("LONG_MATMUL_BLOCK_N"); v != nullptr && *v != '\0') {
        block_n = static_cast<uint32_t>(std::stoul(v));
    }
    TT_FATAL(block_m >= 1 && block_n >= 1, "LONG_MATMUL_BLOCK_M/N must be >= 1");
    // The whole block lives in the destination registers simultaneously.
    TT_FATAL(
        block_m * block_n <= 8,
        "LONG_MATMUL_BLOCK_M * LONG_MATMUL_BLOCK_N ({}) exceeds the 8-tile destination register budget",
        block_m * block_n);
    TT_FATAL(Mt % block_m == 0, "Mt ({}) must be divisible by LONG_MATMUL_BLOCK_M ({})", Mt, block_m);
    TT_FATAL(Nt % block_n == 0, "Nt ({}) must be divisible by LONG_MATMUL_BLOCK_N ({})", Nt, block_n);

    const uint32_t blocks_per_row = Nt / block_n;
    const uint32_t total_output_blocks = (Mt / block_m) * blocks_per_row;
    const double reads_per_multiply = double(block_m + block_n) / double(block_m * block_n);
    fmt::print(
        "Output block: {}x{} tiles  |  {} blocks  |  {:.3f} DRAM tile reads per multiply "
        "(1x1 baseline = 2.000)\n",
        block_m, block_n, total_output_blocks, reads_per_multiply);

    // One line with every effective knob, so the log of any run says exactly what was measured
    // regardless of which env vars were used to get there.
    {
        const char* pc = std::getenv("POWER_CASE");
        fmt::print(
            "Effective knobs: POWER_CASE={} reader={} compute={} writer={} write_amplification_pct={} "
            "op={} block={}x{} mode={} shape={}x{}x{} iterations={}\n",
            (pc != nullptr && *pc != '\0') ? pc : "unset",
            power_cfg.disable_reader ? "idle" : "real",
            power_cfg.disable_compute ? "idle" : "real",
            power_cfg.disable_writer ? "idle" : "real",
            power_cfg.write_amplification_pct,
            power_cfg.op,
            block_m,
            block_n,
            fixed_tiles_per_core > 0 ? fmt::format("fixed({})", fixed_tiles_per_core) : std::string("split"),
            M,
            N,
            K,
            num_iterations);
    }

    // The writer normally issues 1 NoC write per output tile while the reader issues 2*Kt reads.
    // Amplification re-writes the same tile to the same address to load the write-side NoC path
    // symmetrically; 100% matches the reader's read volume exactly. Correctness is unaffected
    // (the app never verifies its output).
    const uint32_t write_repeats = (power_cfg.write_amplification_pct == 0)
        ? 1u
        : std::max<uint32_t>(
              1u,
              static_cast<uint32_t>(std::lround(
                  (static_cast<double>(power_cfg.write_amplification_pct) / 100.0) * 2.0 *
                  static_cast<double>(Kt))));

    try {
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        auto& cq = mesh_device->mesh_command_queue();

        auto max_core_grid = mesh_device->compute_with_storage_grid_size();
        const uint32_t max_x = max_core_grid.x;
        const uint32_t max_y = max_core_grid.y;

        fmt::print("Detected max compute grid: {}x{} ({} cores)\n", max_x, max_y, max_x * max_y);

        const auto test_grids = build_test_grids(max_x, max_y);

        fmt::print("Selected test grids:\n");
        for (const auto& g : test_grids) {
            fmt::print("  {}x{} ({})\n", g.x, g.y, g.x * g.y);
        }
        fmt::print("\n");

        fmt::print("Generating input data once and reusing it for all runs...\n");
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<bfloat16> src0_vec(M * K);
        std::vector<bfloat16> src1_vec(K * N);
        for (auto& v : src0_vec) {
            v = bfloat16(dist(rng));
        }
        for (auto& v : src1_vec) {
            v = bfloat16(dist(rng));
        }

        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

        std::vector<RunReport> summary_reports;
        summary_reports.reserve(test_grids.size());

        for (size_t grid_idx = 0; grid_idx < test_grids.size(); ++grid_idx) {
            if (grid_idx > 0) {
                fmt::print("\nWaiting 5 seconds before next run...\n");
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }

            CoreCoord core_grid{};
            core_grid.x = test_grids[grid_idx].x;
            core_grid.y = test_grids[grid_idx].y;

            fmt::print("\n============================================================\n");
            fmt::print(
                "Starting run for compute grid: {}x{} ({} cores)\n",
                core_grid.x,
                core_grid.y,
                core_grid.x * core_grid.y);
            fmt::print("============================================================\n");

            uint32_t num_cores, work_per_core1, work_per_core2;
            CoreRangeSet all_cores, core_group_1, core_group_2;

            if (fixed_tiles_per_core > 0) {
                TT_FATAL(
                    fixed_tiles_per_core <= total_output_blocks,
                    "fixed_tiles_per_core ({}) must not exceed total_output_blocks ({})",
                    fixed_tiles_per_core, total_output_blocks);
                num_cores     = core_grid.x * core_grid.y;
                all_cores     = CoreRangeSet({CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1})});
                core_group_1  = all_cores;
                core_group_2  = CoreRangeSet();
                work_per_core1 = fixed_tiles_per_core;
                work_per_core2 = 0;
            } else {
                auto [nc, ac, cg1, cg2, wpc1, wpc2] = split_work_to_cores(core_grid, total_output_blocks);
                num_cores = nc; all_cores = ac;
                core_group_1 = cg1; core_group_2 = cg2;
                work_per_core1 = wpc1; work_per_core2 = wpc2;
            }

            Program program{};
            constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

            distributed::DeviceLocalBufferConfig dram_config{
                .page_size = single_tile_size,
                .buffer_type = BufferType::DRAM};

            auto src0_dram = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = single_tile_size * Mt * Kt},
                dram_config,
                mesh_device.get());

            auto src1_dram = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = single_tile_size * Kt * Nt},
                dram_config,
                mesh_device.get());

            auto dst_dram = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = single_tile_size * Mt * Nt},
                dram_config,
                mesh_device.get());

            const auto cb_fmt = tt::DataFormat::Float16_b;
            // Each CB must hold one whole slice (in0/in1) or one whole output block, doubled so
            // the reader can prefetch the next step while compute works on the current one.
            const uint32_t cb0_tiles = 2 * block_m;
            const uint32_t cb1_tiles = 2 * block_n;
            const uint32_t cb_out_tiles = 2 * block_m * block_n;

            tt_metal::CreateCircularBuffer(
                program,
                all_cores,
                CircularBufferConfig(cb0_tiles * single_tile_size, {{CBIndex::c_0, cb_fmt}})
                    .set_page_size(CBIndex::c_0, single_tile_size));

            tt_metal::CreateCircularBuffer(
                program,
                all_cores,
                CircularBufferConfig(cb1_tiles * single_tile_size, {{CBIndex::c_1, cb_fmt}})
                    .set_page_size(CBIndex::c_1, single_tile_size));

            tt_metal::CreateCircularBuffer(
                program,
                all_cores,
                CircularBufferConfig(cb_out_tiles * single_tile_size, {{CBIndex::c_16, cb_fmt}})
                    .set_page_size(CBIndex::c_16, single_tile_size));

            // Passed as JIT defines rather than compile-time args so that changing POWER_CASE or
            // the block size only triggers a kernel recompile -- the host binary never needs
            // rebuilding.
            const std::string bm = std::to_string(block_m), bn = std::to_string(block_n);
            std::map<std::string, std::string> reader_defines{{"BLOCK_M", bm}, {"BLOCK_N", bn}};
            if (power_cfg.disable_reader) {
                reader_defines["LONG_MATMUL_DISABLE_READER"] = "1";
            }
            std::map<std::string, std::string> compute_defines{{"BLOCK_M", bm}, {"BLOCK_N", bn}};
            compute_defines[kComputeOps.at(power_cfg.op)] = "1";
            if (power_cfg.disable_compute) {
                compute_defines["LONG_MATMUL_DISABLE_COMPUTE"] = "1";
            }
            std::map<std::string, std::string> writer_defines{{"BLOCK_M", bm}, {"BLOCK_N", bn}};
            if (power_cfg.disable_writer) {
                writer_defines["LONG_MATMUL_DISABLE_WRITER"] = "1";
            }

            std::vector<uint32_t> reader_ct_args;
            TensorAccessorArgs(*src0_dram).append_to(reader_ct_args);
            TensorAccessorArgs(*src1_dram).append_to(reader_ct_args);

            auto reader_id = tt_metal::CreateKernel(
                program,
                OVERRIDE_KERNEL_PREFIX "long_matmul/kernels/dataflow/reader.cpp",
                all_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = reader_ct_args,
                    .defines = reader_defines});

            std::vector<uint32_t> writer_ct_args;
            TensorAccessorArgs(*dst_dram).append_to(writer_ct_args);

            auto writer_id = tt_metal::CreateKernel(
                program,
                OVERRIDE_KERNEL_PREFIX "long_matmul/kernels/dataflow/writer.cpp",
                all_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = writer_ct_args,
                    .defines = writer_defines});

            auto compute_id = tt_metal::CreateKernel(
                program,
                OVERRIDE_KERNEL_PREFIX "long_matmul/kernels/compute/compute.cpp",
                all_cores,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .defines = compute_defines});

            uint32_t work_offset = 0;
            uint32_t core_linear_idx = 0;
            auto work_groups = {
                std::make_pair(core_group_1, work_per_core1),
                std::make_pair(core_group_2, work_per_core2)
            };

            for (const auto& [ranges, work_per_core] : work_groups) {
                for (const auto& range : ranges.ranges()) {
                    for (const auto& core : range) {
                        // In fixed mode each core starts at a different offset (wrapping) so
                        // cores hit different DRAM addresses and don't serialise on the same bank.
                        const uint32_t effective_offset = (fixed_tiles_per_core > 0)
                            ? (core_linear_idx * fixed_tiles_per_core) % total_output_blocks
                            : work_offset;

                        tt_metal::SetRuntimeArgs(
                            program, reader_id, core,
                            {src0_dram->address(), src1_dram->address(),
                             Mt, Kt, Nt,
                             effective_offset, work_per_core, num_iterations, blocks_per_row});

                        tt_metal::SetRuntimeArgs(
                            program, writer_id, core,
                            {dst_dram->address(), work_per_core, effective_offset, num_iterations,
                             write_repeats, blocks_per_row, Nt});

                        tt_metal::SetRuntimeArgs(
                            program, compute_id, core,
                            {work_per_core, Kt, num_iterations});

                        if (fixed_tiles_per_core == 0) work_offset += work_per_core;
                        ++core_linear_idx;
                    }
                }
            }

            distributed::EnqueueWriteMeshBuffer(cq, src0_dram, src0_vec, false);
            distributed::EnqueueWriteMeshBuffer(cq, src1_dram, src1_vec, false);

            std::string tiles_per_core_str;
            if (fixed_tiles_per_core > 0) {
                tiles_per_core_str = std::to_string(fixed_tiles_per_core);
            } else {
                tiles_per_core_str = std::to_string(work_per_core1);
                if (work_per_core2 > 0) {
                    tiles_per_core_str += " / " + std::to_string(work_per_core2);
                }
            }

            fmt::print(
                "Active cores: {}  |  Tiles/core: {}\n",
                num_cores,
                tiles_per_core_str);

            fmt::print(
                "Running {} iterations of {}x{}x{} HiFi4 matmul on {} cores...\n",
                num_iterations,
                M,
                N,
                K,
                num_cores);

            distributed::MeshWorkload workload;
            distributed::MeshCoordinateRange device_range(mesh_device->shape());
            workload.add_program(device_range, std::move(program));

            auto sys_start = std::chrono::system_clock::now();
            auto t_start = std::chrono::high_resolution_clock::now();

            distributed::EnqueueMeshWorkload(cq, workload, false);

            std::vector<bfloat16> result_vec(Mt * Nt * TILE_HW);
            distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram, true);

            auto t_end = std::chrono::high_resolution_clock::now();
            auto sys_end = std::chrono::system_clock::now();

            // Cheap correctness signal. The workload never verifies its output, so with the
            // blocked path this is the only way to notice a tile-indexing mistake: the inputs
            // are seeded deterministically (mt19937(42)) and accumulation order over the shared
            // dimension is identical for every block size, so this checksum must not depend on
            // LONG_MATMUL_BLOCK_M/N.
            {
                double sum = 0.0, absmax = 0.0;
                for (const auto& v : result_vec) {
                    const double d = static_cast<float>(v);
                    sum += d;
                    absmax = std::max(absmax, std::abs(d));
                }
                fmt::print("Output checksum: sum={:.6e} absmax={:.6e}\n", sum, absmax);
            }


            const std::string start_time_str = format_time(sys_start);
            const std::string end_time_str = format_time(sys_end);

            const double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
            // In fixed mode, actual FLOPs scale with num_cores (each core does the same work).
            // The unit of work per core is an output block of block_m * block_n tiles.
            const double run_flops = (fixed_tiles_per_core > 0)
                ? 2.0 * fixed_tiles_per_core * block_m * block_n * num_cores * Kt * TILE_HEIGHT * TILE_WIDTH *
                      num_iterations
                : total_flops;
            const double tflops = run_flops / elapsed_s / 1e12;
            const double per_iter_ms = elapsed_s * 1000.0 / static_cast<double>(num_iterations);

            fmt::print("\n=== Results for {}x{} ===\n", core_grid.x, core_grid.y);
            fmt::print("Start time:      {}\n", start_time_str);
            fmt::print("End time:        {}\n", end_time_str);
            fmt::print("Total time:      {:.3f} s\n", elapsed_s);
            fmt::print("Throughput:      {:.2f} TFLOPS\n", tflops);
            fmt::print("Per-iteration:   {:.3f} ms\n", per_iter_ms);
            fmt::print("Test Passed\n");

            summary_reports.push_back(RunReport{
                .grid_x = core_grid.x,
                .grid_y = core_grid.y,
                .active_cores = num_cores,
                .tiles_per_core = tiles_per_core_str,
                .elapsed_s = elapsed_s,
                .tflops = tflops,
                .per_iter_ms = per_iter_ms,
                .start_time = start_time_str,
                .end_time = end_time_str
            });
        }

        fmt::print("\n\n========================================================================================================================\n");
        fmt::print("SUMMARY REPORT\n");
        fmt::print("========================================================================================================================\n");
        fmt::print(
            "{:>8} {:>12} {:>16} {:>16} {:>16} {:>18} {:>26} {:>26}\n",
            "Grid",
            "Cores",
            "Tiles/Core",
            "Time [s]",
            "TFLOPS",
            "Per iter [ms]",
            "Start Time",
            "End Time");

        for (const auto& r : summary_reports) {
            fmt::print(
                "{:>3}x{:<3} {:>12} {:>16} {:>16.3f} {:>16.2f} {:>18.3f} {:>26} {:>26}\n",
                r.grid_x,
                r.grid_y,
                r.active_cores,
                r.tiles_per_core,
                r.elapsed_s,
                r.tflops,
                r.per_iter_ms,
                r.start_time,
                r.end_time);
        }

        mesh_device->close();

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed: {}\n", e.what());
        throw;
    }

    return 0;
}
