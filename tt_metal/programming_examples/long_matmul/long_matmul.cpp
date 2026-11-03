// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
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
    std::string blocks_per_core;
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

// Idle time between grids. tt-ember measures the board's idle current in these pauses.
static constexpr std::chrono::seconds kPauseBetweenGrids{5};

// Output tiles one core can hold in the destination registers at once (half of DEST, 16-bit).
static constexpr uint32_t kDestTileBudget = 8;

// The sweep walks the near-square band of the core array: for each width x, it runs the three
// heights x-1, x and x+1, clipped to the device. That gives a dense ascending core-count series
// while also pairing grids of equal core count but different aspect ratio (3x4 vs 4x3, 6x7 vs
// 7x6, 8x7 vs 7x8), which is what separates a core-count effect from a NoC-shape one. The full
// device grid is always the last one: 15 grids on a Wormhole n300 (8x7), 17 on an n150 (8x8),
// 24 on a Blackhole p100a (11x10). On a device wider than it is tall, the widths past the band
// (where even x-1 is taller than the device) run at the full height, so a 13x10 grid gets 26
// grids, ending 11x10, 12x10, 13x10.
static std::vector<GridCandidate> build_test_grids(uint32_t max_x, uint32_t max_y) {
    std::vector<GridCandidate> result;

    for (uint32_t x = kMinGridX; x <= max_x; ++x) {
        if (x - 1 > max_y) {
            result.push_back({x, max_y});
            continue;
        }
        for (uint32_t y = x - 1; y <= x + 1 && y <= max_y; ++y) {
            result.push_back({x, y});
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

// Parses the whole string as a non-negative decimal number. Unlike std::stoul this rejects a sign,
// trailing characters and values that do not fit, so a typo fails loudly instead of turning into
// a huge or truncated knob value.
static uint32_t parse_u32(const char* what, std::string_view text) {
    uint32_t value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    TT_FATAL(
        !text.empty() && ec == std::errc() && end == text.data() + text.size(),
        "{} must be a non-negative integer, got '{}'",
        what,
        text);
    return value;
}

// Reads an optional numeric env var; returns fallback when it is unset or empty.
static uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    return (v == nullptr || *v == '\0') ? fallback : parse_u32(name, v);
}

// Every circular buffer holds this many slices or blocks, so the producer can fill the next one
// while the consumer works on the current one. Must match kCbBufferDepth in kernels/config.h,
// which idle kernels in updated mode rely on to fill every slot with real data.
static constexpr uint32_t kCbBufferDepth = 2;

// Write amplification that matches the reader's NoC read volume.
static constexpr uint32_t kFullAmplificationPct = 100;
static constexpr uint32_t kMaxAmplificationPct = 1000;

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
    if (!kComputeOps.contains(op)) {
        std::string valid;
        for (const auto& [name, _] : kComputeOps) {
            valid += (valid.empty() ? "" : ", ") + name;
        }
        TT_THROW("LONG_MATMUL_OP must be one of [{}], got '{}'", valid, op);
    }
    return op;
}

// LONG_MATMUL_MODE: "updated" (default) or "legacy". Legacy reproduces how the published tt-ember
// results were measured; the README lists every difference.
static bool resolve_legacy_mode() {
    const char* v = std::getenv("LONG_MATMUL_MODE");
    if (v == nullptr || *v == '\0' || std::string_view(v) == "updated") {
        return false;
    }
    TT_FATAL(std::string_view(v) == "legacy", "LONG_MATMUL_MODE must be 'updated' or 'legacy', got '{}'", v);
    return true;
}

static PowerExperimentConfig resolve_power_experiment() {
    PowerExperimentConfig cfg;
    const std::string op = resolve_compute_op();

    // POWER_CASE, when set, overrides all four individual flags with one of the canonical
    // scenarios. Cases 1-5 all hold the writer at 100% amplification so that turning exactly one
    // of reader / compute / writer off compares against case 1 with the other two unchanged.
    if (const char* pc = std::getenv("POWER_CASE"); pc != nullptr && *pc != '\0') {
        const uint32_t c = parse_u32("POWER_CASE", pc);
        constexpr uint32_t full = kFullAmplificationPct;
        switch (c) {
            // regular: everything real, no write amplification
            case 0: cfg = {.write_amplification_pct = 0}; break;
            // writer_amp: everything real, the reference for cases 2-5
            case 1: cfg = {.write_amplification_pct = full}; break;
            // compute_idle
            case 2: cfg = {.disable_compute = true, .write_amplification_pct = full}; break;
            // reader_compute_idle
            case 3: cfg = {.disable_reader = true, .disable_compute = true, .write_amplification_pct = full}; break;
            // reader_idle2
            case 4: cfg = {.disable_reader = true, .write_amplification_pct = full}; break;
            // writer_idle. Case 0 only turns write amplification off, so it measures the cost of
            // the extra writes; case 5 idles the writer, symmetric with cases 2 and 4.
            case 5: cfg = {.disable_writer = true, .write_amplification_pct = full}; break;
            default: TT_THROW("POWER_CASE must be in [0, 5], got {}", c);
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
    cfg.write_amplification_pct = env_u32("LONG_MATMUL_WRITE_AMPLIFICATION_PCT", /*fallback=*/0);
    TT_FATAL(
        cfg.write_amplification_pct <= kMaxAmplificationPct,
        "LONG_MATMUL_WRITE_AMPLIFICATION_PCT must be at most {}, got {}",
        kMaxAmplificationPct,
        cfg.write_amplification_pct);
    cfg.op = op;
    return cfg;
}

int main(int argc, char* argv[]) {
    uint32_t M = 256;
    uint32_t N = 256;
    uint32_t K = 512;
    uint32_t num_iterations = 100000;
    // 0 = split total work (default), >0 = each core does exactly this many output blocks
    uint32_t fixed_blocks_per_core = 0;

    // With no arguments (as tests/scripts/run_sdk_examples.sh runs every installed example) do a
    // quick smoke run: the full grid only, a few iterations, no pauses. Measurements always pass
    // at least the shape and the iteration count.
    const bool smoke_run = argc == 1;
    if (smoke_run) {
        num_iterations = 10;
    }

    if (argc >= 2) {
        M = parse_u32("M", argv[1]);
    }
    if (argc >= 3) {
        N = parse_u32("N", argv[2]);
    }
    if (argc >= 4) {
        K = parse_u32("K", argv[3]);
    }
    if (argc >= 5) {
        num_iterations = parse_u32("iterations", argv[4]);
    }
    if (argc >= 6) {
        fixed_blocks_per_core = parse_u32("fixed_blocks_per_core", argv[5]);
    }

    TT_FATAL(M > 0 && N > 0 && K > 0, "M, N and K must be positive, got {}x{}x{}", M, N, K);
    TT_FATAL(num_iterations > 0, "iterations must be positive");
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
    if (fixed_blocks_per_core > 0) {
        fmt::print(
            "Mode: FIXED per-core ({} blocks/core) — total work scales with core count\n", fixed_blocks_per_core);
    } else {
        fmt::print("Mode: SPLIT total work — tiles divided equally across cores\n");
    }
    fmt::print("Expected FLOPs per full run: {:.2e}\n\n", total_flops);

    const PowerExperimentConfig power_cfg = resolve_power_experiment();
    const bool legacy_mode = resolve_legacy_mode();

    // Output blocking. Each core accumulates a BLOCK_M x BLOCK_N patch of output tiles in the
    // destination registers at once, so one column slice of A and one row slice of B feed
    // BLOCK_M*BLOCK_N multiplies. DRAM tile reads per multiply fall from 2 to
    // (BLOCK_M + BLOCK_N) / (BLOCK_M * BLOCK_N). 1x1 is the original tile-at-a-time behaviour.
    const uint32_t block_m = env_u32("LONG_MATMUL_BLOCK_M", /*fallback=*/1);
    const uint32_t block_n = env_u32("LONG_MATMUL_BLOCK_N", /*fallback=*/1);
    TT_FATAL(block_m >= 1 && block_n >= 1, "LONG_MATMUL_BLOCK_M/N must be >= 1");
    // The whole block lives in the destination registers simultaneously.
    TT_FATAL(
        block_m * block_n <= kDestTileBudget,
        "LONG_MATMUL_BLOCK_M * LONG_MATMUL_BLOCK_N ({}) exceeds the {}-tile destination register budget",
        block_m * block_n,
        kDestTileBudget);
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
            "op={} block={}x{} mode={} shape={}x{}x{} iterations={} LONG_MATMUL_MODE={}\n",
            (pc != nullptr && *pc != '\0') ? pc : "unset",
            power_cfg.disable_reader ? "idle" : "real",
            power_cfg.disable_compute ? "idle" : "real",
            power_cfg.disable_writer ? "idle" : "real",
            power_cfg.write_amplification_pct,
            power_cfg.op,
            block_m,
            block_n,
            fixed_blocks_per_core > 0 ? fmt::format("fixed({})", fixed_blocks_per_core) : std::string("split"),
            M,
            N,
            K,
            num_iterations,
            legacy_mode ? "legacy" : "updated");
    }

    // The writer normally issues 1 NoC write per output tile, while the reader issues
    // Kt * (block_m + block_n) / (block_m * block_n) reads per output tile (2*Kt at 1x1).
    // Amplification re-writes the same tile to the same address to load the write-side NoC path
    // symmetrically; 100% matches the reader's read volume. Legacy mode always uses the 1x1 rate.
    // Correctness is unaffected (the app never verifies its output).
    const double reads_per_output_tile =
        legacy_mode ? 2.0 * static_cast<double>(Kt) : static_cast<double>(Kt) * reads_per_multiply;
    const uint32_t write_repeats =
        (power_cfg.write_amplification_pct == 0)
            ? 1u
            : std::max<uint32_t>(
                  1u,
                  static_cast<uint32_t>(std::lround(
                      (static_cast<double>(power_cfg.write_amplification_pct) / kFullAmplificationPct) *
                      reads_per_output_tile)));

    try {
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        auto& cq = mesh_device->mesh_command_queue();

        auto max_core_grid = mesh_device->compute_with_storage_grid_size();
        const uint32_t max_x = max_core_grid.x;
        const uint32_t max_y = max_core_grid.y;

        fmt::print("Detected max compute grid: {}x{} ({} cores)\n", max_x, max_y, max_x * max_y);

        const auto test_grids = smoke_run ? std::vector<GridCandidate>{{max_x, max_y}} : build_test_grids(max_x, max_y);
        if (smoke_run) {
            fmt::print("No arguments: smoke run on the full grid only, {} iterations.\n", num_iterations);
        }

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
                fmt::print("\nWaiting {} seconds before next run...\n", kPauseBetweenGrids.count());
                std::this_thread::sleep_for(kPauseBetweenGrids);
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

            if (fixed_blocks_per_core > 0) {
                TT_FATAL(
                    fixed_blocks_per_core <= total_output_blocks,
                    "fixed_blocks_per_core ({}) must not exceed total_output_blocks ({})",
                    fixed_blocks_per_core,
                    total_output_blocks);
                num_cores     = core_grid.x * core_grid.y;
                all_cores     = CoreRangeSet({CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1})});
                core_group_1  = all_cores;
                core_group_2  = CoreRangeSet();
                work_per_core1 = fixed_blocks_per_core;
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

            // Sizes in 64 bits: a 32-bit product would wrap for operands of 4 GiB or more.
            const DeviceAddr tile_bytes = single_tile_size;
            auto src0_dram = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = tile_bytes * Mt * Kt}, dram_config, mesh_device.get());

            auto src1_dram = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = tile_bytes * Kt * Nt}, dram_config, mesh_device.get());

            auto dst_dram = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = tile_bytes * Mt * Nt}, dram_config, mesh_device.get());

            // Each CB holds kCbBufferDepth slices of A or B, or output blocks.
            auto make_cb_config = [&](CBIndex cb_index, uint32_t tiles_per_slot) {
                return CircularBufferConfig(
                           kCbBufferDepth * tiles_per_slot * single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(cb_index, single_tile_size);
            };
            tt_metal::CreateCircularBuffer(program, all_cores, make_cb_config(CBIndex::c_0, block_m));
            tt_metal::CreateCircularBuffer(program, all_cores, make_cb_config(CBIndex::c_1, block_n));
            tt_metal::CreateCircularBuffer(program, all_cores, make_cb_config(CBIndex::c_16, block_m * block_n));

            // Passed as JIT defines so the kernels compile only the selected op and leave out an
            // idle writer's writes. An idle reader or idle compute keeps its code, because in
            // updated mode it still does its first kCbBufferDepth steps for real.
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
            if (legacy_mode) {
                for (auto* defines : {&reader_defines, &compute_defines, &writer_defines}) {
                    (*defines)["LONG_MATMUL_LEGACY"] = "1";
                }
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
                        // In fixed mode the cores' start blocks are spread evenly over the output
                        // so they hit different DRAM addresses and don't serialise on the same
                        // bank, whatever fixed_blocks_per_core is. The reader and writer wrap the
                        // block index at total_output_blocks.
                        const uint32_t effective_offset =
                            (fixed_blocks_per_core > 0)
                                ? static_cast<uint32_t>(uint64_t{core_linear_idx} * total_output_blocks / num_cores)
                                : work_offset;

                        tt_metal::SetRuntimeArgs(
                            program,
                            reader_id,
                            core,
                            {src0_dram->address(),
                             src1_dram->address(),
                             Kt,
                             Nt,
                             effective_offset,
                             work_per_core,
                             num_iterations,
                             blocks_per_row,
                             total_output_blocks});

                        tt_metal::SetRuntimeArgs(
                            program, writer_id, core,
                            {dst_dram->address(), work_per_core, effective_offset, num_iterations,
                             write_repeats, blocks_per_row, Nt, total_output_blocks});

                        tt_metal::SetRuntimeArgs(
                            program, compute_id, core,
                            {work_per_core, Kt, num_iterations});

                        if (fixed_blocks_per_core == 0) {
                            work_offset += work_per_core;
                        }
                        ++core_linear_idx;
                    }
                }
            }

            distributed::EnqueueWriteMeshBuffer(cq, src0_dram, src0_vec, /*blocking=*/false);
            distributed::EnqueueWriteMeshBuffer(cq, src1_dram, src1_vec, /*blocking=*/false);
            if (!legacy_mode) {
                // Updated mode waits for the uploads here, so none of them lands in the measured
                // interval. Legacy mode, like the published measurements, does not wait.
                distributed::Finish(cq);
            }

            std::string blocks_per_core_str;
            if (fixed_blocks_per_core > 0) {
                blocks_per_core_str = std::to_string(fixed_blocks_per_core);
            } else {
                blocks_per_core_str = std::to_string(work_per_core1);
                if (work_per_core2 > 0) {
                    blocks_per_core_str += " / " + std::to_string(work_per_core2);
                }
            }

            fmt::print("Active cores: {}  |  Blocks/core: {}\n", num_cores, blocks_per_core_str);

            fmt::print(
                "Running {} iterations of {}x{}x{} {} on {} cores...\n",
                num_iterations,
                M,
                N,
                K,
                power_cfg.op,
                num_cores);

            distributed::MeshWorkload workload;
            distributed::MeshCoordinateRange device_range(mesh_device->shape());
            workload.add_program(device_range, std::move(program));

            // With a cold kernel cache the kernels are JIT-compiled inside EnqueueMeshWorkload
            // (0.75 s for the first grid on n150), before any of the program's commands reach the
            // device. Updated mode starts the clock after it returns, so the compile stays out of
            // the measured interval; legacy mode starts it before, as the published measurements
            // did.
            auto sys_start = std::chrono::system_clock::now();
            auto t_start = std::chrono::high_resolution_clock::now();

            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

            if (!legacy_mode) {
                sys_start = std::chrono::system_clock::now();
                t_start = std::chrono::high_resolution_clock::now();
            }

            std::vector<bfloat16> result_vec(Mt * Nt * TILE_HW);
            distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram, /*blocking=*/true);

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

            // With TT_METAL_DEVICE_PROFILER=1 this drains the kernels' timing markers (see
            // kernels/profile_markers.h) into generated/profiler/.logs/profile_log_device.csv.
            // It runs after t_end, so it is not part of the measured interval.
            ReadMeshDeviceProfilerResults(*mesh_device);

            const std::string start_time_str = format_time(sys_start);
            const std::string end_time_str = format_time(sys_end);

            const double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
            // In fixed mode, actual FLOPs scale with num_cores (each core does the same work).
            // The unit of work per core is an output block of block_m * block_n tiles.
            const double run_flops = (fixed_blocks_per_core > 0)
                                         ? 2.0 * fixed_blocks_per_core * block_m * block_n * num_cores * K *
                                               TILE_HEIGHT * TILE_WIDTH * num_iterations
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
                .blocks_per_core = blocks_per_core_str,
                .elapsed_s = elapsed_s,
                .tflops = tflops,
                .per_iter_ms = per_iter_ms,
                .start_time = start_time_str,
                .end_time = end_time_str});
        }

        fmt::print("\n\n========================================================================================================================\n");
        fmt::print("SUMMARY REPORT\n");
        fmt::print("========================================================================================================================\n");
        fmt::print(
            "{:>8} {:>12} {:>16} {:>16} {:>16} {:>18} {:>26} {:>26}\n",
            "Grid",
            "Cores",
            "Blocks/Core",
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
                r.blocks_per_core,
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
