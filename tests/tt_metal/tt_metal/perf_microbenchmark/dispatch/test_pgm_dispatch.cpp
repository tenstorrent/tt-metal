// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 100;
constexpr uint32_t MAX_KERNEL_SIZE_K = 16;
constexpr uint32_t DEFAULT_KERNEL_SIZE_K = 1;
constexpr uint32_t MAX_CBS = 32;
constexpr uint32_t MAX_ARGS = 255;

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Times dispatching program to M cores, N processors, of various sizes, CBs, runtime args
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

uint32_t iterations_g = DEFAULT_ITERATIONS;
uint32_t warmup_iterations_g = DEFAULT_WARMUP_ITERATIONS;
CoreRange workers_g = {{0, 0}, {0, 0}};;
uint32_t size_g;
uint32_t n_cbs_g;
uint32_t n_args_g;
bool brisc_enabled_g;
bool ncrisc_enabled_g;
bool trisc_enabled_g;

void init(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -w: warm-up iterations before starting timer (default {}), ", DEFAULT_WARMUP_ITERATIONS);
        log_info(LogTest, "  -i: iterations (default {})", DEFAULT_ITERATIONS);
        log_info(LogTest, "  -s: size of kernels in bytes (default {}, max {})", DEFAULT_KERNEL_SIZE_K, MAX_KERNEL_SIZE_K * 1024);
        log_info(LogTest, "  -x: X end of core range (default {})", 1);
        log_info(LogTest, "  -y: Y end of core range (default {})", 1);
        log_info(LogTest, "  -c: number of CBs (default {}, max {})", 0, MAX_CBS);
        log_info(LogTest, "  -a: number of runtime args (default {}, max {})", 0, MAX_ARGS);
        log_info(LogTest, "  -b: disable brisc kernel");
        log_info(LogTest, "  -n: disable ncrisc kernel");
        log_info(LogTest, "  -t: disable trisc kernels");
        exit(0);
    }

    uint32_t core_x = test_args::get_command_option_uint32(input_args, "-x", 1);
    uint32_t core_y = test_args::get_command_option_uint32(input_args, "-y", 1);
    warmup_iterations_g = test_args::get_command_option_uint32(input_args, "-w", DEFAULT_WARMUP_ITERATIONS);
    iterations_g = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    size_g = test_args::get_command_option_uint32(input_args, "-s", DEFAULT_KERNEL_SIZE_K);
    n_cbs_g = test_args::get_command_option_uint32(input_args, "-c", 0);
    n_args_g = test_args::get_command_option_uint32(input_args, "-a", 0);
    if (size_g > MAX_KERNEL_SIZE_K * 1024) {
        log_fatal("CB count must be 0..{}", MAX_KERNEL_SIZE_K * 1024);
        exit(0);
    }

    // Kernel only supports certain sizes, snap to them
    if (size_g < 256) size_g = 16;
    else if (size_g < 512) size_g = 256;
    else if (size_g < 1024) size_g = 512;
    else {
        for (int i = 2; i < 16; i++) {
            if (size_g < 1024 * i) {
                size_g = 1024 * (i - 1);
                break;
            }
        }
    }
    if (n_cbs_g > MAX_CBS) {
        log_fatal("CB count must be 0..{}", MAX_CBS);
        exit(0);
    }
    if (n_args_g > MAX_ARGS) {
        log_fatal("CB count must be 0..{}", MAX_ARGS);
        exit(0);
    }

    brisc_enabled_g = !test_args::has_command_option(input_args, "-b");
    ncrisc_enabled_g = !test_args::has_command_option(input_args, "-n");
    trisc_enabled_g = !test_args::has_command_option(input_args, "-t");

    workers_g = {.start = {0, 0}, .end = {core_x, core_y}};
}

void set_runtime_args(Program& program, tt_metal::KernelHandle kernel_id, vector<uint32_t>& args) {
    for (int core_idx_y = workers_g.start.y; core_idx_y < workers_g.end.y; core_idx_y++) {
        for (int core_idx_x = workers_g.start.x; core_idx_x < workers_g.end.x; core_idx_x++) {
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};
            tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
        }
    }
}

int main(int argc, char **argv) {
    init(argc, argv);

    tt::llrt::OptionsG.set_kernels_nullified(true);

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        CommandQueue& cq = tt::tt_metal::detail::GetCommandQueue(device);

        tt_metal::Program program = tt_metal::CreateProgram();

        std::map<string, string> pad_defines = {
            {"KERNEL_BYTES", std::to_string(size_g)}
        };

        vector<uint32_t> args;
        args.resize(n_args_g);

        for (int i = 0; i < n_cbs_g; i++) {
            tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(16, {{i, tt::DataFormat::Float16_b}})
                .set_page_size(i, 16);
            auto cb = tt_metal::CreateCircularBuffer(program, workers_g, cb_config);
        }

        if (brisc_enabled_g) {
            auto dm0 = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                workers_g,
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .defines = pad_defines});
            set_runtime_args(program, dm0, args);
        }

        if (ncrisc_enabled_g) {
            auto dm1 = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                workers_g,
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .defines = pad_defines});
            set_runtime_args(program, dm1, args);
        }

        if (trisc_enabled_g) {
            auto compute = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                workers_g,
                tt_metal::ComputeConfig{.defines = pad_defines});
            set_runtime_args(program, compute, args);
        }

        // Cache stuff
        for (int i = 0; i < warmup_iterations_g; i++) {
            EnqueueProgram(cq, program, false);
            Finish(cq);
        }

        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < iterations_g; i++) {
            EnqueueProgram(cq, program, false);
        }
        Finish(cq);
        auto end = std::chrono::system_clock::now();

        log_info(LogTest, "Kernel size: {}", size_g);
        log_info(LogTest, "Grid: ({}-{})", workers_g.start.str(), workers_g.end.str());
        log_info(LogTest, "CBs: {}", n_cbs_g);
        log_info(LogTest, "Args: {}", n_args_g);

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);
        log_info(LogTest, "Ran in {}us per iteration", elapsed_seconds.count() * 1000 * 1000 / iterations_g);

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    tt::llrt::OptionsG.set_kernels_nullified(false);

    return 0;
}
