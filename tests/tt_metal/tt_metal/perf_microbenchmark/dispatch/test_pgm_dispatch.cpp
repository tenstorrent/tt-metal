// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "umd/device/types/cluster_descriptor_types.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/device.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include <benchmark/benchmark.h>

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 100;
constexpr uint32_t MIN_KERNEL_SIZE_BYTES = 32;  // overhead
constexpr uint32_t DEFAULT_KERNEL_SIZE_K = 1;
constexpr uint32_t MAX_CBS = 32;
constexpr uint32_t MAX_ARGS = 255;

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Times dispatching program to M cores, N processors, of various sizes, CBs, runtime args
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;

struct TestInfo {
    uint32_t iterations = DEFAULT_ITERATIONS;
    uint32_t warmup_iterations = DEFAULT_WARMUP_ITERATIONS;
    CoreRange workers = {{0, 0}, {0, 0}};
    uint32_t kernel_size{DEFAULT_KERNEL_SIZE_K * 1024};
    uint32_t fast_kernel_cycles{0};
    uint32_t slow_kernel_cycles{0};
    uint32_t nfast_kernels{0};
    uint32_t n_cbs{0};
    uint32_t n_args{0};
    uint32_t n_common_args{0};
    uint32_t n_sems{0};
    uint32_t n_kgs{1};
    bool brisc_enabled{true};
    bool ncrisc_enabled{true};
    bool trisc_enabled{true};
    bool erisc_enabled{false};
    uint32_t erisc_count{1};
    bool lazy{false};
    bool time_just_finish{false};
    bool use_global{false};
    bool use_trace{false};
    bool dispatch_from_eth{false};
    bool use_all_cores{false};
};

std::tuple<uint32_t, uint32_t> get_core_count() {
    uint32_t core_x = 0;
    uint32_t core_y = 0;

    std::string arch_name{getenv("ARCH_NAME")};
    if (arch_name == "grayskull") {
        core_x = 11;
        core_y = 8;
    } else if (arch_name == "wormhole_b0") {
        core_x = 7;
        core_y = 6;
    } else if (arch_name == "blackhole") {
        core_x = 12;
        core_y = 9;
    } else {
        log_fatal("Unexpected ARCH_NAME {}", arch_name);
        exit(0);
    }
    return std::make_tuple(core_x, core_y);
}

void init(const std::vector<std::string>& input_args, TestInfo& info) {
    auto core_count = get_core_count();

    if (test_args::has_command_option(input_args, "-h") || test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -w: warm-up iterations before starting timer (default {}), ", DEFAULT_WARMUP_ITERATIONS);
        log_info(LogTest, "  -i: iterations (default {})", DEFAULT_ITERATIONS);
        log_info(
            LogTest,
            "  -s: size of kernels in powers of 2 bytes (default {}, min {})",
            DEFAULT_KERNEL_SIZE_K * 1024,
            MIN_KERNEL_SIZE_BYTES);
        log_info(LogTest, "  -x: X end of inclusive core range (default {})", 0);
        log_info(LogTest, "  -y: Y end of inclusive core range (default {})", 0);
        log_info(LogTest, "  -c: number of CBs (default {}, max {})", 0, MAX_CBS);
        log_info(LogTest, "  -a: number of runtime args (default {}, max {})", 0, MAX_ARGS);
        log_info(
            LogTest, " -ca: number of common runtime args multicast to all cores (default {}, max {})", 0, MAX_ARGS);
        log_info(LogTest, "  -S: number of semaphores (default {}, max {})", 0, NUM_SEMAPHORES);
        log_info(LogTest, " -kg: number of kernel groups (default 1)");
        log_info(LogTest, "  -g: use a 4 byte global variable (additional spans");
        log_info(LogTest, " -rs: run \"slow\" kernels for exactly <n> cycles (default 0)");
        log_info(LogTest, " -rf: run \"fast\" kernels for exactly <n> cycles (default 0)");
        log_info(LogTest, " -nf: run <n> fast kernels between slow kernels (default 0)");
        log_info(LogTest, "  -b: disable brisc kernel (default enabled)");
        log_info(LogTest, "  -n: disable ncrisc kernel (default enabled)");
        log_info(LogTest, "  -t: disable trisc kernels (default enabled)");
        log_info(LogTest, "  +e: enable erisc kernels (default disabled)");
        log_info(LogTest, " -ec: erisc count (default 1 if enabled)");
        log_info(LogTest, "  -f: time just the finish call (use w/ lazy mode) (default disabled)");
        log_info(LogTest, "  -z: enable dispatch lazy mode (default disabled)");
        log_info(LogTest, " -tr: enable trace (default disabled)");
        log_info(LogTest, " -de: dispatch from eth cores (default tensix)");
        log_info(
            LogTest,
            " -ac: use all viable worker cores (default {}x{})",
            std::get<0>(core_count),
            std::get<1>(core_count));
        exit(0);
    }

    uint32_t core_x = test_args::get_command_option_uint32(input_args, "-x", 0);
    uint32_t core_y = test_args::get_command_option_uint32(input_args, "-y", 0);

    if (test_args::has_command_option(input_args, "-ac")) {
        core_x = std::get<0>(core_count);
        core_y = std::get<1>(core_count);
    }

    info.warmup_iterations = test_args::get_command_option_uint32(input_args, "-w", DEFAULT_WARMUP_ITERATIONS);
    info.iterations = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    info.kernel_size = test_args::get_command_option_uint32(input_args, "-s", DEFAULT_KERNEL_SIZE_K * 1024);
    info.n_cbs = test_args::get_command_option_uint32(input_args, "-c", 0);
    info.n_args = test_args::get_command_option_uint32(input_args, "-a", 0);
    info.n_common_args = test_args::get_command_option_uint32(input_args, "-ca", 0);
    info.n_sems = test_args::get_command_option_uint32(input_args, "-S", 0);
    info.n_kgs = test_args::get_command_option_uint32(input_args, "-kg", 1);
    info.lazy = test_args::has_command_option(input_args, "-z");
    info.use_global = test_args::has_command_option(input_args, "-g");
    info.time_just_finish = test_args::has_command_option(input_args, "-f");
    info.fast_kernel_cycles = test_args::get_command_option_uint32(input_args, "-rf", 0);
    info.slow_kernel_cycles = test_args::get_command_option_uint32(input_args, "-rs", 0);
    info.nfast_kernels = test_args::get_command_option_uint32(input_args, "-nf", 0);
    info.use_trace = test_args::has_command_option(input_args, "-tr");
    info.dispatch_from_eth = test_args::has_command_option(input_args, "-de");
    if (info.kernel_size < MIN_KERNEL_SIZE_BYTES) {
        log_fatal("Minimum kernel size is {} bytes", MIN_KERNEL_SIZE_BYTES);
        exit(0);
    }
    if (info.n_cbs > MAX_CBS) {
        log_fatal("CB count must be 0..{}", MAX_CBS);
        exit(0);
    }
    if (info.n_args > MAX_ARGS) {
        log_fatal("Runtime arg count must be 0..{}", MAX_ARGS);
        exit(0);
    }
    if (info.n_common_args > MAX_ARGS) {
        log_fatal("Common Runtime arg count must be 0..{}", MAX_ARGS);
        exit(0);
    }
    if (info.n_sems > NUM_SEMAPHORES) {
        log_fatal("Sem count must be 0..{}", NUM_SEMAPHORES);
        exit(0);
    }
    if (info.n_kgs > core_x + 1) {
        log_fatal("This test uses columns for kernel groups so number of kernel groups must be <= x core range");
        exit(0);
    }
    info.brisc_enabled = !test_args::has_command_option(input_args, "-b");
    info.ncrisc_enabled = !test_args::has_command_option(input_args, "-n");
    info.trisc_enabled = !test_args::has_command_option(input_args, "-t");
    info.erisc_enabled = test_args::has_command_option(input_args, "+e");
    info.erisc_count = test_args::get_command_option_uint32(input_args, "-ec", 1);

    info.workers = CoreRange({0, 0}, {core_x, core_y});

    if (info.nfast_kernels != 0 && info.slow_kernel_cycles <= info.fast_kernel_cycles) {
        log_error(
            "The number of fast kernels is non-zero, but slow_kernel_ cycles ({}) is <= fast_kernel_cycles ({})",
            info.slow_kernel_cycles,
            info.fast_kernel_cycles);
        log_error("For meaningful results, run multiple fast kernels between single slow kernels");
        exit(0);
    }
}

void set_runtime_args(
    tt_metal::Program& program, tt_metal::KernelHandle kernel_id, vector<uint32_t>& args, CoreRange kg) {
    for (int core_idx_y = kg.start_coord.y; core_idx_y <= kg.end_coord.y; core_idx_y++) {
        for (int core_idx_x = kg.start_coord.x; core_idx_x <= kg.end_coord.x; core_idx_x++) {
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};
            tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
        }
    }
}

bool initialize_program(
    const TestInfo& info, tt_metal::IDevice* device, tt_metal::Program& program, uint32_t run_cycles) {
    program = tt_metal::CreateProgram();

    std::map<string, string> defines = {{"KERNEL_BYTES", std::to_string(info.kernel_size)}};
    if (run_cycles != 0) {
        defines.insert(std::pair<string, string>("KERNEL_RUN_TIME", std::to_string(run_cycles)));
    }
    if (info.use_global) {
        defines.insert(std::pair<string, string>("KERNEL_GLOBAL", "1"));
    }

    for (uint32_t i = 0; i < info.n_sems; i++) {
        tt_metal::CreateSemaphore(program, info.workers, 3);
    }

    vector<uint32_t> args;
    args.resize(info.n_args);
    vector<uint32_t> common_args;
    common_args.resize(info.n_common_args);

    for (int i = 0; i < info.n_cbs; i++) {
        tt_metal::CircularBufferConfig cb_config =
            tt_metal::CircularBufferConfig(16, {{i, tt::DataFormat::Float16_b}}).set_page_size(i, 16);
        auto cb = tt_metal::CreateCircularBuffer(program, info.workers, cb_config);
    }

    // first kernel group is possibly wide, remaining kernel groups are 1 column each
    CoreRange kg = {info.workers.start_coord, {info.workers.end_coord.x - info.n_kgs + 1, info.workers.end_coord.y}};
    for (uint32_t i = 0; i < info.n_kgs; i++) {
        defines.insert(std::pair<string, string>(string("KG_") + std::to_string(i), ""));

        if (info.brisc_enabled) {
            auto dm0 = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                kg,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .defines = defines});
            set_runtime_args(program, dm0, args, kg);
            tt_metal::SetCommonRuntimeArgs(program, dm0, common_args);
        }

        if (info.ncrisc_enabled) {
            auto dm1 = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                kg,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .defines = defines});
            set_runtime_args(program, dm1, args, kg);
            tt_metal::SetCommonRuntimeArgs(program, dm1, common_args);
        }

        if (info.trisc_enabled) {
            auto compute = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                kg,
                tt_metal::ComputeConfig{.defines = defines});
            set_runtime_args(program, compute, args, kg);
            tt_metal::SetCommonRuntimeArgs(program, compute, common_args);
        }

        kg.start_coord = {kg.end_coord.x + 1, kg.end_coord.y};
        kg.end_coord = kg.start_coord;
    }

    if (info.erisc_enabled) {
        auto erisc_cores = device->get_active_ethernet_cores(true);
        if (info.erisc_count > erisc_cores.size()) {
            log_fatal(
                "Requested number of erisc cores {} exceeds actual erisc core count {}",
                info.erisc_count,
                erisc_cores.size());
            return false;
        }
        auto erisc_core = erisc_cores.begin();
        for (uint32_t i = 0; i < info.erisc_count; i++, erisc_core++) {
            auto eth_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                *erisc_core,
                tt::tt_metal::EthernetConfig{
                    .eth_mode = Eth::RECEIVER,
                    .noc = NOC::NOC_0,
                    .defines = defines,
                });
            tt_metal::SetRuntimeArgs(program, eth_kernel, *erisc_core, args);
            tt_metal::SetCommonRuntimeArgs(program, eth_kernel, common_args);
        }
    }
    return true;
}

struct FakeBenchmarkState {
    std::vector<int>::iterator begin() { return range.begin(); }
    std::vector<int>::iterator end() { return range.end(); }

    std::vector<int> range{1};
};

template <typename T>
static int pgm_dispatch(T& state, TestInfo info) {
    if constexpr (std::is_same_v<T, benchmark::State>) {
        log_info(LogTest, "Running {}", state.name());
    }
    if (info.use_all_cores) {
        auto core_count = get_core_count();
        info.workers = CoreRange({0, 0}, {std::get<0>(core_count), std::get<1>(core_count)});
    }
    if constexpr (std::is_same_v<T, benchmark::State>) {
        info.kernel_size = state.range(0);
    }

    if (info.use_trace) {
        log_info(LogTest, "Running with trace enabled");
    }
    log_info(LogTest, "Warmup iterations: {}", info.warmup_iterations);
    log_info(LogTest, "Iterations: {}", info.iterations);
    log_info(
        LogTest,
        "Grid: ({}-{}) ({} cores)",
        info.workers.start_coord.str(),
        info.workers.end_coord.str(),
        info.workers.size());
    log_info(LogTest, "Kernel size: {}", info.kernel_size);
    if (info.nfast_kernels != 0) {
        log_info(LogTest, "Fast kernel cycles: {}", info.fast_kernel_cycles);
        log_info(LogTest, "Slow kernel cycles: {}", info.slow_kernel_cycles);
        log_info(LogTest, "{} fast kernels between slow kernels", info.nfast_kernels);
    } else {
        log_info(LogTest, "Kernel cycles: {}", info.slow_kernel_cycles);
    }
    log_info(LogTest, "KGs: {}", info.n_kgs);
    log_info(LogTest, "CBs: {}", info.n_cbs);
    log_info(LogTest, "UniqueRTArgs: {}", info.n_args);
    log_info(LogTest, "CommonRTArgs: {}", info.n_common_args);
    log_info(LogTest, "Sems: {}", info.n_sems);
    log_info(LogTest, "Lazy: {}", info.lazy);

    tt::llrt::RunTimeOptions::get_instance().set_kernels_nullified(true);

    bool pass = true;
    try {
        const chip_id_t device_id = 0;
        DispatchCoreType dispatch_core_type = info.dispatch_from_eth ? DispatchCoreType::ETH : DispatchCoreType::WORKER;
        tt_metal::IDevice* device = tt_metal::CreateDevice(
            device_id, 1, DEFAULT_L1_SMALL_SIZE, 900000000, DispatchCoreConfig{dispatch_core_type});
        CommandQueue& cq = device->command_queue();

        tt_metal::Program program[2];
        if (!initialize_program(info, device, program[0], info.slow_kernel_cycles)) {
            if constexpr (std::is_same_v<T, benchmark::State>) {
                state.SkipWithError("Program creation failed");
            }
            return 1;
        }
        if (!initialize_program(info, device, program[1], info.fast_kernel_cycles)) {
            if constexpr (std::is_same_v<T, benchmark::State>) {
                state.SkipWithError("Program creation failed");
            }
            return 1;
        }

        // Cache stuff
        for (int i = 0; i < info.warmup_iterations; i++) {
            EnqueueProgram(cq, program[0], false);
            for (int j = 0; j < info.nfast_kernels; j++) {
                EnqueueProgram(cq, program[1], false);
            }
        }

        auto main_program_loop = [&]() {
            for (int i = 0; i < info.iterations; i++) {
                EnqueueProgram(cq, program[0], false);
                for (int j = 0; j < info.nfast_kernels; j++) {
                    EnqueueProgram(cq, program[1], false);
                }
            }
        };
        uint32_t tid = 0;
        if (info.use_trace) {
            tid = BeginTraceCapture(device, cq.id());
            main_program_loop();
            EndTraceCapture(device, cq.id(), tid);
            Finish(cq);
        }

        if (info.lazy) {
            tt_metal::detail::SetLazyCommandQueueMode(true);
        }

        for (auto _ : state) {
            auto start = std::chrono::system_clock::now();
            if (info.use_trace) {
                EnqueueTrace(cq, tid, false);
            } else {
                main_program_loop();
            }
            if (info.time_just_finish) {
                start = std::chrono::system_clock::now();
            }
            Finish(cq);
            auto end = std::chrono::system_clock::now();

            if constexpr (std::is_same_v<T, benchmark::State>) {
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                state.SetIterationTime(elapsed_seconds.count());
            } else {
                std::chrono::duration<double> elapsed_seconds = (end - start);
                log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);
                log_info(LogTest, "Ran in {}us per iteration", elapsed_seconds.count() * 1000 * 1000 / info.iterations);
            }
        }

        if constexpr (std::is_same_v<T, benchmark::State>) {
            state.counters["IterationTime"] = benchmark::Counter(
                info.iterations, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
        }

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    tt::llrt::RunTimeOptions::get_instance().set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        if constexpr (std::is_same_v<T, benchmark::State>) {
            state.SkipWithError("Test failed");
        } else {
            log_info(LogTest, "Test failed");
        }
        return 1;
    }
}

static void BM_pgm_dispatch(benchmark::State& state, TestInfo info) { pgm_dispatch(state, info); }

static void Max12288Args(benchmark::internal::Benchmark* b) {
    b->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->Arg(12288);
}

static void Max8192Args(benchmark::internal::Benchmark* b) {
    b->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192);
}

BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    brisc_only_trace,
    TestInfo{.warmup_iterations = 5000, .ncrisc_enabled = false, .trisc_enabled = false, .use_trace = true})
    ->Apply(Max12288Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    ncrisc_only_trace,
    TestInfo{.warmup_iterations = 5000, .brisc_enabled = false, .trisc_enabled = false, .use_trace = true})
    ->Apply(Max12288Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    trisc_only_trace,
    TestInfo{.warmup_iterations = 5000, .brisc_enabled = false, .ncrisc_enabled = false, .use_trace = true})
    ->Apply(Max12288Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    brisc_trisc_only_trace,
    TestInfo{.warmup_iterations = 5000, .ncrisc_enabled = false, .use_trace = true})
    ->Apply(Max12288Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_pgm_dispatch, all_processors_trace, TestInfo{.warmup_iterations = 5000, .use_trace = true})
    ->Apply(Max12288Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_all_cores_trace,
    TestInfo{.warmup_iterations = 5000, .use_trace = true, .use_all_cores = true})
    ->Apply(Max12288Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_all_cores_1cb,
    TestInfo{.warmup_iterations = 5000, .n_cbs = 1, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_all_cores_32cb,
    TestInfo{.warmup_iterations = 5000, .n_cbs = 32, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch, all_processors_1_core_1_rta, TestInfo{.warmup_iterations = 5000, .n_args = 1, .use_trace = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    one_processor_all_cores_128_rta,
    TestInfo{
        .warmup_iterations = 5000,
        .n_args = 128,
        .ncrisc_enabled = false,
        .trisc_enabled = false,
        .use_trace = true,
        .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    one_processors_all_cores_1_rta,
    TestInfo{
        .warmup_iterations = 5000,
        .n_args = 1,
        .ncrisc_enabled = false,
        .trisc_enabled = false,
        .use_trace = true,
        .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_all_cores_1_rta,
    TestInfo{.warmup_iterations = 5000, .n_args = 1, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_all_cores_32_rta,
    TestInfo{.warmup_iterations = 5000, .n_args = 32, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_all_cores_128_rta,
    TestInfo{.warmup_iterations = 5000, .n_args = 128, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    sems_1_core_1_processor_trace,
    TestInfo{
        .warmup_iterations = 5000, .n_sems = 4, .ncrisc_enabled = false, .trisc_enabled = false, .use_trace = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    sems_all_cores_1_processor_trace,
    TestInfo{
        .warmup_iterations = 5000,
        .n_sems = 4,
        .ncrisc_enabled = false,
        .trisc_enabled = false,
        .use_trace = true,
        .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    maxed_config_params_trace,
    TestInfo{
        .warmup_iterations = 5000, .n_cbs = 32, .n_args = 128, .n_sems = 4, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    kernel_groups_trace,
    TestInfo{.warmup_iterations = 5000, .n_kgs = 8, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    if (test_args::has_command_option(input_args, "--custom")) {
        TestInfo info;
        init(input_args, info);
        FakeBenchmarkState state;
        return pgm_dispatch(state, info);
    }

    auto core_count = get_core_count();

    benchmark::RegisterBenchmark(
        "BM_pgm_dispatch/kernel_groups_4_shadow",
        BM_pgm_dispatch,
        TestInfo{
            .warmup_iterations = 5000,
            .slow_kernel_cycles = 40000,
            .nfast_kernels = 4,
            .n_kgs = std::get<0>(core_count),
            .use_trace = true,
            .use_all_cores = true})
        ->Apply(Max8192Args)
        ->UseManualTime();
    benchmark::RegisterBenchmark(
        "BM_pgm_dispatch/kernel_groups_5_shadow",
        BM_pgm_dispatch,
        TestInfo{
            .warmup_iterations = 5000,
            .slow_kernel_cycles = 40000,
            .nfast_kernels = 5,
            .n_kgs = std::get<0>(core_count),
            .use_trace = true,
            .use_all_cores = true})
        ->Apply(Max8192Args)
        ->UseManualTime();
    if (getenv("ARCH_NAME") == std::string("wormhole_b0")) {
        benchmark::RegisterBenchmark(
            "BM_pgm_dispatch/eth_dispatch",
            BM_pgm_dispatch,
            TestInfo{
                .warmup_iterations = 5000,
                .brisc_enabled = false,
                .ncrisc_enabled = false,
                .trisc_enabled = false,
                .erisc_enabled = true,
                .use_trace = true})
            ->Apply(Max8192Args)
            ->UseManualTime();
        benchmark::RegisterBenchmark(
            "BM_pgm_dispatch/tensix_eth_2",
            BM_pgm_dispatch,
            TestInfo{
                .warmup_iterations = 5000,
                .n_args = 16,
                .n_kgs = std::get<0>(core_count),
                .erisc_enabled = true,
                .use_trace = true,
                .use_all_cores = true})
            ->Apply(Max8192Args)
            ->UseManualTime();
        benchmark::RegisterBenchmark(
            "BM_pgm_dispatch/tensix_eth_2_4_shadow",
            BM_pgm_dispatch,
            TestInfo{
                .warmup_iterations = 5000,
                .slow_kernel_cycles = 40000,
                .nfast_kernels = 4,
                .n_args = 16,
                .n_kgs = std::get<0>(core_count),
                .erisc_enabled = true,
                .use_trace = true,
                .use_all_cores = true})
            ->Apply(Max8192Args)
            ->UseManualTime();
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
