// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include <chrono>
#include <fmt/base.h>
#include <cstdint>
#include "impl/dispatch/command_queue.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <array>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/buffers/semaphore.hpp"
#include <tt_stl/span.hpp>
#include "test_common.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/math.hpp>
#include "tt_metal/impl/dispatch/device_command.hpp"
#include <tt-metalium/sub_device.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <distributed/mesh_device_impl.hpp>

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 100;
constexpr uint32_t MIN_KERNEL_SIZE_BYTES = 32;  // overhead
constexpr uint32_t DEFAULT_KERNEL_SIZE_K = 1;
constexpr uint32_t MAX_CBS = 32;
constexpr uint32_t MAX_ARGS = 341;

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Times dispatching program to M cores, N processors, of various sizes, CBs, runtime args
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal::distributed;

static bool dump_test_info = false;

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
    uint32_t n_cb_gs{1};
    uint32_t n_subdevice_ranges{1};
    bool brisc_enabled{true};
    bool ncrisc_enabled{true};
    bool trisc_enabled{true};
    bool erisc_enabled{false};
    uint32_t erisc_count{1};
    bool time_just_finish{false};
    bool use_global{false};
    bool use_trace{false};
    bool dispatch_from_eth{false};
    bool use_all_cores{false};
    bool load_prefetcher{false};
    // Use the entire leftmost column of cores.
    bool use_left_cores{false};
};

// Returns the address of the last core in the last column of the mesh. Picks a configuration that works with worst-case
// harvesting, for consistency.
std::tuple<uint32_t, uint32_t> get_core_count() {
    uint32_t core_x = 0;
    uint32_t core_y = 0;

    std::string arch_name = tt::tt_metal::hal::get_arch_name();
    if (arch_name == "wormhole_b0") {
        // Three rows harvested.
        core_x = 7;
        core_y = 6;
    } else if (arch_name == "blackhole") {
        // Two columns and one row harvested.
        core_x = 11;
        core_y = 8;
    } else {
        log_fatal(tt::LogTest, "Unexpected ARCH_NAME {}", arch_name);
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
        log_info(LogTest, " -sd: number of subdevices core ranges (default 1)");
        log_info(LogTest, "  -g: use a 4 byte global variable (additional spans");
        log_info(LogTest, " -rs: run \"slow\" kernels for exactly <n> cycles (default 0)");
        log_info(LogTest, " -rf: run \"fast\" kernels for exactly <n> cycles (default 0)");
        log_info(LogTest, " -nf: run <n> fast kernels between slow kernels (default 0)");
        log_info(LogTest, "  -b: disable brisc kernel (default enabled)");
        log_info(LogTest, "  -n: disable ncrisc kernel (default enabled)");
        log_info(LogTest, "  -t: disable trisc kernels (default enabled)");
        log_info(LogTest, "  +e: enable erisc kernels (default disabled)");
        log_info(LogTest, " -ec: erisc count (default 1 if enabled)");
        log_info(LogTest, "  -f: time just the finish call (default disabled)");
        log_info(LogTest, " -tr: enable trace (default disabled)");
        log_info(LogTest, " -de: dispatch from eth cores (default tensix)");
        log_info(
            LogTest,
            " -ac: use all viable worker cores (default {}x{})",
            std::get<0>(core_count),
            std::get<1>(core_count));
        log_info(LogTest, " -pfl: test prefetcher cache load performance (default disabled)");
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
    info.n_subdevice_ranges = test_args::get_command_option_uint32(input_args, "-sd", 1);
    info.use_global = test_args::has_command_option(input_args, "-g");
    info.time_just_finish = test_args::has_command_option(input_args, "-f");
    info.fast_kernel_cycles = test_args::get_command_option_uint32(input_args, "-rf", 0);
    info.slow_kernel_cycles = test_args::get_command_option_uint32(input_args, "-rs", 0);
    info.nfast_kernels = test_args::get_command_option_uint32(input_args, "-nf", 0);
    info.use_trace = test_args::has_command_option(input_args, "-tr");
    info.dispatch_from_eth = test_args::has_command_option(input_args, "-de");
    info.load_prefetcher = test_args::has_command_option(input_args, "-pfl");
    if (info.kernel_size < MIN_KERNEL_SIZE_BYTES) {
        log_fatal(tt::LogTest, "Minimum kernel size is {} bytes", MIN_KERNEL_SIZE_BYTES);
        exit(0);
    }
    if (info.n_cbs > MAX_CBS) {
        log_fatal(tt::LogTest, "CB count must be 0..{}", MAX_CBS);
        exit(0);
    }
    if (info.n_args > MAX_ARGS) {
        log_fatal(tt::LogTest, "Runtime arg count must be 0..{}", MAX_ARGS);
        exit(0);
    }
    if (info.n_common_args > MAX_ARGS) {
        log_fatal(tt::LogTest, "Common Runtime arg count must be 0..{}", MAX_ARGS);
        exit(0);
    }
    if (info.n_sems > NUM_SEMAPHORES) {
        log_fatal(tt::LogTest, "Sem count must be 0..{}", NUM_SEMAPHORES);
        exit(0);
    }
    if (info.n_kgs > core_x + 1) {
        log_fatal(
            tt::LogTest, "This test uses columns for kernel groups so number of kernel groups must be <= x core range");
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
            tt::LogTest,
            "The number of fast kernels is non-zero, but slow_kernel_ cycles ({}) is <= fast_kernel_cycles ({})",
            info.slow_kernel_cycles,
            info.fast_kernel_cycles);
        log_error(tt::LogTest, "For meaningful results, run multiple fast kernels between single slow kernels");
        exit(0);
    }
}

void set_runtime_args(
    tt_metal::Program& program, tt_metal::KernelHandle kernel_id, vector<uint32_t>& args, const CoreRangeSet& kgset) {
    for (const auto& kg : kgset.ranges()) {
        for (int core_idx_y = kg.start_coord.y; core_idx_y <= kg.end_coord.y; core_idx_y++) {
            for (int core_idx_x = kg.start_coord.x; core_idx_x <= kg.end_coord.x; core_idx_x++) {
                CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};
                tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
            }
        }
    }
}
tt_metal::CoreRangeSet get_subdevice_core_range_set(const TestInfo& info, CoreRange all_core_range) {
    std::set<CoreRange> core_range_set;
    uint32_t total_core_x = all_core_range.end_coord.x - all_core_range.start_coord.x + 1;
    if (info.n_subdevice_ranges > all_core_range.end_coord.x + 1) {
        log_fatal(tt::LogTest, "Too many subdevice ranges for Worker core width");
    }
    if (info.n_subdevice_ranges > all_core_range.end_coord.y + 1) {
        log_fatal(tt::LogTest, "Too many subdevice ranges for Worker core height");
    }
    // Construct/filter each column individually.
    for (size_t i = 0; i < total_core_x; i++) {
        uint32_t subdevice_subtract_amount = 0;
        if (i >= total_core_x - info.n_subdevice_ranges) {
            // First subdevice range is wide, remaining columns shrink by 1 each time.
            uint32_t offset = i - (total_core_x - info.n_subdevice_ranges);
            subdevice_subtract_amount = offset;
        }

        CoreRange column_core_range{
            CoreCoord(all_core_range.start_coord.x + i, all_core_range.start_coord.y),
            CoreCoord(all_core_range.start_coord.x + i, all_core_range.end_coord.y - subdevice_subtract_amount)};

        core_range_set.insert(column_core_range);
    }

    return tt_metal::CoreRangeSet{core_range_set};
}

uint32_t get_num_kernels(const TestInfo& info) {
    uint32_t num_kernels = 0;
    if (info.brisc_enabled) {
        num_kernels++;
    }
    if (info.ncrisc_enabled) {
        num_kernels++;
    }
    if (info.trisc_enabled) {
        num_kernels += 3;  // 3 compute kernels when enabled
    }
    if (info.erisc_enabled) {
        num_kernels++;
    }
    return num_kernels;
}

bool initialize_program(
    const TestInfo& info,
    const std::shared_ptr<MeshDevice>& mesh_device,
    tt_metal::Program& program,
    uint32_t run_cycles) {
    program = tt_metal::CreateProgram();

    std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(info.kernel_size)}};
    if (run_cycles != 0) {
        defines.insert(std::pair<std::string, std::string>("KERNEL_RUN_TIME", std::to_string(run_cycles)));
    }
    if (info.use_global) {
        defines.insert(std::pair<std::string, std::string>("KERNEL_GLOBAL", "1"));
    }

    for (uint32_t i = 0; i < info.n_sems; i++) {
        tt_metal::CreateSemaphore(program, info.workers, 3);
    }

    vector<uint32_t> args;
    args.resize(info.n_args);
    vector<uint32_t> common_args;
    common_args.resize(info.n_common_args);

    CoreRange cbg = {info.workers.start_coord, {info.workers.end_coord.x - info.n_cb_gs + 1, info.workers.end_coord.y}};

    for (uint32_t i = 0; i < info.n_cb_gs; i++) {
        for (int j = 0; j < info.n_cbs; j++) {
            tt_metal::CircularBufferConfig cb_config =
                tt_metal::CircularBufferConfig(16, {{j, tt::DataFormat::Float16_b}}).set_page_size(j, 16);
            tt_metal::CreateCircularBuffer(program, cbg, cb_config);
        }
        cbg.start_coord = {cbg.end_coord.x + 1, cbg.end_coord.y};
        cbg.end_coord = cbg.start_coord;
    }

    // first kernel group is possibly wide, remaining kernel groups are 1 column each
    CoreRange total_kg = {
        info.workers.start_coord, {info.workers.end_coord.x - info.n_kgs + 1, info.workers.end_coord.y}};
    std::array<CoreRangeSet, NumHalProgrammableCoreTypes> core_ranges;
    auto grid_size = mesh_device->compute_with_storage_grid_size();
    CoreRange all_core_range{{0, 0}, {grid_size.x - 1, grid_size.y - 1}};
    CoreRangeSet subdevice_core_ranges_set = get_subdevice_core_range_set(info, all_core_range);
    for (uint32_t i = 0; i < info.n_kgs; i++) {
        defines.insert(std::pair<std::string, std::string>(std::string("KG_") + std::to_string(i), ""));
        CoreRangeSet kg = CoreRangeSet{total_kg}.intersection(subdevice_core_ranges_set);

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

        total_kg.end_coord.x++;
        total_kg.start_coord.x = total_kg.end_coord.x;
    }

    if (info.erisc_enabled) {
        auto erisc_cores = mesh_device->impl().get_device(0, 0)->get_active_ethernet_cores(true);
        if (info.erisc_count > erisc_cores.size()) {
            log_fatal(
                tt::LogTest,
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

// Helper structure to encapsulate program execution logic
struct ProgramExecutor {
    std::function<void()> execute_programs;
    std::function<void()> warmup_programs;
    uint32_t total_program_iterations;

    ProgramExecutor(std::function<void()> exec, std::function<void()> warm, uint32_t total_iters) :
        execute_programs(std::move(exec)), warmup_programs(std::move(warm)), total_program_iterations(total_iters) {}
};

// Helper function to create program executor for standard test
ProgramExecutor create_standard_executor(
    const TestInfo& info,
    std::vector<MeshWorkload>& mesh_workloads,
    std::array<tt_metal::Program, 2>& programs,
    MeshCommandQueue& mesh_cq) {
    // Create mesh workloads
    mesh_workloads.resize(programs.size());
    for (auto i = 0; i < programs.size(); i++) {
        mesh_workloads[i].add_program(
            MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)), std::move(programs[i]));
    }
    std::function warmup_func{[&info, &mesh_cq, &mesh_workloads]() {
        for (int i = 0; i < info.warmup_iterations; i++) {
            EnqueueMeshWorkload(mesh_cq, mesh_workloads[0], false);
            for (int j = 0; j < info.nfast_kernels; j++) {
                EnqueueMeshWorkload(mesh_cq, mesh_workloads[1], false);
            }
        }
    }};

    std::function execute_func{[&info, &mesh_cq, &mesh_workloads]() {
        for (int i = 0; i < info.iterations; i++) {
            EnqueueMeshWorkload(mesh_cq, mesh_workloads[0], false);
            for (int j = 0; j < info.nfast_kernels; j++) {
                EnqueueMeshWorkload(mesh_cq, mesh_workloads[1], false);
            }
        }
    }};

    return ProgramExecutor(execute_func, warmup_func, info.iterations);
}

// Helper function to create program executor for prefetcher cache load test
ProgramExecutor create_load_prefetcher_executor(
    const TestInfo& info,
    std::vector<MeshWorkload>& mesh_workloads,
    std::vector<tt_metal::Program>& programs,
    MeshCommandQueue& mesh_cq) {
    // Create mesh workload
    mesh_workloads.resize(programs.size());
    for (auto i = 0; i < programs.size(); i++) {
        mesh_workloads[i].add_program(
            MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)), std::move(programs[i]));
    }

    std::function warmup_func{[&info, &mesh_cq, &mesh_workloads]() {
        for (int i = 0; i < info.warmup_iterations; i++) {
            for (auto& mesh_workload : mesh_workloads) {
                EnqueueMeshWorkload(mesh_cq, mesh_workload, false);
            }
        }
    }};

    std::function execute_func{[&info, &mesh_cq, &mesh_workloads]() {
        for (int i = 0; i < info.iterations; i++) {
            for (auto& mesh_workload : mesh_workloads) {
                EnqueueMeshWorkload(mesh_cq, mesh_workload, false);
            }
        }
    }};

    return ProgramExecutor(execute_func, warmup_func, info.iterations * programs.size());
}

// Helper function to setup trace if enabled
template <typename T>
MeshTraceId setup_trace_if_enabled(
    const TestInfo& info, const std::shared_ptr<MeshDevice>& mesh_device, ProgramExecutor& executor) {
    MeshTraceId tid;
    if (info.use_trace) {
        const std::size_t cq_id = 0;
        tid = BeginTraceCapture(mesh_device.get(), cq_id);
        executor.execute_programs();
        mesh_device->end_mesh_trace(cq_id, tid);
        Finish(mesh_device->mesh_command_queue(cq_id));
    }
    return tid;
}

// Helper function to run the benchmark timing loop
template <typename T>
void run_benchmark_timing_loop(
    T& state,
    const TestInfo& info,
    MeshCommandQueue& mesh_cq,
    ProgramExecutor& executor,
    MeshTraceId tid,
    const std::shared_ptr<MeshDevice>& mesh_device) {
    constexpr std::size_t cq_id = 0;
    auto execute_func = executor.execute_programs;
    for ([[maybe_unused]] auto _ : state) {
        auto start = std::chrono::system_clock::now();
        if (info.use_trace) {
            mesh_device->replay_mesh_trace(cq_id, tid, false);
        } else {
            execute_func();
        }
        if (info.time_just_finish) {
            start = std::chrono::system_clock::now();
        }
        Finish(mesh_cq);
        auto end = std::chrono::system_clock::now();

        if constexpr (std::is_same_v<T, benchmark::State>) {
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            state.SetIterationTime(elapsed_seconds.count());
        } else {
            std::chrono::duration<double> elapsed_seconds = (end - start);
            log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);
            log_info(
                LogTest,
                "Ran in {}us per iteration",
                elapsed_seconds.count() * 1000 * 1000 / executor.total_program_iterations);
        }
    }
}

// Helper function to set benchmark counters
template <typename T>
void set_benchmark_counters(
    T& state,
    const TestInfo& info,
    tt_metal::IDevice* device,
    uint32_t total_iterations,
    const std::unordered_map<std::string, uint32_t>& extra_counters = {}) {
    if constexpr (std::is_same_v<T, benchmark::State>) {
        if (dump_test_info) {
            state.counters["cores"] = benchmark::Counter(info.workers.size(), benchmark::Counter::kDefaults);
            state.counters["kernel_size"] = benchmark::Counter(info.kernel_size, benchmark::Counter::kDefaults);
            state.counters["fast_kernel_cycles"] =
                benchmark::Counter(info.fast_kernel_cycles, benchmark::Counter::kDefaults);
            state.counters["slow_kernel_cycles"] =
                benchmark::Counter(info.slow_kernel_cycles, benchmark::Counter::kDefaults);
            state.counters["kgs"] = benchmark::Counter(info.n_kgs, benchmark::Counter::kDefaults);
            state.counters["cbs"] = benchmark::Counter(info.n_cbs, benchmark::Counter::kDefaults);
            state.counters["rtas"] = benchmark::Counter(info.n_args, benchmark::Counter::kDefaults);
            state.counters["crtas"] = benchmark::Counter(info.n_common_args, benchmark::Counter::kDefaults);
            state.counters["sems"] = benchmark::Counter(info.n_sems, benchmark::Counter::kDefaults);
            state.counters["brisc_enabled"] = benchmark::Counter(info.brisc_enabled, benchmark::Counter::kDefaults);
            state.counters["ncrisc_enabled"] = benchmark::Counter(info.ncrisc_enabled, benchmark::Counter::kDefaults);
            state.counters["trisc_enabled"] = benchmark::Counter(info.trisc_enabled, benchmark::Counter::kDefaults);
            state.counters["erisc_enabled"] = benchmark::Counter(info.erisc_enabled, benchmark::Counter::kDefaults);
            state.counters["cb_gs"] = benchmark::Counter(info.n_cb_gs, benchmark::Counter::kDefaults);

            // Add extra counters
            for (const auto& [key, value] : extra_counters) {
                state.counters[key] = benchmark::Counter(value, benchmark::Counter::kDefaults);
            }
        }

        state.counters["IterationTime"] = benchmark::Counter(
            total_iterations, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
        state.counters["Clock"] = benchmark::Counter(get_tt_npu_clock(device), benchmark::Counter::kDefaults);
    }
}

// Helper function to convert from DispatchCoreType to CoreType
CoreType dispatch_core_type_to_core_type(DispatchCoreType dispatch_core_type) {
    switch (dispatch_core_type) {
        case DispatchCoreType::WORKER: return CoreType::WORKER;
        case DispatchCoreType::ETH: return CoreType::ETH;
        default: TT_THROW("invalid dispatch core type");
    }
}

// Helper function to create standard programs
std::array<tt_metal::Program, 2> create_standard_programs(
    const TestInfo& info, const std::shared_ptr<MeshDevice>& mesh_device, DispatchCoreType /*dispatch_core_type*/) {
    std::array<tt_metal::Program, 2> programs;
    if (!initialize_program(info, mesh_device, programs[0], info.slow_kernel_cycles) ||
        !initialize_program(info, mesh_device, programs[1], info.fast_kernel_cycles)) {
        throw std::runtime_error("Standard program creation failed");
    }
    return programs;
}
// Helper function to create prefetcher cache load programs
std::pair<std::vector<tt_metal::Program>, std::unordered_map<std::string, uint32_t>> create_load_prefetcher_programs(
    const TestInfo& info, const std::shared_ptr<MeshDevice>& mesh_device, DispatchCoreType dispatch_core_type) {
    uint32_t prefetcher_cache_size = tt::tt_metal::MetalContext::instance()
                                         .dispatch_mem_map(dispatch_core_type_to_core_type(dispatch_core_type))
                                         .ringbuffer_size();
    uint32_t target_total_size = (3 * prefetcher_cache_size) / 2;
    uint32_t num_kernels = get_num_kernels(info);
    uint32_t estimated_program_size =
        tt::align(info.kernel_size * num_kernels, tt::tt_metal::HostMemDeviceCommand::PROGRAM_PAGE_SIZE);
    uint32_t num_programs = tt::div_up(target_total_size, estimated_program_size);

    log_info(LogTest, "Prefetcher cache load test: prefetcher cache size = {} bytes", prefetcher_cache_size);
    log_info(
        LogTest,
        "Estimated program size = {} bytes, target total program size = {} bytes (1.5x cache)",
        estimated_program_size,
        target_total_size);
    log_info(LogTest, "Creating {} programs for cache overflow test", num_programs);

    std::vector<tt_metal::Program> programs(num_programs);
    uint32_t kernel_runtime = 2000;  // cycles - short enough to focus on dispatch time

    for (auto& program : programs) {
        if (!initialize_program(info, mesh_device, program, kernel_runtime)) {
            throw std::runtime_error("Program creation failed for prefetcher cache load test");
        }
    }

    std::unordered_map<std::string, uint32_t> extra_counters = {
        {"num_programs", num_programs},
        {"prefetcher_cache_size", prefetcher_cache_size},
        {"target_program_size", target_total_size}};

    return {std::move(programs), extra_counters};
}

// Helper function to log test configuration
void log_test_configuration(const TestInfo& info) {
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
    log_info(LogTest, "Subdevice core ranges: {}", info.n_subdevice_ranges);
    log_info(LogTest, "CBs: {}", info.n_cbs);
    log_info(LogTest, "UniqueRTArgs: {}", info.n_args);
    log_info(LogTest, "CommonRTArgs: {}", info.n_common_args);
    log_info(LogTest, "Sems: {}", info.n_sems);

    if (info.load_prefetcher) {
        log_info(LogTest, "Prefetcher cache load test: ENABLED");
    }
}

template <typename T>
static int pgm_dispatch(T& state, TestInfo info) {
    if constexpr (std::is_same_v<T, benchmark::State>) {
        log_info(LogTest, "Running {}", state.name());
    }

    // Apply configuration adjustments
    if (info.use_all_cores) {
        auto core_count = get_core_count();
        info.workers = CoreRange({0, 0}, {std::get<0>(core_count), std::get<1>(core_count)});
    }
    if (info.use_left_cores) {
        auto core_count = get_core_count();
        info.workers = CoreRange({0, 0}, {0, std::get<1>(core_count)});
    }

    log_test_configuration(info);

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    bool pass = true;
    std::shared_ptr<MeshDevice> mesh_device;
    try {
        const ChipId device_id = 0;
        const std::size_t cq_id = 0;
        DispatchCoreType dispatch_core_type = info.dispatch_from_eth ? DispatchCoreType::ETH : DispatchCoreType::WORKER;
        size_t trace_region_size = 1'000'000'000;
        std::string arch_name = tt::tt_metal::hal::get_arch_name();
        if (arch_name == std::string("blackhole")) {
            // Blackhole has more cores, so we need more room to store RTAs.
            trace_region_size = 2'500'000'000;
        }
        mesh_device = MeshDevice::create_unit_mesh(
            device_id, DEFAULT_L1_SMALL_SIZE, trace_region_size, 1, DispatchCoreConfig{dispatch_core_type});
        auto& mesh_cq = mesh_device->mesh_command_queue(cq_id);

        std::vector<tt_metal::SubDevice> sub_devices;
        if (info.n_subdevice_ranges > 1) {
            std::array<CoreRangeSet, NumHalProgrammableCoreTypes> core_ranges;
            auto grid_size = mesh_device->compute_with_storage_grid_size();
            CoreRange all_core_range{{0, 0}, {grid_size.x - 1, grid_size.y - 1}};
            core_ranges[static_cast<size_t>(HalProgrammableCoreType::TENSIX)] =
                get_subdevice_core_range_set(info, all_core_range);
            sub_devices.push_back(tt_metal::SubDevice(core_ranges));
            auto manager = mesh_device->create_sub_device_manager(sub_devices, 1024);
            mesh_device->load_sub_device_manager(manager);
        }

        // Declare program storage at function scope to ensure proper lifetime
        ProgramExecutor executor([]() {}, []() {}, 0);  // Initialize with placeholder
        std::vector<MeshWorkload> mesh_workloads;
        if (info.load_prefetcher) {
            auto [programs, extra_counters] = create_load_prefetcher_programs(info, mesh_device, dispatch_core_type);
            executor = create_load_prefetcher_executor(info, mesh_workloads, programs, mesh_cq);
            // Store extra counters for later use
            if constexpr (std::is_same_v<T, benchmark::State>) {
                if (dump_test_info) {
                    for (const auto& [key, value] : extra_counters) {
                        state.counters[key] = benchmark::Counter(value, benchmark::Counter::kDefaults);
                    }
                }
            }
        } else {
            auto programs = create_standard_programs(info, mesh_device, dispatch_core_type);
            executor = create_standard_executor(info, mesh_workloads, programs, mesh_cq);
        }

        // Set benchmark counters before timing (all values are known at this point)
        set_benchmark_counters(state, info, mesh_device->impl().get_device(0, 0), executor.total_program_iterations);

        // Run warmup
        executor.warmup_programs();

        // Setup trace if enabled
        MeshTraceId tid = setup_trace_if_enabled<T>(info, mesh_device, executor);

        // Run benchmark timing loop
        run_benchmark_timing_loop(state, info, mesh_cq, executor, tid, mesh_device);

        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        pass = false;
        mesh_device->close();
        log_fatal(tt::LogTest, "{}", e.what());
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    }
    if constexpr (std::is_same_v<T, benchmark::State>) {
        state.SkipWithError("Test failed");
    } else {
        log_info(LogTest, "Test failed");
    }
    return 1;
}

static void BM_pgm_dispatch(benchmark::State& state, TestInfo info) {
    info.kernel_size = state.range(0);
    pgm_dispatch(state, info);
}

static void BM_pgm_dispatch_vary_slow_cycles(benchmark::State& state, TestInfo info) {
    info.slow_kernel_cycles = state.range(0);
    pgm_dispatch(state, info);
}

static void BM_pgm_dispatch_random(benchmark::State& state, TestInfo info) { pgm_dispatch(state, info); }

static void Max12288Args(benchmark::internal::Benchmark* b) {
    b->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->Arg(12288);
}

static void Max8192Args(benchmark::internal::Benchmark* b) {
    b->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192);
}
static void Range512To12KArgs(benchmark::internal::Benchmark* b) {
    b->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->Arg(12288);
}

static void KernelCycleArgs(benchmark::internal::Benchmark* b) {
    // Dispatch time for most normal kernels is around 3000-4000 cycles.
    b->Arg(0)->Arg(1000)->Arg(2000)->Arg(3000)->Arg(4000)->Arg(5000)->Arg(10000);
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
    BM_pgm_dispatch,
    all_processors_all_cores_1cb_8g,
    TestInfo{.warmup_iterations = 5000, .n_cbs = 1, .n_cb_gs = 8, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_all_cores_32cb_8g,
    TestInfo{.warmup_iterations = 5000, .n_cbs = 32, .n_cb_gs = 8, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_all_cores_1cb_1sem,
    TestInfo{.warmup_iterations = 5000, .n_cbs = 1, .n_sems = 1, .use_trace = true, .use_all_cores = true})
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
    all_processors_1_core_1_crta,
    TestInfo{.warmup_iterations = 5000, .n_common_args = 1, .use_trace = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_1_core_128_crta,
    TestInfo{.warmup_iterations = 5000, .n_common_args = 128, .use_trace = true})
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
    all_processors_all_cores_1_crta,
    TestInfo{.warmup_iterations = 5000, .n_common_args = 1, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_all_cores_32_crta,
    TestInfo{.warmup_iterations = 5000, .n_common_args = 32, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    all_processors_all_cores_128_crta,
    TestInfo{.warmup_iterations = 5000, .n_common_args = 128, .use_trace = true, .use_all_cores = true})
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
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    kernel_groups_1_rta_trace,
    TestInfo{.warmup_iterations = 5000, .n_args = 1, .n_kgs = 8, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    kernel_groups_128_rta_trace,
    TestInfo{.warmup_iterations = 5000, .n_args = 128, .n_kgs = 8, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();

BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    kernel_groups_1_cb_trace,
    TestInfo{.warmup_iterations = 5000, .n_cbs = 1, .n_kgs = 8, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    kernel_groups_32_cb_trace,
    TestInfo{.warmup_iterations = 5000, .n_cbs = 32, .n_kgs = 8, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();

BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    10000_kernel_all_cores_all_processors_32_cbs_trace,
    TestInfo{
        .warmup_iterations = 5000, .slow_kernel_cycles = 10000, .n_cbs = 32, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    5000_kernel_all_cores_all_processors_32_cbs_trace,
    TestInfo{
        .warmup_iterations = 5000, .slow_kernel_cycles = 5000, .n_cbs = 32, .use_trace = true, .use_all_cores = true})
    ->Apply(Max8192Args)
    ->UseManualTime();
// Intended to be GO-latency-bound
BENCHMARK_CAPTURE(
    BM_pgm_dispatch_vary_slow_cycles,
    256_bytes_brisc_only_all_processors_trace,
    TestInfo{
        .warmup_iterations = 5000,
        .kernel_size = 256,
        .ncrisc_enabled = false,
        .trisc_enabled = false,
        .use_trace = true,
        .use_all_cores = true})
    ->Apply(KernelCycleArgs)
    ->UseManualTime();
BENCHMARK_CAPTURE(
    BM_pgm_dispatch_vary_slow_cycles,
    256_bytes_brisc_only_left_processors_subdevices_trace,
    TestInfo{
        .warmup_iterations = 5000,
        .kernel_size = 256,
        .n_subdevice_ranges = 6,
        .ncrisc_enabled = false,
        .trisc_enabled = false,
        .use_trace = true,
        // Use only the left column to allow for a single CoreRange in the kernel group.
        .use_left_cores = true})
    ->Apply(KernelCycleArgs)
    ->UseManualTime();

// Prefetcher cache load performance test - measures dispatch time with programs 2x cache size
BENCHMARK_CAPTURE(
    BM_pgm_dispatch,
    load_prefetcher_test,
    TestInfo{.iterations = 5000, .warmup_iterations = 1000, .use_trace = true, .load_prefetcher = true})
    ->Apply(Range512To12KArgs)
    ->UseManualTime();

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    if (test_args::has_command_option(input_args, "--custom")) {
        TestInfo info;
        init(input_args, info);
        FakeBenchmarkState state;
        return pgm_dispatch(state, info);
    }

    if (test_args::has_command_option(input_args, "--dump-test-info")) {
        dump_test_info = true;
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

    if (test_args::has_command_option(input_args, "--random-tests")) {
        // Generate random tests to use when generating a model of dispatch time.
        std::mt19937 gen;

        for (size_t i = 0; i < 400; i++) {
            uint32_t core_count_x = std::uniform_int_distribution{1u, std::get<0>(core_count)}(gen);
            uint32_t core_count_y = std::uniform_int_distribution{1u, std::get<1>(core_count)}(gen);
            TestInfo test_info{};
            test_info.kernel_size = std::uniform_int_distribution(0, 12288)(gen);
            test_info.workers = CoreRange({0, 0}, {core_count_x - 1, core_count_y - 1});
            bool has_cbs = std::uniform_int_distribution(0, 1)(gen);
            if (has_cbs) {
                test_info.n_cbs = std::uniform_int_distribution(0, 32)(gen);
            }
            bool has_args = std::uniform_int_distribution(0, 1)(gen);
            if (has_args) {
                test_info.n_args = std::uniform_int_distribution(0, 128)(gen);
            }
            bool has_sems = std::uniform_int_distribution(0, 1)(gen);
            if (has_sems) {
                test_info.n_sems = std::uniform_int_distribution(0, 4)(gen);
            }
            test_info.n_kgs = std::uniform_int_distribution(1u, core_count_x)(gen);
            test_info.n_cb_gs = std::uniform_int_distribution(1u, core_count_x)(gen);
            // Ensure 1 core is enabled.
            uint32_t cores_enabled = std::uniform_int_distribution(1, 7)(gen);
            test_info.brisc_enabled = cores_enabled & 1;
            test_info.ncrisc_enabled = cores_enabled & 2;
            test_info.trisc_enabled = cores_enabled & 4;
            test_info.use_trace = true;
            benchmark::RegisterBenchmark(
                "BM_pgm_dispatch_random/" + std::to_string(i), BM_pgm_dispatch_random, test_info)
                ->UseManualTime();
        }
    }
    std::string arch_name = tt::tt_metal::hal::get_arch_name();
    if (arch_name == std::string("wormhole_b0")) {
        benchmark::RegisterBenchmark(
            "BM_pgm_dispatch/eth_dispatch",
            BM_pgm_dispatch,
            TestInfo{
                .warmup_iterations = 5000,
                .brisc_enabled = true,
                .ncrisc_enabled = false,
                .trisc_enabled = false,
                .erisc_enabled = false,
                .use_trace = true,
                .dispatch_from_eth = true})
            ->Apply(Max8192Args)
            ->UseManualTime();
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
