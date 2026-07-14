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
// mode. The compute kernel emits per-tile DeviceZoneScopedN("MATH") zones
// (plus TILE_IDX data) so a profiler-enabled build dumps timestamps into
// generated/profiler/.logs/profile_log_device.csv.
//
// For **program-level** op-to-op (between host enqueues), use
// `--use-realtime-profiler` (ProgramRealtimeRecord gaps), not nested
// DeviceZoneScopedMainN in user TRISC code (see compute kernel comments).
//
// CSV / in-kernel timing: use per-tile MATH zones on the MATH TRISC row,
// or TRISC-KERNEL zones emitted by firmware (trisck.cc) for whole-kernel
// envelope on each TRISC.
//
// CLI:
//   --num-pages-per-core N  pages of interleaved DRAM per core (default 4)
//   --compute-nops N        TTI_NOPs in the per-tile body (default 0;
//                           tune up so the math zone is comfortably > 10us)
//   --num-programs N        back-to-back enqueues per measurement (default 8)
//   --use-trace             capture once + replay (default: FD mode)
//   --use-device-profiler   call ReadMeshDeviceProfilerResults after Finish
//                           (requires Tracy-enabled build — default unless
//                            build_metal.sh --disable-profiler; plus env var
//                            TT_METAL_DEVICE_PROFILER=1 before process start)
//   --use-realtime-profiler register ProgramRealtimeProfilerCallback for the
//                           timed portion only (FD: N enqueues + Finish; trace:
//                           replay + Finish). Logs per-chip op-to-op gaps
//                           (next start - previous end, ns). Inactive on some
//                           dispatch setups; see
//                           tech_reports/real_time_profiler/getting-started.md
//   --trace-warmup-replays N  untimed trace replays before the measured replay (default 0)
//   --trace-region-size N   trace buffer size, bytes (default 1 MiB)
//   --device-id N           device under test (default 0)
//   --output-cb-depth-tiles N  output CB depth in tiles (default 2)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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
#include <tt-metalium/experimental/realtime_profiler.hpp>
#include <hostdevcommon/common_values.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"

#include "test_common.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Defaults:
//   num_pages_per_core = 4 (each core processes 4 interleaved DRAM pages, so
//                        every Tensix sees ≥ 4 per-tile MATH-zone entries
//                        and we still touch every DRAM bank in a chip-wide
//                        interleaved layout).
//   num_nops_per_tile  = 0 keeps the math zone short by default; tune this
//                        up via --compute-nops until the per-tile MATH zone
//                        in profile_log_device.csv is comfortably > 10us
//                        (the target program runtime for op-to-op latency
//                        measurement to be meaningful vs dispatch noise).
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
    uint32_t num_pages_per_core = 4;
    uint32_t num_nops_per_tile = 0;
    uint32_t num_programs = 8;
    uint32_t device_id = 0;
    uint32_t reader_push_tile_count = 2;
    // 0 = auto (2 * reader_push_tile_count) for input CB depth.
    uint32_t input_cb_depth_tiles = 0;
    // 0 = default 2 tiles (compute->writer still pops 1 at a time).
    uint32_t output_cb_depth_tiles = 0;
    // 0 = reserve N then read+push one tile at a time; 1 = reserve N, read all, push N;
    // 2 = per-trid double-buffer (N reads in flight per TRID).
    uint32_t reader_mode = 0;
    // Reads in flight per TRID for reader_mode 2. CB depth must be
    // >= 2 * reader_trid_in_flight * page_size_tiles.
    uint32_t reader_trid_in_flight = 2;
    // DRAM page size in tiles. Larger pages = fewer, larger NoC transactions
    // per program. CB pages stay 1 tile each so compute kernel is unchanged.
    uint32_t page_size_tiles = 1;
    // Cap active core count. 0 = use full grid (default). Used to test whether
    // brisc_done_to_go scales with core count.
    uint32_t num_active_cores = 0;
    // Core placement for the active set: false = row-major (fill a row then next row),
    // true = column-major (fix x, vary y first). Lets us probe NoC-direction asymmetry
    // (a row of cores contends differently on the torus than a column).
    bool core_layout_col = false;
    // Reverse the core fill order (take the LAST `want` cores instead of the first) -- the
    // mirror-image core set, to test whether which specific cores are used drives results.
    bool core_reverse = false;
    // NoC assignment for reader / writer (0 = NOC0, 1 = NOC1). The two NoCs route opposite
    // directions on the torus; measured read-BW table shows reads scale to ~206 GB/s on
    // NOC0 but cap at ~67 on NOC1, so the defaults are reader=NOC0, writer=NOC1 (the
    // opposite of the framework default reader=NOC1/writer=NOC0). Override per kernel.
    uint32_t reader_noc = 0;
    uint32_t writer_noc = 1;
    bool warmup = true;
    bool use_trace = false;
    bool use_device_profiler = false;
    bool use_realtime_profiler = false;
    size_t trace_region_size = 1ull << 20;  // 1 MiB
    // Untimed trace replays before the measured replay (steady-state trace path).
    uint32_t trace_warmup_replays = 0;
    // Read-only mode: writer pops from output CB but skips DRAM writes.
    // Isolates pure DRAM read BW through the reader pipeline.
    bool read_only = false;
    // Lean compute: drop per-tile TILE_IDX / MATH profiler markers (keep tile 0) so the
    // consumer drains at full speed and does not back-pressure the reader. Use when
    // measuring read bandwidth; compute cost is then copy + NOPs only.
    bool lean_compute = false;
    // Skip the host output-data check. The BW reader writes dummy data to a fixed L1
    // address, so the written DRAM won't match the input; for latency/BW measurement
    // (which needs real DRAM writes, i.e. not --read-only) we don't care about data.
    bool skip_output_validation = false;
    // Writer: 0 = noc_async_writes_flushed every page; 1 = flush only when output
    // CB back-pressure requires it (recommended back-pressure write path).
    bool writer_flush_on_pressure = false;
    // Each program reads/writes a disjoint per-program tile slice rather than the
    // same tiles every program. Forces DRAM controller to open new rows per
    // program (more app-like streaming pattern); allocates a buffer big enough
    // for (num_programs + 1) slices.
    bool cross_program_dram_offset = false;
    // Writer end-of-kernel barrier (Batch-8 latency experiment for 's HW
    // barrier-elimination proposal):
    //   0 = noc_async_write_barrier()  (DEFAULT; wait for DRAM ACK; safe)
    //   1 = noc_async_writes_flushed() (cheaper; writes left source but not yet ACKed)
    //   2 = no end barrier             (UNSAFE for real workloads; simulates HW
    //                                   giving us this guarantee for free)
    uint32_t writer_end_barrier_mode = 0;
};

BenchmarkConfig parse_args(const std::vector<std::string>& args) {
    BenchmarkConfig cfg;
    cfg.num_pages_per_core = test_args::get_command_option_uint32(args, "--num-pages-per-core", cfg.num_pages_per_core);
    cfg.reader_push_tile_count =
        test_args::get_command_option_uint32(args, "--reader-push-tiles", cfg.reader_push_tile_count);
    cfg.input_cb_depth_tiles =
        test_args::get_command_option_uint32(args, "--input-cb-depth-tiles", cfg.input_cb_depth_tiles);
    cfg.output_cb_depth_tiles =
        test_args::get_command_option_uint32(args, "--output-cb-depth-tiles", cfg.output_cb_depth_tiles);
    if (test_args::has_command_option(args, "--reader-batch-push")) {
        cfg.reader_mode = 1;
    }
    if (test_args::has_command_option(args, "--reader-dbuf-trid")) {
        // Per-trid double-buffer reader (pipelined design).
        cfg.reader_mode = 2;
    }
    cfg.page_size_tiles = test_args::get_command_option_uint32(args, "--page-size-tiles", cfg.page_size_tiles);
    cfg.reader_trid_in_flight =
        test_args::get_command_option_uint32(args, "--reader-trid-in-flight", cfg.reader_trid_in_flight);
    cfg.num_active_cores = test_args::get_command_option_uint32(args, "--num-active-cores", cfg.num_active_cores);
    if (test_args::has_command_option(args, "--core-layout-col")) {
        cfg.core_layout_col = true;
    }
    if (test_args::has_command_option(args, "--core-reverse")) {
        cfg.core_reverse = true;
    }
    cfg.reader_noc = test_args::get_command_option_uint32(args, "--reader-noc", cfg.reader_noc);
    cfg.writer_noc = test_args::get_command_option_uint32(args, "--writer-noc", cfg.writer_noc);
    if (test_args::has_command_option(args, "--swap-nocs")) {
        // Back-compat: legacy framework assignment (reader=NOC1, writer=NOC0).
        cfg.reader_noc = 1;
        cfg.writer_noc = 0;
    }
    cfg.num_nops_per_tile = test_args::get_command_option_uint32(args, "--compute-nops", cfg.num_nops_per_tile);
    cfg.num_programs = test_args::get_command_option_uint32(args, "--num-programs", cfg.num_programs);
    cfg.device_id = test_args::get_command_option_uint32(args, "--device-id", cfg.device_id);
    cfg.trace_region_size =
        test_args::get_command_option_uint32(args, "--trace-region-size", static_cast<uint32_t>(cfg.trace_region_size));
    cfg.trace_warmup_replays =
        test_args::get_command_option_uint32(args, "--trace-warmup-replays", cfg.trace_warmup_replays);
    if (test_args::has_command_option(args, "--no-warmup")) {
        cfg.warmup = false;
    }
    if (test_args::has_command_option(args, "--use-trace")) {
        cfg.use_trace = true;
    }
    if (test_args::has_command_option(args, "--use-device-profiler")) {
        cfg.use_device_profiler = true;
    }
    if (test_args::has_command_option(args, "--use-realtime-profiler")) {
        cfg.use_realtime_profiler = true;
    }
    if (test_args::has_command_option(args, "--read-only")) {
        cfg.read_only = true;
    }
    if (test_args::has_command_option(args, "--skip-output-validation")) {
        cfg.skip_output_validation = true;
    }
    if (test_args::has_command_option(args, "--lean-compute")) {
        cfg.lean_compute = true;
    }
    if (test_args::has_command_option(args, "--writer-flush-on-pressure")) {
        cfg.writer_flush_on_pressure = true;
    }
    if (test_args::has_command_option(args, "--cross-program-dram-offset")) {
        cfg.cross_program_dram_offset = true;
    }
    cfg.writer_end_barrier_mode =
        test_args::get_command_option_uint32(args, "--writer-end-barrier-mode", cfg.writer_end_barrier_mode);
    if (cfg.writer_end_barrier_mode > 2) {
        log_fatal(LogTest, "--writer-end-barrier-mode must be 0 (barrier), 1 (flushed), or 2 (none)");
        cfg.writer_end_barrier_mode = 0;
    }
    return cfg;
}

// Real-time profiler callbacks run on a worker thread; collect records here
// and analyse after UnregisterProgramRealtimeProfilerCallback.
void log_realtime_program_go_done(const std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord>& records) {
    std::map<uint32_t, std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord>> by_chip;
    for (const auto& r : records) {
        by_chip[r.chip_id].push_back(r);
    }
    for (auto& [chip_id, chip_records] : by_chip) {
        std::sort(chip_records.begin(), chip_records.end(), [](const auto& a, const auto& b) {
            return a.start_timestamp < b.start_timestamp;
        });
        for (const auto& r : chip_records) {
            const uint64_t duration_cycles =
                (r.end_timestamp >= r.start_timestamp) ? (r.end_timestamp - r.start_timestamp) : 0;
            const double duration_ns = (r.frequency > 0.0) ? static_cast<double>(duration_cycles) / r.frequency : 0.0;
            log_info(
                LogTest,
                "Real-time profiler chip {} program {}: go_cycles={} done_cycles={} duration={:.2f} ns",
                chip_id,
                r.runtime_id,
                r.start_timestamp,
                r.end_timestamp,
                duration_ns);
        }
    }
}

void log_realtime_op_to_op_gaps(const std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord>& records) {
    if (records.size() < 2) {
        log_info(LogTest, "Real-time profiler: got {} record(s); need >= 2 to compute op-to-op gaps.", records.size());
        return;
    }
    std::map<uint32_t, std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord>> by_chip;
    for (const auto& r : records) {
        by_chip[r.chip_id].push_back(r);
    }
    for (auto& [chip_id, chip_records] : by_chip) {
        std::sort(chip_records.begin(), chip_records.end(), [](const auto& a, const auto& b) {
            return a.start_timestamp < b.start_timestamp;
        });
        double sum_gap_ns = 0.0;
        uint32_t gap_count = 0;
        double min_gap_ns = 0.0;
        double max_gap_ns = 0.0;
        std::vector<double> gaps_ns;
        for (size_t i = 1; i < chip_records.size(); ++i) {
            const auto& prev = chip_records[i - 1];
            const auto& cur = chip_records[i];
            if (cur.runtime_id == prev.runtime_id) {
                log_warning(
                    LogTest,
                    "Real-time profiler chip {}: skipping dispatch gap between duplicate runtime_id {} "
                    "(trace replay may report the same host runtime id twice)",
                    chip_id,
                    cur.runtime_id);
                continue;
            }
            if (cur.start_timestamp < prev.end_timestamp) {
                log_warning(
                    LogTest,
                    "Real-time profiler chip {}: record {} start {} < prev end {} (skip gap)",
                    chip_id,
                    i,
                    cur.start_timestamp,
                    prev.end_timestamp);
                continue;
            }
            const uint64_t gap_cycles = cur.start_timestamp - prev.end_timestamp;
            const double freq = (cur.frequency > 0.0) ? cur.frequency : prev.frequency;
            if (freq <= 0.0) {
                continue;
            }
            const double gap_ns = static_cast<double>(gap_cycles) / freq;
            if (gap_count == 0) {
                min_gap_ns = max_gap_ns = gap_ns;
            } else {
                min_gap_ns = std::min(min_gap_ns, gap_ns);
                max_gap_ns = std::max(max_gap_ns, gap_ns);
            }
            sum_gap_ns += gap_ns;
            ++gap_count;
            gaps_ns.push_back(gap_ns);
        }
        if (gap_count > 0) {
            std::sort(gaps_ns.begin(), gaps_ns.end());
            const double median_ns = gaps_ns[gaps_ns.size() / 2];
            log_info(
                LogTest,
                "Real-time profiler chip {}: {} op-to-op gap(s) — min {:.2f} ns, median {:.2f} ns, max {:.2f} ns, "
                "mean {:.2f} ns (next program start − previous program end)",
                chip_id,
                gap_count,
                min_gap_ns,
                median_ns,
                max_gap_ns,
                sum_gap_ns / gap_count);
            std::string all = "Real-time profiler chip " + std::to_string(chip_id) + " sorted gaps (ns):";
            for (double g : gaps_ns) {
                all += fmt::format(" {:.0f}", g);
            }
            log_info(LogTest, "{}", all);
        }
    }
}

struct RealtimeProfilerSession {
    std::mutex mutex;
    std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord> records;
    std::optional<tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle> handle;

    void register_callback() {
        handle = tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback(
            [this](const tt::tt_metal::experimental::ProgramRealtimeRecord& record) {
                std::lock_guard<std::mutex> lock(mutex);
                records.push_back(record);
            });
    }

    void unregister_and_drain() {
        if (handle.has_value()) {
            tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback(*handle);
            handle.reset();
        }
    }

    std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord> copy_records() {
        std::lock_guard<std::mutex> lock(mutex);
        return records;
    }
};

void write_realtime_profiler_csv(
    const std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord>& records, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        log_warning(LogTest, "Real-time profiler: could not open {} for write", path);
        return;
    }
    out << "program_id,chip_id,go_cycles,done_cycles,gap_to_next_go_cycles,duration_cycles,duration_ns,frequency_ghz\n";
    std::map<uint32_t, std::vector<tt::tt_metal::experimental::ProgramRealtimeRecord>> by_chip;
    for (const auto& r : records) {
        by_chip[r.chip_id].push_back(r);
    }
    for (auto& [chip_id, chip_records] : by_chip) {
        std::sort(chip_records.begin(), chip_records.end(), [](const auto& a, const auto& b) {
            return a.start_timestamp < b.start_timestamp;
        });
        for (size_t i = 0; i < chip_records.size(); ++i) {
            const auto& r = chip_records[i];
            const uint64_t duration_cycles =
                (r.end_timestamp >= r.start_timestamp) ? (r.end_timestamp - r.start_timestamp) : 0;
            const double duration_ns = (r.frequency > 0.0) ? static_cast<double>(duration_cycles) / r.frequency : 0.0;
            uint64_t gap_to_next_go = 0;
            if (i + 1 < chip_records.size()) {
                const auto& next = chip_records[i + 1];
                if (next.runtime_id != r.runtime_id && next.start_timestamp >= r.end_timestamp) {
                    gap_to_next_go = next.start_timestamp - r.end_timestamp;
                }
            }
            out << r.runtime_id << "," << chip_id << "," << r.start_timestamp << "," << r.end_timestamp << ","
                << gap_to_next_go << "," << duration_cycles << "," << duration_ns << "," << r.frequency << "\n";
        }
    }
    log_info(LogTest, "Real-time profiler: wrote {} record(s) to {}", records.size(), path);
}

// Registers RT profiler callback after warmup; on destruction quiesces, waits for
// D2H records, unregisters, then logs per-chip op-to-op gaps (the RT metric:
// next start - previous end). Must outlive the timed enqueue burst only.
struct RealtimeProfilerDrainGuard {
    distributed::MeshDevice* mesh = nullptr;
    std::unique_ptr<RealtimeProfilerSession> session;

    RealtimeProfilerDrainGuard(distributed::MeshDevice* mesh_in, bool use_realtime_profiler, bool profiler_active) :
        mesh(mesh_in) {
        if (!use_realtime_profiler) {
            return;
        }
        if (!profiler_active) {
            log_warning(
                LogTest,
                "Real-time profiler: --use-realtime-profiler set but IsProgramRealtimeProfilerActive() is false; "
                "skipping (ETH dispatch / remote chip / resources — see "
                "tech_reports/real_time_profiler/getting-started.md).");
            return;
        }
        session = std::make_unique<RealtimeProfilerSession>();
        session->register_callback();
        log_info(LogTest, "Real-time profiler: callback registered for timed section.");
    }

    RealtimeProfilerDrainGuard(const RealtimeProfilerDrainGuard&) = delete;
    RealtimeProfilerDrainGuard& operator=(const RealtimeProfilerDrainGuard&) = delete;

    ~RealtimeProfilerDrainGuard() {
        if (!session) {
            return;
        }
        mesh->quiesce_devices();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        session->unregister_and_drain();
        const auto records = session->copy_records();
        log_realtime_program_go_done(records);
        log_realtime_op_to_op_gaps(records);
        const char* metal_home = std::getenv("TT_METAL_HOME");
        if (metal_home != nullptr) {
            write_realtime_profiler_csv(
                records, std::string(metal_home) + "/generated/profiler/.logs/profile_log_device_rt.csv");
        }
    }
};

struct BuiltProgram {
    Program program;
    KernelHandle reader_kernel = 0;
    KernelHandle writer_kernel = 0;
    KernelHandle compute_kernel = 0;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t tiles_per_core_group_1 = 0;
    uint32_t tiles_per_core_group_2 = 0;
    uint32_t reader_push_tile_count = 2;
    uint32_t reader_mode = 0;
    uint32_t reader_trid_in_flight = 2;
    uint32_t input_cb_depth_tiles = 0;
    uint32_t output_cb_depth_tiles = 2;
    uint32_t page_size_tiles = 1;
};

// Set reader/writer/compute runtime args (only the truly per-launch ones; kernel knobs
// like reader_mode / push_tile_count / tiles_per_page / trid_in_flight are compile-time
// args on the kernel so they fold into constants and dead code gets eliminated).
void set_program_launch_args(
    Program& program,
    const BuiltProgram& built,
    uint32_t input_buffer_addr,
    uint32_t output_buffer_addr,
    uint32_t program_id) {
    const auto work_groups = {
        std::make_pair(built.core_group_1, built.tiles_per_core_group_1),
        std::make_pair(built.core_group_2, built.tiles_per_core_group_2)};
    uint32_t start_tile_id = 0;
    for (const auto& [group, tiles_per_core] : work_groups) {
        if (tiles_per_core == 0) {
            continue;
        }
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                SetRuntimeArgs(
                    program, built.reader_kernel, core, {input_buffer_addr, tiles_per_core, start_tile_id, program_id});
                SetRuntimeArgs(
                    program,
                    built.writer_kernel,
                    core,
                    {output_buffer_addr, tiles_per_core, start_tile_id, program_id});
                SetRuntimeArgs(program, built.compute_kernel, core, {tiles_per_core, program_id});
                start_tile_id += tiles_per_core;
            }
        }
    }
}

// Configure host runtime id (RT profiler) and device CSV program_id before each launch.
void configure_program_launch(
    Program& program,
    const BuiltProgram& built,
    uint32_t input_buffer_addr,
    uint32_t output_buffer_addr,
    uint32_t profiler_program_id,
    uint32_t host_runtime_id) {
    set_program_launch_args(program, built, input_buffer_addr, output_buffer_addr, profiler_program_id);
    program.set_runtime_id(host_runtime_id);
}

// Build the reader -> compute -> writer program covering every Tensix core,
// with the per-core tile slice assigned via runtime args.
BuiltProgram build_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const BenchmarkConfig& cfg,
    const std::shared_ptr<distributed::MeshBuffer>& input_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& output_buffer,
    uint32_t total_num_tiles,
    uint32_t single_tile_size_bytes) {
    Program program = CreateProgram();

    const auto full_grid = mesh_device->compute_with_storage_grid_size();
    uint32_t want = full_grid.x * full_grid.y;
    if (cfg.num_active_cores > 0 && cfg.num_active_cores < want) {
        want = cfg.num_active_cores;
    }
    // Build the full grid order (row-major, or column-major with --core-layout-col), then
    // take the first `want`. --core-reverse takes the LAST `want` instead (reversed fill
    // order) so we can use the mirror-image core set and rule out which specific cores /
    // direction drive the curve.
    std::vector<CoreCoord> full_order;
    if (cfg.core_layout_col) {
        for (uint32_t x = 0; x < full_grid.x; ++x) {
            for (uint32_t y = 0; y < full_grid.y; ++y) {
                full_order.push_back(CoreCoord{x, y});
            }
        }
    } else {
        for (uint32_t y = 0; y < full_grid.y; ++y) {
            for (uint32_t x = 0; x < full_grid.x; ++x) {
                full_order.push_back(CoreCoord{x, y});
            }
        }
    }
    if (cfg.core_reverse) {
        std::reverse(full_order.begin(), full_order.end());
    }
    std::vector<CoreCoord> core_list(
        full_order.begin(), full_order.begin() + std::min<size_t>(want, full_order.size()));
    const uint32_t num_cores = static_cast<uint32_t>(core_list.size());
    std::vector<CoreRange> ranges;
    ranges.reserve(num_cores);
    for (const auto& c : core_list) {
        ranges.emplace_back(c, c);
    }
    CoreRangeSet all_cores(ranges);
    CoreRangeSet core_group_1 = all_cores;
    CoreRangeSet core_group_2;
    const uint32_t num_tiles_per_core_group_1 = num_cores > 0 ? total_num_tiles / num_cores : 0;
    const uint32_t num_tiles_per_core_group_2 = 0;

    log_info(
        LogTest,
        "Active cores: {} ({}-major{}), full_grid {}x{}, total_num_tiles={} ({} tiles/core); first={} last={}",
        num_cores,
        cfg.core_layout_col ? "column" : "row",
        cfg.core_reverse ? " reversed" : "",
        full_grid.x,
        full_grid.y,
        total_num_tiles,
        num_tiles_per_core_group_1,
        num_cores ? core_list.front().str() : "none",
        num_cores ? core_list.back().str() : "none");

    // Input CB depth defaults to 2x reader push chunk (e.g. push 2 -> depth 4).
    constexpr uint32_t kInputCbId = tt::CBIndex::c_0;
    constexpr uint32_t kOutputCbId = tt::CBIndex::c_16;
    constexpr auto kDataFormat = tt::DataFormat::Float16_b;

    const uint32_t push_tiles = cfg.reader_push_tile_count > 0 ? cfg.reader_push_tile_count : 1;
    const uint32_t input_cb_depth = cfg.input_cb_depth_tiles > 0 ? cfg.input_cb_depth_tiles : (2 * push_tiles);
    // Compute -> writer unchanged: still wait_front(1) / pack one tile at a time.
    const uint32_t output_cb_depth = cfg.output_cb_depth_tiles > 0 ? cfg.output_cb_depth_tiles : 2;

    TT_FATAL(
        input_cb_depth >= push_tiles,
        "input CB depth ({}) must be >= --reader-push-tiles ({})",
        input_cb_depth,
        push_tiles);
    TT_FATAL(
        cfg.num_pages_per_core >= push_tiles,
        "--num-pages-per-core ({}) must be >= --reader-push-tiles ({})",
        cfg.num_pages_per_core,
        push_tiles);

    const uint32_t page_size_tiles = cfg.page_size_tiles > 0 ? cfg.page_size_tiles : 1;
    TT_FATAL(
        cfg.num_pages_per_core % page_size_tiles == 0,
        "--num-pages-per-core ({}) must be a multiple of --page-size-tiles ({})",
        cfg.num_pages_per_core,
        page_size_tiles);
    TT_FATAL(
        input_cb_depth >= page_size_tiles,
        "input CB depth ({}) must be >= --page-size-tiles ({})",
        input_cb_depth,
        page_size_tiles);
    TT_FATAL(
        output_cb_depth >= page_size_tiles,
        "output CB depth ({}) must be >= --page-size-tiles ({}) so the writer can accumulate a full page",
        output_cb_depth,
        page_size_tiles);
    const uint32_t trid_in_flight = cfg.reader_trid_in_flight > 0 ? cfg.reader_trid_in_flight : 2;
    if (cfg.reader_mode == 2) {
        TT_FATAL(
            trid_in_flight >= 2,
            "--reader-trid-in-flight ({}) must be >= 2 for mode 2 (per-trid double-buffer needs both sides active)",
            trid_in_flight);
        TT_FATAL(
            input_cb_depth >= 2 * trid_in_flight * page_size_tiles,
            "per-trid DB needs input CB depth ({}) >= 2 * --reader-trid-in-flight ({}) * --page-size-tiles ({})",
            input_cb_depth,
            trid_in_flight,
            page_size_tiles);
        TT_FATAL(
            cfg.num_pages_per_core / page_size_tiles >= 1,
            "--num-pages-per-core ({}) / --page-size-tiles ({}) must be >= 1 for reader_mode 2",
            cfg.num_pages_per_core,
            page_size_tiles);
    } else if (page_size_tiles > 1) {
        TT_FATAL(
            false,
            "--page-size-tiles > 1 currently requires --reader-dbuf-trid (reader_mode 2); "
            "legacy modes 0/1 use tile_id as page_id and would read wrong data.");
    }

    log_info(
        LogTest,
        "Reader: push_tiles={}, input_cb_depth={}, output_cb_depth={}, page_size_tiles={}, trid_in_flight={} "
        "(reader_mode={}: 0=incremental, 1=batch, 2=per-trid DB with N reads in flight per TRID)",
        push_tiles,
        input_cb_depth,
        output_cb_depth,
        page_size_tiles,
        trid_in_flight,
        cfg.reader_mode);

    CircularBufferConfig cb_in_config =
        CircularBufferConfig(input_cb_depth * single_tile_size_bytes, {{kInputCbId, kDataFormat}})
            .set_page_size(kInputCbId, single_tile_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_in_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(output_cb_depth * single_tile_size_bytes, {{kOutputCbId, kDataFormat}})
            .set_page_size(kOutputCbId, single_tile_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_out_config);

    // Reader: NOC1 / RISCV1, pulls from interleaved DRAM via TensorAccessor.
    // Compile-time args order MUST match reader_interleaved.cpp:
    //   [0]=cb_in, [1]=READER_MODE, [2]=PUSH_TILE_COUNT, [3]=TILES_PER_PAGE,
    //   [4]=TRID_IN_FLIGHT, [5]=CROSS_PROGRAM_OFFSET_TILES, then TensorAccessorArgs starting at index 6.
    const uint32_t cross_program_offset_tiles = cfg.cross_program_dram_offset ? total_num_tiles : 0u;
    std::vector<uint32_t> reader_compile_time_args = {
        kInputCbId, cfg.reader_mode, push_tiles, page_size_tiles, trid_in_flight, cross_program_offset_tiles};
    // Defaults: reader=NOC0, writer=NOC1 (measured best). --reader-noc/--writer-noc override.
    // HARD CONSTRAINT: the two data-movement kernels on a core must be on DIFFERENT NoCs.
    // Putting both on the same NoC hangs the device (requires a chip reset), so fail fast.
    TT_FATAL(
        cfg.reader_noc <= 1 && cfg.writer_noc <= 1,
        "--reader-noc/--writer-noc must be 0 or 1 (got reader={}, writer={})",
        cfg.reader_noc,
        cfg.writer_noc);
    TT_FATAL(
        cfg.reader_noc != cfg.writer_noc,
        "reader and writer must use DIFFERENT NoCs (both NOC{} requested) -- same NoC hangs the device",
        cfg.reader_noc);
    const NOC reader_noc = (cfg.reader_noc == 1) ? NOC::NOC_1 : NOC::NOC_0;
    const NOC writer_noc = (cfg.writer_noc == 1) ? NOC::NOC_1 : NOC::NOC_0;
    log_info(LogTest, "NoC assignment: reader=NOC{}, writer=NOC{}", cfg.reader_noc, cfg.writer_noc);
    TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/kernels/reader_interleaved.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = reader_noc, .compile_args = reader_compile_time_args});

    // Writer: NOC0 / RISCV0, pushes to interleaved DRAM.
    // Compile-time args order MUST match writer_interleaved.cpp:
    //   [0]=cb_out, [1]=TILES_PER_PAGE, [2]=READ_ONLY, [3]=WRITER_FLUSH_MODE,
    //   [4]=OUTPUT_CB_DEPTH_TILES, [5]=CROSS_PROGRAM_OFFSET_TILES,
    //   [6]=END_BARRIER_MODE, then TensorAccessorArgs starting at index 7.
    std::vector<uint32_t> writer_compile_time_args = {
        kOutputCbId,
        page_size_tiles,
        cfg.read_only ? 1u : 0u,
        cfg.writer_flush_on_pressure ? 1u : 0u,
        output_cb_depth,
        cross_program_offset_tiles,
        cfg.writer_end_barrier_mode};
    TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);
    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/kernels/writer_interleaved.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = writer_noc, .compile_args = writer_compile_time_args});

    // Compute: copy_tile + tunable NOP spin.
    std::vector<uint32_t> compute_compile_time_args = {
        kInputCbId, kOutputCbId, cfg.num_nops_per_tile, cfg.lean_compute ? 0u : 1u};
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
                SetRuntimeArgs(
                    program, reader_kernel, core, {input_buffer->address(), tiles_per_core, start_tile_id, 0});
                SetRuntimeArgs(
                    program, writer_kernel, core, {output_buffer->address(), tiles_per_core, start_tile_id, 0});
                SetRuntimeArgs(program, compute_kernel, core, {tiles_per_core, 0});
                start_tile_id += tiles_per_core;
            }
        }
    }
    TT_FATAL(
        start_tile_id == total_num_tiles,
        "Tile distribution mismatch: assigned {} but expected {}",
        start_tile_id,
        total_num_tiles);

    BuiltProgram built;
    built.program = std::move(program);
    built.reader_kernel = reader_kernel;
    built.writer_kernel = writer_kernel;
    built.compute_kernel = compute_kernel;
    built.core_group_1 = core_group_1;
    built.core_group_2 = core_group_2;
    built.tiles_per_core_group_1 = num_tiles_per_core_group_1;
    built.tiles_per_core_group_2 = num_tiles_per_core_group_2;
    built.reader_push_tile_count = push_tiles;
    built.reader_mode = cfg.reader_mode;
    built.reader_trid_in_flight = trid_in_flight;
    built.input_cb_depth_tiles = input_cb_depth;
    built.output_cb_depth_tiles = output_cb_depth;
    built.page_size_tiles = page_size_tiles;
    return built;
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const std::vector<std::string> args(argv, argv + argc);
        const BenchmarkConfig cfg = parse_args(args);

        log_info(
            LogTest,
            "op_to_op_latency: device_id={}, pages_per_core={}, reader_push_tiles={}, input_cb_depth_tiles={}, "
            "output_cb_depth_tiles={}, reader_mode={}, compute_nops={}, num_programs={}, warmup={}, use_trace={}, "
            "use_device_profiler={}, use_realtime_profiler={}, trace_region_size={}, trace_warmup_replays={}, "
            "read_only={}",
            cfg.device_id,
            cfg.num_pages_per_core,
            cfg.reader_push_tile_count,
            cfg.input_cb_depth_tiles,
            cfg.output_cb_depth_tiles,
            cfg.reader_mode,
            cfg.num_nops_per_tile,
            cfg.num_programs,
            cfg.warmup,
            cfg.use_trace,
            cfg.use_device_profiler,
            cfg.use_realtime_profiler,
            cfg.trace_region_size,
            cfg.trace_warmup_replays,
            cfg.read_only);
        TT_FATAL(cfg.num_programs >= 1, "--num-programs must be >= 1");
        TT_FATAL(cfg.num_pages_per_core >= 1, "--num-pages-per-core must be >= 1");
        TT_FATAL(cfg.reader_push_tile_count >= 1, "--reader-push-tiles must be >= 1");

        // If the user asked for a CSV dump, make sure the runtime profiler is
        // actually enabled (Tracy-enabled build + TT_METAL_DEVICE_PROFILER=1
        // env var before process start). Without this check, the CSV would just be empty and the
        // failure mode would be silent. Pattern mirrors test_dram_offchip.cpp.
        if (cfg.use_device_profiler) {
            const bool profiler_runtime_enabled =
                tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_enabled();
            TT_FATAL(
                profiler_runtime_enabled,
                "--use-device-profiler requires the device profiler to be enabled at runtime. "
                "Use a Tracy-enabled build (default: omit `build_metal.sh --disable-profiler`) and "
                "export TT_METAL_DEVICE_PROFILER=1 before starting the process "
                "(e.g. `TT_METAL_DEVICE_PROFILER=1 ./test_op_to_op_latency --use-device-profiler ...`). "
                "If kernels were JIT-cached without profiler, clear ~/.cache/tt-metal-cache and re-run.");
        }

        // trace_region_size = 0 would disable trace capture entirely (legacy
        // path) -- in trace mode we always want a non-zero value. The default
        // is 1 MiB which covers tens of small programs comfortably.
        const size_t trace_region_size = cfg.use_trace ? cfg.trace_region_size : 0;
        auto mesh_device =
            distributed::MeshDevice::create_unit_mesh(cfg.device_id, DEFAULT_L1_SMALL_SIZE, trace_region_size);

        log_info(LogTest, "Clock: {} MHz", get_tt_npu_clock(mesh_device->get_devices()[0]));

        const auto grid = mesh_device->compute_with_storage_grid_size();
        uint32_t effective_cores = grid.x * grid.y;
        if (cfg.num_active_cores > 0 && cfg.num_active_cores < effective_cores) {
            effective_cores = cfg.num_active_cores;
        }
        const uint32_t num_cores = effective_cores;
        const uint32_t total_num_tiles = num_cores * cfg.num_pages_per_core;

        constexpr auto kDataFormat = tt::DataFormat::Float16_b;
        const uint32_t single_tile_size_bytes = tile_size(kDataFormat);
        // When cross-program DRAM offset is on, allocate enough room for warmup
        // (pid=0) + num_programs disjoint slices so each timed program touches
        // fresh DRAM pages.
        const uint32_t buffer_num_tiles =
            cfg.cross_program_dram_offset ? (cfg.num_programs + 1u) * total_num_tiles : total_num_tiles;
        const uint32_t buffer_size_bytes = buffer_num_tiles * single_tile_size_bytes;
        if (cfg.cross_program_dram_offset) {
            log_info(
                LogTest,
                "cross_program_dram_offset: enabled; buffer={} tiles ({} per-program slice x {} = {} MB)",
                buffer_num_tiles,
                total_num_tiles,
                cfg.num_programs + 1u,
                buffer_size_bytes / (1024 * 1024));
        }

        // Interleaved DRAM buffer: page_size = page_size_tiles * tile_bytes
        // (default 1 tile per page), so each NoC transaction transfers one
        // page from one DRAM bank; pages are spread across banks round-robin.
        const uint32_t dram_page_size_bytes =
            (cfg.page_size_tiles > 0 ? cfg.page_size_tiles : 1) * single_tile_size_bytes;
        distributed::DeviceLocalBufferConfig dram_local_config{
            .page_size = dram_page_size_bytes,
            .buffer_type = BufferType::DRAM,
        };
        distributed::ReplicatedBufferConfig dram_buf_config{.size = buffer_size_bytes};

        auto input_buffer = distributed::MeshBuffer::create(dram_buf_config, dram_local_config, mesh_device.get());
        auto output_buffer = distributed::MeshBuffer::create(dram_buf_config, dram_local_config, mesh_device.get());

        // Random bf16 input data, packed two values per uint32_t.
        const auto seed = static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<uint32_t> input_data =
            create_random_vector_of_bfloat16(buffer_size_bytes, /*rand_max_float=*/100, seed);

        auto& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data, /*blocking=*/false);
        // Drain the input upload before trace capture: host writes are not allowed while
        // a trace is being recorded (see FDMeshCommandQueue "Writes are not supported
        // during trace capture"). Warmup's Finish used to hide this; --no-warmup needs
        // an explicit Finish here.
        distributed::Finish(cq);

        BuiltProgram built =
            build_program(mesh_device, cfg, input_buffer, output_buffer, total_num_tiles, single_tile_size_bytes);

        distributed::MeshWorkload workload;
        const distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(built.program));
        Program& program = workload.get_programs().begin()->second;

        // Device CSV uses PROG_ID=0 for warmup/pre-compile; export skips id < 1.
        // Host runtime_id=0 is not reported by the real-time profiler (dropped by Metal).
        constexpr uint32_t kPrecompileProfilerProgramId = 0;

        const uint32_t input_buffer_addr = input_buffer->address();
        const uint32_t output_buffer_addr = output_buffer->address();
        auto launch = [&](uint32_t profiler_program_id, uint32_t host_runtime_id) {
            configure_program_launch(
                program, built, input_buffer_addr, output_buffer_addr, profiler_program_id, host_runtime_id);
        };

        // Warmup (untimed): one eager enqueue absorbs the JIT-load /
        // dispatcher prefetch cost so they do not pollute either the FD timing
        // loop or the trace-capture step.
        if (cfg.warmup) {
            launch(kPrecompileProfilerProgramId, kPrecompileProfilerProgramId);
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
            distributed::Finish(cq);
        }

        const bool rt_profiler_active = tt::tt_metal::experimental::IsProgramRealtimeProfilerActive();

        constexpr uint8_t kCqId = 0;
        long long elapsed_us = 0;
        const char* mode_label = nullptr;

        if (cfg.use_trace) {
            // First enqueue inside trace capture triggers JIT / host-side setup, which
            // is not allowed while recording ("Writes are not supported during trace
            // capture"). Warmup already pre-compiles; with --no-warmup do it here.
            if (!cfg.warmup) {
                log_info(LogTest, "Trace mode: pre-compiling kernels before capture (PROG_ID=0).");
                launch(kPrecompileProfilerProgramId, 1);
                distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
                distributed::Finish(cq);
            }

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
                const uint32_t program_index = i + 1;
                launch(program_index, program_index);
                distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
            }
            mesh_device->end_mesh_trace(kCqId, tid);
            distributed::Finish(cq);  // make sure capture is fully committed before timing

            // Untimed replays warm up the trace path; we measure the steady-state replay.
            for (uint32_t replay = 0; replay < cfg.trace_warmup_replays; ++replay) {
                mesh_device->replay_mesh_trace(kCqId, tid, /*blocking=*/false);
                distributed::Finish(cq);
                if (cfg.use_device_profiler) {
                    // Flush device profiler so warmup markers do not pollute the timed replay.
                    tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh_device);
                }
            }
            if (cfg.trace_warmup_replays > 0) {
                log_info(
                    LogTest, "Trace: {} untimed warmup replay(s) before measured replay", cfg.trace_warmup_replays);
            }

            {
                RealtimeProfilerDrainGuard rt_guard(mesh_device.get(), cfg.use_realtime_profiler, rt_profiler_active);
                const auto t_begin = std::chrono::steady_clock::now();
                mesh_device->replay_mesh_trace(kCqId, tid, /*blocking=*/false);
                distributed::Finish(cq);
                const auto t_end = std::chrono::steady_clock::now();
                elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
            }

            mesh_device->release_mesh_trace(tid);
            mode_label = cfg.trace_warmup_replays > 0 ? "Trace replay (after warmup)" : "Trace replay";
        } else {
            {
                RealtimeProfilerDrainGuard rt_guard(mesh_device.get(), cfg.use_realtime_profiler, rt_profiler_active);
                // Step 2: enqueue the same MeshWorkload back-to-back num_programs
                // times under one Finish.
                const auto t_begin = std::chrono::steady_clock::now();
                for (uint32_t i = 0; i < cfg.num_programs; ++i) {
                    const uint32_t program_index = i + 1;
                    launch(program_index, program_index);
                    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
                }
                distributed::Finish(cq);
                const auto t_end = std::chrono::steady_clock::now();
                elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
            }
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
        // generated/profiler/.logs/profile_log_device.csv. For in-kernel timing,
        // filter per-tile `MATH` rows (MATH TRISC) or firmware `TRISC-KERNEL`
        // rows. Program-level op-to-op between host enqueues is from
        // `--use-realtime-profiler`, not from nesting DeviceZoneScopedMainN in
        // user TRISC kernels. tools/tracy/process_device_log.py can post-process
        // the CSV.
        if (cfg.use_device_profiler) {
            tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh_device);
            log_info(
                LogTest,
                "Device profiler results dumped. Inspect "
                "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv "
                "(PROG_ID, read/write barriers, TRISC_0 TILE_IDX, TRISC_2 FINISH_LAST_PUSH; "
                "dispatch done/go: --use-realtime-profiler -> profile_log_device_rt.csv).");
        }

        if (!cfg.read_only && !cfg.skip_output_validation) {
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
        }  // !read_only validation

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
