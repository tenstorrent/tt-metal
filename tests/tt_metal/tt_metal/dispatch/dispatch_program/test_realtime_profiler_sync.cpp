// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sync-accuracy coverage for the real-time (RT) profiler: the published sync_error distribution, and the mapping's
// residual against an independent read of the device clock. Their limits are nanosecond-scale and
// hardware-timing-sensitive (PCIe read brackets, chip thermal state), so this file is deliberately NOT part of the
// merge gate -- it runs from UNIT_TESTS_DISPATCH_SLOW_SOURCES alongside the stress tests. Merge-gate coverage of the
// same pipeline (well-formedness, delivery, sources) lives in test_realtime_profiler_sanity.cpp; the didt suite
// (tests/didt/test_rt_profiler_sync_error.py) asserts the same distribution under compute load.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <tt-logger/tt-logger.hpp>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "impl/realtime_profiler/device_clock_sync.hpp"

namespace tt::tt_metal {
namespace {

using namespace std::chrono_literals;
using tt::tt_metal::experimental::IsProgramRealtimeProfilerActive;
using tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle;
using tt::tt_metal::experimental::ProgramRealtimeRecord;
using tt::tt_metal::experimental::ProgramRealtimeRecordBatch;
using tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback;
using tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback;

// The helpers below are Sync-prefixed copies of test_realtime_profiler_sanity.cpp's, because the two files share a
// Unity build TU.

// Inlined (200x200 unrolled NOPs) rather than loaded from a file under tt_metal/programming_examples/...: those files
// ship in the `metalium-examples` deb, while this test runs from `tt-metalium-validation` in CI.
std::string make_sync_kernel_source(uint32_t runtime_id) {
    return "#include <cstdint>\n"
           "// rt_profiler_sync_marker_" +
           std::to_string(runtime_id) +
           "\n"
           "void kernel_main() {\n"
           "    for (int i = 0; i < 200; i++) {\n"
           "#pragma GCC unroll 65534\n"
           "        for (int j = 0; j < 200; j++) {\n"
           "            asm(\"nop\");\n"
           "        }\n"
           "    }\n"
           "}\n";
}

// runtime_id == 0 is reserved for infrastructure traffic and filtered host-side.
Program make_sync_program(const std::string& kernel_src, const CoreRange& cores, uint32_t runtime_id) {
    Program program = CreateProgram();
    CreateKernelFromString(
        program,
        kernel_src,
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernelFromString(
        program,
        kernel_src,
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernelFromString(program, kernel_src, cores, ComputeConfig{});
    program.set_runtime_id(static_cast<uint64_t>(runtime_id));
    return program;
}

void enqueue_sync_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t runtime_id, const CoreRange& cores) {
    distributed::MeshWorkload workload;
    workload.add_program(
        distributed::MeshCoordinateRange(mesh_device->shape()),
        make_sync_program(make_sync_kernel_source(runtime_id), cores, runtime_id));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
}

// Accumulates everything delivered to one registered callback. Unregistering in the destructor is what makes the
// accumulated records safe to read: the API guarantees no callback is in flight once it returns.
class SyncRecordCollector {
public:
    SyncRecordCollector() {
        handle_ = RegisterProgramRealtimeProfilerCallback([this](const ProgramRealtimeRecordBatch& batch) {
            std::lock_guard lock(mutex_);
            dropped_ += batch.dropped;
            records_.insert(records_.end(), batch.records.begin(), batch.records.end());
        });
        registered_ = true;
    }
    ~SyncRecordCollector() { stop(); }
    SyncRecordCollector(const SyncRecordCollector&) = delete;
    SyncRecordCollector& operator=(const SyncRecordCollector&) = delete;

    void stop() {
        if (std::exchange(registered_, false)) {
            UnregisterProgramRealtimeProfilerCallback(handle_);
        }
    }

    std::vector<ProgramRealtimeRecord> records() const {
        std::lock_guard lock(mutex_);
        return records_;
    }
    std::set<uint32_t> runtime_ids_in_range(uint32_t count) const {
        std::set<uint32_t> ids;
        for (const auto& record : records()) {
            if (record.runtime_id >= 1 && record.runtime_id <= count) {
                ids.insert(record.runtime_id);
            }
        }
        return ids;
    }

private:
    mutable std::mutex mutex_;
    std::vector<ProgramRealtimeRecord> records_;
    uint64_t dropped_ = 0;
    ProgramRealtimeProfilerCallbackHandle handle_{};
    bool registered_ = false;
};

// Opens a unit mesh with the RT profiler active, or skips.
class RealtimeProfilerSyncTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_device_ = distributed::MeshDevice::create_unit_mesh(
            /*device_id=*/0,
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            /*num_command_queues=*/1,
            DispatchCoreConfig{DispatchCoreType::WORKER});
        ASSERT_NE(mesh_device_, nullptr);
        if (!IsProgramRealtimeProfilerActive()) {
            mesh_device_->close();
            mesh_device_.reset();
            GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
        }
    }

    void TearDown() override {
        if (mesh_device_ != nullptr) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    CoreRange all_cores() const {
        const CoreCoord grid = mesh_device_->compute_with_storage_grid_size();
        return CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1});
    }

    template <typename Predicate>
    void quiesce_and_wait_for(Predicate delivered) {
        mesh_device_->quiesce_devices();
        const auto deadline = std::chrono::steady_clock::now() + 10s;
        while (!delivered() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(5ms);
        }
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
};

// Several times the worst measured claim on either architecture (p99 under 2us idle or under didt load), so a
// systematic regression trips it while thermal variance does not. The mean bound is what catches a systematic
// inflation that stays under the per-record ceiling; measured means sit around 0.6us.
constexpr auto kMaxSyncError = std::chrono::microseconds(15);
constexpr auto kMaxMeanSyncError = std::chrono::microseconds(5);

// The didt suite owns the sync_error distribution under load; this only catches a claim broken outright:
// unpopulated, or inflated past the scale every measurement of it sits at.
TEST_F(RealtimeProfilerSyncTest, SyncAccuracy) {
    SyncRecordCollector collector;
    // A fixed runtime_id keeps the kernel source (and its JIT compile) shared across iterations.
    constexpr uint32_t kIterations = 300;
    for (uint32_t i = 0; i < kIterations; ++i) {
        enqueue_sync_program(mesh_device_, /*runtime_id=*/1, all_cores());
        std::this_thread::sleep_for(DeviceClockSync::sync_interval());
    }
    quiesce_and_wait_for([&] { return collector.records().size() >= kIterations; });
    collector.stop();

    const auto records = collector.records();
    ASSERT_GE(records.size(), kIterations);
    std::chrono::nanoseconds worst{};
    std::chrono::nanoseconds sum{};
    for (const auto& record : records) {
        ASSERT_GT(record.clock_sync.sync_error, std::chrono::nanoseconds::zero())
            << "sync_error should be populated once the clock is anchored";
        ASSERT_LT(record.clock_sync.sync_error, kMaxSyncError)
            << "a claim this large means the probe path is degraded (wide brackets accepted, or bad placement)";
        worst = std::max(worst, record.clock_sync.sync_error);
        sum += record.clock_sync.sync_error;
    }
    const auto mean = sum / records.size();
    log_info(
        tt::LogTest,
        "[RT profiler sync] claimed sync_error over {} records: mean={}ns worst={}ns",
        records.size(),
        mean.count(),
        worst.count());
    EXPECT_LT(mean, kMaxMeanSyncError)
        << "the typical claim is inflated even though no single record tripped the per-record ceiling";
}

// Independent check of the clock mapping the RT profiler publishes on every record. Shares no code or mechanism with
// production sync: production has the host drive a round trip and bracket a device timestamp inside it, while this
// reads the device's own clock and brackets that read between two host clock reads.
//
// Nothing is fitted here -- a record's device_cycle_offset/frequency already states the mapping, so only its
// residual is measured. A single read can't resolve it (the host bracket is wider than the error sought); averaging
// many does, to about one bracket over sqrt(sample count).
//
// The kernel never waits on the host: it stamps WALL_CLOCK into one L1 slot in a bounded loop and exits, so the host
// can stop reading (or fail an assertion) at any time without stranding a spinning kernel and hanging Finish.
constexpr uint32_t kClockStampIterations = 200'000'000;  // ~5s of stamping; only reached if the stop flag never lands
constexpr uint32_t kClockCheckRounds = 3;
constexpr uint32_t kClockReadsPerRound = 500;
constexpr uint32_t kFirstClockCheckRuntimeId = 9101;
// Fixed rather than derived from the record's own sync_error -- a bound computed from the claim cannot test the
// claim. Measured disagreement is ~210ns; this leaves an order of magnitude of headroom.
constexpr double kMaxMeanMappingErrorNs = 2'000.0;
// How far a read may put the mapping out *beyond what that read itself could be wrong by*. Measured zero on
// Blackhole; sized well above that so ordinary jitter can't trip it while an uncorrected DVFS step still would.
constexpr double kMaxMappingErrorBeyondReadResolutionNs = 500.0;

// Runs on every core the stamper is not using, so the reads below are taken while the part is drawing power: on
// Blackhole, DVFS only steps AICLK under load, and that step is what this test needs to see the mapping survive.
std::string make_clock_load_kernel_source() {
    return R"(
#include <cstdint>
#include "risc_common.h"

void kernel_main() {
    const uint32_t stop_addr = get_arg_val<uint32_t>(0);
    const uint32_t max_iterations = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr const uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);

    for (uint32_t i = 0; i < max_iterations; i++) {
#pragma GCC unroll 128
        for (int j = 0; j < 128; j++) {
            asm("nop");
        }
        if ((i & 0xFF) == 0 && stop[0] != 0) {
            break;
        }
    }
}
)";
}

std::string make_clock_stamp_kernel_source() {
    return R"(
#include <cstdint>
#include "risc_common.h"

void kernel_main() {
    const uint32_t stamp_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t max_iterations = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* stamp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stamp_addr);
    volatile tt_l1_ptr const uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);

    for (uint32_t i = 0; i < max_iterations; i++) {
        const uint64_t now = get_timestamp();
        stamp[1] = static_cast<uint32_t>(now >> 32);
        stamp[0] = static_cast<uint32_t>(now & 0xFFFFFFFF);
        if ((i & 0xFF) == 0 && stop[0] != 0) {
            break;
        }
    }
}
)";
}

// Blackhole exposes a static TLB window onto the core's L1, so the stamp can be read with one load instead of going
// through UMD's generic path -- only software overhead comes off, since an MMIO read is a non-posted PCIe
// transaction either way. Null where no such window exists; the caller falls back to the generic read.
volatile uint64_t* try_map_l1_word(Cluster& cluster, ChipId chip_id, CoreCoord virtual_core, uint32_t addr) {
    try {
        const tt_xy_pair tlb_core(virtual_core.x, virtual_core.y);
        auto* tlb_manager = cluster.get_driver()->get_chip(chip_id)->get_tlb_manager();
        if (tlb_manager == nullptr || !tlb_manager->is_tlb_mapped(tlb_core, addr, sizeof(uint64_t))) {
            return nullptr;
        }
        // A lookup, never an allocation: get_tlb_window throws unless UMD already mapped this core at init, and
        // is_tlb_mapped has just confirmed the mapping covers these bytes.
        auto* window = tlb_manager->get_tlb_window(tlb_core);
        if (window == nullptr) {
            return nullptr;
        }
        const uint64_t window_offset = window->get_base_address() - window->handle_ref().get_config().local_offset;
        return reinterpret_cast<volatile uint64_t*>(window->handle_ref().get_base() + addr + window_offset);
    } catch (const std::exception&) {
        return nullptr;
    }
}

struct ClockBracket {
    int64_t host_before_ns = 0;
    int64_t host_after_ns = 0;
    uint64_t device_ticks = 0;
    uint32_t round = 0;

    double host_mid_ns() const {
        return static_cast<double>(host_before_ns) + static_cast<double>(host_after_ns - host_before_ns) / 2.0;
    }
    double half_width_ns() const { return static_cast<double>(host_after_ns - host_before_ns) / 2.0; }
};

TEST_F(RealtimeProfilerSyncTest, RecordMappingMatchesAnIndependentClockRead) {
    SyncRecordCollector collector;

    IDevice* device = mesh_device_->get_devices().front();
    const uint32_t l1_base = mesh_device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t stamp_addr = l1_base;
    const uint32_t stop_addr = l1_base + 16;
    const CoreCoord core{0, 0};
    const CoreCoord vcore = device->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto& cluster = MetalContext::instance().get_cluster();

    volatile uint64_t* mapped_stamp = try_map_l1_word(cluster, device->id(), vcore, stamp_addr);

    const auto host_now_ns = [] { return std::chrono::steady_clock::now().time_since_epoch().count(); };
    const auto read_device_ticks = [&]() -> uint64_t {
        if (mapped_stamp != nullptr) {
            return *mapped_stamp;
        }
        uint32_t words[2] = {0, 0};
        cluster.read_core(words, sizeof(words), tt_cxy_pair(device->id(), vcore), stamp_addr);
        return (static_cast<uint64_t>(words[1]) << 32) | words[0];
    };

    // Each round brackets one device clock read, then runs one program, pairing the two so the comparison below is
    // against the mapping that was live beside that read.
    std::vector<ClockBracket> brackets;
    for (uint32_t round = 0; round < kClockCheckRounds; ++round) {
        const std::vector<uint32_t> zeros(8, 0);
        cluster.write_core(zeros.data(), zeros.size() * sizeof(uint32_t), tt_cxy_pair(device->id(), vcore), stamp_addr);

        Program stamp_program = CreateProgram();
        auto kernel = CreateKernelFromString(
            stamp_program,
            make_clock_stamp_kernel_source(),
            CoreRangeSet(CoreRange(core, core)),
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(stamp_program, kernel, core, {stamp_addr, stop_addr, kClockStampIterations});

        // Same program as the stamper, so the load starts and is torn down by the same stop flag.
        const CoreRange grid = all_cores();
        CoreRangeSet load_cores = CoreRangeSet(grid).subtract(CoreRangeSet(CoreRange(core, core)));
        const std::string load_src = make_clock_load_kernel_source();
        for (auto processor : {DataMovementProcessor::RISCV_1}) {
            auto load_kernel = CreateKernelFromString(
                stamp_program,
                load_src,
                load_cores,
                DataMovementConfig{.processor = processor, .noc = NOC::RISCV_1_default});
            for (const auto& cr : load_cores.ranges()) {
                SetRuntimeArgs(stamp_program, load_kernel, cr, {stop_addr, kClockStampIterations});
            }
        }

        distributed::MeshWorkload stamp_workload;
        stamp_workload.add_program(distributed::MeshCoordinateRange(mesh_device_->shape()), std::move(stamp_program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), stamp_workload, /*blocking=*/false);

        // Wait for the slot to go non-zero rather than assume a launch latency; a timeout fails the assertion
        // instead of hanging, since the kernel ends itself either way.
        const auto launch_deadline = std::chrono::steady_clock::now() + 10s;
        while (read_device_ticks() == 0 && std::chrono::steady_clock::now() < launch_deadline) {
            std::this_thread::sleep_for(1ms);
        }
        ASSERT_NE(read_device_ticks(), 0u) << "the stamping kernel never started";

        // Queued behind the still-running stamper before any reads are taken, so it starts the instant the stop flag
        // lands. Every millisecond between the reads and the record they are checked against adds that much of the
        // record's frequency scatter to the residual as extrapolation error the record's claim does not cover, so
        // the gap is what this test's sensitivity is made of.
        enqueue_sync_program(mesh_device_, kFirstClockCheckRuntimeId + round, all_cores());

        for (uint32_t i = 0; i < kClockReadsPerRound; ++i) {
            ClockBracket bracket;
            bracket.host_before_ns = host_now_ns();
            bracket.device_ticks = read_device_ticks();
            bracket.host_after_ns = host_now_ns();
            bracket.round = round;
            brackets.push_back(bracket);
        }
        ASSERT_NE(brackets.back().device_ticks, brackets[brackets.size() - kClockReadsPerRound].device_ticks)
            << "the stamp did not advance across the reads; the stamper exited (hit its iteration ceiling?) and the "
               "residuals below would be against a frozen tick";

        const uint32_t stop = 1;  // written to every core: the load kernels poll their own L1 for this flag
        const CoreCoord grid_size = mesh_device_->compute_with_storage_grid_size();
        for (uint32_t y = 0; y < grid_size.y; ++y) {
            for (uint32_t x = 0; x < grid_size.x; ++x) {
                const CoreCoord v = device->virtual_core_from_logical_core(CoreCoord{x, y}, CoreType::WORKER);
                cluster.write_core_immediate(&stop, sizeof(stop), tt_cxy_pair(device->id(), v), stop_addr);
            }
        }
        distributed::Finish(mesh_device_->mesh_command_queue());
    }

    quiesce_and_wait_for([&] {
        return collector.runtime_ids_in_range(kFirstClockCheckRuntimeId + kClockCheckRounds - 1).size() >=
               kClockCheckRounds;
    });
    collector.stop();

    std::map<uint32_t, ProgramRealtimeRecord> record_by_runtime_id;
    for (const auto& record : collector.records()) {
        if (record.frequency > 0.0) {
            record_by_runtime_id.insert({record.runtime_id, record});
        }
    }

    double sum_residual_ns = 0.0;
    size_t num_reads = 0;
    double worst_abs_residual_ns = 0.0;
    double worst_excess_ns = 0.0;
    size_t reads_with_excess = 0;
    size_t reads_beyond_claim = 0;
    std::chrono::nanoseconds claimed_error{};
    uint32_t rounds_checked = 0;
    for (uint32_t round = 0; round < kClockCheckRounds; ++round) {
        const auto it = record_by_runtime_id.find(kFirstClockCheckRuntimeId + round);
        if (it == record_by_runtime_id.end()) {
            continue;
        }
        ++rounds_checked;
        const ProgramRealtimeRecord& record = it->second;
        for (const auto& bracket : brackets) {
            if (bracket.round != round) {
                continue;
            }
            const double mapped_host_ns = (static_cast<double>(bracket.device_ticks) -
                                           static_cast<double>(record.clock_sync.device_cycle_offset)) /
                                          record.frequency;
            const double residual = mapped_host_ns - bracket.host_mid_ns();
            sum_residual_ns += residual;
            worst_abs_residual_ns = std::max(worst_abs_residual_ns, std::abs(residual));
            ++num_reads;
            // The excess is what the mapping can be blamed for: a read locates the device clock to no better than
            // half its own bracket, so a residual inside that reflects a slow read, not a wrong mapping.
            const double excess = std::max(0.0, std::abs(residual) - bracket.half_width_ns());
            worst_excess_ns = std::max(worst_excess_ns, excess);
            reads_with_excess += excess > kMaxMappingErrorBeyondReadResolutionNs;
            reads_beyond_claim += excess > static_cast<double>(record.clock_sync.sync_error.count());
            claimed_error = std::max(claimed_error, record.clock_sync.sync_error);
        }
    }
    ASSERT_GE(rounds_checked, kClockCheckRounds / 2) << "too few rounds produced a record to check against";
    ASSERT_GT(num_reads, 0u);
    const double mean_residual_ns = sum_residual_ns / static_cast<double>(num_reads);

    log_info(
        tt::LogTest,
        "[RT profiler sync] independent clock read ({}): {} reads over {} rounds; actual sync error mean={:.0f}ns "
        "max={:.0f}ns; worst error beyond a read's own resolution {:.0f}ns, {} read(s) beyond {:.0f}ns, {} beyond the "
        "record's claim (claim {}ns)",
        mapped_stamp != nullptr ? "mapped load" : "generic read",
        num_reads,
        rounds_checked,
        mean_residual_ns,
        worst_abs_residual_ns,
        worst_excess_ns,
        reads_with_excess,
        kMaxMappingErrorBeyondReadResolutionNs,
        reads_beyond_claim,
        claimed_error.count());

    // Mean is not expected to be zero: a read's true sampling instant is not the bracket midpoint, since the two
    // legs of a PCIe access are unequal -- a constant of the measurement, not an error in the mapping.
    EXPECT_LE(std::abs(mean_residual_ns), kMaxMeanMappingErrorNs)
        << "an independent read of the device clock puts the published mapping further out than a few microseconds";
    // A few reads may land beyond their own bracket when the two legs of the PCIe access split unevenly; more than
    // 1% of them disagreeing beyond their own resolution means the disagreement is the mapping's.
    EXPECT_LE(reads_with_excess, num_reads / 100)
        << "reads disagree with the published mapping by more than they could be wrong by themselves";
    EXPECT_LE(worst_excess_ns, 4 * kMaxMappingErrorBeyondReadResolutionNs)
        << "a single read disagrees with the published mapping far beyond its own resolution";
    // The fixed limits above catch a degraded mapping whatever it claims; this tests the claim itself.
    EXPECT_LE(reads_beyond_claim, num_reads / 100)
        << "independent reads put the mapping outside its own claimed sync_error too often; the published bound "
           "understates the real error";
}

// A ~4s program against a 2s probe ring: its start outlives the ring, so this exercises the peek end to end --
// dispatch_s holds the start timestamp in its L1 for the program's whole run, the idle-floor path pins it while its
// probes are fresh, and the record maps through the pinned placement. The sync_error ceiling is the discriminator:
// the pinned path claims placement-scale (~0.5us), while the unpinned ring-wide fallback would claim
// ride x brackets/ring-span, well past 1.5us at this length. Historically this record was dropped outright.
TEST_F(RealtimeProfilerSyncTest, LongProgramIsDeliveredIntact) {
    SyncRecordCollector collector;
    constexpr uint32_t kLongRuntimeId = 7201;
    // ~4e9 cycles: comfortably past the ring's span at any plausible AICLK.
    const std::string long_kernel_src = R"(
void kernel_main() {
    for (unsigned i = 0; i < 62500000u; ++i) {
#pragma GCC unroll 64
        for (int j = 0; j < 64; ++j) {
            asm("nop");
        }
    }
}
)";
    Program program = CreateProgram();
    CreateKernelFromString(
        program,
        long_kernel_src,
        CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    program.set_runtime_id(static_cast<uint64_t>(kLongRuntimeId));

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device_->shape()), std::move(program));
    const auto window_start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/true);
    const auto window_end = std::chrono::steady_clock::now();

    quiesce_and_wait_for([&] {
        return std::ranges::any_of(collector.records(), [](const ProgramRealtimeRecord& record) {
            return record.runtime_id == kLongRuntimeId;
        });
    });
    collector.stop();

    const auto records = collector.records();
    const auto it = std::ranges::find_if(
        records, [](const ProgramRealtimeRecord& record) { return record.runtime_id == kLongRuntimeId; });
    ASSERT_NE(it, records.end()) << "the long-running program's record was dropped";
    EXPECT_GT(it->duration(), std::chrono::seconds(1));
    EXPECT_LT(it->duration(), std::chrono::seconds(30));
    EXPECT_GE(it->host_start(), window_start - std::chrono::milliseconds(2));
    EXPECT_GE(it->host_end(), window_start);
    EXPECT_LE(it->host_end(), window_end + std::chrono::milliseconds(2));
    EXPECT_GT(it->clock_sync.sync_error, std::chrono::nanoseconds::zero());
    EXPECT_LT(it->clock_sync.sync_error, std::chrono::nanoseconds(1500))
        << "a claim past the placement scale means the start was not pinned and fell back to the ride estimate";
    log_info(
        tt::LogTest,
        "[RT profiler sync] long program: duration={:.3f}s sync_error={}us",
        std::chrono::duration<double>{it->duration()}.count(),
        std::chrono::duration<double, std::micro>{it->clock_sync.sync_error}.count());
}

}  // namespace
}  // namespace tt::tt_metal
