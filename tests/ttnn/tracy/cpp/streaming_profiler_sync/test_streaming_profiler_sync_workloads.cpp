// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The sync check's device workloads, one per subcommand: idle, host_sync and didt. Each opens the system mesh with the
// streaming profiler on; host_sync and didt check the timeline it records. Needs a Blackhole system.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"
#include "kernels/workload_layout.hpp"
#include "llrt/tt_cluster.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

// What the workloads share: opening the system mesh and the per-core flag a kernel sets when it stops waiting for a
// peer.
namespace {

constexpr std::string_view kKernelDir = "tests/ttnn/tracy/cpp/streaming_profiler_sync/kernels/";

// One region of every worker's L1, from the allocator: the kernels' flag (FlagWord) and the multicast's round values.
struct WorkerL1 {
    static constexpr uint32_t kFlagBytes = 64;
    static constexpr uint32_t kValuesBytes = 64 * 1024;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;
    uint32_t flag() const { return static_cast<uint32_t>(buffer->address()); }
    uint32_t ack() const { return flag() + kAckWord * sizeof(uint32_t); }
    uint32_t values() const { return flag() + kFlagBytes; }
};

// One page per L1 bank, and every worker is one bank, so each worker holds the region at the same address.
WorkerL1 reserve_worker_l1(tt::tt_metal::distributed::MeshDevice& mesh) {
    constexpr uint32_t kPerCore = WorkerL1::kFlagBytes + WorkerL1::kValuesBytes;
    const uint32_t banks = mesh.allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    return {tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = uint64_t{kPerCore} * banks},
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = kPerCore, .buffer_type = tt::tt_metal::BufferType::L1},
        &mesh)};
}

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

double median(std::vector<double> values) {
    std::ranges::nth_element(values, values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2));
    return values[values.size() / 2];
}

double run_once(tt::tt_metal::distributed::MeshCommandQueue& cq, tt::tt_metal::distributed::MeshWorkload& workload) {
    const auto start = Clock::now();
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    tt::tt_metal::distributed::Finish(cq);
    return seconds_since(start);
}

std::shared_ptr<tt::tt_metal::distributed::MeshDevice> open_system_mesh(const char* tag) {
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create(
        tt::tt_metal::distributed::MeshDeviceConfig(tt::tt_metal::distributed::SystemMesh::instance().shape()),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1);
    if (!tt::tt_metal::experimental::streaming_profiler::IsActive()) {
        std::fprintf(stderr, "[%s] the streaming profiler is not running\n", tag);
        mesh_device->close();
        return nullptr;
    }
    return mesh_device;
}

tt::tt_metal::CoreRange worker_grid(tt::tt_metal::distributed::MeshDevice& mesh) {
    const tt::tt_metal::CoreCoord grid = mesh.compute_with_storage_grid_size();
    return tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(grid.x - 1, grid.y - 1));
}

struct FlagCore {
    tt::tt_metal::IDevice* device;
    tt::tt_metal::CoreCoord core;
};

void clear_flags(const WorkerL1& worker_l1, const std::vector<FlagCore>& cores) {
    std::vector<uint32_t> zero(WorkerL1::kFlagBytes / sizeof(uint32_t), 0);
    for (const FlagCore& flag_core : cores) {
        tt::tt_metal::detail::WriteToDeviceL1(flag_core.device, flag_core.core, worker_l1.flag(), zero);
    }
}

uint32_t count_gave_up(const WorkerL1& worker_l1, const std::vector<FlagCore>& cores, const char* tag) {
    uint32_t gave_up = 0;
    for (const FlagCore& flag_core : cores) {
        std::vector<uint32_t> words;
        tt::tt_metal::detail::ReadFromDeviceL1(
            flag_core.device, flag_core.core, worker_l1.flag(), (kGaveUpRoundWord + 1) * sizeof(uint32_t), words);
        if (words[kGaveUpRoundWord] != 0) {
            std::printf(
                "[%s] chip %d core (%zu,%zu) gave up waiting for round %u (flag %u)\n",
                tag,
                flag_core.device->id(),
                flag_core.core.x,
                flag_core.core.y,
                words[kGaveUpRoundWord],
                words[kRoundWord]);
            gave_up++;
        }
    }
    return gave_up;
}

}  // namespace

using namespace tt;
using namespace tt::tt_metal;
namespace sp = tt::tt_metal::experimental::streaming_profiler;

// Opens the mesh and sleeps for --seconds, so the sync check measures clocks no workload disturbs.
namespace idle {

int run(int argc, char** argv) {
    double seconds = 10.0;
    for (int i = 1; i < argc; i++) {
        const std::string_view arg = argv[i];
        if (arg == "--seconds" && i + 1 < argc) {
            seconds = std::strtod(argv[++i], nullptr);
        } else {
            std::fprintf(stderr, "usage: %s [--seconds S]\n", argv[0]);
            return 2;
        }
    }
    auto mesh_device = open_system_mesh("idle");
    if (!mesh_device) {
        return 1;
    }
    std::printf("[idle] %zu chips, %.1f s\n", mesh_device->num_devices(), seconds);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    mesh_device->close();
    return 0;
}

}  // namespace idle

// Checks the device-to-host part of the timeline. For each round the host writes a round number into one worker's L1
// on every chip and polls for that worker's ack, and the worker records an HOST_RX zone in between. Placed on the host
// timeline, every zone must start after the host began its write and before it read the ack. The window is the host's
// MMIO round trip, a few microseconds wide, so the check is loose: it catches a host mapping off by more than that. It
// also fails unless a callback that unregisters itself on its first call runs exactly once.
namespace host_sync {

namespace {
constexpr uint32_t kRounds = 2000;
constexpr auto kAckTimeout = std::chrono::seconds(1);
// Spaced so the capture runs long enough for the sync check to measure, and spans seconds of clock drift.
constexpr auto kRoundGap = std::chrono::milliseconds(1);

struct Window {
    Clock::time_point before, after;
};

double to_us(Clock::duration duration) { return std::chrono::duration<double, std::micro>(duration).count(); }
}  // namespace

int run(int, char**) {
    std::map<uint32_t, std::vector<Clock::time_point>> starts;
    uint64_t dropped_bytes = 0;
    auto registration = sp::RegisterCallback(
        [&](const sp::Batch<sp::RecordType::Zones>& batch) {
            dropped_bytes += batch.dropped_bytes();
            for (const sp::Zone& zone : batch.zones()) {
                if (std::string_view(zone.site().name) == "HOST_RX") {
                    starts[zone.core().chip_id].push_back(zone.start_time());
                }
            }
        },
        "host_sync");
    std::atomic<uint32_t> once_calls{0};
    sp::Callback once;
    once = sp::RegisterCallback(
        [&](const sp::Batch<sp::RecordType::Zones>&) {
            if (once_calls++ == 0) {
                once = {};
            }
        },
        "host_sync-once");

    auto mesh_device = open_system_mesh("host_sync");
    if (!mesh_device) {
        return 1;
    }
    const WorkerL1 worker_l1 = reserve_worker_l1(*mesh_device);
    const CoreCoord core(0, 0);
    std::vector<FlagCore> cores;
    for (IDevice* device : mesh_device->get_devices()) {
        cores.push_back({device, core});
    }
    clear_flags(worker_l1, cores);

    Program program = CreateProgram();
    const auto kernel = CreateKernel(
        program,
        std::string(kKernelDir) + "host_poke_dm.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, core, {worker_l1.flag(), kRounds});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    // ReadFromDeviceL1 barriers every core on the chip first, which takes about 0.5 ms and would widen each window by
    // that much, so the ack poll reads the word directly.
    const auto& cluster = MetalContext::instance().get_cluster();
    std::map<uint32_t, std::vector<Window>> windows;
    bool timed_out = false;
    std::vector<uint32_t> round_word(1, 0);
    for (uint32_t round = 1; round <= kRounds && !timed_out; round++) {
        std::this_thread::sleep_for(kRoundGap);
        round_word[0] = round;
        for (const FlagCore& flag_core : cores) {
            const tt_cxy_pair ack_core(
                flag_core.device->id(),
                flag_core.device->virtual_core_from_logical_core(flag_core.core, CoreType::WORKER));
            const Clock::time_point before = Clock::now();
            detail::WriteToDeviceL1(flag_core.device, flag_core.core, worker_l1.flag(), round_word);
            while (true) {
                uint32_t ack = 0;
                cluster.read_core(&ack, sizeof(ack), ack_core, worker_l1.ack());
                if (ack == round) {
                    break;
                }
                if (Clock::now() - before > kAckTimeout) {
                    std::printf("[host_sync] chip %d gave no ack for round %u\n", flag_core.device->id(), round);
                    timed_out = true;
                    break;
                }
            }
            windows[flag_core.device->id()].push_back({before, Clock::now()});
        }
    }
    distributed::Finish(cq);
    const uint32_t gave_up = count_gave_up(worker_l1, cores, "host_sync");
    mesh_device->close();
    registration = {};

    size_t failed = 0;
    for (const auto& [chip, chip_windows] : windows) {
        const std::vector<Clock::time_point>& zone_starts = starts[chip];
        if (zone_starts.size() != chip_windows.size()) {
            std::printf(
                "[host_sync] chip %u: FAIL, %zu zones for %zu rounds\n", chip, zone_starts.size(), chip_windows.size());
            failed++;
            continue;
        }
        size_t outside = 0;
        std::vector<double> after_write, before_ack, width;
        for (size_t k = 0; k < chip_windows.size(); k++) {
            const double write_to_zone_us = to_us(zone_starts[k] - chip_windows[k].before);
            const double zone_to_ack_us = to_us(chip_windows[k].after - zone_starts[k]);
            outside += write_to_zone_us < 0.0 || zone_to_ack_us < 0.0 ? 1 : 0;
            after_write.push_back(write_to_zone_us);
            before_ack.push_back(zone_to_ack_us);
            width.push_back(to_us(chip_windows[k].after - chip_windows[k].before));
        }
        const bool pass = outside == 0;
        failed += pass ? 0 : 1;
        std::printf(
            "[host_sync] chip %u: %s, %zu of %zu zones outside their window; zone after the write p50 %.2f us (min "
            "%.2f), before the ack p50 %.2f us (min %.2f); window p50 %.2f us\n",
            chip,
            pass ? "PASS" : "FAIL",
            outside,
            chip_windows.size(),
            median(after_write),
            std::ranges::min(after_write),
            median(before_ack),
            std::ranges::min(before_ack),
            median(width));
    }
    const bool pass = failed == 0 && !timed_out && gave_up == 0 && dropped_bytes == 0 && once_calls == 1;
    std::printf(
        "[host_sync] %s: %zu of %zu chips failed, %u cores gave up, %llu bytes dropped, a callback that unregisters "
        "itself ran %u times%s\n",
        pass ? "PASS" : "FAIL",
        failed,
        windows.size(),
        gave_up,
        static_cast<unsigned long long>(dropped_bytes),
        once_calls.load(),
        timed_out ? ", an ack timed out" : "");
    return pass ? 0 : 1;
}

}  // namespace host_sync

// Checks each chip's timeline core to core. One Tensix core per chip broadcasts rounds over each NoC, and every worker
// records the arrival as an MC_RX zone. A multicast takes a fixed 9 cycles per router hop, so once each arrival is on
// the host timeline and its hops are subtracted, every core should report the same instant. A run fails if any core's
// mean differs from the others' by a cycle or more, a round goes missing or records are dropped.
namespace multicast {

namespace {
constexpr uint32_t kRounds = 4000;
static_assert(kRoundValueStrideBytes * (kRounds + 1) <= WorkerL1::kValuesBytes);
static_assert(kRounds % 2 == 0);
// 9 cycles per router hop (BlackholeA0/NoC README.md; measured 9.00).
constexpr double kCyclesPerHop = 9.0;

struct Arrival {
    int64_t tsc;
    int64_t cycles;
    uint16_t chip;
    uint8_t x, y;
    uint8_t physical_x, physical_y;
};

distributed::MeshWorkload make_multicast(
    distributed::MeshDevice& mesh, const WorkerL1& worker_l1, uint32_t runtime_id) {
    const CoreRange cores = worker_grid(mesh);
    const CoreCoord &low = cores.start_coord, &high = cores.end_coord;
    const CoreCoord virtual_low = mesh.worker_core_from_logical_core(low);
    const CoreCoord virtual_high = mesh.worker_core_from_logical_core(high);
    const uint32_t num_dests = static_cast<uint32_t>(cores.size() - 1);
    Program program = CreateProgram();
    program.set_runtime_id(runtime_id);
    const auto kernel = CreateKernel(
        program,
        std::string(kKernelDir) + "multicast_dm.cpp",
        CoreRangeSet(cores),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    for (const CoreCoord& core : cores) {
        const MulticastRole role = core == low ? MulticastRole::Noc0Source
                                               : (core == high ? MulticastRole::Noc1Source : MulticastRole::Receiver);
        // A NoC 1 rectangle runs from its start corner at the high coordinates down to the low ones.
        const CoreCoord& rect_start = role == MulticastRole::Noc1Source ? virtual_high : virtual_low;
        const CoreCoord& rect_end = role == MulticastRole::Noc1Source ? virtual_low : virtual_high;
        SetRuntimeArgs(
            program,
            kernel,
            core,
            {static_cast<uint32_t>(role),
             static_cast<uint32_t>(rect_start.x),
             static_cast<uint32_t>(rect_start.y),
             static_cast<uint32_t>(rect_end.x),
             static_cast<uint32_t>(rect_end.y),
             num_dests,
             worker_l1.flag(),
             worker_l1.values(),
             kRounds});
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    return workload;
}

using Round = std::map<CoreCoord, const Arrival*>;
struct Lane {
    CoreCoord src;
    std::vector<Round> rounds;
};
struct LaneKey {
    uint32_t chip = 0, noc = 0;
    auto operator<=>(const LaneKey&) const = default;
};
struct Lanes {
    std::map<LaneKey, Lane> by_chip_noc;
    bool incomplete = false;
};

// The NoC 0 source sends the odd rounds and the NoC 1 source the even ones, and a source never receives its own
// broadcast.
size_t round_of_arrival(
    const CoreCoord& core, size_t arrival_index, const CoreCoord& noc0_source, const CoreCoord& noc1_source) {
    if (core == noc0_source) {
        return 2 * arrival_index + 2;
    }
    if (core == noc1_source) {
        return 2 * arrival_index + 1;
    }
    return arrival_index + 1;
}

Lanes group_lanes(const std::vector<Arrival>& arrivals, const CoreRange& cores, size_t num_chips) {
    std::map<uint32_t, std::vector<const Arrival*>> by_chip;
    for (const Arrival& arrival : arrivals) {
        by_chip[arrival.chip].push_back(&arrival);
    }
    const CoreCoord &noc0_source = cores.start_coord, &noc1_source = cores.end_coord;
    Lanes lanes;
    lanes.incomplete = by_chip.size() != num_chips;
    for (auto& [chip, chip_arrivals] : by_chip) {
        std::map<CoreCoord, std::vector<const Arrival*>> by_core;
        for (const Arrival* arrival : chip_arrivals) {
            by_core[CoreCoord{arrival->x, arrival->y}].push_back(arrival);
        }
        bool counts_ok = by_core.size() == cores.size();
        for (auto& [core, core_arrivals] : by_core) {
            std::ranges::sort(core_arrivals, {}, &Arrival::cycles);
            const size_t expected = core == noc0_source || core == noc1_source ? kRounds / 2 : kRounds;
            if (core_arrivals.size() != expected) {
                std::printf(
                    "[multicast] chip %u core (%zu,%zu): %zu arrivals, expected %zu\n",
                    chip,
                    core.x,
                    core.y,
                    core_arrivals.size(),
                    expected);
                counts_ok = false;
            }
        }
        if (!counts_ok) {
            lanes.incomplete = true;
            continue;
        }
        Lane& noc0_lane = lanes.by_chip_noc[{.chip = chip, .noc = 0}];
        Lane& noc1_lane = lanes.by_chip_noc[{.chip = chip, .noc = 1}];
        const Arrival* noc0_source_arrival = by_core.at(noc0_source).front();
        const Arrival* noc1_source_arrival = by_core.at(noc1_source).front();
        noc0_lane.src = CoreCoord(noc0_source_arrival->physical_x, noc0_source_arrival->physical_y);
        noc1_lane.src = CoreCoord(noc1_source_arrival->physical_x, noc1_source_arrival->physical_y);
        noc0_lane.rounds.resize(kRounds / 2);
        noc1_lane.rounds.resize(kRounds / 2);
        for (const auto& [core, core_arrivals] : by_core) {
            for (size_t k = 0; k < core_arrivals.size(); k++) {
                const size_t round = round_of_arrival(core, k, noc0_source, noc1_source);
                ((round & 1u) ? noc0_lane : noc1_lane).rounds[(round - 1) / 2][core] = core_arrivals[k];
            }
        }
    }
    lanes.incomplete = lanes.incomplete || lanes.by_chip_noc.empty();
    return lanes;
}

struct LaneSpan {
    size_t cores = 0, rounds = 0;
    double ns_per_cycle = 0.0, span_ns = 0.0, noise_ns = 0.0;
};

LaneSpan solve_lane(uint32_t chip, uint32_t noc, const Lane& lane) {
    const double ns_per_tsc = sp::NsPerTscTick();
    const std::vector<Round>& rounds = lane.rounds;
    const CoreCoord src = lane.src;
    std::vector<CoreCoord> cores;
    std::vector<double> hops;
    for (const auto& [core, arrival] : rounds.front()) {
        const CoreCoord physical(arrival->physical_x, arrival->physical_y);
        const int hops_x = noc == 0 ? static_cast<int>(physical.x) - static_cast<int>(src.x)
                                    : static_cast<int>(src.x) - static_cast<int>(physical.x);
        const int hops_y = noc == 0 ? static_cast<int>(physical.y) - static_cast<int>(src.y)
                                    : static_cast<int>(src.y) - static_cast<int>(physical.y);
        TT_FATAL(
            hops_x >= 0 && hops_y >= 0,
            "chip {} NoC {}: core {} is behind the source {}",
            chip,
            noc,
            physical.str(),
            src.str());
        cores.push_back(core);
        hops.push_back(static_cast<double>(hops_x + hops_y));
    }
    const size_t num_cores = cores.size(), num_rounds = rounds.size();
    std::vector<std::vector<double>> residual_ns(num_cores, std::vector<double>(num_rounds));
    double ns_per_cycle_sum = 0.0;
    for (size_t k = 0; k < num_rounds; k++) {
        const size_t neighbour_round = k + 1 < num_rounds ? k + 1 : k - 1;
        const Arrival* ref = rounds[k].at(cores[0]);
        const Arrival* ref_in_neighbour = rounds[neighbour_round].at(cores[0]);
        const double round_ns_per_cycle = static_cast<double>(ref_in_neighbour->tsc - ref->tsc) * ns_per_tsc /
                                          static_cast<double>(ref_in_neighbour->cycles - ref->cycles);
        ns_per_cycle_sum += round_ns_per_cycle;
        double round_residual_sum = 0.0;
        for (size_t i = 0; i < num_cores; i++) {
            const Arrival* arrival = rounds[k].at(cores[i]);
            residual_ns[i][k] = static_cast<double>(arrival->tsc - ref->tsc) * ns_per_tsc -
                                kCyclesPerHop * hops[i] * round_ns_per_cycle;
            round_residual_sum += residual_ns[i][k];
        }
        for (size_t i = 0; i < num_cores; i++) {
            residual_ns[i][k] -= round_residual_sum / static_cast<double>(num_cores);
        }
    }
    const double ns_per_cycle = ns_per_cycle_sum / static_cast<double>(num_rounds);
    double min_mean_ns = std::numeric_limits<double>::max(), max_mean_ns = std::numeric_limits<double>::lowest();
    double stderr_ns = 0.0;
    for (size_t i = 0; i < num_cores; i++) {
        double sum = 0.0, sum_squares = 0.0;
        for (const double residual : residual_ns[i]) {
            sum += residual;
            sum_squares += residual * residual;
        }
        const double mean = sum / static_cast<double>(num_rounds);
        stderr_ns = std::max(
            stderr_ns,
            std::sqrt(std::max(sum_squares / static_cast<double>(num_rounds) - mean * mean, 0.0) / num_rounds));
        min_mean_ns = std::min(min_mean_ns, mean);
        max_mean_ns = std::max(max_mean_ns, mean);
    }
    return LaneSpan{
        .cores = num_cores,
        .rounds = num_rounds,
        .ns_per_cycle = ns_per_cycle,
        .span_ns = max_mean_ns - min_mean_ns,
        .noise_ns = stderr_ns};
}

// Collects every run's arrivals by the run's runtime id. Records reach the host by the capture's end, so verify() runs
// after the mesh has closed.
class Check {
public:
    Check() :
        registration_(sp::RegisterCallback(
            [this](const sp::Batch<sp::RecordType::Zones>& batch) {
                dropped_bytes_ += batch.dropped_bytes();
                for (const sp::Zone& zone : batch.zones()) {
                    if (std::string_view(zone.site().name) != "MC_RX") {
                        continue;
                    }
                    const sp::Core core = zone.core();
                    by_run_[zone.runtime_id()].push_back(Arrival{
                        .tsc = zone.start_tsc(),
                        .cycles = static_cast<int64_t>(zone.start_device_cycles()),
                        .chip = static_cast<uint16_t>(core.chip_id),
                        .x = static_cast<uint8_t>(core.logical.x),
                        .y = static_cast<uint8_t>(core.logical.y),
                        .physical_x = static_cast<uint8_t>(core.physical.x),
                        .physical_y = static_cast<uint8_t>(core.physical.y)});
                }
            },
            "multicast")) {}
    Check(const Check&) = delete;
    Check& operator=(const Check&) = delete;

    void run(distributed::MeshDevice& mesh, std::string_view name) {
        cores_ = worker_grid(mesh);
        num_chips_ = mesh.get_devices().size();
        const WorkerL1 worker_l1 = reserve_worker_l1(mesh);
        std::vector<FlagCore> cores;
        for (IDevice* device : mesh.get_devices()) {
            for (const CoreCoord& core : cores_) {
                cores.push_back({device, core});
            }
        }
        clear_flags(worker_l1, cores);
        const auto runtime_id = static_cast<uint32_t>(runs_.size() + 1);
        distributed::MeshWorkload workload = make_multicast(mesh, worker_l1, runtime_id);
        run_once(mesh.mesh_command_queue(), workload);
        runs_.push_back({std::string(name), count_gave_up(worker_l1, cores, "multicast")});
        std::printf(
            "[multicast] %s: %zux%zu Tensix cores x %u rounds (%u a NoC) on %zu chips\n",
            runs_.back().name.c_str(),
            cores_.grid_size().x,
            cores_.grid_size().y,
            kRounds,
            kRounds / 2,
            num_chips_);
        std::fflush(stdout);
    }

    bool verify() {
        registration_ = {};
        double worst_span_cycles = 0.0;
        uint32_t total_gave_up = 0;
        size_t failed = 0;
        for (size_t k = 0; k < runs_.size(); k++) {
            const Run& result = runs_[k];
            std::printf("[multicast] run %zu/%zu %s\n", k + 1, runs_.size(), result.name.c_str());
            const Lanes lanes = group_lanes(by_run_[static_cast<uint32_t>(k + 1)], cores_, num_chips_);
            double run_worst_span_cycles = 0.0;
            std::printf("chip noc cores rounds  ns/cycle  span ns (cycles)  noise ns\n");
            for (const auto& [key, lane] : lanes.by_chip_noc) {
                const LaneSpan span = solve_lane(key.chip, key.noc, lane);
                const double span_cycles = span.span_ns / span.ns_per_cycle;
                run_worst_span_cycles = std::max(run_worst_span_cycles, span_cycles);
                std::printf(
                    "%4u %3u %5zu  %5zu  %.4f   %6.3f (%5.2f)    %6.3f\n",
                    key.chip,
                    key.noc,
                    span.cores,
                    span.rounds,
                    span.ns_per_cycle,
                    span.span_ns,
                    span_cycles,
                    span.noise_ns);
            }
            const bool pass = run_worst_span_cycles < 1.0 && result.gave_up == 0 && !lanes.incomplete;
            std::printf(
                "[multicast] %s: %s, worst span %.2f cycles; %u cores gave up%s\n",
                result.name.c_str(),
                pass ? "PASS" : "FAIL",
                run_worst_span_cycles,
                result.gave_up,
                lanes.incomplete ? "; SOME CHIPS MISSING OR WITH WRONG ARRIVAL COUNTS" : "");
            worst_span_cycles = std::max(worst_span_cycles, run_worst_span_cycles);
            total_gave_up += result.gave_up;
            failed += pass ? 0 : 1;
        }
        const bool pass = failed == 0 && dropped_bytes_ == 0;
        std::printf(
            "[multicast] %s: worst core-to-core span of the device-local timeline %.2f cycles over %zu runs, "
            "%zu failed; %u cores gave up, %llu bytes dropped\n",
            pass ? "PASS" : "FAIL",
            worst_span_cycles,
            runs_.size(),
            failed,
            total_gave_up,
            static_cast<unsigned long long>(dropped_bytes_));
        return pass;
    }

private:
    struct Run {
        std::string name;
        uint32_t gave_up;
    };
    std::vector<Run> runs_;
    CoreRange cores_{CoreCoord{0, 0}};
    size_t num_chips_ = 0;
    std::map<uint32_t, std::vector<Arrival>> by_run_;
    uint64_t dropped_bytes_ = 0;
    sp::Callback registration_;
};
}  // namespace

}  // namespace multicast

// Fabric traffic under the profiler. Over 2D fabric, one worker on each side of every linked chip pair ping-pongs
// atomic increments for kSeconds, recording a PP_TX zone per send and a PP_RX zone per arrival. A run fails if a kernel
// gives up waiting for its peer or any round's zones don't all reach the host. As a sanity check of the synced
// timeline, it also fails if a pair's first rounds break causality or its two directions' replies take different times.
// The sync gate runs it with the sync check on, so its traffic shares the routers with the link sync.
namespace pingpong {

namespace {
constexpr double kSeconds = 10.0;
// A pass takes at least 36 ms (1.81 us a round trip), so the host work between passes is the minority.
constexpr uint32_t kRounds = 20000;

struct Pair {
    uint32_t chip_a, chip_b;
    CoreCoord core_a, core_b;
    uint32_t link_a, link_b;
};

constexpr uint32_t kTimedRounds = 1000;
struct Stamp {
    uint64_t cycles;
    int64_t tsc;
    bool tx;
};
// Every zone is counted; the first kTimedRounds rounds' zones are also kept for the timeline check.
struct Stamps {
    size_t tx = 0, rx = 0;
    std::vector<Stamp> first;
};
using StampsByCore = std::map<std::pair<uint32_t, CoreCoord>, Stamps>;
struct Chip {
    distributed::MeshCoordinate coord;
    IDevice* device;
    tt::tt_fabric::FabricNodeId node;
};
using Chips = std::map<uint32_t, Chip>;

std::vector<Pair> find_pairs(distributed::MeshDevice& mesh_device, const Chips& chips) {
    const CoreCoord grid = mesh_device.compute_with_storage_grid_size();
    std::map<uint32_t, uint32_t> next_worker;
    const auto take_worker = [&](uint32_t chip) {
        const uint32_t worker = next_worker[chip]++;
        TT_FATAL(worker < grid.x * grid.y, "chip {} has more fabric neighbours than worker cores", chip);
        return CoreCoord(worker % grid.x, worker / grid.x);
    };
    std::vector<Pair> pairs;
    for (const auto& [id_a, chip_a] : chips) {
        for (const auto& [id_b, chip_b] : chips) {
            if (id_a >= id_b || tt::tt_fabric::get_neighbor_eth_directions(chip_a.node, chip_b.node).empty()) {
                continue;
            }
            const auto links_ab = tt::tt_fabric::get_forwarding_link_indices(chip_a.node, chip_b.node);
            const auto links_ba = tt::tt_fabric::get_forwarding_link_indices(chip_b.node, chip_a.node);
            TT_FATAL(!links_ab.empty() && !links_ba.empty(), "no fabric link between chips {} and {}", id_a, id_b);
            pairs.push_back(Pair{
                .chip_a = id_a,
                .chip_b = id_b,
                .core_a = take_worker(id_a),
                .core_b = take_worker(id_b),
                .link_a = links_ab.front(),
                .link_b = links_ba.front()});
        }
    }
    return pairs;
}

void arm(
    Program& program,
    distributed::MeshDevice& mesh_device,
    const WorkerL1& worker_l1,
    const CoreCoord& core,
    uint32_t role,
    const CoreCoord& peer,
    const tt::tt_fabric::FabricNodeId& src,
    const tt::tt_fabric::FabricNodeId& dst,
    uint32_t link_idx) {
    const auto kernel = CreateKernel(
        program,
        std::string(kKernelDir) + "pingpong_fabric_dm.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    const CoreCoord virtual_peer = mesh_device.worker_core_from_logical_core(peer);
    std::vector<uint32_t> args = {
        role,
        static_cast<uint32_t>(virtual_peer.x),
        static_cast<uint32_t>(virtual_peer.y),
        worker_l1.flag(),
        kRounds,
        static_cast<uint32_t>(dst.chip_id),
        static_cast<uint32_t>(dst.mesh_id.get())};
    tt::tt_fabric::append_fabric_connection_rt_args(src, dst, link_idx, program, core, args);
    SetRuntimeArgs(program, kernel, core, args);
}

distributed::MeshWorkload build_workload(
    distributed::MeshDevice& mesh_device,
    const WorkerL1& worker_l1,
    const std::vector<Pair>& pairs,
    const Chips& chips,
    uint32_t runtime_id) {
    std::map<uint32_t, Program> programs;
    for (const uint32_t chip : std::views::keys(chips)) {
        programs.emplace(chip, CreateProgram()).first->second.set_runtime_id(runtime_id);
    }
    for (const Pair& pair : pairs) {
        const auto &node_a = chips.at(pair.chip_a).node, &node_b = chips.at(pair.chip_b).node;
        arm(programs.at(pair.chip_a), mesh_device, worker_l1, pair.core_a, 0, pair.core_b, node_a, node_b, pair.link_a);
        arm(programs.at(pair.chip_b), mesh_device, worker_l1, pair.core_b, 1, pair.core_a, node_b, node_a, pair.link_b);
    }
    distributed::MeshWorkload workload;
    for (auto& [chip, program] : programs) {
        const auto& coord = chips.at(chip).coord;
        workload.add_program(distributed::MeshCoordinateRange(coord, coord), std::move(program));
    }
    return workload;
}

uint32_t count_missing(const std::vector<Pair>& pairs, const StampsByCore& by_core, size_t expected) {
    uint32_t failures = 0;
    for (const Pair& pair : pairs) {
        const auto found_a = by_core.find({pair.chip_a, pair.core_a});
        const auto found_b = by_core.find({pair.chip_b, pair.core_b});
        const size_t atx = found_a == by_core.end() ? 0 : found_a->second.tx;
        const size_t arx = found_a == by_core.end() ? 0 : found_a->second.rx;
        const size_t btx = found_b == by_core.end() ? 0 : found_b->second.tx;
        const size_t brx = found_b == by_core.end() ? 0 : found_b->second.rx;
        if (atx != expected || brx != expected || btx != expected || arx != expected) {
            std::printf(
                "chip %u - chip %u: stamps %zu/%zu/%zu/%zu of %zu\n",
                pair.chip_a,
                pair.chip_b,
                atx,
                brx,
                btx,
                arx,
                expected);
            failures++;
        }
    }
    return failures;
}
// Half the reply difference is the pair's clock error plus half the link's asymmetry; on a LoudBox it stays within
// +-5 ns.
constexpr double kReplySkewBoundNs = 10.0;

uint32_t check_timeline(const std::vector<Pair>& pairs, StampsByCore& by_core) {
    const double ns_per_tsc = sp::NsPerTscTick();
    uint32_t failures = 0;
    for (const Pair& pair : pairs) {
        std::vector<Stamp>& stamps_a = by_core[{pair.chip_a, pair.core_a}].first;
        std::vector<Stamp>& stamps_b = by_core[{pair.chip_b, pair.core_b}].first;
        for (std::vector<Stamp>* stamps : {&stamps_a, &stamps_b}) {
            std::ranges::sort(*stamps, {}, &Stamp::cycles);
        }
        enum Leg : size_t { kPingAB, kPingBA, kReplyAB, kReplyBA, kLegs };
        std::array<std::vector<double>, kLegs> legs;
        uint32_t acausal = 0, misread = 0;
        for (size_t k = 0; k < std::min(stamps_a.size(), stamps_b.size()) / 2; k++) {
            // Round k + 1: core_b (role 1) sends first on odd rounds, core_a on even ones.
            const bool b_first = (k & 1u) == 0;
            const Stamp* sender = &(b_first ? stamps_b : stamps_a)[2 * k];
            const Stamp* replier = &(b_first ? stamps_a : stamps_b)[2 * k];
            if (!sender[0].tx || sender[1].tx || replier[0].tx || !replier[1].tx) {
                misread++;
                continue;
            }
            const double ping_ns = static_cast<double>(replier[0].tsc - sender[0].tsc) * ns_per_tsc;
            const double reply_ns = static_cast<double>(sender[1].tsc - replier[1].tsc) * ns_per_tsc;
            acausal += (ping_ns <= 0.0 ? 1u : 0u) + (reply_ns <= 0.0 ? 1u : 0u);
            legs[b_first ? kPingBA : kPingAB].push_back(ping_ns);
            legs[b_first ? kReplyAB : kReplyBA].push_back(reply_ns);
        }
        if (std::ranges::any_of(legs, [](const std::vector<double>& leg) { return leg.empty(); })) {
            std::printf("chip %u - chip %u: FAIL, no timed rounds in some direction\n", pair.chip_a, pair.chip_b);
            failures++;
            continue;
        }
        std::array<double, kLegs> leg_median{};
        std::ranges::transform(legs, leg_median.begin(), [](const std::vector<double>& leg) { return median(leg); });
        const double reply_skew_ns = (leg_median[kReplyAB] - leg_median[kReplyBA]) / 2;
        const bool pair_ok = acausal == 0 && misread == 0 && std::abs(reply_skew_ns) <= kReplySkewBoundNs;
        std::printf(
            "chip %u - chip %u: %s, ping %.0f / %.0f ns, reply %.0f / %.0f ns (a to b / b to a), half the reply "
            "difference %+.1f ns; %u acausal, %u misread of %zu rounds\n",
            pair.chip_a,
            pair.chip_b,
            pair_ok ? "ok" : "FAIL",
            leg_median[kPingAB],
            leg_median[kPingBA],
            leg_median[kReplyAB],
            leg_median[kReplyBA],
            reply_skew_ns,
            acausal,
            misread,
            legs[kPingAB].size() + legs[kPingBA].size() + misread);
        failures += pair_ok ? 0 : 1;
    }
    return failures;
}

// Like multicast::Check, for the PP_TX and PP_RX zones.
class Check {
public:
    Check() :
        registration_(sp::RegisterCallback(
            [this](const sp::Batch<sp::RecordType::Zones>& batch) {
                for (const sp::Zone& zone : batch.zones()) {
                    const std::string_view name = zone.site().name;
                    if (name == "PP_TX" || name == "PP_RX") {
                        Stamps& stamps = by_run_[zone.runtime_id()][{zone.core().chip_id, zone.core().logical}];
                        (name == "PP_TX" ? stamps.tx : stamps.rx)++;
                        if (stamps.first.size() < 2 * kTimedRounds) {
                            stamps.first.push_back({zone.start_device_cycles(), zone.start_tsc(), name == "PP_TX"});
                        }
                    }
                }
            },
            "pingpong")) {}
    Check(const Check&) = delete;
    Check& operator=(const Check&) = delete;

    // Needs the mesh opened with 2D fabric.
    void run(distributed::MeshDevice& mesh_device, std::string_view name) {
        Chips chips;
        for (const auto& coord : distributed::MeshCoordinateRange(mesh_device.shape())) {
            IDevice* device = mesh_device.get_device(coord);
            chips.emplace(
                device->id(),
                Chip{coord, device, tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id())});
        }
        const auto runtime_id = static_cast<uint32_t>(runs_.size() + 1);
        Run& result = runs_.emplace_back(Run{.name = std::string(name), .pairs = find_pairs(mesh_device, chips)});
        TT_FATAL(!result.pairs.empty(), "no fabric-linked chip pairs");
        std::vector<FlagCore> flag_cores;
        for (const Pair& pair : result.pairs) {
            flag_cores.push_back({chips.at(pair.chip_a).device, pair.core_a});
            flag_cores.push_back({chips.at(pair.chip_b).device, pair.core_b});
        }
        const WorkerL1 worker_l1 = reserve_worker_l1(mesh_device);
        distributed::MeshWorkload workload = build_workload(mesh_device, worker_l1, result.pairs, chips, runtime_id);
        distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
        const auto start = Clock::now();
        double last_pass_s = 0.0;
        do {
            clear_flags(worker_l1, flag_cores);
            last_pass_s = run_once(cq, workload);
            result.gave_up += count_gave_up(worker_l1, flag_cores, "pingpong");
            result.passes++;
        } while (seconds_since(start) < kSeconds);
        std::printf(
            "[pingpong] %s: %zu linked pairs on %zu chips, %u passes of %u rounds, %.1f ms each\n",
            result.name.c_str(),
            result.pairs.size(),
            chips.size(),
            result.passes,
            kRounds,
            last_pass_s * 1e3);
        std::fflush(stdout);
    }

    bool verify() {
        registration_ = {};
        size_t failed = 0;
        for (size_t k = 0; k < runs_.size(); k++) {
            const Run& result = runs_[k];
            std::printf("[pingpong] run %zu/%zu %s\n", k + 1, runs_.size(), result.name.c_str());
            StampsByCore& by_core = by_run_[static_cast<uint32_t>(k + 1)];
            const uint32_t failures = result.gave_up +
                                      count_missing(result.pairs, by_core, size_t{result.passes} * kRounds) +
                                      check_timeline(result.pairs, by_core);
            std::printf("[pingpong] %s: %s\n", result.name.c_str(), failures == 0 ? "PASS" : "FAIL");
            failed += failures == 0 ? 0 : 1;
        }
        std::printf("[pingpong] %s: %zu of %zu runs failed\n", failed == 0 ? "PASS" : "FAIL", failed, runs_.size());
        return failed == 0;
    }

private:
    struct Run {
        std::string name;
        std::vector<Pair> pairs;
        uint32_t passes = 0, gave_up = 0;
    };
    std::vector<Run> runs_;
    std::map<uint32_t, StampsByCore> by_run_;
    sp::Callback registration_;
};
}  // namespace

}  // namespace pingpong

// The FF1 matmul and SDPA di/dt ops (tests/didt/test_ff1_matmul.py's "all and without_gelu" and test_sdpa_op.py's "all
// and bf16_HiFi2"), whose throttling moves AICLK, on one mesh with 2D fabric. The multicast and ping-pong checks run
// before, between and after them, and every run of each must pass.
namespace didt {

namespace {
constexpr uint32_t kFf1Iterations = 3000;
constexpr uint32_t kSdpaIterations = 500;

ttnn::Tensor random_normal(
    distributed::MeshDevice& mesh, const ttnn::Shape& shape, DataType dtype, std::mt19937& generator) {
    std::normal_distribution<float> normal;
    std::vector<float> values(shape.volume());
    std::ranges::generate(values, [&] { return normal(generator); });
    const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    return ttnn::Tensor::from_vector(std::move(values), spec, &mesh);
}

template <typename Op>
void loop(distributed::MeshDevice& mesh, uint32_t iterations, const Op& op) {
    for (uint32_t i = 0; i < iterations; i++) {
        ttnn::Tensor out = op();
        distributed::Synchronize(mesh, std::nullopt);
        out.deallocate(/*force=*/true);
    }
}

void ff1_matmul(distributed::MeshDevice& mesh) {
    constexpr uint32_t kPerCoreM = 4, kPerCoreN = 72, kShardWidth = 576;
    const CoreCoord grid = mesh.compute_with_storage_grid_size();
    std::mt19937 generator(1234);
    const ttnn::MemoryConfig in0_config(
        TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        ShardSpec(CoreRangeSet(worker_grid(mesh)), {32 * kPerCoreM, kShardWidth}, ShardOrientation::ROW_MAJOR));
    const ttnn::Tensor in0 = ttnn::to_memory_config(
        random_normal(
            mesh, ttnn::Shape({1, 1, 32 * kPerCoreM * grid.y, kShardWidth * grid.x}), DataType::BFLOAT16, generator),
        in0_config);
    const ttnn::Tensor in1 = random_normal(
        mesh, ttnn::Shape({1, 1, kShardWidth * grid.x, 32 * kPerCoreN * grid.x}), DataType::BFLOAT8_B, generator);
    const ttnn::operations::matmul::MatmulProgramConfig program_config =
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
            .compute_with_storage_grid_size = grid,
            .in0_block_w = 3,
            .out_subblock_h = 1,
            .out_subblock_w = 8,
            .out_block_h = kPerCoreM,
            .out_block_w = kPerCoreN,
            .per_core_M = kPerCoreM,
            .per_core_N = kPerCoreN,
            .transpose_mcast = false};
    const ttnn::BlackholeComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::LoFi,
        .math_approx_mode = false,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true};
    loop(mesh, kFf1Iterations, [&] {
        return ttnn::matmul(
            in0,
            in1,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG,
            DataType::BFLOAT16,
            program_config,
            /*activation=*/std::nullopt,
            compute_config);
    });
}

void sdpa(distributed::MeshDevice& mesh) {
    const ttnn::Shape shape({1, 10, 9472, 128});
    std::mt19937 generator(1234);
    const ttnn::Tensor q = random_normal(mesh, shape, DataType::BFLOAT16, generator);
    const ttnn::Tensor k = random_normal(mesh, shape, DataType::BFLOAT16, generator);
    const ttnn::Tensor v = random_normal(mesh, shape, DataType::BFLOAT16, generator);
    const ttnn::operations::transformer::SDPAProgramConfig program_config{
        .compute_with_storage_grid_size = mesh.compute_with_storage_grid_size(),
        .q_chunk_size = 256,
        .k_chunk_size = 256,
        .exp_approx_mode = false};
    const ttnn::DeviceComputeKernelConfig compute_config = ttnn::init_device_compute_kernel_config(
        mesh.arch(),
        std::nullopt,
        MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false);
    loop(mesh, kSdpaIterations, [&] {
        return ttnn::transformer::scaled_dot_product_attention(
            q,
            k,
            v,
            /*attn_mask=*/std::nullopt,
            /*is_causal=*/false,
            /*scale=*/std::nullopt,
            /*sliding_window_size=*/std::nullopt,
            /*memory_config=*/std::nullopt,
            program_config,
            compute_config);
    });
}
}  // namespace

int run(int argc, char** argv) {
    if (argc > 1) {
        std::fprintf(stderr, "usage: %s\n", argv[0]);
        return 2;
    }
    multicast::Check multicast_check;
    pingpong::Check pingpong_check;
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
    auto mesh_device = open_system_mesh("didt");
    if (!mesh_device) {
        return 1;
    }
    const auto checks = [&](std::string_view name) {
        multicast_check.run(*mesh_device, name);
        pingpong_check.run(*mesh_device, name);
    };
    checks("before the ops");
    ff1_matmul(*mesh_device);
    std::printf("[didt] FF1 matmul: %u iterations\n", kFf1Iterations);
    checks("after the FF1 matmul");
    sdpa(*mesh_device);
    std::printf("[didt] SDPA: %u iterations\n", kSdpaIterations);
    checks("after SDPA");
    mesh_device->close();
    const bool multicast_pass = multicast_check.verify();
    const bool pingpong_pass = pingpong_check.verify();
    return multicast_pass && pingpong_pass ? 0 : 1;
}

}  // namespace didt

int main(int argc, char** argv) {
    const std::string_view workload = argc > 1 ? argv[1] : "";
    if (workload == "idle") {
        return idle::run(argc - 1, argv + 1);
    }
    if (workload == "host_sync") {
        return host_sync::run(argc - 1, argv + 1);
    }
    if (workload == "didt") {
        return didt::run(argc - 1, argv + 1);
    }
    std::fprintf(stderr, "usage: %s idle [--seconds S] | host_sync | didt\n", argv[0]);
    return 2;
}
