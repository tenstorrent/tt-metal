// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device workloads for test_streaming_profiler_sync.py, which runs them with the sync check on to measure the accuracy
// of the clock sync between chips. There is one workload per subcommand: idle, host_sync, didt and ccl. Each runs on
// all of the system's chips with the streaming profiler on. host_sync also checks, on the host's clock, that the chip
// sees each host write after the host makes it and before the host gets the chip's ack. didt also checks that multicast
// arrivals line up within each chip and that fabric ping-pongs between chips arrive after they're sent, according to
// the synced timeline.

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
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
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
#include "kernels/sync_workload.hpp"
#include "llrt/tt_cluster.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/all_reduce/all_reduce.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

using namespace tt;
using namespace tt::tt_metal;
namespace streaming_profiler = tt::tt_metal::experimental::streaming_profiler;

namespace {

constexpr std::string_view kKernelDir = "tests/ttnn/tracy/cpp/streaming_profiler_sync/kernels/";

// One region of every worker's L1, taken from the allocator, that holds the kernels' flag (FlagWord) and the
// multicast's round values.
struct WorkerL1 {
    static constexpr uint32_t kFlagBytes = 64;
    static constexpr uint32_t kValuesBytes = 64 * 1024;
    std::shared_ptr<distributed::MeshBuffer> buffer;
    uint32_t flag() const { return static_cast<uint32_t>(buffer->address()); }
    uint32_t ack() const { return flag() + sync_workload::kAckWord * sizeof(uint32_t); }
    uint32_t values() const { return flag() + kFlagBytes; }
};

// Reserves a WorkerL1 through the allocator. It allocates one page per L1 bank, and each worker's L1 is one bank, so
// every worker holds its page at the same address.
WorkerL1 reserve_worker_l1(distributed::MeshDevice& mesh) {
    constexpr uint32_t kPerCore = WorkerL1::kFlagBytes + WorkerL1::kValuesBytes;
    const uint32_t banks = mesh.allocator()->get_num_banks(BufferType::L1);
    return {distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = uint64_t{kPerCore} * banks},
        distributed::DeviceLocalBufferConfig{.page_size = kPerCore, .buffer_type = BufferType::L1},
        &mesh)};
}

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

double median(std::vector<double> values) {
    std::ranges::nth_element(values, values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2));
    return values[values.size() / 2];
}

void run_once(distributed::MeshCommandQueue& cq, distributed::MeshWorkload& workload) {
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);
}

std::shared_ptr<distributed::MeshDevice> open_system_mesh() {
    return distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()));
}

CoreRange worker_grid(distributed::MeshDevice& mesh) {
    const CoreCoord grid = mesh.compute_with_storage_grid_size();
    return CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
}

struct FlagCore {
    IDevice* device;
    CoreCoord core;
};

void clear_flags(const WorkerL1& worker_l1, const std::vector<FlagCore>& cores) {
    std::vector<uint32_t> zero(WorkerL1::kFlagBytes / sizeof(uint32_t), 0);
    for (const FlagCore& flag_core : cores) {
        detail::WriteToDeviceL1(flag_core.device, flag_core.core, worker_l1.flag(), zero);
    }
}

// Opens the mesh and sleeps for `seconds`, so the sync check measures clocks no workload disturbs.
namespace idle {

int run(double seconds) {
    auto mesh_device = open_system_mesh();
    std::printf("[idle] %zu chips, %.1f s\n", mesh_device->num_devices(), seconds);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    mesh_device->close();
    return 0;
}

}  // namespace idle

// Checks the conversion of device times to host times. For each round the host writes a round number into one worker's
// L1 on every chip and then polls that worker's ack, and the worker records a HOST_RX zone between the two. Converted
// to host time, every zone must start after the host began its write and before it read the ack. That window is the
// host's MMIO round trip, a few microseconds wide, so the check is loose and catches only a conversion that is off by
// more than that. The run also fails unless a callback that unregisters itself on its first call runs exactly once.
namespace host_sync {

constexpr uint32_t kRounds = 2000;
constexpr auto kAckTimeout = std::chrono::seconds(1);
// Rounds 1 ms apart make the capture about 2 s long, enough for the sync check and for the clocks to drift.
constexpr auto kRoundGap = std::chrono::milliseconds(1);

struct Window {
    Clock::time_point before, after;
};

double to_us(Clock::duration duration) { return std::chrono::duration<double, std::micro>(duration).count(); }

int run() {
    std::map<ChipId, std::vector<Clock::time_point>> starts;
    uint64_t dropped_bytes = 0;
    auto registration = streaming_profiler::RegisterCallback(
        [&](const streaming_profiler::Batch<streaming_profiler::Zone>& batch) {
            dropped_bytes += batch.dropped_bytes();
            for (const streaming_profiler::Zone& zone : batch.records<streaming_profiler::Zone>()) {
                if (zone.site().name == "HOST_RX") {
                    starts[zone.core().chip_id].push_back(zone.start_time());
                }
            }
        },
        "host_sync");
    std::atomic<uint32_t> once_calls{0};
    streaming_profiler::Callback once;
    once = streaming_profiler::RegisterCallback(
        [&](const streaming_profiler::Batch<streaming_profiler::Zone>&) {
            if (once_calls++ == 0) {
                once.reset();
            }
        },
        "host_sync-once");

    auto mesh_device = open_system_mesh();
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
    std::map<ChipId, std::vector<Window>> windows;
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
    mesh_device->close();
    registration.reset();

    size_t failed = 0;
    for (const auto& [chip, chip_windows] : windows) {
        const std::vector<Clock::time_point>& zone_starts = starts[chip];
        if (zone_starts.size() != chip_windows.size()) {
            std::printf(
                "[host_sync] chip %d: FAIL, %zu zones for %zu rounds\n", chip, zone_starts.size(), chip_windows.size());
            failed++;
            continue;
        }
        size_t outside = 0;
        for (size_t k = 0; k < chip_windows.size(); k++) {
            const double write_to_zone_us = to_us(zone_starts[k] - chip_windows[k].before);
            const double zone_to_ack_us = to_us(chip_windows[k].after - zone_starts[k]);
            if (write_to_zone_us < 0.0 || zone_to_ack_us < 0.0) {
                if (outside++ == 0) {
                    std::printf(
                        "[host_sync] chip %d round %zu: the zone is %.2f us after the write and "
                        "%.2f us before the ack\n",
                        chip,
                        k + 1,
                        write_to_zone_us,
                        zone_to_ack_us);
                }
            }
        }
        const bool pass = outside == 0;
        failed += pass ? 0 : 1;
        std::printf(
            "[host_sync] chip %d: %s, %zu of %zu zones outside their window\n",
            chip,
            pass ? "PASS" : "FAIL",
            outside,
            chip_windows.size());
    }
    const bool pass = failed == 0 && !timed_out && dropped_bytes == 0 && once_calls == 1;
    std::printf(
        "[host_sync] %s: %zu of %zu chips failed, %llu bytes dropped, a callback that unregisters itself ran %u "
        "times%s\n",
        pass ? "PASS" : "FAIL",
        failed,
        windows.size(),
        static_cast<unsigned long long>(dropped_bytes),
        once_calls.load(),
        timed_out ? ", an ack timed out" : "");
    return pass ? 0 : 1;
}

}  // namespace host_sync

// Checks that each chip's cores agree on the time. On every chip, the first worker multicasts the odd rounds over NoC 0
// and the last worker multicasts the even rounds over NoC 1, and every worker records an MC_RX zone for each round it
// receives. A multicast takes a fixed number of cycles per router hop, so once each arrival is converted to host time
// and its hops are subtracted, every core should give the same time. A run fails if any two cores' mean times differ by
// a cycle or more, a round goes missing, or records are dropped.
namespace multicast {

constexpr uint32_t kRounds = 4000;
static_assert(sync_workload::kRoundValueStrideBytes * (kRounds + 1) <= WorkerL1::kValuesBytes);
static_assert(kRounds % 2 == 0);
// The Blackhole NoC documentation gives 9 cycles per router hop.
constexpr double kCyclesPerHop = 9.0;

struct Arrival {
    int64_t host_ns;
    int64_t cycles;
    ChipId chip;
    uint8_t logical_x, logical_y;
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
    using sync_workload::MulticastRole;
    for (const CoreCoord& core : cores) {
        MulticastRole role = MulticastRole::Receiver;
        if (core == low) {
            role = MulticastRole::Noc0Source;
        } else if (core == high) {
            role = MulticastRole::Noc1Source;
        }
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

// A chip's arrivals by receiving core, each core's in arrival order.
using CoreArrivals = std::map<CoreCoord, std::vector<const Arrival*>>;

// Groups the arrivals by chip and core, and leaves out any chip that is missing a core or has a core with the wrong
// number of arrivals. A source receives only the other source's rounds, so it gets half as many as a receiver.
std::map<ChipId, CoreArrivals> group_arrivals(const std::vector<Arrival>& arrivals, const CoreRange& cores) {
    std::map<ChipId, CoreArrivals> by_chip;
    for (const Arrival& arrival : arrivals) {
        by_chip[arrival.chip][CoreCoord{arrival.logical_x, arrival.logical_y}].push_back(&arrival);
    }
    std::erase_if(by_chip, [&](auto& chip_arrivals) {
        auto& [chip, by_core] = chip_arrivals;
        bool counts_ok = by_core.size() == cores.size();
        for (auto& [core, core_arrivals] : by_core) {
            std::ranges::sort(core_arrivals, {}, &Arrival::cycles);
            const size_t expected = core == cores.start_coord || core == cores.end_coord ? kRounds / 2 : kRounds;
            if (core_arrivals.size() != expected) {
                std::printf(
                    "[multicast] chip %d core (%zu,%zu): %zu arrivals, expected %zu\n",
                    chip,
                    core.x,
                    core.y,
                    core_arrivals.size(),
                    expected);
                counts_ok = false;
            }
        }
        return !counts_ok;
    });
    return by_chip;
}

// Returns, in cycles, the spread of the cores' mean arrival times for `noc`'s multicasts once hops are subtracted. Each
// arrival is measured from the reference core's arrival in the same round, because summed absolute host times would
// lose nanoseconds in a double. Subtracting the same time from every core's arrival in a round shifts every mean
// equally, so the spread is unchanged.
double noc_span_cycles(const CoreArrivals& by_core, const CoreRange& cores, uint32_t noc) {
    const CoreCoord& source = noc == 0 ? cores.start_coord : cores.end_coord;
    const CoreCoord& other_source = noc == 0 ? cores.end_coord : cores.start_coord;
    // A receiver's arrivals alternate the NoC 0 source's odd rounds and the NoC 1 source's even ones.
    const auto arrival = [&](const CoreCoord& core, size_t round) {
        return by_core.at(core)[core == other_source ? round : 2 * round + noc];
    };
    const CoreCoord src(by_core.at(source).front()->physical_x, by_core.at(source).front()->physical_y);
    const CoreCoord& reference =
        by_core.begin()->first == source ? std::next(by_core.begin())->first : by_core.begin()->first;
    constexpr size_t rounds = kRounds / 2;
    std::vector<double> ns_per_cycle(rounds);
    for (size_t k = 0; k < rounds; k++) {
        const Arrival* next = arrival(reference, k + 1 < rounds ? k + 1 : k - 1);
        ns_per_cycle[k] = static_cast<double>(next->host_ns - arrival(reference, k)->host_ns) /
                          static_cast<double>(next->cycles - arrival(reference, k)->cycles);
    }
    double min_mean_ns = std::numeric_limits<double>::max(), max_mean_ns = std::numeric_limits<double>::lowest();
    for (const auto& [core, arrivals] : by_core) {
        if (core == source) {
            continue;
        }
        const CoreCoord physical(arrivals.front()->physical_x, arrivals.front()->physical_y);
        const int hops_x = noc == 0 ? static_cast<int>(physical.x) - static_cast<int>(src.x)
                                    : static_cast<int>(src.x) - static_cast<int>(physical.x);
        const int hops_y = noc == 0 ? static_cast<int>(physical.y) - static_cast<int>(src.y)
                                    : static_cast<int>(src.y) - static_cast<int>(physical.y);
        double sum_ns = 0.0;
        for (size_t k = 0; k < rounds; k++) {
            sum_ns += static_cast<double>(arrival(core, k)->host_ns - arrival(reference, k)->host_ns) -
                      kCyclesPerHop * (hops_x + hops_y) * ns_per_cycle[k];
        }
        const double mean_ns = sum_ns / static_cast<double>(rounds);
        min_mean_ns = std::min(min_mean_ns, mean_ns);
        max_mean_ns = std::max(max_mean_ns, mean_ns);
    }
    return (max_mean_ns - min_mean_ns) /
           (std::reduce(ns_per_cycle.begin(), ns_per_cycle.end()) / static_cast<double>(rounds));
}

void run(distributed::MeshDevice& mesh, uint32_t runtime_id) {
    const CoreRange cores = worker_grid(mesh);
    const WorkerL1 worker_l1 = reserve_worker_l1(mesh);
    std::vector<FlagCore> flag_cores;
    for (IDevice* device : mesh.get_devices()) {
        for (const CoreCoord& core : cores) {
            flag_cores.push_back({device, core});
        }
    }
    clear_flags(worker_l1, flag_cores);
    distributed::MeshWorkload workload = make_multicast(mesh, worker_l1, runtime_id);
    run_once(mesh.mesh_command_queue(), workload);
    std::printf(
        "[multicast] %zux%zu Tensix cores x %u rounds (%u per NoC) on %zu chips\n",
        cores.grid_size().x,
        cores.grid_size().y,
        kRounds,
        kRounds / 2,
        mesh.get_devices().size());
}

// Returns whether every core on all `num_chips` chips received every round, and whether on each chip the cores' mean
// arrival times, once hops are subtracted, are within a cycle of each other.
bool verify(const std::vector<Arrival>& arrivals, const CoreRange& cores, size_t num_chips) {
    const std::map<ChipId, CoreArrivals> by_chip = group_arrivals(arrivals, cores);
    double worst_span_cycles = 0.0;
    for (const auto& [chip, by_core] : by_chip) {
        for (const uint32_t noc : {0u, 1u}) {
            const double span_cycles = noc_span_cycles(by_core, cores, noc);
            worst_span_cycles = std::max(worst_span_cycles, span_cycles);
            std::printf("chip %d NoC %u: span %.2f cycles\n", chip, noc, span_cycles);
        }
    }
    const bool complete = by_chip.size() == num_chips;
    std::printf(
        "[multicast] worst core-to-core span of the device-local timeline %.2f cycles%s\n",
        worst_span_cycles,
        complete ? "" : "; some chips missing or with wrong arrival counts");
    return worst_span_cycles < 1.0 && complete;
}

}  // namespace multicast

// Runs fabric traffic under the profiler. Over the 2D fabric, one worker on each side of every linked chip pair
// ping-pongs atomic increments for kSeconds, recording a PP_TX zone per send and a PP_RX zone per arrival. A run fails
// if a kernel gives up on its peer or a zone doesn't reach the host. It also fails if, in a pair's first kTimedRounds
// rounds, an increment arrives before it was sent or the median reply times differ by more than 2 * kReplySkewBoundNs.
namespace pingpong {

constexpr double kSeconds = 10.0;
// Each pass is kRounds round trips of at least 1.81 us (36 ms), so host work between passes is a small part.
constexpr uint32_t kRounds = 20000;

struct End {
    ChipId chip;
    CoreCoord core;
    uint32_t link;
};
// The two ends of a linked chip pair. Each end's PingpongRole is its index.
using Pair = std::array<End, 2>;

constexpr uint32_t kTimedRounds = 1000;
struct ZoneStart {
    uint64_t cycles;
    Clock::time_point start;
    bool tx;
};
// One core's ping-pong zones. Every zone is counted, and the first kTimedRounds rounds' zones are also kept for the
// timeline check.
struct CoreZones {
    size_t tx = 0, rx = 0;
    std::vector<ZoneStart> timed;
};
using ZonesByCore = std::map<std::pair<ChipId, CoreCoord>, CoreZones>;
struct Chip {
    distributed::MeshCoordinate coord;
    IDevice* device;
    tt::tt_fabric::FabricNodeId node;
};
using Chips = std::map<ChipId, Chip>;

std::vector<Pair> find_pairs(distributed::MeshDevice& mesh_device, const Chips& chips) {
    const CoreCoord grid = mesh_device.compute_with_storage_grid_size();
    std::map<ChipId, uint32_t> next_worker;
    const auto take_worker = [&](ChipId chip) {
        const uint32_t worker = next_worker[chip]++;
        return CoreCoord(worker % grid.x, worker / grid.x);
    };
    std::vector<Pair> pairs;
    for (const auto& [id_a, chip_a] : chips) {
        for (const auto& [id_b, chip_b] : chips) {
            if (id_a >= id_b || !tt::tt_fabric::are_intra_mesh_neighbors(mesh_device, chip_a.node, chip_b.node)) {
                continue;
            }
            const auto links_ab = tt::tt_fabric::get_forwarding_link_indices(chip_a.node, chip_b.node);
            const auto links_ba = tt::tt_fabric::get_forwarding_link_indices(chip_b.node, chip_a.node);
            TT_FATAL(!links_ab.empty() && !links_ba.empty(), "no fabric link between chips {} and {}", id_a, id_b);
            pairs.push_back(
                {End{.chip = id_a, .core = take_worker(id_a), .link = links_ab.front()},
                 End{.chip = id_b, .core = take_worker(id_b), .link = links_ba.front()}});
        }
    }
    return pairs;
}

distributed::MeshWorkload build_workload(
    distributed::MeshDevice& mesh_device,
    const WorkerL1& worker_l1,
    const std::vector<Pair>& pairs,
    const Chips& chips,
    uint32_t runtime_id) {
    std::map<ChipId, Program> programs;
    for (const ChipId chip : std::views::keys(chips)) {
        programs.emplace(chip, CreateProgram()).first->second.set_runtime_id(runtime_id);
    }
    for (const Pair& pair : pairs) {
        for (uint32_t index = 0; index < pair.size(); index++) {
            const End &self = pair[index], &peer = pair[1 - index];
            const tt::tt_fabric::FabricNodeId &src = chips.at(self.chip).node, &dst = chips.at(peer.chip).node;
            Program& program = programs.at(self.chip);
            const auto kernel = CreateKernel(
                program,
                std::string(kKernelDir) + "pingpong_fabric_dm.cpp",
                self.core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            const CoreCoord virtual_peer = mesh_device.worker_core_from_logical_core(peer.core);
            std::vector<uint32_t> args = {
                static_cast<uint32_t>(sync_workload::PingpongRole{index}),
                static_cast<uint32_t>(virtual_peer.x),
                static_cast<uint32_t>(virtual_peer.y),
                worker_l1.flag(),
                kRounds,
                static_cast<uint32_t>(dst.chip_id),
                static_cast<uint32_t>(dst.mesh_id.get())};
            tt::tt_fabric::append_fabric_connection_rt_args(src, dst, self.link, program, self.core, args);
            SetRuntimeArgs(program, kernel, self.core, args);
        }
    }
    distributed::MeshWorkload workload;
    for (auto& [chip, program] : programs) {
        const auto& coord = chips.at(chip).coord;
        workload.add_program(distributed::MeshCoordinateRange(coord, coord), std::move(program));
    }
    return workload;
}

uint32_t count_missing(const std::vector<Pair>& pairs, ZonesByCore& by_core, size_t expected) {
    uint32_t failures = 0;
    for (const Pair& pair : pairs) {
        for (const End& end : pair) {
            const CoreZones& zones = by_core[{end.chip, end.core}];
            if (zones.tx != expected || zones.rx != expected) {
                std::printf(
                    "chip %d core (%zu,%zu): FAIL, %zu PP_TX and %zu PP_RX zones, of %zu each\n",
                    end.chip,
                    end.core.x,
                    end.core.y,
                    zones.tx,
                    zones.rx,
                    expected);
                failures++;
            }
        }
    }
    return failures;
}
// Half the difference between the two directions' median reply times is the pair's clock error plus half the difference
// between their software paths. Under di/dt load it changes by up to 9.5 ns between checks, with or without the chips
// at the same AICLK, while the sync check bounds the clock error at about 4 ns, so most of that change comes from the
// software paths. The bound is above the largest value seen (10.5 ns) and below one refclk tick (20 ns), the smallest
// whole-tick sync error.
constexpr double kReplySkewBoundNs = 15.0;

uint32_t check_timeline(const std::vector<Pair>& pairs, ZonesByCore& by_core) {
    uint32_t failures = 0;
    for (const Pair& pair : pairs) {
        const auto& [a, b] = pair;
        std::vector<ZoneStart>& timed_a = by_core[{a.chip, a.core}].timed;
        std::vector<ZoneStart>& timed_b = by_core[{b.chip, b.core}].timed;
        for (std::vector<ZoneStart>* timed : {&timed_a, &timed_b}) {
            std::ranges::sort(*timed, {}, &ZoneStart::cycles);
        }
        enum Direction : size_t { kAToB, kBToA };
        std::array<std::vector<double>, 2> replies;
        uint32_t acausal = 0, misread = 0;
        for (size_t k = 0; k < std::min(timed_a.size(), timed_b.size()) / 2; k++) {
            // In round k + 1, b sends first if the round is odd and a sends first if it is even.
            const bool b_first = (k & 1u) == 0;
            const ZoneStart* sender = &(b_first ? timed_b : timed_a)[2 * k];
            const ZoneStart* replier = &(b_first ? timed_a : timed_b)[2 * k];
            if (!sender[0].tx || sender[1].tx || replier[0].tx || !replier[1].tx) {
                misread++;
                continue;
            }
            const double ping_ns = std::chrono::duration<double, std::nano>(replier[0].start - sender[0].start).count();
            const double reply_ns =
                std::chrono::duration<double, std::nano>(sender[1].start - replier[1].start).count();
            acausal += (ping_ns <= 0.0 ? 1u : 0u) + (reply_ns <= 0.0 ? 1u : 0u);
            replies[b_first ? kAToB : kBToA].push_back(reply_ns);
        }
        if (replies[kAToB].empty() || replies[kBToA].empty()) {
            std::printf("chip %d - chip %d: FAIL, no timed rounds in some direction\n", a.chip, b.chip);
            failures++;
            continue;
        }
        const double reply_skew_ns = (median(replies[kAToB]) - median(replies[kBToA])) / 2;
        const bool pair_ok = acausal == 0 && misread == 0 && std::abs(reply_skew_ns) <= kReplySkewBoundNs;
        std::printf(
            "chip %d - chip %d: %s, half the reply difference %+.1f ns; %u acausal, %u misread of %zu rounds\n",
            a.chip,
            b.chip,
            pair_ok ? "ok" : "FAIL",
            reply_skew_ns,
            acausal,
            misread,
            replies[kAToB].size() + replies[kBToA].size() + misread);
        failures += pair_ok ? 0 : 1;
    }
    return failures;
}

struct Run {
    std::vector<Pair> pairs;
    uint32_t passes = 0;
};

Run run(distributed::MeshDevice& mesh_device, uint32_t runtime_id) {
    Chips chips;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device.shape())) {
        IDevice* device = mesh_device.get_device(coord);
        chips.emplace(device->id(), Chip{coord, device, mesh_device.get_fabric_node_id(coord)});
    }
    Run result{.pairs = find_pairs(mesh_device, chips)};
    TT_FATAL(!result.pairs.empty(), "no fabric-linked chip pairs");
    std::vector<FlagCore> flag_cores;
    for (const Pair& pair : result.pairs) {
        for (const End& end : pair) {
            flag_cores.push_back({chips.at(end.chip).device, end.core});
        }
    }
    const WorkerL1 worker_l1 = reserve_worker_l1(mesh_device);
    distributed::MeshWorkload workload = build_workload(mesh_device, worker_l1, result.pairs, chips, runtime_id);
    distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
    const auto start = Clock::now();
    do {
        clear_flags(worker_l1, flag_cores);
        run_once(cq, workload);
        result.passes++;
    } while (seconds_since(start) < kSeconds);
    std::printf(
        "[pingpong] %zu linked pairs on %zu chips, %u passes of %u rounds\n",
        result.pairs.size(),
        chips.size(),
        result.passes,
        kRounds);
    return result;
}

void add_zone(CoreZones& zones, const streaming_profiler::Zone& zone, bool tx) {
    (tx ? zones.tx : zones.rx)++;
    if (zones.timed.size() < 2 * kTimedRounds) {
        zones.timed.push_back({zone.start_device_cycles(), zone.start_time(), tx});
    }
}

bool verify(const Run& result, ZonesByCore& by_core) {
    const uint32_t failures =
        count_missing(result.pairs, by_core, size_t{result.passes} * kRounds) + check_timeline(result.pairs, by_core);
    return failures == 0;
}

}  // namespace pingpong

// Runs the FF1 matmul and SDPA di/dt ops on one mesh with the 2D fabric. Their di/dt throttling changes AICLK while
// they run. They use the configs that tests/didt/test_ff1_matmul.py and test_sdpa_op.py run under -k "all and
// without_gelu" and -k "all and bf16_HiFi2". The multicast and ping-pong checks run before, between and after the ops,
// and every run of each must pass. The ops are rebuilt here rather than run from those tests because the checks must
// share the open mesh and the profiler capture with them in one process.
namespace didt {

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
        ShardSpec(
            CoreRangeSet(worker_grid(mesh)),
            {constants::TILE_HEIGHT * kPerCoreM, kShardWidth},
            ShardOrientation::ROW_MAJOR));
    const ttnn::Tensor in0 = ttnn::to_memory_config(
        random_normal(
            mesh,
            ttnn::Shape({1, 1, constants::TILE_HEIGHT * kPerCoreM * grid.y, kShardWidth * grid.x}),
            DataType::BFLOAT16,
            generator),
        in0_config);
    const ttnn::Tensor in1 = random_normal(
        mesh,
        ttnn::Shape({1, 1, kShardWidth * grid.x, constants::TILE_WIDTH * kPerCoreN * grid.x}),
        DataType::BFLOAT8_B,
        generator);
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

// The didt workload's multicast and ping-pong checks of the recorded timeline. Each run() runs both checks once. Every
// record has reached the host once the capture ends, when the mesh closes, so verify() runs after that.
class Checks {
public:
    Checks() :
        callback_(streaming_profiler::RegisterCallback(
            [this](const streaming_profiler::Batch<streaming_profiler::Zone>& batch) {
                dropped_bytes_ += batch.dropped_bytes();
                for (const streaming_profiler::Zone& zone : batch.records<streaming_profiler::Zone>()) {
                    const std::string_view name = zone.site().name;
                    const streaming_profiler::Core core = zone.core();
                    if (name == "MC_RX") {
                        arrivals_[zone.runtime_id()].push_back(multicast::Arrival{
                            .host_ns = zone.start_time().time_since_epoch().count(),
                            .cycles = static_cast<int64_t>(zone.start_device_cycles()),
                            .chip = core.chip_id,
                            .logical_x = static_cast<uint8_t>(core.logical.x),
                            .logical_y = static_cast<uint8_t>(core.logical.y),
                            .physical_x = static_cast<uint8_t>(core.physical.x),
                            .physical_y = static_cast<uint8_t>(core.physical.y)});
                    } else if (name == "PP_TX" || name == "PP_RX") {
                        // A batch holds each core's zones together, so a zone usually belongs to the same core as the
                        // zone before it.
                        const auto key = std::tuple(zone.runtime_id(), core.chip_id, core.logical);
                        if (last_zones_ == nullptr || key != last_key_) {
                            last_key_ = key;
                            last_zones_ = &zones_[zone.runtime_id()][{core.chip_id, core.logical}];
                        }
                        pingpong::add_zone(*last_zones_, zone, name == "PP_TX");
                    }
                }
            },
            "didt")) {}
    Checks(const Checks&) = delete;
    Checks& operator=(const Checks&) = delete;

    void run(distributed::MeshDevice& mesh, std::string_view name) {
        const auto runtime_id = static_cast<uint32_t>(phases_.size() + 1);
        cores_ = worker_grid(mesh);
        num_chips_ = mesh.get_devices().size();
        multicast::run(mesh, runtime_id);
        phases_.push_back({std::string(name), pingpong::run(mesh, runtime_id)});
        std::fflush(stdout);
    }

    bool verify() {
        callback_.reset();
        size_t failed = 0;
        for (size_t k = 0; k < phases_.size(); k++) {
            const Phase& phase = phases_[k];
            std::printf("[didt] checks %s\n", phase.name.c_str());
            const bool multicast_pass = multicast::verify(arrivals_[k + 1], cores_, num_chips_);
            const bool pingpong_pass = pingpong::verify(phase.pingpong, zones_[k + 1]);
            std::printf(
                "[didt] checks %s: multicast %s, ping-pong %s\n",
                phase.name.c_str(),
                multicast_pass ? "PASS" : "FAIL",
                pingpong_pass ? "PASS" : "FAIL");
            failed += multicast_pass && pingpong_pass ? 0 : 1;
        }
        const bool pass = failed == 0 && dropped_bytes_ == 0;
        std::printf(
            "[didt] %s: %zu of %zu phases failed, %llu bytes dropped\n",
            pass ? "PASS" : "FAIL",
            failed,
            phases_.size(),
            static_cast<unsigned long long>(dropped_bytes_));
        return pass;
    }

private:
    struct Phase {
        std::string name;
        pingpong::Run pingpong;
    };
    std::vector<Phase> phases_;
    CoreRange cores_{CoreCoord{0, 0}};
    size_t num_chips_ = 0;
    std::map<uint32_t, std::vector<multicast::Arrival>> arrivals_;
    std::map<uint32_t, pingpong::ZonesByCore> zones_;
    std::tuple<uint32_t, ChipId, CoreCoord> last_key_;
    pingpong::CoreZones* last_zones_ = nullptr;
    uint64_t dropped_bytes_ = 0;
    streaming_profiler::Callback callback_;
};

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

int run() {
    Checks checks;
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
    auto mesh_device = open_system_mesh();
    checks.run(*mesh_device, "before the ops");
    ff1_matmul(*mesh_device);
    std::printf("[didt] FF1 matmul: %u iterations\n", kFf1Iterations);
    checks.run(*mesh_device, "after the FF1 matmul");
    sdpa(*mesh_device);
    std::printf("[didt] SDPA: %u iterations\n", kSdpaIterations);
    checks.run(*mesh_device, "after SDPA");
    mesh_device->close();
    return checks.verify() ? 0 : 1;
}

}  // namespace didt

// Runs a ttnn CCL op along the mesh's longer side for `seconds`, with `load` matmuls before each op.
namespace ccl {

constexpr uint32_t kOpsPerSync = 20;

int run(std::string_view op, std::string_view fabric, uint32_t load, double seconds) {
    const bool ring = fabric == "ring";
    tt::tt_fabric::SetFabricConfig(
        ring ? tt::tt_fabric::FabricConfig::FABRIC_1D_RING : tt::tt_fabric::FabricConfig::FABRIC_2D);
    const tt::tt_fabric::Topology topology = ring ? tt::tt_fabric::Topology::Ring : tt::tt_fabric::Topology::Linear;
    const distributed::MeshShape system = distributed::SystemMesh::instance().shape();
    auto mesh = distributed::MeshDevice::create(distributed::MeshDeviceConfig(
        distributed::MeshShape(std::max(system[0], system[1]), std::min(system[0], system[1]))));
    std::mt19937 generator(1234);
    const ttnn::Tensor input =
        didt::random_normal(*mesh, ttnn::Shape({1, 1, 512, 2048}), DataType::BFLOAT16, generator);
    const ttnn::Tensor matmul_a =
        didt::random_normal(*mesh, ttnn::Shape({1, 1, 2048, 4096}), DataType::BFLOAT16, generator);
    const ttnn::Tensor matmul_b =
        didt::random_normal(*mesh, ttnn::Shape({1, 1, 4096, 4096}), DataType::BFLOAT16, generator);
    uint64_t ops = 0;
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < end) {
        for (uint32_t i = 0; i < kOpsPerSync; i++) {
            for (uint32_t m = 0; m < load; m++) {
                ttnn::matmul(matmul_a, matmul_b);
            }
            if (op == "all_gather") {
                ttnn::all_gather(
                    input,
                    /*dim=*/3,
                    /*cluster_axis=*/0,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    topology);
            } else {
                ttnn::all_reduce(input, /*cluster_axis=*/0, std::nullopt, std::nullopt, std::nullopt, topology);
            }
        }
        distributed::Synchronize(*mesh, std::nullopt);
        ops += kOpsPerSync;
    }
    std::printf(
        "[ccl] %.*s over %.*s fabric, load %u: %llu ops in %.0f s\n",
        static_cast<int>(op.size()),
        op.data(),
        static_cast<int>(fabric.size()),
        fabric.data(),
        load,
        static_cast<unsigned long long>(ops),
        seconds);
    mesh->close();
    return 0;
}

}  // namespace ccl

}  // namespace

int main(int argc, char** argv) {
    const std::string_view workload = argc > 1 ? argv[1] : "";
    if (workload == "idle" && argc == 4 && std::string_view(argv[2]) == "--seconds") {
        return idle::run(std::strtod(argv[3], nullptr));
    }
    if (workload == "host_sync" && argc == 2) {
        return host_sync::run();
    }
    if (workload == "didt" && argc == 2) {
        return didt::run();
    }
    if (workload == "ccl" && argc == 6) {
        return ccl::run(argv[2], argv[3], std::strtoul(argv[4], nullptr, 10), std::strtod(argv[5], nullptr));
    }
    std::fprintf(
        stderr,
        "usage: %s idle --seconds S | host_sync | didt | ccl all_gather|all_reduce ring|2d MATMULS SECONDS\n",
        argv[0]);
    return 2;
}
