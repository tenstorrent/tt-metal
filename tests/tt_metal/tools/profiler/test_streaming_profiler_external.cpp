// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Runs a kernel that publishes a Reading to each destination while an external process, this binary run with --reader,
// receives the records sent to it through a callback of its own. The profiled process must see every reading, and the
// external process every "both" reading and no "metal" one.
// Both sum the high words of the "both" readings' device times, and take their earliest and latest host times, which
// must agree, since the external process places records on the profiled process's clock map. Under fast dispatch, the
// profiled process must also see zones from the dispatch kernels.

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <fmt/format.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "external_types.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace external_test;
namespace sp = tt::tt_metal::experimental::streaming_profiler;

namespace {

constexpr uint32_t kIterations = 2000;
constexpr uint32_t kGridSide = 4;
constexpr std::pair<DataMovementProcessor, NOC> kDmRiscs[] = {
    {DataMovementProcessor::RISCV_0, NOC::RISCV_0_default}, {DataMovementProcessor::RISCV_1, NOC::RISCV_1_default}};
constexpr uint64_t kPerDestination = uint64_t{kIterations} * kGridSide * kGridSide * std::size(kDmRiscs);
constexpr std::string_view kNames[] = {"metal", "both"};
// How far apart the two processes may put a "both" reading on the host's steady clock: each process maps the TSC onto
// it from samples of its own.
constexpr int64_t kSteadyAgreementNs = 1000;

// The file the external process creates once it listens, which the profiled process waits for before it opens the
// device; the one the profiled process creates once it has closed the device; and where the external process leaves
// what it saw of the "both" readings.
std::string ready_path(const std::string& path) { return path + ".ready"; }
std::string closed_path(const std::string& path) { return path + ".closed"; }
std::string both_path(const std::string& path) { return path + ".both"; }

bool wait_for_file(const std::string& path, std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!std::ifstream(path)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

// What a process saw of the "both" readings: the sum of their device times' high words, and their earliest and latest
// host times.
struct BothSeen {
    uint64_t hi = 0;
    int64_t first_ns = std::numeric_limits<int64_t>::max();
    int64_t last_ns = std::numeric_limits<int64_t>::min();

    void add(const sp::Timestamp& time) {
        hi += time.device_cycles() >> 32;
        const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time.time().time_since_epoch()).count();
        first_ns = std::min(first_ns, ns);
        last_ns = std::max(last_ns, ns);
    }
};

// Counts the readings, by destination, that are what their call site sent, and notes the "both" ones.
struct Readings {
    std::mutex mutex;
    std::array<uint64_t, 2> counts{};
    uint64_t wrong = 0;
    BothSeen both;

    void add(const sp::Record<Reading, Name>& record) {
        const Reading& reading = record.payload();
        if (reading.destination > 1 || record.metadata().name != kNames[reading.destination] ||
            reading.value != reading_value(reading.destination, reading.iteration)) {
            wrong++;
            return;
        }
        counts[reading.destination]++;
        if (reading.destination == 1) {
            both.add(record.published_at());
        }
    }
};

// The external process: having no MetalContext, it listens once it registers a callback. With `drain`, it only
// receives, until it is killed.
int run_reader(const std::string& path, bool drain) {
    Readings readings;
    auto callback = sp::RegisterCallback(
        [&](const sp::Batch<sp::Record<Reading, Name>>& batch) {
            std::lock_guard lock(readings.mutex);
            for (const auto& record : batch.records<sp::Record<Reading, Name>>()) {
                readings.add(record);
            }
        },
        "external-reader");
    std::ofstream(ready_path(path)) << 1;
    if (drain) {
        while (true) {
            std::this_thread::sleep_for(std::chrono::hours(1));
        }
    }
    // The profiled process closes its device, which stops the relays' copies, before this process stops receiving.
    const bool closed = wait_for_file(closed_path(path), std::chrono::seconds(300));
    std::remove(closed_path(path).c_str());
    const auto complete = [&] {
        std::lock_guard lock(readings.mutex);
        return readings.counts[1] >= kPerDestination;
    };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (closed && !complete() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    callback.reset();
    std::ofstream(both_path(path)) << readings.both.hi << ' ' << readings.both.first_ns << ' ' << readings.both.last_ns;
    fmt::print(
        "[external] external process: {} both, {} metal, {} wrong; want {} both{}\n",
        readings.counts[1],
        readings.counts[0],
        readings.wrong,
        kPerDestination,
        readings.counts[1] == 0 ? " (another external process running would have taken them)" : "");
    return readings.counts[0] == 0 && readings.counts[1] == kPerDestination && readings.wrong == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string_view(argv[1]) == "--reader") {
        return run_reader(argv[2], argc == 4 && std::string_view(argv[3]) == "--drain");
    }
    const std::string path = fmt::format("/tmp/tt-external-test-{}", ::getpid());
    const pid_t reader = ::fork();
    if (reader == 0) {
        ::execl("/proc/self/exe", argv[0], "--reader", path.c_str(), nullptr);
        ::_exit(127);
    }
    // Once the external process listens, opening the device connects to it and waits for it to attach.
    wait_for_file(ready_path(path), std::chrono::seconds(120));
    std::remove(ready_path(path).c_str());

    auto mesh_device =
        distributed::MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1);
    // Registered once the device is open, so this process has a MetalContext and doesn't listen as an external process.
    Readings readings;
    uint64_t dispatch_zones = 0;
    auto callback = sp::RegisterCallback(
        [&](const sp::Batch<sp::Record<Reading, Name>, sp::Zone>& batch) {
            std::lock_guard lock(readings.mutex);
            for (const sp::Zone& zone : batch.records<sp::Zone>()) {
                dispatch_zones += zone.location().file.find("impl/dispatch/kernels/") != std::string_view::npos;
            }
            for (const auto& record : batch.records<sp::Record<Reading, Name>>()) {
                readings.add(record);
            }
        },
        "external-test");
    Program program = CreateProgram();
    const CoreRange cores(CoreCoord{0, 0}, CoreCoord{kGridSide - 1, kGridSide - 1});
    for (const auto& [processor, noc] : kDmRiscs) {
        CreateKernel(
            program,
            "tests/tt_metal/tools/profiler/kernels/external_dm.cpp",
            cores,
            DataMovementConfig{
                .processor = processor, .noc = noc, .defines = {{"N_ITERS", std::to_string(kIterations) + "u"}}});
    }
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);
    mesh_device->close();
    callback.reset();
    std::ofstream(closed_path(path)) << 1;
    int status = 0;
    ::waitpid(reader, &status, 0);

    const bool reader_ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    BothSeen theirs{.hi = ~uint64_t{0}};
    std::ifstream(both_path(path)) >> theirs.hi >> theirs.first_ns >> theirs.last_ns;
    std::remove(both_path(path).c_str());
    const BothSeen& ours = readings.both;
    const int64_t first_apart = theirs.first_ns - ours.first_ns, last_apart = theirs.last_ns - ours.last_ns;
    fmt::print(
        "[external] \"both\" readings: time high words {} here, {} in the external process; there, the first is {} ns "
        "and the last {} ns from here on the steady clock\n",
        ours.hi,
        theirs.hi,
        first_apart,
        last_apart);
    const bool slow_dispatch = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;
    fmt::print("[external] {} zones from the dispatch kernels\n", dispatch_zones);
    const bool ok = readings.counts[0] == kPerDestination && readings.counts[1] == kPerDestination &&
                    readings.wrong == 0 && ours.hi == theirs.hi && std::abs(first_apart) <= kSteadyAgreementNs &&
                    std::abs(last_apart) <= kSteadyAgreementNs && (slow_dispatch || dispatch_zones != 0);
    fmt::print(
        "[external] profiled process: {} metal, {} both, {} wrong; want {} each\n",
        readings.counts[0],
        readings.counts[1],
        readings.wrong,
        kPerDestination);
    fmt::print("[external] {}\n", ok && reader_ok ? "PASS" : "FAIL");
    return ok && reader_ok ? 0 : 1;
}
