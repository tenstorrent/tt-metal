// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Callbacks registered and unregistered between captures each receive exactly the records published while they are
// registered, and two callbacks on one record type each receive all of them. A callback may unregister itself but not
// register another, and one that falls behind is told how many bytes it missed. A Timestamp maps onto the steady clock
// only once a capture has synced. Run with TT_METAL_STREAMING_PROFILER=1. Exits nonzero on any mismatch.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <fmt/format.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/device_observability.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "records_types.hpp"

using namespace tt::tt_metal;
using namespace records_test;
namespace api = tt::tt_metal::experimental::device_observability;

namespace {

using Count = api::Record<uint32_t, Name>;
using BigRecord = api::Record<Big, Name>;

constexpr uint32_t kRecords = 1000;
// 4 MiB from each RISC, through a 1 MiB FIFO.
constexpr uint32_t kBig = 8192;
constexpr std::pair<DataMovementProcessor, NOC> kDmRiscs[] = {
    {DataMovementProcessor::RISCV_0, NOC::RISCV_0_default}, {DataMovementProcessor::RISCV_1, NOC::RISCV_1_default}};
constexpr uint64_t kCountsPerCapture = uint64_t{kRecords} * std::size(kDmRiscs);

bool failed = false;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        fmt::print("[callbacks] FAIL {}\n", what);
        failed = true;
    }
}

bool throws(const std::function<void()>& f, std::string_view message) {
    try {
        f();
    } catch (const std::exception& e) {
        return std::string_view(e.what()).find(message) != std::string_view::npos;
    }
    return false;
}

// Runs the kernel on one core's data-movement RISCs, publishing `records` counts and then `bigs` large records, and
// waits for it.
void run(distributed::MeshDevice& mesh_device, uint32_t records, uint32_t bigs) {
    Program program = CreateProgram();
    const std::map<std::string, std::string> defines{
        {"N_RECORDS", std::to_string(records) + "u"}, {"N_BIG", std::to_string(bigs) + "u"}};
    for (const auto& [processor, noc] : kDmRiscs) {
        CreateKernel(
            program,
            "tests/tt_metal/tools/profiler/kernels/callbacks_dm.cpp",
            CoreCoord{0, 0},
            DataMovementConfig{.processor = processor, .noc = noc, .defines = defines});
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/false);
    distributed::Finish(mesh_device.mesh_command_queue());
}

// One capture: opens the device, runs the counts and then the large records, and closes the device, which delivers
// every record before it returns. `between` runs between the two runs and `finished` after both, with the device open.
void capture(
    uint32_t records,
    uint32_t bigs,
    const std::function<void()>& between = [] {},
    const std::function<void()>& finished = [] {}) {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    run(*mesh_device, records, 0);
    between();
    if (bigs != 0) {
        run(*mesh_device, 0, bigs);
    }
    finished();
    mesh_device->close();
}

}  // namespace

int main() {
    // Small enough for the last capture to overrun while its callback waits.
    setenv("TT_METAL_DEVICE_OBSERVABILITY_FIFO_MB", "1", 1);

    expect(
        throws(
            [] { static_cast<void>(api::Timestamp{}.time()); },
            "a TSC tick maps to steady_clock only after a capture's first host sync burst"),
        "Timestamp::time() before any capture throws");

    // Two callbacks on one type each get every record; one unregistered after the first capture gets none of the
    // second's, and one registered after it gets only the second's.
    std::atomic<uint64_t> first{0}, second{0}, late{0};
    {
        api::Callback first_callback = api::RegisterCallback(
            [&](const api::Batch<Count>& batch) { first += batch.records<Count>().size(); }, "first");
        api::Callback second_callback = api::RegisterCallback(
            [&](const api::Batch<Count>& batch) { second += batch.records<Count>().size(); }, "second");
        capture(kRecords, 0);
        first_callback.reset();
        api::Callback late_callback = api::RegisterCallback(
            [&](const api::Batch<Count>& batch) { late += batch.records<Count>().size(); }, "late");
        capture(kRecords, 0);
    }
    expect(
        first == kCountsPerCapture,
        fmt::format("first callback got {} records, want {}", first.load(), kCountsPerCapture));
    expect(
        second == 2 * kCountsPerCapture,
        fmt::format("second callback got {} records, want {}", second.load(), 2 * kCountsPerCapture));
    expect(
        late == kCountsPerCapture,
        fmt::format("late callback got {} records, want {}", late.load(), kCountsPerCapture));

    // A callback that unregisters itself gets no more batches, here or in the next capture; one that registers another
    // callback gets the rule's error.
    std::atomic<uint64_t> once_batches{0};
    std::string register_error;
    api::Callback once;
    once = api::RegisterCallback(
        [&](const api::Batch<Count>&) {
            ++once_batches;
            once.reset();
        },
        "once");
    api::Callback registers = api::RegisterCallback(
        [&](const api::Batch<Count>&) {
            if (register_error.empty()) {
                try {
                    api::Callback inner = api::RegisterCallback([](const api::Batch<Count>&) {}, "inner");
                    register_error = "none";
                } catch (const std::exception& e) {
                    register_error = e.what();
                }
            }
        },
        "registers");
    capture(kRecords, 0);
    registers.reset();
    expect(
        register_error.find("add_consumer must not be called from a consumer callback") != std::string::npos,
        "a callback that registers a callback gets the rule's error, got: " + register_error);

    // A callback that blocks in its first batch while the kernel sends many FIFOs' worth misses records, and its next
    // batch says how many bytes.
    std::mutex mutex;
    std::condition_variable cv;
    bool blocked = false, finished = false;
    std::atomic<uint64_t> dropped{0}, bigs{0};
    api::Callback slow = api::RegisterCallback(
        [&](const api::Batch<Count, BigRecord>& batch) {
            {
                std::unique_lock lock(mutex);
                if (!blocked) {
                    blocked = true;
                    cv.notify_all();
                    cv.wait(lock, [&] { return finished; });
                }
            }
            dropped += batch.dropped_bytes();
            bigs += batch.records<BigRecord>().size();
        },
        "slow");
    capture(
        kRecords,
        kBig,
        [&] {
            // The large records go out only once the callback is blocked.
            std::unique_lock lock(mutex);
            expect(
                cv.wait_for(lock, std::chrono::seconds(60), [&] { return blocked; }),
                "a callback gets the counts while the device is open");
        },
        [&] {
            {
                const std::lock_guard lock(mutex);
                finished = true;
            }
            cv.notify_all();
        });
    slow.reset();
    expect(
        once_batches == 1,
        fmt::format("a callback that unregistered itself got {} batches, want 1", once_batches.load()));
    expect(dropped > 0, "a callback that falls behind is told the bytes it missed");
    expect(
        bigs < uint64_t{kBig} * std::size(kDmRiscs),
        fmt::format("a callback that falls behind got all {} large records", bigs.load()));

    fmt::print("[callbacks] {}\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
