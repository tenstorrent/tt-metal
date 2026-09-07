// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device.hpp>

namespace tt::tt_metal {
namespace {

using namespace distributed;
using namespace experimental;

class RealtimeProfilerSubdevices : public ::testing::TestWithParam<bool> {};

// Runtime IDs are shared by all workers of a program. An early worker in the
// slow subdevice must not complete its record, nor may the other subdevice's
// completion replace its end timestamp.
TEST_P(RealtimeProfilerSubdevices, LastWorkerCompletionPreservesOverlap) {
    constexpr std::array<uint32_t, 2> requested_cycles = {50'000'000, 5'000'000};
    constexpr uint32_t repetitions = 5;
    auto device =
        MeshDevice::create_unit_mesh(0, 32768, 8 * 1024 * 1024, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    if (!IsProgramRealtimeProfilerActive()) {
        device->close();
        GTEST_SKIP() << "Real-time profiler unavailable on this dispatch configuration";
    }
    const std::array<CoreRangeSet, 2> cores = {
        CoreRangeSet(CoreRange({0, 0}, {1, 0})), CoreRangeSet(CoreRange({2, 0}, {2, 0}))};
    auto manager =
        device->create_sub_device_manager({SubDevice(std::array{cores[0]}), SubDevice(std::array{cores[1]})}, 0);
    device->load_sub_device_manager(manager);
    std::array<MeshWorkload, 2> workloads;
    for (uint32_t subdevice = 0; subdevice < 2; ++subdevice) {
        auto program = CreateProgram();
        const uint32_t num_cores = subdevice == 0 ? 2 : 1;
        for (uint32_t core = 0; core < num_cores; ++core) {
            const uint32_t delay = subdevice == 0 && core == 0 ? requested_cycles[1] : requested_cycles[subdevice];
            const CoreCoord logical{subdevice == 0 ? core : 2, 0};
            const std::string source =
                "#include <cstdint>\n#include \"api/dataflow/dataflow_api.h\"\n"
                "void kernel_main() {\n"
                "volatile tt_reg_ptr uint32_t* clock = reinterpret_cast<volatile tt_reg_ptr "
                "uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);\n"
                "uint32_t start = *clock;\n"
                "while (static_cast<uint32_t>(*clock - start) < " +
                std::to_string(delay) + "u) {}\n}\n";
            CreateKernelFromString(
                program,
                source,
                logical,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        }
        program.set_runtime_id(101 + subdevice);
        workloads[subdevice].add_program(MeshCoordinateRange(device->shape()), std::move(program));
    }
    auto& cq = device->mesh_command_queue();
    for (auto& workload : workloads) {
        EnqueueMeshWorkload(cq, workload, true);
    }
    std::vector<MeshTraceId> traces;
    if (GetParam()) {
        for (auto& workload : workloads) {
            auto id = device->begin_mesh_trace(cq);
            EnqueueMeshWorkload(cq, workload, false);
            device->end_mesh_trace(cq, id);
            traces.push_back(id);
        }
    }
    Finish(cq);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::mutex mutex;
    std::array<std::vector<ProgramRealtimeRecord>, 2> records;
    uint64_t dropped = 0;
    uint64_t unexpected = 0;
    auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        std::lock_guard lock(mutex);
        dropped += batch.dropped;
        for (const auto& record : batch.records) {
            if (record.runtime_id == 101 || record.runtime_id == 102) {
                records[record.runtime_id - 101].push_back(record);
            } else {
                ++unexpected;
            }
        }
    });
    // Always unregister before the callback's captured storage is destroyed,
    // including when a runtime exception unwinds the test.
    struct CallbackGuard {
        ProgramRealtimeProfilerCallbackHandle handle;
        ~CallbackGuard() { UnregisterProgramRealtimeProfilerCallback(handle); }
    };
    std::vector<std::pair<double, double>> completion_ms;
    {
        CallbackGuard guard{handle};
        for (uint32_t repeat = 0; repeat < repetitions; ++repeat) {
            auto start = std::chrono::steady_clock::now();
            if (GetParam()) {
                for (auto id : traces) {
                    device->replay_mesh_trace(cq, id, false);
                }
            } else {
                for (auto& workload : workloads) {
                    EnqueueMeshWorkload(cq, workload, false);
                }
            }
            const std::array fast_group{SubDeviceId{1}};
            device->set_sub_device_stall_group(fast_group);
            Finish(cq);
            const double fast_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            device->reset_sub_device_stall_group();
            Finish(cq);
            const double all_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            completion_ms.emplace_back(fast_ms, all_ms);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    for (auto id : traces) {
        device->release_mesh_trace(id);
    }
    device->close();

    ASSERT_EQ(dropped, 0u);
    ASSERT_EQ(unexpected, 0u);
    ASSERT_EQ(records[0].size(), repetitions);
    ASSERT_EQ(records[1].size(), repetitions);
    for (uint32_t repeat = 0; repeat < repetitions; ++repeat) {
        const auto& slow = records[0][repeat];
        const auto& fast = records[1][repeat];
        EXPECT_EQ(slow.chip_id, fast.chip_id);
        EXPECT_LT(slow.start_timestamp, fast.start_timestamp);
        EXPECT_LT(fast.end_timestamp, slow.end_timestamp);
        for (uint32_t subdevice = 0; subdevice < 2; ++subdevice) {
            const auto& record = records[subdevice][repeat];
            ASSERT_GE(record.end_timestamp, record.start_timestamp);
            const uint64_t duration = record.end_timestamp - record.start_timestamp;
            EXPECT_GE(duration, requested_cycles[subdevice]);
            // A generous device-cycle bound includes GO delivery, firmware,
            // and monitor polling, while rejecting another worker's completion.
            EXPECT_LT(duration, requested_cycles[subdevice] + 100'000u);
        }
        EXPECT_LT(completion_ms[repeat].first, completion_ms[repeat].second * 0.75)
            << "Finishing the fast subdevice waited for the slow subdevice";
    }
}

INSTANTIATE_TEST_SUITE_P(
    OrdinaryAndSegmentedTrace,
    RealtimeProfilerSubdevices,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "SegmentedTrace" : "Ordinary"; });

}  // namespace
}  // namespace tt::tt_metal
