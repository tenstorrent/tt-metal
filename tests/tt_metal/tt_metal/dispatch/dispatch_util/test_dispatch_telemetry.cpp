// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

#include <tt-metalium/experimental/dispatch_telemetry.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "command_queue_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/command_queue_common.hpp"
#include "impl/dispatch/dispatch_telemetry.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"

namespace tt::tt_metal {
namespace {

bool worker_reached_l1_wait(
    IDevice* device,
    const CoreCoord& core,
    uint32_t addr,
    uint32_t expected,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    std::vector<uint32_t> readback(1);

    do {
        detail::ReadFromDeviceL1(device, core, addr, sizeof(uint32_t), readback);
        if (readback[0] == expected) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);

    return false;
}

Program create_blank_program(const CoreCoord& core) {
    Program program = CreateProgram();
    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    return program;
}

}  // namespace

class DispatchTelemetryReadApiTest : public UnitMeshCQFixture {
protected:
    IDevice* device() const { return devices_.at(0)->get_devices().front(); }

    void write_telemetry(const CoreCoord& core, const DispatchCoreTelemetry& telemetry) {
        auto telemetry_addr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
            CommandQueueDeviceAddrType::DISPATCH_TELEMETRY);
        auto bytes =
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&telemetry), sizeof(DispatchCoreTelemetry));
        ASSERT_TRUE(detail::WriteToDeviceL1(device(), core, telemetry_addr, bytes, CoreType::WORKER));
    }

    void write_telemetry(const CoreCoord& core, const PrefetchCoreTelemetry& telemetry) {
        auto telemetry_addr = MetalContext::instance().dispatch_mem_map().get_device_command_queue_addr(
            CommandQueueDeviceAddrType::DISPATCH_TELEMETRY);
        auto bytes =
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&telemetry), sizeof(PrefetchCoreTelemetry));
        ASSERT_TRUE(detail::WriteToDeviceL1(device(), core, telemetry_addr, bytes, CoreType::WORKER));
    }
};

class DispatchTelemetryHostL1WaitTest : public DispatchTelemetryReadApiTest {
protected:
    void SetUp() override {
        DispatchTelemetryReadApiTest::SetUp();
        if (IsSkipped()) {
            return;
        }

        release_addr_ = devices_.at(0)->allocator()->get_base_allocator_addr(HalMemType::L1);
        started_addr_ = release_addr_ + sizeof(uint32_t);
        std::vector<uint32_t> zero_word{0};

        ASSERT_TRUE(detail::WriteToDeviceL1(device(), worker_core_, release_addr_, zero_word));
        ASSERT_TRUE(detail::WriteToDeviceL1(device(), worker_core_, started_addr_, zero_word));

        Program waiting_program = CreateProgram();
        auto waiting_kernel = CreateKernel(
            waiting_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/wait_for_host_l1_write.cpp",
            worker_core_,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(
            waiting_program,
            waiting_kernel,
            worker_core_,
            {release_addr_, release_value_, started_addr_, started_value_});
        waiting_workload_.add_program(device_range_, std::move(waiting_program));
    }

    void release_worker_and_finish() {
        std::vector<uint32_t> release_word{release_value_};
        EXPECT_TRUE(detail::WriteToDeviceL1(device(), worker_core_, release_addr_, release_word));
        Finish(devices_.at(0)->mesh_command_queue());
    }

    static constexpr uint32_t release_value_ = 0x67216721;
    static constexpr uint32_t started_value_ = 0x5A5A5A5A;

    uint32_t release_addr_ = 0;
    uint32_t started_addr_ = 0;
    CoreCoord worker_core_{0, 0};
    distributed::MeshWorkload waiting_workload_;
};

TEST_F(DispatchTelemetryReadApiTest, ReadDispatchTelemetryFromL1) {
    const CoreCoord core{0, 0};
    DispatchCoreTelemetry telemetry;
    telemetry.upstream_blocked_count = 17;
    telemetry.upstream_unblocked_count = 19;
    telemetry.program_count = 21;
    write_telemetry(core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_dispatch_core_telemetry(device()->id(), virtual_core);

    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual->upstream_blocked_count, telemetry.upstream_blocked_count);
    EXPECT_EQ(actual->upstream_unblocked_count, telemetry.upstream_unblocked_count);
    EXPECT_EQ(actual->program_count, telemetry.program_count);
}

TEST_F(DispatchTelemetryReadApiTest, ReadDispatchTelemetryRejectsBadSignature) {
    const CoreCoord core{0, 0};
    DispatchCoreTelemetry telemetry;
    telemetry.signature = INVALID_TELEMETRY_SIGNATURE;
    write_telemetry(core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_dispatch_core_telemetry(device()->id(), virtual_core);

    EXPECT_FALSE(actual.has_value());
}

TEST_F(DispatchTelemetryReadApiTest, ReadDispatchTelemetryRejectsBadVersion) {
    const CoreCoord core{0, 0};
    DispatchCoreTelemetry telemetry;
    telemetry.version = DISPATCH_TELEMETRY_VERSION + 1;
    write_telemetry(core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_dispatch_core_telemetry(device()->id(), virtual_core);

    EXPECT_FALSE(actual.has_value());
}

TEST_F(DispatchTelemetryReadApiTest, ReadPrefetchTelemetryFromL1) {
    const CoreCoord core{0, 0};
    PrefetchCoreTelemetry telemetry;
    telemetry.upstream_blocked_count = 23;
    telemetry.upstream_unblocked_count = 29;
    telemetry.command_count = 31;
    write_telemetry(core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_prefetch_core_telemetry(device()->id(), virtual_core);

    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual->upstream_blocked_count, telemetry.upstream_blocked_count);
    EXPECT_EQ(actual->upstream_unblocked_count, telemetry.upstream_unblocked_count);
    EXPECT_EQ(actual->command_count, telemetry.command_count);
}

TEST_F(DispatchTelemetryReadApiTest, ReadPrefetchTelemetryRejectsBadSignature) {
    const CoreCoord core{0, 0};
    PrefetchCoreTelemetry telemetry;
    telemetry.signature = INVALID_TELEMETRY_SIGNATURE;
    write_telemetry(core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_prefetch_core_telemetry(device()->id(), virtual_core);

    EXPECT_FALSE(actual.has_value());
}

TEST_F(DispatchTelemetryReadApiTest, ReadPrefetchTelemetryRejectsBadVersion) {
    const CoreCoord core{0, 0};
    PrefetchCoreTelemetry telemetry;
    telemetry.version = DISPATCH_TELEMETRY_VERSION + 1;
    write_telemetry(core, telemetry);

    const CoreCoord virtual_core = device()->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto actual = read_prefetch_core_telemetry(device()->id(), virtual_core);

    EXPECT_FALSE(actual.has_value());
}

TEST_F(DispatchTelemetryReadApiTest, DispatchProgramCountIncrementsAfterProgramRuns) {
    IDevice* device = this->device();
    auto& cq = devices_.at(0)->mesh_command_queue();
    const CoreCoord worker_core{0, 0};
    constexpr size_t total_runs = 10;
    constexpr size_t num_blank_programs = 4;

    DispatchTelemetry telemetry(*device);
    ASSERT_FALSE(telemetry.read_info().empty());

    distributed::MeshWorkload workload;
    workload.add_program(device_range_, create_blank_program(worker_core));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    auto after_one = telemetry.read_info();
    ASSERT_FALSE(after_one.empty());
    EXPECT_EQ(after_one.front().dispatch_program_count_since_last_read, 1);

    // Multiple runs to ensure the delta is calculated correctly
    for (size_t run = 0; run < total_runs; ++run) {
        for (size_t i = 0; i < num_blank_programs; ++i) {
            distributed::MeshWorkload workload;
            workload.add_program(device_range_, create_blank_program(worker_core));
            distributed::EnqueueMeshWorkload(cq, workload, false);
        }
        Finish(cq);

        auto after_many = telemetry.read_info();
        ASSERT_FALSE(after_many.empty());
        EXPECT_EQ(after_many.front().dispatch_program_count_since_last_read, num_blank_programs);
    }
}

TEST_F(DispatchTelemetryReadApiTest, PrefetchCommandCountAdvancesForEveryEnqueuedProgram) {
    IDevice* device = this->device();
    auto& cq = devices_.at(0)->mesh_command_queue();
    const CoreCoord worker_core{0, 0};
    constexpr size_t num_blank_programs = 4;

    DispatchTelemetry telemetry(*device);
    ASSERT_FALSE(telemetry.read_info().empty());

    for (size_t i = 0; i < num_blank_programs; ++i) {
        distributed::MeshWorkload workload;
        workload.add_program(device_range_, create_blank_program(worker_core));
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    Finish(cq);

    auto after = telemetry.read_info();
    ASSERT_FALSE(after.empty());
    EXPECT_GE(after.front().prefetch_command_count_since_last_read, num_blank_programs);
}

TEST_F(DispatchTelemetryHostL1WaitTest, WorkerWaitReportsUpstreamBlockedState) {
    IDevice* device = this->device();
    auto& cq = devices_.at(0)->mesh_command_queue();

    DispatchTelemetry telemetry(*device);
    ASSERT_FALSE(telemetry.read_info().empty());

    // Enqueue kernel that waits on host write to complete
    distributed::EnqueueMeshWorkload(cq, waiting_workload_, false);
    EXPECT_TRUE(worker_reached_l1_wait(device, worker_core_, started_addr_, started_value_));

    // Worker has the host blocked kernel in progress
    // Dispatch queue is empty
    // Prefetch queue is empty
    auto while_waiting = telemetry.read_info();
    EXPECT_FALSE(while_waiting.empty());
    if (!while_waiting.empty()) {
        // Prefetch is waiting on upstream host for the next workload
        EXPECT_TRUE(while_waiting.front().prefetch_waiting_on_upstream);
        // Dispatch is waiting on upstream prefetch for the next workload
        EXPECT_TRUE(while_waiting.front().dispatch_waiting_on_upstream);
    }

    constexpr size_t num_blank_programs = 4;
    for (size_t i = 0; i < num_blank_programs; ++i) {
        distributed::MeshWorkload workload;
        workload.add_program(device_range_, create_blank_program(worker_core_));
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }

    // Worker still has the host blocked kernel in progress
    // Prefetch is stalled waiting on downstream sync
    // Dispatch is waiting for worker progress
    auto after_enqueue = telemetry.read_info();
    EXPECT_FALSE(after_enqueue.empty());
    if (!after_enqueue.empty()) {
        // Prefetch is no longer waiting on upstream host
        EXPECT_FALSE(after_enqueue.front().prefetch_waiting_on_upstream);
        // Dispatch is no longer waiting on upstream prefetch
        EXPECT_FALSE(after_enqueue.front().dispatch_waiting_on_upstream);
    }

    release_worker_and_finish();

    // Worker is no longer blocked, it has finished processing all enqueued work loads
    // Dispatch queue is empty
    // Prefetch queue is empty
    auto after_finish = telemetry.read_info();
    EXPECT_FALSE(after_finish.empty());
    if (!after_finish.empty()) {
        // Prefetch is waiting on upstream host for the next workload
        EXPECT_TRUE(after_finish.front().prefetch_waiting_on_upstream);
        // Dispatch is waiting on upstream prefetch for the next workload
        EXPECT_TRUE(after_finish.front().dispatch_waiting_on_upstream);
    }
}

}  // namespace tt::tt_metal
