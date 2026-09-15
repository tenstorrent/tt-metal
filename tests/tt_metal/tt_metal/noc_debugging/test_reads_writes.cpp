// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdlib>
#include <thread>
#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <impl/context/metal_context.hpp>

#include "hal_types.hpp"
#include "noc_debugging_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr int BRISC_PROCESSOR_ID = 0;
constexpr int NCRISC_PROCESSOR_ID = 1;

using IssueChecker = std::function<bool(ChipId, CoreCoord, int)>;

void VerifyIssuesOnAllCores(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    CoreCoord grid_start,
    CoreCoord grid_end,
    bool expect_issue,
    const IssueChecker& has_issue,
    const std::string& issue_type) {
    auto* device = mesh_device->get_devices()[0];
    auto device_id = device->id();

    for (uint32_t x = grid_start.x; x <= grid_end.x; ++x) {
        for (uint32_t y = grid_start.y; y <= grid_end.y; ++y) {
            CoreCoord logical_core = {x, y};
            auto virtual_core = mesh_device->worker_core_from_logical_core(logical_core);

            bool brisc_issue = has_issue(device_id, virtual_core, BRISC_PROCESSOR_ID);
            bool ncrisc_issue = has_issue(device_id, virtual_core, NCRISC_PROCESSOR_ID);

            const auto make_error_string = [&issue_type, &device_id](
                                               const std::string& msg, const CoreCoord& virtual_core) {
                return fmt::format(
                    "NOC debugger should have detected {} {} issue at device {} core {}",
                    msg,
                    issue_type,
                    device_id,
                    virtual_core.str());
            };

            if (expect_issue) {
                EXPECT_TRUE(brisc_issue) << make_error_string("brisc", virtual_core);
                EXPECT_TRUE(ncrisc_issue) << make_error_string("ncrisc", virtual_core);
            } else {
                EXPECT_FALSE(brisc_issue) << make_error_string("NO brisc issue", virtual_core);
                EXPECT_FALSE(ncrisc_issue) << make_error_string("NO ncrisc issue", virtual_core);
            }
        }
    }
}

void RunWritesTest(
    NOCDebuggingFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    bool use_barrier,
    bool use_trid = false,
    bool use_trid_barrier = false) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();

    CoreCoord grid_start = {0, 0};
    CoreCoord grid_end = {compute_grid_size.x - 1, compute_grid_size.y - 1};
    CoreRange core_range(grid_start, grid_end);

    auto dest_core_virtual = mesh_device->worker_core_from_logical_core(grid_end);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr uint32_t buffer_page_size = 4096;
    constexpr uint32_t buffer_size = buffer_page_size * 4;

    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"L1_BUFFER_ADDR", std::to_string(l1_buffer->address())},
        {"OTHER_CORE_X", std::to_string(dest_core_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(dest_core_virtual.y)},
        {"DST_ADDR", std::to_string(l1_buffer->address())},
        {"NUM_ITERATIONS", "10"},
    };

    if (use_barrier) {
        defines["USE_WRITE_BARRIER"] = "1";
    }
    if (use_trid) {
        defines["USE_TRID"] = "1";
    }
    if (use_trid_barrier) {
        defines["USE_TRID_BARRIER"] = "1";
    }

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_writes.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_writes.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload);

    ReadMeshDeviceProfilerResults(*mesh_device);

    VerifyIssuesOnAllCores(
        mesh_device,
        grid_start,
        grid_end,
        /*expect_issue=*/!(use_barrier || use_trid_barrier),
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_write_barrier_issue(chip_id, core, processor_id);
        },
        "write barrier");
}

// Every core issues repeated posted writes from the same source to one destination core. Posted writes are drained
// by a posted-writes flush (noc_async_posted_writes_flushed), not a regular write barrier; without an in-loop flush
// the source-reuse hazard is reported. Exercises the posted-flush device emission + the WRITE_FLUSH posted mapping.
void RunPostedWriteTest(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool use_flush) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();

    CoreCoord grid_start = {0, 0};
    CoreCoord grid_end = {compute_grid_size.x - 1, compute_grid_size.y - 1};
    CoreRange core_range(grid_start, grid_end);

    auto dest_core_virtual = mesh_device->worker_core_from_logical_core(grid_end);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr uint32_t buffer_page_size = 4096;
    constexpr uint32_t buffer_size = buffer_page_size * 4;

    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"L1_BUFFER_ADDR", std::to_string(l1_buffer->address())},
        {"OTHER_CORE_X", std::to_string(dest_core_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(dest_core_virtual.y)},
        {"DST_ADDR", std::to_string(l1_buffer->address())},
        {"NUM_ITERATIONS", "10"},
    };

    if (use_flush) {
        defines["USE_POSTED_FLUSH"] = "1";
    }

    for (auto processor : {tt_metal::DataMovementProcessor::RISCV_0, tt_metal::DataMovementProcessor::RISCV_1}) {
        auto noc = processor == tt_metal::DataMovementProcessor::RISCV_0 ? tt_metal::NOC::RISCV_0_default
                                                                         : tt_metal::NOC::RISCV_1_default;
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_posted_writes.cpp",
            core_range,
            tt_metal::DataMovementConfig{.processor = processor, .noc = noc, .defines = defines});
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload);

    ReadMeshDeviceProfilerResults(*mesh_device);

    VerifyIssuesOnAllCores(
        mesh_device,
        grid_start,
        grid_end,
        /*expect_issue=*/!use_flush,
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_write_barrier_issue(chip_id, core, processor_id);
        },
        "posted write flush");
}

// Every core issues repeated non-posted remote atomic increments (noc_semaphore_inc) to one destination core.
// Atomics are released only by an atomic/full barrier (they use a NIU counter separate from writes), so without a
// barrier they remain outstanding at kernel end and the tool reports an unflushed-semaphore issue; an atomic
// barrier drains them. Exercises the Stage-3c SEMAPHORE_INC + ATOMIC_BARRIER host mapping and atomics tracking.
void RunSemaphoreIncTest(
    NOCDebuggingFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    bool use_barrier,
    bool use_full_barrier = false,
    bool use_wrong_write_barrier = false) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();

    CoreCoord grid_start = {0, 0};
    CoreCoord grid_end = {compute_grid_size.x - 1, compute_grid_size.y - 1};
    CoreRange core_range(grid_start, grid_end);

    auto dest_core_virtual = mesh_device->worker_core_from_logical_core(grid_end);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr uint32_t buffer_page_size = 4096;
    constexpr uint32_t buffer_size = buffer_page_size * 4;

    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"OTHER_CORE_X", std::to_string(dest_core_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(dest_core_virtual.y)},
        {"DST_ADDR", std::to_string(l1_buffer->address())},
        {"NUM_ITERATIONS", "10"},
    };

    if (use_barrier) {
        defines["USE_ATOMIC_BARRIER"] = "1";
    } else if (use_full_barrier) {
        defines["USE_FULL_BARRIER"] = "1";
    } else if (use_wrong_write_barrier) {
        defines["USE_WRITE_BARRIER"] = "1";
    }

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_semaphore_inc.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_semaphore_inc.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload);

    ReadMeshDeviceProfilerResults(*mesh_device);

    VerifyIssuesOnAllCores(
        mesh_device,
        grid_start,
        grid_end,
        /*expect_issue=*/!(use_barrier || use_full_barrier),
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_unflushed_semaphore_issue(chip_id, core, processor_id);
        },
        "unflushed semaphore inc");
}

// A single sender issues repeated multicast atomic increments (noc_semaphore_inc_multicast) to a rectangle of
// cores (excluding itself, since the atomic-inc multicast sender cannot be a destination). Without an atomic/full
// barrier they remain outstanding at kernel end -> unflushed (multicast) semaphore issue. Exercises the
// SEMAPHORE_INC_MULTICAST host mapping + the multicast device record path.
void RunSemaphoreIncMulticastTest(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool use_barrier) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();
    if (compute_grid_size.x < 2) {
        // need at least one column besides the sender's for a sender-excluding multicast rectangle
        GTEST_SKIP() << "Multicast semaphore-inc test requires a compute grid at least 2 columns wide";
    }

    CoreCoord sender_core = {0, 0};
    CoreRange sender_range(sender_core, sender_core);

    // Multicast rectangle starts at column 1 so the sender (0,0) is not one of the destinations.
    CoreCoord mcast_start = {1, 0};
    CoreCoord mcast_end = {compute_grid_size.x - 1, compute_grid_size.y - 1};
    auto mcast_start_virtual = mesh_device->worker_core_from_logical_core(mcast_start);
    auto mcast_end_virtual = mesh_device->worker_core_from_logical_core(mcast_end);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    uint32_t num_dest_cores = CoreRange(mcast_start, mcast_end).size();

    constexpr uint32_t buffer_page_size = 64;
    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_page_size};
    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"USE_MULTICAST", "1"},
        {"MCAST_START_X", std::to_string(mcast_start_virtual.x)},
        {"MCAST_START_Y", std::to_string(mcast_start_virtual.y)},
        {"MCAST_END_X", std::to_string(mcast_end_virtual.x)},
        {"MCAST_END_Y", std::to_string(mcast_end_virtual.y)},
        {"NUM_DEST_CORES", std::to_string(num_dest_cores)},
        {"DST_ADDR", std::to_string(l1_buffer->address())},
        {"NUM_ITERATIONS", "10"},
    };

    if (use_barrier) {
        defines["USE_ATOMIC_BARRIER"] = "1";
    }

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_semaphore_inc.cpp",
        sender_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload);

    ReadMeshDeviceProfilerResults(*mesh_device);

    auto* device = mesh_device->get_devices()[0];
    auto device_id = device->id();
    auto sender_core_virtual = mesh_device->worker_core_from_logical_core(sender_core);

    bool has_issue = fixture->has_unflushed_semaphore_mcast_issue(device_id, sender_core_virtual, BRISC_PROCESSOR_ID);
    if (use_barrier) {
        EXPECT_FALSE(has_issue) << "With atomic barrier, should NOT have unflushed multicast semaphore issue at device "
                                << device_id << " core " << sender_core_virtual.str();
    } else {
        EXPECT_TRUE(has_issue) << "Without atomic barrier, should have unflushed multicast semaphore issue at device "
                               << device_id << " core " << sender_core_virtual.str();
    }
}

// Every core issues repeated inline dword writes (4-byte immediate value, no L1 source buffer) to one destination
// core. Because there is no source buffer, the same-src write-barrier check must NOT fire (that would be a false
// positive); inline writes are released by a normal write barrier, so without one they are reported as unflushed at
// kernel end. Exercises the Stage-3d WRITE_INLINE device emission + host mapping + has_source_buffer handling.
void RunInlineWriteTest(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool use_barrier) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();

    CoreCoord grid_start = {0, 0};
    CoreCoord grid_end = {compute_grid_size.x - 1, compute_grid_size.y - 1};
    CoreRange core_range(grid_start, grid_end);

    auto dest_core_virtual = mesh_device->worker_core_from_logical_core(grid_end);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr uint32_t buffer_page_size = 4096;
    constexpr uint32_t buffer_size = buffer_page_size * 4;

    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"OTHER_CORE_X", std::to_string(dest_core_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(dest_core_virtual.y)},
        {"DST_ADDR", std::to_string(l1_buffer->address())},
        {"NUM_ITERATIONS", "10"},
    };

    if (use_barrier) {
        defines["USE_WRITE_BARRIER"] = "1";
    }

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_inline_writes.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_inline_writes.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload);

    ReadMeshDeviceProfilerResults(*mesh_device);

    // Primary: an inline write left unflushed at kernel end is reported iff there is no write barrier.
    VerifyIssuesOnAllCores(
        mesh_device,
        grid_start,
        grid_end,
        /*expect_issue=*/!use_barrier,
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_unflushed_write_issue(chip_id, core, processor_id);
        },
        "unflushed inline write");

    // Guard: the same-src write-barrier check must never fire for inline writes (they have no source buffer, so
    // repeated inline writes to the same destination are not a source-reuse hazard).
    VerifyIssuesOnAllCores(
        mesh_device,
        grid_start,
        grid_end,
        /*expect_issue=*/false,
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_write_barrier_issue(chip_id, core, processor_id);
        },
        "inline write same-src false positive");
}

// Every core issues repeated stateful writes (noc_async_write_one_packet_with_state) from the same source address
// to one destination core. Same-source-without-barrier must still be detected. Exercises the Stage-3e
// WRITE_WITH_STATE host mapping + counter whitelist.
void RunStatefulWriteTest(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool use_barrier) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();

    CoreCoord grid_start = {0, 0};
    CoreCoord grid_end = {compute_grid_size.x - 1, compute_grid_size.y - 1};
    CoreRange core_range(grid_start, grid_end);

    auto dest_core_virtual = mesh_device->worker_core_from_logical_core(grid_end);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr uint32_t buffer_page_size = 4096;
    constexpr uint32_t buffer_size = buffer_page_size * 4;

    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"L1_BUFFER_ADDR", std::to_string(l1_buffer->address())},
        {"OTHER_CORE_X", std::to_string(dest_core_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(dest_core_virtual.y)},
        {"DST_ADDR", std::to_string(l1_buffer->address())},
        {"NUM_ITERATIONS", "10"},
    };

    if (use_barrier) {
        defines["USE_WRITE_BARRIER"] = "1";
    }

    for (auto processor : {tt_metal::DataMovementProcessor::RISCV_0, tt_metal::DataMovementProcessor::RISCV_1}) {
        auto noc = processor == tt_metal::DataMovementProcessor::RISCV_0 ? tt_metal::NOC::RISCV_0_default
                                                                         : tt_metal::NOC::RISCV_1_default;
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_stateful_writes.cpp",
            core_range,
            tt_metal::DataMovementConfig{.processor = processor, .noc = noc, .defines = defines});
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload);

    ReadMeshDeviceProfilerResults(*mesh_device);

    VerifyIssuesOnAllCores(
        mesh_device,
        grid_start,
        grid_end,
        /*expect_issue=*/!use_barrier,
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_write_barrier_issue(chip_id, core, processor_id);
        },
        "stateful write barrier");
}

// Every core issues repeated stateful reads (noc_async_read_one_packet_with_state) that land at the same local
// address. Same-destination-without-barrier must still be detected. Exercises the Stage-3e READ_WITH_STATE host
// mapping + counter whitelist.
void RunStatefulReadTest(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool use_barrier) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();

    CoreCoord grid_start = {0, 0};
    CoreCoord grid_end = {compute_grid_size.x - 1, compute_grid_size.y - 1};
    CoreRange core_range(grid_start, grid_end);

    auto dest_core_virtual = mesh_device->worker_core_from_logical_core(grid_end);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr uint32_t buffer_page_size = 4096;
    constexpr uint32_t buffer_size = buffer_page_size * 4;

    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"L1_BUFFER_ADDR", std::to_string(l1_buffer->address())},
        {"OTHER_CORE_X", std::to_string(dest_core_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(dest_core_virtual.y)},
        {"SRC_ADDR", std::to_string(l1_buffer->address())},
        {"NUM_ITERATIONS", "10"},
    };

    if (use_barrier) {
        defines["USE_READ_BARRIER"] = "1";
    }

    for (auto processor : {tt_metal::DataMovementProcessor::RISCV_0, tt_metal::DataMovementProcessor::RISCV_1}) {
        auto noc = processor == tt_metal::DataMovementProcessor::RISCV_0 ? tt_metal::NOC::RISCV_0_default
                                                                         : tt_metal::NOC::RISCV_1_default;
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_stateful_reads.cpp",
            core_range,
            tt_metal::DataMovementConfig{.processor = processor, .noc = noc, .defines = defines});
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload);

    ReadMeshDeviceProfilerResults(*mesh_device);

    VerifyIssuesOnAllCores(
        mesh_device,
        grid_start,
        grid_end,
        /*expect_issue=*/!use_barrier,
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_read_barrier_issue(chip_id, core, processor_id);
        },
        "stateful read barrier");
}

// Single core + single RISC issues writes and then a full barrier. Exercises the FULL_BARRIER host
// mapping via the end-of-kernel unflushed-write check: with the mapping the full barrier clears the
// pending writes so nothing is reported; without it the full barrier is ignored and the writes are
// falsely reported as unflushed at kernel end.
void RunFullBarrierWritesSingleCore(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();
    if (compute_grid_size.x < 2) {
        GTEST_SKIP() << "Single-core full-barrier write test requires a compute grid at least 2 columns wide";
    }

    const CoreCoord writer_core = {0, 0};
    const CoreCoord dest_core = {1, 0};
    auto writer_virtual = mesh_device->worker_core_from_logical_core(writer_core);
    auto dest_virtual = mesh_device->worker_core_from_logical_core(dest_core);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr uint32_t buffer_page_size = 4096;
    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_page_size};
    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"L1_BUFFER_ADDR", std::to_string(l1_buffer->address())},
        {"OTHER_CORE_X", std::to_string(dest_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(dest_virtual.y)},
        {"DST_ADDR", std::to_string(l1_buffer->address())},
        {"NUM_ITERATIONS", "10"},
        {"USE_FULL_BARRIER", "1"},
    };

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_writes.cpp",
        CoreRange(writer_core),
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);
    ReadMeshDeviceProfilerResults(*mesh_device);

    for (IDevice* device : mesh_device->get_devices()) {
        EXPECT_FALSE(fixture->has_unflushed_write_issue(device->id(), writer_virtual, 0))
            << "A full barrier must flush pending writes by kernel end; the FULL_BARRIER host mapping was not "
               "applied so the writes were falsely reported as unflushed.";
    }
}

// Single core + single RISC issues writes and then only an atomic barrier. An atomic barrier waits on a NIU
// counter separate from writes, so it must NOT drain the pending writes: they must remain reported as unflushed at
// kernel end. This guards the barrier/counter separation (write vs atomic) from a future regression that would let
// an atomic barrier clear writes. The mirror case (a write barrier must not drain atomics) is covered by
// SemaphoreIncWriteBarrierDoesNotFlush.
void RunAtomicBarrierWritesSingleCore(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();
    if (compute_grid_size.x < 2) {
        GTEST_SKIP() << "Single-core atomic-barrier write test requires a compute grid at least 2 columns wide";
    }

    const CoreCoord writer_core = {0, 0};
    const CoreCoord dest_core = {1, 0};
    auto writer_virtual = mesh_device->worker_core_from_logical_core(writer_core);
    auto dest_virtual = mesh_device->worker_core_from_logical_core(dest_core);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr uint32_t buffer_page_size = 4096;
    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_page_size};
    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"L1_BUFFER_ADDR", std::to_string(l1_buffer->address())},
        {"OTHER_CORE_X", std::to_string(dest_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(dest_virtual.y)},
        {"DST_ADDR", std::to_string(l1_buffer->address())},
        {"NUM_ITERATIONS", "10"},
        {"USE_ATOMIC_BARRIER", "1"},
    };

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_writes.cpp",
        CoreRange(writer_core),
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);
    ReadMeshDeviceProfilerResults(*mesh_device);

    for (IDevice* device : mesh_device->get_devices()) {
        EXPECT_TRUE(fixture->has_unflushed_write_issue(device->id(), writer_virtual, 0))
            << "An atomic barrier must not flush pending writes (writes use a NIU counter separate from atomics); "
               "the writes should still be reported as unflushed at kernel end.";
    }
}

void RunReadsTest(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool use_barrier) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();

    CoreCoord grid_start = {0, 0};
    CoreCoord grid_end = {compute_grid_size.x - 1, compute_grid_size.y - 1};
    CoreRange core_range(grid_start, grid_end);

    auto src_core_virtual = mesh_device->worker_core_from_logical_core(grid_end);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr uint32_t buffer_page_size = 4096;
    constexpr uint32_t buffer_size = buffer_page_size * 4;

    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"L1_BUFFER_ADDR", std::to_string(l1_buffer->address())},
        {"OTHER_CORE_X", std::to_string(src_core_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(src_core_virtual.y)},
        {"NUM_ITERATIONS", "10"},
    };

    if (use_barrier) {
        defines["USE_READ_BARRIER"] = "1";
    }

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_reads.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_reads.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload);

    ReadMeshDeviceProfilerResults(*mesh_device);

    VerifyIssuesOnAllCores(
        mesh_device,
        grid_start,
        grid_end,
        /*expect_issue=*/!use_barrier,
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_read_barrier_issue(chip_id, core, processor_id);
        },
        "read barrier");
}

void RunInterleavedReadsWritesTest(
    NOCDebuggingFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    bool use_read_barrier,
    bool use_write_barrier) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();

    CoreCoord grid_start = {0, 0};
    CoreCoord grid_end = {compute_grid_size.x - 1, compute_grid_size.y - 1};
    CoreRange core_range(grid_start, grid_end);

    auto other_core_virtual = mesh_device->worker_core_from_logical_core(grid_end);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr uint32_t buffer_page_size = 4096;
    constexpr uint32_t buffer_size = buffer_page_size * 4;

    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    auto src_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
    auto dst_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"SRC_ADDR", std::to_string(src_buffer->address())},
        {"OTHER_CORE_X", std::to_string(other_core_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(other_core_virtual.y)},
        {"DST_ADDR", std::to_string(dst_buffer->address())},
    };

    if (use_read_barrier) {
        defines["USE_READ_BARRIER"] = "1";
    }
    if (use_write_barrier) {
        defines["USE_WRITE_BARRIER"] = "1";
    }

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/interleaved_async_reads_writes.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/interleaved_async_reads_writes.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload);

    ReadMeshDeviceProfilerResults(*mesh_device);

    VerifyIssuesOnAllCores(
        mesh_device,
        grid_start,
        grid_end,
        /*expect_issue=*/!use_write_barrier,
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_write_barrier_issue(chip_id, core, processor_id);
        },
        "write barrier");

    VerifyIssuesOnAllCores(
        mesh_device,
        grid_start,
        grid_end,
        /*expect_issue=*/!use_read_barrier,
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_read_barrier_issue(chip_id, core, processor_id);
        },
        "read barrier");
}

TEST_F(NOCDebuggingFixture, WritesNoBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunWritesTest(fixture, mesh_device, false);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, WritesWithBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunWritesTest(fixture, mesh_device, true);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, WritesWithFullBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunFullBarrierWritesSingleCore(fixture, mesh_device);
            },
            mesh_device);
    }
}

// An atomic barrier must not drain pending writes (writes and atomics use separate NIU counters): the writes must
// still be reported as unflushed at kernel end.
TEST_F(NOCDebuggingFixture, AtomicBarrierDoesNotFlushWrites) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunAtomicBarrierWritesSingleCore(fixture, mesh_device);
            },
            mesh_device);
    }
}

// Transaction-id writes are modeled as ordinary address-keyed writes: the same-src-without-barrier issue must
// still be detected (no barrier), and a regular write barrier must still clear them (with barrier).
TEST_F(NOCDebuggingFixture, TridWritesNoBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunWritesTest(fixture, mesh_device, /*use_barrier=*/false, /*use_trid=*/true);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, TridWritesWithBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunWritesTest(fixture, mesh_device, /*use_barrier=*/true, /*use_trid=*/true);
            },
            mesh_device);
    }
}

// The dedicated trid write barrier (noc_async_write_barrier_with_trid) must clear the trid writes it waits on,
// so no same-src issue is reported. Exercises the Stage-3b device emission + host mapping of WRITE_BARRIER_WITH_TRID.
TEST_F(NOCDebuggingFixture, TridWritesWithTridBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunWritesTest(
                    fixture, mesh_device, /*use_barrier=*/false, /*use_trid=*/true, /*use_trid_barrier=*/true);
            },
            mesh_device);
    }
}

// Posted writes reusing the same source without a posted flush must still be flagged; a posted flush clears them.
TEST_F(NOCDebuggingFixture, PostedWritesNoFlush) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunPostedWriteTest(fixture, mesh_device, /*use_flush=*/false);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, PostedWritesWithFlush) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunPostedWriteTest(fixture, mesh_device, /*use_flush=*/true);
            },
            mesh_device);
    }
}

// Sums the sizes of the profiler's per-core/per-risc Tracy marker set for a device (this is the map that
// device_markers.emplace() at profiler.cpp:2068 grows, and that the NOC-debug dedup relies on).
size_t total_device_marker_count(ChipId device_id) {
    auto& psm = tt::tt_metal::MetalContext::instance().profiler_state_manager();
    std::lock_guard<std::recursive_mutex> lock{psm->device_profiler_map_mutex};
    auto it = psm->device_profiler_map.find(device_id);
    if (it == psm->device_profiler_map.end()) {
        return 0;
    }
    size_t count = 0;
    for (const auto& [core, risc_map] : it->second.device_markers_per_core_risc_map) {
        for (const auto& [risc, markers] : risc_map) {
            count += markers.size();
        }
    }
    return count;
}

// Runs the stress kernel once from a single core (num_iterations wrapped-source non-posted writes, no barrier). By
// default it then drains via the profiler read (where process_accumulated_events + finish_cores run). Set
// read_after=false to leave the results undrained so a test can observe what the background periodic poll accumulated
// while the kernel ran, before any host force-read. wait_iters>0 adds an on-device idle after every burst_size writes
// (models a long-running kernel that spends time not emitting events).
void run_stress_write_program(
    NOCDebuggingFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_iterations,
    bool read_after = true,
    uint32_t wait_iters = 0,
    uint32_t burst_size = 0) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();
    auto dest_core_virtual =
        mesh_device->worker_core_from_logical_core(CoreCoord{compute_grid_size.x - 1, compute_grid_size.y - 1});
    CoreRange sender_range(CoreCoord{0, 0}, CoreCoord{0, 0});

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();
    // Fixed-size buffer regardless of num_iterations; the kernel wraps its source within SRC_SLOTS slots so a huge
    // iteration count (e.g. 100000) stays in-bounds while still generating that many distinct profiler events.
    constexpr uint32_t kBufferBytes = 4096;
    constexpr uint32_t kSlotBytes = 32;
    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = kBufferBytes, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = kBufferBytes};
    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
    std::map<std::string, std::string> defines = {
        {"SRC_BASE_ADDR", std::to_string(l1_buffer->address())},
        {"SRC_SLOTS", std::to_string(kBufferBytes / kSlotBytes)},
        {"OTHER_CORE_X", std::to_string(dest_core_virtual.x)},
        {"OTHER_CORE_Y", std::to_string(dest_core_virtual.y)},
        {"DST_ADDR", std::to_string(l1_buffer->address())},
        {"NUM_ITERATIONS", std::to_string(num_iterations)},
    };
    if (wait_iters > 0) {
        defines["WAIT_ITERS"] = std::to_string(wait_iters);
        defines["BURST_SIZE"] = std::to_string(burst_size > 0 ? burst_size : num_iterations);
    }
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_stress_writes.cpp",
        sender_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});
    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);
    if (read_after) {
        ReadMeshDeviceProfilerResults(*mesh_device);
    }
}

// Iteration count for the stress test below, env-configurable via NOC_DEBUG_STRESS_ITERATIONS (default 20, kept
// small for CI). Set it high (e.g. 100000) to stress the profiler event/marker path.
uint32_t marker_test_iterations() {
    if (const char* env = std::getenv("NOC_DEBUG_STRESS_ITERATIONS"); env != nullptr) {
        if (const uint32_t parsed = static_cast<uint32_t>(std::strtoul(env, nullptr, 10)); parsed > 0) {
            return parsed;
        }
    }
    return 20;
}

// Retunes the background debug-dump thread for the lifetime of this object, then restores the previous settings.
//
// The thread reads the three knobs once and captures them BY VALUE when it starts (see
// ProfilerStateManager::start_debug_dump_thread), and it is started at device open. So setting rtoptions from a test
// body has no effect on the running thread -- the knobs only take hold across a stop/relaunch, which is what this
// does. Restoring in the destructor means the rest of the suite keeps the defaults even if the test body aborts
// early on a failed assertion.
class ScopedDebugDumpTuning {
public:
    ScopedDebugDumpTuning(
        std::vector<tt::tt_metal::IDevice*> devices,
        std::chrono::milliseconds poll,
        std::chrono::milliseconds full_read,
        std::chrono::milliseconds margin) :
        devices_{std::move(devices)},
        prev_poll_{rtoptions().get_noc_debug_poll_interval()},
        prev_full_read_{rtoptions().get_noc_debug_full_read_interval()},
        prev_margin_{rtoptions().get_noc_debug_watermark_margin()} {
        restart(poll, full_read, margin);
    }

    ~ScopedDebugDumpTuning() { restart(prev_poll_, prev_full_read_, prev_margin_); }

    ScopedDebugDumpTuning(const ScopedDebugDumpTuning&) = delete;
    ScopedDebugDumpTuning& operator=(const ScopedDebugDumpTuning&) = delete;

private:
    static tt::llrt::RunTimeOptions& rtoptions() { return tt::tt_metal::MetalContext::instance().rtoptions(); }

    void restart(
        std::chrono::milliseconds poll, std::chrono::milliseconds full_read, std::chrono::milliseconds margin) {
        auto& psm = tt::tt_metal::MetalContext::instance().profiler_state_manager();
        if (psm->debug_dump_thread.joinable()) {
            psm->stop_debug_dump_thread = true;
            psm->stop_debug_dump_thread_cv.notify_all();
            psm->debug_dump_thread.join();
        }
        rtoptions().set_noc_debug_poll_interval(poll);
        rtoptions().set_noc_debug_full_read_interval(full_read);
        rtoptions().set_noc_debug_watermark_margin(margin);
        // start_debug_dump_thread() clears the stop flag, so the thread is live again after this.
        // Must cover EVERY device the thread originally covered (profiler_initializer.cpp launches it with all of
        // them): a device left out stops being drained, and later tests running on it see no events at all.
        tt::tt_metal::LaunchIntervalBasedProfilerReadThread(devices_);
    }

    std::vector<tt::tt_metal::IDevice*> devices_;
    std::chrono::milliseconds prev_poll_;
    std::chrono::milliseconds prev_full_read_;
    std::chrono::milliseconds prev_margin_;
};

// Exposes the core "days-long workload" problem. During a long kernel the background debug-dump poll READS events off
// the device but does not PROCESS or REPORT them -- detection, reporting and discharge only happen on a user read. So
// before any user read the pending-event queue and the marker set grow with total events and NO issue has been
// detected yet, even though the kernel already committed violations (here: same source reused with no barrier). The
// fix makes the poll process + report + discharge as events arrive, so before any user read: issues are already
// detected and host memory stays bounded regardless of run length.
//
// Large-N stress test: the device profiler buffer must overflow so the periodic poll actually runs mid-kernel.
// Skipped at the small CI default; drive with NOC_DEBUG_STRESS_ITERATIONS>=100000.
TEST_F(NOCDebuggingFixture, IncrementalProcessingDuringLongKernel) {
    if (marker_test_iterations() < 100000) {
        GTEST_SKIP() << "set NOC_DEBUG_STRESS_ITERATIONS>=100000 to run this stress test";
    }
    this->RunTestOnDevice<NOCDebuggingFixture>(
        [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state();
            ASSERT_TRUE(noc_debug_state != nullptr);
            const auto device_id = mesh_device->get_devices()[0]->id();
            const uint32_t writes = marker_test_iterations();
            const uint32_t burst = writes / 4;
            constexpr uint32_t wait_iters = 50'000'000u;

            // One long kernel, NO user read: only the background periodic poll drains the device.
            run_stress_write_program(fixture, mesh_device, writes, /*read_after=*/false, wait_iters, burst);
            // Let the poll catch up on anything still stalled after the kernel finished.
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));

            const auto summary = noc_debug_state->get_state_summary();
            const size_t markers = total_device_marker_count(device_id);
            log_info(
                tt::LogMetal,
                "[incremental] before any user read: pending_events={}, issues={}, markers={} (writes={})",
                summary.pending_events,
                summary.issues,
                markers,
                writes);

            // With incremental processing the poll detects issues as they happen (not deferred to a read)...
            EXPECT_GT(summary.issues, 0u) << "no issue detected before a user read -- processing is deferred to reads";
            // ...and discharges events + markers so host memory stays bounded regardless of run length.
            EXPECT_LT(summary.pending_events, writes / 2) << "pending events accumulated unprocessed";
            EXPECT_LT(markers, writes / 2) << "marker set accumulated undischarged";
        },
        this->devices_[0]);
}

// Same behaviour as the stress test above, but cheap enough to run in CI on every commit, so the background
// processing path (self-triggered full read -> watermark -> incremental report -> marker discharge) actually has
// regression protection.
//
// It is a short kernel rather than a huge one because the event COUNT was never what the stress test needed. What
// matters is the event time SPAN: process_accumulated_events_up_to() holds back everything within margin_ticks of the
// newest event it has seen, so events only become processable once the span exceeds the margin. At the 3000 ms
// default no short kernel can ever qualify. Shrinking the margin to 60 ms (and the full-read period to 20 ms, so
// several passes land while the kernel is still running) gets the same coverage from a sub-second kernel.
TEST_F(NOCDebuggingFixture, IncrementalProcessingFastCycle) {
    // Collected here (where the fixture's device list is in scope) because the retuned thread has to be relaunched
    // covering every device, not just the one this test runs on -- see ScopedDebugDumpTuning.
    std::vector<tt::tt_metal::IDevice*> all_devices;
    for (const auto& md : this->devices_) {
        for (auto* d : md->get_devices()) {
            all_devices.push_back(d);
        }
    }
    this->RunTestOnDevice<NOCDebuggingFixture>(
        [all_devices](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state();
            ASSERT_TRUE(noc_debug_state != nullptr);
            const auto device_id = mesh_device->get_devices()[0]->id();

            // The margin must exceed the poll interval (start_debug_dump_thread enforces that) and must be well
            // under the kernel's event span, or nothing ever falls behind the watermark. That threshold is what this
            // test actually pins: raising the margin above the kernel's span flips the results to
            // pending_events=401/issues=0 (nothing processed) instead of 121/1, so the assertions below fail if
            // mid-run processing regresses to being deferred to a user read.
            ScopedDebugDumpTuning tuning{
                all_devices,
                /*poll=*/std::chrono::milliseconds(10),
                /*full_read=*/std::chrono::milliseconds(10),
                /*margin=*/std::chrono::milliseconds(30)};
            // Relaunching the thread above drains the device once, which can push leftovers from earlier tests.
            noc_debug_state->reset_state();

            // 400 writes in 10 bursts, each burst followed by an on-device idle, giving a sub-second kernel. The
            // source slot wraps after SRC_SLOTS(=128) writes, so the unbarriered source reuse -- the violation this
            // asserts on -- happens around burst 4, early enough to fall behind the watermark before the kernel ends.
            constexpr uint32_t writes = 400;
            constexpr uint32_t burst = 40;
            constexpr uint32_t wait_iters = 8'000'000u;

            // NO user read: only the background thread drains, processes, reports and discharges.
            run_stress_write_program(fixture, mesh_device, writes, /*read_after=*/false, wait_iters, burst);
            // Let the last full-read pass land after the kernel finished.
            std::this_thread::sleep_for(std::chrono::milliseconds(300));

            const auto summary = noc_debug_state->get_state_summary();
            const size_t markers = total_device_marker_count(device_id);
            log_info(
                tt::LogMetal,
                "[incremental-fast] before any user read: pending_events={}, issues={}, markers={} (writes={})",
                summary.pending_events,
                summary.issues,
                markers,
                writes);

            EXPECT_GT(summary.issues, 0u) << "no issue detected before a user read -- processing is deferred to reads";
            EXPECT_LT(summary.pending_events, writes / 2) << "pending events accumulated unprocessed";
            EXPECT_LT(markers, writes / 2) << "marker set accumulated undischarged";
        },
        this->devices_[0]);
}

// Non-posted semaphore increments with no barrier stay outstanding at kernel end -> unflushed-semaphore issue.
TEST_F(NOCDebuggingFixture, SemaphoreIncNoBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunSemaphoreIncTest(fixture, mesh_device, /*use_barrier=*/false);
            },
            mesh_device);
    }
}

// An atomic barrier drains the outstanding increments, so nothing is reported.
TEST_F(NOCDebuggingFixture, SemaphoreIncWithBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunSemaphoreIncTest(fixture, mesh_device, /*use_barrier=*/true);
            },
            mesh_device);
    }
}

// A full barrier also drains outstanding atomics (it waits on reads, writes AND atomics), so nothing is reported.
TEST_F(NOCDebuggingFixture, SemaphoreIncWithFullBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunSemaphoreIncTest(fixture, mesh_device, /*use_barrier=*/false, /*use_full_barrier=*/true);
            },
            mesh_device);
    }
}

// A write barrier must not drain outstanding atomics (writes and atomics use separate NIU counters): issuing a
// write barrier instead of an atomic barrier must still leave the increments reported as unflushed. Mirror of
// AtomicBarrierDoesNotFlushWrites.
TEST_F(NOCDebuggingFixture, SemaphoreIncWriteBarrierDoesNotFlush) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunSemaphoreIncTest(
                    fixture,
                    mesh_device,
                    /*use_barrier=*/false,
                    /*use_full_barrier=*/false,
                    /*use_wrong_write_barrier=*/true);
            },
            mesh_device);
    }
}

// Multicast atomic increments with no barrier stay outstanding at kernel end -> unflushed multicast semaphore issue.
TEST_F(NOCDebuggingFixture, SemaphoreIncMulticastNoBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunSemaphoreIncMulticastTest(fixture, mesh_device, /*use_barrier=*/false);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, SemaphoreIncMulticastWithBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunSemaphoreIncMulticastTest(fixture, mesh_device, /*use_barrier=*/true);
            },
            mesh_device);
    }
}

// Inline dword writes with no barrier stay outstanding at kernel end -> unflushed-write issue (and never a
// same-src false positive, since they carry no source buffer).
TEST_F(NOCDebuggingFixture, InlineWritesNoBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunInlineWriteTest(fixture, mesh_device, /*use_barrier=*/false);
            },
            mesh_device);
    }
}

// A write barrier drains the inline writes, so nothing is reported.
TEST_F(NOCDebuggingFixture, InlineWritesWithBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunInlineWriteTest(fixture, mesh_device, /*use_barrier=*/true);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, StatefulWritesNoBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunStatefulWriteTest(fixture, mesh_device, /*use_barrier=*/false);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, StatefulWritesWithBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunStatefulWriteTest(fixture, mesh_device, /*use_barrier=*/true);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, StatefulReadsNoBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunStatefulReadTest(fixture, mesh_device, /*use_barrier=*/false);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, StatefulReadsWithBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunStatefulReadTest(fixture, mesh_device, /*use_barrier=*/true);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, ReadsNoBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunReadsTest(fixture, mesh_device, false);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, ReadsWithBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunReadsTest(fixture, mesh_device, true);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, InterleavedReadsWritesNoBarrier) {
    // Only run it on device 0 as it's taking too long
    this->RunTestOnDevice<NOCDebuggingFixture>(
        [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunInterleavedReadsWritesTest(fixture, mesh_device, false, false);
        },
        this->devices_[0]);
}

TEST_F(NOCDebuggingFixture, InterleavedReadsWritesWithBarrier) {
    // Only run it on device 0 as it's taking too long
    this->RunTestOnDevice<NOCDebuggingFixture>(
        [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunInterleavedReadsWritesTest(fixture, mesh_device, true, true);
        },
        this->devices_[0]);
}

void RunMcastTest(
    NOCDebuggingFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    bool use_write_mcast_flush,
    bool use_semaphore_mcast_flush) {
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();

    CoreCoord sender_core = {0, 0};
    CoreRange sender_range(sender_core, sender_core);

    CoreCoord mcast_start = {0, 0};
    CoreCoord mcast_end = {compute_grid_size.x - 1, compute_grid_size.y - 1};

    auto mcast_start_virtual = mesh_device->worker_core_from_logical_core(mcast_start);
    auto mcast_end_virtual = mesh_device->worker_core_from_logical_core(mcast_end);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    uint32_t num_dest_cores = CoreRange(mcast_start, mcast_end).size();

    constexpr uint32_t buffer_page_size = 64;
    constexpr uint32_t buffer_size = buffer_page_size;
    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = buffer_page_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

    std::map<std::string, std::string> defines = {
        {"MCAST_START_X", std::to_string(mcast_start_virtual.x)},
        {"MCAST_START_Y", std::to_string(mcast_start_virtual.y)},
        {"MCAST_END_X", std::to_string(mcast_end_virtual.x)},
        {"MCAST_END_Y", std::to_string(mcast_end_virtual.y)},
        {"NUM_DEST_CORES", std::to_string(num_dest_cores)},
        {"L1_BUFFER_ADDR", std::to_string(l1_buffer->address())},
        {"WRITE_SIZE", std::to_string(buffer_page_size)},
    };

    if (use_write_mcast_flush) {
        defines["USE_WRITE_MCAST_FLUSH"] = "1";
    }
    if (use_semaphore_mcast_flush) {
        defines["USE_SEMAPHORE_MCAST_FLUSH"] = "1";
    }

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/noc_debugging/async_mcast_semaphore.cpp",
        sender_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload);

    ReadMeshDeviceProfilerResults(*mesh_device);

    auto* device = mesh_device->get_devices()[0];
    auto device_id = device->id();
    auto sender_core_virtual = mesh_device->worker_core_from_logical_core(sender_core);

    bool has_write_mcast_issue =
        fixture->has_unflushed_write_mcast_issue(device_id, sender_core_virtual, BRISC_PROCESSOR_ID);

    if (use_write_mcast_flush) {
        EXPECT_FALSE(has_write_mcast_issue)
            << "With write mcast barrier, should NOT have unflushed write mcast issue at device " << device_id
            << " core " << sender_core_virtual.str();
    } else {
        EXPECT_TRUE(has_write_mcast_issue)
            << "Without write mcast barrier, should have unflushed write mcast issue at device " << device_id
            << " core " << sender_core_virtual.str();
    }

    bool has_semaphore_mcast_issue =
        fixture->has_unflushed_semaphore_mcast_issue(device_id, sender_core_virtual, BRISC_PROCESSOR_ID);

    if (use_semaphore_mcast_flush) {
        EXPECT_FALSE(has_semaphore_mcast_issue)
            << "With semaphore mcast barrier, should NOT have unflushed semaphore mcast issue at device " << device_id
            << " core " << sender_core_virtual.str();
    } else {
        EXPECT_TRUE(has_semaphore_mcast_issue)
            << "Without semaphore mcast barrier, should have unflushed semaphore mcast issue at device " << device_id
            << " core " << sender_core_virtual.str();
    }
}

TEST_F(NOCDebuggingFixture, McastNoFlushes) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunMcastTest(
                    fixture, mesh_device, /*use_write_mcast_flush=*/false, /*use_semaphore_mcast_flush=*/false);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, McastOnlyWriteFlush) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunMcastTest(fixture, mesh_device, /*use_write_mcast_flush=*/true, /*use_semaphore_mcast_flush=*/false);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, McastWithAllFlushes) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunMcastTest(fixture, mesh_device, /*use_write_mcast_flush=*/true, /*use_semaphore_mcast_flush=*/true);
            },
            mesh_device);
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
