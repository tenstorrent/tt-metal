// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
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
    bool use_full_barrier = false) {
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

    uint32_t num_dest_cores =
        (mcast_end_virtual.x - mcast_start_virtual.x + 1) * (mcast_end_virtual.y - mcast_start_virtual.y + 1);

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

    uint32_t num_dest_cores =
        (mcast_end_virtual.x - mcast_start_virtual.x + 1) * (mcast_end_virtual.y - mcast_start_virtual.y + 1);

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
