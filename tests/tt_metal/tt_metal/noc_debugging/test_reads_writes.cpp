// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
        /*expect_issue=*/!use_barrier,
        [fixture](ChipId chip_id, CoreCoord core, int processor_id) {
            return fixture->has_write_barrier_issue(chip_id, core, processor_id);
        },
        "write barrier");
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
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunInterleavedReadsWritesTest(fixture, mesh_device, false, false);
            },
            mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, InterleavedReadsWritesWithBarrier) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice<NOCDebuggingFixture>(
            [](NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunInterleavedReadsWritesTest(fixture, mesh_device, true, true);
            },
            mesh_device);
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
