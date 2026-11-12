// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Watcher check on two dedicated-NOC data movement kernels that share one NOC
// (DataMovement1XXConfig::dedicated_noc_sharing): they must program disjoint NoC command buffers. A pair that
// keeps to its own command buffers runs clean; a pair that both write trips the check on the second writer.

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-logger/tt-logger.hpp>
#include "debug_tools_fixture.hpp"
#include "impl/context/metal_context.hpp"

namespace {

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t kTransferBytes = 2048;
// The write command buffer in dedicated-NOC mode, on Wormhole and Blackhole (noc_nonblocking_api.h).
constexpr uint32_t kWriteCmdBuf = 0;
const std::string kKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/shared_noc_cmd_buf.cpp";

std::shared_ptr<distributed::MeshBuffer> make_l1_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t size) {
    distributed::DeviceLocalBufferConfig local_config{.page_size = size, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = size};
    return distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
}

// BRISC writes its source buffer to its destination buffer; NCRISC reads its own pair, or also writes when
// ncrisc_writes. Both kernels share NOC 0 in dedicated mode (shared_noc_cmd_buf.cpp).
void RunSharedNocCmdBufTest(
    MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool ncrisc_writes) {
    const auto& hal = MetalContext::instance().hal();
    if (MetalContext::instance().rtoptions().watcher_noc_sanitize_disabled()) {
        GTEST_SKIP();
    }
    if (hal.get_arch() == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Sharing a dedicated NOC is a TT-1.x.x data movement option";
    }

    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    auto brisc_src = make_l1_buffer(mesh_device, kTransferBytes);
    auto brisc_dst = make_l1_buffer(mesh_device, kTransferBytes);
    auto ncrisc_src = make_l1_buffer(mesh_device, kTransferBytes);
    auto ncrisc_dst = make_l1_buffer(mesh_device, kTransferBytes);
    // flags[0]: BRISC's transfer is done; flags[1]: NCRISC's.
    constexpr uint32_t kFlagsBytes = 32;
    auto flags = make_l1_buffer(mesh_device, kFlagsBytes);

    // Each buffer is one interleaved page, which lives in L1 bank 0. The kernels address the buffers as local
    // L1, so they run on bank 0's core.
    const auto& allocator = *brisc_src->get_backing_buffer()->allocator();
    const CoreCoord core = allocator.get_logical_core_from_bank_id(0);
    const CoreCoord virtual_core = device->worker_core_from_logical_core(core);
    const auto l1_addr = [&](const std::shared_ptr<distributed::MeshBuffer>& buffer) {
        return static_cast<uint32_t>(buffer->address() + allocator.get_bank_offset(BufferType::L1, 0));
    };

    // WriteShard is non-blocking and keeps a pointer into its source, so these outlive the program run.
    std::vector<uint32_t> brisc_data = create_random_vector_of_bfloat16(kTransferBytes, 100, 0);
    std::vector<uint32_t> ncrisc_data = create_random_vector_of_bfloat16(kTransferBytes, 100, 1);
    std::vector<uint32_t> zeros(kTransferBytes / sizeof(uint32_t), 0);
    std::vector<uint32_t> flags_clear(kFlagsBytes / sizeof(uint32_t), 0);
    distributed::WriteShard(cq, brisc_src, brisc_data, zero_coord);
    distributed::WriteShard(cq, ncrisc_src, ncrisc_data, zero_coord);
    distributed::WriteShard(cq, brisc_dst, zeros, zero_coord);
    distributed::WriteShard(cq, ncrisc_dst, zeros, zero_coord);
    distributed::WriteShard(cq, flags, flags_clear, zero_coord);

    experimental::KernelSpec::CompilerOptions::Defines defines;
    if (fixture->IsSlowDispatch()) {
        defines["WATCHER_KERNEL_SLOW_DISPATCH"] = "1";
    }
    auto make_kernel = [&](const experimental::KernelSpecName& name,
                           DataMovementProcessor processor,
                           const std::shared_ptr<distributed::MeshBuffer>& src,
                           const std::shared_ptr<distributed::MeshBuffer>& dst,
                           uint32_t my_flag_addr,
                           uint32_t other_flag_addr,
                           bool do_write,
                           bool wait_for_other_first,
                           bool signal_done_early) {
        return experimental::KernelSpec{
            .unique_id = name,
            .source = kKernel,
            .compiler_options = {.defines = defines},
            .compile_time_args =
                {{"src_addr", l1_addr(src)},
                 {"dst_addr", l1_addr(dst)},
                 {"size", kTransferBytes},
                 {"my_flag_addr", my_flag_addr},
                 {"other_flag_addr", other_flag_addr},
                 {"do_write", do_write},
                 {"wait_for_other_first", wait_for_other_first},
                 {"signal_done_early", signal_done_early}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .config_1xx =
                        experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = processor,
                            .noc = NOC::NOC_0,
                            .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                            .dedicated_noc_sharing = experimental::DedicatedNocSharing::SHARED_DISJOINT_TRAFFIC,
                        },
                },
        };
    };
    const experimental::KernelSpecName brisc_name{"brisc_writer"};
    const experimental::KernelSpecName ncrisc_name{ncrisc_writes ? "ncrisc_writer" : "ncrisc_reader"};
    const uint32_t brisc_flag_addr = l1_addr(flags);
    const uint32_t ncrisc_flag_addr = brisc_flag_addr + sizeof(uint32_t);
    // When both write, BRISC goes first and reports the core done up front, since NCRISC's write then trips
    // the watcher and both kernels hang.
    experimental::ProgramSpec spec{
        .name = "watcher_shared_noc_cmd_buf",
        .kernels =
            {make_kernel(
                 brisc_name,
                 DataMovementProcessor::RISCV_0,
                 brisc_src,
                 brisc_dst,
                 brisc_flag_addr,
                 ncrisc_flag_addr,
                 /*do_write=*/true,
                 /*wait_for_other_first=*/false,
                 /*signal_done_early=*/ncrisc_writes),
             make_kernel(
                 ncrisc_name,
                 DataMovementProcessor::RISCV_1,
                 ncrisc_src,
                 ncrisc_dst,
                 ncrisc_flag_addr,
                 brisc_flag_addr,
                 /*do_write=*/ncrisc_writes,
                 /*wait_for_other_first=*/ncrisc_writes,
                 /*signal_done_early=*/false)},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {brisc_name, ncrisc_name},
            .target_nodes = experimental::NodeCoord{core},
        }},
    };

    distributed::MeshWorkload workload;
    workload.add_program(device_range, experimental::MakeProgramFromSpec(*mesh_device, spec));

    if (!ncrisc_writes) {
        fixture->RunProgram(mesh_device, workload, /*wait_for_dump=*/true);
        std::vector<uint32_t> brisc_result;
        std::vector<uint32_t> ncrisc_result;
        distributed::ReadShard(cq, brisc_result, brisc_dst, zero_coord);
        distributed::ReadShard(cq, ncrisc_result, ncrisc_dst, zero_coord);
        EXPECT_EQ(brisc_result, brisc_data);
        EXPECT_EQ(ncrisc_result, ncrisc_data);
        EXPECT_TRUE(MetalContext::instance().watcher_server()->exception_message().empty())
            << MetalContext::instance().watcher_server()->exception_message();
        return;
    }

    fixture->RunProgramExpectingWatcherError(mesh_device, workload);

    // Under fast dispatch BRISC's early done report also uses a command buffer, so the report's list of
    // BRISC's command buffers depends on the dispatch mode. Match up to the one that collided.
    const std::string expected = fmt::format(
        "Device {} worker core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} issued a NoC command on noc0 command "
        "buffer {}, which {} has also used on it",
        device->id(),
        core.x,
        core.y,
        virtual_core.x,
        virtual_core.y,
        hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, 1, false),
        kWriteCmdBuf,
        hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, 0, false));
    log_info(LogTest, "Expected error: {}", expected);
    std::string exception;
    do {
        exception = MetalContext::instance().watcher_server()->exception_message();
    } while (exception.empty());
    log_info(LogTest, "Reported error: {}", exception);
    EXPECT_NE(exception.find(expected), std::string::npos);
}

}  // namespace

TEST_F(MeshWatcherFixture, TensixTestWatcherSharedNocDisjointCmdBufs) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunSharedNocCmdBufTest(fixture, mesh_device, /*ncrisc_writes=*/false);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSharedNocSameCmdBuf) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunSharedNocCmdBufTest(fixture, mesh_device, /*ncrisc_writes=*/true);
        },
        this->devices_[0]);
}
