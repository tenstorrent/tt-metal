// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar emulator: DRAM above the first 64 MiB of a bank.

#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>

#include "common/device_fixture.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kSixtyFourMiB = 1u << 26;
constexpr uint32_t kPatternWords = 256;

bool on_simulator() { return MetalContext::instance().rtoptions().is_simulator_or_emulated(); }

uint32_t dram_unreserved_base() { return MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED); }

std::vector<uint32_t> pattern(uint32_t seed) {
    std::vector<uint32_t> values(kPatternWords);
    std::iota(values.begin(), values.end(), seed);
    return values;
}

}  // namespace

// Three distinct patterns 64 MiB apart in one DRAM channel must all read back. Written and read by
// the host, so this checks the DRAM model, not the device's address path: it fails if the model
// folds addresses above 64 MiB onto the first 64 MiB.
TEST_F(QuasarMeshDeviceSingleCardFixture, DramChannelDoesNotAliasAt64MiB) {
    if (!on_simulator()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator.";
    }
    const uint32_t base = dram_unreserved_base();
    const uint64_t channel_size = this->device().dram_size_per_channel();
    if (channel_size < uint64_t{base} + 2 * kSixtyFourMiB + kPatternWords * sizeof(uint32_t)) {
        GTEST_SKIP() << "DRAM channel 0 is smaller than 192 MiB";
    }
    const uint32_t addresses[3] = {base, base + kSixtyFourMiB, base + 2 * kSixtyFourMiB};
    std::vector<uint32_t> patterns[3] = {pattern(0x11110000u), pattern(0x22220000u), pattern(0x33330000u)};
    for (int i = 0; i < 3; i++) {
        slow_dispatch::WriteToDRAMChannel(this->device(), 0, addresses[i], patterns[i]);
    }
    MetalContext::instance().get_cluster().dram_barrier(this->device().get_device_ids()[0]);
    for (int i = 0; i < 3; i++) {
        std::vector<uint32_t> readback;
        slow_dispatch::ReadFromDRAMChannel(this->device(), 0, addresses[i], kPatternWords * sizeof(uint32_t), readback);
        EXPECT_EQ(readback, patterns[i]) << "DRAM channel 0 did not keep the data written at " << std::hex
                                         << addresses[i] << " (another 64 MiB region aliases onto it)";
    }
}

// A top-down interleaved DRAM buffer lands above 64 MiB once the bank spans the whole view; it must
// write and read back through the command queue (under fast dispatch: the dispatcher's paged path).
TEST_F(QuasarMeshDeviceSingleCardFixture, DramBufferAboveSixtyFourMiB) {
    if (!on_simulator()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator.";
    }
    distributed::MeshDevice& mesh = this->device();
    if (mesh.allocator()->get_bank_size(BufferType::DRAM) <= kSixtyFourMiB) {
        GTEST_SKIP() << "The DRAM bank is bounded to 64 MiB";
    }
    constexpr uint32_t buffer_size = 4096;
    distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = 512, .buffer_type = BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig global_cfg{.size = buffer_size};
    std::shared_ptr<distributed::MeshBuffer> buf = distributed::MeshBuffer::create(global_cfg, local_cfg, &mesh);
    ASSERT_GE(buf->address(), kSixtyFourMiB) << "the top-down buffer did not land above 64 MiB";

    std::vector<uint32_t> src(buffer_size / sizeof(uint32_t));
    std::iota(src.begin(), src.end(), 0x5a5a0000u);
    distributed::MeshCommandQueue& cq = mesh.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, buf, src);
    std::vector<uint32_t> dst;
    distributed::EnqueueReadMeshBuffer(cq, dst, buf, /*blocking=*/true);
    ASSERT_EQ(dst, src);
}

// A DM kernel reads a word the host placed 128 MiB into DRAM bank 0 through the typed DRAM address
// path (the map's DRAM window) into L1.
TEST_F(QuasarMeshDeviceSingleCardFixture, DmReadsDramAboveSixtyFourMiB) {
    if (!on_simulator()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator.";
    }
    const uint32_t dram_address = dram_unreserved_base() + 2 * kSixtyFourMiB;
    if (this->device().dram_size_per_channel() < uint64_t{dram_address} + sizeof(uint32_t)) {
        GTEST_SKIP() << "DRAM channel 0 is smaller than 128 MiB";
    }
    const experimental::NodeCoord node{0, 0};
    const uint32_t l1_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    std::vector<uint32_t> value = {0x0C0FFEE1u};
    slow_dispatch::WriteToDRAMChannel(this->device(), 0, dram_address, value);
    MetalContext::instance().get_cluster().dram_barrier(this->device().get_device_ids()[0]);
    std::vector<uint32_t> zero = {0u};
    slow_dispatch::WriteToL1(this->device(), node, l1_address, zero);

    const experimental::KernelSpecName kernel_name{"dram_to_l1"};
    experimental::KernelSpec kernel_spec{
        .unique_id = kernel_name,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem"}, .accessor_name = "sem"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"dram_addr", "l1_addr", "dram_buffer_size", "dram_bank_id", "signal_value"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::SemaphoreSpec sem{.unique_id = experimental::SemaphoreSpecName{"sem"}, .target_nodes = node};
    experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {kernel_name}, .target_nodes = node};
    experimental::ProgramSpec spec{
        .name = "dram_high_read", .kernels = {kernel_spec}, .semaphores = {sem}, .work_units = {main_wu}};
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = kernel_name,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"dram_addr", dram_address},
             {"l1_addr", l1_address},
             {"dram_buffer_size", static_cast<uint32_t>(sizeof(uint32_t))},
             {"dram_bank_id", 0u},
             {"signal_value", 0u}})});
    experimental::SetProgramRunArgs(program, params);
    LaunchProgram(this->device(), std::move(program));

    std::vector<uint32_t> readback;
    slow_dispatch::ReadFromL1(this->device(), node, l1_address, sizeof(uint32_t), readback);
    ASSERT_EQ(readback.size(), 1u);
    EXPECT_EQ(readback[0], value[0]) << "the kernel read " << std::hex << readback[0] << " from DRAM " << dram_address;
}
