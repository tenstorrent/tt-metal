// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <thread>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include "device_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "hostdev/remote_dfb_config_layout.h"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/tensor_prefetcher_fifo.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/kernel_config/tensor_prefetcher.hpp"
#include "impl/host_api/temp_quasar_api.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

namespace {

namespace fifo = tensor_prefetcher_fifo;

// The emulator runs a dispatch-engine poll loop orders of magnitude slower than silicon.
constexpr auto kRetireTimeout = std::chrono::seconds(300);

std::vector<uint32_t> read_words(
    distributed::MeshDevice& mesh_device, const CoreCoord& core, uint32_t addr, uint32_t bytes, CoreType core_type) {
    std::vector<uint32_t> words;
    slow_dispatch::ReadFromL1(mesh_device, core, addr, bytes, words, core_type);
    return words;
}

uint32_t read_word(distributed::MeshDevice& mesh_device, const CoreCoord& core, uint32_t addr) {
    return read_words(mesh_device, core, addr, sizeof(uint32_t), CoreType::DISPATCH).at(0);
}

void write_words(
    distributed::MeshDevice& mesh_device, const CoreCoord& core, uint32_t addr, std::vector<uint32_t> words) {
    slow_dispatch::WriteToL1(mesh_device, core, addr, words, CoreType::DISPATCH);
    mesh_device.impl().metal_env().get_cluster().l1_barrier(mesh_device.get_device_ids()[0]);
}

// The Tensor prefetcher region on the dispatch engine of `mesh_device`'s one device.
struct DispatchEngineRegion {
    explicit DispatchEngineRegion(distributed::MeshDevice& mesh_device) :
        mesh_device(mesh_device),
        base(mesh_device.impl().metal_context().dispatch_mem_map().unused_l1_base()),
        core(dram_sender_dispatch_engine(mesh_device.impl().get_context_id(), mesh_device.get_device_ids()[0])) {}

    uint32_t read(uint32_t offset) const { return read_word(mesh_device, core, base + offset); }

    // Slots every Tensor prefetcher kernel has retired. The counts are ordered as the host orders them. Fast
    // dispatch zeroes them at launch, so they cannot wrap here; the static_asserts on fifo::count_behind cover
    // the wrap.
    uint32_t slots_retired_by_every_kernel() const {
        uint32_t retired = read(fifo::fifo_rd_offset(0));
        for (uint32_t k = 1; k < tensor_prefetcher_kernels().size(); ++k) {
            const uint32_t kernel_rd = read(fifo::fifo_rd_offset(k));
            if (fifo::count_behind(kernel_rd, retired)) {
                retired = kernel_rd;
            }
        }
        return retired;
    }

    // Waits up to kRetireTimeout for every kernel to retire `target` slots; returns how many they have.
    uint32_t wait_for_retired(uint32_t target) const {
        const auto deadline = std::chrono::steady_clock::now() + kRetireTimeout;
        uint32_t retired = slots_retired_by_every_kernel();
        while (fifo::count_behind(retired, target) && std::chrono::steady_clock::now() < deadline) {
            retired = slots_retired_by_every_kernel();
        }
        return retired;
    }

    // Waits up to kRetireTimeout for the host to have written `target` slots; returns how many it has.
    uint32_t wait_for_written(uint32_t target) const {
        const auto deadline = std::chrono::steady_clock::now() + kRetireTimeout;
        uint32_t written = read(fifo::kFifoWrOffset);
        while (fifo::count_behind(written, target) && std::chrono::steady_clock::now() < deadline) {
            written = read(fifo::kFifoWrOffset);
        }
        return written;
    }

    distributed::MeshDevice& mesh_device;
    uint32_t base = 0;
    CoreCoord core;
};

// Raises Tensor prefetcher op signal `signal_addr` `num_signals` times from a data-movement kernel on `core`.
void raise_op_signal(
    distributed::MeshDevice& mesh_device, const CoreCoord& core, uint32_t signal_addr, uint32_t num_signals) {
    Program program;
    const KernelHandle kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/tensor_prefetcher_signal.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1, .compile_args = {/*signal_noc=*/0}});
    SetRuntimeArgs(program, kernel, core, {signal_addr, num_signals});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);
}

}  // namespace

// Fast dispatch launches the Tensor prefetcher kernels on the dispatch engine together with the CQ kernels.
// Each must retire every request slot the host writes, including after the FIFO wraps, and each must exit
// when the device closes (a kernel that ignored its termination signal would hang teardown).
TEST_F(QuasarMeshDeviceSingleCardFixture, TensorPrefetcherDispatchEngineKernelsRetireFifoSlots) {
    if (!dram_senders_on_dispatch_engine(this->device())) {
        GTEST_SKIP() << "Requires fast dispatch on a dispatch engine serving one CQ";
    }
    const DispatchMemMap& mem_map = this->device().impl().metal_context().dispatch_mem_map();
    const fifo::RegionLayout layout = tensor_prefetcher_region_layout(this->device().impl().metal_env().get_cluster());
    ASSERT_GE(mem_map.unused_l1_size(), layout.min_region_bytes());
    const DispatchEngineRegion region(this->device());
    const uint32_t base = region.base;
    const CoreCoord core = region.core;
    const uint32_t num_kernels = tensor_prefetcher_kernels().size();

    // More slots than the FIFO holds, taking credit the way the host-side producer does.
    constexpr uint32_t kNumSlots = fifo::kFifoSlots + 4;
    fifo::SlotHeader nop{};
    nop.command = fifo::kSlotNop;
    std::vector<uint32_t> nop_words(sizeof(nop) / sizeof(uint32_t));
    std::memcpy(nop_words.data(), &nop, sizeof(nop));
    for (uint32_t written = 0; written < kNumSlots; ++written) {
        if (written >= fifo::kFifoSlots) {
            ASSERT_GE(region.wait_for_retired(written - fifo::kFifoSlots + 1), written - fifo::kFifoSlots + 1)
                << "Tensor prefetcher kernels stopped retiring slots";
        }
        write_words(this->device(), core, base + layout.slot_offset(written), nop_words);
        write_words(this->device(), core, base + fifo::kFifoWrOffset, {written + 1});
    }
    EXPECT_EQ(region.wait_for_retired(kNumSlots), kNumSlots);
    for (uint32_t k = 0; k < num_kernels; ++k) {
        EXPECT_EQ(read_word(this->device(), core, base + fifo::fifo_rd_offset(k)), kNumSlots) << "kernel " << k;
    }
}

// The dispatch engine keeps the one signal slot table its kernels wait on. A Tensix kernel's op signal raises
// its slot once per DRAM bank, since every bank's counter is that slot, and a CQ fence is the dispatcher's
// write to the CQ's slot. No kernel retires a WAIT_SIGNAL slot before its signal arrives.
TEST_F(QuasarMeshDeviceSingleCardFixture, TensorPrefetcherDispatchEngineWaitsOnSignalSlots) {
    if (!dram_senders_on_dispatch_engine(this->device())) {
        GTEST_SKIP() << "Requires fast dispatch on a dispatch engine serving one CQ";
    }
    constexpr uint32_t kSignalId = 3;
    // How long a kernel that ignored the wait would have had to retire it.
    constexpr auto kWaitHoldTime = std::chrono::seconds(5);
    auto& mesh_device = this->device();
    const DispatchEngineRegion region(mesh_device);
    const uint32_t num_banks = mesh_device.allocator()->get_num_banks(BufferType::DRAM);

    experimental::StartTensorPrefetcher(mesh_device, {});
    const uint32_t signal_addr = experimental::GetTensorPrefetcherSignalAddress(mesh_device, kSignalId);
    EXPECT_EQ(signal_addr, region.base + fifo::signal_slot_offset(op_signal_slot_index(kSignalId)));
    for (uint32_t slot = 0; slot < kNumSignalSlots; ++slot) {
        EXPECT_EQ(region.read(fifo::signal_slot_offset(slot)), 0u) << "Start must zero signal slot " << slot;
    }

    const uint32_t written = region.read(fifo::kFifoWrOffset);
    experimental::QueueTensorPrefetcherWaitForSignal(mesh_device, kSignalId, /*trace_capture_cq=*/nullptr);
    ASSERT_EQ(region.wait_for_written(written + 1), written + 1) << "the host never wrote the wait's slot";
    std::this_thread::sleep_for(kWaitHoldTime);
    EXPECT_EQ(region.slots_retired_by_every_kernel(), written) << "a kernel passed a wait before its signal";

    raise_op_signal(mesh_device, CoreCoord{0, 0}, signal_addr, /*num_signals=*/1);
    EXPECT_EQ(region.read(fifo::signal_slot_offset(op_signal_slot_index(kSignalId))), num_banks)
        << "one op signal must raise the engine's slot once per DRAM bank";
    for (uint32_t neighbour : {kSignalId - 1, kSignalId + 1}) {
        EXPECT_EQ(region.read(fifo::signal_slot_offset(op_signal_slot_index(neighbour))), 0u)
            << "a signal landed on op signal " << neighbour << "'s slot";
    }
    EXPECT_EQ(region.wait_for_retired(written + 1), written + 1) << "the kernels never passed the signalled wait";

    experimental::WaitForCqOnTensorPrefetcher(mesh_device.mesh_command_queue(), std::nullopt);
    // The fence is a dispatcher write. Let it run before looking for its effect: a simulator advances fast
    // dispatch only while the host drives the command queue, not while it polls L1.
    distributed::Finish(mesh_device.mesh_command_queue());
    EXPECT_EQ(region.wait_for_retired(written + 2), written + 2) << "the kernels never passed the CQ fence";
    EXPECT_NE(region.read(fifo::signal_slot_offset(mesh_device.mesh_command_queue().id())), 0u)
        << "the dispatcher never wrote the CQ's fence slot";
    experimental::StopTensorPrefetcher(mesh_device);
}

// Pipes carved for the Tensor prefetcher through the ordinary DRAM-sender API land on the dispatch
// engine: one endpoint per bank, whose sender config page is in the engine's L1 and whose receivers
// return credits to the engine.
TEST_F(QuasarMeshDeviceSingleCardFixture, TensorPrefetcherDramSenderPipesLiveOnDispatchEngine) {
    if (!dram_senders_on_dispatch_engine(this->device())) {
        GTEST_SKIP() << "Requires fast dispatch on a dispatch engine serving one CQ";
    }
    auto& mesh_device = this->device();
    const CoreCoord grid = mesh_device.compute_with_storage_grid_size();
    ASSERT_GE(grid.x * grid.y, 2u);
    // Two receivers on bank 0: one pipe, from the bank's one endpoint on the dispatch engine.
    const std::vector<CoreCoord> receivers = {CoreCoord(0, 0), grid.x > 1 ? CoreCoord(1, 0) : CoreCoord(0, 1)};

    experimental::PrefetcherPipeSpaceConfig config;
    config.num_dram_senders = 1;
    config.receiver_domain = CoreRangeSet(CoreRange(receivers[0])).merge(CoreRangeSet(CoreRange(receivers[1])));
    config.ring_size = 4096;
    config.max_receivers_per_pipe = 2;
    auto space = experimental::CreatePrefetcherPipeSpace(mesh_device.mesh_command_queue(0), config);
    const std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers = {{0, config.receiver_domain}};
    auto pipes = experimental::CreatePrefetcherPipesForTensorPrefetcher(space, bank_to_receivers);
    ASSERT_EQ(pipes.size(), 1u);
    const experimental::PrefetcherPipe& pipe = pipes[0];
    EXPECT_EQ(pipe.sender_core_type(), experimental::SenderCoreType::Dram);
    EXPECT_EQ(pipe.sender_core(), CoreCoord(0, 0)) << "an endpoint keeps naming its bank and role";
    ASSERT_EQ(pipe.receiver_cores(), config.receiver_domain);

    const ChipId chip = mesh_device.get_device_ids()[0];
    const CoreCoord dispatch_engine = dram_sender_dispatch_engine(mesh_device.impl().get_context_id(), chip);
    const CoreCoord dispatch_engine_virtual =
        mesh_device.virtual_core_from_logical_core(dispatch_engine, CoreType::DISPATCH);
    const uint32_t page_bytes = pipe.config_page_size();
    const auto sender_page = read_words(
        mesh_device,
        dispatch_engine,
        static_cast<uint32_t>(pipe.impl().sender_state_drisc_l1_base()),
        page_bytes,
        CoreType::DISPATCH);
    EXPECT_EQ(sender_page.at(REMOTE_DFB_CFG_IS_SENDER), 1u);
    EXPECT_EQ(sender_page.at(REMOTE_DFB_CFG_NUM_RECEIVERS), 2u);
    for (const CoreCoord& receiver : receivers) {
        const auto receiver_page =
            read_words(mesh_device, receiver, pipe.config_address(), page_bytes, CoreType::WORKER);
        EXPECT_EQ(receiver_page, pipe.impl().config_page(receiver)) << "receiver " << receiver.str();
        EXPECT_EQ(receiver_page.at(PREFETCHER_PIPE_CONFIG_HEADER_WORDS), dispatch_engine_virtual.x)
            << "receiver " << receiver.str();
        EXPECT_EQ(receiver_page.at(PREFETCHER_PIPE_CONFIG_HEADER_WORDS + 1), dispatch_engine_virtual.y)
            << "receiver " << receiver.str();
    }
}

}  // namespace tt::tt_metal
