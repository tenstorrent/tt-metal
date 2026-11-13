// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
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

bool dram_senders_on_dispatch_engine(const distributed::MeshDevice& mesh_device) {
    return mesh_device.impl().dram_sender_placement() == DramSenderPlacement::DispatchEngine;
}

}  // namespace

// Fast dispatch launches the Tensor prefetcher's reader and pusher on the dispatch engine together with the CQ
// kernels. Both must retire every request slot the host writes, including after the FIFO wraps, and both must
// exit when the device closes (a kernel that ignored its termination signal would hang teardown).
TEST_F(QuasarMeshDeviceSingleCardFixture, TensorPrefetcherDispatchEngineKernelsRetireFifoSlots) {
    if (!dram_senders_on_dispatch_engine(this->device())) {
        GTEST_SKIP() << "Requires fast dispatch on a dispatch engine serving one CQ";
    }
    const DispatchMemMap& mem_map = this->device().impl().metal_context().dispatch_mem_map();
    ASSERT_GE(mem_map.unused_l1_size(), fifo::kMinRegionBytes);
    const uint32_t base = mem_map.unused_l1_base();
    const CoreCoord core =
        dram_sender_dispatch_engine(this->device().impl().get_context_id(), this->device().get_device_ids()[0]);

    auto slots_retired_by_both = [&]() {
        return std::min(
            read_word(this->device(), core, base + fifo::kReaderRdOffset),
            read_word(this->device(), core, base + fifo::kPusherRdOffset));
    };
    auto wait_for_retired = [&](uint32_t target) {
        const auto deadline = std::chrono::steady_clock::now() + kRetireTimeout;
        uint32_t retired = slots_retired_by_both();
        while (retired < target && std::chrono::steady_clock::now() < deadline) {
            retired = slots_retired_by_both();
        }
        return retired;
    };

    // More slots than the FIFO holds, taking credit the way the host-side producer does.
    constexpr uint32_t kNumSlots = fifo::kFifoSlots + 4;
    fifo::SlotHeader nop{};
    nop.command = fifo::kSlotNop;
    std::vector<uint32_t> nop_words(sizeof(nop) / sizeof(uint32_t));
    std::memcpy(nop_words.data(), &nop, sizeof(nop));
    for (uint32_t written = 0; written < kNumSlots; ++written) {
        if (written >= fifo::kFifoSlots) {
            ASSERT_GE(wait_for_retired(written - fifo::kFifoSlots + 1), written - fifo::kFifoSlots + 1)
                << "Tensor prefetcher kernels stopped retiring slots";
        }
        write_words(this->device(), core, base + fifo::slot_offset(written), nop_words);
        write_words(this->device(), core, base + fifo::kFifoWrOffset, {written + 1});
    }
    EXPECT_EQ(wait_for_retired(kNumSlots), kNumSlots);
    EXPECT_EQ(read_word(this->device(), core, base + fifo::kReaderRdOffset), kNumSlots);
    EXPECT_EQ(read_word(this->device(), core, base + fifo::kPusherRdOffset), kNumSlots);
}

// Pipes carved for the Tensor prefetcher through the ordinary DRAM-sender API land on the dispatch
// engine: each endpoint's sender config page is in the engine's L1, and its receivers return credits
// to the engine.
TEST_F(QuasarMeshDeviceSingleCardFixture, TensorPrefetcherDramSenderPipesLiveOnDispatchEngine) {
    if (!dram_senders_on_dispatch_engine(this->device())) {
        GTEST_SKIP() << "Requires fast dispatch on a dispatch engine serving one CQ";
    }
    auto& mesh_device = this->device();
    const CoreCoord grid = mesh_device.compute_with_storage_grid_size();
    ASSERT_GE(grid.x * grid.y, 2u);
    // Two receivers on bank 0, which splits them across the bank's two sender endpoints: two pipes
    // whose endpoints share the one dispatch engine.
    const std::vector<CoreCoord> receivers = {CoreCoord(0, 0), grid.x > 1 ? CoreCoord(1, 0) : CoreCoord(0, 1)};

    experimental::PrefetcherPipeSpaceConfig config;
    config.num_dram_senders = 2;
    config.receiver_domain = CoreRangeSet(CoreRange(receivers[0])).merge(CoreRangeSet(CoreRange(receivers[1])));
    config.ring_size = 4096;
    config.max_receivers_per_pipe = 1;
    auto space = experimental::CreatePrefetcherPipeSpace(mesh_device, config);
    const std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers = {{0, config.receiver_domain}};
    auto pipes = experimental::CreatePrefetcherPipesForTensorPrefetcher(space, bank_to_receivers);
    ASSERT_EQ(pipes.size(), 2u);

    const ChipId chip = mesh_device.get_device_ids()[0];
    const CoreCoord dispatch_engine = dram_sender_dispatch_engine(mesh_device.impl().get_context_id(), chip);
    const CoreCoord dispatch_engine_virtual =
        mesh_device.virtual_core_from_logical_core(dispatch_engine, CoreType::DISPATCH);
    const uint32_t page_bytes = pipes[0].config_page_size();
    for (uint32_t b = 0; b < pipes.size(); ++b) {
        const experimental::PrefetcherPipe& pipe = pipes[b];
        EXPECT_EQ(pipe.sender_core_type(), experimental::SenderCoreType::Dram);
        EXPECT_EQ(pipe.sender_core(), CoreCoord(0, b)) << "an endpoint keeps naming its bank and role";
        ASSERT_EQ(pipe.receiver_cores(), CoreRangeSet(CoreRange(receivers[b])));

        const auto sender_page = read_words(
            mesh_device,
            dispatch_engine,
            static_cast<uint32_t>(pipe.impl().sender_state_drisc_l1_base()),
            page_bytes,
            CoreType::DISPATCH);
        EXPECT_EQ(sender_page.at(REMOTE_DFB_CFG_IS_SENDER), 1u) << "endpoint " << b;
        EXPECT_EQ(sender_page.at(REMOTE_DFB_CFG_NUM_RECEIVERS), 1u) << "endpoint " << b;

        const auto receiver_page =
            read_words(mesh_device, receivers[b], pipe.config_address(), page_bytes, CoreType::WORKER);
        EXPECT_EQ(receiver_page, pipe.impl().config_page(receivers[b])) << "endpoint " << b;
        EXPECT_EQ(receiver_page.at(PREFETCHER_PIPE_CONFIG_HEADER_WORDS), dispatch_engine_virtual.x) << "endpoint " << b;
        EXPECT_EQ(receiver_page.at(PREFETCHER_PIPE_CONFIG_HEADER_WORDS + 1), dispatch_engine_virtual.y)
            << "endpoint " << b;
    }
    // Two endpoints on one engine must not share a sender page.
    EXPECT_NE(pipes[0].impl().sender_state_drisc_l1_base(), pipes[1].impl().sender_state_drisc_l1_base());
}

}  // namespace tt::tt_metal
