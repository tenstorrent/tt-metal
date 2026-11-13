// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include "device_fixture.hpp"
#include "impl/buffers/tensor_prefetcher_fifo.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

namespace {

namespace fifo = tensor_prefetcher_fifo;

// The emulator runs a dispatch-engine poll loop orders of magnitude slower than silicon.
constexpr auto kRetireTimeout = std::chrono::seconds(300);

uint32_t read_word(distributed::MeshDevice& mesh_device, const CoreCoord& core, uint32_t addr) {
    std::vector<uint32_t> word;
    slow_dispatch::ReadFromL1(mesh_device, core, addr, sizeof(uint32_t), word, CoreType::DISPATCH);
    return word.at(0);
}

void write_words(
    distributed::MeshDevice& mesh_device, const CoreCoord& core, uint32_t addr, std::vector<uint32_t> words) {
    slow_dispatch::WriteToL1(mesh_device, core, addr, words, CoreType::DISPATCH);
    MetalContext::instance().get_cluster().l1_barrier(mesh_device.get_device_ids()[0]);
}

}  // namespace

// Fast dispatch launches the Tensor prefetcher's reader and pusher on the dispatch engine together with the CQ
// kernels. Both must retire every request slot the host writes, including after the FIFO wraps, and both must
// exit when the device closes (a kernel that ignored its termination signal would hang teardown).
TEST_F(QuasarMeshDeviceSingleCardFixture, TensorPrefetcherDispatchEngineKernelsRetireFifoSlots) {
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        GTEST_SKIP() << "Requires fast dispatch";
    }
    auto& dispatch_core_manager = MetalContext::instance().get_dispatch_core_manager();
    if (dispatch_core_manager.get_dispatch_core_type() != CoreType::DISPATCH || this->device().num_hw_cqs() != 1) {
        GTEST_SKIP() << "Requires fast dispatch on a dispatch engine serving one CQ";
    }

    const DispatchMemMap& mem_map = MetalContext::instance().dispatch_mem_map();
    ASSERT_GE(mem_map.tensor_prefetcher_region_size(), fifo::kMinRegionBytes);
    const uint32_t base = mem_map.tensor_prefetcher_region_base();

    const ChipId chip = this->device().get_device_ids()[0];
    const uint16_t channel = MetalContext::instance().get_cluster().get_assigned_channel_for_device(chip);
    const tt_cxy_pair dispatch_engine = dispatch_core_manager.dispatcher_core(chip, channel, 0);
    const CoreCoord core(dispatch_engine.x, dispatch_engine.y);

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

}  // namespace tt::tt_metal
