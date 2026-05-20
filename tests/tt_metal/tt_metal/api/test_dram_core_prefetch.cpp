// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Standalone test for the dram_core_prefetcher DRISC kernel:
//  - Seed bank 0's DRAM with a known pattern.
//  - Run the kernel on bank 0's unused subchannel to fetch it via GDDR DMA
//    and push to a single worker receiver via the DRAM-sender GCB.
//  - Verify the receiver's L1 byte-for-byte.
//  - A second case exercises ping-pong staging with multiple blocks.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/experimental/dram_subchannel.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>

#include "impl/kernels/kernel.hpp"  // DramConfig
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"

namespace tt::tt_metal {

class DramCorePrefetcherFixture : public BlackholeSingleCardFixture {
protected:
    void SetUp() override {
        BlackholeSingleCardFixture::SetUp();
        if (devices_.empty()) {
            return;
        }
        const auto& hal = MetalContext::instance().hal();
        if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
        mesh_device_ = devices_[0].get();
        device_ = mesh_device_->get_devices()[0];
    }

    void run_one_test(uint32_t num_blocks, uint32_t block_size) {
        constexpr uint32_t kNumReceivers = 1;
        constexpr uint32_t kRemoteCBId = 31;

        const uint32_t total_bytes = num_blocks * block_size;

        // Seed bank 0 DRAM with random data.
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::vector<uint32_t> data = create_random_vector_of_bfloat16(total_bytes, 1000.0f, seed);

        CoreCoord dram_logical{0, 0};
        uint32_t dram_channel = device_->dram_channel_from_logical_core(dram_logical);

        auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
            .device = device_,
            .size = total_bytes,
            .page_size = total_bytes,
            .buffer_type = BufferType::DRAM,
        });
        const uint32_t dram_addr = dram_buffer->address();
        tt::tt_metal::detail::WriteToDeviceDRAMChannel(device_, dram_channel, dram_addr, data);

        // Sender = bank 0; one worker receiver.
        const uint32_t bank_id = 0;
        CoreRangeSet receiver_cores(CoreRange({0, 0}, {0, 0}));

        // GCB sized to hold at least num_blocks blocks per receiver.
        const uint32_t gcb_size = num_blocks * block_size;
        std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers = {{bank_id, receiver_cores}};
        auto gcb = experimental::CreateGlobalCircularBufferWithDramSenders(
            mesh_device_, bank_to_receivers, gcb_size, BufferType::L1);
        const uint32_t unused_sub = experimental::pick_unused_dram_subchannel(device_, bank_id);
        CoreCoord sender_logical{bank_id, unused_sub};

        // DRISC L1 layout.
        const auto& hal = MetalContext::instance().hal();
        const uint32_t drisc_l1_unreserved =
            hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
        auto align_up = [&](uint32_t a) { return (a + l1_alignment - 1) & ~(l1_alignment - 1); };

        const uint32_t pages_sent_addr = drisc_l1_unreserved;
        uint32_t cursor = pages_sent_addr + 2 * l1_alignment * kNumReceivers;
        cursor = align_up(cursor);
        const uint32_t noc_xy_addr = cursor;
        cursor += 2 * sizeof(uint32_t) * kNumReceivers;
        cursor = align_up(cursor);
        const uint32_t config_addr = cursor;
        cursor += 16;
        cursor = align_up(cursor);
        const uint32_t stage_a_addr = cursor;
        cursor += block_size;
        cursor = align_up(cursor);
        const uint32_t stage_b_addr = cursor;

        // Build program.
        distributed::MeshCoordinateRange device_range(distributed::MeshCoordinate(0, 0));
        Program program = CreateProgram();

        constexpr uint32_t kNumLayers = 1;
        constexpr uint32_t kNumTensors = 1;
        std::vector<uint32_t> sender_compile_args = {
            kNumLayers,
            kNumTensors,
            num_blocks,
            kNumReceivers,
            block_size,
            stage_a_addr,
            stage_b_addr,
            kRemoteCBId,
            pages_sent_addr,
            noc_xy_addr,
            config_addr,
            gcb_size,
            static_cast<uint32_t>(gcb.buffer_address()),
            static_cast<uint32_t>(experimental::pages_sent_worker_l1_base(gcb)),
        };
        KernelHandle sender_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/dram_core_prefetcher.cpp",
            sender_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = sender_compile_args});

        // RT args: bank_id, tensor_offsets[], dma_block_sizes[], push_page_sizes[], recv_xy[]
        std::vector<uint32_t> sender_rt_args = {
            bank_id,
            /*tensor_offset[0]=*/dram_addr,
            /*dma_block_size[0]=*/block_size,
            /*push_page_size[0]=*/block_size / kNumReceivers,
        };
        const auto& receiver_phys = experimental::receiver_coords_per_sender(gcb).at(0);
        for (const auto& c : receiver_phys) {
            sender_rt_args.push_back(c.x);
            sender_rt_args.push_back(c.y);
        }
        SetRuntimeArgs(program, sender_kernel_id, sender_logical, sender_rt_args);

        CircularBufferConfig cb_config(block_size);
        cb_config.remote_index(kRemoteCBId).set_page_size(block_size).set_data_format(tt::DataFormat::Float16_b);
        experimental::CreateCircularBuffer(program, receiver_cores, cb_config, gcb);

        std::vector<uint32_t> receiver_compile_args = {kRemoteCBId, num_blocks};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/gcb_smoke_receiver.cpp",
            receiver_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = receiver_compile_args});

        distributed::MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device_->mesh_command_queue());

        // Verify receiver L1: blocks are pushed sequentially into the ring buffer starting at
        // buffer_address. With num_blocks * block_size <= gcb_size we have no wrap, so the
        // first num_blocks * block_size bytes of receiver L1 should match `data`.
        std::vector<uint32_t> result;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device_, CoreCoord{0, 0}, gcb.buffer_address(), total_bytes, result, CoreType::WORKER);
        EXPECT_EQ(result, data);
    }

    distributed::MeshDevice* mesh_device_{};
    IDevice* device_{};
};

TEST_F(DramCorePrefetcherFixture, OneBlockOneReceiver) {
    constexpr uint32_t kBlockSize = 64;
    run_one_test(/*num_blocks=*/1, kBlockSize);
}

TEST_F(DramCorePrefetcherFixture, PingPongMultipleBlocks) {
    constexpr uint32_t kBlockSize = 64;
    run_one_test(/*num_blocks=*/4, kBlockSize);
}

}  // namespace tt::tt_metal
