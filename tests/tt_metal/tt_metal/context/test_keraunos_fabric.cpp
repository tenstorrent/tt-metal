// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 5 of keraunos_plan.md: fabric-shaped tests, the Blackhole fabric data paths on Keraunos.
//   Worker -> router -> destination: a Mimir CCE writes a packet into a K CCE0 channel buffer and bumps a counter;
//     the K CCE forwards the payload to another HSIO tile's SRAM or to Mimir GDDR and returns a credit.
//   Sender/receiver split: CCE0 of a tile owns the channel buffer and CCE1 forwards, synchronised with peer flags.

#include "keraunos_cce_fixture.hpp"

#include <tt-umd/coordinates/grendel_noc_address_resolver.hpp>

namespace tt::tt_metal {
namespace {

using namespace keraunos_test;

constexpr uint32_t kChannelOffset = 0x1000;   // relative to the unreserved base
constexpr uint32_t kCounterOffset = 0x800;
constexpr uint32_t kRouterStatusOffset = 0x0;
constexpr uint32_t kPayloadWords = 64;

class KeraunosFabric : public KeraunosCceTest {
protected:
    uint32_t mimir_cce_offset() const {
        return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    }
    uint64_t mimir_cce_noc_addr() const {
        return MetalContext::instance().hal().get_dev_noc_addr(
            HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    }
    static std::vector<uint32_t> payload(uint32_t tag) {
        std::vector<uint32_t> p(kPayloadWords);
        for (uint32_t i = 0; i < p.size(); i++) {
            p[i] = (tag << 16) | i;
        }
        return p;
    }
    // The packet the worker builds: header word 0 = 0xFAB10000 | payload bytes, word 1 = sequence.
    static std::vector<uint32_t> packet(uint32_t tag) {
        std::vector<uint32_t> p = {0xFAB10000u | (kPayloadWords * 4), 1};
        const auto data = payload(tag);
        p.insert(p.end(), data.begin(), data.end());
        return p;
    }

    enum class Dest { HsioSram, M1Gddr };

    // Worker on M1's first CCE, router on HSIO 2 CCE0, destination as given.
    void expect_worker_router(Dest dest, uint32_t tag) {
        const auto router = hsio_core(2, 0);
        ASSERT_TRUE(router.has_value());
        constexpr uint32_t kMimir = 1;
        uint32_t mimir_cce = 0;
        for (; mimir_cce < num_mimir_cces_on(device_); ++mimir_cce) {
            if (mimir_index_for_cce(soc_desc_of(device_), mimir_cce) == kMimir) {
                break;
            }
        }
        ASSERT_LT(mimir_cce, num_mimir_cces_on(device_));
        const CoreCoord mimir_core = logical_mimir_cce_core(soc_desc_of(device_), mimir_cce);
        const uint64_t noc = mimir_cce_noc_addr();
        const uint32_t worker_status = mimir_cce_offset();
        const uint32_t worker_credit = mimir_cce_offset() + 0x100;

        const auto data = payload(tag);
        const std::vector<uint32_t> zeros(data.size(), 0);
        // Clean slate.
        write_words(*router, kChannelOffset, std::vector<uint32_t>(kPayloadWords + 2, 0));
        write_words(*router, kCounterOffset, {0});
        write_words(*router, kRouterStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
        const std::vector<uint32_t> zero_words(80, 0);
        write_mimir_cce_sram(device_, mimir_core, noc, zero_words.data(), zero_words.size() * sizeof(uint32_t));

        std::optional<Hsio> dest_core;
        uint32_t dest_index = 0, dest_cce = 0, dest_offset = 0;
        uint32_t kind = 0;
        GddrBackingScope gddr_backing;
        const uint32_t gddr_offset = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
        if (dest == Dest::HsioSram) {
            dest_core = hsio_core(3, 0);
            ASSERT_TRUE(dest_core.has_value());
            dest_index = 3;
            dest_offset = unreserved_ + kDstOffset;
            write_words(*dest_core, kDstOffset, zeros);
        } else {
            kind = 2;
            dest_index = kMimir;
            dest_offset = gddr_offset;
            write_mimir_gddr(*mesh_, device_, kMimir, gddr_offset, zeros);
        }

        Program program = CreateProgram();
        const KernelHandle router_kernel = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_fabric_router.cpp",
            router->logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
        SetRuntimeArgs(
            program, router_kernel, router->logical,
            {unreserved_ + kChannelOffset, unreserved_ + kCounterOffset, kind, dest_index, dest_cce, dest_offset,
             mimir_cce, worker_credit, unreserved_ + kRouterStatusOffset});
        const KernelHandle worker_kernel = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_fabric_worker.cpp",
            mimir_core,
            DramConfig{.noc = NOC::NOC_0});
        std::vector<uint32_t> worker_args = split(global_addr(*router, kChannelOffset));
        const auto counter = split(global_addr(*router, kCounterOffset));
        worker_args.insert(worker_args.end(), counter.begin(), counter.end());
        worker_args.insert(worker_args.end(), {kPayloadWords, tag, worker_credit, worker_status});
        SetRuntimeArgs(program, worker_kernel, mimir_core, worker_args);

        try {
            distributed::MeshWorkload workload;
            workload.add_program(distributed::MeshCoordinateRange(mesh_->shape()), std::move(program));
            distributed::EnqueueMeshWorkload(mesh_->mesh_command_queue(), workload, /*blocking=*/true);
        } catch (const std::exception& e) {
            const auto rs = read_words(*router, kRouterStatusOffset, kStatusWords);
            std::vector<uint32_t> ws(8, 0);
            read_mimir_cce_sram(device_, mimir_core, noc, ws.data(), ws.size() * sizeof(uint32_t));
            FAIL() << "launch failed: " << e.what() << "\nrouter status: " << fmt::format("{:#x}", fmt::join(rs, " "))
                   << "\nworker status: " << fmt::format("{:#x}", fmt::join(ws, " "));
        }

        const auto router_status = read_words(*router, kRouterStatusOffset, kStatusWords);
        EXPECT_EQ(router_status[1], 1u);
        EXPECT_EQ(router_status[2], 1u);
        EXPECT_EQ(router_status[4], kAcked);
        std::vector<uint32_t> worker_status_words(8, 0);
        read_mimir_cce_sram(device_, mimir_core, noc, worker_status_words.data(), worker_status_words.size() * sizeof(uint32_t));
        EXPECT_EQ(worker_status_words[2], 1u) << "credit not seen";
        EXPECT_EQ(worker_status_words[4], kAcked);
        // The packet the worker wrote is in the router's channel buffer, and the payload reached the destination.
        EXPECT_EQ(read_words(*router, kChannelOffset, kPayloadWords + 2), packet(tag));
        if (dest == Dest::HsioSram) {
            EXPECT_EQ(read_words(*dest_core, kDstOffset, data.size()), data);
        } else {
            std::vector<uint32_t> got;
            read_mimir_gddr(*mesh_, device_, kMimir, gddr_offset, kPayloadWords * 4, got);
            EXPECT_EQ(got, data);
        }
    }

    // CCE0 sender, CCE1 receiver of HSIO tile 0, destination as given. The host plays the worker and fills CCE0's
    // channel buffer: word 0 = payload bytes, payload from word 1.
    void expect_sender_receiver(Dest dest, uint32_t tag) {
        const auto sender = hsio_core(0, 0);
        const auto receiver = hsio_core(0, 1);
        ASSERT_TRUE(sender.has_value() && receiver.has_value());
        constexpr uint32_t kReady = 0x400;
        constexpr uint32_t kCredit = 0x440;
        constexpr uint32_t kStatus = 0x0;
        const auto data = payload(tag);
        std::vector<uint32_t> buffer = {kPayloadWords * 4};
        buffer.insert(buffer.end(), data.begin(), data.end());
        write_words(*sender, kChannelOffset, buffer);
        write_words(*sender, kReady, {0});
        write_words(*sender, kCredit, {0});
        write_words(*sender, kStatus, std::vector<uint32_t>(kStatusWords, 0));
        write_words(*receiver, kStatus, std::vector<uint32_t>(kStatusWords, 0));

        GddrBackingScope gddr_backing;
        const uint32_t gddr_offset = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
        const std::vector<uint32_t> zeros(data.size(), 0);
        std::optional<Hsio> dest_core;
        uint32_t kind = 0, dest_index = 1, dest_cce = 1, dest_offset = unreserved_ + kDstOffset;
        if (dest == Dest::HsioSram) {
            dest_core = hsio_core(1, 1);
            ASSERT_TRUE(dest_core.has_value());
            write_words(*dest_core, kDstOffset, zeros);
        } else {
            kind = 2;
            dest_index = 1;
            dest_cce = 0;
            dest_offset = gddr_offset;
            write_mimir_gddr(*mesh_, device_, 1, gddr_offset, zeros);
        }

        Program program = CreateProgram();
        uint32_t role = 0;
        for (const Hsio* core : {&*sender, &*receiver}) {
            const KernelHandle k = CreateKernel(
                program,
                std::string(kKernelDir) + "keraunos_fabric_split.cpp",
                core->logical,
                EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
            SetRuntimeArgs(
                program, k, core->logical,
                {role++, unreserved_ + kChannelOffset, unreserved_ + kReady, unreserved_ + kCredit, kind, dest_index,
                 dest_cce, dest_offset, unreserved_ + kStatus});
        }
        launch(program, *sender);
        for (const Hsio* core : {&*sender, &*receiver}) {
            const auto status = read_words(*core, kStatus, kStatusWords);
            EXPECT_EQ(status[1], 1u) << fmt::format("status: {:#x}", fmt::join(status, " "));
            EXPECT_EQ(status[4], kAcked);
        }
        if (dest == Dest::HsioSram) {
            EXPECT_EQ(read_words(*dest_core, kDstOffset, data.size()), data);
        } else {
            std::vector<uint32_t> got;
            read_mimir_gddr(*mesh_, device_, 1, gddr_offset, kPayloadWords * 4, got);
            EXPECT_EQ(got, data);
        }
    }
};

TEST_F(KeraunosFabric, WorkerRouterToOtherTileSram) { expect_worker_router(Dest::HsioSram, 0x31); }
TEST_F(KeraunosFabric, WorkerRouterToM1Gddr) { expect_worker_router(Dest::M1Gddr, 0x32); }
TEST_F(KeraunosFabric, SenderReceiverSplitToOtherTileCce1Sram) { expect_sender_receiver(Dest::HsioSram, 0x33); }
TEST_F(KeraunosFabric, SenderReceiverSplitToM1Gddr) { expect_sender_receiver(Dest::M1Gddr, 0x34); }

}  // namespace
}  // namespace tt::tt_metal
