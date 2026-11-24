// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "fabric_fixture.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/host_rdma/host_rdma.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_fabric::fabric_router_tests {

namespace {

using tt::tt_fabric::host_rdma::Completion;
using tt::tt_fabric::host_rdma::WriteRequest;

constexpr uint32_t kCounterOffset = 0;
constexpr uint32_t kDataOffset = 64;
constexpr uint32_t kSrcOffset = 768 * 1024;
constexpr uint32_t kServiceDramBank = 0;

template <tt_fabric::FabricConfig CONFIG>
class HostRdmaFixture : public BaseFabricFixture {
public:
    static void SetUpTestSuite() {
        tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_host_rdma(true);
        BaseFabricFixture::DoSetUpTestSuite(CONFIG);
    }
    static void TearDownTestSuite() {
        BaseFabricFixture::DoTearDownTestSuite();
        tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_host_rdma(false);
    }

    void SetUp() override {
        BaseFabricFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Host-initiated RDMA is Blackhole only";
        }
        auto& context = tt::tt_metal::MetalContext::instance();
        if (!context.hal().has_programmable_core_type(tt::tt_metal::HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "The host_rdma service runs on a DRAM programmable core";
        }
        const auto& control_plane = context.get_control_plane();
        l1_base_ = static_cast<uint32_t>(context.hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED));
        for (const auto& [chip, mesh_device] : devices_map_) {
            const FabricNodeId node = control_plane.get_fabric_node_id_from_physical_chip_id(chip);
            chips_.push_back(
                Chip{.chip = chip, .node = node, .core = mesh_device->worker_core_from_logical_core({0, 0})});
        }
    }

    void TearDown() override {
        for (auto& [chip, svc] : services_) {
            svc->drain();
        }
        services_.clear();
        chips_.clear();
    }

protected:
    struct Chip {
        ChipId chip;
        FabricNodeId node;
        tt::tt_metal::CoreCoord core;
    };

    tt::tt_fabric::host_rdma::HostRdmaService& service(const Chip& chip) {
        auto it = services_.find(chip.chip);
        if (it == services_.end()) {
            // A DRAM core that serves no DRAM view: both of its NIUs are in stream mode, so it suits either NoC the
            // fabric routers credit the service on.
            const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(chip.chip);
            const tt::tt_metal::CoreCoord service_core =
                tt::tt_metal::pick_unused_dram_logical_core(soc_desc, kServiceDramBank, chip.chip);
            it = services_
                     .emplace(
                         chip.chip,
                         tt::tt_fabric::host_rdma::HostRdmaService::create(
                             devices_map_.at(chip.chip), service_core, tt::tt_fabric::host_rdma::ServiceCore::Dram))
                     .first;
        }
        return *it->second;
    }

    void write_l1(const Chip& chip, uint32_t addr, const std::vector<uint32_t>& words) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            words.data(), words.size() * sizeof(uint32_t), tt_cxy_pair(chip.chip, chip.core), addr);
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(chip.chip, {chip.core});
    }

    std::vector<uint32_t> read_l1(const Chip& chip, uint32_t addr, size_t num_words) {
        std::vector<uint32_t> words(num_words);
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            words.data(), num_words * sizeof(uint32_t), tt_cxy_pair(chip.chip, chip.core), addr);
        return words;
    }

    void wait_for_counter(const Chip& chip, uint32_t expected) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        uint32_t value = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            value = read_l1(chip, l1_base_ + kCounterOffset, 1)[0];
            if (value == expected) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        std::string service_states;
        for (const auto& [service_chip, svc] : services_) {
            service_states += "\n  " + svc->describe();
        }
        FAIL() << "Completion counter on chip " << chip.chip << " is " << value << ", expected " << expected
               << service_states;
    }

    void reset_destination(const Chip& chip, size_t data_words) {
        write_l1(chip, l1_base_ + kCounterOffset, {0});
        write_l1(chip, l1_base_ + kDataOffset, std::vector<uint32_t>(data_words, 0xDEADBEEF));
    }

    static std::vector<uint32_t> pattern(size_t num_words, uint32_t seed) {
        std::mt19937 gen(seed);
        std::vector<uint32_t> words(num_words);
        for (auto& w : words) {
            w = gen();
        }
        return words;
    }

    void write_and_check(Chip& src, const Chip& dst, size_t num_words, uint32_t seed) {
        reset_destination(dst, num_words);
        const auto data = pattern(num_words, seed);
        const Completion completion{.virtual_core = dst.core, .addr = l1_base_ + kCounterOffset, .increment = 1};
        write_l1(src, l1_base_ + kSrcOffset, data);
        service(src).write(WriteRequest{
            .src = {src.core, l1_base_ + kSrcOffset},
            .dst = dst.node,
            .dst_at = {dst.core, l1_base_ + kDataOffset},
            .size = static_cast<uint32_t>(num_words * sizeof(uint32_t)),
            .completion = completion});
        wait_for_counter(dst, 1);
        EXPECT_EQ(read_l1(dst, l1_base_ + kDataOffset, num_words), data)
            << "chip " << src.chip << " -> chip " << dst.chip << ", " << num_words * 4 << " B";
    }

    void check_one_worker_channel_per_routing_plane() {
        auto& context = tt::tt_metal::MetalContext::instance();
        const auto& control_plane = context.get_control_plane();
        const auto& cluster = context.get_cluster();
        const auto& edm_config = control_plane.get_fabric_context().get_builder_context().get_fabric_router_config();
        const auto table_addr = context.hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS);
        size_t routers = 0;
        for (const auto& chip : chips_) {
            const auto any_worker = devices_map_.at(chip.chip)->worker_core_from_logical_core({2, 2});
            tt::tt_fabric::tensix_fabric_connections_l1_info_t table{};
            cluster.read_core(&table, sizeof(table), tt_cxy_pair(chip.chip, any_worker), table_addr);
            uint32_t routing_planes = 0;
            for (const auto direction :
                 {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
                const auto eth_chans = control_plane.get_active_fabric_eth_channels_in_direction(chip.node, direction);
                EXPECT_EQ(
                    eth_chans.size(),
                    control_plane.get_active_fabric_eth_routing_planes_in_direction(chip.node, direction).size())
                    << "chip " << chip.chip << " direction " << static_cast<int>(direction);
                for (const chan_id_t eth_chan : eth_chans) {
                    routing_planes++;
                    const auto host_rdma = control_plane.get_host_rdma_connection_info(chip.node, eth_chan);
                    ASSERT_NE((table.valid_connections_mask >> eth_chan) & 1u, 0u)
                        << "chip " << chip.chip << " eth chan " << eth_chan << " has no worker connection";
                    const auto& entry = table.read_only[eth_chan];
                    EXPECT_EQ(
                        entry.edm_worker_location_info_addr,
                        edm_config.sender_channels_worker_conn_info_base_address[0]);
                    EXPECT_EQ(
                        entry.edm_connection_handshake_addr,
                        edm_config.sender_channels_connection_semaphore_address[0]);
                    EXPECT_NE(entry.edm_worker_location_info_addr, host_rdma.edm_worker_location_info_addr);
                    EXPECT_NE(entry.edm_connection_handshake_addr, host_rdma.edm_connection_handshake_addr);
                    EXPECT_NE(entry.buffer_index_semaphore_id, host_rdma.buffer_index_semaphore_id);
                    EXPECT_NE(entry.edm_buffer_base_addr, host_rdma.edm_buffer_base_addr);
                    EXPECT_NE(entry.worker_free_slots_stream_id, host_rdma.worker_free_slots_stream_id);
                    routers++;
                }
            }
            EXPECT_EQ(static_cast<uint32_t>(std::popcount(table.valid_connections_mask)), routing_planes)
                << "chip " << chip.chip << ": worker connections, one per routing plane";
        }
        log_info(tt::LogTest, "host_rdma: {} routers, one worker sender channel each", routers);
        EXPECT_GT(routers, 0u);
    }

    bool neighbours(const Chip& a, const Chip& b) const {
        return tt::tt_fabric::are_intra_mesh_neighbors(*devices_map_.at(a.chip), a.node, b.node);
    }

    void run_a_to_b(bool neighbours_only) {
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        const size_t max_payload_words = [&] {
            const auto& fabric_context = control_plane.get_fabric_context();
            return fabric_context.get_fabric_max_payload_size_bytes() / sizeof(uint32_t);
        }();
        const size_t smallest_aligned = 16 / sizeof(uint32_t);
        const size_t under_one_packet = 256;
        const size_t several_packets_partial_last = 5 * max_payload_words + 12;
        const std::vector<size_t> sizes = {
            smallest_aligned, under_one_packet, max_payload_words, several_packets_partial_last};
        uint32_t seed = 1;
        size_t pairs = 0;
        for (auto& src : chips_) {
            for (const auto& dst : chips_) {
                if (src.chip == dst.chip) {
                    continue;
                }
                if (neighbours_only && !neighbours(src, dst)) {
                    continue;
                }
                for (const size_t words : sizes) {
                    write_and_check(src, dst, words, seed++);
                    if (HasFatalFailure() || HasFailure()) {
                        return;
                    }
                }
                pairs++;
            }
        }
        log_info(tt::LogTest, "host_rdma A->B: {} chip pairs, {} sizes each", pairs, sizes.size());
        EXPECT_GT(pairs, 0u);
    }

    void run_all_gather(size_t shard_words, uint32_t rounds) {
        std::vector<Chip*> group;
        group.reserve(chips_.size());
        for (auto& c : chips_) {
            group.push_back(&c);
        }
        const size_t n = group.size();
        ASSERT_GE(n, 2u);
        log_info(tt::LogTest, "host_rdma all-gather over {} chips", n);
        for (uint32_t round = 0; round < rounds; ++round) {
            std::vector<std::vector<uint32_t>> shards;
            shards.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                shards.push_back(pattern(shard_words, 1000 * (round + 1) + static_cast<uint32_t>(i)));
            }
            for (size_t d = 0; d < n; ++d) {
                reset_destination(*group[d], n * shard_words);
                write_l1(*group[d], l1_base_ + kDataOffset + d * shard_words * sizeof(uint32_t), shards[d]);
            }
            for (size_t s = 0; s < n; ++s) {
                std::vector<WriteRequest> batch;
                for (size_t d = 0; d < n; ++d) {
                    if (s == d) {
                        continue;
                    }
                    const uint32_t slot =
                        l1_base_ + kDataOffset + static_cast<uint32_t>(s * shard_words * sizeof(uint32_t));
                    const Completion completion{
                        .virtual_core = group[d]->core, .addr = l1_base_ + kCounterOffset, .increment = 1};
                    batch.push_back(WriteRequest{
                        .src = {group[s]->core, slot},
                        .dst = group[d]->node,
                        .dst_at = {group[d]->core, slot},
                        .size = static_cast<uint32_t>(shard_words * sizeof(uint32_t)),
                        .completion = completion});
                }
                service(*group[s]).write_batch(batch);
            }
            for (size_t d = 0; d < n; ++d) {
                wait_for_counter(*group[d], static_cast<uint32_t>(n - 1));
                if (HasFatalFailure() || HasFailure()) {
                    return;
                }
                const auto gathered = read_l1(*group[d], l1_base_ + kDataOffset, n * shard_words);
                for (size_t s = 0; s < n; ++s) {
                    const std::vector<uint32_t> got(
                        gathered.begin() + s * shard_words, gathered.begin() + (s + 1) * shard_words);
                    EXPECT_EQ(got, shards[s]) << "round " << round << ": chip " << group[d]->chip << " shard " << s;
                }
            }
        }
    }

    std::vector<Chip> chips_;
    uint32_t l1_base_ = 0;
    std::map<ChipId, std::shared_ptr<tt::tt_fabric::host_rdma::HostRdmaService>> services_;
};

using HostRdmaFabric2DFixture = HostRdmaFixture<tt_fabric::FabricConfig::FABRIC_2D>;

}  // namespace

TEST_F(HostRdmaFabric2DFixture, AToBNeighbours) { run_a_to_b(/*neighbours_only=*/true); }
TEST_F(HostRdmaFabric2DFixture, AToBAllPairs) { run_a_to_b(/*neighbours_only=*/false); }
TEST_F(HostRdmaFabric2DFixture, AllGather) { run_all_gather(/*shard_words=*/4096, /*rounds=*/3); }

TEST_F(HostRdmaFabric2DFixture, OneWorkerChannelPerRoutingPlane) { check_one_worker_channel_per_routing_plane(); }

// A benchmark: --gtest_also_run_disabled_tests --gtest_filter=*HostRdma*Bandwidth*
TEST_F(HostRdmaFabric2DFixture, DISABLED_Bandwidth) {
    auto& src = chips_[0];
    const Chip* dst = nullptr;
    for (const auto& c : chips_) {
        if (c.chip != src.chip && neighbours(src, c)) {
            dst = &c;
            break;
        }
    }
    ASSERT_NE(dst, nullptr);
    {
        constexpr uint32_t window_bytes = 256 * 1024;
        constexpr size_t total_bytes = 64ull * 1024 * 1024;
        write_l1(src, l1_base_ + kSrcOffset, pattern(window_bytes / sizeof(uint32_t), 11));
        auto& svc = service(src);
        // Without a warm-up the first measurement reads about 40% low.
        reset_destination(*dst, 1);
        constexpr uint32_t warmup_requests = total_bytes / 1024;
        for (uint32_t i = 0; i < warmup_requests; ++i) {
            const bool last = i + 1 == warmup_requests;
            svc.write(WriteRequest{
                .src = {src.core, l1_base_ + kSrcOffset},
                .dst = dst->node,
                .dst_at = {dst->core, l1_base_ + kDataOffset},
                .size = 1024,
                .completion = last ? std::optional<Completion>(
                                         Completion{.virtual_core = dst->core, .addr = l1_base_ + kCounterOffset})
                                   : std::nullopt});
        }
        wait_for_counter(*dst, 1);
        for (const uint32_t request_bytes : {1024u, 4096u, 16384u, 65536u}) {
            for (const size_t batch_size : {size_t{1}, size_t{8}, size_t{32}}) {
                reset_destination(*dst, 1);
                const size_t num_requests = total_bytes / request_bytes;
                std::vector<WriteRequest> batch;
                batch.reserve(batch_size);
                const auto t0 = std::chrono::steady_clock::now();
                uint32_t offset = 0;
                for (size_t i = 0; i < num_requests; ++i) {
                    if (offset + request_bytes > window_bytes) {
                        offset = 0;
                    }
                    const bool last = i + 1 == num_requests;
                    batch.push_back(WriteRequest{
                        .src = {src.core, l1_base_ + kSrcOffset + offset},
                        .dst = dst->node,
                        .dst_at = {dst->core, l1_base_ + kDataOffset + offset},
                        .size = request_bytes,
                        .completion = last ? std::optional<Completion>(Completion{
                                                 .virtual_core = dst->core, .addr = l1_base_ + kCounterOffset})
                                           : std::nullopt});
                    if (batch.size() == batch_size || last) {
                        svc.write_batch(batch);
                        batch.clear();
                    }
                    offset += request_bytes;
                }
                const auto t_issued = std::chrono::steady_clock::now();
                wait_for_counter(*dst, 1);
                const auto t1 = std::chrono::steady_clock::now();
                ASSERT_FALSE(HasFailure());
                const double secs = std::chrono::duration<double>(t1 - t0).count();
                log_info(
                    tt::LogTest,
                    "host_rdma bw service: request {:6} B batch {:2}: {:6.3f} GB/s ({:7.1f} K req/s); host {:6.2f} us "
                    "per request",
                    request_bytes,
                    batch_size,
                    total_bytes / secs / 1e9,
                    num_requests / secs / 1e3,
                    std::chrono::duration<double, std::micro>(t_issued - t0).count() / num_requests);
            }
        }
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
