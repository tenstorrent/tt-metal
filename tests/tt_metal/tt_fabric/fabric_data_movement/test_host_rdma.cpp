// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-initiated RDMA over TT-Fabric, single domain: the host posts write requests to a chip's request service, whose
// Tensix kernel reads the data, forms the packets and injects them into a router's host_rdma channel
// (TT_METAL_ENABLE_FABRIC_HOST_RDMA); each write's last packet bumps a completion counter on the destination chip (the
// self-domain ack).

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

// Where writes land on a destination chip: L1 of logical worker core (0, 0), past the reserved region. The completion
// counter sits at the base, the data 64 B after it.
constexpr uint32_t kCounterOffset = 0;
constexpr uint32_t kDataOffset = 64;
// Where the tests stage a write's source on the source chip (past any destination window).
constexpr uint32_t kSrcOffset = 768 * 1024;

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
        // Drained first: stopping a service does not wait for its traffic.
        for (auto& [chip, svc] : services_) {
            svc->drain();
        }
        services_.clear();  // stops the service kernels, which close their router connections
        chips_.clear();
    }

protected:
    struct Chip {
        ChipId chip;
        FabricNodeId node;
        tt::tt_metal::CoreCoord core;  // virtual coordinates of the destination worker core
    };

    // The request service of `chip`, on logical core (1, 1) (not (0, 0), which holds the tests' source and
    // destination data).
    tt::tt_fabric::host_rdma::HostRdmaService& service(const Chip& chip) {
        auto it = services_.find(chip.chip);
        if (it == services_.end()) {
            it = services_
                     .emplace(
                         chip.chip,
                         tt::tt_fabric::host_rdma::HostRdmaService::create(
                             devices_map_.at(chip.chip), tt::tt_metal::CoreCoord{1, 1}))
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

    // Waits for the completion counter on `chip` to reach `expected`.
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
        FAIL() << "Completion counter on chip " << chip.chip << " is " << value << ", expected " << expected;
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

    // One write from `src` to `dst` of `num_words` words, then check the counter and the data.
    void write_and_check(Chip& src, const Chip& dst, size_t num_words, uint32_t seed) {
        reset_destination(dst, num_words);
        const auto data = pattern(num_words, seed);
        const Completion completion{.virtual_core = dst.core, .addr = l1_base_ + kCounterOffset, .increment = 1};
        // The source is on the source chip: the service reads it there.
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

    // Workers see the fabric as they would without host_rdma: one worker sender channel per routing plane. The
    // connection table every worker core reads (written at device init) holds exactly one entry per active router
    // (one per routing plane in each direction), and each names the router's sender channel 0, never its host_rdma
    // channel.
    void check_one_worker_channel_per_routing_plane() {
        auto& context = tt::tt_metal::MetalContext::instance();
        const auto& control_plane = context.get_control_plane();
        const auto& cluster = context.get_cluster();
        const auto& edm_config = control_plane.get_fabric_context().get_builder_context().get_fabric_router_config();
        const auto table_addr = context.hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS);
        size_t routers = 0;
        for (const auto& chip : chips_) {
            // Any worker core: every one holds the same table.
            const auto worker = devices_map_.at(chip.chip)->worker_core_from_logical_core({2, 2});
            tt::tt_fabric::tensix_fabric_connections_l1_info_t table{};
            cluster.read_core(&table, sizeof(table), tt_cxy_pair(chip.chip, worker), table_addr);
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

    // For every ordered pair of chips (optionally neighbours only), writes of each size.
    void run_a_to_b(bool neighbours_only) {
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        const size_t max_payload_words = [&] {
            const auto& fabric_context = control_plane.get_fabric_context();
            return fabric_context.get_fabric_max_payload_size_bytes() / sizeof(uint32_t);
        }();
        // One word; under one packet; exactly one packet; several packets with a partial last one.
        const std::vector<size_t> sizes = {4, 256, max_payload_words, 5 * max_payload_words + 12};
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

    // All-gather: chip i holds shard i; the host writes every shard to every other chip at slot i, each write
    // completing on the destination's counter. Every chip ends up with every shard.
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
                // A chip's own shard is local: the host puts it there directly.
                write_l1(*group[d], l1_base_ + kDataOffset + d * shard_words * sizeof(uint32_t), shards[d]);
            }
            // Issue every write before waiting on any, so the writes of every source are in flight together; each
            // source chip's writes go to its service as one batch.
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
                    // Chip s's own slot holds its shard: its service copies it to the same slot on chip d.
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

// With the host_rdma channel built, a worker still sees one worker sender channel per routing plane.
TEST_F(HostRdmaFabric2DFixture, OneWorkerChannelPerRoutingPlane) { check_one_worker_channel_per_routing_plane(); }

// Host-injection bandwidth from one chip to a neighbour (disabled by default: a benchmark, not a check). Run with
// --gtest_also_run_disabled_tests --gtest_filter=*HostRdma*Bandwidth*.
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
    // The request service: requests of each size from a source window in the source chip's L1, round-robin over a
    // destination window, the last with a completion. The service reads, packetizes and injects; the host only posts.
    {
        constexpr uint32_t window_bytes = 256 * 1024;
        constexpr size_t total_bytes = 64ull * 1024 * 1024;
        write_l1(src, l1_base_ + kSrcOffset, pattern(window_bytes / sizeof(uint32_t), 11));
        auto& svc = service(src);
        // Warm-up, untimed: the service's kernel starts and opens its router connection, and the host's posting path
        // warms up (without it the first measurement reads about 40% low).
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
