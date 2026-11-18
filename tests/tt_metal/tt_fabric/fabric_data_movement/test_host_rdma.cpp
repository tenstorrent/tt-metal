// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-initiated RDMA over TT-Fabric, single domain: the host writes packets straight into a router's host_rdma
// channel (TT_METAL_ENABLE_FABRIC_HOST_RDMA) and is credited in pinned host memory; each write's last packet bumps a
// completion counter on the destination chip (the self-domain ack). No kernels run: the host is the only producer.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <thread>
#include <utility>
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
using tt::tt_fabric::host_rdma::HostRdmaChannel;
using tt::tt_fabric::host_rdma::PinnedHostMemoryTarget;
using tt::tt_fabric::host_rdma::UmdDmaDeviceWriter;
using tt::tt_fabric::host_rdma::WriteRequest;

// Where writes land on a destination chip: L1 of logical worker core (0, 0), past the reserved region. The completion
// counter sits at the base, the data 64 B after it.
constexpr uint32_t kCounterOffset = 0;
constexpr uint32_t kDataOffset = 64;
// Where the service tests stage a write's source on the source chip (past any destination window).
constexpr uint32_t kSrcOffset = 768 * 1024;
// Words of credit memory per source chip: one per router channel the host opens there.
constexpr size_t kCreditWordsPerChip = 64;

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
        writer_ = std::make_shared<UmdDmaDeviceWriter>(context.get_cluster());
        const auto& control_plane = context.get_control_plane();
        l1_base_ = static_cast<uint32_t>(context.hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED));
        for (const auto& [chip, mesh_device] : devices_map_) {
            const FabricNodeId node = control_plane.get_fabric_node_id_from_physical_chip_id(chip);
            chips_.push_back(Chip{
                .chip = chip,
                .node = node,
                .core = mesh_device->worker_core_from_logical_core({0, 0}),
                .credits = std::make_shared<PinnedHostMemoryTarget>(*mesh_device, kCreditWordsPerChip)});
        }
    }

    void TearDown() override {
        channels_.clear();  // drains and withdraws every channel's credit target
        services_.clear();  // drains and stops the service kernels
        chips_.clear();
        writer_.reset();
    }

protected:
    struct Chip {
        ChipId chip;
        FabricNodeId node;
        tt::tt_metal::CoreCoord core;  // virtual coordinates of the destination worker core
        std::shared_ptr<PinnedHostMemoryTarget> credits;
        size_t next_credit_word = 0;
    };

    // The channel from `src` toward `dst`: one per router, opened on first use.
    HostRdmaChannel& channel(Chip& src, const Chip& dst) {
        const chan_id_t eth_chan = HostRdmaChannel::eth_chan_toward(src.node, dst.node);
        auto key = std::make_pair(src.chip, eth_chan);
        auto it = channels_.find(key);
        if (it == channels_.end()) {
            TT_FATAL(src.next_credit_word < kCreditWordsPerChip, "Out of credit words on chip {}", src.chip);
            auto opened =
                std::make_unique<HostRdmaChannel>(src.node, eth_chan, writer_, src.credits, src.next_credit_word++);
            it = channels_.emplace(key, std::move(opened)).first;
        }
        return *it->second;
    }

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

    static std::span<const std::byte> bytes(const std::vector<uint32_t>& words) {
        return std::as_bytes(std::span<const uint32_t>(words));
    }

    // One write from `src` to `dst` of `num_words` words, then check the counter and the data.
    void write_and_check(Chip& src, const Chip& dst, size_t num_words, uint32_t seed) {
        reset_destination(dst, num_words);
        const auto data = pattern(num_words, seed);
        const Completion completion{.virtual_core = dst.core, .addr = l1_base_ + kCounterOffset, .increment = 1};
        if (use_service_) {
            // The source is on the source chip: the service reads it there.
            write_l1(src, l1_base_ + kSrcOffset, data);
            service(src).write(WriteRequest{
                .src = {src.core, l1_base_ + kSrcOffset},
                .dst = dst.node,
                .dst_at = {dst.core, l1_base_ + kDataOffset},
                .size = static_cast<uint32_t>(num_words * sizeof(uint32_t)),
                .completion = completion});
        } else {
            channel(src, dst).write(dst.node, dst.core, l1_base_ + kDataOffset, bytes(data), completion);
        }
        wait_for_counter(dst, 1);
        EXPECT_EQ(read_l1(dst, l1_base_ + kDataOffset, num_words), data)
            << "chip " << src.chip << " -> chip " << dst.chip << ", " << num_words * 4 << " B";
    }

    // Whether a write from `src` can reach `dst`: on 2D, any chip; on 1D, a chip on the same mesh row or column (a 1D
    // fabric runs lines, and a 1D header counts hops along one).
    static bool reachable(FabricNodeId src, FabricNodeId dst) {
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        if (control_plane.get_fabric_context().is_2D_routing_enabled()) {
            return true;
        }
        const auto& mesh_graph = control_plane.get_mesh_graph();
        const auto a = mesh_graph.chip_to_coordinate(src.mesh_id, src.chip_id);
        const auto b = mesh_graph.chip_to_coordinate(dst.mesh_id, dst.chip_id);
        return src.mesh_id == dst.mesh_id && (a[0] == b[0] || a[1] == b[1]);
    }

    // Chips crossed from `src` to `dst` on the route the host_rdma channel takes.
    static uint32_t hops(FabricNodeId src, FabricNodeId dst) {
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        uint32_t count = 0;
        FabricNodeId previous = src;
        for (const auto& [node, chan] :
             control_plane.get_fabric_route(src, dst, HostRdmaChannel::eth_chan_toward(src, dst))) {
            if (node != previous) {
                count++;
                previous = node;
            }
        }
        return count;
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
                if (!reachable(src.node, dst.node) || (neighbours_only && hops(src.node, dst.node) != 1)) {
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
        // The participants: every chip on 2D; on 1D, the longer of the mesh row and column through the first chip.
        std::vector<Chip*> group;
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        if (control_plane.get_fabric_context().is_2D_routing_enabled()) {
            for (auto& c : chips_) {
                group.push_back(&c);
            }
        } else {
            const auto& mesh_graph = control_plane.get_mesh_graph();
            const auto& first = chips_.front().node;
            const auto origin = mesh_graph.chip_to_coordinate(first.mesh_id, first.chip_id);
            for (const size_t axis : {0u, 1u}) {
                std::vector<Chip*> line;
                for (auto& c : chips_) {
                    const auto at = mesh_graph.chip_to_coordinate(c.node.mesh_id, c.node.chip_id);
                    if (c.node.mesh_id == first.mesh_id && at[axis] == origin[axis]) {
                        line.push_back(&c);
                    }
                }
                if (line.size() > group.size()) {
                    group = std::move(line);
                }
            }
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
            // Issue every write before waiting on any, so the writes of every source are in flight together. Through
            // the services, each source chip's writes go as one batch.
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
                    if (use_service_) {
                        // Chip s's own slot holds its shard: its service copies it to the same slot on chip d.
                        batch.push_back(WriteRequest{
                            .src = {group[s]->core, slot},
                            .dst = group[d]->node,
                            .dst_at = {group[d]->core, slot},
                            .size = static_cast<uint32_t>(shard_words * sizeof(uint32_t)),
                            .completion = completion});
                    } else {
                        channel(*group[s], *group[d])
                            .write(group[d]->node, group[d]->core, slot, bytes(shards[s]), completion);
                    }
                }
                if (use_service_) {
                    service(*group[s]).write_batch(batch);
                }
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

    std::shared_ptr<UmdDmaDeviceWriter> writer_;
    std::vector<Chip> chips_;
    std::map<std::pair<ChipId, chan_id_t>, std::unique_ptr<HostRdmaChannel>> channels_;
    uint32_t l1_base_ = 0;
    bool use_service_ = false;  // writes go through the chips' request services (else the host pushes packets)
    std::map<ChipId, std::shared_ptr<tt::tt_fabric::host_rdma::HostRdmaService>> services_;
};

using HostRdmaFabric2DFixture = HostRdmaFixture<tt_fabric::FabricConfig::FABRIC_2D>;
using HostRdmaFabric1DFixture = HostRdmaFixture<tt_fabric::FabricConfig::FABRIC_1D>;

}  // namespace

TEST_F(HostRdmaFabric2DFixture, AToBNeighbours) { run_a_to_b(/*neighbours_only=*/true); }
TEST_F(HostRdmaFabric2DFixture, AToBAllPairs) { run_a_to_b(/*neighbours_only=*/false); }
TEST_F(HostRdmaFabric2DFixture, AllGather) { run_all_gather(/*shard_words=*/4096, /*rounds=*/3); }

TEST_F(HostRdmaFabric1DFixture, AToBAllPairs) { run_a_to_b(/*neighbours_only=*/false); }
TEST_F(HostRdmaFabric1DFixture, AllGather) { run_all_gather(/*shard_words=*/4096, /*rounds=*/3); }

// The same through the chips' request services: the host posts requests, a Tensix core on the source chip reads the
// data, forms the packets and injects them.
TEST_F(HostRdmaFabric2DFixture, ServiceAToBAllPairs) {
    use_service_ = true;
    run_a_to_b(/*neighbours_only=*/false);
}
TEST_F(HostRdmaFabric2DFixture, ServiceAllGather) {
    use_service_ = true;
    run_all_gather(/*shard_words=*/4096, /*rounds=*/3);
}
TEST_F(HostRdmaFabric1DFixture, ServiceAToBAllPairs) {
    use_service_ = true;
    run_a_to_b(/*neighbours_only=*/false);
}
TEST_F(HostRdmaFabric1DFixture, ServiceAllGather) {
    use_service_ = true;
    run_all_gather(/*shard_words=*/4096, /*rounds=*/3);
}

namespace {

// A DeviceWriter that times each kind of call of the writer it wraps (the benchmark's breakdown of the host's cost).
class TimedDeviceWriter final : public tt::tt_fabric::host_rdma::DeviceWriter {
public:
    explicit TimedDeviceWriter(std::shared_ptr<DeviceWriter> inner) : inner_(std::move(inner)) {}

    void write(
        ChipId chip, const tt::tt_metal::CoreCoord& core, uint64_t addr, std::span<const std::byte> data) override {
        const auto t = Clock::now();
        inner_->write(chip, core, addr, data);
        write_ns += elapsed(t);
        writes++;
    }
    void fence(ChipId chip, const tt::tt_metal::CoreCoord& core) override {
        const auto t = Clock::now();
        inner_->fence(chip, core);
        fence_ns += elapsed(t);
    }
    void write_word(ChipId chip, const tt::tt_metal::CoreCoord& core, uint64_t addr, uint32_t value) override {
        const auto t = Clock::now();
        inner_->write_word(chip, core, addr, value);
        word_ns += elapsed(t);
    }
    uint32_t read_word(ChipId chip, const tt::tt_metal::CoreCoord& core, uint64_t addr) override {
        return inner_->read_word(chip, core, addr);
    }

    void reset() { write_ns = fence_ns = word_ns = writes = 0; }

    uint64_t write_ns = 0, fence_ns = 0, word_ns = 0, writes = 0;

private:
    using Clock = std::chrono::steady_clock;
    static uint64_t elapsed(Clock::time_point t) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count();
    }
    std::shared_ptr<DeviceWriter> inner_;
};

}  // namespace

// The host's write paths into device L1, measured raw (no fabric): UMD PCIe DMA, MMIO (Relaxed and Strict ordering)
// and the L1 barrier, by transfer size, into an Ethernet core (a router's idle host_rdma slot ring) and a worker core.
// Run with --gtest_also_run_disabled_tests.
TEST_F(HostRdmaFabric2DFixture, DISABLED_DmaPath) {
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    const auto& driver = cluster.get_driver();
    auto& src = chips_[0];
    const Chip* dst = nullptr;
    for (const auto& c : chips_) {
        if (c.chip != src.chip && hops(src.node, c.node) == 1) {
            dst = &c;
            break;
        }
    }
    ASSERT_NE(dst, nullptr);
    // Opening the channel gives the slot ring's layout; nothing is rung in, so the router never reads the slots.
    auto& ch = channel(src, *dst);
    const auto eth_core = cluster.get_virtual_eth_core_from_channel(src.chip, ch.eth_chan());
    const auto& builder_context = context.get_control_plane().get_fabric_context().get_builder_context();
    const auto& layout = builder_context.get_router_channel_layout(src.node, ch.eth_chan());
    const uint64_t eth_addr = layout.senders[0].back().base_address;
    const size_t eth_bytes = static_cast<size_t>(layout.senders[0].back().num_slots) *
                             builder_context.get_fabric_router_config().channel_buffer_size_bytes;
    const uint64_t worker_addr = l1_base_;

    struct Target {
        const char* name;
        tt::tt_metal::CoreCoord core;
        uint64_t addr;
        size_t max_bytes;
    };
    const std::vector<Target> targets = {
        {"eth", eth_core, eth_addr, eth_bytes}, {"worker", src.core, worker_addr, 1024 * 1024}};
    std::vector<std::byte> buffer(1024 * 1024);
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = static_cast<std::byte>(i * 131);
    }

    using Clock = std::chrono::steady_clock;
    const auto& soc = cluster.get_soc_desc(src.chip);
    for (const auto& target : targets) {
        const tt::umd::CoreCoord umd_core = soc.get_coord_at(target.core, CoordSystem::TRANSLATED);
        for (size_t size = 64; size <= target.max_bytes; size *= 4) {
            const int iters = size <= 16384 ? 2000 : (size <= 262144 ? 200 : 50);
            const auto time_us = [&](auto&& op) {
                op();  // warm
                const auto t = Clock::now();
                for (int i = 0; i < iters; ++i) {
                    op();
                }
                return std::chrono::duration<double, std::micro>(Clock::now() - t).count() / iters;
            };
            const double dma =
                time_us([&] { driver->dma_write_to_device(buffer.data(), size, src.chip, umd_core, target.addr); });
            const double mmio_relaxed = time_us([&] {
                driver->write_to_device(
                    buffer.data(), size, src.chip, umd_core, target.addr, tt::umd::IoOrdering::Relaxed);
            });
            const double mmio_strict = time_us([&] {
                driver->write_to_device(
                    buffer.data(), size, src.chip, umd_core, target.addr, tt::umd::IoOrdering::Strict);
            });
            const double fence = time_us([&] { cluster.l1_barrier(src.chip, {target.core}); });
            const double dma_fence = time_us([&] {
                driver->dma_write_to_device(buffer.data(), size, src.chip, umd_core, target.addr);
                cluster.l1_barrier(src.chip, {target.core});
            });
            const double relaxed_fence = time_us([&] {
                driver->write_to_device(
                    buffer.data(), size, src.chip, umd_core, target.addr, tt::umd::IoOrdering::Relaxed);
                cluster.l1_barrier(src.chip, {target.core});
            });
            const auto gbps = [&](double us) { return size / (us * 1e3); };
            log_info(
                tt::LogTest,
                "dma path {:6} {:8} B: dma {:8.2f} us ({:5.2f} GB/s) | mmio relaxed {:8.2f} us ({:5.2f}) | mmio strict "
                "{:8.2f} us ({:5.2f}) | l1 barrier {:6.2f} us | dma+barrier {:8.2f} us ({:5.2f}) | relaxed+barrier "
                "{:8.2f} us ({:5.2f})",
                target.name,
                size,
                dma,
                gbps(dma),
                mmio_relaxed,
                gbps(mmio_relaxed),
                mmio_strict,
                gbps(mmio_strict),
                fence,
                dma_fence,
                gbps(dma_fence),
                relaxed_fence,
                gbps(relaxed_fence));
        }
    }
}

// Single-hop host-injected bandwidth, by packet payload size: a stream of packet-sized writes into a window of the
// neighbour's L1, timed from the first write until the last write's completion lands. Run with
// --gtest_also_run_disabled_tests.
TEST_F(HostRdmaFabric2DFixture, DISABLED_Bandwidth) {
    auto& src = chips_[0];
    const Chip* dst = nullptr;
    for (const auto& c : chips_) {
        if (c.chip != src.chip && hops(src.node, c.node) == 1) {
            dst = &c;
            break;
        }
    }
    ASSERT_NE(dst, nullptr);
    {
        auto timed = std::make_shared<TimedDeviceWriter>(writer_);
        HostRdmaChannel ch(
            src.node,
            HostRdmaChannel::eth_chan_toward(src.node, dst->node),
            timed,
            src.credits,
            src.next_credit_word++,
            tt::tt_fabric::host_rdma::host_push_transport(timed));

        constexpr size_t window_bytes = 256 * 1024;
        constexpr size_t total_bytes = 64ull * 1024 * 1024;
        // (payload, batch) pairs: every payload at the default batch, then the largest payload at each batch size.
        std::vector<std::pair<size_t, uint32_t>> runs;
        for (const size_t p : {size_t{256}, size_t{1024}, size_t{2048}, ch.max_payload_bytes()}) {
            runs.emplace_back(p, ch.max_batch_packets());
        }
        for (const uint32_t b : {1u, 4u, 8u, ch.num_slots()}) {
            runs.emplace_back(ch.max_payload_bytes(), b);
        }
        const auto data = pattern(window_bytes / sizeof(uint32_t), 7);
        const auto all = bytes(data);

        for (const auto& [payload, batch] : runs) {
            ch.set_max_batch_packets(batch);
            reset_destination(*dst, 1);
            timed->reset();
            const size_t num_packets = total_bytes / payload;
            const auto t0 = std::chrono::steady_clock::now();
            size_t offset = 0;
            for (size_t i = 0; i < num_packets; ++i) {
                if (offset + payload > window_bytes) {
                    offset = 0;
                }
                const bool last = i + 1 == num_packets;
                ch.write(
                    dst->node,
                    dst->core,
                    l1_base_ + kDataOffset + offset,
                    all.subspan(offset, payload),
                    last ? std::optional<Completion>(
                               Completion{.virtual_core = dst->core, .addr = l1_base_ + kCounterOffset})
                         : std::nullopt);
                offset += payload;
            }
            const auto t_issued = std::chrono::steady_clock::now();
            wait_for_counter(*dst, 1);
            const auto t1 = std::chrono::steady_clock::now();
            ASSERT_FALSE(HasFailure());

            const double secs = std::chrono::duration<double>(t1 - t0).count();
            const double issue_secs = std::chrono::duration<double>(t_issued - t0).count();
            const double n = static_cast<double>(num_packets);
            log_info(
                tt::LogTest,
                "host_rdma bw {}: payload {:5} B batch {:2}: {:6.3f} GB/s ({:7.0f} K pkt/s); per packet: {:5.2f} us "
                "total, "
                "write {:5.2f} us, fence {:5.2f} us, doorbell {:5.2f} us, credit wait + staging {:5.2f} us",
                "push",
                payload,
                batch,
                total_bytes / secs / 1e9,
                num_packets / secs / 1e3,
                issue_secs / n * 1e6,
                timed->write_ns / n / 1e3,
                timed->fence_ns / n / 1e3,
                timed->word_ns / n / 1e3,
                (issue_secs * 1e9 - timed->write_ns - timed->fence_ns - timed->word_ns) / n / 1e3);
        }
    }

    // The request service: requests of each size from a source window in the source chip's L1, round-robin over a
    // destination window, the last with a completion. The service reads, packetizes and injects; the host only posts.
    {
        constexpr uint32_t window_bytes = 256 * 1024;
        constexpr size_t total_bytes = 64ull * 1024 * 1024;
        write_l1(src, l1_base_ + kSrcOffset, pattern(window_bytes / sizeof(uint32_t), 11));
        auto& svc = service(src);
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
