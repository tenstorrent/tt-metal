// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Placement of the prefill ops on a [RING, RING] 4x4 mesh that two hosts share (mesh_host_rank 0 holds rows 0-1, rank 1
// rows 2-3): every rank must end up with the same placement for every chip, although it can read the physical layout of
// its own chips only. Run with two ranks, e.g. on a mock cluster:
//   tt-run --mock-cluster-rank-binding <mapping.yaml> --rank-binding <binding.yaml> unit_tests_dual_rank_4x4_ring
// A single rank (one Galaxy, same mesh shape) runs the same tests without any exchange.

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <vector>

#include <internal/mesh_device.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/operations/ccl/common/host/mesh_chip_exchange.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_placement.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/dispatch_fabric2d/device/dispatch_fabric2d_placement.hpp"

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::MeshDeviceFixtureBase;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshShape;
using namespace ttnn::operations::ccl::common;
namespace dispatch = ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d;
namespace combine = ttnn::operations::experimental::deepseek_prefill::combine_fabric2d;

class RingRing4x4Fixture : public MeshDeviceFixtureBase {
protected:
    RingRing4x4Fixture() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = MeshShape{4, 4},
            .num_cqs = 1,
            .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY}) {}

    bool is_local(const MeshCoordinate& coord) const { return mesh_device_->is_local(coord); }

    // Every rank's copy of `mine`, concatenated in rank order. Without a context, just `mine`.
    std::vector<uint32_t> gather_over_ranks(std::vector<uint32_t> mine) const {
        const auto context = mesh_distributed_context(*mesh_device_);
        if (context == nullptr) {
            return mine;
        }
        std::vector<uint32_t> all(mine.size() * *context->size());
        context->all_gather(
            ttsl::as_writable_bytes(ttsl::Span<uint32_t>(mine)), ttsl::as_writable_bytes(ttsl::Span<uint32_t>(all)));
        return all;
    }

    void expect_same_on_every_rank(const std::vector<uint32_t>& mine) const {
        const auto all = gather_over_ranks(mine);
        ASSERT_EQ(all.size() % mine.size(), 0u);
        for (std::size_t rank = 0; rank * mine.size() < all.size(); ++rank) {
            EXPECT_TRUE(std::equal(mine.begin(), mine.end(), all.begin() + rank * mine.size()))
                << "rank " << rank << " computed a different placement than this rank";
        }
    }

    CoreRangeSet whole_grid() const {
        const auto grid = mesh_device_->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}));
    }
};

void append_core(std::vector<uint32_t>& words, const CoreCoord& core) {
    words.push_back(static_cast<uint32_t>(core.x));
    words.push_back(static_cast<uint32_t>(core.y));
}

void append_coord(std::vector<uint32_t>& words, const MeshCoordinate& coord) {
    words.insert(words.end(), coord.coords().begin(), coord.coords().end());
}

template <class Streams>
void append_streams(std::vector<uint32_t>& words, const Streams& streams) {
    for (const auto& [stream, placement] : streams) {
        words.push_back(stream);
        append_core(words, placement.worker_logical);
        append_core(words, placement.worker_virtual);
        append_coord(words, placement.downstream_coord);
        words.push_back(placement.downstream_node.chip_id);
    }
}

TEST_F(RingRing4x4Fixture, ExchangeGivesEveryRankEveryChip) {
    const auto& shape = mesh_device_->shape();
    std::size_t calls = 0;
    std::size_t local = 0;
    for (const auto& coord : MeshCoordinateRange(shape)) {
        local += is_local(coord) ? 1 : 0;
    }
    const auto records = exchange_chip_records(
        *mesh_device_, mesh_distributed_context(*mesh_device_), 2, [&calls, &shape](const MeshCoordinate& coord) {
            ++calls;
            const auto index = static_cast<uint32_t>(coord.to_linear_index(shape));
            return std::vector<uint32_t>{index, index * 7 + 1};
        });
    EXPECT_EQ(calls, local) << "a chip's record must be computed by its owner only";
    ASSERT_EQ(records.size(), shape.mesh_size());
    std::vector<uint32_t> flat;
    for (std::size_t index = 0; index < records.size(); ++index) {
        ASSERT_EQ(records[index].size(), 2u);
        EXPECT_EQ(records[index][0], index);
        EXPECT_EQ(records[index][1], index * 7 + 1);
        flat.insert(flat.end(), records[index].begin(), records[index].end());
    }
    expect_same_on_every_rank(flat);
}

TEST_F(RingRing4x4Fixture, ExchangeIsCachedPerConfig) {
    std::size_t calls = 0;
    const auto compute = [&calls](const MeshCoordinate&) {
        ++calls;
        return std::vector<uint32_t>{5};
    };
    const auto context = mesh_distributed_context(*mesh_device_);
    const auto first = exchange_chip_records(*mesh_device_, context, 1, compute, "cache-test");
    const std::size_t calls_after_first = calls;
    const auto second = exchange_chip_records(*mesh_device_, context, 1, compute, "cache-test");
    EXPECT_EQ(calls, calls_after_first) << "the second call must not recompute";
    EXPECT_EQ(first, second);
    exchange_chip_records(*mesh_device_, context, 1, compute, "cache-test-other");
    EXPECT_GT(calls, calls_after_first) << "a different config must not hit the cache";
}

TEST_F(RingRing4x4Fixture, DispatchPlacementAgreesAcrossRanks) {
    for (const uint32_t axis : {0u, 1u}) {
        for (const uint32_t num_links : {1u, 2u}) {
            const auto placement = dispatch::decide_placement(mesh_device_.get(), axis, num_links, whole_grid());
            ASSERT_EQ(placement.size(), mesh_device_->shape().mesh_size());
            std::vector<uint32_t> words;
            for (const auto& [coord, streams] : placement) {
                ASSERT_EQ(streams.size(), dispatch::stream_count(num_links));
                std::set<CoreCoord> cores;
                for (const auto& [stream, stream_placement] : streams) {
                    EXPECT_TRUE(cores.insert(stream_placement.worker_logical).second)
                        << "two streams of chip " << coord << " share a core";
                    if (is_local(coord)) {
                        auto* device = tt::tt_metal::internal::get_device(*mesh_device_, coord);
                        EXPECT_EQ(
                            stream_placement.worker_virtual,
                            device->virtual_core_from_logical_core(
                                stream_placement.worker_logical, tt::CoreType::WORKER));
                    }
                    EXPECT_EQ(
                        stream_placement.downstream_node,
                        mesh_device_->get_fabric_node_id(stream_placement.downstream_coord));
                }
                append_coord(words, coord);
                append_streams(words, streams);
            }
            expect_same_on_every_rank(words);
        }
    }
}

TEST_F(RingRing4x4Fixture, CombinePlacementAgreesAcrossRanks) {
    for (const uint32_t axis : {0u, 1u}) {
        for (const uint32_t num_links : {1u, 2u}) {
            for (const bool with_collector : {false, true}) {
                const auto placement = combine::decide_placement(
                    mesh_device_.get(), axis, num_links, combine::DEFAULT_UNTILIZERS_PER_GROUP, with_collector);
                ASSERT_EQ(placement.size(), mesh_device_->shape().mesh_size());
                std::vector<uint32_t> words;
                for (const auto& [coord, device_placement] : placement) {
                    ASSERT_EQ(device_placement.streams.size(), combine::stream_count(num_links));
                    EXPECT_EQ(device_placement.collector.has_value(), with_collector);
                    if (is_local(coord)) {
                        auto* device = tt::tt_metal::internal::get_device(*mesh_device_, coord);
                        for (const auto& [stream, stream_placement] : device_placement.streams) {
                            EXPECT_EQ(
                                stream_placement.worker_virtual,
                                device->virtual_core_from_logical_core(
                                    stream_placement.worker_logical, tt::CoreType::WORKER));
                        }
                        for (const auto& group : device_placement.untilizers) {
                            EXPECT_EQ(group.size(), combine::DEFAULT_UNTILIZERS_PER_GROUP);
                            for (const auto& untilizer : group) {
                                EXPECT_EQ(
                                    untilizer.worker_virtual,
                                    device->virtual_core_from_logical_core(untilizer.logical, tt::CoreType::WORKER));
                            }
                        }
                    }
                    append_coord(words, coord);
                    append_streams(words, device_placement.streams);
                    for (const auto& group : device_placement.untilizers) {
                        for (const auto& untilizer : group) {
                            append_core(words, untilizer.logical);
                            append_core(words, untilizer.worker_virtual);
                        }
                    }
                    append_core(words, device_placement.collector.value_or(CoreCoord{0, 0}));
                }
                expect_same_on_every_rank(words);
            }
        }
    }
}

}  // namespace
