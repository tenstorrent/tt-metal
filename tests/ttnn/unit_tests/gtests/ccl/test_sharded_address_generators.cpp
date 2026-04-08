// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>

#include <tt-metalium/buffer_types.hpp>
#include "gtest/gtest.h"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"

static constexpr std::array<noc_grid_index_t, 8> worker_to_routing_x_wormhole = {1, 2, 3, 4, 6, 7, 8, 9};

static constexpr std::array<noc_grid_index_t, 10> worker_to_routing_y_wormhole = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11};

namespace tt::tt_metal {

struct UnharvestedWormholeWorkerToNocLookup
    : address_generators::WorkerToNocCoordLookup<UnharvestedWormholeWorkerToNocLookup> {
    static constexpr std::array<noc_grid_index_t, 8> worker_to_routing_x = {1, 2, 3, 4, 6, 7, 8, 9};
    static constexpr std::array<noc_grid_index_t, 10> worker_to_routing_y = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11};

    noc_grid_index_t get_noc_x_from_worker_x(noc_grid_index_t worker_x) const {
        // ASSERT worker_x < worker_to_routing_x_wormhole.size()
        return worker_to_routing_x[worker_x];
    }

    noc_grid_index_t get_noc_y_from_worker_y(noc_grid_index_t worker_y) const {
        // ASSERT worker_y < worker_to_routing_y_wormhole.size()
        return worker_to_routing_y[worker_y];
    }
};

static void run_width_sharded_tensor_slice_indexer_get_page_location_test(
    address_generators::WidthShardedAddressGenerator<
        UnharvestedWormholeWorkerToNocLookup,
        address_generators::DeviceWidthShardSpec>& addrgen,

    std::size_t pages_per_shard_y,
    std::size_t pages_per_shard_x,

    std::size_t shard_grid_height,
    std::size_t shard_grid_width,

    std::size_t worker_shard_cores_start_y,
    std::size_t worker_shard_cores_start_x,
    bool is_shard_grid_transposed) {
    std::size_t page_id = 0;
    // Takes a long time to sweep really large numbers so instead stride through the range.
    // Really the only reason to test larger numbers is to catch overflow issues with smaller
    // number types that may be carried around in the addrgen structs
    std::size_t py_increment = pages_per_shard_y > 32 ? 7 : 1;
    std::size_t px_increment = pages_per_shard_x > 32 ? 7 : 1;

    if (!is_shard_grid_transposed) {
        for (std::size_t py = 0; py < pages_per_shard_y; py++) {
            for (std::size_t y_logical = worker_shard_cores_start_y;
                 y_logical < worker_shard_cores_start_y + shard_grid_height;
                 y_logical++) {
                for (std::size_t x_logical = worker_shard_cores_start_x;
                     x_logical < worker_shard_cores_start_x + shard_grid_width;
                     x_logical++) {
                    for (std::size_t px = 0; px < pages_per_shard_x; px++) {
                        if ((px % px_increment == 0 && py % py_increment == 0) ||
                            (py == (pages_per_shard_y - 1) || px != (pages_per_shard_x - 1))) {
                            const auto& result = addrgen.get_page_location(page_id);
                            ASSERT_EQ(result.core_location.noc_x, worker_to_routing_x_wormhole.at(x_logical));
                            ASSERT_EQ(result.core_location.noc_y, worker_to_routing_y_wormhole.at(y_logical));
                            ASSERT_EQ(result.page_offset, px + (py * pages_per_shard_x));

                            const auto& result2 =
                                addrgen.get_page_location_with_contiguous_pages_in_row_in_bank(page_id);
                            ASSERT_EQ(result2.core_location.noc_x, result.core_location.noc_x);
                            ASSERT_EQ(result2.core_location.noc_y, result.core_location.noc_y);
                            ASSERT_EQ(result2.page_offset, result.page_offset);
                            ASSERT_EQ(result2.contig_pages_in_row, pages_per_shard_x - px);
                        }

                        page_id++;
                    }
                }
            }
        }
    } else {
        for (std::size_t py = 0; py < pages_per_shard_y; py++) {
            for (std::size_t x_logical = worker_shard_cores_start_x;
                 x_logical < worker_shard_cores_start_x + shard_grid_width;
                 x_logical++) {
                for (std::size_t y_logical = worker_shard_cores_start_y;
                     y_logical < worker_shard_cores_start_y + shard_grid_height;
                     y_logical++) {
                    for (std::size_t px = 0; px < pages_per_shard_x; px++) {
                        if ((px % px_increment == 0 && py % py_increment == 0) ||
                            (py == (pages_per_shard_y - 1) || px != (pages_per_shard_x - 1))) {
                            const auto& result = addrgen.get_page_location(page_id);
                            ASSERT_EQ(result.core_location.noc_x, worker_to_routing_x_wormhole.at(x_logical));
                            ASSERT_EQ(result.core_location.noc_y, worker_to_routing_y_wormhole.at(y_logical));
                            ASSERT_EQ(result.page_offset, px + (py * pages_per_shard_x));

                            const auto& result2 =
                                addrgen.get_page_location_with_contiguous_pages_in_row_in_bank(page_id);
                            ASSERT_EQ(result2.core_location.noc_x, result.core_location.noc_x);
                            ASSERT_EQ(result2.core_location.noc_y, result.core_location.noc_y);
                            ASSERT_EQ(result2.page_offset, result.page_offset);
                            ASSERT_EQ(result2.contig_pages_in_row, pages_per_shard_x - px);
                        }
                        page_id++;
                    }
                }
            }
        }
    }
}

static void run_width_sharded_tensor_slice_indexer_get_page_location_test(
    std::size_t pages_per_shard_y,
    std::size_t pages_per_shard_x,

    std::size_t shard_grid_height,
    std::size_t shard_grid_width,

    std::size_t worker_shard_cores_start_y,
    std::size_t worker_shard_cores_start_x,

    bool is_shard_grid_transposed) {
    auto addrgen = address_generators::
        WidthShardedAddressGenerator<UnharvestedWormholeWorkerToNocLookup, address_generators::DeviceWidthShardSpec>(
            UnharvestedWormholeWorkerToNocLookup(),
            address_generators::DeviceShardSpecTypeGetter<TensorMemoryLayout::WIDTH_SHARDED>::type(
                pages_per_shard_y,
                pages_per_shard_x,
                shard_grid_height,
                shard_grid_width,
                worker_shard_cores_start_y,
                worker_shard_cores_start_x,
                is_shard_grid_transposed),
            1024,
            0x0);

    run_width_sharded_tensor_slice_indexer_get_page_location_test(
        addrgen,
        pages_per_shard_y,
        pages_per_shard_x,

        shard_grid_height,
        shard_grid_width,

        worker_shard_cores_start_y,
        worker_shard_cores_start_x,

        is_shard_grid_transposed);
}

TEST(CclWidthShardedTensorSliceIndexer_Wormhole, basic_test_case) {
    static constexpr std::size_t pages_per_shard_y = 1;
    static constexpr std::size_t pages_per_shard_x = 8;

    static constexpr std::size_t shard_grid_height = 2;
    static constexpr std::size_t shard_grid_width = 1;

    static constexpr std::size_t worker_shard_cores_start_y = 0;
    static constexpr std::size_t worker_shard_cores_start_x = 0;

    bool is_shard_grid_transposed = false;

    run_width_sharded_tensor_slice_indexer_get_page_location_test(
        pages_per_shard_y,
        pages_per_shard_x,

        shard_grid_height,
        shard_grid_width,

        worker_shard_cores_start_y,
        worker_shard_cores_start_x,

        is_shard_grid_transposed);
}

static void run_height_sharded_tensor_slice_indexer_get_page_location_test(
    address_generators::HeightShardedAddressGenerator<
        UnharvestedWormholeWorkerToNocLookup,
        address_generators::DeviceHeightShardSpec>& addrgen,
    std::size_t pages_per_shard_y,
    std::size_t pages_per_shard_x,

    std::size_t shard_grid_height,
    std::size_t shard_grid_width,

    std::size_t worker_shard_cores_start_y,
    std::size_t worker_shard_cores_start_x,

    bool is_shard_grid_transposed) {
    std::size_t page_id = 0;

    // Takes a long time to sweep really large numbers so instead stride through the range.
    // Really the only reason to test larger numbers is to catch overflow issues with smaller
    // number types that may be carried around in the addrgen structs
    std::size_t py_increment = pages_per_shard_y > 32 ? 7 : 1;
    std::size_t px_increment = pages_per_shard_x > 32 ? 7 : 1;

    if (!is_shard_grid_transposed) {
        for (std::size_t x_logical = worker_shard_cores_start_x;
             x_logical < worker_shard_cores_start_x + shard_grid_width;
             x_logical++) {
            for (std::size_t y_logical = worker_shard_cores_start_y;
                 y_logical < worker_shard_cores_start_y + shard_grid_height;
                 y_logical++) {
                for (std::size_t py = 0; py < pages_per_shard_y; py++) {
                    for (std::size_t px = 0; px < pages_per_shard_x; px++) {
                        if ((px % px_increment == 0 && py % py_increment == 0) ||
                            (py == (pages_per_shard_y - 1) || px != (pages_per_shard_x - 1))) {
                            const auto& result = addrgen.get_page_location(page_id);
                            ASSERT_EQ(result.core_location.noc_x, worker_to_routing_x_wormhole.at(x_logical));
                            ASSERT_EQ(result.core_location.noc_y, worker_to_routing_y_wormhole.at(y_logical));
                            ASSERT_EQ(result.page_offset, px + (py * pages_per_shard_x));
                        }

                        page_id++;
                    }
                }
            }
        }
    } else {
        for (std::size_t y_logical = worker_shard_cores_start_y;
             y_logical < worker_shard_cores_start_y + shard_grid_height;
             y_logical++) {
            for (std::size_t x_logical = worker_shard_cores_start_x;
                 x_logical < worker_shard_cores_start_x + shard_grid_width;
                 x_logical++) {
                for (std::size_t py = 0; py < pages_per_shard_y; py++) {
                    for (std::size_t px = 0; px < pages_per_shard_x; px++) {
                        if ((px % px_increment == 0 && py % py_increment == 0) ||
                            (py == (pages_per_shard_y - 1) || px != (pages_per_shard_x - 1))) {
                            const auto& result = addrgen.get_page_location(page_id);
                            ASSERT_EQ(result.core_location.noc_x, worker_to_routing_x_wormhole.at(x_logical));
                            ASSERT_EQ(result.core_location.noc_y, worker_to_routing_y_wormhole.at(y_logical));
                            ASSERT_EQ(result.page_offset, px + (py * pages_per_shard_x));
                        }
                        page_id++;
                    }
                }
            }
        }
    }
}

static void run_height_sharded_tensor_slice_indexer_get_page_location_test(
    std::size_t pages_per_shard_y,
    std::size_t pages_per_shard_x,

    std::size_t shard_grid_height,
    std::size_t shard_grid_width,

    std::size_t worker_shard_cores_start_y,
    std::size_t worker_shard_cores_start_x,

    bool is_shard_grid_transposed) {
    auto addrgen = address_generators::
        HeightShardedAddressGenerator<UnharvestedWormholeWorkerToNocLookup, address_generators::DeviceHeightShardSpec>(
            UnharvestedWormholeWorkerToNocLookup(),
            address_generators::DeviceShardSpecTypeGetter<TensorMemoryLayout::HEIGHT_SHARDED>::type(
                pages_per_shard_y,
                pages_per_shard_x,
                shard_grid_height,
                shard_grid_width,
                worker_shard_cores_start_y,
                worker_shard_cores_start_x,
                is_shard_grid_transposed),
            1024,
            0x0);

    run_height_sharded_tensor_slice_indexer_get_page_location_test(
        addrgen,
        pages_per_shard_y,
        pages_per_shard_x,
        shard_grid_height,
        shard_grid_width,
        worker_shard_cores_start_y,
        worker_shard_cores_start_x,

        is_shard_grid_transposed);
}

TEST(CclHeightShardedTensorSliceIndexer_Wormhole, basic_test_case) {
    static constexpr std::size_t pages_per_shard_y = 8;
    static constexpr std::size_t pages_per_shard_x = 1;

    static constexpr std::size_t shard_grid_height = 1;
    static constexpr std::size_t shard_grid_width = 2;

    static constexpr std::size_t worker_shard_cores_start_y = 0;
    static constexpr std::size_t worker_shard_cores_start_x = 0;

    bool is_shard_grid_transposed = false;

    run_height_sharded_tensor_slice_indexer_get_page_location_test(
        pages_per_shard_y,
        pages_per_shard_x,

        shard_grid_height,
        shard_grid_width,

        worker_shard_cores_start_y,
        worker_shard_cores_start_x,

        is_shard_grid_transposed);
}

static void run_block_sharded_tensor_slice_indexer_get_page_location_test(
    address_generators::BlockShardedAddressGenerator<
        UnharvestedWormholeWorkerToNocLookup,
        address_generators::DeviceBlockShardSpec>& addrgen,
    std::size_t pages_per_shard_y,
    std::size_t pages_per_shard_x,

    std::size_t shard_grid_height,
    std::size_t shard_grid_width,

    std::size_t worker_shard_cores_start_y,
    std::size_t worker_shard_cores_start_x,

    bool is_shard_grid_transposed) {
    std::size_t page_id = 0;

    // Takes a long time to sweep really large numbers so instead stride through the range.
    // Really the only reason to test larger numbers is to catch overflow issues with smaller
    // number types that may be carried around in the addrgen structs
    std::size_t py_increment = pages_per_shard_y > 32 ? 7 : 1;
    std::size_t px_increment = pages_per_shard_x > 32 ? 7 : 1;

    if (!is_shard_grid_transposed) {
        for (std::size_t y_logical = worker_shard_cores_start_y;
             y_logical < worker_shard_cores_start_y + shard_grid_height;
             y_logical++) {
            for (std::size_t py = 0; py < pages_per_shard_y; py++) {
                for (std::size_t x_logical = worker_shard_cores_start_x;
                     x_logical < worker_shard_cores_start_x + shard_grid_width;
                     x_logical++) {
                    for (std::size_t px = 0; px < pages_per_shard_x; px++) {
                        if ((px % px_increment == 0 && py % py_increment == 0) ||
                            (py == (pages_per_shard_y - 1) || px != (pages_per_shard_x - 1))) {
                            const auto& result = addrgen.get_page_location(page_id);
                            ASSERT_EQ(result.core_location.noc_x, worker_to_routing_x_wormhole.at(x_logical));
                            ASSERT_EQ(result.core_location.noc_y, worker_to_routing_y_wormhole.at(y_logical));
                            ASSERT_EQ(result.page_offset, px + (py * pages_per_shard_x));

                            const auto& result2 =
                                addrgen.get_page_location_with_contiguous_pages_in_row_in_bank(page_id);
                            ASSERT_EQ(result2.core_location.noc_x, result.core_location.noc_x);
                            ASSERT_EQ(result2.core_location.noc_y, result.core_location.noc_y);
                            ASSERT_EQ(result2.page_offset, result.page_offset);
                            ASSERT_EQ(result2.contig_pages_in_row, pages_per_shard_x - px);
                        }

                        page_id++;
                    }
                }
            }
        }
    } else {
        ASSERT_EQ(true, false);  //"Transposed grid not supported in testing yet"
    }
}

static void run_block_sharded_tensor_slice_indexer_get_page_location_test(
    std::size_t pages_per_shard_y,
    std::size_t pages_per_shard_x,

    std::size_t shard_grid_height,
    std::size_t shard_grid_width,

    std::size_t worker_shard_cores_start_y,
    std::size_t worker_shard_cores_start_x,

    bool is_shard_grid_transposed) {
    auto addrgen = address_generators::
        BlockShardedAddressGenerator<UnharvestedWormholeWorkerToNocLookup, address_generators::DeviceBlockShardSpec>(
            UnharvestedWormholeWorkerToNocLookup(),
            address_generators::DeviceShardSpecTypeGetter<TensorMemoryLayout::BLOCK_SHARDED>::type(
                pages_per_shard_y,
                pages_per_shard_x,
                shard_grid_height,
                shard_grid_width,
                worker_shard_cores_start_y,
                worker_shard_cores_start_x,
                is_shard_grid_transposed),
            1024,
            0x0);

    run_block_sharded_tensor_slice_indexer_get_page_location_test(
        addrgen,
        pages_per_shard_y,
        pages_per_shard_x,
        shard_grid_height,
        shard_grid_width,
        worker_shard_cores_start_y,
        worker_shard_cores_start_x,
        is_shard_grid_transposed);
}

TEST(CclBlockShardedTensorSliceIndexer_Wormhole, basic_test_case) {
    static constexpr std::size_t pages_per_shard_y = 8;
    static constexpr std::size_t pages_per_shard_x = 2;

    static constexpr std::size_t shard_grid_height = 3;
    static constexpr std::size_t shard_grid_width = 2;

    static constexpr std::size_t worker_shard_cores_start_y = 0;
    static constexpr std::size_t worker_shard_cores_start_x = 0;

    bool is_shard_grid_transposed = false;

    run_block_sharded_tensor_slice_indexer_get_page_location_test(
        pages_per_shard_y,
        pages_per_shard_x,

        shard_grid_height,
        shard_grid_width,

        worker_shard_cores_start_y,
        worker_shard_cores_start_x,

        is_shard_grid_transposed);
}

TEST(CclShardedTensorAddrGenBuilder, TestBuildWidthSharded) {
    static constexpr std::size_t pages_per_shard_y = 1;
    static constexpr std::size_t pages_per_shard_x = 8;

    static constexpr std::size_t shard_grid_height = 2;
    static constexpr std::size_t shard_grid_width = 1;

    static constexpr std::size_t worker_shard_cores_start_y = 0;
    static constexpr std::size_t worker_shard_cores_start_x = 0;

    bool is_shard_grid_transposed = false;
    auto addrgen = build_sharded_addr_gen<TensorMemoryLayout::WIDTH_SHARDED>(
        UnharvestedWormholeWorkerToNocLookup(),
        address_generators::DeviceShardSpecTypeGetter<TensorMemoryLayout::WIDTH_SHARDED>::type(
            pages_per_shard_y,
            pages_per_shard_x,
            shard_grid_height,
            shard_grid_width,
            worker_shard_cores_start_y,
            worker_shard_cores_start_x,
            is_shard_grid_transposed),
        1024,
        0x0);

    run_width_sharded_tensor_slice_indexer_get_page_location_test(
        addrgen,
        pages_per_shard_y,
        pages_per_shard_x,

        shard_grid_height,
        shard_grid_width,

        worker_shard_cores_start_y,
        worker_shard_cores_start_x,

        is_shard_grid_transposed);
}
TEST(CclShardedTensorAddrGenBuilder, TestBuildHeightSharded) {
    static constexpr std::size_t pages_per_shard_y = 8;
    static constexpr std::size_t pages_per_shard_x = 1;

    static constexpr std::size_t shard_grid_height = 1;
    static constexpr std::size_t shard_grid_width = 2;

    static constexpr std::size_t worker_shard_cores_start_y = 0;
    static constexpr std::size_t worker_shard_cores_start_x = 0;

    bool is_shard_grid_transposed = false;
    auto addrgen = build_sharded_addr_gen<TensorMemoryLayout::HEIGHT_SHARDED>(
        UnharvestedWormholeWorkerToNocLookup(),
        address_generators::DeviceShardSpecTypeGetter<TensorMemoryLayout::HEIGHT_SHARDED>::type(
            pages_per_shard_y,
            pages_per_shard_x,
            shard_grid_height,
            shard_grid_width,
            worker_shard_cores_start_y,
            worker_shard_cores_start_x,
            is_shard_grid_transposed),
        1024,
        0x0);

    run_height_sharded_tensor_slice_indexer_get_page_location_test(
        addrgen,
        pages_per_shard_y,
        pages_per_shard_x,

        shard_grid_height,
        shard_grid_width,

        worker_shard_cores_start_y,
        worker_shard_cores_start_x,

        is_shard_grid_transposed);
}
TEST(CclShardedTensorAddrGenBuilder, TestBuildBlockSharded) {
    static constexpr std::size_t pages_per_shard_y = 8;
    static constexpr std::size_t pages_per_shard_x = 2;

    static constexpr std::size_t shard_grid_height = 3;
    static constexpr std::size_t shard_grid_width = 2;

    static constexpr std::size_t worker_shard_cores_start_y = 0;
    static constexpr std::size_t worker_shard_cores_start_x = 0;

    bool is_shard_grid_transposed = false;

    auto addrgen = build_sharded_addr_gen<TensorMemoryLayout::BLOCK_SHARDED>(
        UnharvestedWormholeWorkerToNocLookup(),
        address_generators::DeviceShardSpecTypeGetter<TensorMemoryLayout::BLOCK_SHARDED>::type(
            pages_per_shard_y,
            pages_per_shard_x,
            shard_grid_height,
            shard_grid_width,
            worker_shard_cores_start_y,
            worker_shard_cores_start_x,
            is_shard_grid_transposed),
        1024,
        0x0);

    run_block_sharded_tensor_slice_indexer_get_page_location_test(
        addrgen,
        pages_per_shard_y,
        pages_per_shard_x,

        shard_grid_height,
        shard_grid_width,

        worker_shard_cores_start_y,
        worker_shard_cores_start_x,

        is_shard_grid_transposed);
}

}  // namespace tt::tt_metal
