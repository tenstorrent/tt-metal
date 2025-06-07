// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/impl/dispatch/ringbuffer_cache.hpp"
#include <memory>
#include <numeric>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <random>

namespace tt::tt_metal {
class RingbufferCacheTestFixture : public ::testing::Test {
protected:
    RingbufferCacheTestFixture() = default;
    ~RingbufferCacheTestFixture() override = default;

    // Define cache of size 256 blocks, one block of size 4 unsigned.
    constexpr static size_t cache_block_sizeB = 4;
    constexpr static size_t cache_size_blocks = 256;
    constexpr static size_t initial_manager_size = 64;  // initial size of the manager array, less than cache size
    std::unique_ptr<RingbufferCacheManager> rbCache;

    // This function is called before each test in this test case.
    void SetUp() override {
        rbCache = std::make_unique<RingbufferCacheManager>(cache_block_sizeB, cache_size_blocks, initial_manager_size);
    }

    // This function is called after each test in this test case.
    void TearDown() override {}

    // define accessors to the private members of the RingbufferCacheManager
    auto get_next_block_offset() const { return rbCache->manager_.next_block_offset; }
    auto get_oldest_idx() const { return rbCache->manager_.oldest_idx; }
    auto get_next_idx() const { return rbCache->manager_.next_idx; }
    auto get_manager_entry_size() const { return rbCache->manager_.entry.size(); }
    auto get_manager_entry(size_t idx) const { return rbCache->manager_.entry[idx]; }
    auto get_valid_entry(size_t idx) const { return rbCache->valid_[idx]; }
    constexpr static auto invalid_entry_ = RingbufferCacheManager::invalid_cache_entry_;
};

TEST_F(RingbufferCacheTestFixture, SimpleAllocate) {
    auto result = rbCache->get_cache_offset(0, 10);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->is_cached, false);
    EXPECT_EQ(result->offset, 0);
}

TEST_F(RingbufferCacheTestFixture, AllocateMoreThanCacheSize) {
    // Try to allocate more than 256 blocks, expect failure
    auto result = rbCache->get_cache_offset(0, (cache_size_blocks + 1) * cache_block_sizeB);
    ASSERT_FALSE(result);
}

TEST_F(RingbufferCacheTestFixture, ConfirmCachedBlocks) {
    // Allocate blocks of sizes {2, 4, 8, 16, 32, 64, 128}, confirm they are cached
    std::vector<size_t> block_sizes = {2, 4, 8, 16, 32, 64, 128};
    std::vector<size_t> pgm_ids;
    auto last_offset = 0;
    for (auto i = 0; i < block_sizes.size(); ++i) {
        auto pgm_size = block_sizes[i] * cache_block_sizeB;
        auto result = rbCache->get_cache_offset(i, pgm_size);
        pgm_ids.push_back(i);
        ASSERT_TRUE(result);
        if (result) {
            EXPECT_EQ(result->is_cached, false);
            EXPECT_EQ(result->offset, last_offset);
            last_offset += block_sizes[i];
        }
    }
    last_offset = 0;
    // cache hits
    for (auto i = 0; i < block_sizes.size(); ++i) {
        auto pgm_size = block_sizes[i] * cache_block_sizeB;
        auto result = rbCache->get_cache_offset(pgm_ids[i], pgm_size);
        ASSERT_TRUE(result);
        if (result) {
            EXPECT_EQ(result->is_cached, true);
            EXPECT_EQ(result->offset, last_offset);
            last_offset += block_sizes[i];
        }
    }
    EXPECT_EQ(get_manager_entry_size(), initial_manager_size);
    EXPECT_EQ(get_oldest_idx(), 0);
    EXPECT_EQ(get_next_idx(), block_sizes.size());
}

TEST_F(RingbufferCacheTestFixture, WraparoundAllocateLargeBlock) {
    // Allocate blocks of sizes {2, 4, 8, 16, 32, 64, 128, 128, 8}, confirm wraparound allocation and eviction of
    // previous entries
    std::vector<size_t> block_sizes = {2, 4, 8, 16, 32, 64, 128, 128};
    std::vector<size_t> pgm_ids;
    size_t last_offset = 0;
    int i;
    for (i = 0; i < block_sizes.size() - 1; ++i) {
        auto pgm_size = block_sizes[i] * cache_block_sizeB;
        auto result = rbCache->get_cache_offset(i, pgm_size);
        EXPECT_EQ(get_oldest_idx(), 0);
        pgm_ids.push_back(i);
        ASSERT_TRUE(result);
        if (result) {
            EXPECT_EQ(result->is_cached, false);
            EXPECT_EQ(result->offset, last_offset);
            last_offset += block_sizes[i];
        }
    }
    EXPECT_EQ(get_next_block_offset(), std::reduce(block_sizes.begin(), block_sizes.end() - 1));
    // Allocate last block to invalidate all previous entries. Then check everything is invalidated. Check internal
    // manager state is as expected
    auto pgm_size = block_sizes[i] * cache_block_sizeB;
    auto result = rbCache->get_cache_offset(i, pgm_size);
    pgm_ids.push_back(i);
    ASSERT_TRUE(result);
    if (result) {
        EXPECT_EQ(result->is_cached, false);
        EXPECT_EQ(result->offset, 0);
    }

    for (i = 0; i < block_sizes.size() - 1; ++i) {
        EXPECT_EQ(get_valid_entry(i), invalid_entry_);
    }
    EXPECT_EQ(get_oldest_idx(), 7);
    EXPECT_EQ(get_next_block_offset(), block_sizes.back() & (cache_size_blocks - 1));

    EXPECT_EQ(get_manager_entry_size(), initial_manager_size);
    EXPECT_EQ(get_next_idx(), block_sizes.size());
}

TEST_F(RingbufferCacheTestFixture, EvictOldestEntries) {
    // Allocate blocks of sizes {48, 48, 128}, then allocate 64 and confirm eviction
    std::vector<size_t> block_sizes = {48, 48, 128};
    std::vector<size_t> pgm_ids;
    size_t last_offset = 0;
    int i;
    for (i = 0; i < block_sizes.size(); ++i) {
        auto pgm_size = block_sizes[i] * cache_block_sizeB;
        rbCache->get_cache_offset(i, pgm_size);
        pgm_ids.push_back(i);
    }
    rbCache->get_cache_offset(i, 64 * cache_block_sizeB);
    // pgm_ids.push_back(i);

    // Confirm evictions
    for (auto it_pgm_id = pgm_ids.begin(); it_pgm_id != pgm_ids.end() - 1; ++it_pgm_id) {
        EXPECT_EQ(get_valid_entry(*it_pgm_id), invalid_entry_);
    }

    // Confirm oldest_idx
    EXPECT_EQ(get_oldest_idx(), 2);
}

TEST_F(RingbufferCacheTestFixture, ComplexEvictionAndAllocation) {
    // Allocate blocks of sizes {128, 64, 32}, then allocate 64, 40, and 50
    std::vector<size_t> block_sizes = {128, 64, 32};
    int last_offset = 0;
    for (auto i = 0; i < block_sizes.size(); ++i) {
        auto result = rbCache->get_cache_offset(i, block_sizes[i] * cache_block_sizeB);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->is_cached, false);
        EXPECT_EQ(result->offset, last_offset);
        last_offset += block_sizes[i];
    }
    EXPECT_EQ(get_oldest_idx(), 0);
    EXPECT_EQ(get_next_block_offset(), std::reduce(block_sizes.begin(), block_sizes.end()));

    auto result = rbCache->get_cache_offset(block_sizes.size(), 64 * cache_block_sizeB);  // Evict first block
    EXPECT_EQ(result->offset, 0);
    EXPECT_EQ(get_valid_entry(0), invalid_entry_);

    result = rbCache->get_cache_offset(block_sizes.size() + 1, 40 * cache_block_sizeB);  // No eviction
    EXPECT_EQ(result->offset, 64);
    EXPECT_EQ(get_valid_entry(1), 1);

    result = rbCache->get_cache_offset(block_sizes.size() + 2, 50 * cache_block_sizeB);  // Evict second block
    EXPECT_EQ(result->offset, (64 + 40));
    EXPECT_EQ(get_valid_entry(1), invalid_entry_);
    EXPECT_EQ(get_valid_entry(2), 2);
    EXPECT_EQ(get_oldest_idx(), 2);
    EXPECT_EQ(get_next_block_offset(), 64 + 40 + 50);
}

TEST_F(RingbufferCacheTestFixture, ValidateCacheFullBehavior) {
    // Validate behavior when cache is completely full
    std::vector<size_t> block_sizes(256, 1);  // Fill cache with 256 blocks of size 1
    int last_offset = 0;
    for (size_t i = 0; i < block_sizes.size(); ++i) {
        auto result = rbCache->get_cache_offset(i + 100, block_sizes[i] * cache_block_sizeB);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->is_cached, false);
        EXPECT_EQ(result->offset, last_offset);
        last_offset += block_sizes[i];
        EXPECT_EQ(get_valid_entry(i + 100), result->offset);
    }

    // Allocate a new block to trigger eviction
    rbCache->get_cache_offset(10, 1 * cache_block_sizeB);

    EXPECT_EQ(get_manager_entry_size(), std::min(block_sizes.size() * 2, cache_size_blocks));
    // Validate the new block is added at the correct offset
    EXPECT_EQ(get_manager_entry(0).offset, 0);
    EXPECT_EQ(get_manager_entry(0).valid_idx, 10);
    EXPECT_EQ(get_valid_entry(10), 0);

    EXPECT_EQ(get_next_idx(), 1);
    EXPECT_EQ(get_oldest_idx(), 1);
}

TEST_F(RingbufferCacheTestFixture, FillResetRefill) {
    // Fill the cache
    std::vector<size_t> block_sizes(4, 64);  // Fill cache with 256 blocks of size 1
    int last_offset = 0;
    for (size_t i = 0; i < block_sizes.size(); ++i) {
        auto result = rbCache->get_cache_offset(i + 1000, block_sizes[i] * cache_block_sizeB);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->is_cached, false);
        EXPECT_EQ(result->offset, last_offset);
        last_offset += block_sizes[i];
        EXPECT_EQ(get_valid_entry(i + 1000), i);
    }

    // Reset the cache
    rbCache->reset();

    // Refill the cache
    last_offset = 0;
    for (size_t i = 0; i < block_sizes.size(); ++i) {
        auto result = rbCache->get_cache_offset(i + 10, block_sizes[i] * cache_block_sizeB);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->is_cached, false);
        EXPECT_EQ(result->offset, last_offset);
        last_offset += block_sizes[i];
        EXPECT_EQ(get_valid_entry(i + 10), i);
    }
    // Validate the cache is filled correctly
    last_offset = 0;
    for (size_t i = 0; i < block_sizes.size(); ++i) {
        EXPECT_EQ(get_manager_entry(i).valid_idx, i + 10);
        EXPECT_EQ(get_manager_entry(i).offset, last_offset);
        last_offset += block_sizes[i];
        EXPECT_EQ(get_valid_entry(i + 10), i);
    }
    EXPECT_EQ(get_next_block_offset(), 0);
    EXPECT_EQ(get_oldest_idx(), 0);
    EXPECT_EQ(get_next_idx(), block_sizes.size());
}

TEST_F(RingbufferCacheTestFixture, ValidateWraparoundAllocate) {
    // fillup cache
    int i;
    for (i = 0; i < cache_size_blocks; ++i) {
        rbCache->get_cache_offset(i, 1 * cache_block_sizeB);
    }
    EXPECT_EQ(get_manager_entry_size(), cache_size_blocks);
    EXPECT_EQ(get_oldest_idx(), 0);
    EXPECT_EQ(get_next_block_offset(), 0);
    auto next_idx = 0;
    EXPECT_EQ(get_next_idx(), 0);

    std::vector<size_t> block_sizes{1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89};  // first 11 fibonacci
    std::reverse(block_sizes.begin(), block_sizes.end());
    auto fibonacci_total = std::reduce(block_sizes.begin(), block_sizes.end());

    // Fill the cache
    int last_offset = 0;
    for (i = 0; i < block_sizes.size(); ++i) {
        auto result = rbCache->get_cache_offset(cache_size_blocks + i, block_sizes[i] * cache_block_sizeB);
    }
    EXPECT_EQ(get_next_block_offset(), fibonacci_total);
    next_idx = block_sizes.size();
    EXPECT_EQ(get_next_idx(), next_idx);
    EXPECT_EQ(
        get_oldest_idx(),
        fibonacci_total);  // because had to free up fib total number of 1 block allocations that were made before

    // Allocate a new block to trigger wraparound
    auto eviction_blocks_num = 48;
    auto result = rbCache->get_cache_offset(cache_size_blocks + i, eviction_blocks_num * cache_block_sizeB);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->is_cached, false);
    EXPECT_EQ(result->offset, 0);
    // Check current state:
    //  next_idx was at end of fibonacci sequence, i.e 11, so now it must be 12. Since the cache wrapped around, the
    //  older entries from fib total to end of cache (single block allocations) must have been evicted. That would take
    //  oldest_idx all the way to the end of the manager and wrapped to the front of the manager array. Finally, since
    //  the next block was allocated to cache[0], oldest_idx must've incremented to 1.
    EXPECT_EQ(get_oldest_idx(), 1);
    EXPECT_EQ(get_next_idx(), 12);
    // verify that oldest_idx is pointing to cache block at higher address than next_block_offset
    EXPECT_GE(get_manager_entry(get_oldest_idx()).offset, get_next_block_offset());
    EXPECT_EQ(get_valid_entry(get_manager_entry(get_oldest_idx()).valid_idx), get_oldest_idx());
    EXPECT_EQ(get_next_block_offset(), eviction_blocks_num);

    // The last allocation, as explained above, must have been at offset 11. Let's confirm it is correct.
    auto current_idx = get_next_idx() - 1;
    EXPECT_EQ(get_manager_entry(current_idx).offset, 0);
    EXPECT_EQ(get_manager_entry(current_idx).valid_idx, cache_size_blocks + i);
    EXPECT_EQ(get_valid_entry(cache_size_blocks + i), current_idx);

    for (i = fibonacci_total; i < block_sizes.size(); ++i) {
        EXPECT_EQ(get_valid_entry(i), invalid_entry_);
    }
}

struct CacheTestParams {
    size_t cache_block_sizeB;
    size_t cache_size_blocks;
    size_t initial_manager_size;
    std::pair<int, int> pgm_ids;
    std::pair<int, int> pgm_sizes;
};
class RingbufferCacheRandomizedTestsFixture : public ::testing::TestWithParam<CacheTestParams> {
protected:
    RingbufferCacheRandomizedTestsFixture() = default;
    ~RingbufferCacheRandomizedTestsFixture() override = default;

    std::unique_ptr<RingbufferCacheManager> rbCache;

    // This function is called before each test in this test case.
    void setup(CacheTestParams& params) {
        rbCache = std::make_unique<RingbufferCacheManager>(
            params.cache_block_sizeB, params.cache_size_blocks, params.initial_manager_size);
    }

    // define accessors to the private members of the RingbufferCacheManager
    auto get_next_block_offset() const { return rbCache->manager_.next_block_offset; }
    auto get_oldest_block_offset() const { return rbCache->manager_.entry[rbCache->manager_.oldest_idx].offset; }
    auto get_oldest_idx() const { return rbCache->manager_.oldest_idx; }
    auto get_next_idx() const { return rbCache->manager_.next_idx; }
    auto get_manager_entry_size() const { return rbCache->manager_.entry.size(); }
    auto get_manager_entry(size_t idx) const { return rbCache->manager_.entry[idx]; }
    auto get_valid_entry(size_t idx) const { return rbCache->valid_[idx]; }
    constexpr static auto invalid_entry_ = RingbufferCacheManager::invalid_cache_entry_;
};

INSTANTIATE_TEST_SUITE_P(
    RingbufferCacheRandomSuite,
    RingbufferCacheRandomizedTestsFixture,
    testing::Values(
        CacheTestParams{
            .cache_block_sizeB = 4,
            .cache_size_blocks = 1024,
            .initial_manager_size = 64,
            .pgm_ids = std::make_pair(0, 1000),
            .pgm_sizes = std::make_pair(1, 769)},
        CacheTestParams{
            .cache_block_sizeB = 4,
            .cache_size_blocks = 1024,
            .initial_manager_size = 32,
            .pgm_ids = std::make_pair(0, 4000),
            .pgm_sizes = std::make_pair(4, 20)},
        CacheTestParams{
            .cache_block_sizeB = 4,
            .cache_size_blocks = 256,
            .initial_manager_size = 64,
            .pgm_ids = std::make_pair(0, 10000),
            .pgm_sizes = std::make_pair(1, 4)},
        CacheTestParams{
            .cache_block_sizeB = 4,
            .cache_size_blocks = 1024,
            .initial_manager_size = 2,
            .pgm_ids = std::make_pair(0, 10000),
            .pgm_sizes = std::make_pair(1, 100)},
        CacheTestParams{
            .cache_block_sizeB = 4,
            .cache_size_blocks = 512,
            .initial_manager_size = 4,
            .pgm_ids = std::make_pair(0, 1000),
            .pgm_sizes = std::make_pair(1, 10)},
        CacheTestParams{
            .cache_block_sizeB = 4,
            .cache_size_blocks = 1024,
            .initial_manager_size = 1024,
            .pgm_ids = std::make_pair(0, 4000),
            .pgm_sizes = std::make_pair(4, 20)},
        CacheTestParams{
            .cache_block_sizeB = 4,
            .cache_size_blocks = 1024,
            .initial_manager_size = 2048,
            .pgm_ids = std::make_pair(0, 4000),
            .pgm_sizes = std::make_pair(16, 64)},
        CacheTestParams{// high hits test
                        .cache_block_sizeB = 4,
                        .cache_size_blocks = 1024,
                        .initial_manager_size = 2,
                        .pgm_ids = std::make_pair(0, 512),
                        .pgm_sizes = std::make_pair(1, 8)},
        CacheTestParams{// high hits test
                        .cache_block_sizeB = 4,
                        .cache_size_blocks = 4096,
                        .initial_manager_size = 32,
                        .pgm_ids = std::make_pair(0, 512),
                        .pgm_sizes = std::make_pair(4, 24)},
        CacheTestParams{// high hits test
                        .cache_block_sizeB = 4,
                        .cache_size_blocks = 4096,
                        .initial_manager_size = 16,
                        .pgm_ids = std::make_pair(0, 256),
                        .pgm_sizes = std::make_pair(4, 16)},
        CacheTestParams{
            .cache_block_sizeB = 4,
            .cache_size_blocks = 512,
            .initial_manager_size = 128,
            .pgm_ids = std::make_pair(0, 4000),
            .pgm_sizes = std::make_pair(100, 200)}));

TEST_P(RingbufferCacheRandomizedTestsFixture, RandomizedQueries) {
    CacheTestParams params = GetParam();
    setup(params);
    auto pgm_ids = params.pgm_ids;
    auto pgm_sizes = params.pgm_sizes;

    std::unordered_map<int, int> pgm_id_size_map;

    int hits_count = 0;

    std::random_device rd;
    uint64_t rd1 = rd(), rd2 = rd();
    std::mt19937 gen_pgm_id(rd1);
    std::mt19937 gen_pgm_size(rd2);
    std::uniform_int_distribution<uint64_t> dist_pgm_id(pgm_ids.first, pgm_ids.second - 1);
    std::uniform_int_distribution<uint64_t> dist_pgm_size(pgm_sizes.first, pgm_sizes.second - 1);
    uint64_t pgm_id, pgm_size;
    for (size_t i = 0; i < 10'000'000; ++i) {
        pgm_id = dist_pgm_id(gen_pgm_id);
        if (pgm_id_size_map.find(pgm_id) != pgm_id_size_map.end()) {
            pgm_size = pgm_id_size_map[pgm_id];
        } else {
            pgm_size = dist_pgm_size(gen_pgm_size);
            pgm_id_size_map[pgm_id] = pgm_size;
        }
    }

    gen_pgm_id.seed(rd1);  // restart from seed
    auto start_rbcache = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10'000'000; ++i) {
        pgm_id = dist_pgm_id(gen_pgm_id);
        pgm_size = pgm_id_size_map[pgm_id];

        auto result = rbCache->get_cache_offset(pgm_id, pgm_size * params.cache_block_sizeB);
        ASSERT_TRUE(result);
        ASSERT_GE(result->offset, 0);
        ASSERT_LT(result->offset, params.cache_size_blocks);
        if (!result->is_cached) {
            ASSERT_TRUE((result->offset + pgm_size) % params.cache_size_blocks == get_next_block_offset())
                << "Failed check (iter:" << i << "): cache size: " << params.cache_size_blocks << ", pgm_id: " << pgm_id
                << ", pgm_size: " << pgm_size << ", offset: " << result->offset
                << ", next_block_offset: " << get_next_block_offset() << std::endl;
        }
        ASSERT_TRUE(get_manager_entry_size() >= std::min(params.initial_manager_size, params.cache_size_blocks))
            << "Manager size: " << get_manager_entry_size() << ", cache size: " << params.cache_size_blocks
            << ", initial manager size: " << params.initial_manager_size << ", oldest_idx: " << get_oldest_idx()
            << ", next_index: " << get_next_idx() << ", oldest_block_offset: " << get_oldest_block_offset()
            << ", next_block_offset: " << get_next_block_offset() << std::endl;
        if (result->is_cached) {
            ++hits_count;
        }
    }
    auto end_rbcache = std::chrono::high_resolution_clock::now();
    auto duration_rbcache = std::chrono::duration_cast<std::chrono::milliseconds>(end_rbcache - start_rbcache).count();
    std::cout << "Ringbuffer cache runtime: " << duration_rbcache << " ms, hits: " << hits_count << std::endl;
    ASSERT_TRUE(get_manager_entry_size() <= params.cache_size_blocks)
        << "Manager size: " << get_manager_entry_size() << ", cache size: " << params.cache_size_blocks << std::endl;
    std::cout << "Cache size: " << params.cache_size_blocks << ", initial manager size: " << params.initial_manager_size
              << ", final manager size: " << get_manager_entry_size() << std::endl;
}

TEST(DISABLED_MapComparison, OrderedUnorderedMapPerformance) {
    std::unordered_map<int, int> map_unordered;
    std::map<int, int> map_ordered;

    constexpr std::pair<size_t, size_t> dram_address_range(0x10000000, 0x10005000);
    constexpr size_t cache_size = 1024 * 16;
    int hits_count = 0;

    std::random_device rd;
    std::array<uint64_t, cache_size> cache_offsets = {};
    size_t cache_index = 0;

    // Initialize ringbuffer for ordered map
    size_t ringbuffer_start = 0;
    size_t ringbuffer_size = 0;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(dram_address_range.first, dram_address_range.second - 1);

    auto start_unordered = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10'000'000; ++i) {
        uint64_t dram_address = dist(gen);

        // Using unordered_map
        if (map_unordered.find(dram_address) != map_unordered.end()) {
            ++hits_count;
        } else {
            if (cache_index < cache_size) {
                map_unordered[dram_address] = cache_index;
                cache_offsets[cache_index] = dram_address;
                ++cache_index;
            } else {
                // Evict the oldest entry
                uint64_t evicted_address = cache_offsets[cache_index % cache_size];
                map_unordered.erase(evicted_address);

                // Add the new address
                map_unordered[dram_address] = cache_index % cache_size;
                cache_offsets[cache_index % cache_size] = dram_address;
                ++cache_index;
            }
        }
    }
    auto end_unordered = std::chrono::high_resolution_clock::now();
    auto duration_unordered =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_unordered - start_unordered).count();
    std::cout << "Unordered map runtime: " << duration_unordered << " ms, hits: " << hits_count << std::endl;

    hits_count = 0;  // Reset hit count for ordered map

    auto start_ordered = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < 10'000'000; ++i) {
        uint64_t dram_address = dist(gen);

        // Using unordered_map
        if (map_ordered.find(dram_address) != map_ordered.end()) {
            ++hits_count;
        } else {
            if (cache_index < cache_size) {
                map_ordered[dram_address] = cache_index;
                cache_offsets[cache_index] = dram_address;
                ++cache_index;
            } else {
                // Evict the oldest entry
                uint64_t evicted_address = cache_offsets[cache_index % cache_size];
                map_ordered.erase(evicted_address);

                // Add the new address
                map_ordered[dram_address] = cache_index % cache_size;
                cache_offsets[cache_index % cache_size] = dram_address;
                ++cache_index;
            }
        }
    }
    auto end_ordered = std::chrono::high_resolution_clock::now();
    auto duration_ordered = std::chrono::duration_cast<std::chrono::milliseconds>(end_ordered - start_ordered).count();
    std::cout << "Ordered map runtime: " << duration_ordered << " ms, hits: " << hits_count << std::endl;
}

}  // namespace tt::tt_metal
