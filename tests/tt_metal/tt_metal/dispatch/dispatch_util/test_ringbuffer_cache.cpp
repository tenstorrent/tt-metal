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

    std::unique_ptr<RingbufferCacheManager> rb_cache_;

    // This function is called before each test in this test case.
    void setup(CacheTestParams& params) {
        rb_cache_ = std::make_unique<RingbufferCacheManager>(
            params.cache_block_sizeB, params.cache_size_blocks, params.initial_manager_size);
    }

    // define accessors to the private members of the RingbufferCacheManager
    auto get_next_block_offset() const { return rb_cache_->manager_.next_block_offset; }
    auto get_oldest_block_offset() const { return rb_cache_->manager_.entry[rb_cache_->manager_.oldest_idx].offset; }
    auto get_oldest_idx() const { return rb_cache_->manager_.oldest_idx; }
    auto get_next_idx() const { return rb_cache_->manager_.next_idx; }
    auto get_manager_entry_size() const { return rb_cache_->manager_.entry.size(); }
    auto get_manager_entry(size_t idx) const { return rb_cache_->manager_.entry[idx]; }
    auto get_valid_entry(size_t idx) const { return rb_cache_->valid_[idx]; }
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
    uint64_t pgm_id = 0, pgm_size = 0;
    constexpr size_t num_iterations = 10'000'000;
    for (size_t i = 0; i < num_iterations; ++i) {
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
    for (size_t i = 0; i < num_iterations; ++i) {
        pgm_id = dist_pgm_id(gen_pgm_id);
        pgm_size = pgm_id_size_map[pgm_id];

        auto result = rb_cache_->get_cache_offset(pgm_id, pgm_size * params.cache_block_sizeB);
        ASSERT_TRUE(result);
        ASSERT_GE(result->offset, 0);
        ASSERT_LT(result->offset, params.cache_size_blocks);
        if (!result->is_cached) {
            ASSERT_TRUE((result->offset + pgm_size) % params.cache_size_blocks == get_next_block_offset())
                << "Failed check (iter:" << i << "): cache size: " << params.cache_size_blocks << ", pgm_id: " << pgm_id
                << ", pgm_size: " << pgm_size << ", offset: " << result->offset
                << ", next_block_offset: " << get_next_block_offset() << std::endl;
        }
        ASSERT_GE(get_manager_entry_size(), std::min(params.initial_manager_size, params.cache_size_blocks))
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
    ASSERT_LE(get_manager_entry_size(), params.cache_size_blocks)
        << "Manager size: " << get_manager_entry_size() << ", cache size: " << params.cache_size_blocks << std::endl;
    std::cout << "Cache size: " << params.cache_size_blocks << ", initial manager size: " << params.initial_manager_size
              << ", final manager size: " << get_manager_entry_size() << std::endl;
}

}  // namespace tt::tt_metal
