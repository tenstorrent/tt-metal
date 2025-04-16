// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefetcher cache implementation

#pragma once

#include <cstdint>
#include <array>

namespace tt::tt_metal {

#define DISPATCH_PREFETCH_CACHE_SIZE_LOG2 20
#define DISPATCH_PREFETCH_CACHE_BLOCK_SIZE_LOG2 10
#define DISPATCH_PREFETCH_CACHE_BLOCK_SIZE (1 << DISPATCH_PREFETCH_CACHE_BLOCK_SIZE_LOG2)  // 1KB
#define DISPATCH_PREFETCH_CACHE_NUM_BLOCKS \
    (1 << (DISPATCH_PREFETCH_CACHE_SIZE_LOG2 - DISPATCH_PREFETCH_CACHE_BLOCK_SIZE_LOG2))  // 1MB
constexpr int offset_width{
    DISPATCH_PREFETCH_CACHE_SIZE_LOG2 -
    DISPATCH_PREFETCH_CACHE_BLOCK_SIZE_LOG2};  // # of bits to represent offset in cache, in blocks
constexpr int length_width{10};                // # of bits to represent amount of data in cache, in blocks
constexpr int valid_width{offset_width};       // # of bits to represent valid table index, in blocks
struct DispatchPrefetchCacheMgrEntry {
    uint32_t offset : offset_width;
    uint32_t length : length_width;
    uint32_t valid : valid_width;
};
static_assert(
    (offset_width + length_width + valid_width) <= 8 * sizeof(uint32_t),
    "DispatchPrefetchCacheMgrEntry bitwidth sizes are ill defined");

class DispatchPrefetchCacheMgr {
public:
    DispatchPrefetchCacheMgr() {
        std::fill(this->valid_.begin(), this->valid_.end(), -1);
        // Initialize the cache manager entries
        std::fill(this->mgr_.begin(), this->mgr_.end(), 0);
    };

    ~DispatchPrefetchCacheMgr() = default;

    /*! @brief  Transfer dram buffer to dispatcher L1
     *  @return success/failure
     */
    bool copy_dram_to_dispatcher_l1(PrefetchExecBufState exec_buf_state, uint32_t addr);

private:
    /*! @brief base address of prefetcher L1 cache */
    const uint32_t cache_base_addr_{0};

    /*! @brief cache entries meta data */
    struct {
        std::array<DispatchPrefetchCacheMgrEntry, DISPATCH_PREFETCH_CACHE_NUM_BLOCKS> entry;
        uint16_t oldest_idx_{0};
        uint16_t next_idx_{0};
    } mgr_;

    /*! @brief if dram --> cache assoc is there then valid_id is not needed? */
    std::array<int16_t, DISPATCH_PREFETCH_CACHE_NUM_BLOCKS> valid_;

    bool is_data_in_cache(DispatchPrefetchCacheMgrEntry& m) {
        std::ptrdiff_t m_idx = &m - &mgr_[0];
        // ASSERT (m_idx >= 0 && m_idx < DISPATCH_PREFETCH_CACHE_NUM_BLOCKS);
        int8_t val_id = valid_[m.valid];
        return (valid_[m.valid] != -1);
    }

    void set_data_in_cache(DispatchPrefetchCacheMgrEntry& m) {
        std::ptrdiff_t m_idx = &m - &mgr_[0];
        // ASSERT (m_idx >= 0 && m_idx < DISPATCH_PREFETCH_CACHE_NUM_BLOCKS);
        uint32_t addr_d = this->base_addr_ + m.offset;
    }

    void read_data_from_cache_write_dispatcher(uint32_t cache_add, uint32_t size, uint32_t dispatcher_addr) {
        std::ptrdiff_t m_idx = &m - &mgr_[0];
        // ASSERT (m_idx >= 0 && m_idx < DISPATCH_PREFETCH_CACHE_NUM_BLOCKS);
        uint32_t addr_c = this->base_addr_ + m.offset;
        this->dram_cache_map_[addr_d] = addr_c;
        this->valid_[m.valid] = m_idx;
    }
};

}  // namespace tt::tt_metal
