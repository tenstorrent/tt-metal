// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/ringbuffer_cache.hpp"
#include "assert.hpp"
// #include "tt_metal/hw/inc/dataflow_api.h"

namespace tt::tt_metal {

// this function encapsulates the common code for adding a new entry to the ringbuffer manager
// also, this function should be called when the cache is empty
void RingbufferCacheManager::add_manager_entry_no_evict(uint64_t pgm_id, uint32_t offset, uint32_t length) {
    TT_ASSERT(
        offset + length <= cache_size_blocks_, "RingbufferCacheManager new allocation: offset + length > cache size");

    auto idx = this->manager_.next_idx;
    this->manager_.entry[idx] = {offset, length, pgm_id};

    TT_ASSERT(pgm_id <= UINT32_MAX, "RingbufferCacheManager new allocation: pgm_id > UINT32_MAX");
    this->valid_[pgm_id] = idx;

    this->manager_.next_block_offset = offset + length == cache_size_blocks_ ? 0 : offset + length;
    this->manager_.update_next_idx();
}

// this function should be called to allocate a new cache entry if the cache is not empty
void RingbufferCacheManager::add_manager_entry(uint64_t pgm_id, uint32_t offset, uint32_t length) {
    auto idx = this->manager_.next_idx;
    if (idx == this->manager_.oldest_idx) {
        // if the next index is the same as the oldest index, then we need to invalidate the oldest entry
        this->invalidate_manager_entry();
    }
    add_manager_entry_no_evict(pgm_id, offset, length);
}

// this function will invalidate the oldest entry in the ringbuffer manager
// caution: oldest entry must be valid
void RingbufferCacheManager::invalidate_manager_entry(void) {
    auto& entry = this->manager_.entry[this->manager_.oldest_idx];
    auto valid_idx = entry.valid_idx;
    TT_ASSERT(valid_idx < this->valid_.size(), "RingbufferCacheManager invalidation: pgm_id out of range");
    TT_ASSERT(
        this->valid_[valid_idx] != RingbufferCacheManager::invalid_cache_entry_,
        "RingbufferCacheManager invalidation: entry not valid (mgr: oldest_idx:{}, next_idx:{}, offset:{}, length:{}, "
        "valid idx (pgm id):{}, valid[pgm id]:{})",
        this->manager_.oldest_idx,
        this->manager_.next_idx,
        entry.offset,
        entry.length,
        valid_idx,
        this->valid_[valid_idx]);
    // invalidate the entry
    this->valid_[valid_idx] = RingbufferCacheManager::invalid_cache_entry_;
    this->manager_.update_oldest_idx();
}

// this function will invalidate all oldest entries until the cache wraps around
// caution: should not be called if the cache is empty
bool RingbufferCacheManager::invalidate_oldest_until_wraparound(void) {
    if (this->manager_.oldest_idx == this->manager_.next_idx) {
        this->invalidate_manager_entry();
    }
    while ((this->manager_.entry[this->manager_.oldest_idx].offset >= this->manager_.next_block_offset) and
           (this->manager_.oldest_idx != this->manager_.next_idx)) {
        invalidate_manager_entry();
    }
    if (this->manager_.oldest_idx == this->manager_.next_idx) {
        // this means cache is empty and caller should call add_manager_entry_no_evict
        return true;
    } else {
        return false;
    }
}
// this function is used to invalidate the oldest entry until there is sufficient space
// caution: should not be called if the cache is empty
bool RingbufferCacheManager::invalidate_sufficient_blocks(int required_space, int offset) {
    if (this->manager_.oldest_idx == this->manager_.next_idx) {
        this->invalidate_manager_entry();
    }
    int oldest_block_offset = this->manager_.entry[this->manager_.oldest_idx].offset;
    int available_space = oldest_block_offset - offset;
    while (available_space < required_space and this->manager_.oldest_idx != this->manager_.next_idx) {
        available_space += this->manager_.entry[this->manager_.oldest_idx].length;
        invalidate_manager_entry();  // invalidate oldest entry
    }
    if (this->manager_.oldest_idx == this->manager_.next_idx) {
        // this means cache is empty and caller should call add_manager_entry_no_evict
        return true;
    } else {
        return false;
    }
}

std::optional<typename RingbufferCacheManager::CacheOffset> RingbufferCacheManager::get_cache_offset(
    uint64_t pgm_id, uint32_t lengthB) {
    uint32_t cache_offset;
    CacheOffset query_result;

    const int required_space = (lengthB + cache_block_sizeB_ - 1) / cache_block_sizeB_;
    if (required_space > cache_size_blocks_) [[unlikely]] {
        return std::nullopt;  // cannot fit in cache
    } else if (manager_.entry.size() == 0) [[unlikely]] {
        // first entry, so we can just add it
        this->valid_.resize(pgm_id + 1, RingbufferCacheManager::invalid_cache_entry_);
        manager_.entry.resize(std::min(this->cache_manager_initial_entry_size_, this->cache_size_blocks_));
        add_manager_entry_no_evict(pgm_id, 0, required_space);
        query_result.is_cached = false;
        query_result.offset = 0;
        return query_result;
    } else if (pgm_id >= this->valid_.size()) {
        this->valid_.resize(pgm_id + 1, RingbufferCacheManager::invalid_cache_entry_);
    } else if (this->valid_[pgm_id] != invalid_cache_entry_) {
        unsigned mgr_idx = this->valid_[pgm_id];
        TT_ASSERT(
            mgr_idx < this->manager_.entry.size(),
            "RingbufferCacheManager invalid cache hit: manager index out of bounds. pgm_id:{}, valid idx:{}, entry "
            "size:{}",
            pgm_id,
            mgr_idx,
            this->manager_.entry.size());
        auto mgr_entry = manager_.entry[mgr_idx];
        TT_ASSERT(
            mgr_entry.length * cache_block_sizeB_ >= lengthB,
            "RingbufferCacheManager invalid cache hit: requested length:{} > entry length:{}, manager idx:{}, cache "
            "offset:{}, valid_idx/pgm_id requested:{}/{}",
            lengthB,
            mgr_entry.length * cache_block_sizeB_,
            mgr_idx,
            mgr_entry.offset,
            mgr_entry.valid_idx,
            pgm_id);
        query_result.is_cached = true;
        query_result.offset = mgr_entry.offset;
        return query_result;
    }

    query_result.is_cached = false;

    // find space in RB for new entry, or if full, then invalidate oldest entry(ies)
    int next_block_offset = this->manager_.next_block_offset;
    int oldest_block_offset = this->manager_.entry[this->manager_.oldest_idx].offset;
    int free_space_to_end = this->cache_size_blocks_ - next_block_offset;
    bool cache_emptied = false;
    if (next_block_offset > oldest_block_offset) {
        if (free_space_to_end < required_space) [[unlikely]] {
            cache_emptied = invalidate_sufficient_blocks(required_space);  // free up space from beginning
            cache_offset = 0;
        } else {
            cache_offset = next_block_offset;  // cache has space
        }
    } else {
        if (free_space_to_end >= required_space) {
            cache_emptied = invalidate_sufficient_blocks(required_space, next_block_offset);
            cache_offset = next_block_offset;
        } else [[unlikely]] {
            // cache is not full, but must wraparound for sufficient space
            cache_emptied = invalidate_oldest_until_wraparound();
            if (not cache_emptied) {
                cache_emptied = invalidate_sufficient_blocks(required_space);  // free up space from beginning
            }
            cache_offset = 0;
        }
    }
    // made room, now allocate the new entry
    if (cache_emptied) {
        add_manager_entry_no_evict(pgm_id, cache_offset, required_space);
    } else {
        add_manager_entry(pgm_id, cache_offset, required_space);
    }

    query_result.offset = cache_offset;
    return query_result;
}

}  // namespace tt::tt_metal
