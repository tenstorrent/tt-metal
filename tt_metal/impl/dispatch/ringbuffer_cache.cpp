// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/ringbuffer_cache.hpp"
#include "assert.hpp"
// #include "tt_metal/hw/inc/dataflow_api.h"

namespace tt::tt_metal {

RingbufferCacheManager::RingbufferCacheManager(
    int cache_block_sizeB, int cache_size_blocks, int manager_entry_initial_size) :
    cache_block_sizeB_(cache_block_sizeB),
    cache_size_blocks_(cache_size_blocks),
    cache_manager_initial_entry_size_{manager_entry_initial_size} {
    TT_ASSERT(cache_size_blocks > 0, "Ringbuffer cache size must be greater than 0");
    TT_ASSERT(cache_block_sizeB > 0, "Ringbuffer cache block size must be greater than 0");
    TT_ASSERT(
        cache_manager_initial_entry_size_ > 0,
        "Ringbuffer cache manager initial entry size ({}) must be greater than 0 ({}, {})",
        cache_manager_initial_entry_size_,
        cache_block_sizeB,
        cache_size_blocks_);
    TT_ASSERT(
        (cache_manager_initial_entry_size_ & (cache_manager_initial_entry_size_ - 1)) == 0,
        "Ringbuffer cache manager initial entry size ({}) is not a power of 2 ({}, {})",
        cache_manager_initial_entry_size_,
        cache_block_sizeB,
        cache_size_blocks_);
}

// this function encapsulates the common code for adding a new entry to the ringbuffer manager
// also, this function should be called when the cache is empty
void RingbufferCacheManager::add_manager_entry_no_evict(uint64_t pgm_id, uint32_t offset, uint32_t length) {
    TT_ASSERT(
        offset + length <= cache_size_blocks_, "RingbufferCacheManager new allocation: offset + length > cache size");

    auto idx = manager_.next_idx;
    manager_.entry[idx] = {offset, length, pgm_id};

    TT_ASSERT(pgm_id <= UINT32_MAX, "RingbufferCacheManager new allocation: pgm_id > UINT32_MAX");
    valid_[pgm_id] = idx;

    manager_.next_block_offset = offset + length == cache_size_blocks_ ? 0 : offset + length;
    manager_.update_next_idx();
}

// this function should be called to allocate a new cache entry if the cache is not empty
void RingbufferCacheManager::add_manager_entry(uint64_t pgm_id, uint32_t offset, uint32_t length) {
    auto idx = manager_.next_idx;
    if (idx == manager_.oldest_idx) {
        // if the next index is the same as the oldest index, then we need to invalidate the oldest entry
        invalidate_manager_entry();
    }
    add_manager_entry_no_evict(pgm_id, offset, length);
}

// this function will invalidate the oldest entry in the ringbuffer manager
// caution: oldest entry must be valid
void RingbufferCacheManager::invalidate_manager_entry() {
    auto& entry = manager_.entry[manager_.oldest_idx];
    auto valid_idx = entry.valid_idx;
    TT_ASSERT(valid_idx < valid_.size(), "RingbufferCacheManager invalidation: pgm_id out of range");
    TT_ASSERT(
        valid_[valid_idx] != RingbufferCacheManager::invalid_cache_entry_,
        "RingbufferCacheManager invalidation: entry not valid (mgr: oldest_idx:{}, next_idx:{}, offset:{}, length:{}, "
        "valid idx (pgm id):{}, valid[pgm id]:{})",
        manager_.oldest_idx,
        manager_.next_idx,
        entry.offset,
        entry.length,
        valid_idx,
        valid_[valid_idx]);
    // invalidate the entry
    valid_[valid_idx] = RingbufferCacheManager::invalid_cache_entry_;
    manager_.update_oldest_idx();
}

// this function will invalidate all oldest entries until the cache wraps around
// caution: should not be called if the cache is empty
bool RingbufferCacheManager::invalidate_oldest_until_wraparound() {
    if (manager_.oldest_idx == manager_.next_idx) {
        invalidate_manager_entry();
    }
    while ((manager_.entry[manager_.oldest_idx].offset >= manager_.next_block_offset) and
           (manager_.oldest_idx != manager_.next_idx)) {
        invalidate_manager_entry();
    }
    // true ==> cache is empty and caller should call add_manager_entry_no_evict
    return manager_.oldest_idx == manager_.next_idx;
}
// this function is used to invalidate the oldest entry until there is sufficient space
// caution: should not be called if the cache is empty
bool RingbufferCacheManager::invalidate_sufficient_blocks(int required_space, int offset) {
    if (manager_.oldest_idx == manager_.next_idx) {
        invalidate_manager_entry();
    }
    int oldest_block_offset = manager_.entry[manager_.oldest_idx].offset;
    int available_space = oldest_block_offset - offset;
    while (available_space < required_space and manager_.oldest_idx != manager_.next_idx) {
        available_space += manager_.entry[manager_.oldest_idx].length;
        invalidate_manager_entry();  // invalidate oldest entry
    }

    // true ==> cache is empty and caller should call add_manager_entry_no_evict
    return manager_.oldest_idx == manager_.next_idx;
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
        valid_.resize(pgm_id + 1, RingbufferCacheManager::invalid_cache_entry_);
        manager_.entry.resize(std::min(cache_manager_initial_entry_size_, cache_size_blocks_));
        add_manager_entry_no_evict(pgm_id, 0, required_space);
        query_result.is_cached = false;
        query_result.offset = 0;
        return query_result;
    } else if (pgm_id >= valid_.size()) {
        valid_.resize(
            std::max((uint64_t)cache_manager_initial_entry_size_, pgm_id + 1),
            RingbufferCacheManager::invalid_cache_entry_);
    } else if (valid_[pgm_id] != invalid_cache_entry_) {
        unsigned mgr_idx = valid_[pgm_id];
        TT_ASSERT(
            mgr_idx < manager_.entry.size(),
            "RingbufferCacheManager invalid cache hit: manager index out of bounds. pgm_id:{}, valid idx:{}, entry "
            "size:{}",
            pgm_id,
            mgr_idx,
            manager_.entry.size());
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
    int next_block_offset = manager_.next_block_offset;
    int oldest_block_offset = manager_.entry[manager_.oldest_idx].offset;
    int free_space_to_end = cache_size_blocks_ - next_block_offset;
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

void RingbufferCacheManager::reset() {
    this->manager_.clear();
    std::vector<RingbufferCacheManagerEntry> temp_entry;
    this->manager_.entry.swap(temp_entry);

    std::vector<int32_t> temp_valid;
    this->valid_.swap(temp_valid);
}

void swap(RingbufferCacheManager& a, RingbufferCacheManager& b) noexcept {
    TT_ASSERT(
        a.cache_block_sizeB_ == b.cache_block_sizeB_,
        "Ringbuffer cache block size mismatch: {} != {}",
        a.cache_block_sizeB_,
        b.cache_block_sizeB_);
    TT_ASSERT(
        a.cache_size_blocks_ == b.cache_size_blocks_,
        "Ringbuffer cache size mismatch: {} != {}",
        a.cache_size_blocks_,
        b.cache_size_blocks_);
    TT_ASSERT(
        a.cache_manager_initial_entry_size_ == b.cache_manager_initial_entry_size_,
        "Ringbuffer cache manager initial entry size mismatch: {} != {}",
        a.cache_manager_initial_entry_size_,
        b.cache_manager_initial_entry_size_);
    using std::swap;
    a.valid_.swap(b.valid_);
    a.manager_.swap(b.manager_);
}

void RingbufferCacheManager::InternalManager::clear() {
    this->oldest_idx = 0;
    this->next_idx = 0;
    this->next_block_offset = 0;
}

RingbufferCacheManager::InternalManager& RingbufferCacheManager::InternalManager::swap(InternalManager& b) {
    using std::swap;
    swap(this->entry, b.entry);
    swap(this->oldest_idx, b.oldest_idx);
    swap(this->next_idx, b.next_idx);
    swap(this->next_block_offset, b.next_block_offset);
    return *this;
}

}  // namespace tt::tt_metal
