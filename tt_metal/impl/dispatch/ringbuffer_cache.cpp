// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/ringbuffer_cache.hpp"

template <int CACHE_BLOCK_SIZE, int CACHE_SIZE>
void RingbufferCacheManager::add_manager_entry(uint16_t pgm_id, uint32_t offset, uint32_t length, int idx) {
    ASSERT(offset + length <= CACHE_SIZE, "RingbufferCacheManager new allocation: offset + length > cache size");
    auto& entry = this->manager_.entry[idx];
    entry.offset = offset;
    entry.length = length;
    entry.valid_idx = pgm_id;
    this->valid_[pgm_id] = idx;
    this->manager_.next_block_offset = (offset + length) & (cache_size_ - 1);
    this->manager_.next_idx = (idx + 1) & (cache_size_ - 1);
    if (offset == 0) {
        this->manager_.zero_offset_idx = idx;
    }
}

template <int CACHE_BLOCK_SIZE, int CACHE_SIZE>
void RingbufferCacheManager::invalidate_manager_entry(int idx) {
    auto& entry = this->manager_.entry[idx];
    auto pgm_id = entry.valid_idx;
    ASSERT(pgm_id < this->valid_.size(), "RingbufferCacheManager invalidation: pgm_id out of range");
    this->valid_[pgm_id] = -1;
    this->manager_.oldest_idx = (idx + 1) & (cache_size_ - 1);
}

template <int CACHE_BLOCK_SIZE, int CACHE_SIZE>
std::optional<std::pair<bool, uint32_t>> RingbufferCacheManager::is_cached(uint16_t pgm_id, size_t lengthB) {
    // Check if the entry is already present
    if (valid_.size() > pgm_id and valid_[pgm_id] != -1) {
        int mgr_idx = valid_[pgm_id];
        auto mgr_entry = manager_.entry[mgr_idx];
        ASSERT(mgr_entry.length * cache_block_sizeB >= lengthB);
        return {true, mgr_entry.offset};
    }
    // find space in RB for new entry, or if full, then invalidate oldest entry(ies)
    const int required_space = (lengthB + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE;
    if (__builtin_expect(required_space > cache_size_, false)) {
        return std::nullopt;  // cannot fit in cache
    }

    uint32_t cache_offset;
    int next_block_offset = this->manager_.next_block_offset;
    size_t oldest_idx = this->manager_.oldest_idx;
    int oldest_block_offset = this->manager_.entry[oldest_idx].offset;
    size_t free_space_to_end = cache_size_ - next_block_offset;
    if (next_block_offset > oldest_block_offset) {
        // cache is not full, but we need to check if there is enough space
        if (free_space_to_end >= required_space) {
            add_manager_entry(pgm_id, next_block_offset, required_space);
            cache_offset = next_block_offset;
        } else if (oldest_block_offset > required_space) {
            // cache is not full, but must wraparound for sufficient space
            add_manager_entry(pgm_id, 0, required_space);
            cache_offset = 0;
        } else {
            // cache is full, need to invalidate oldest entry(ies)
            size_t available_space = oldest_block_offset;
            while (available_space < required_space) {
                available_space += this->manager_.entry[this->manager_.oldest_idx].length;
                // invalidate oldest entry
                invalidate_manager_entry();
            }
            add_manager_entry(pgm_id, 0, required_space);
            cache_offset = 0;
        }
    } else {  // oldest_block_offset >= next_block_offset
        if (free_space_to_end >= required_space) {
            size_t available_space = oldest_block_offset - next_block_offset;
            while (available_space < required_space) {
                available_space += this->manager_.entry[this->manager_.oldest_idx].length;
                // invalidate oldest entry
                invalidate_manager_entry();
            }
            add_manager_entry(pgm_id, next_block_offset, required_space);
            cache_offset = next_block_offset;
        } else {
            // cache is not full, but must wraparound for sufficient space. We will invalidate some intermediate entries
            // while possibly leaving older entries intact
            size_t freed_space = 0;
            uint16_t wrap_idx = this->manager_.zero_offset_idx;
            while (freed_space < required_space) {
                freed_space += this->manager_.entry[wrap_idx].length;
                // invalidate entry at wrap_idx
                invalidate_manager_entry(wrap_idx);
                wrap_idx = (wrap_idx + 1) & (cache_size_ - 1);
            }
            add_manager_entry(pgm_id, 0, required_space);
            if (this->manager_.next_block_offset > oldest_block_offset) {
                // this means we freed up the oldest cache entry, and must upate the oldest_idx
                this->manager_.oldest_idx = wrap_idx;
            }
            cache_offset = 0;
        }
    }

    return {false, cache_offset};
}
