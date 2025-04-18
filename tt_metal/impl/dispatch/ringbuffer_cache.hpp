// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ringbuffer cache implementation

#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <bit>
#include <climits>

namespace tt::tt_metal {

/*! @brief Ringbuffer cache manager
 *  This class manages the ringbuffer cache for the prefetcher.
 *  It keeps track of the entries in the ringbuffer and their offsets.
 *  The ringbuffer is divided into blocks of size cache_block_sizeB.
 *  The cache manager keeps track of the entries in the ringbuffer and their offsets.
 */
template <int CACHE_BLOCK_SIZE, int CACHE_SIZE>
class RingbufferCacheManager {
public:
    RingbufferCacheManager() {
        std::fill(this->valid_.begin(), this->valid_.end(), -1);
        // Initialize the cache manager entries
        static_assert((CACHE_SIZE & (CACHE_SIZE - 1)) == 0, "Ringbuffer manager array size must be a power of 2");
        std::fill(this->manager_.entry.begin(), this->manager_.entry.end(), 0);
        this->manager_.entry[0].length = CACHE_SIZE;  // to get over the first allocation without special handling
    };

    ~RingbufferCacheManager() = default;

    /*! @brief Check if the program is present in the ringbuffer
     *  Add a new entry to the ringbuffer manager if not present
     *  @param pgm_id program id
     *  @param lengthB size of program in bytes
     *  @return std::pair of {bool, offset}, where bool::false indicates that the program was added
     */
    std::optional<std::pair<bool, uint32_t>> is_cached(uint16_t pgm_id, size_t lengthB);

    /*! @brief if program is present in rb, then return its offset
     *   @param pgm_id program id
     *   @return offset of the program in ringbuffer, or -1 if not present
     */
    // int32_t get_ringbuffer_offset (uint64_t pgm_id);

private:
    constexpr static size_t offset_width_ = std::bit_width(CACHE_SIZE - 1);
    constexpr static size_t length_width_ = std::bit_width(CACHE_SIZE - 1);
    constexpr static size_t valid_width_ = CHAR_BIT * sizeof(uint32_t) - offset_width_ - length_width_;
    static_assert(valid_width_ > 11, "valid_width must be greater than 11 bits to cover all program ids");

    /*! @brief cache manager entry */
    struct RingbufferCacheManagerEntry {
        uint32_t offset : offset_width_;    // offset in ringbuffer
        uint32_t length : length_width_;    // length of the program in blocks
        uint32_t valid_idx : valid_width_;  // index into the valid_ array
    };

    /*! @brief cache manager  */
    struct {
        std::array<RingbufferCacheManagerEntry, CACHE_SIZE> entry;
        // the following indexes are for the ringbuffer manager, not the ringbuffer
        uint16_t oldest_idx{0};  // update this whenever an entry is evicted from cache
        uint16_t next_idx{0};    // update this whenever an entry is added to cache
        // the following is saved for convenience
        uint16_t next_block_offset{0};  // offset of the next block to allocate in ringbuffer
        // the following is needed when allocations must wraparound
        uint16_t zero_offset_idx{0};  // index in entry array for block zero allocation
    } manager_;

    /*! @brief indexed by program id. Contains index to entry in cache manager if cache hit */
    std::vector<int16_t> valid_;

    void add_manager_entry(uint16_t pgm_id, uint32_t offset, uint32_t length, uint16_t idx);
    void add_manager_entry(uint16_t pgm_id, uint32_t offset, uint32_t length) {
        add_manager_entry(pgm_id, offset, length, this->manager_.next_idx);
    }
    void invalidate_manager_entry(uint16_t idx);
    void invalidate_manager_entry(void) { invalidate_manager_entry(this->manager_.oldest_idx); }
};

}  // namespace tt::tt_metal
