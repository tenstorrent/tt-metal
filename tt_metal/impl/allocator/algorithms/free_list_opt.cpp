// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/allocator/algorithms/free_list_opt.hpp"

#include <assert.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "allocator_algorithm.hpp"

inline size_t intlg2(size_t n) {
    // std::log2() is slow
    size_t count = 0;
    while (n > 0) {
        count++;
        n >>= 1;
    }
    return count;
}

inline size_t num_segerated_classes(size_t max_size_bytes, size_t size_segregated_base) {
    size_t n = max_size_bytes / size_segregated_base;
    ssize_t count = intlg2(n);
    // 128MB as the last seggregated class size should be enough
    // avoid having too many classes as iterating them is not free
    ssize_t max_count = intlg2(128 * 1024 * 1024 / size_segregated_base);
    return std::clamp(count, ssize_t{2}, max_count);
}

namespace tt {

namespace tt_metal {

namespace allocator {

FreeListOpt::FreeListOpt(
    DeviceAddr max_size_bytes,
    DeviceAddr offset_bytes,
    DeviceAddr min_allocation_size,
    DeviceAddr alignment,
    SearchPolicy policy) :
    policy_(policy),
    size_segregated_count((num_segerated_classes(max_size_bytes, size_segregated_base))),
    Algorithm(max_size_bytes, offset_bytes, min_allocation_size, alignment) {
    // Reduce reallocations by reserving memory for free list components
    constexpr size_t initial_block_count = 64;
    block_address_.reserve(initial_block_count);
    block_size_.reserve(initial_block_count);
    block_prev_block_.reserve(initial_block_count);
    block_next_block_.reserve(initial_block_count);
    block_is_allocated_.reserve(initial_block_count);
    free_meta_block_indices_.reserve(initial_block_count);
    meta_block_is_allocated_.reserve(initial_block_count);
    free_blocks_segregated_by_size_.resize(size_segregated_count);
    for (auto& free_blocks : free_blocks_segregated_by_size_) {
        free_blocks.reserve(initial_block_count);
    }
    allocated_block_table_.resize(n_alloc_table_buckets);
    for (auto& bucket : allocated_block_table_) {
        bucket.reserve(n_alloc_table_init_bucket_size);
    }

    init();
}

void FreeListOpt::init() {
    max_size_bytes_ += shrink_size_;
    shrink_size_ = 0;

    block_address_.clear();
    block_size_.clear();
    block_prev_block_.clear();
    block_next_block_.clear();
    block_is_allocated_.clear();
    free_meta_block_indices_.clear();
    meta_block_is_allocated_.clear();
    for (auto& bucket : allocated_block_table_) {
        bucket.clear();
    }
    for (auto& free_blocks : free_blocks_segregated_by_size_) {
        free_blocks.clear();
    }

    // Create a single block that spans the entire memory
    block_address_.push_back(0);
    block_size_.push_back(max_size_bytes_);
    block_prev_block_.push_back(-1);
    block_next_block_.push_back(-1);
    block_is_allocated_.push_back(false);
    meta_block_is_allocated_.push_back(true);
    free_blocks_segregated_by_size_[get_size_segregated_index(max_size_bytes_)].push_back(0);
}

std::optional<DeviceAddr> FreeListOpt::allocate(DeviceAddr size_bytes, bool bottom_up, DeviceAddr address_limit) {
    DeviceAddr alloc_size = align(std::max(size_bytes, min_allocation_size_));

    // Find the best free block by looking at the segregated free blocks, if we can find a block in it's size class
    // we can be confident that it's the best block to allocate from. Else, look at the next size class. However the
    // blocks within a size class are not sorted by size, so we may not always find the best block.

    ssize_t target_block_index = -1;
    size_t size_segregated_index = get_size_segregated_index(alloc_size);
    TT_ASSERT(size_segregated_index < size_segregated_count, "Size segregated index out of bounds");
    std::vector<size_t>* segregated_list = nullptr;
    size_t segregated_item_index = 0;
    DeviceAddr best_address = bottom_up ? ~(DeviceAddr)0 : 0;

    for (size_t i = size_segregated_index; i < free_blocks_segregated_by_size_.size(); i++) {
        auto& free_blocks = free_blocks_segregated_by_size_[i];
        ssize_t increment = bottom_up ? 1 : -1;
        for (ssize_t j = bottom_up ? 0 : free_blocks.size() - 1; j >= 0 && j < free_blocks.size(); j += increment) {
            size_t block_index = free_blocks[j];
            if (policy_ == SearchPolicy::BEST) {
                if (block_size_[block_index] == alloc_size) {
                    target_block_index = block_index;
                    segregated_list = &free_blocks;
                    segregated_item_index = j;
                    break;
                } else if (
                    block_size_[block_index] >= alloc_size &&
                    (target_block_index == -1 || block_size_[block_index] < block_size_[target_block_index])) {
                    target_block_index = block_index;
                    segregated_list = &free_blocks;
                    segregated_item_index = j;
                }
                if (target_block_index != -1) {
                    break;
                }
            } else {
                if (block_size_[block_index] < alloc_size) {
                    continue;
                }

                bool address_better =
                    bottom_up ? block_address_[block_index] < best_address : block_address_[block_index] > best_address;
                if (target_block_index == -1 || address_better) {
                    target_block_index = block_index;
                    segregated_list = &free_blocks;
                    segregated_item_index = j;
                    best_address = block_address_[block_index];
                    break;
                }
            }
        }
    }

    if (target_block_index == -1) {
        return std::nullopt;
    }
    TT_ASSERT(segregated_list != nullptr, "Segregated list is null");
    TT_ASSERT(segregated_item_index < segregated_list->size(), "Segregated item index out of bounds");
    TT_ASSERT(
        block_is_allocated_[target_block_index] == false, "Block we are trying allocate from is already allocated");
    segregated_list->erase(segregated_list->begin() + segregated_item_index);

    // Allocate the block
    size_t offset = 0;
    if (!bottom_up) {
        offset = block_size_[target_block_index] - alloc_size;
    }
    size_t allocated_block_index = allocate_in_block(target_block_index, alloc_size, offset);
    DeviceAddr start_address = block_address_[allocated_block_index];
    if (start_address + offset_bytes_ < address_limit) {
        TT_THROW(
            "Out of Memory: Cannot allocate at an address below {}. Allocation at {}",
            address_limit,
            start_address + offset_bytes_);
    }
    update_lowest_occupied_address(start_address);
    return start_address + offset_bytes_;
}

std::optional<DeviceAddr> FreeListOpt::allocate_at_address(DeviceAddr absolute_start_address, DeviceAddr size_bytes) {
    // Nothing we can do but scan the free list
    size_t alloc_size = align(std::max(size_bytes, min_allocation_size_));
    ssize_t target_block_index = -1;
    DeviceAddr start_address = absolute_start_address - offset_bytes_;
    for (size_t i = 0; i < block_address_.size(); i++) {
        size_t block_start = block_address_[i];
        size_t block_end = block_start + block_size_[i];
        if (start_address >= block_start && start_address + alloc_size <= block_end) {
            target_block_index = i;
            break;
        }
    }

    if (target_block_index == -1 || block_is_allocated_[target_block_index]) {
        return std::nullopt;
    }

    // Find the relevant size segregated list
    size_t size_segregated_index = get_size_segregated_index(block_size_[target_block_index]);
    std::vector<size_t>& segregated_list = free_blocks_segregated_by_size_[size_segregated_index];
    auto it = std::find(segregated_list.begin(), segregated_list.end(), target_block_index);
    TT_ASSERT(it != segregated_list.end(), "Block not found in size segregated list");
    segregated_list.erase(it);

    size_t offset = start_address - block_address_[target_block_index];
    size_t alloc_block_index = allocate_in_block(target_block_index, alloc_size, offset);
    update_lowest_occupied_address(start_address);
    return absolute_start_address;
}

size_t FreeListOpt::allocate_in_block(size_t block_index, DeviceAddr alloc_size, size_t offset) {
    if (block_size_[block_index] == alloc_size && offset == 0) {
        block_is_allocated_[block_index] = true;
        insert_block_to_alloc_table(block_address_[block_index], block_index);
        return block_index;
    }

    bool left_aligned = offset == 0;
    bool right_aligned = offset + alloc_size == block_size_[block_index];

    // Create free space if not left/right aligned
    if (!left_aligned) {
        size_t free_block_size = offset;
        DeviceAddr free_block_address = block_address_[block_index];
        ssize_t prev_block = block_prev_block_[block_index];
        ssize_t next_block = block_next_block_[block_index];
        block_size_[block_index] -= offset;
        block_address_[block_index] += offset;
        size_t new_block_index = alloc_meta_block(free_block_address, free_block_size, prev_block, block_index, false);
        if (prev_block != -1) {
            block_next_block_[prev_block] = new_block_index;
        }
        block_prev_block_[block_index] = new_block_index;

        insert_block_to_segregated_list(new_block_index);
    }

    if (!right_aligned) {
        size_t free_block_size = block_size_[block_index] - alloc_size;
        DeviceAddr free_block_address = block_address_[block_index] + alloc_size;
        ssize_t prev_block = block_index;
        ssize_t next_block = block_next_block_[block_index];
        block_size_[block_index] -= free_block_size;
        size_t new_block_index = alloc_meta_block(free_block_address, free_block_size, prev_block, next_block, false);
        if (next_block != -1) {
            block_prev_block_[next_block] = new_block_index;
        }
        block_next_block_[block_index] = new_block_index;

        insert_block_to_segregated_list(new_block_index);
    }
    block_is_allocated_[block_index] = true;
    insert_block_to_alloc_table(block_address_[block_index], block_index);

    return block_index;
}

void FreeListOpt::deallocate(DeviceAddr absolute_address) {
    // The existing FreeList implementation does not check if the address is actually allocated. Just return if it's not
    // Do we want to keep this behavior?

    DeviceAddr addr = absolute_address - offset_bytes_;
    auto block_index_opt = get_and_remove_from_alloc_table(addr);
    if (!block_index_opt.has_value()) {
        return;
    }
    size_t block_index = *block_index_opt;
    block_is_allocated_[block_index] = false;
    ssize_t prev_block = block_prev_block_[block_index];
    ssize_t next_block = block_next_block_[block_index];

    // Merge with previous block if it's free
    if (prev_block != -1 && !block_is_allocated_[prev_block]) {
        // Look into the size segregated list to remove the block
        size_t size_segregated_index = get_size_segregated_index(block_size_[prev_block]);
        std::vector<size_t>& segregated_list = free_blocks_segregated_by_size_[size_segregated_index];
        auto it = std::find(segregated_list.begin(), segregated_list.end(), prev_block);
        TT_ASSERT(
            it != segregated_list.end(),
            "Prev block {} not found in size segregated list during deallocation of block {}",
            prev_block,
            block_index);
        segregated_list.erase(it);

        block_size_[prev_block] += block_size_[block_index];
        block_next_block_[prev_block] = next_block;
        if (next_block != -1) {
            block_prev_block_[next_block] = prev_block;
        }
        free_meta_block(block_index);
        block_index = prev_block;
    }

    // Merge with next block if it's free
    if (next_block != -1 && !block_is_allocated_[next_block]) {
        // Look into the size segregated list to remove the block
        size_t size_segregated_index = get_size_segregated_index(block_size_[next_block]);
        std::vector<size_t>& segregated_list = free_blocks_segregated_by_size_[size_segregated_index];
        auto it = std::find(segregated_list.begin(), segregated_list.end(), next_block);
        TT_ASSERT(
            it != segregated_list.end(),
            "Next block {} not found in size segregated list during deallocation of block {}",
            next_block,
            block_index);
        segregated_list.erase(it);

        block_size_[block_index] += block_size_[next_block];
        block_next_block_[block_index] = block_next_block_[next_block];
        if (block_next_block_[next_block] != -1) {
            block_prev_block_[block_next_block_[next_block]] = block_index;
        }
        free_meta_block(next_block);
    }

    TT_ASSERT(lowest_occupied_address_.has_value(), "Lowest occupied address should have a value");
    if (addr <= *lowest_occupied_address_) {
        lowest_occupied_address_ = std::nullopt;
        ssize_t curr_block_index = block_next_block_[block_index];
        while (curr_block_index != -1) {
            if (block_is_allocated_[curr_block_index]) {
                lowest_occupied_address_ = block_address_[curr_block_index];
                break;
            }
            curr_block_index = block_next_block_[curr_block_index];
        }
    }
    // Update the segregated list
    insert_block_to_segregated_list(block_index);
}

std::vector<std::pair<DeviceAddr, DeviceAddr>> FreeListOpt::available_addresses(DeviceAddr size_bytes) const {
    size_t alloc_size = align(std::max(size_bytes, min_allocation_size_));
    size_t size_segregated_index = get_size_segregated_index(alloc_size);
    std::vector<std::pair<DeviceAddr, DeviceAddr>> addresses;

    for (size_t i = size_segregated_index; i < size_segregated_count; i++) {
        for (size_t j = 0; j < free_blocks_segregated_by_size_[i].size(); j++) {
            size_t block_index = free_blocks_segregated_by_size_[i][j];
            if (block_size_[block_index] >= alloc_size) {
                addresses.push_back(
                    {block_address_[block_index], block_address_[block_index] + block_size_[block_index]});
            }
        }
    }
    return addresses;
}

size_t FreeListOpt::alloc_meta_block(
    DeviceAddr address, DeviceAddr size, ssize_t prev_block, ssize_t next_block, bool is_allocated) {
    size_t idx;
    if (free_meta_block_indices_.empty()) {
        idx = block_address_.size();
        block_address_.push_back(address);
        block_size_.push_back(size);
        block_prev_block_.push_back(prev_block);
        block_next_block_.push_back(next_block);
        block_is_allocated_.push_back(is_allocated);
        meta_block_is_allocated_.push_back(true);
    } else {
        idx = free_meta_block_indices_.back();
        free_meta_block_indices_.pop_back();
        block_address_[idx] = address;
        block_size_[idx] = size;
        block_prev_block_[idx] = prev_block;
        block_next_block_[idx] = next_block;
        block_is_allocated_[idx] = is_allocated;
        meta_block_is_allocated_[idx] = true;
    }
    return idx;
}

void FreeListOpt::free_meta_block(size_t block_index) {
    free_meta_block_indices_.push_back(block_index);
    meta_block_is_allocated_[block_index] = false;
}

void FreeListOpt::clear() { init(); }

Statistics FreeListOpt::get_statistics() const {
    // TODO: Cache the statistics
    size_t total_allocated_bytes = 0;
    size_t total_free_bytes = 0;
    size_t largest_free_block_bytes = 0;
    std::vector<uint32_t> largest_free_block_addrs;

    for (size_t i = 0; i < block_address_.size(); i++) {
        if (block_is_allocated_[i]) {
            total_allocated_bytes += block_size_[i];
        } else {
            total_free_bytes += block_size_[i];
            if (block_size_[i] >= largest_free_block_bytes) {
                largest_free_block_bytes = block_size_[i];
                // XXX: This is going to overflow
                largest_free_block_addrs.push_back(block_address_[i] + offset_bytes_);
            }
        }
    }

    if (total_allocated_bytes == 0) {
        total_free_bytes = max_size_bytes_;
        largest_free_block_bytes = max_size_bytes_;
    }

    return Statistics{
        .total_allocatable_size_bytes = max_size_bytes_,
        .total_allocated_bytes = total_allocated_bytes,
        .total_free_bytes = total_free_bytes,
        .largest_free_block_bytes = largest_free_block_bytes,
        // Why do we need largest_free_block_addrs? Without it the entire loop can be removed
        // and statistics can be tracked during allocation and deallocation
        .largest_free_block_addrs = std::move(largest_free_block_addrs),
    };
}

void FreeListOpt::dump_blocks(std::ostream& out) const {
    out << "FreeListOpt allocator info:" << std::endl;
    out << "segregated free blocks by size:" << std::endl;
    for (size_t i = 0; i < free_blocks_segregated_by_size_.size(); i++) {
        if (i != free_blocks_segregated_by_size_.size() - 1) {
            out << "  Size class " << i << ": (" << size_t(size_segregated_base * (size_t{1} << i)) << " - "
                << size_t(size_segregated_base * (size_t{1} << (i + 1))) << ") blocks: ";
        } else {
            out << "  Size class " << i << ": (" << size_t(size_segregated_base * (size_t{1} << i))
                << " - inf) blocks: ";
        }
        for (size_t j = 0; j < free_blocks_segregated_by_size_[i].size(); j++) {
            out << free_blocks_segregated_by_size_[i][j] << " ";
        }

        out << std::endl;
    }

    out << "Free slots in block table: ";
    for (size_t i = 0; i < free_meta_block_indices_.size(); i++) {
        out << free_meta_block_indices_[i] << " ";
    }
    out << std::endl;

    out << "Block table:" << std::endl;
    auto leftpad = [](std::string str, size_t width) {
        if (str.size() >= width) {
            return str;
        }
        return std::string(width - str.size(), ' ') + str;
    };
    auto leftpad_num = [leftpad](auto num, size_t width) {
        // HACK: -1 for us means none
        if (num == -1) {
            return leftpad("none", width);
        }
        return leftpad(std::to_string(num), width);
    };
    const size_t pad = 12;
    std::array<std::string, 6> headers = {"Block", "Address", "Size", "PrevID", "NextID", "Allocated"};
    for (auto& header : headers) {
        out << leftpad(header, pad) << " ";
    }
    out << std::endl;
    for (size_t i = 0; i < block_address_.size(); i++) {
        if (!meta_block_is_allocated_[i]) {
            continue;
        }
        out << leftpad_num(i, pad) << " " << leftpad_num(block_address_[i], pad) << " "
            << leftpad_num(block_size_[i], pad) << " " << leftpad_num(block_prev_block_[i], pad) << " "
            << leftpad_num(block_next_block_[i], pad) << " " << leftpad(block_is_allocated_[i] ? "yes" : "no", pad)
            << std::endl;
    }
}

MemoryBlockTable FreeListOpt::get_memory_block_table() const {
    MemoryBlockTable blocks;

    for (size_t i = 0; i < block_address_.size(); i++) {
        std::unordered_map<std::string, std::string> block_entry;

        if (!meta_block_is_allocated_[i]) {
            continue;
        }

        block_entry["blockID"] = std::to_string(i);
        block_entry["address"] = std::to_string(block_address_[i]);  // bytes
        block_entry["size"] = std::to_string(block_size_[i]);        // bytes
        block_entry["prevID"] = std::to_string(block_prev_block_[i]);
        block_entry["nextID"] = std::to_string(block_next_block_[i]);
        block_entry["allocated"] = block_is_allocated_[i] ? "yes" : "no";
        blocks.push_back(block_entry);
    }

    return blocks;
}

void FreeListOpt::shrink_size(DeviceAddr shrink_size, bool bottom_up) {
    if (shrink_size == 0) {
        return;
    }
    TT_FATAL(bottom_up, "Shrinking from the top is currently not supported");
    TT_FATAL(
        shrink_size <= this->max_size_bytes_,
        "Shrink size {} must be smaller than max size {}",
        shrink_size,
        max_size_bytes_);

    // loop and scan the block list to find if the shrink cut into any allocated block
    size_t block_to_shrink = -1;
    DeviceAddr shrunk_address = shrink_size_ + shrink_size;
    // TODO: There must be a way to force the beginning of all blocks be at index 0
    for (size_t i = 0; i < block_address_.size(); i++) {
        if (!meta_block_is_allocated_[i]) {
            continue;
        } else if (block_is_allocated_[i]) {
            TT_FATAL(
                block_address_[i] >= shrunk_address,
                "Shrink size {} cuts into allocated block at address {}",
                shrunk_address,
                block_address_[i]);
        } else if (block_address_[i] <= shrunk_address && block_address_[i] + block_size_[i] >= shrunk_address) {
            block_to_shrink = i;
            break;
        }
    }

    TT_FATAL(block_to_shrink != -1, "Shrink size {} does not align with any block. This must be a bug", shrunk_address);

    // Find the relevant size segregated list
    size_t size_segregated_index = get_size_segregated_index(block_size_[block_to_shrink]);
    std::vector<size_t>& segregated_list = free_blocks_segregated_by_size_[size_segregated_index];
    for (size_t i = 0; i < segregated_list.size(); i++) {
        if (segregated_list[i] == block_to_shrink) {
            segregated_list.erase(segregated_list.begin() + i);
            break;
        }
    }

    // Shrink the block
    block_size_[block_to_shrink] -= shrink_size;
    max_size_bytes_ -= shrink_size;
    shrink_size_ += shrink_size;
    if (block_size_[block_to_shrink] == 0) {
        block_prev_block_[block_next_block_[block_to_shrink]] = block_prev_block_[block_to_shrink];
        free_meta_block(block_to_shrink);
    } else {
        block_address_[block_to_shrink] += shrink_size;
        insert_block_to_segregated_list(block_to_shrink);
    }
}

void FreeListOpt::reset_size() {
    if (shrink_size_ == 0) {
        return;
    }

    // Create a new block, mark it as allocated and deallocate the old block so coalescing can happen
    ssize_t lowest_block_index = -1;
    for (size_t i = 0; i < block_address_.size(); i++) {
        if (!meta_block_is_allocated_[i]) {
            continue;
        }
        if (block_address_[i] == shrink_size_) {
            lowest_block_index = i;
            break;
        }
    }
    TT_ASSERT(lowest_block_index != -1, "Lowest block not found during reset size");

    // There 2 cases to consider:
    // 1. The lowest block is is free, which means we can just modify it's attributes
    // 2. The lowest block is allocated, which means we need to create a new block and deallocate the old one
    if (!block_is_allocated_[lowest_block_index]) {
        auto* segregated_list =
            &free_blocks_segregated_by_size_[get_size_segregated_index(block_size_[lowest_block_index])];
        for (size_t i = 0; i < segregated_list->size(); i++) {
            if ((*segregated_list)[i] == lowest_block_index) {
                segregated_list->erase(segregated_list->begin() + i);
                break;
            }
        }
        block_size_[lowest_block_index] += shrink_size_;
        block_address_[lowest_block_index] = 0;
        insert_block_to_segregated_list(lowest_block_index);
    } else {
        size_t new_block_index = alloc_meta_block(0, shrink_size_, -1, lowest_block_index, false);
        TT_ASSERT(block_prev_block_[lowest_block_index] == -1, "Lowest block should not have a previous block");
        block_prev_block_[lowest_block_index] = new_block_index;
        insert_block_to_segregated_list(new_block_index);
    }

    max_size_bytes_ += shrink_size_;
    shrink_size_ = 0;
}

void FreeListOpt::insert_block_to_segregated_list(size_t block_index) {
    const size_t size_segregated_index = get_size_segregated_index(block_size_[block_index]);
    auto& free_blocks = free_blocks_segregated_by_size_[size_segregated_index];
    // Pushing to the back is faster than sorted insertion. But it increases fragmentation
    // free_blocks.push_back(block_index);
    // The overhead is not worth it in benchmarks. Need real world data to confirm. But certainly it'll help with
    // fragmentation
    std::vector<size_t>::iterator it;
    // from experience, the lower bound is only faster after a certain number of elements
    if (free_blocks.size() < 30) {
        for (it = free_blocks.begin(); it != free_blocks.end(); it++) {
            if (block_address_[*it] > block_address_[block_index]) {
                break;
            }
        }
    } else {
        it = std::lower_bound(free_blocks.begin(), free_blocks.end(), block_index, [this](size_t a, size_t b) {
            return block_address_[a] < block_address_[b];
        });
    }
    free_blocks.insert(it, block_index);
}

inline size_t FreeListOpt::hash_device_address(DeviceAddr address) {
    // HACK: This hash is critical for performance, empirically found to be good for
    // the specific usecase
    return ((address) ^ (address >> 12) * 3) % n_alloc_table_buckets;
}
void FreeListOpt::insert_block_to_alloc_table(DeviceAddr address, size_t block_index) {
    size_t bucket = hash_device_address(address);
    allocated_block_table_[bucket].emplace_back(address, block_index);
}
bool FreeListOpt::is_address_in_alloc_table(DeviceAddr address) const {
    size_t bucket = hash_device_address(address);
    for (const auto& [addr, block_index] : allocated_block_table_[bucket]) {
        if (addr == address) {
            return true;
        }
    }
    return false;
}
std::optional<size_t> FreeListOpt::get_and_remove_from_alloc_table(DeviceAddr address) {
    size_t bucket = hash_device_address(address);
    // It's common to deallocate the last allocated block, so search from the back
    for (ssize_t i = allocated_block_table_[bucket].size() - 1; i >= 0; i--) {
        if (allocated_block_table_[bucket][i].first == address) {
            auto res = allocated_block_table_[bucket][i].second;
            allocated_block_table_[bucket].erase(allocated_block_table_[bucket].begin() + i);
            return res;
        }
    }
    return std::nullopt;
}

void FreeListOpt::update_lowest_occupied_address(DeviceAddr address) {
    if (!lowest_occupied_address_.has_value() || address < lowest_occupied_address_.value()) {
        lowest_occupied_address_ = address;
    }
}

}  // namespace allocator
}  // namespace tt_metal
}  // namespace tt
