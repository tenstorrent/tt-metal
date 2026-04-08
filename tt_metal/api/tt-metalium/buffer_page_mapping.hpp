// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>

#include <vector>
#include <optional>
#include <unordered_map>
#include <array>

namespace tt::tt_metal {

struct UncompressedBufferPageMapping {
    // Represents a page on device which doesn't match any host page within core_host_page_indices.
    static constexpr uint32_t PADDING = std::numeric_limits<uint32_t>::max();

    std::vector<CoreCoord> all_cores;
    // For each core, a vector of host page indices (or PADDING if there's no corresponding host page).
    std::vector<std::vector<uint32_t>> core_host_page_indices;
};

// Represents a contiguous range of device pages for a single core.
struct BufferCorePageMapping {
    struct ContiguousHostPages {
        uint32_t device_page_offset = 0;
        uint32_t host_page_start = 0;
        uint32_t num_pages = 0;
    };

    uint32_t device_start_page = 0;
    uint32_t num_pages = 0;
    // Vector of contiguous host page ranges within this contiguous range of device pages.
    std::vector<ContiguousHostPages> host_ranges;

    // Iterator over host pages within this contiguous range of device pages.
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::optional<uint32_t>;

        Iterator() = default;
        Iterator(const BufferCorePageMapping* mapping, uint32_t device_page_offset, uint32_t range_index) :
            mapping_(mapping), device_page_offset_(device_page_offset), range_index_(range_index) {}

        Iterator& operator++();
        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        value_type operator*() const;
        uint32_t device_page_offset() const { return device_page_offset_; }

        bool operator==(const Iterator& other) const { return device_page_offset_ == other.device_page_offset_; }
        bool operator!=(const Iterator& other) const { return device_page_offset_ != other.device_page_offset_; }

        struct Range {
            uint32_t device_page_offset = 0;
            uint32_t host_page_start = 0;
            uint32_t num_pages = 0;
        };
        Range next_range(uint32_t end_device_page_offset);

    private:
        const BufferCorePageMapping* mapping_ = nullptr;
        uint32_t device_page_offset_ = 0;
        uint32_t range_index_ = 0;
    };

    Iterator begin() const { return Iterator(this, 0, 0); }
    Iterator end() const { return Iterator(this, num_pages, host_ranges.size()); }
};

// Represents a mapping between host pages and device pages for a given buffer.
struct BufferPageMapping {
    BufferPageMapping() = default;
    BufferPageMapping(const UncompressedBufferPageMapping& page_mapping);

    BufferPageMapping filter_by_host_range(uint32_t start_host_page, uint32_t end_host_page) const;

    std::vector<CoreCoord> all_cores;
    std::unordered_map<CoreCoord, uint32_t> core_to_core_id;
    // For each core, a vector of BufferCorePageMapping, representing contiguous range of device pages.
    std::vector<std::vector<BufferCorePageMapping>> core_page_mappings;

    // Iterator over all host <-> device page mapping pairs.
    struct Iterator {
        struct MappedPage {
            uint32_t core_id = 0;
            uint32_t device_page = 0;
            uint32_t host_page = 0;
        };

        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = MappedPage;

        Iterator() = default;
        Iterator(
            const BufferPageMapping* page_mapping,
            uint32_t core_id,
            uint32_t page_mapping_index,
            uint32_t host_range_index,
            uint32_t host_page_index) :
            page_mapping_(page_mapping),
            core_id_(core_id),
            page_mapping_index_(page_mapping_index),
            host_range_index_(host_range_index),
            host_page_index_(host_page_index) {}

        Iterator& operator++();
        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        value_type operator*() const;

        bool operator==(const Iterator& other) const = default;
        bool operator!=(const Iterator& other) const = default;

    private:
        const BufferPageMapping* page_mapping_ = nullptr;
        uint32_t core_id_ = 0;
        uint32_t page_mapping_index_ = 0;
        uint32_t host_range_index_ = 0;
        uint32_t host_page_index_ = 0;
    };

    Iterator begin() const;
    Iterator end() const;
};

}  // namespace tt::tt_metal
