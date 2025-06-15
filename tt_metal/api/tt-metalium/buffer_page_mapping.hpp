// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>

#include <vector>
#include <optional>
#include <unordered_map>
#include <array>

namespace tt::tt_metal {

struct BufferPageMapping {
    static constexpr uint32_t PADDING = std::numeric_limits<uint32_t>::max();

    std::vector<CoreCoord> all_cores;
    std::vector<std::vector<uint32_t>> core_host_page_indices;
};

struct BufferCorePageMapping {
    struct ContiguousHostPages {
        uint32_t device_page_offset = 0;
        uint32_t host_page_start = 0;
        uint32_t num_pages = 0;
    };

    uint32_t start_page = 0;
    uint32_t num_pages = 0;
    std::vector<ContiguousHostPages> host_ranges;
};

struct CompressedBufferPageMapping {
    CompressedBufferPageMapping() = default;
    CompressedBufferPageMapping(const BufferPageMapping& page_mapping);

    CompressedBufferPageMapping filter_by_host_range(uint32_t start_host_page, uint32_t end_host_page) const;

    std::vector<CoreCoord> all_cores;
    std::unordered_map<CoreCoord, uint32_t> core_to_core_id;
    std::vector<std::vector<BufferCorePageMapping>> core_page_mappings;
};

class BufferCorePageMappingIterator {
public:
    BufferCorePageMappingIterator() = default;
    BufferCorePageMappingIterator(const BufferCorePageMapping* mapping) : mapping_(mapping) {}

    uint32_t device_page() const { return device_page_; }

    BufferCorePageMappingIterator& operator++() {
        device_page_++;
        if (range_index_ >= mapping_->host_ranges.size()) {
            return *this;
        }
        const auto& host_range = mapping_->host_ranges[range_index_];
        if (device_page_ == host_range.device_page_offset + host_range.num_pages) {
            range_index_++;
        }
        return *this;
    }

    std::optional<uint32_t> operator*() const {
        if (range_index_ >= mapping_->host_ranges.size()) {
            return std::nullopt;
        }
        const auto& host_range = mapping_->host_ranges[range_index_];
        if (device_page_ < host_range.device_page_offset) {
            return std::nullopt;
        }
        return host_range.host_page_start + device_page_ - host_range.device_page_offset;
    }

    struct Range {
        uint32_t device_page_start = 0;
        uint32_t host_page_start = 0;
        uint32_t num_pages = 0;
    };
    Range next_range(uint32_t end_device_page) {
        Range result;
        if (range_index_ >= mapping_->host_ranges.size() || device_page_ >= end_device_page) {
            return result;
        }
        const auto& host_range = mapping_->host_ranges[range_index_];
        uint32_t num_pages_left = host_range.num_pages - (device_page_ - host_range.device_page_offset);
        uint32_t num_pages = std::min(num_pages_left, end_device_page - device_page_);
        result = Range{
            .device_page_start = device_page_,
            .host_page_start = host_range.host_page_start + device_page_ - host_range.device_page_offset,
            .num_pages = num_pages,
        };
        device_page_ += num_pages;
        if (device_page_ == host_range.device_page_offset + host_range.num_pages) {
            range_index_++;
        }

        return result;
    }

private:
    const BufferCorePageMapping* mapping_ = nullptr;
    uint32_t device_page_ = 0;
    uint32_t range_index_ = 0;
};

}  // namespace tt::tt_metal
