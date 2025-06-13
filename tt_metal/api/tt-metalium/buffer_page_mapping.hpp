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

struct BufferCorePageMapping;

class BufferCorePageMappingIterator {
public:
    BufferCorePageMappingIterator() = default;
    BufferCorePageMappingIterator(const BufferCorePageMapping* mapping, uint32_t device_offset, uint32_t range_index) :
        mapping(mapping), device_offset(device_offset), range_index(range_index) {}

    bool operator==(const BufferCorePageMappingIterator& other) const = default;
    bool operator!=(const BufferCorePageMappingIterator& other) const = default;

    void next();
    std::optional<uint32_t> operator*() const;

private:
    const BufferCorePageMapping* mapping = nullptr;
    uint32_t device_offset = 0;
    uint32_t range_index = 0;
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

    BufferCorePageMappingIterator begin() const;
    BufferCorePageMappingIterator end() const;
};

struct CompressedBufferPageMapping {
    CompressedBufferPageMapping() = default;
    CompressedBufferPageMapping(const BufferPageMapping& page_mapping);

    CompressedBufferPageMapping filter_by_host_range(uint32_t start_host_page, uint32_t end_host_page) const;

    std::vector<CoreCoord> all_cores;
    std::unordered_map<CoreCoord, uint32_t> core_to_core_id;
    std::vector<std::vector<BufferCorePageMapping>> core_page_mappings;
};

}  // namespace tt::tt_metal
