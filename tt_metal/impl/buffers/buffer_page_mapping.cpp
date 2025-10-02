// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer_page_mapping.hpp>

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

std::vector<BufferCorePageMapping::ContiguousHostPages> to_host_page_ranges(
    tt::stl::Span<const uint32_t> host_page_indices) {
    std::vector<BufferCorePageMapping::ContiguousHostPages> result;

    uint32_t start_host_page_idx = 0;
    bool is_processing_range = false;

    auto add_page = [&](uint32_t end_host_page_idx) {
        uint32_t start_host_page = host_page_indices[start_host_page_idx];
        uint32_t end_host_page = host_page_indices[end_host_page_idx - 1];
        result.push_back({
            .device_page_offset = start_host_page_idx,
            .host_page_start = start_host_page,
            .num_pages = end_host_page - start_host_page + 1,
        });
    };

    for (size_t i = 0; i < host_page_indices.size(); i++) {
        uint32_t host_page = host_page_indices[i];
        if (!is_processing_range) {
            if (host_page != UncompressedBufferPageMapping::PADDING) {
                start_host_page_idx = i;
                is_processing_range = true;
            }
        } else if (host_page == UncompressedBufferPageMapping::PADDING) {
            add_page(i);
            start_host_page_idx = i + 1;
            is_processing_range = false;
        } else if (host_page_indices[i - 1] + 1 != host_page) {
            add_page(i);
            start_host_page_idx = i;
            is_processing_range = true;
        }
    }
    if (is_processing_range) {
        add_page(host_page_indices.size());
    }

    return result;
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

BufferPageMapping::BufferPageMapping(const UncompressedBufferPageMapping& page_mapping) :
    all_cores(page_mapping.all_cores) {
    for (size_t i = 0; i < all_cores.size(); i++) {
        core_to_core_id[all_cores[i]] = i;
    }

    size_t num_cores = page_mapping.all_cores.size();
    core_page_mappings.resize(num_cores);
    for (size_t core_id = 0; core_id < num_cores; core_id++) {
        auto& core_page_mapping = core_page_mappings[core_id];
        const auto& core_host_page_indices = page_mapping.core_host_page_indices[core_id];
        if (core_host_page_indices.empty()) {
            continue;
        }
        auto host_ranges = CMAKE_UNIQUE_NAMESPACE::to_host_page_ranges(core_host_page_indices);
        if (host_ranges.empty()) {
            continue;
        }
        core_page_mapping.push_back(BufferCorePageMapping{
            .device_start_page = 0,
            .num_pages = static_cast<uint32_t>(core_host_page_indices.size()),
            .host_ranges = std::move(host_ranges),
        });
    }
}

BufferPageMapping BufferPageMapping::filter_by_host_range(uint32_t start_host_page, uint32_t end_host_page) const {
    BufferPageMapping result;
    result.all_cores = all_cores;
    result.core_to_core_id = core_to_core_id;
    result.core_page_mappings.resize(all_cores.size());

    BufferCorePageMapping result_core_mapping;

    auto add_core_mapping = [&](size_t core_id, uint32_t original_device_start_page) {
        if (result_core_mapping.host_ranges.empty()) {
            return;
        }

        uint32_t min_device_page_offset = std::numeric_limits<uint32_t>::max();
        uint32_t max_device_page_offset = 0;
        for (const auto& host_range : result_core_mapping.host_ranges) {
            min_device_page_offset = std::min(min_device_page_offset, host_range.device_page_offset);
            max_device_page_offset =
                std::max(max_device_page_offset, host_range.device_page_offset + host_range.num_pages);
        }
        result_core_mapping.device_start_page = original_device_start_page + min_device_page_offset;
        result_core_mapping.num_pages = max_device_page_offset - min_device_page_offset;
        for (auto& host_range : result_core_mapping.host_ranges) {
            host_range.device_page_offset -= min_device_page_offset;
        }

        result.core_page_mappings[core_id].push_back(result_core_mapping);
        result_core_mapping.host_ranges.clear();
    };

    for (size_t core_id = 0; core_id < all_cores.size(); core_id++) {
        for (const auto& core_page_mapping : core_page_mappings[core_id]) {
            for (const auto& host_range : core_page_mapping.host_ranges) {
                auto host_range_start = std::max(start_host_page, host_range.host_page_start);
                auto host_range_end = std::min(end_host_page, host_range.host_page_start + host_range.num_pages);

                if (host_range_start != host_range.host_page_start) {
                    add_core_mapping(core_id, core_page_mapping.device_start_page);
                }

                if (host_range_start >= host_range_end) {
                    add_core_mapping(core_id, core_page_mapping.device_start_page);
                    continue;
                }

                result_core_mapping.host_ranges.push_back({
                    .device_page_offset = host_range.device_page_offset + host_range_start - host_range.host_page_start,
                    .host_page_start = host_range_start - start_host_page,
                    .num_pages = host_range_end - host_range_start,
                });

                if (host_range_end != host_range.host_page_start + host_range.num_pages) {
                    add_core_mapping(core_id, core_page_mapping.device_start_page);
                }
            }
            add_core_mapping(core_id, core_page_mapping.device_start_page);
        }
    }

    return result;
}

BufferCorePageMapping::Iterator& BufferCorePageMapping::Iterator::operator++() {
    device_page_offset_++;
    if (range_index_ >= mapping_->host_ranges.size()) {
        return *this;
    }
    const auto& host_range = mapping_->host_ranges[range_index_];
    if (device_page_offset_ == host_range.device_page_offset + host_range.num_pages) {
        range_index_++;
    }
    return *this;
}

std::optional<uint32_t> BufferCorePageMapping::Iterator::operator*() const {
    if (range_index_ >= mapping_->host_ranges.size()) {
        return std::nullopt;
    }
    const auto& host_range = mapping_->host_ranges[range_index_];
    if (device_page_offset_ < host_range.device_page_offset) {
        return std::nullopt;
    }
    return host_range.host_page_start + device_page_offset_ - host_range.device_page_offset;
}

BufferCorePageMapping::Iterator::Range BufferCorePageMapping::Iterator::next_range(uint32_t end_device_page_offset) {
    Range result;
    if (range_index_ >= mapping_->host_ranges.size() || device_page_offset_ >= end_device_page_offset) {
        return result;
    }
    const auto& host_range = mapping_->host_ranges[range_index_];
    uint32_t num_pages_left = host_range.num_pages - (device_page_offset_ - host_range.device_page_offset);
    uint32_t num_pages = std::min(num_pages_left, end_device_page_offset - device_page_offset_);
    result = Range{
        .device_page_offset = device_page_offset_,
        .host_page_start = host_range.host_page_start + device_page_offset_ - host_range.device_page_offset,
        .num_pages = num_pages,
    };
    device_page_offset_ += num_pages;
    if (device_page_offset_ == host_range.device_page_offset + host_range.num_pages) {
        range_index_++;
    }

    return result;
}

BufferPageMapping::Iterator& BufferPageMapping::Iterator::operator++() {
    host_page_index_++;

    auto& core_page_mapping = page_mapping_->core_page_mappings[core_id_][page_mapping_index_];
    if (host_page_index_ < core_page_mapping.host_ranges[host_range_index_].num_pages) {
        return *this;
    }
    host_page_index_ = 0;
    host_range_index_++;

    if (host_range_index_ < core_page_mapping.host_ranges.size()) {
        return *this;
    }
    host_range_index_ = 0;
    page_mapping_index_++;

    while (core_id_ < page_mapping_->all_cores.size() &&
           page_mapping_index_ == page_mapping_->core_page_mappings[core_id_].size()) {
        page_mapping_index_ = 0;
        core_id_++;
    }
    return *this;
}

BufferPageMapping::Iterator::MappedPage BufferPageMapping::Iterator::operator*() const {
    const auto& core_page_mapping = page_mapping_->core_page_mappings[core_id_][page_mapping_index_];
    const auto& host_range = core_page_mapping.host_ranges[host_range_index_];
    return MappedPage{
        .core_id = core_id_,
        .device_page = core_page_mapping.device_start_page + host_range.device_page_offset + host_page_index_,
        .host_page = host_range.host_page_start + host_page_index_,
    };
}

BufferPageMapping::Iterator BufferPageMapping::begin() const {
    for (uint32_t core_id = 0; core_id < all_cores.size(); core_id++) {
        if (!core_page_mappings[core_id].empty()) {
            return BufferPageMapping::Iterator(this, core_id, 0, 0, 0);
        }
    }
    return end();
}

BufferPageMapping::Iterator BufferPageMapping::end() const {
    return BufferPageMapping::Iterator(this, all_cores.size(), 0, 0, 0);
}

}  // namespace tt::tt_metal
