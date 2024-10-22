// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_memory.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <span> // FIXME: No C++20 desired in tt_metal
#include <vector>

#include "tt_elffile.hpp"
#include "tt_metal/common/assert.hpp"

namespace ll_api {

memory::memory() {
    data_.reserve(initial_data_space_);
    link_spans_.reserve(initial_span_space_);
    text_size_ = 0;
    packed_size_ = 0;
}

memory::memory(std::string const &path) : memory() {
    ElfFile elf;

    elf.ReadImage(path);

    // The ELF file puts the text segment first, but memory wants
    // ordered spans.
    // FIXME: Perhaps we can relax that?
    uint32_t total_size = 0;
    auto emit_segment = [&](ElfFile::Segment const& segment) {
        link_spans_.emplace_back(
            segment.address, segment.contents.size());
        data_.insert(data_.end(), segment.contents.begin(), segment.contents.end());
        total_size += segment.contents.size();
    };
    auto* text = &elf.GetSegments()[0];
    for (auto& segment : std::span(elf.GetSegments()).subspan(1)) {
        if (text && segment.address > text->address) {
            emit_segment(*text);
            text = nullptr;
        }
        emit_segment(segment);
    }
    if (text)
        emit_segment(*text);

    set_text_size(elf.GetSegments()[0].contents.size() * sizeof(uint32_t));
    set_packed_size(total_size * sizeof(uint32_t));
}

bool memory::operator==(const memory& other) const {
    return
        data_ == other.data_ &&
        link_spans_ == other.link_spans_;
}

void memory::fill_from_mem_template(const memory& mem_template, const std::function<void (std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback) {
    link_spans_ = mem_template.link_spans_;
    data_.resize(mem_template.data_.size());
    process_spans(callback);
}

void memory::process_spans(const std::function<void (std::vector<uint32_t>::const_iterator, uint64_t addr, uint32_t len)>& callback) const {
    uint32_t offset = 0;
    for (const auto& span : link_spans_) {
        std::vector<uint32_t>::const_iterator cit = data_.cbegin() + offset;
        callback(cit, span.addr, span.len);
        offset += span.len;
    }
}

void memory::process_spans(const std::function<void (std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback) {
    uint32_t offset = 0;
    for (const auto& span : link_spans_) {
        std::vector<uint32_t>::iterator it = data_.begin() + offset;
        callback(it, span.addr, span.len);
        offset += span.len;
    }
}

// Takes spans and merges the data to the text span
// Used for kernels (not firmware)
// Spans get packed for kernels so they can be loaded in one NOC transaction
// A symbol at the end of the text segment allows the FW to find the data segment to copy into place
void memory::pack_data_into_text(std::uint64_t text_start, std::uint64_t data_start) {

    uint64_t text_end, data_end;
    if (text_start > data_start) {
        text_end = std::numeric_limits<uint64_t>::max();
        data_end = text_start;
    } else {
        text_end = data_start;
        data_end = std::numeric_limits<uint64_t>::max();
    }

    TT_ASSERT(this->link_spans_.size() != 0);

    std::vector<word_t> new_data;
    new_data.resize(this->data_.size());
    struct span new_span;
    size_t new_len = 0;

    bool first_text = true;
    size_t offset = 0;
    // Copy text spans.  May start after data span (ncrisc)
    // TODO: Ideally would be just 1, sometimes init doesn't merge w/ text and we get 2
    // TODO: (and init is just a jump to text and should be removed)
    for (const auto& span : this->link_spans_) {
        if (span.addr >= text_start && span.addr < text_end) {
            if (first_text) {
                new_span.addr = span.addr;
                first_text = false;
            } else if (span.addr > new_span.addr + new_len * sizeof(uint32_t)) {
                uint64_t delta = span.addr - (new_span.addr + new_len * sizeof(uint32_t));
                delta /= sizeof(uint32_t);
                // Pad the prior span
                new_data.resize(new_data.size() + delta);
                new_len += delta;
            }
            memcpy(&new_data[new_len], &this->data_[offset], span.len * sizeof(uint32_t));
            new_len += span.len;
        }

        offset += span.len;
    }
    TT_ASSERT(!first_text);

    // Copy data spans.  Should be just 1.  May start before text span (ncrisc)
    offset = 0;
    for (const auto& span : this->link_spans_) {
        if (span.addr >= data_start && span.addr < data_end) {
            memcpy(&new_data[new_len], &this->data_[offset], span.len * sizeof(uint32_t));
            new_len += span.len;
        }
        offset += span.len;
    }

    new_span.len = new_len;
    this->link_spans_.resize(1);
    this->link_spans_[0] = new_span;
    this->data_ = new_data;
}

}  // namespace ll_api
