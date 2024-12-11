// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_memory.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "tt_elffile.hpp"
#include "tt_metal/common/assert.hpp"

namespace ll_api {

memory::memory() {
    constexpr uint32_t initial_data_space_ = 0x400;
    constexpr uint32_t initial_span_space_ = 4;

    data_.reserve(initial_data_space_);
    link_spans_.reserve(initial_span_space_);
}

memory::memory(std::string const& path, Packing pack_type, Relocate relo_type) {
    ElfFile elf;

    elf.ReadImage(path);
    if (relo_type == Relocate::XIP) {
        elf.MakeExecuteInPlace();
    }

    auto const& segments = elf.GetSegments();

    // The ELF file puts the text segment first, but one set of
    // binaries (ncrisc) places data a lower address, and at least one
    // consumer (unknown) requires spans in order.  So generate a
    // mapping table.
    // TODO: Perhaps we can relax this?
    std::vector<unsigned> map;
    map.reserve(segments.size());
    for (unsigned ix = 0; ix != segments.size(); ix++) {
        map.push_back(ix);
    }
    std::sort(
        map.begin(), map.end(), [&](unsigned a, unsigned b) { return segments[a].address < segments[b].address; });

    for (unsigned ix : map) {
        auto const& segment = segments[map[ix]];
        if (!segment.relocs.empty()) {
            TT_THROW("{}: contains dynamic relocations", path);
        }
        if (segment.contents.size()) {
            link_spans_.emplace_back(segment.address, segment.contents.size());
            data_.insert(data_.end(), segment.contents.begin(), segment.contents.end());
        }
    };
    text_addr_ = segments[0].address;
    set_text_size(segments[0].contents.size() * sizeof(word_t));
    set_packed_size(data_.size() * sizeof(uint32_t));

    if (pack_type == Packing::CONTIGUOUS) {
        TT_ASSERT(this->link_spans_.size() != 0);
        TT_ASSERT(link_spans_.size() <= 2);

        std::vector<word_t> new_data2;

        bool text_is_second = link_spans_.size() == 2 && link_spans_[1].addr == text_addr_;
        auto const& text = link_spans_[text_is_second];

        span new_span2 = text;

        uint32_t offset = text_is_second ? link_spans_[0].len : 0;
        new_data2.insert(new_data2.end(), &data_[offset], &data_[offset] + text.len);

        if (link_spans_.size() == 2) {
            offset = text_is_second ? 0 : text.len;
            auto const& data = link_spans_[!text_is_second];
            new_span2.len += data.len;
            new_data2.insert(new_data2.end(), &data_[offset], &data_[offset] + data.len);
        }

        this->link_spans_.resize(1);
        this->link_spans_[0] = new_span2;
        this->data_ = new_data2;
    }
}

bool memory::operator==(const memory& other) const { return data_ == other.data_ && link_spans_ == other.link_spans_; }

void memory::fill_from_mem_template(
    const memory& mem_template,
    const std::function<void(std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback) {
    link_spans_ = mem_template.link_spans_;
    data_.resize(mem_template.data_.size());
    process_spans(callback);
}

void memory::process_spans(
    const std::function<void(std::vector<uint32_t>::const_iterator, uint64_t addr, uint32_t len)>& callback) const {
    uint32_t offset = 0;
    for (const auto& span : link_spans_) {
        std::vector<uint32_t>::const_iterator cit = data_.cbegin() + offset;
        callback(cit, span.addr, span.len);
        offset += span.len;
    }
}

void memory::process_spans(
    const std::function<void(std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback) {
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
void memory::pack_data_into_text(std::uint64_t, std::uint64_t) {
#if 0
    TT_ASSERT(this->link_spans_.size() != 0);
    TT_ASSERT(link_spans_.size() <= 2);

    std::vector<word_t> new_data2;

    bool text_is_second = link_spans_.size() == 2 && link_spans_[1].addr == text_addr_;
    auto const& text = link_spans_[text_is_second];

    span new_span2 = text;

    uint32_t offset = text_is_second ? link_spans_[0].len : 0;
    new_data2.insert(new_data2.end(), &data_[offset], &data_[offset] + text.len);

    if (link_spans_.size() == 2) {
        offset = text_is_second ? 0 : text.len;
        auto const& data = link_spans_[!text_is_second];
        new_span2.len += data.len;
        new_data2.insert(new_data2.end(), &data_[offset], &data_[offset] + data.len);
    }

    this->link_spans_.resize(1);
    this->link_spans_[0] = new_span2;
    this->data_ = new_data2;
#endif
}

}  // namespace ll_api
