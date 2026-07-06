// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "tt_elffile.hpp"

namespace ll_api {

class memory {
public:
    using address_t = std::uint64_t;
    using word_t = std::uint32_t;
    enum class Loading : std::uint8_t { DISCRETE, CONTIGUOUS, CONTIGUOUS_XIP };

private:
    struct span {
        // Note: the offset of the data for a span in data_ is generated on the
        // fly by processing spans in order
        address_t addr;  // byte address in device memory
        size_t len;
        bool operator==(const span& other) const { return addr == other.addr && len == other.len; }
    };

    std::vector<word_t> data_;
    std::vector<struct span> link_spans_;
    std::uint32_t text_size_ = 0;
    std::uint32_t text_addr_ = 0;
    Loading loading_{Loading::DISCRETE};

    // Populate link_spans_/data_/text_addr_/text_size_ from ELF segments, ordering by address for
    // DISCRETE loads. Split out of the (path, loading) constructor so the ordering logic can be
    // exercised by unit tests with synthetic segments (no on-disk ELF needed).
    void pack_from_segments(const std::string& path, const std::vector<ElfFile::Segment>& segments);

    // Grants the ordering unit test access to loading_ and pack_from_segments.
    friend class MemorySegmentOrderingTest;

public:
    memory();
    memory(const std::string& path, Loading loading);

    // These can be large objects, so ban copying ...
    memory(const memory&) = delete;
    memory& operator=(const memory&) = delete;
    // ... but permit moving.
    memory(memory&&) = default;
    memory& operator=(memory&&) = default;

    const std::vector<word_t>& data() const { return this->data_; }

    bool operator==(const memory& other) const;
    Loading get_loading() const { return loading_; }

    std::uint32_t get_text_size() const { return this->text_size_; }
    std::uint32_t get_packed_size() const { return data_.size() * sizeof(word_t); }
    std::uint32_t get_text_addr() const { return this->text_addr_; }
    void set_text_addr(const std::uint32_t& addr) { this->text_addr_ = addr; }

    size_t size() const { return data_.size(); }

    size_t num_spans() const { return link_spans_.size(); }

    // Process spans in arg mem to fill data in *this (eg, from device)
    void fill_from_mem_template(
        const memory& mem_template,
        const std::function<void(std::vector<std::uint32_t>::iterator, std::uint64_t addr, std::uint32_t len)>&
            callback);

    // Iterate over spans_ to act on data_ (eg., to device)
    void process_spans(
        const std::function<void(std::vector<std::uint32_t>::const_iterator, std::uint64_t addr, std::uint32_t len)>&
            callback) const;
    void process_spans(
        const std::function<void(std::vector<std::uint32_t>::iterator, std::uint64_t addr, std::uint32_t len)>&
            callback);
    void update_spans(std::function<void(std::uint64_t& addr)>& callback);
};

}  // namespace ll_api
