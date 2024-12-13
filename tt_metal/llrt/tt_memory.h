// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ll_api {

class memory {
public:
    typedef std::uint64_t address_t;
    typedef std::uint32_t word_t;
    enum class Loading : std::uint8_t { DISCRETE, CONTIGUOUS, CONTIGUOUS_XIP };

private:
    struct span {
        // Note: the offset of the data for a span in data_ is generated on the
        // fly by processing spans in order
        address_t addr;    // byte address in device memory
        size_t len;
        bool operator==(const span& other) const { return addr == other.addr && len == other.len; }
    };

    std::vector<word_t> data_;
    std::vector<struct span> link_spans_;
    uint32_t text_size_ = 0;
    uint32_t text_addr_ = 0;
    Loading loading_;

public:
    memory();
    memory(std::string const &path, Loading loading);

public:
    // These can be large objects, so ban copying ...
    memory(memory const&) = delete;
    memory& operator=(memory const&) = delete;
    // ... but permit moving.
    memory(memory&&) = default;
    memory& operator=(memory&&) = default;

public:
    const std::vector<word_t>& data() const { return this->data_; }

    bool operator==(const memory& other) const;
    Loading get_loading() const {return loading_;}

    uint32_t get_text_size() const { return this->text_size_; }
    uint32_t get_packed_size() const { return data_.size() * sizeof(word_t); }
    uint32_t get_text_addr() const { return this->text_addr_; }

    size_t size() const { return data_.size(); }

    size_t num_spans() const { return link_spans_.size(); }

public:
    // Process spans in arg mem to fill data in *this (eg, from device)
    void fill_from_mem_template(const memory& mem_template, const std::function<void (std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback);

    // Iterate over spans_ to act on data_ (eg., to device)
    void process_spans(const std::function<void (std::vector<uint32_t>::const_iterator, uint64_t addr, uint32_t len)>& callback) const;
    void process_spans(const std::function<void (std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback);
};

}  // namespace ll_api
