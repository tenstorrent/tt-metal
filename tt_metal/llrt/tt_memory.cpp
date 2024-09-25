// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <cassert>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "tensix.h"
#include "tt_memory.h"
#include "tt_hexfile.h"

using std::numeric_limits;
using std::runtime_error;
using std::string;
using std::vector;

namespace ll_api {

// We use stoul to parse an address. We cast address_t to unsigned long to format with %08lX.
static_assert(
    numeric_limits<unsigned long>::max() >= numeric_limits<memory::address_t>::max(),
    "unsigned long can't cover whole range of addresses");

// We cast word_t to unsigned long to format with %08lX.
static_assert(
    numeric_limits<unsigned long>::max() >= numeric_limits<memory::word_t>::max(),
    "unsigned long can't cover whole range of words");


memory::memory() {
    data_.reserve(initial_data_space_);
    link_spans_.reserve(initial_span_space_);
}

memory::memory(std::string const &path) : memory() {
  fill_from_discontiguous_hex(path);
}

bool memory::operator==(const memory& other) const {
    return
        data_ == other.data_ &&
        link_spans_ == other.link_spans_;
}

void memory::fill_from_discontiguous_hex(std::string const &path) {
    std::ifstream is(path);

    // Intended to start empty
    assert(data_.empty());
    bool first = true;
    address_t last_addr = 0;
    // hex files run low address to high address
    read_discontiguous_hex_file(is, [&](memory::address_t word_addr, memory::word_t value) {
        if (first || word_addr != last_addr + 1) {
            link_spans_.push_back({word_addr << 2, 0});
            first = false;
        }

        data_.push_back(value);
        link_spans_.back().len++;
        last_addr = word_addr;
    });
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

}  // namespace ll_api
